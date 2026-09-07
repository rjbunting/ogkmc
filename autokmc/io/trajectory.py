"""Trajectory persistence helpers."""

from __future__ import annotations

import io
import operator
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, SupportsIndex, cast

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from autokmc.core.constants import TRAJ_DUMP_EVERY
from autokmc.io._files import atomic_output_path, ensure_directory, fsync_directory
from autokmc.io.schemas import (
	TRAJECTORY_ARTIFACT_TYPE,
	TRAJECTORY_METADATA_SCHEMA_VERSION,
)


def _validated_step(value: object, *, context: str) -> int:
	"""Return an integer KMC step or raise a persistence-focused error."""
	if isinstance(value, bool):
		raise ValueError(f"{context} must be an integer, got {value!r}")
	try:
		step = operator.index(cast(SupportsIndex, value))
	except TypeError as exc:
		raise ValueError(f"{context} must be an integer, got {value!r}") from exc
	if step < 0:
		raise ValueError(f"{context} cannot be negative, got {step}")
	return int(step)


def _copy_prefix(source, destination, length: int) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("trajectory changed during atomic replacement")
        destination.write(chunk)
        remaining -= len(chunk)


def _last_frame_offset(path: Path) -> int:
    """Locate the final frame without materializing the trajectory."""
    last = 0
    with path.open("rb") as source:
        while True:
            offset = source.tell()
            header = source.readline()
            if not header:
                return last
            if not header.strip():
                continue
            count = int(header)
            if count < 0:
                raise ValueError("negative atom count in trajectory")
            source.readline()
            for _ in range(count):
                if not source.readline():
                    raise ValueError("incomplete final trajectory frame")
            last = offset


def _metadata_matches(
	current: Mapping[str, Any],
	expected: Mapping[str, Any],
) -> bool:
	"""Compare metadata while tolerating extxyz's numeric schema coercion."""
	for key, value in expected.items():
		existing = current.get(key)
		if key == "trajectory_schema_version":
			if str(existing) != str(value):
				return False
		elif existing != value:
			return False
	return True


def reconcile_trajectory(
    output_path: str | Path,
    *,
    checkpoint_step: int,
    committed_offset: int | None = None,
) -> int:
    """Validate the committed frames and remove an interrupted append tail.

    New checkpoints record the exact durable byte prefix. Older checkpoints
    use frame steps; an incomplete final frame is disposable only when its
    header or the preceding frames establish that it is uncommitted.
    """
    from ase.io.extxyz import key_val_str_to_dict

    path = Path(output_path)
    checkpoint_step = _validated_step(checkpoint_step, context="checkpoint_step")
    if committed_offset is not None:
        committed_offset = _validated_step(committed_offset, context="committed_offset")
    if not path.is_file():
        if committed_offset:
            raise ValueError(f"cannot reconcile {path}: committed trajectory is missing")
        return 0
    size = path.stat().st_size
    if committed_offset is not None and committed_offset > size:
        raise ValueError(f"cannot reconcile {path}: trajectory is shorter than its committed prefix")
    retained_end = 0
    previous_step = -1
    tail_started = False
    removed = 0
    limit = size if committed_offset is None else committed_offset
    with path.open("rb") as source:
        def read_line():
            return source.readline(max(0, limit - source.tell()))

        frame_number = 0
        while source.tell() < limit:
            header = read_line()
            if not header.strip():
                if not tail_started:
                    retained_end = source.tell()
                continue
            frame_number += 1
            try:
                count = int(header)
                if count < 0:
                    raise ValueError("negative atom count")
            except ValueError as exc:
                if committed_offset is None and not header.endswith(b"\n") and (
                    tail_started or previous_step == checkpoint_step
                ):
                    removed += 1
                    break
                raise ValueError(f"cannot reconcile {path}: invalid frame {frame_number} atom count") from exc
            comment = read_line()
            if committed_offset is None and not comment.endswith(b"\n") and (
                tail_started or previous_step == checkpoint_step
            ):
                removed += 1
                break
            try:
                metadata = key_val_str_to_dict(comment.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"cannot reconcile {path}: invalid frame {frame_number} metadata") from exc
            if "kmc_step" not in metadata:
                raise ValueError(f"cannot reconcile {path}: frame {frame_number} has no kmc_step")
            step = _validated_step(
                metadata["kmc_step"], context=f"{path} frame {frame_number} kmc_step",
            )
            body = []
            for _ in range(count):
                row = read_line()
                if not row:
                    break
                body.append(row)
            if step > checkpoint_step:
                if committed_offset is not None:
                    raise ValueError(f"cannot reconcile {path}: committed frame exceeds checkpoint step")
                tail_started = True
                removed += 1
                continue
            if tail_started:
                raise ValueError(f"cannot reconcile {path}: committed frame follows an uncommitted crash-tail frame")
            if step <= previous_step:
                raise ValueError(f"cannot reconcile {path}: committed kmc_step values are not strictly increasing")
            try:
                if len(body) != count or any(not row.endswith(b"\n") for row in body) or not comment.endswith(b"\n"):
                    raise ValueError("incomplete committed frame")
                frame_text = (header + comment + b"".join(body)).decode("utf-8")
                ase_read(io.StringIO(frame_text), index=0, format="extxyz")
            except Exception as exc:
                raise ValueError(f"cannot reconcile {path}: invalid committed extxyz frame {frame_number}") from exc
            previous_step = step
            retained_end = source.tell()
    if committed_offset is not None:
        retained_end = committed_offset
        removed = int(size > committed_offset)
    if retained_end < size:
        with atomic_output_path(path) as temporary:
            with path.open("rb") as source, temporary.open("wb") as destination:
                _copy_prefix(source, destination, retained_end)

    return removed


class TrajectoryWriter:
	"""Periodic extended-XYZ dumper."""

	supports_frame_metadata = True

	def __init__(
		self,
		output_path: str | Path,
		*,
		dump_every: int = TRAJ_DUMP_EVERY,
		append: bool = False,
		resume_checkpoint_step: int | None = None,
		resume_committed_offset: int | None = None,
	):
		if resume_committed_offset is not None and resume_checkpoint_step is None:
			raise ValueError("resume_committed_offset requires resume_checkpoint_step")
		self.output_path = Path(output_path)
		self.dump_every = int(dump_every)
		self.append = bool(append)
		self._n_frames: int = 0
		self._dirty = False
		self._checkpoint_offset = resume_committed_offset or 0
		self._directory_entry_dirty = self.append and not self.output_path.is_file()
		self._last_step: int | None = None
		self._last_frame_metadata: dict[str, Any] | None = None
		if resume_checkpoint_step is not None:
			if not self.append:
				raise ValueError(
					"resume_checkpoint_step requires append=True"
				)
			reconcile_trajectory(
				self.output_path,
				checkpoint_step=resume_checkpoint_step,
				committed_offset=resume_committed_offset,
			)
			if self.output_path.is_file() and self.output_path.stat().st_size:
				last_frame = ase_read(self.output_path, index=-1, format="extxyz")
				self._last_step = _validated_step(
					last_frame.info.get("kmc_step"),
					context=f"{self.output_path} final frame kmc_step",
				)
				self._last_frame_metadata = dict(last_frame.info)
		if self.dump_every > 0:
			ensure_directory(self.output_path.parent)
			if not self.append:
				with atomic_output_path(self.output_path):
					pass

	@property
	def n_frames(self) -> int:
		return self._n_frames

	@property
	def last_step(self) -> int | None:
		return self._last_step

	@property
	def enabled(self) -> bool:
		return self.dump_every > 0

	def should_write(self, *, step: int) -> bool:
		"""Return whether *step* is a configured trajectory boundary."""
		return bool(
			self.enabled
			and (int(step) == 0 or (int(step) % self.dump_every) == 0)
		)

	def maybe_write(
		self,
		atoms: Atoms,
		*,
		step: int,
		metadata: Mapping[str, Any] | None = None,
	) -> bool:
		if not self.should_write(step=step):
			return False
		return self._write(atoms, step=step, metadata=metadata)

	def maybe_write_snapshot(
		self,
		snapshot_factory: Callable[[], Atoms],
		*,
		step: int,
		metadata: Mapping[str, Any] | None = None,
	) -> bool:
		"""Build and write an owned snapshot only when *step* is due.

		The factory contract lets graph-backed callers avoid both constructing
		skipped frames and copying a freshly-created :class:`~ase.Atoms`.
		"""
		if not self.should_write(step=step):
			return False
		return self._write(
			snapshot_factory(),
			step=step,
			copy_atoms=False,
			metadata=metadata,
		)

	def write(
		self,
		atoms: Atoms,
		*,
		step: int = 0,
		metadata: Mapping[str, Any] | None = None,
	) -> bool:
		if not self.enabled:
			return False
		return self._write(atoms, step=step, metadata=metadata)

	def write_final_snapshot(
		self,
		snapshot_factory: Callable[[], Atoms],
		*,
		step: int,
		metadata: Mapping[str, Any] | None = None,
	) -> bool:
		"""Write the final state exactly once, irrespective of periodic cadence."""
		if not self.enabled:
			return False
		resolved_step = _validated_step(step, context="final trajectory step")
		if self._last_step == resolved_step:
			return self._upgrade_final_frame(
				step=resolved_step,
				metadata=metadata,
			)
		if self._last_step is not None and self._last_step > resolved_step:
			raise ValueError(
				f"final trajectory step {resolved_step} precedes existing "
				f"frame step {self._last_step}"
			)
		final_metadata = dict(metadata or {})
		final_metadata["frame_kind"] = "final"
		return self._write(
			snapshot_factory(),
			step=resolved_step,
			copy_atoms=False,
			metadata=final_metadata,
		)

	def _upgrade_final_frame(
		self,
		*,
		step: int,
		metadata: Mapping[str, Any] | None,
	) -> bool:
		"""Mark an existing same-step frame final without duplicating the step."""
		final_metadata = dict(metadata or {})
		final_metadata["frame_kind"] = "final"
		final_metadata["artifact_type"] = TRAJECTORY_ARTIFACT_TYPE
		final_metadata[
			"trajectory_schema_version"
		] = TRAJECTORY_METADATA_SCHEMA_VERSION
		final_metadata["kmc_step"] = step
		if (
			self._last_frame_metadata is not None
			and _metadata_matches(self._last_frame_metadata, final_metadata)
		):
			return False
		if not self.output_path.is_file() or self.output_path.stat().st_size == 0:
			return False
		offset = _last_frame_offset(self.output_path)
		with self.output_path.open("rb") as source:
			source.seek(offset)
			raw_frame = source.read()
		final_frame = ase_read(io.StringIO(raw_frame.decode("utf-8")), format="extxyz")
		existing_step = _validated_step(
			final_frame.info.get("kmc_step"),
			context=f"{self.output_path} final frame kmc_step",
		)
		if existing_step != step:
			raise ValueError(
				f"trajectory bookkeeping expected final step {step}, "
				f"but the persisted final frame is step {existing_step}"
			)
		if _metadata_matches(final_frame.info, final_metadata):
			self._last_frame_metadata = dict(final_frame.info)
			return False
		if self._checkpoint_offset > offset:
			# A published checkpoint may still refer to this frame if the process
			# dies during finalization. Preserve its byte boundary and recorded
			# segment metadata; only change the final marker, padding its value.
			lines = raw_frame.split(b"\n", 2)
			comment, replacements = re.subn(
				rb"(?<!\S)frame_kind=(initial|periodic)(?=\s|$)",
				lambda match: b"frame_kind=final".ljust(len(match.group(0))),
				lines[1],
				count=1,
			)
			if not replacements:
				return False
			replacement = b"\n".join((lines[0], comment, lines[2]))
			final_frame.info["frame_kind"] = "final"
		else:
			final_frame.info.update(final_metadata)
			buffer = io.StringIO()
			ase_write(buffer, final_frame, format="extxyz")
			replacement = buffer.getvalue().encode("utf-8")
		with atomic_output_path(self.output_path) as temporary:
			with self.output_path.open("rb") as source, temporary.open("wb") as destination:
				_copy_prefix(source, destination, offset)
				destination.write(replacement)
		self._last_frame_metadata = dict(final_frame.info)
		self._dirty = True
		return True

	def _write(
		self,
		atoms: Atoms,
		*,
		step: int,
		copy_atoms: bool = True,
		metadata: Mapping[str, Any] | None = None,
	) -> bool:
		snap = atoms.copy() if copy_atoms else atoms
		resolved_step = _validated_step(step, context="trajectory step")
		frame_metadata = dict(metadata or {})
		frame_metadata.setdefault(
			"frame_kind",
			"initial" if resolved_step == 0 else "periodic",
		)
		snap.info.update(frame_metadata)
		snap.info["artifact_type"] = TRAJECTORY_ARTIFACT_TYPE
		snap.info["trajectory_schema_version"] = TRAJECTORY_METADATA_SCHEMA_VERSION
		snap.info["kmc_step"] = resolved_step
		ase_write(self.output_path, snap, format="extxyz", append=True)
		self._n_frames += 1
		self._last_step = resolved_step
		self._last_frame_metadata = dict(snap.info)
		self._dirty = True
		return True

	def sync_for_checkpoint(self) -> None:
		"""Make all appended frames durable before publishing a checkpoint."""
		if not self.output_path.is_file():
			return
		if not self._dirty:
			self._checkpoint_offset = self.output_path.stat().st_size
			return
		with self.output_path.open("rb") as handle:
			os.fsync(handle.fileno())
			self._checkpoint_offset = os.fstat(handle.fileno()).st_size
		if self._directory_entry_dirty:
			fsync_directory(self.output_path.parent)
			self._directory_entry_dirty = False
		self._dirty = False

	@property
	def committed_offset(self) -> int | None:
		"""Byte boundary to record after ``sync_for_checkpoint`` succeeds."""
		if not self.enabled:
			return None
		return self.output_path.stat().st_size if self.output_path.is_file() else 0

	def close(self) -> None:
		self.sync_for_checkpoint()

	def __enter__(self):  # pragma: no cover
		return self

	def __exit__(self, *exc):  # pragma: no cover
		self.close()


__all__ = ["TrajectoryWriter", "reconcile_trajectory"]
