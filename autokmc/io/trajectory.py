"""Trajectory persistence helpers."""

from __future__ import annotations

import operator
import os
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


def _atomic_write_frames(path: Path, frames: list[Atoms]) -> None:
	"""Replace *path* with *frames* without exposing a partial trajectory."""
	with atomic_output_path(path) as temporary:
		if frames:
			ase_write(temporary, frames, format="extxyz")


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
) -> int:
	"""Remove trajectory frames not represented by a resumed checkpoint.

	Every retained frame must carry a non-negative integer ``kmc_step`` and the
	committed steps must be strictly increasing. Frames after the first step
	beyond ``checkpoint_step`` form a disposable crash tail. A later committed
	frame is rejected because it would make that prefix ambiguous.

	Missing and empty files are accepted for compatibility with trajectories
	that were disabled or absent in older runs. The number of removed frames is
	returned.
	"""
	path = Path(output_path)
	resolved_checkpoint_step = _validated_step(
		checkpoint_step,
		context="checkpoint_step",
	)
	if not path.is_file() or path.stat().st_size == 0:
		return 0

	try:
		loaded = ase_read(path, index=":", format="extxyz")
	except Exception as exc:
		raise ValueError(f"cannot reconcile {path}: invalid extxyz trajectory") from exc
	frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)

	retained: list[Atoms] = []
	previous_step = -1
	tail_started = False
	for frame_number, frame in enumerate(frames, start=1):
		try:
			step_value = frame.info["kmc_step"]
		except KeyError as exc:
			raise ValueError(
				f"cannot reconcile {path}: frame {frame_number} has no kmc_step"
			) from exc
		step = _validated_step(
			step_value,
			context=f"{path} frame {frame_number} kmc_step",
		)
		if step > resolved_checkpoint_step:
			tail_started = True
			continue
		if tail_started:
			raise ValueError(
				f"cannot reconcile {path}: committed frame {frame_number} "
				f"(kmc_step={step}) follows an uncommitted crash-tail frame"
			)
		if step <= previous_step:
			raise ValueError(
				f"cannot reconcile {path}: committed kmc_step values are not "
				f"strictly increasing at frame {frame_number}"
			)
		retained.append(frame)
		previous_step = step

	removed = len(frames) - len(retained)
	if removed:
		_atomic_write_frames(path, retained)
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
	):
		self.output_path = Path(output_path)
		self.dump_every = int(dump_every)
		self.append = bool(append)
		self._n_frames: int = 0
		self._dirty = False
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
		loaded = ase_read(self.output_path, index=":", format="extxyz")
		frames = [loaded] if isinstance(loaded, Atoms) else list(loaded)
		if not frames:
			return False
		final_frame = frames[-1]
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
		final_frame.info.update(final_metadata)
		_atomic_write_frames(self.output_path, frames)
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
		if not self._dirty or not self.output_path.is_file():
			return
		with self.output_path.open("rb") as handle:
			os.fsync(handle.fileno())
		if self._directory_entry_dirty:
			fsync_directory(self.output_path.parent)
			self._directory_entry_dirty = False
		self._dirty = False

	def close(self) -> None:
		self.sync_for_checkpoint()

	def __enter__(self):  # pragma: no cover
		return self

	def __exit__(self, *exc):  # pragma: no cover
		self.close()


__all__ = ["TrajectoryWriter", "reconcile_trajectory"]
