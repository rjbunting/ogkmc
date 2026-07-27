"""Checkpoint-boundary reconciliation for the append-only KMC event log."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
import json
import math
import os
from pathlib import Path
from typing import Any

from autokmc.core.constants import PERSISTENCE_SCHEMA_VERSION
from autokmc.io._files import atomic_output_path
from autokmc.io.reaction_layout import kind_subdir
from autokmc.io.summary import ReactionSummary
from autokmc.species.smiles import smiles_to_dirname


ReactionKey = tuple[str, str, int, int]


def _history_record(event: dict[str, Any]) -> tuple:
    """Return the stable public KMC history tuple for one event row."""
    delta_e = event.get("rate_delta_ev")
    if delta_e is None:
        delta_e = event["delta_e_ev"]
    barrier = event.get("rate_barrier_ev")
    if barrier is None:
        barrier = event["barrier_ev"]
    return (
        int(event["step"]),
        float(event["time_s"]),
        str(event["kind"]),
        int(event["iso_class"]),
        int(event["member_index"]),
        int(event["lateral_class"]),
        float(delta_e),
        float(barrier),
        float(event["rate_hz"]),
    )


class EventHistory(Sequence[tuple]):
    """Lazy committed history prefix plus an in-memory continuation suffix.

    A resumed run can append new tuples without materialising the pre-checkpoint
    history. Iteration or indexing explicitly streams that committed prefix
    from ``events.jsonl``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        committed_count: int,
        committed_offset: int,
    ) -> None:
        self.path = Path(path)
        self.committed_count = int(committed_count)
        self.committed_offset = int(committed_offset)
        self._suffix: list[tuple] = []

    @property
    def in_memory_count(self) -> int:
        """Number of post-resume records currently retained in memory."""
        return len(self._suffix)

    def append(self, record: tuple) -> None:
        self._suffix.append(tuple(record))

    def mark_committed(self, *, count: int, offset: int) -> None:
        """Move the in-memory suffix into a newly durable JSONL prefix."""
        resolved_count = int(count)
        if resolved_count != len(self):
            raise ValueError(
                "event-history commit count does not match prefix plus suffix: "
                f"{resolved_count} != {len(self)}"
            )
        self.committed_count = resolved_count
        self.committed_offset = int(offset)
        self._suffix.clear()

    def __len__(self) -> int:
        return self.committed_count + len(self._suffix)

    def __iter__(self) -> Iterator[tuple]:
        seen = 0
        if self.committed_offset:
            with self.path.open("rb") as handle:
                line_number = 0
                while handle.tell() < self.committed_offset:
                    remaining = self.committed_offset - handle.tell()
                    raw = handle.readline(remaining)
                    line_number += 1
                    if not raw.endswith(b"\n"):
                        raise ValueError(
                            "committed event-history offset no longer ends at "
                            "a JSONL boundary"
                        )
                    if not raw.strip():
                        continue
                    event = _decode_event_line(self.path, raw, line_number)
                    yield _history_record(event)
                    seen += 1
        if seen != self.committed_count:
            raise ValueError(
                "committed event-history count no longer matches its byte prefix: "
                f"{self.committed_count} != {seen}"
            )
        yield from self._suffix

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        return next(islice(iter(self), resolved, resolved + 1))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sequence):
            return False
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other)
        )

    def __repr__(self) -> str:
        return (
            "EventHistory("
            f"committed={self.committed_count}, "
            f"in_memory={len(self._suffix)})"
        )

    def materialize(self) -> list[tuple]:
        """Explicitly return the complete history as a list."""
        return list(self)


@dataclass
class ReactionEventState:
    """Compact event-derived state for one persisted reaction folder."""

    count: int = 0
    first_step: int | None = None
    last_step: int | None = None
    last_event: dict[str, Any] | None = None
    rate_energy_bases: set[str] = field(default_factory=set)

    def add(self, event: dict[str, Any]) -> None:
        step = int(event["step"])
        self.count += 1
        if self.first_step is None:
            self.first_step = step
        self.last_step = step
        self.last_event = event
        basis = event.get("rate_energy_basis")
        if basis:
            self.rate_energy_bases.add(str(basis))


@dataclass
class EventLogRecovery:
    """One-pass recovery products shared by resume consumers."""

    history: list[tuple] | EventHistory = field(default_factory=list)
    summary: ReactionSummary = field(default_factory=ReactionSummary)
    reaction_states: dict[ReactionKey, ReactionEventState] = field(
        default_factory=dict
    )
    count: int = 0
    history_complete: bool = True
    summary_complete: bool = True
    collect_history: bool = field(default=True, repr=False)

    def add(self, event: dict[str, Any]) -> None:
        """Accumulate public history, summary, and folder counters."""
        self.count += 1
        try:
            record = _history_record(event)
            if self.collect_history:
                self.history.append(record)
        except (KeyError, TypeError, ValueError):
            self.history_complete = False

        if self.summary_complete:
            try:
                self.summary.add_event(event)
            except (KeyError, TypeError, ValueError):
                self.summary_complete = False

        try:
            if event.get("legacy_history_recovered"):
                return
            subdir = kind_subdir(str(event["kind"]))
            smiles = str(event.get("reactant_smiles", ""))
            key = (
                subdir,
                smiles_to_dirname(smiles) if smiles else "unknown",
                int(event["iso_class"]),
                int(event["lateral_class"]),
            )
            self.reaction_states.setdefault(key, ReactionEventState()).add(event)
        except (KeyError, TypeError, ValueError):
            # Minimal legacy/test rows are still valid reconciliation records;
            # they simply cannot restore per-reaction metadata.
            pass

    def attach_history_view(
        self,
        path: str | Path,
        *,
        committed_count: int,
        committed_offset: int,
    ) -> None:
        """Attach an O(1)-memory history view after successful reconciliation."""
        if self.history_complete and not self.collect_history:
            self.history = EventHistory(
                path,
                committed_count=committed_count,
                committed_offset=committed_offset,
            )


@dataclass(frozen=True)
class EventLogCommit:
    """Durable ``events.jsonl`` prefix represented by a checkpoint."""

    count: int
    offset: int
    recovery: EventLogRecovery | None = field(
        default=None,
        compare=False,
        repr=False,
    )


def _truncate_event_log(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(int(offset))
        handle.flush()
        os.fsync(handle.fileno())


def _canonicalise_legacy_prefix(
    path: Path,
    *,
    retained_offset: int,
    run_id: str,
) -> int:
    """Atomically stamp a run UUID onto one already-validated legacy prefix."""
    offset = 0
    with atomic_output_path(path) as temporary_path:
        with temporary_path.open("wb") as handle:
            with path.open("rb") as source:
                line_number = 0
                while source.tell() < retained_offset:
                    raw = source.readline(retained_offset - source.tell())
                    line_number += 1
                    if not raw.strip():
                        continue
                    event = _decode_event_line(path, raw, line_number)
                    if event.get("run_id") is None:
                        event["run_id"] = run_id
                    handle.write(json.dumps(event).encode("utf-8") + b"\n")
            offset = handle.tell()
    return offset


def _seed_legacy_history_prefix(
    path: Path,
    history: Iterable[tuple] | None,
    *,
    checkpoint_step: int,
    run_id: str | None,
) -> bool:
    """Create a canonical event prefix from a legacy checkpoint history.

    Legacy checkpoints can outlive their original run directory. Seeding their
    public history before the first continuation ensures the next compact
    checkpoint still points at a cumulative event source of truth.
    """
    if history is None:
        return False
    iterator = iter(history)
    try:
        first = next(iterator)
    except StopIteration:
        return False

    previous_step: int | None = None
    previous_time = 0.0
    count = 0
    with atomic_output_path(path) as temporary_path:
        with temporary_path.open("wb") as handle:
            for history_index, record in enumerate(
                chain((first,), iterator),
                start=1,
            ):
                try:
                    (
                        step,
                        time_s,
                        kind,
                        iso_class,
                        member_index,
                        lateral_class,
                        delta_e,
                        barrier,
                        rate_hz,
                    ) = record[:9]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "legacy checkpoint history row "
                        f"{history_index} is not a nine-field KMC record"
                    ) from exc
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or step < 1
                ):
                    raise ValueError(
                        "legacy checkpoint history row "
                        f"{history_index} has invalid step {step!r}"
                    )
                if previous_step is not None and step != previous_step + 1:
                    raise ValueError(
                        "legacy checkpoint history steps are not consecutive "
                        f"at row {history_index}: expected {previous_step + 1}, "
                        f"got {step}"
                    )
                numeric_time = float(time_s)
                numeric_values = (
                    numeric_time,
                    float(delta_e),
                    float(barrier),
                    float(rate_hz),
                )
                if not all(math.isfinite(value) for value in numeric_values):
                    raise ValueError(
                        "legacy checkpoint history row "
                        f"{history_index} contains non-finite values"
                    )
                if numeric_time < previous_time:
                    raise ValueError(
                        "legacy checkpoint history time decreases at row "
                        f"{history_index}"
                    )
                event = {
                    "schema_version": PERSISTENCE_SCHEMA_VERSION,
                    "step": step,
                    "time_s": numeric_time,
                    "tau_s": numeric_time - previous_time,
                    "kind": str(kind),
                    "direction": None,
                    "reactant_smiles": "",
                    "iso_class": int(iso_class),
                    "member_index": int(member_index),
                    "lateral_class": int(lateral_class),
                    "rate_hz": float(rate_hz),
                    "delta_e_ev": float(delta_e),
                    "barrier_ev": float(barrier),
                    "rate_delta_ev": float(delta_e),
                    "rate_barrier_ev": float(barrier),
                    "rate_energy_basis": "legacy_history",
                    "description": "Recovered from legacy checkpoint history.",
                    "reaction_dir": None,
                    "inputs": [],
                    "outputs": [],
                    "run_id": run_id,
                    "legacy_history_recovered": True,
                }
                handle.write(
                    json.dumps(event, allow_nan=False).encode("utf-8") + b"\n"
                )
                previous_step = step
                previous_time = numeric_time
                count += 1
        if previous_step != int(checkpoint_step):
            raise ValueError(
                "legacy checkpoint history is behind checkpoint step "
                f"{checkpoint_step}: last history step is {previous_step!r}"
            )
    return count > 0


def _decode_event_line(path: Path, raw: bytes, line_number: int) -> dict[str, Any]:
    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot reconcile {path}: invalid event line {line_number}: {exc}"
        ) from exc
    if not isinstance(event, dict):
        raise ValueError(
            f"cannot reconcile {path}: event line {line_number} is not an object"
        )
    return event


def _event_step(path: Path, event: dict[str, Any], line_number: int) -> int:
    """Return a JSON event step without accepting bools or numeric coercion."""
    try:
        step = event["step"]
    except KeyError as exc:
        raise ValueError(
            f"cannot reconcile {path}: event line {line_number} has no valid step"
        ) from exc
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(
            f"cannot reconcile {path}: event line {line_number} has no valid step"
        )
    return step


class _CommittedEventValidator:
    """Streaming validator which never retains serialized event rows."""

    def __init__(
        self,
        path: Path,
        *,
        checkpoint_step: int,
        run_id: str | None,
        allow_missing_run_id: bool,
        collect_history: bool,
    ) -> None:
        self.path = path
        self.checkpoint_step = int(checkpoint_step)
        self.run_id = run_id
        self.allow_missing_run_id = allow_missing_run_id
        self.previous_step: int | None = None
        self.count = 0
        self.missing_run_id = False
        self.recovery = EventLogRecovery(collect_history=collect_history)

    def add(self, event: dict[str, Any], *, line_number: int) -> None:
        path = self.path
        step = _event_step(path, event, line_number)
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError(
                f"cannot reconcile {path}: event steps are not consecutive "
                f"at line {line_number}: expected {self.previous_step + 1}, got {step}"
            )
        if step > self.checkpoint_step:
            raise ValueError(
                f"checkpoint prefix contains event step {step} beyond "
                f"checkpoint step {self.checkpoint_step}"
            )
        event_run_id = event.get("run_id")
        if (
            self.run_id is not None
            and event_run_id != self.run_id
            and not (self.allow_missing_run_id and event_run_id is None)
        ):
            raise ValueError(
                f"cannot reconcile {path}: event line {line_number} run_id "
                f"{event.get('run_id')!r} does not match checkpoint run_id "
                f"{self.run_id!r}"
            )
        self.missing_run_id |= event_run_id is None
        self.previous_step = step
        self.count += 1
        self.recovery.add(event)

    def finish(self) -> None:
        if self.count == 0 and self.checkpoint_step > 0:
            raise ValueError(
                "event log has no committed events for checkpoint step "
                f"{self.checkpoint_step}"
            )
        if (
            self.previous_step is not None
            and self.previous_step != self.checkpoint_step
        ):
            raise ValueError(
                f"event log is behind checkpoint step {self.checkpoint_step}: "
                f"last committed event is step {self.previous_step!r}"
            )


def reconcile_event_log(
    path: str | Path,
    *,
    checkpoint_step: int,
    committed_event_count: int | None,
    committed_event_offset: int | None,
    run_id: str | None = None,
    legacy_history: Iterable[tuple] | None = None,
    collect_history: bool = False,
) -> EventLogCommit:
    """Restore ``events.jsonl`` to the prefix represented by a checkpoint.

    Schema-v4 checkpoints provide an exact byte offset and line count. Older
    checkpoints are reconciled conservatively by retaining the valid prefix up
    through ``checkpoint_step``. If a legacy event log is missing but its
    checkpoint contains public history, that history is canonicalised into a
    new cumulative prefix before continuation.

    By default, recovered history is represented by an O(1)-memory
    :class:`EventHistory`. Set ``collect_history=True`` only when an eager list
    is explicitly required.
    """
    path = Path(path)
    exact = committed_event_count is not None or committed_event_offset is not None
    if (committed_event_count is None) != (committed_event_offset is None):
        raise ValueError(
            "checkpoint event commit is incomplete: count and offset must both be set"
        )
    if not path.is_file():
        if exact:
            expected_count = int(committed_event_count or 0)
            expected_offset = int(committed_event_offset or 0)
            if expected_count < 0 or expected_offset < 0:
                raise ValueError(
                    "checkpoint event commit cannot contain negative count/offset: "
                    f"count={expected_count}, offset={expected_offset}"
                )
            if expected_count > 0 or expected_offset > 0:
                raise ValueError(
                    f"checkpoint commits {expected_count} events at byte "
                    f"offset {expected_offset} but {path} is missing"
                )
            if int(checkpoint_step) > 0:
                raise ValueError(
                    f"checkpoint step {checkpoint_step} cannot commit an empty "
                    "event prefix"
                )
            return EventLogCommit(count=0, offset=0)
        seeded = _seed_legacy_history_prefix(
            path,
            legacy_history,
            checkpoint_step=checkpoint_step,
            run_id=run_id,
        )
        if not seeded:
            return EventLogCommit(count=0, offset=0)

    if exact:
        expected_count = int(committed_event_count or 0)
        expected_offset = int(committed_event_offset or 0)
        size = path.stat().st_size
        if expected_count < 0 or expected_offset < 0 or expected_offset > size:
            raise ValueError(
                "checkpoint event commit is inconsistent with the event log: "
                f"count={expected_count}, offset={expected_offset}, size={size}"
            )
        validator = _CommittedEventValidator(
            path,
            checkpoint_step=checkpoint_step,
            run_id=run_id,
            allow_missing_run_id=False,
            collect_history=collect_history,
        )
        with path.open("rb") as handle:
            line_number = 0
            while handle.tell() < expected_offset:
                remaining = expected_offset - handle.tell()
                raw = handle.readline(remaining)
                line_number += 1
                if not raw.endswith(b"\n"):
                    raise ValueError(
                        "checkpoint event offset does not end at a JSONL boundary"
                    )
                if not raw.strip():
                    continue
                event = _decode_event_line(path, raw, line_number)
                validator.add(event, line_number=line_number)
        if validator.count != expected_count:
            raise ValueError(
                "checkpoint event count does not match its committed byte prefix: "
                f"{expected_count} != {validator.count}"
            )
        validator.finish()
        if size > expected_offset:
            _truncate_event_log(path, expected_offset)
        validator.recovery.attach_history_view(
            path,
            committed_count=expected_count,
            committed_offset=expected_offset,
        )
        return EventLogCommit(
            count=expected_count,
            offset=expected_offset,
            recovery=validator.recovery,
        )

    # Legacy fallback: retain only complete, valid rows through the checkpoint
    # step. A malformed newline-terminated row is corruption; an incomplete
    # final row is an expected crash tail and is discarded.
    validator = _CommittedEventValidator(
        path,
        checkpoint_step=checkpoint_step,
        run_id=run_id,
        allow_missing_run_id=True,
        collect_history=collect_history,
    )
    retained_offset = 0
    truncate_at: int | None = None
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                retained_offset = handle.tell()
                continue
            if not raw.endswith(b"\n"):
                truncate_at = line_start
                break
            try:
                event = _decode_event_line(path, raw, line_number)
            except ValueError:
                if not raw.endswith(b"\n"):
                    truncate_at = line_start
                    break
                raise
            step = _event_step(path, event, line_number)
            if step > int(checkpoint_step):
                truncate_at = line_start
                break
            validator.add(event, line_number=line_number)
            retained_offset = handle.tell()

    validator.finish()
    if run_id is not None and validator.missing_run_id:
        # Older event rows predate the run UUID. Canonicalise the fully
        # validated retained prefix before append so the exact v3 checkpoint
        # written by this continuation remains resumable later.
        retained_offset = _canonicalise_legacy_prefix(
            path,
            retained_offset=retained_offset,
            run_id=run_id,
        )
    elif truncate_at is not None or path.stat().st_size > retained_offset:
        _truncate_event_log(path, retained_offset)
    validator.recovery.attach_history_view(
        path,
        committed_count=validator.count,
        committed_offset=retained_offset,
    )
    return EventLogCommit(
        count=validator.count,
        offset=retained_offset,
        recovery=validator.recovery,
    )


__all__ = [
    "EventLogCommit",
    "EventHistory",
    "EventLogRecovery",
    "ReactionEventState",
    "reconcile_event_log",
]
