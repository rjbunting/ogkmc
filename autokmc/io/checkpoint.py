"""Checkpoint/restart support for long KMC runs."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase import Atoms


CHECKPOINT_SCHEMA_VERSION = "2"


@dataclass
class CheckpointState:
	"""Serializable restart bundle.

	The calculators are intentionally stripped before writing; they are rebuilt
	from the run config on resume.
	"""
	schema_version: str
	step: int
	time_s: float
	graph: Any
	adsorbate_sites: list
	diffusion_sites: list
	bond_sites: list
	reactants: list
	frozen_indices: list[int] | None
	history: list
	reaction_counts: dict
	metadata: dict[str, Any]
	rng_state: dict[str, Any] | None = None


def _is_hashable(value) -> bool:
	try:
		hash(value)
	except TypeError:
		return False
	return True


def _hashable_fallback(value):
	try:
		out = copy.deepcopy(value)
	except Exception:
		out = value
	if _is_hashable(out):
		return out
	return value


def _strip_calculators(obj, *, _memo: dict[int, Any] | None = None):
	"""Return a deep-copied object with ASE calculator handles removed."""
	if _memo is None:
		_memo = {}
	oid = id(obj)
	if oid in _memo:
		return _memo[oid]
	if isinstance(obj, Atoms):
		out = obj.copy()
		out.calc = None
		_memo[oid] = out
		return out
	if isinstance(obj, dict):
		out = {}
		_memo[oid] = out
		for k, v in obj.items():
			key = _strip_calculators(k, _memo=_memo)
			if not _is_hashable(key):
				key = _hashable_fallback(k)
			out[key] = _strip_calculators(v, _memo=_memo)
		return out
	if isinstance(obj, list):
		out = []
		_memo[oid] = out
		out.extend(_strip_calculators(v, _memo=_memo) for v in obj)
		return out
	if isinstance(obj, tuple):
		out = tuple(_strip_calculators(v, _memo=_memo) for v in obj)
		_memo[oid] = out
		return out
	if isinstance(obj, set):
		out = set()
		_memo[oid] = out
		for v in obj:
			item = _strip_calculators(v, _memo=_memo)
			if not _is_hashable(item):
				item = _hashable_fallback(v)
			out.add(item)
		return out
	try:
		out = copy.deepcopy(obj)
	except Exception:
		return obj
	_memo[oid] = out
	d = getattr(out, "__dict__", None)
	if isinstance(d, dict):
		for key, val in list(d.items()):
			if key == "calc":
				setattr(out, key, None)
			else:
				setattr(out, key, _strip_calculators(val, _memo=_memo))
	return out


def make_checkpoint_state(
	*,
	step: int,
	time_s: float,
	graph,
	adsorbate_sites: list,
	diffusion_sites: list | None = None,
	bond_sites: list | None = None,
	reactants: list | None = None,
	frozen_indices: list[int] | None = None,
	history: list | None = None,
	reaction_counts: dict | None = None,
	rng_state: dict[str, Any] | None = None,
	metadata: dict[str, Any] | None = None,
) -> CheckpointState:
	return CheckpointState(
		schema_version=CHECKPOINT_SCHEMA_VERSION,
		step=int(step),
		time_s=float(time_s),
		graph=_strip_calculators(graph),
		adsorbate_sites=_strip_calculators(list(adsorbate_sites or [])),
		diffusion_sites=_strip_calculators(list(diffusion_sites or [])),
		bond_sites=_strip_calculators(list(bond_sites or [])),
		reactants=_strip_calculators(list(reactants or [])),
		frozen_indices=(None if frozen_indices is None else list(frozen_indices)),
		history=list(history or []),
		reaction_counts=dict(reaction_counts or {}),
		metadata={
			"written_at": datetime.now(timezone.utc).isoformat(),
			**dict(metadata or {}),
		},
		rng_state=_strip_calculators(rng_state),
	)


def save_checkpoint(path: str | Path, state: CheckpointState | dict) -> Path:
	"""Write *state* to *path* atomically and return the final path."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_suffix(path.suffix + ".tmp")
	with tmp.open("wb") as fp:
		pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)
	tmp.replace(path)
	return path


def load_checkpoint(path: str | Path) -> CheckpointState:
	"""Load and validate a checkpoint created by :func:`save_checkpoint`."""
	with Path(path).open("rb") as fp:
		state = pickle.load(fp)
	if isinstance(state, dict):
		state = CheckpointState(**state)
	if not isinstance(state, CheckpointState):
		raise TypeError(f"checkpoint {path!s} does not contain a CheckpointState")
	if state.schema_version == "1":
		# Version 1 did not persist the random-number-generator state.  It can
		# still be resumed, but only version-2 continuations are bitwise
		# equivalent to an uninterrupted run.
		state.rng_state = None
		state.schema_version = CHECKPOINT_SCHEMA_VERSION
	if state.schema_version != CHECKPOINT_SCHEMA_VERSION:
		raise ValueError(
			f"checkpoint schema_version={state.schema_version!r} does not match "
			f"{CHECKPOINT_SCHEMA_VERSION!r}"
		)
	return state


class CheckpointWriter:
	"""Periodic checkpoint writer used by the KMC loop."""

	def __init__(
		self,
		path: str | Path,
		*,
		every_n_steps: int = 1,
		metadata: dict[str, Any] | None = None,
	):
		self.path = Path(path)
		self.every_n_steps = max(1, int(every_n_steps or 1))
		self.metadata = dict(metadata or {})
		self.last_path: Path | None = None

	def maybe_write(self, *, step: int, force: bool = False, **state_kwargs) -> Path | None:
		if not force and (int(step) <= 0 or (int(step) % self.every_n_steps) != 0):
			return None
		state = make_checkpoint_state(step=step, metadata=self.metadata, **state_kwargs)
		self.last_path = save_checkpoint(self.path, state)
		return self.last_path


__all__ = [
	"CHECKPOINT_SCHEMA_VERSION",
	"CheckpointState",
	"CheckpointWriter",
	"make_checkpoint_state",
	"save_checkpoint",
	"load_checkpoint",
]
