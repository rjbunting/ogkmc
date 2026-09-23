"""Checkpoint/restart support for long KMC runs."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase import Atoms

from autokmc.io._files import atomic_output_path
from autokmc.io.atoms import copy_atoms_with_results
from autokmc.utils.telemetry import instrument


CHECKPOINT_SCHEMA_VERSION = "4"

# These graph attributes are pure acceleration structures.  Persisting them can
# dwarf the scientific state (``surface_apsp`` is potentially quadratic), and
# every producer already rebuilds them lazily after a cache miss.
_TRANSIENT_GRAPH_CACHE_KEYS = frozenset({
	"surface_apsp",
	"_surface_shells_cache",
	"_clique_position_index_cache",
	"_surface_atoms_array_cache",
})


@dataclass
class CheckpointState:
	"""Serializable restart bundle.

	Live calculators are stripped before writing and rebuilt from the run config
	on resume. Safe single-point energy/force snapshots may remain attached to
	optimized structures.
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
	# Exact committed prefix of events.jsonl represented by this state.  These
	# are optional only for checkpoints written before schema version 3.
	committed_event_count: int | None = None
	committed_event_offset: int | None = None
	# Optional for older v4 snapshots and runs without trajectory output.
	committed_trajectory_offset: int | None = None


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
	"""Deep-copy state, replacing live calculators with cached results."""
	if _memo is None:
		_memo = {}
	oid = id(obj)
	if oid in _memo:
		return _memo[oid]
	if isinstance(obj, Atoms):
		out = copy_atoms_with_results(obj)
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
		# Give ``deepcopy`` the same memo used by the container branches.  This
		# is essential for NetworkX graphs: their graph metadata and reverse
		# indexes contain the same site objects that are also stored in the
		# top-level checkpoint lists.
		out = copy.deepcopy(obj, _memo)
	except Exception:
		return obj
	_memo[oid] = out
	_clear_calculators_inplace(out)
	return out


def _clear_calculators_inplace(obj, *, _seen: set[int] | None = None) -> None:
	"""Remove live calculators without copying an already-copied graph."""
	if _seen is None:
		_seen = set()
	identity = id(obj)
	if identity in _seen:
		return
	_seen.add(identity)
	if isinstance(obj, Atoms):
		obj.calc = copy_atoms_with_results(obj).calc
		return
	if isinstance(obj, dict):
		for value in obj.values():
			_clear_calculators_inplace(value, _seen=_seen)
		return
	if isinstance(obj, (list, tuple, set)):
		for value in obj:
			_clear_calculators_inplace(value, _seen=_seen)
		return
	d = getattr(obj, "__dict__", None)
	if not isinstance(d, dict):
		return
	for key, value in list(d.items()):
		if key == "calc":
			setattr(obj, key, None)
		else:
				_clear_calculators_inplace(value, _seen=_seen)


def _graph_without_transient_caches(graph):
	"""Return a shallow graph shell excluding lazily reconstructible caches."""
	if not hasattr(graph, "graph") or not isinstance(graph.graph, dict):
		return graph
	out = copy.copy(graph)
	out.graph = dict(graph.graph)
	for key in _TRANSIENT_GRAPH_CACHE_KEYS:
		out.graph.pop(key, None)
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
	committed_event_count: int | None = None,
	committed_event_offset: int | None = None,
	committed_trajectory_offset: int | None = None,
	metadata: dict[str, Any] | None = None,
) -> CheckpointState:
	# Copy one persistent root mapping in a single traversal.  This both keeps a
	# site shared by graph indexes and channel lists as one object *and* avoids
	# memoizing the id of a short-lived ``list(...)`` temporary.  CPython may
	# immediately reuse such an id for the next root, which can otherwise alias
	# unrelated collections (for example bond_sites and reactants).
	roots = {
		"graph": _graph_without_transient_caches(graph),
		"adsorbate_sites": list(adsorbate_sites or []),
		"diffusion_sites": list(diffusion_sites or []),
		"bond_sites": list(bond_sites or []),
		"reactants": list(reactants or []),
		"history": list(history or []),
		"reaction_counts": dict(reaction_counts or {}),
		"rng_state": rng_state,
	}
	stripped = _strip_calculators(roots)
	return CheckpointState(
		schema_version=CHECKPOINT_SCHEMA_VERSION,
		step=int(step),
		time_s=float(time_s),
		graph=stripped["graph"],
		adsorbate_sites=stripped["adsorbate_sites"],
		diffusion_sites=stripped["diffusion_sites"],
		bond_sites=stripped["bond_sites"],
		reactants=stripped["reactants"],
		frozen_indices=(None if frozen_indices is None else list(frozen_indices)),
		history=stripped["history"],
		reaction_counts=stripped["reaction_counts"],
		metadata={
			"written_at": datetime.now(timezone.utc).isoformat(),
			**dict(metadata or {}),
		},
		rng_state=stripped["rng_state"],
		committed_event_count=(
			None if committed_event_count is None else int(committed_event_count)
		),
		committed_event_offset=(
			None if committed_event_offset is None else int(committed_event_offset)
		),
		committed_trajectory_offset=(
			None if committed_trajectory_offset is None else int(committed_trajectory_offset)
		),
	)


@instrument("checkpoint.save")
def save_checkpoint(path: str | Path, state: CheckpointState | dict) -> Path:
	"""Write *state* to *path* atomically and return the final path."""
	path = Path(path)
	with atomic_output_path(path) as temporary:
		with temporary.open("wb") as fp:
			pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)
	return path


def load_checkpoint(path: str | Path) -> CheckpointState:
	"""Load and validate a checkpoint created by :func:`save_checkpoint`."""
	with Path(path).open("rb") as fp:
		state = pickle.load(fp)
	if isinstance(state, dict):
		state = CheckpointState(**state)
	if not isinstance(state, CheckpointState):
		raise TypeError(f"checkpoint {path!s} does not contain a CheckpointState")
	legacy_version = str(state.schema_version)
	if legacy_version == "1":
		# Version 1 did not persist the random-number-generator state.  It can
		# still be resumed, but only later continuations are bitwise
		# equivalent to an uninterrupted run.
		state.rng_state = None
	if legacy_version in {"1", "2"}:
		# Version 3 records the exact committed events.jsonl prefix.  Legacy
		# checkpoints fall back to step-based reconciliation on resume.
		state.committed_event_count = None
		state.committed_event_offset = None
	if legacy_version in {"1", "2", "3"}:
		# Version 4 treats events.jsonl as the canonical event history and omits
		# reconstructible graph caches from newly-written checkpoints.
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

	def should_write(self, *, step: int, force: bool = False) -> bool:
		"""Return whether *step* is a configured checkpoint boundary."""
		return bool(
			force
			or (int(step) > 0 and (int(step) % self.every_n_steps) == 0)
		)

	def maybe_write(self, *, step: int, force: bool = False, **state_kwargs) -> Path | None:
		if not self.should_write(step=step, force=force):
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
