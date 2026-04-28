"""
autokmc.persistence
===================
Persistence of KMC events, trajectories and run summaries.

Output layout (under ``output.dir``)
------------------------------------

::

    <output.dir>/
    ├── events.jsonl          # one JSON line per executed KMC event
    ├── summary.json          # per-reaction-type aggregates + run metadata
    ├── kmc.extxyz            # extended-XYZ trajectory (initial + every Nth state)
    └── reactions/
        ├── adsorption/
        │   ├── iso0_lat0/
        │   │   ├── reaction.json     # description + energies + ΔE / barrier / rate
        │   │   ├── occupied.extxyz   # relaxed atoms used to compute E_occ
        │   │   └── unoccupied.extxyz # relaxed atoms used to compute E_unocc
        │   ├── iso0_lat1/
        │   │   └── …
        │   └── …
        └── diffusion/
            ├── diff_iso0_lat0/
            │   ├── reaction.json
            │   ├── state_a.extxyz    # relaxed endpoint A
            │   ├── state_b.extxyz    # relaxed endpoint B
            │   └── ts.extxyz         # NEB transition-state image
            └── …

Reactions are split by **kind** into a sub-folder of ``reactions/`` so that
adsorption / desorption events and diffusion (hop) events do not collide on
the ``(iso, lateral)`` namespace.  ``adsorption/`` and ``desorption/`` share
the same sub-folder (``adsorption/``) since they are forward / reverse
directions of the same lateral class.

Each unique reaction is identified by ``(iso_class, lateral_class)`` — it
is the lateral interaction class that owns the actual relaxed structures
used in the energy calculation.  The atoms files are the **as-calculated**
structures from
:func:`autokmc.check_adsorbate_sites.check_site_stability`, not snapshots
of the live KMC graph.  They are written exactly **once** per
``(iso, lat)`` pair (the first time the KMC fires that reaction); every
subsequent firing simply appends a thin row to ``events.jsonl`` referring
to the existing folder.

For an adsorption event the *initial* state is ``unoccupied.extxyz`` and
the *final* state is ``occupied.extxyz``.  The reverse holds for
desorption — both directions re-use the same per-lateral-class folder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import networkx as nx

from ase import Atoms
from ase.io import write as ase_write

from autokmc.constants import (
    PERSISTENCE_SCHEMA_VERSION,
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,         # noqa: F401  (re-exported for convenience)
    TRAJECTORY_FILENAME,      # noqa: F401  (re-exported for convenience)
    REACTIONS_DIR,
    REACTION_DESCRIPTION_FMT,
    TRAJ_DUMP_EVERY,
)
from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_atoms_copy(atoms: Atoms) -> Atoms:
    """Return a copy of *atoms* with a sanitised calculator.

    If the attached calculator has result arrays (forces, energies, …) whose
    first dimension does not match ``len(atoms)`` — which can happen when a
    NequIP / MACE calculator caches results from a previous, smaller system —
    the calculator is stripped so that ASE's extxyz writer does not raise a
    broadcast error.  Energy is preserved as ``atoms.info["energy"]`` when
    available.
    """
    snap = atoms.copy()
    calc = snap.calc
    if calc is None:
        return snap
    try:
        results = getattr(calc, "results", {}) or {}
        for key, val in results.items():
            if hasattr(val, "__len__") and len(val) != len(snap):
                # Stale results — strip the calculator entirely.
                # Try to salvage the scalar energy first.
                e = results.get("energy")
                if e is not None:
                    snap.info.setdefault("energy", float(e))
                snap.calc = None
                return snap
    except Exception:
        snap.calc = None
    return snap


# ---------------------------------------------------------------------------
# Atoms <-> graph helper
# ---------------------------------------------------------------------------

def atoms_from_graph(G: nx.Graph) -> Atoms:
    """Build an :class:`~ase.Atoms` from the live graph state.

    Includes every ``type ∈ {"bulk", "surface"}`` node (sorted by their
    original ASE atom index) plus every ``type == "adsorbate"`` node whose
    ``occupied`` flag is set.  Anchor bookkeeping nodes are skipped.
    """
    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True)
         if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )
    ads_nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("type") == "adsorbate" and d.get("occupied", False)
    )

    all_ids = slab_nodes + ads_nodes
    symbols   = [G.nodes[n]["element"]  for n in all_ids]
    positions = [G.nodes[n]["position"] for n in all_ids]

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)

    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)


# ---------------------------------------------------------------------------
# ReactionRecord (one persisted event row in events.jsonl)
# ---------------------------------------------------------------------------

@dataclass
class ReactionRecord:
    """One persisted KMC event row.

    Heavy data (atoms) lives in the per-lateral-class folder referenced by
    ``reaction_dir``; this row is intentionally lightweight so
    ``events.jsonl`` stays diff-friendly and streamable.
    """
    schema_version:  str
    step:            int
    time_s:          float
    tau_s:           float
    kind:            str
    reactant_smiles: str
    iso_class:       int
    member_index:    int
    lateral_class:   int
    rate_hz:         float
    delta_e_ev:      float
    barrier_ev:      float
    description:     str
    reaction_dir:    str  # relative to output.dir, e.g. "reactions/iso0_lat3"

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


# ---------------------------------------------------------------------------
# ReactionWriter
# ---------------------------------------------------------------------------

#: Map ``reaction.kind`` → sub-folder name under ``reactions/``.  Adsorption
#: and desorption are forward / reverse of the same lateral class so they
#: share a single ``adsorption/`` folder; diffusion gets its own.
KIND_SUBDIR: dict[str, str] = {
    "adsorption": "adsorption",
    "desorption": "adsorption",
    "diffusion":  "diffusion",
}

def _kind_subdir(kind: str) -> str:
    return KIND_SUBDIR.get(str(kind), str(kind))


def _reaction_folder_name(iso_class: int, lateral_class: int) -> str:
    return f"iso{int(iso_class)}_lat{int(lateral_class)}"


def _diffusion_folder_name(iso_class: int, lateral_class: int) -> str:
    return f"diff_iso{int(iso_class)}_lat{int(lateral_class)}"


def _reaction_relative_dir(kind: str, iso: int, lat: int) -> str:
    """Return ``"reactions/<sub>/<folder>"`` (POSIX-style, for JSON output)."""
    sub = _kind_subdir(kind)
    folder = (
        _diffusion_folder_name(iso, lat) if sub == "diffusion"
        else _reaction_folder_name(iso, lat)
    )
    return f"{REACTIONS_DIR}/{sub}/{folder}"


def _json_safe(o):
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(x) for x in o]
    return o


class ReactionWriter:
    """Writes per-unique-reaction folders + an events JSONL.

    Each unique ``(iso_class, lateral_class)`` pair gets a folder under
    ``output_dir/reactions/`` containing:

    * ``occupied.extxyz``   — relaxed atoms behind ``E_occupied``.
    * ``unoccupied.extxyz`` — relaxed atoms behind ``E_unoccupied``.
    * ``reaction.json``     — description + energies + ΔE / barrier / rate
      / fired-event count.

    The folder is materialised lazily — it is written the **first time**
    a reaction with that ``(iso, lat)`` is fired, then re-used by every
    subsequent firing of either direction.  ``reaction.json`` is updated
    in place (counts, last_step) on every event, but the .extxyz files
    are written exactly once.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        reactions_filename: str = REACTIONS_FILENAME,
        reactions_dir: str = REACTIONS_DIR,
        calculator_meta: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reactions_root = self.output_dir / reactions_dir
        self.reactions_root.mkdir(parents=True, exist_ok=True)

        self._jsonl_path: Path = self.output_dir / reactions_filename
        self._fp: TextIO | None = self._jsonl_path.open("w", encoding="utf-8")
        self._calc_meta: dict[str, Any] = dict(calculator_meta or {})
        self._n_written: int = 0

        # Per (sub, iso, lat) bookkeeping for the on-disk reaction.json files.
        self._folder_meta: dict[tuple[str, int, int], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def n_written(self) -> int:
        return self._n_written

    @property
    def n_unique_reactions(self) -> int:
        return len(self._folder_meta)

    # ------------------------------------------------------------------
    def _ensure_reaction_folder(self, reaction) -> Path:
        """Create + populate the per-lateral-class folder if not yet done.

        Folders are nested by ``KIND_SUBDIR[reaction.kind]`` so that
        adsorption/desorption events live under
        ``reactions/adsorption/iso{X}_lat{Y}/`` and diffusion events under
        ``reactions/diffusion/diff_iso{X}_lat{Y}/``.
        """
        iso = int(reaction.site.iso_class)
        lat = int(reaction.lateral_class.lateral_class)
        sub = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        key = (sub, iso, lat)

        sub_root = self.reactions_root / sub
        if sub == "diffusion":
            folder = sub_root / _diffusion_folder_name(iso, lat)
        else:
            folder = sub_root / _reaction_folder_name(iso, lat)

        if key in self._folder_meta:
            return folder

        folder.mkdir(parents=True, exist_ok=True)

        lc = reaction.lateral_class
        if sub == "diffusion":
            atoms_a  = getattr(lc, "atoms_a",  None)
            atoms_b  = getattr(lc, "atoms_b",  None)
            atoms_ts = getattr(lc, "atoms_ts", None)
            if atoms_a is not None:
                ase_write(folder / "state_a.extxyz", _safe_atoms_copy(atoms_a), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: diffusion lateral_class iso=%d lat=%d "
                    "has no atoms_a — state_a.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_b is not None:
                ase_write(folder / "state_b.extxyz", _safe_atoms_copy(atoms_b), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: diffusion lateral_class iso=%d lat=%d "
                    "has no atoms_b — state_b.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_ts is not None:
                ase_write(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: diffusion lateral_class iso=%d lat=%d "
                    "has no atoms_ts — ts.extxyz will not be written.",
                    iso, lat,
                )
            # Optional full NEB band (one frame per image: A, n_images
            # intermediates, then B).  Stamped on by check_diffusion_stability
            # only when persist_neb_path=True.
            atoms_neb_path = getattr(lc, "atoms_neb_path", None)
            if atoms_neb_path:
                ase_write(
                    folder / "neb_path.extxyz",
                    [_safe_atoms_copy(im) for im in atoms_neb_path],
                    format="extxyz",
                )
        else:
            # Stamped onto the lateral class by check_site_stability().
            atoms_occ   = getattr(lc, "atoms_occupied",   None)
            atoms_unocc = getattr(lc, "atoms_unoccupied", None)
            if atoms_occ is not None:
                ase_write(folder / "occupied.extxyz",   _safe_atoms_copy(atoms_occ),   format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: lateral_class iso=%d lat=%d has no "
                    "atoms_occupied — occupied.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_unocc is not None:
                ase_write(folder / "unoccupied.extxyz", _safe_atoms_copy(atoms_unocc), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: lateral_class iso=%d lat=%d has no "
                    "atoms_unoccupied — unoccupied.extxyz will not be written.",
                    iso, lat,
                )

        self._folder_meta[key] = {
            "count":      0,
            "first_step": None,
            "last_step":  None,
        }
        return folder

    # ------------------------------------------------------------------
    def _write_reaction_json(
        self,
        folder: Path,
        reaction,
        step: int,
        gas_energies: dict[str, float] | None,
        *,
        fired: bool,
    ) -> None:
        iso = int(reaction.site.iso_class)
        lat = int(reaction.lateral_class.lateral_class)
        sub = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        meta = self._folder_meta[(sub, iso, lat)]
        if fired:
            meta["count"] += 1
            if meta["first_step"] is None:
                meta["first_step"] = int(step)
            meta["last_step"] = int(step)

        smiles = getattr(reaction.site, "reactant", "")
        lc = reaction.lateral_class

        if sub == "diffusion":
            # Per-direction barriers from the *raw* NEB TS energy.  The
            # KMC-side barrier is floored at EA_MIN (see kmc_adsorption.EA_MIN
            # docstring) but the *raw* values are persisted unchanged so the
            # floor is auditable from disk.
            from autokmc.kmc_adsorption import EA_MIN as _EA_MIN
            e_a  = getattr(lc, "energy_a",  None)
            e_b  = getattr(lc, "energy_b",  None)
            e_ts = getattr(lc, "energy_ts", None)
            if e_a is not None and e_ts is not None:
                ea_fwd_raw = float(e_ts) - float(e_a)
                ea_fwd_kmc = max(_EA_MIN, ea_fwd_raw)
            else:
                ea_fwd_raw = ea_fwd_kmc = None
            if e_b is not None and e_ts is not None:
                ea_rev_raw = float(e_ts) - float(e_b)
                ea_rev_kmc = max(_EA_MIN, ea_rev_raw)
            else:
                ea_rev_raw = ea_rev_kmc = None

            payload = {
                "schema_version":   PERSISTENCE_SCHEMA_VERSION,
                "kind":             "diffusion",
                "iso_class":        iso,
                "lateral_class":    lat,
                "reactant_smiles":  smiles,
                "kind_directions":  ["a_to_b", "b_to_a"],
                "description":      REACTION_DESCRIPTION_FMT.format(
                    kind     = reaction.kind,
                    smiles   = smiles,
                    iso      = iso,
                    member   = reaction.member_index,
                    lateral  = lat,
                    delta_e  = float(reaction.delta_e),
                    barrier  = float(reaction.barrier),
                    rate     = float(reaction.rate),
                ),
                "energies_ev": {
                    "state_a":      None if e_a  is None else float(e_a),
                    "state_b":      None if e_b  is None else float(e_b),
                    "transition":   None if e_ts is None else float(e_ts),
                },
                "barriers_ev": {
                    "forward_raw": ea_fwd_raw,
                    "forward_kmc": ea_fwd_kmc,
                    "reverse_raw": ea_rev_raw,
                    "reverse_kmc": ea_rev_kmc,
                    "ea_min_floor": _EA_MIN,
                },
                "last_event": (
                    {
                        "kind":       str(reaction.kind),
                        "direction":  getattr(reaction, "direction", None),
                        "delta_e_ev": float(reaction.delta_e),
                        "barrier_ev": float(reaction.barrier),
                        "rate_hz":    float(reaction.rate),
                        "step":       int(step),
                    }
                    if fired else None
                ),
                "stats": {
                    "count":      int(meta["count"]),
                    "first_step": meta["first_step"],
                    "last_step":  meta["last_step"],
                },
                "atoms": {
                    "state_a":    "state_a.extxyz",
                    "state_b":    "state_b.extxyz",
                    "transition": "ts.extxyz",
                    "neb_path":   ("neb_path.extxyz"
                                   if getattr(lc, "atoms_neb_path", None)
                                   else None),
                },
                "calculator": dict(self._calc_meta),
            }
        else:
            e_gas  = float((gas_energies or {}).get(smiles, float("nan")))
            e_occ   = getattr(lc, "energy_occupied",   None)
            e_unocc = getattr(lc, "energy_unoccupied", None)

            payload = {
                "schema_version":   PERSISTENCE_SCHEMA_VERSION,
                "kind":             "adsorption",
                "iso_class":        iso,
                "lateral_class":    lat,
                "reactant_smiles":  smiles,
                "kind_directions":  ["adsorption", "desorption"],
                "description":      REACTION_DESCRIPTION_FMT.format(
                    kind     = reaction.kind,
                    smiles   = smiles,
                    iso      = iso,
                    member   = reaction.member_index,
                    lateral  = lat,
                    delta_e  = float(reaction.delta_e),
                    barrier  = float(reaction.barrier),
                    rate     = float(reaction.rate),
                ),
                "energies_ev": {
                    "occupied":   None if e_occ   is None else float(e_occ),
                    "unoccupied": None if e_unocc is None else float(e_unocc),
                    "gas_phase":  e_gas,
                },
                "last_event": (
                    {
                        "kind":       str(reaction.kind),
                        "delta_e_ev": float(reaction.delta_e),
                        "barrier_ev": float(reaction.barrier),
                        "rate_hz":    float(reaction.rate),
                        "step":       int(step),
                    }
                    if fired else None
                ),
                "stats": {
                    "count":      int(meta["count"]),
                    "first_step": meta["first_step"],
                    "last_step":  meta["last_step"],
                },
                "atoms": {
                    "occupied":   "occupied.extxyz",
                    "unoccupied": "unoccupied.extxyz",
                },
                "calculator": dict(self._calc_meta),
            }
        with (folder / "reaction.json").open("w", encoding="utf-8") as fp:
            json.dump(_json_safe(payload), fp, indent=2)

    # ------------------------------------------------------------------
    def ensure_reaction(
        self,
        reaction,
        *,
        step: int = 0,
        gas_energies: dict[str, float] | None = None,
    ) -> Path:
        """Ensure the on-disk folder + ``reaction.json`` exist for *reaction*.

        Idempotent: the first call materialises the folder, writes
        ``occupied.extxyz`` / ``unoccupied.extxyz`` and an initial
        ``reaction.json`` (with ``stats.count = 0``).  Subsequent calls
        with the same ``(iso_class, lateral_class)`` key are no-ops.

        Used by the KMC driver to persist **every** reaction in the
        current applicable list — not just the one chosen for execution —
        so the run directory mirrors the full discovered reaction network.
        """
        iso = int(reaction.site.iso_class)
        lat = int(reaction.lateral_class.lateral_class)
        sub = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        if (sub, iso, lat) in self._folder_meta:
            sub_root = self.reactions_root / sub
            if sub == "diffusion":
                return sub_root / _diffusion_folder_name(iso, lat)
            return sub_root / _reaction_folder_name(iso, lat)
        folder = self._ensure_reaction_folder(reaction)
        self._write_reaction_json(folder, reaction, step, gas_energies, fired=False)
        return folder

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        step: int,
        time_s: float,
        tau_s: float,
        reaction,
        gas_energies: dict[str, float] | None = None,
        # Backwards-compat — ignored under the new per-reaction-folder layout.
        atoms_initial: Atoms | None = None,
        atoms_final:   Atoms | None = None,
    ) -> ReactionRecord:
        """Persist one KMC event (lazy folder creation + JSONL append)."""
        if self._fp is None:
            raise RuntimeError("ReactionWriter has been closed")

        smiles = getattr(reaction.site, "reactant", "")
        folder = self._ensure_reaction_folder(reaction)
        self._write_reaction_json(folder, reaction, step, gas_energies, fired=True)

        description = REACTION_DESCRIPTION_FMT.format(
            kind     = reaction.kind,
            smiles   = smiles,
            iso      = reaction.site.iso_class,
            member   = reaction.member_index,
            lateral  = reaction.lateral_class.lateral_class,
            delta_e  = float(reaction.delta_e),
            barrier  = float(reaction.barrier),
            rate     = float(reaction.rate),
        )

        rec = ReactionRecord(
            schema_version  = PERSISTENCE_SCHEMA_VERSION,
            step            = int(step),
            time_s          = float(time_s),
            tau_s           = float(tau_s),
            kind            = str(reaction.kind),
            reactant_smiles = str(smiles),
            iso_class       = int(reaction.site.iso_class),
            member_index    = int(reaction.member_index),
            lateral_class   = int(reaction.lateral_class.lateral_class),
            rate_hz         = float(reaction.rate),
            delta_e_ev      = float(reaction.delta_e),
            barrier_ev      = float(reaction.barrier),
            description     = description,
            reaction_dir    = str(folder.relative_to(self.output_dir)),
        )

        self._fp.write(json.dumps(rec.to_jsonable()) + "\n")
        self._fp.flush()
        self._n_written += 1
        return rec

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, *exc):  # pragma: no cover
        self.close()


# ---------------------------------------------------------------------------
# TrajectoryWriter — extended XYZ append
# ---------------------------------------------------------------------------

class TrajectoryWriter:
    """Periodic extended-XYZ dumper.

    Frames are appended to a single ``.extxyz`` file via
    :func:`ase.io.write` with ``append=True``.  Extended XYZ is
    human-readable, ASE-native, and re-attaches to any calculator via
    :func:`ase.io.read`.
    """

    def __init__(self, output_path: str | Path, *, dump_every: int = TRAJ_DUMP_EVERY):
        self.output_path = Path(output_path)
        self.dump_every  = int(dump_every)
        self._n_frames: int = 0
        if self.dump_every > 0:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any pre-existing file from a previous run.
            self.output_path.write_text("")

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def enabled(self) -> bool:
        return self.dump_every > 0

    def maybe_write(self, atoms: Atoms, *, step: int) -> bool:
        if not self.enabled:
            return False
        if step != 0 and (step % self.dump_every) != 0:
            return False
        return self._write(atoms, step=step)

    def write(self, atoms: Atoms, *, step: int = 0) -> bool:
        if not self.enabled:
            return False
        return self._write(atoms, step=step)

    def _write(self, atoms: Atoms, *, step: int) -> bool:
        snap = atoms.copy()
        snap.info["kmc_step"] = int(step)
        ase_write(self.output_path, snap, format="extxyz", append=True)
        self._n_frames += 1
        return True

    def close(self) -> None:
        # Nothing to flush — ase_write closes the file each call.
        return

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, *exc):  # pragma: no cover
        self.close()


# ---------------------------------------------------------------------------
# ReactionSummary
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "min":  float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=0)),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


class ReactionSummary:
    """Aggregator for per-reaction-type statistics.

    Reactions are bucketed by ``(kind, reactant_smiles, iso_class,
    lateral_class)``.
    """

    __slots__ = ("_buckets", "_total_by_kind", "_n", "_first_step", "_last_step")

    def __init__(self):
        self._buckets: dict[tuple, dict[str, list[float]]] = {}
        self._total_by_kind: dict[str, int] = {}
        self._n: int = 0
        self._first_step: dict[tuple, int] = {}
        self._last_step:  dict[tuple, int] = {}

    def add(self, reaction, *, step: int) -> None:
        smiles = getattr(reaction.site, "reactant", "")
        key = (
            str(reaction.kind),
            str(smiles),
            int(reaction.site.iso_class),
            int(reaction.lateral_class.lateral_class),
        )
        b = self._buckets.setdefault(
            key, {"rate": [], "delta_e": [], "barrier": []},
        )
        b["rate"].append(float(reaction.rate))
        b["delta_e"].append(float(reaction.delta_e))
        b["barrier"].append(float(reaction.barrier))

        self._total_by_kind[key[0]] = self._total_by_kind.get(key[0], 0) + 1
        self._n += 1
        self._first_step.setdefault(key, int(step))
        self._last_step[key] = int(step)

    def to_dict(
        self,
        *,
        run_meta: dict[str, Any] | None = None,
        final_occupancy: dict[int, int] | None = None,
    ) -> dict[str, Any]:
        by_type: list[dict[str, Any]] = []
        for key, b in self._buckets.items():
            kind, smiles, iso, lat = key
            by_type.append({
                "kind":             kind,
                "reactant_smiles":  smiles,
                "iso_class":        iso,
                "lateral_class":    lat,
                "reaction_dir":     _reaction_relative_dir(kind, iso, lat),
                "count":            len(b["rate"]),
                "first_step":       self._first_step[key],
                "last_step":        self._last_step[key],
                "rate_hz":          _stats(b["rate"]),
                "delta_e_ev":       _stats(b["delta_e"]),
                "barrier_ev":       _stats(b["barrier"]),
            })
        by_type.sort(key=lambda d: (-d["count"], d["kind"], d["iso_class"],
                                    d["lateral_class"]))

        return {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "run":            dict(run_meta or {}),
            "totals": {
                "reactions":             self._n,
                "by_kind":               dict(self._total_by_kind),
                "unique_reaction_types": len(self._buckets),
            },
            "by_reaction_type": by_type,
            "final_occupancy":  {str(k): int(v)
                                 for k, v in (final_occupancy or {}).items()},
        }

    def write(
        self,
        path: str | Path,
        *,
        run_meta: dict[str, Any] | None = None,
        final_occupancy: dict[int, int] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(run_meta=run_meta, final_occupancy=final_occupancy)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        return path

    @property
    def n(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Run-meta helper
# ---------------------------------------------------------------------------

def make_run_meta(
    *,
    config_path: str | None = None,
    temperature_k: float | None = None,
    n_steps_requested: int | None = None,
    steps_executed: int | None = None,
    total_time_s: float | None = None,
    random_seed: int | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _iso(dt: datetime | None) -> str | None:
        return None if dt is None else dt.astimezone(timezone.utc).isoformat()
    out: dict[str, Any] = {
        "config_path":       config_path,
        "started_at":        _iso(started_at),
        "finished_at":       _iso(finished_at),
        "temperature_k":     temperature_k,
        "n_steps_requested": n_steps_requested,
        "steps_executed":    steps_executed,
        "total_time_s":      total_time_s,
        "random_seed":       random_seed,
    }
    if extra:
        out.update(extra)
    return out

