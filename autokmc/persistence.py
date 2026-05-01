"""
autokmc.persistence
===================
Persistence of KMC events, trajectories and run summaries.

Output layout (under ``output.dir``)
-------------------------------------

::

    <output.dir>/
    ├── events.jsonl          # one JSON line per executed KMC event
    ├── summary.json          # per-reaction-type aggregates + run metadata
    ├── kmc.extxyz            # extended-XYZ trajectory (initial + every Nth state)
    └── reactions/
        ├── adsorption/
        │   ├── (O)/                      ← species dir (SMILES-derived)
        │   │   ├── iso0_lat0/
        │   │   │   ├── reaction.json
        │   │   │   ├── occupied.extxyz
        │   │   │   └── unoccupied.extxyz
        │   │   └── iso0_lat1/
        │   │       └── …
        │   └── (H)/
        │       └── …
        ├── diffusion/
        │   └── (O)/                      ← species dir
        │       ├── diff_iso0_lat0/
        │       │   ├── reaction.json
        │       │   ├── state_a.extxyz
        │       │   ├── state_b.extxyz
        │       │   └── ts.extxyz
        │       └── …
        └── bond/
            └── (O)+(H)~(OH)/             ← reaction-process dir (A+B↔C, sanitised)
                ├── bond_iso0_lat0/
                │   ├── reaction.json
                │   ├── state_ab.extxyz
                │   ├── state_c.extxyz
                │   └── ts.extxyz
                └── …

The species / reaction-process level prevents ``iso0_lat0`` collisions when
two different adsorbates happen to share the same iso/lateral index counters.
Adsorption and desorption are forward / reverse of the same lateral class so
they share a single ``adsorption/<species>/`` folder.

Each unique reaction is identified by ``(kind, species, iso_class,
lateral_class)``.  Structures are written exactly **once** per that key; every
subsequent event just appends a row to ``events.jsonl`` and updates the
``reaction.json`` counters.
"""

from __future__ import annotations

import json
import math
import re as _re
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
    BOND_DESCRIPTION_FMT,
    BOND_FOLDER_FMT,
    DIFFUSION_FOLDER_FMT,
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


def _smiles_to_dirname(label: str) -> str:
    """Convert a SMILES string or reaction-process label to a filesystem-safe dir name.

    Rules applied (in order):

    * ``↔`` → ``~``  (cross-platform ASCII alternative for the coupling arrow)
    * ``[``, ``]``   → ``(``, ``)``  (square brackets confuse shells)
    * Characters illegal on Windows: ``\\ / : * ? " < > |`` → ``_``
    * Leading / trailing spaces and dots stripped.
    * Truncated to 64 characters to avoid ``PATH_MAX`` issues.

    The result is guaranteed non-empty (falls back to ``"unknown"``).
    """
    s = str(label)
    s = s.replace("↔", "~")
    s = s.replace("[", "(").replace("]", ")")
    s = _re.sub(r'[\\/:*?"<>|]', "_", s)
    s = s.strip(". ")
    return s[:64] or "unknown"


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
    reaction_dir:    str  # relative to output.dir, e.g. "reactions/adsorption/(O)/iso0_lat3"
    delta_g_ev:      float | None = None
    barrier_g_ev:    float | None = None

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
#: share a single ``adsorption/`` folder; diffusion and bond reactions get
#: their own.
KIND_SUBDIR: dict[str, str] = {
    "adsorption": "adsorption",
    "desorption": "adsorption",
    "diffusion":  "diffusion",
    "bond":       "bond",
}

def _kind_subdir(kind: str) -> str:
    return KIND_SUBDIR.get(str(kind), str(kind))


def _reaction_folder_name(iso_class: int, lateral_class: int) -> str:
    return f"iso{int(iso_class)}_lat{int(lateral_class)}"


def _diffusion_folder_name(iso_class: int, lateral_class: int) -> str:
    return DIFFUSION_FOLDER_FMT.format(iso=int(iso_class), lat=int(lateral_class))


def _bond_folder_name(iso_class: int, lateral_class: int) -> str:
    return BOND_FOLDER_FMT.format(iso=int(iso_class), lat=int(lateral_class))


def _kind_folder_name(sub: str, iso: int, lat: int) -> str:
    if sub == "diffusion":
        return _diffusion_folder_name(iso, lat)
    if sub == "bond":
        return _bond_folder_name(iso, lat)
    return _reaction_folder_name(iso, lat)


def _reaction_smiles(reaction) -> str:
    """Return a human-readable SMILES label for *reaction*.

    Adsorption / diffusion sites carry ``site.reactant`` (a single SMILES);
    bond reactions instead carry a ``site.template`` with three SMILES that
    we render as ``"A+B↔C"``.
    """
    site = reaction.site
    smiles = getattr(site, "reactant", None)
    if smiles:
        return str(smiles)
    tpl = getattr(site, "template", None)
    if tpl is not None:
        return f"{tpl.smiles_a}+{tpl.smiles_b}↔{tpl.smiles_c}"
    return ""


def _reaction_relative_dir(kind: str, iso: int, lat: int, smiles: str = "") -> str:
    """Return ``"reactions/<sub>/<species>/<folder>"`` (POSIX-style, for JSON)."""
    sub     = _kind_subdir(kind)
    species = _smiles_to_dirname(smiles) if smiles else "unknown"
    return f"{REACTIONS_DIR}/{sub}/{species}/{_kind_folder_name(sub, iso, lat)}"


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

    Each unique ``(kind, species, iso_class, lateral_class)`` tuple gets a
    folder under ``output_dir/reactions/<sub>/<species>/`` containing:

    * ``occupied.extxyz``   — relaxed atoms behind ``E_occupied``.
    * ``unoccupied.extxyz`` — relaxed atoms behind ``E_unoccupied``.
    * ``reaction.json``     — description + energies + ΔE / barrier / rate
      / fired-event count.

    The species sub-level prevents ``iso0_lat0`` collisions when two
    different adsorbates share the same iso/lateral index counters.

    The folder is materialised lazily — it is written the **first time**
    a reaction with that key is fired, then re-used by every subsequent
    firing of either direction.  ``reaction.json`` is updated in-place
    (counts, last_step) on every event; the .extxyz files are written once.
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

        # Per (sub, species, iso, lat) bookkeeping for reaction.json files.
        self._folder_meta: dict[tuple[str, str, int, int], dict[str, Any]] = {}

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

        Folders are nested as::

            reactions/<sub>/<species>/<iso_lat_folder>/

        where *species* is the SMILES (adsorption/diffusion) or the
        reaction-process label ``A+B~C`` (bond), sanitised for filesystem
        use by :func:`_smiles_to_dirname`.  This prevents ``iso0_lat0``
        collisions when two different adsorbates share the same iso/lateral
        index counters.
        """
        iso     = int(reaction.site.iso_class)
        lat     = int(reaction.lateral_class.lateral_class)
        sub     = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        species = _smiles_to_dirname(_reaction_smiles(reaction))
        key     = (sub, species, iso, lat)

        sub_root = self.reactions_root / sub
        folder   = sub_root / species / _kind_folder_name(sub, iso, lat)

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
        elif sub == "bond":
            atoms_ab = getattr(lc, "atoms_ab", None)
            atoms_c  = getattr(lc, "atoms_c",  None)
            atoms_ts = getattr(lc, "atoms_ts", None)
            if atoms_ab is not None:
                ase_write(folder / "state_ab.extxyz", _safe_atoms_copy(atoms_ab), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_ab — state_ab.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_c is not None:
                ase_write(folder / "state_c.extxyz", _safe_atoms_copy(atoms_c), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_c — state_c.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_ts is not None:
                ase_write(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts), format="extxyz")
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_ts — ts.extxyz will not be written.",
                    iso, lat,
                )
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
        iso     = int(reaction.site.iso_class)
        lat     = int(reaction.lateral_class.lateral_class)
        sub     = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        species = _smiles_to_dirname(_reaction_smiles(reaction))
        meta    = self._folder_meta[(sub, species, iso, lat)]
        if fired:
            meta["count"] += 1
            if meta["first_step"] is None:
                meta["first_step"] = int(step)
            meta["last_step"] = int(step)

        smiles = _reaction_smiles(reaction)
        lc = reaction.lateral_class

        if sub == "diffusion":
            # Barriers use the same effective TS as the KMC engine.
            # e_ts_eff = max(e_ts, max(e_a, e_b) + EA_MIN) so that both
            # forward and reverse barriers are derived from the same TS level,
            # preserving detailed balance (Ea_fwd − Ea_rev = E_b − E_a).
            # Raw NEB energies are also persisted for auditability.
            from autokmc.kmc_adsorption import EA_MIN as _EA_MIN
            e_a  = getattr(lc, "energy_a",  None)
            e_b  = getattr(lc, "energy_b",  None)
            e_ts = getattr(lc, "energy_ts", None)
            if e_a is not None and e_b is not None and e_ts is not None:
                _e_a  = float(e_a)
                _e_b  = float(e_b)
                _e_ts = float(e_ts)
                e_ts_eff   = max(_e_ts, max(_e_a, _e_b) + _EA_MIN)
                ea_fwd_raw = _e_ts    - _e_a
                ea_rev_raw = _e_ts    - _e_b
                ea_fwd_kmc = max(_EA_MIN, e_ts_eff - _e_a)
                ea_rev_kmc = max(_EA_MIN, e_ts_eff - _e_b)
            else:
                e_ts_eff = ea_fwd_raw = ea_fwd_kmc = None
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
                    "state_a":        None if e_a      is None else float(e_a),
                    "state_b":        None if e_b      is None else float(e_b),
                    "transition_raw": None if e_ts     is None else float(e_ts),
                    "transition_eff": None if e_ts_eff is None else float(e_ts_eff),
                },
                "free_energies_ev": {
                    "g_a":  None if getattr(lc, "g_a",  None) is None else float(lc.g_a),
                    "g_b":  None if getattr(lc, "g_b",  None) is None else float(lc.g_b),
                    "g_ts": None if getattr(lc, "g_ts", None) is None else float(lc.g_ts),
                },
                "vibrations": {
                    "state_a": {
                        "real_cm":          list(getattr(lc, "frequencies_a_cm",  []) or []),
                        "imag_cm":          list(getattr(lc, "imaginary_a_cm",    []) or []),
                        "zpe_ev":           getattr(lc, "zpe_a",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_a", None),
                    },
                    "state_b": {
                        "real_cm":          list(getattr(lc, "frequencies_b_cm",  []) or []),
                        "imag_cm":          list(getattr(lc, "imaginary_b_cm",    []) or []),
                        "zpe_ev":           getattr(lc, "zpe_b",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_b", None),
                    },
                    "transition": {
                        "real_cm":          list(getattr(lc, "frequencies_ts_cm", []) or []),
                        "imag_cm":          list(getattr(lc, "imaginary_ts_cm",   []) or []),
                        "zpe_ev":           getattr(lc, "zpe_ts",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_ts", None),
                    },
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
        elif sub == "bond":
            from autokmc.kmc_adsorption import EA_MIN as _EA_MIN
            tpl = getattr(reaction.site, "template", None)
            e_ab = getattr(lc, "energy_ab", None)
            e_c  = getattr(lc, "energy_c",  None)
            e_ts = getattr(lc, "energy_ts", None)
            if e_ab is not None and e_c is not None and e_ts is not None:
                _e_ab = float(e_ab)
                _e_c  = float(e_c)
                _e_ts = float(e_ts)
                e_ts_eff      = max(_e_ts, max(_e_ab, _e_c) + _EA_MIN)
                ea_couple_raw = _e_ts    - _e_ab
                ea_dissoc_raw = _e_ts    - _e_c
                ea_couple_kmc = max(_EA_MIN, e_ts_eff - _e_ab)
                ea_dissoc_kmc = max(_EA_MIN, e_ts_eff - _e_c)
            else:
                e_ts_eff = ea_couple_raw = ea_couple_kmc = None
                ea_dissoc_raw = ea_dissoc_kmc = None

            payload = {
                "schema_version":  PERSISTENCE_SCHEMA_VERSION,
                "kind":            "bond",
                "iso_class":       iso,
                "lateral_class":   lat,
                "reactant_smiles": smiles,
                "template": (
                    {
                        "smiles_a": tpl.smiles_a,
                        "smiles_b": tpl.smiles_b,
                        "smiles_c": tpl.smiles_c,
                        "bond_type": getattr(tpl, "bond_type", None),
                        "source":    getattr(tpl, "source", None),
                    }
                    if tpl is not None else None
                ),
                "kind_directions": ["couple", "dissoc"],
                "description": BOND_DESCRIPTION_FMT.format(
                    smiles_a  = getattr(tpl, "smiles_a", ""),
                    smiles_b  = getattr(tpl, "smiles_b", ""),
                    smiles_c  = getattr(tpl, "smiles_c", ""),
                    iso       = iso,
                    member    = reaction.member_index,
                    lateral   = lat,
                    direction = getattr(reaction, "direction", ""),
                    delta_e   = float(reaction.delta_e),
                    barrier   = float(reaction.barrier),
                    rate      = float(reaction.rate),
                ),
                "energies_ev": {
                    "state_ab":       None if e_ab     is None else float(e_ab),
                    "state_c":        None if e_c      is None else float(e_c),
                    "transition_raw": None if e_ts     is None else float(e_ts),
                    "transition_eff": None if e_ts_eff is None else float(e_ts_eff),
                },
                "barriers_ev": {
                    "couple_raw": ea_couple_raw,
                    "couple_kmc": ea_couple_kmc,
                    "dissoc_raw": ea_dissoc_raw,
                    "dissoc_kmc": ea_dissoc_kmc,
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
                    "state_ab":   "state_ab.extxyz",
                    "state_c":    "state_c.extxyz",
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
            g_occ   = getattr(lc, "g_occupied",        None)
            g_unocc = getattr(lc, "g_unoccupied",      None)

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
                "free_energies_ev": {
                    "g_occupied":   None if g_occ   is None else float(g_occ),
                    "g_unoccupied": None if g_unocc is None else float(g_unocc),
                    "g_gas":        None,  # populated by writer caller via gas_g
                },
                "vibrations": {
                    "occupied": {
                        "real_cm":          list(getattr(lc, "frequencies_occupied_cm",   []) or []),
                        "imag_cm":          list(getattr(lc, "imaginary_occupied_cm",     []) or []),
                        "zpe_ev":           getattr(lc, "zpe_occupied",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_occupied", None),
                    },
                    "unoccupied": {
                        "real_cm":          list(getattr(lc, "frequencies_unoccupied_cm", []) or []),
                        "imag_cm":          list(getattr(lc, "imaginary_unoccupied_cm",   []) or []),
                        "zpe_ev":           getattr(lc, "zpe_unoccupied",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_unoccupied", None),
                    },
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
        with the same ``(kind, species, iso_class, lateral_class)`` key are
        no-ops.

        Used by the KMC driver to persist **every** reaction in the
        current applicable list — not just the one chosen for execution —
        so the run directory mirrors the full discovered reaction network.
        """
        iso     = int(reaction.site.iso_class)
        lat     = int(reaction.lateral_class.lateral_class)
        sub     = _kind_subdir(getattr(reaction, "kind", "adsorption"))
        species = _smiles_to_dirname(_reaction_smiles(reaction))
        key     = (sub, species, iso, lat)
        if key in self._folder_meta:
            return (self.reactions_root / sub / species
                    / _kind_folder_name(sub, iso, lat))
        folder = self._ensure_reaction_folder(reaction)
        self._write_reaction_json(folder, reaction, step, gas_energies, fired=False)
        return folder

    # ------------------------------------------------------------------
    def write_invalid_diffusion(self, ds, lc) -> Path:
        """Write an on-disk record for a diffusion lateral class that failed NEB.

        Creates ``reactions/diffusion/<species>/diff_iso{X}_lat{Y}/`` and
        writes a ``reaction.json`` with ``"valid": false`` and the failure
        reason.  Any partial atoms already stored on *lc* are written as
        ``state_a.extxyz`` / ``state_b.extxyz`` for post-mortem inspection.

        Idempotent — a second call for the same ``(iso, lat, species)`` is
        a no-op.
        """
        iso     = int(ds.iso_class)
        lat     = int(lc.lateral_class)
        smiles  = getattr(ds, "reactant", "")
        species = _smiles_to_dirname(smiles) if smiles else "unknown"
        key     = ("diffusion_invalid", species, iso, lat)
        sub_root = self.reactions_root / "diffusion"
        folder   = sub_root / species / _diffusion_folder_name(iso, lat)

        if key in self._folder_meta:
            return folder

        folder.mkdir(parents=True, exist_ok=True)

        atoms_a  = getattr(lc, "atoms_a",  None)
        atoms_b  = getattr(lc, "atoms_b",  None)
        atoms_ts = getattr(lc, "atoms_ts", None)
        if atoms_a is not None:
            ase_write(folder / "state_a.extxyz", _safe_atoms_copy(atoms_a), format="extxyz")
        if atoms_b is not None:
            ase_write(folder / "state_b.extxyz", _safe_atoms_copy(atoms_b), format="extxyz")
        if atoms_ts is not None:
            ase_write(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts), format="extxyz")

        payload = {
            "schema_version":  PERSISTENCE_SCHEMA_VERSION,
            "kind":            "diffusion",
            "iso_class":       iso,
            "lateral_class":   lat,
            "reactant_smiles": smiles,
            "valid":           False,
            "invalid_reason":  getattr(lc, "invalid_reason", None),
            "energies_ev": {
                "state_a":   None if lc.energy_a  is None else float(lc.energy_a),
                "state_b":   None if lc.energy_b  is None else float(lc.energy_b),
                "transition": None if lc.energy_ts is None else float(lc.energy_ts),
            },
            "calculator": dict(self._calc_meta),
        }
        with (folder / "reaction.json").open("w", encoding="utf-8") as fp:
            json.dump(_json_safe(payload), fp, indent=2)

        self._folder_meta[key] = {"count": 0, "first_step": None, "last_step": None}
        _log.info(
            "ReactionWriter: wrote invalid diffusion folder "
            "species=%s iso=%d lat=%d  reason=%s",
            species, iso, lat, getattr(lc, "invalid_reason", None),
        )
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

        smiles = _reaction_smiles(reaction)
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

    Parameters
    ----------
    reactant_smiles : set[str] | None
        The SMILES of user-supplied feed-gas reactants (those listed under
        ``reactants:`` in the config).  Any species that desorbs whose SMILES
        is **not** in this set is treated as a *product* of the simulation
        (created on the surface by bond-coupling events) and is counted
        separately in the ``production_summary`` section of the output JSON.
        Pass ``None`` or an empty set to include *all* desorbing species in
        ``production_summary`` (useful when no bond channel is active).
    """

    __slots__ = (
        "_buckets", "_total_by_kind", "_n", "_first_step", "_last_step",
        "_reactant_smiles",
    )

    def __init__(self, reactant_smiles: set[str] | None = None):
        self._buckets: dict[tuple, dict[str, list[float]]] = {}
        self._total_by_kind: dict[str, int] = {}
        self._n: int = 0
        self._first_step: dict[tuple, int] = {}
        self._last_step:  dict[tuple, int] = {}
        # frozenset for O(1) membership test; empty means "show all desorbates"
        self._reactant_smiles: frozenset[str] = frozenset(reactant_smiles or [])

    def add(self, reaction, *, step: int) -> None:
        smiles = _reaction_smiles(reaction)
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

    # ------------------------------------------------------------------
    def _production_summary(
        self,
        run_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the ``production_summary`` block for the output JSON.

        A *product* species is any species that desorbs whose SMILES is
        **not** in ``self._reactant_smiles``.  When ``_reactant_smiles`` is
        empty every desorbing species is included (no feed-gas / product
        distinction has been configured).

        The production rate in Hz is computed as::

            production_rate_hz[species] = desorption_count[species]
                                          / total_kmc_time_s

        where ``total_kmc_time_s`` comes from ``run_meta["total_time_s"]``
        (the total *simulated* time accumulated by the BKL sampler, **not**
        wall-clock time).  If the KMC time is unavailable or zero, the rate
        fields are ``null`` and only raw counts are reported.
        """
        kmc_time: float | None = None
        if run_meta:
            t = run_meta.get("total_time_s")
            if t is not None and float(t) > 0:
                kmc_time = float(t)

        steps_executed: int | None = None
        if run_meta:
            se = run_meta.get("steps_executed")
            if se is not None:
                steps_executed = int(se)

        # ── Collect desorption events bucketed by product species ──────────
        # key: smiles  →  list of per-(iso,lat) breakdown dicts
        product_map: dict[str, list[dict[str, Any]]] = {}

        for key, b in self._buckets.items():
            kind, smiles, iso, lat = key
            if kind != "desorption":
                continue
            # Skip user-supplied feed-gas reactants when the set is non-empty.
            if self._reactant_smiles and smiles in self._reactant_smiles:
                continue
            product_map.setdefault(smiles, []).append({
                "iso_class":     iso,
                "lateral_class": lat,
                "count":         len(b["rate"]),
                "first_step":    self._first_step[key],
                "last_step":     self._last_step[key],
            })

        # ── Build per-species entries ──────────────────────────────────────
        by_species: dict[str, Any] = {}
        total_count = 0
        for smiles in sorted(product_map):
            breakdowns = sorted(
                product_map[smiles],
                key=lambda d: (d["iso_class"], d["lateral_class"]),
            )
            count = sum(d["count"] for d in breakdowns)
            total_count += count
            by_species[smiles] = {
                "desorption_count":   count,
                "production_rate_hz": (
                    count / kmc_time if kmc_time is not None else None
                ),
                "events_per_step": (
                    count / steps_executed
                    if steps_executed and steps_executed > 0 else None
                ),
                "iso_breakdown": breakdowns,
            }

        note = (
            "Desorption events of species not in the user-supplied reactants list "
            "(partial_pressure_bar=0 species created on-the-fly by bond coupling)."
            if self._reactant_smiles
            else "Desorption events of all species (no reactant set configured)."
        )

        return {
            "note":                      note,
            "kmc_time_s":                kmc_time,
            "steps_executed":            steps_executed,
            "reactant_smiles":           sorted(self._reactant_smiles),
            "product_species":           sorted(by_species),
            "by_species":                by_species,
            "total_product_desorptions": total_count,
            "total_production_rate_hz":  (
                total_count / kmc_time if kmc_time is not None else None
            ),
        }

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
                "reaction_dir":     _reaction_relative_dir(kind, iso, lat, smiles=smiles),
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
            "by_reaction_type":  by_type,
            "production_summary": self._production_summary(run_meta),
            "final_occupancy":   {str(k): int(v)
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

