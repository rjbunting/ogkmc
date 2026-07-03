"""
autokmc.io.persistence
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
from pathlib import Path
from typing import Any, TextIO

from ase import Atoms
from ase.io import write as ase_write

from autokmc.core.constants import (
    PERSISTENCE_SCHEMA_VERSION,
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,         # noqa: F401  (re-exported for convenience)
    TRAJECTORY_FILENAME,      # noqa: F401  (re-exported for convenience)
    REACTIONS_DIR,
    REACTION_DESCRIPTION_FMT,
    BOND_DESCRIPTION_FMT,
    DIFFUSION_DESCRIPTION_FMT,
    BOND_FOLDER_FMT,
    DIFFUSION_FOLDER_FMT,
)
from autokmc.io.records import ReactionRecord
from autokmc.species.smiles import smiles_to_dirname as _smiles_to_dirname
from autokmc.utils.logging import get_logger

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
    """Return the reaction folder path used in JSON records."""
    sub     = _kind_subdir(kind)
    species = _smiles_to_dirname(smiles) if smiles else "unknown"
    return f"{REACTIONS_DIR}/{sub}/{species}/{_kind_folder_name(sub, iso, lat)}"


def _reaction_description(
    reaction,
    smiles: str,
    *,
    delta_ev: float | None = None,
    barrier_ev: float | None = None,
    energy_basis: str | None = None,
) -> str:
    """Build the human-readable one-line event description for *reaction*."""
    kind = str(getattr(reaction, "kind", "adsorption"))
    delta = float(reaction.delta_e) if delta_ev is None else float(delta_ev)
    barrier = float(reaction.barrier) if barrier_ev is None else float(barrier_ev)
    if _kind_subdir(kind) == "diffusion":
        out = DIFFUSION_DESCRIPTION_FMT.format(
            smiles    = smiles,
            iso       = reaction.site.iso_class,
            member    = reaction.member_index,
            lateral   = reaction.lateral_class.lateral_class,
            direction = getattr(reaction, "direction", ""),
            delta_e   = delta,
            barrier   = barrier,
            rate      = float(reaction.rate),
        )
    elif _kind_subdir(kind) == "bond":
        tpl = getattr(reaction.site, "template", None)
        out = BOND_DESCRIPTION_FMT.format(
            smiles_a  = getattr(tpl, "smiles_a", ""),
            smiles_b  = getattr(tpl, "smiles_b", ""),
            smiles_c  = getattr(tpl, "smiles_c", ""),
            iso       = reaction.site.iso_class,
            member    = reaction.member_index,
            lateral   = reaction.lateral_class.lateral_class,
            direction = getattr(reaction, "direction", ""),
            delta_e   = delta,
            barrier   = barrier,
            rate      = float(reaction.rate),
        )
    else:
        out = REACTION_DESCRIPTION_FMT.format(
            kind     = kind,
            smiles   = smiles,
            iso      = reaction.site.iso_class,
            member   = reaction.member_index,
            lateral  = reaction.lateral_class.lateral_class,
            delta_e  = delta,
            barrier  = barrier,
            rate     = float(reaction.rate),
        )
    if energy_basis == "free_energy":
        out = out.replace("ΔE=", "ΔG=").replace("Ea=", "G‡=")
    return out


def _json_safe(o):
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(x) for x in o]
    return o


def _finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _event_free_energetics(
    reaction,
    gas_free_energies: dict[str, float] | None = None,
) -> tuple[float | None, float | None]:
    """Derive event-direction ΔG and G-barrier from cached lateral data."""
    explicit_delta = _finite_float(getattr(reaction, "delta_g", None))
    explicit_barrier = _finite_float(getattr(reaction, "barrier_g", None))
    if explicit_delta is not None or explicit_barrier is not None:
        return explicit_delta, explicit_barrier

    from autokmc.reactions.rates import EA_MIN as _EA_MIN

    kind = str(getattr(reaction, "kind", ""))
    sub = _kind_subdir(kind)
    lc = getattr(reaction, "lateral_class", None)
    if lc is None:
        return None, None

    if sub == "adsorption":
        smiles = _reaction_smiles(reaction)
        g_gas = (
            None if gas_free_energies is None
            else _finite_float(gas_free_energies.get(smiles))
        )
        g_occ = _finite_float(getattr(lc, "g_occupied", None))
        g_unocc = _finite_float(getattr(lc, "g_unoccupied", None))
        if g_occ is None or g_unocc is None or g_gas is None:
            return None, None
        if kind == "desorption":
            delta_g = g_unocc + g_gas - g_occ
        else:
            delta_g = g_occ - (g_unocc + g_gas)
        return float(delta_g), float(max(_EA_MIN, delta_g + _EA_MIN))

    if sub == "diffusion":
        g_a = _finite_float(getattr(lc, "g_a", None))
        g_b = _finite_float(getattr(lc, "g_b", None))
        g_ts = _finite_float(getattr(lc, "g_ts", None))
        if g_a is None or g_b is None or g_ts is None:
            return None, None
        g_ts_eff = max(g_ts, max(g_a, g_b) + _EA_MIN)
        if getattr(reaction, "direction", None) == "b_to_a":
            delta_g = g_a - g_b
            barrier_g = max(_EA_MIN, g_ts_eff - g_b)
        else:
            delta_g = g_b - g_a
            barrier_g = max(_EA_MIN, g_ts_eff - g_a)
        return float(delta_g), float(barrier_g)

    if sub == "bond":
        g_ab = _finite_float(getattr(lc, "g_ab", None))
        g_c = _finite_float(getattr(lc, "g_c", None))
        g_ts = _finite_float(getattr(lc, "g_ts", None))
        if g_ab is None or g_c is None or g_ts is None:
            return None, None
        g_ts_eff = max(g_ts, max(g_ab, g_c) + _EA_MIN)
        if getattr(reaction, "direction", None) == "dissoc":
            delta_g = g_ab - g_c
            barrier_g = max(_EA_MIN, g_ts_eff - g_c)
        else:
            delta_g = g_c - g_ab
            barrier_g = max(_EA_MIN, g_ts_eff - g_ab)
        return float(delta_g), float(barrier_g)

    return None, None


def _event_electronic_energetics(
    reaction,
    gas_energies: dict[str, float] | None = None,
) -> tuple[float | None, float | None]:
    """Derive event-direction electronic ΔE and barrier from cached data."""
    from autokmc.reactions.rates import EA_MIN as _EA_MIN

    kind = str(getattr(reaction, "kind", ""))
    sub = _kind_subdir(kind)
    lc = getattr(reaction, "lateral_class", None)
    if lc is None:
        return None, None

    if sub == "adsorption":
        smiles = _reaction_smiles(reaction)
        e_gas = (
            None if gas_energies is None
            else _finite_float(gas_energies.get(smiles))
        )
        e_occ = _finite_float(getattr(lc, "energy_occupied", None))
        e_unocc = _finite_float(getattr(lc, "energy_unoccupied", None))
        if e_occ is None or e_unocc is None or e_gas is None:
            return None, None
        if kind == "desorption":
            delta_e = e_unocc + e_gas - e_occ
        else:
            delta_e = e_occ - (e_unocc + e_gas)
        return float(delta_e), float(max(_EA_MIN, delta_e + _EA_MIN))

    if sub == "diffusion":
        e_a = _finite_float(getattr(lc, "energy_a", None))
        e_b = _finite_float(getattr(lc, "energy_b", None))
        e_ts = _finite_float(getattr(lc, "energy_ts", None))
        if e_a is None or e_b is None or e_ts is None:
            return None, None
        e_ts_eff = max(e_ts, max(e_a, e_b) + _EA_MIN)
        if getattr(reaction, "direction", None) == "b_to_a":
            delta_e = e_a - e_b
            barrier_e = max(_EA_MIN, e_ts_eff - e_b)
        else:
            delta_e = e_b - e_a
            barrier_e = max(_EA_MIN, e_ts_eff - e_a)
        return float(delta_e), float(barrier_e)

    if sub == "bond":
        e_ab = _finite_float(getattr(lc, "energy_ab", None))
        e_c = _finite_float(getattr(lc, "energy_c", None))
        e_ts = _finite_float(getattr(lc, "energy_ts", None))
        if e_ab is None or e_c is None or e_ts is None:
            return None, None
        e_ts_eff = max(e_ts, max(e_ab, e_c) + _EA_MIN)
        if getattr(reaction, "direction", None) == "dissoc":
            delta_e = e_ab - e_c
            barrier_e = max(_EA_MIN, e_ts_eff - e_c)
        else:
            delta_e = e_c - e_ab
            barrier_e = max(_EA_MIN, e_ts_eff - e_ab)
        return float(delta_e), float(barrier_e)

    return None, None


def _event_rate_basis(
    *,
    rate_delta_ev: float | None,
    rate_barrier_ev: float | None,
    delta_e_ev: float | None,
    barrier_ev: float | None,
    delta_g_ev: float | None,
    barrier_g_ev: float | None,
) -> str:
    def _same(a: float | None, b: float | None) -> bool:
        return a is not None and b is not None and math.isclose(
            a, b, rel_tol=1e-9, abs_tol=1e-12,
        )

    if _same(rate_delta_ev, delta_g_ev) and _same(rate_barrier_ev, barrier_g_ev):
        return "free_energy"
    if _same(rate_delta_ev, delta_e_ev) and _same(rate_barrier_ev, barrier_ev):
        return "electronic"
    if delta_g_ev is None and barrier_g_ev is None:
        if delta_e_ev is None and barrier_ev is None:
            return "electronic"
        return "unknown"
    return "unknown"


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
        append: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reactions_root = self.output_dir / reactions_dir
        self.reactions_root.mkdir(parents=True, exist_ok=True)

        self._jsonl_path: Path = self.output_dir / reactions_filename
        mode = "a" if append else "w"
        self._fp: TextIO | None = self._jsonl_path.open(mode, encoding="utf-8")
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
        folder = sub_root / species / _kind_folder_name(sub, iso, lat)

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
        gas_free_energies: dict[str, float] | None,
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
            from autokmc.reactions.rates import EA_MIN as _EA_MIN
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
                "description":      _reaction_description(reaction, smiles),
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
                        "real_ev":          list(getattr(lc, "frequencies_a_ev",  []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_a_ev",    []) or []),
                        "zpe_ev":           getattr(lc, "zpe_a",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_a", None),
                    },
                    "state_b": {
                        "real_ev":          list(getattr(lc, "frequencies_b_ev",  []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_b_ev",    []) or []),
                        "zpe_ev":           getattr(lc, "zpe_b",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_b", None),
                    },
                    "transition": {
                        "real_ev":          list(getattr(lc, "frequencies_ts_ev", []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_ts_ev",   []) or []),
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
            from autokmc.reactions.rates import EA_MIN as _EA_MIN
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
                "free_energies_ev": {
                    "g_ab": None if getattr(lc, "g_ab", None) is None else float(lc.g_ab),
                    "g_c":  None if getattr(lc, "g_c",  None) is None else float(lc.g_c),
                    "g_ts": None if getattr(lc, "g_ts", None) is None else float(lc.g_ts),
                },
                "vibrations": {
                    "state_ab": {
                        "real_ev":          list(getattr(lc, "frequencies_ab_ev", []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_ab_ev",   []) or []),
                        "zpe_ev":           getattr(lc, "zpe_ab",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_ab", None),
                    },
                    "state_c": {
                        "real_ev":          list(getattr(lc, "frequencies_c_ev", []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_c_ev",   []) or []),
                        "zpe_ev":           getattr(lc, "zpe_c",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_c", None),
                    },
                    "transition": {
                        "real_ev":          list(getattr(lc, "frequencies_ts_ev", []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_ts_ev",   []) or []),
                        "zpe_ev":           getattr(lc, "zpe_ts",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_ts", None),
                    },
                },
                "atom_matching": {
                    "method": getattr(lc, "atom_matching_method", None),
                    "atom_mapping": list(getattr(lc, "atom_mapping", []) or []),
                    "diagnostics": dict(getattr(lc, "matching_diagnostics", {}) or {}),
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
            g_gas  = (
                None
                if gas_free_energies is None or smiles not in gas_free_energies
                else float(gas_free_energies[smiles])
            )
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
                    "g_gas":        g_gas,
                },
                "vibrations": {
                    "occupied": {
                        "real_ev":          list(getattr(lc, "frequencies_occupied_ev",   []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_occupied_ev",     []) or []),
                        "zpe_ev":           getattr(lc, "zpe_occupied",     None),
                        "entropy_ev_per_k": getattr(lc, "entropy_occupied", None),
                    },
                    "unoccupied": {
                        "real_ev":          list(getattr(lc, "frequencies_unoccupied_ev", []) or []),
                        "imag_ev":          list(getattr(lc, "imaginary_unoccupied_ev",   []) or []),
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
        gas_free_energies: dict[str, float] | None = None,
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
            return self.reactions_root / sub / species / _kind_folder_name(sub, iso, lat)
        folder = self._ensure_reaction_folder(reaction)
        self._write_reaction_json(
            folder, reaction, step, gas_energies, gas_free_energies, fired=False,
        )
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
                "state_a":   None if getattr(lc, "energy_a",  None) is None else float(lc.energy_a),
                "state_b":   None if getattr(lc, "energy_b",  None) is None else float(lc.energy_b),
                "transition": None if getattr(lc, "energy_ts", None) is None else float(lc.energy_ts),
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
        gas_free_energies: dict[str, float] | None = None,
        # Backwards-compat — ignored under the new per-reaction-folder layout.
        atoms_initial: Atoms | None = None,
        atoms_final:   Atoms | None = None,
    ) -> ReactionRecord:
        """Persist one KMC event (lazy folder creation + JSONL append)."""
        if self._fp is None:
            raise RuntimeError("ReactionWriter has been closed")

        smiles = _reaction_smiles(reaction)
        folder = self._ensure_reaction_folder(reaction)
        self._write_reaction_json(
            folder, reaction, step, gas_energies, gas_free_energies, fired=True,
        )

        rate_delta_ev = _finite_float(getattr(reaction, "delta_e", None))
        rate_barrier_ev = _finite_float(getattr(reaction, "barrier", None))
        delta_e_ev, barrier_ev = _event_electronic_energetics(
            reaction,
            gas_energies=gas_energies,
        )
        delta_g_ev, barrier_g_ev = _event_free_energetics(
            reaction,
            gas_free_energies=gas_free_energies,
        )
        rate_energy_basis = _event_rate_basis(
            rate_delta_ev=rate_delta_ev,
            rate_barrier_ev=rate_barrier_ev,
            delta_e_ev=delta_e_ev,
            barrier_ev=barrier_ev,
            delta_g_ev=delta_g_ev,
            barrier_g_ev=barrier_g_ev,
        )
        if delta_e_ev is None and rate_energy_basis != "free_energy":
            delta_e_ev = rate_delta_ev
        if barrier_ev is None and rate_energy_basis != "free_energy":
            barrier_ev = rate_barrier_ev

        description = _reaction_description(
            reaction,
            smiles,
            delta_ev=rate_delta_ev,
            barrier_ev=rate_barrier_ev,
            energy_basis=rate_energy_basis,
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
            delta_e_ev      = delta_e_ev,
            barrier_ev      = barrier_ev,
            description     = description,
            reaction_dir    = str(folder.relative_to(self.output_dir)),
            direction       = getattr(reaction, "direction", None),
            delta_g_ev      = delta_g_ev,
            barrier_g_ev    = barrier_g_ev,
            rate_energy_basis = rate_energy_basis,
            rate_delta_ev     = rate_delta_ev,
            rate_barrier_ev   = rate_barrier_ev,
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
