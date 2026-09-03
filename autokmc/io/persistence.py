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
subsequent event appends a row to ``events.jsonl``.  ``reaction.json`` counters
are batched at checkpoint boundaries and writer close.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, TextIO, cast

from ase import Atoms
from ase.io import write as ase_write

from autokmc.core.constants import (
    PERSISTENCE_SCHEMA_VERSION,  # noqa: F401 - legacy compatibility re-export
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,         # noqa: F401  (re-exported for convenience)
    TRAJECTORY_FILENAME,      # noqa: F401  (re-exported for convenience)
    REACTIONS_DIR,
    REACTION_DESCRIPTION_FMT,
    BOND_DESCRIPTION_FMT,
    DIFFUSION_DESCRIPTION_FMT,
)
from autokmc.io._files import (
    atomic_output_path,
    ensure_directory,
    fsync_directory,
    replace_path_atomic,
    write_json_atomic,
)
from autokmc.io.event_log import (  # noqa: F401
    EventLogCommit,
    EventLogRecovery,
    reconcile_event_log,
)
from autokmc.io.event_transitions import reaction_transition
from autokmc.io.atoms import copy_atoms_with_results
from autokmc.io.reaction_layout import (
    diffusion_folder_name as _diffusion_folder_name,
    kind_folder_name as _kind_folder_name,
    kind_subdir as _kind_subdir,
    reaction_smiles as _reaction_smiles,
)
from autokmc.io.reaction_index import (
    REACTION_INDEX_FILENAME,
    ReactionIndexWriter,
    reaction_definition_from_document,
    stable_event_id,
    stable_reaction_id,
)
from autokmc.io.reaction_payloads import build_reaction_payload
from autokmc.io.records import ReactionRecord
from autokmc.io.schemas import (
    EVENT_SCHEMA_VERSION,
    REACTION_DOCUMENT_ARTIFACT_TYPE,
    REACTION_DOCUMENT_SCHEMA_VERSION,
)
from autokmc.species.smiles import smiles_to_dirname as _smiles_to_dirname
from autokmc.utils.logging import get_logger
from autokmc.utils.telemetry import instrument

_log = get_logger(__name__)

UNCOMMITTED_REACTIONS_DIR = "uncommitted_reactions"
DIAGNOSTICS_DIR = "diagnostics"
INVALID_ADSORPTION_DIR = "invalid_adsorption"
INVALID_DIFFUSION_DIR = "invalid_diffusion"
INVALID_BOND_DIR = "invalid_bond"


def _atomic_json(path: Path, payload: Any) -> None:
    """Write one complete JSON document without exposing partial contents."""
    write_json_atomic(path, payload, transform=_json_safe)


def _atomic_extxyz(path: Path, images: Atoms | list[Atoms]) -> None:
    """Write structures atomically and durably before publishing their names."""
    with atomic_output_path(path) as temporary:
        ase_write(temporary, images, format="extxyz")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_atoms_copy(atoms: Atoms) -> Atoms:
    """Return an extxyz-ready copy without retaining a live calculator."""
    return copy_atoms_with_results(atoms)


def write_invalid_adsorption_diagnostic(
    diagnostics_dir: str | Path,
    site,
    *,
    reactant_smiles: str,
    atoms_initial: Atoms,
    atoms_optimized: Atoms | None,
    invalid_reason: str,
    details: dict[str, Any] | None = None,
) -> Path:
    """Persist one adsorption iso-class rejected by MLIP pruning.

    These structures are diagnostic candidates rather than KMC reactions, so
    they live outside the authoritative reaction tree under
    ``diagnostics/invalid_adsorption/<species>/ads_isoX/``.
    """
    species = (
        _smiles_to_dirname(reactant_smiles)
        if reactant_smiles
        else "unknown"
    )
    iso_class = int(site.iso_class)
    folder = (
        Path(diagnostics_dir)
        / INVALID_ADSORPTION_DIR
        / species
        / f"ads_iso{iso_class}"
    )
    ensure_directory(folder)
    _atomic_extxyz(
        folder / "initial.extxyz",
        _safe_atoms_copy(atoms_initial),
    )
    if atoms_optimized is not None:
        _atomic_extxyz(
            folder / "optimized.extxyz",
            _safe_atoms_copy(atoms_optimized),
        )
    _atomic_json(
        folder / "diagnostic.json",
        {
            "artifact_type": "autokmc-invalid-adsorption-diagnostic",
            "schema_version": "1",
            "kind": "adsorption",
            "iso_class": iso_class,
            "reactant_smiles": str(reactant_smiles),
            "invalid_reason": str(invalid_reason),
            "structures": {
                "initial": "initial.extxyz",
                "optimized": (
                    "optimized.extxyz"
                    if atoms_optimized is not None
                    else None
                ),
            },
            "details": dict(details or {}),
        },
    )
    return folder


def _read_discovery_step(path: Path) -> int | None:
    """Return a persisted discovery step, preserving legacy metadata as unknown."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reaction metadata {path}: {exc}") from exc

    value = payload.get("discovery_step")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"reaction metadata {path} has invalid discovery_step={value!r}"
        )
    return value


def _available_quarantine_path(destination: Path) -> Path:
    """Choose a recovery path without replacing an earlier quarantined folder."""
    if not destination.exists():
        return destination
    suffix = 1
    while True:
        candidate = destination.with_name(f"{destination.name}.{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _quarantine_uncommitted_reaction_folders(
    output_dir: Path,
    reactions_root: Path,
    *,
    checkpoint_step: int,
) -> list[Path]:
    """Atomically move folders discovered after a resumed checkpoint.

    The checkpoint and event log describe one committed prefix.  A process can
    crash after materialising reactions found while advancing the next state
    but before committing that state's checkpoint.  Such folders must not stay
    under the authoritative ``reactions/`` tree when the older checkpoint is
    resumed.
    """
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")

    quarantined: list[Path] = []
    recovery_root = (
        output_dir
        / UNCOMMITTED_REACTIONS_DIR
        / f"after_checkpoint_step_{checkpoint_step}"
    )
    invalid_diffusion_root = (
        output_dir / DIAGNOSTICS_DIR / INVALID_DIFFUSION_DIR
    )
    invalid_bond_root = output_dir / DIAGNOSTICS_DIR / INVALID_BOND_DIR
    roots = (
        (reactions_root, Path()),
        (
            invalid_diffusion_root,
            Path(DIAGNOSTICS_DIR) / INVALID_DIFFUSION_DIR,
        ),
        (
            invalid_bond_root,
            Path(DIAGNOSTICS_DIR) / INVALID_BOND_DIR,
        ),
    )
    leaf_folders: list[tuple[Path, Path, Path]] = []
    for root, recovery_prefix in roots:
        pattern = "*/*/*" if root == reactions_root else "*/*"
        leaf_folders.extend(
            (path, root, recovery_prefix)
            for path in root.glob(pattern)
            if path.is_dir()
        )
    for folder, authoritative_root, recovery_prefix in sorted(leaf_folders):
        metadata_path = folder / "reaction.json"
        incomplete = not metadata_path.is_file()
        discovery_step = (
            None if incomplete else _read_discovery_step(metadata_path)
        )
        # Legacy reaction documents have no reliable discovery boundary.
        # Retaining them is the only backward-compatible choice.  A leaf
        # without reaction.json is different: folder creation and structure
        # writes precede metadata publication, so it is an incomplete crash
        # artifact and cannot be authoritative for any checkpoint.
        if (
            not incomplete
            and (discovery_step is None or discovery_step <= checkpoint_step)
        ):
            continue

        relative = folder.relative_to(authoritative_root)
        destination = _available_quarantine_path(
            recovery_root / recovery_prefix / relative
        )
        ensure_directory(destination.parent)
        replace_path_atomic(folder, destination)
        quarantined.append(destination)
        if incomplete:
            _log.warning(
                "Moved incomplete reaction folder with no reaction.json "
                "outside the authoritative hierarchy to %s",
                destination,
            )
        else:
            _log.warning(
                "Moved uncommitted reaction folder discovered at step %d "
                "past checkpoint step %d to %s",
                discovery_step,
                checkpoint_step,
                destination,
            )

        # Leave the active hierarchy tidy without deleting any recovery data.
        parent = folder.parent
        while parent != authoritative_root:
            try:
                parent.rmdir()
            except OSError:
                break
            fsync_directory(parent.parent)
            parent = parent.parent
    return quarantined

# ---------------------------------------------------------------------------
# ReactionWriter
# ---------------------------------------------------------------------------

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


def _prepare_bond_gas_reference_assets(site, lateral_class) -> None:
    """Attach reproducible gas-reference snapshots when they can be recovered.

    Gas-product endpoint atoms always use ``[environment | gas molecule]``
    ordering.  That lets persistence recover the relaxed empty environment
    from ``atoms_c`` even when the run was loaded from an older calculation
    cache that predates the explicit gas-reference fields.
    """
    if not bool(getattr(site, "gas_product", False)):
        return

    gas_reactant = getattr(site, "gas_reactant", None)
    gas_atoms = getattr(gas_reactant, "atoms", None)
    if (
        getattr(lateral_class, "atoms_gas_molecule", None) is None
        and isinstance(gas_atoms, Atoms)
    ):
        lateral_class.atoms_gas_molecule = copy_atoms_with_results(gas_atoms)

    if getattr(lateral_class, "atoms_c_gas_reference", None) is None:
        atoms_c = getattr(lateral_class, "atoms_c", None)
        atoms_ab = getattr(lateral_class, "atoms_ab", None)
        molecule = getattr(lateral_class, "atoms_gas_molecule", None)
        if isinstance(atoms_c, Atoms) and isinstance(molecule, Atoms):
            n_gas = len(molecule)
            n_environment = (
                len(atoms_ab) - n_gas
                if isinstance(atoms_ab, Atoms) and len(atoms_ab) >= n_gas
                else len(atoms_c) - n_gas
            )
            if n_environment > 0 and len(atoms_c) >= n_environment:
                snapshot = atoms_c[:n_environment].copy()
                snapshot.calc = None
                lateral_class.atoms_c_gas_reference = snapshot

    if getattr(lateral_class, "energy_c_gas_reference", None) is None:
        energy_c = _finite_float(getattr(lateral_class, "energy_c", None))
        energy_gas = _finite_float(getattr(gas_reactant, "energy", None))
        if energy_c is not None and energy_gas is not None:
            lateral_class.energy_c_gas_reference = energy_c - energy_gas


def _write_missing_bond_gas_reference_assets(
    folder: Path,
    site,
    lateral_class,
) -> bool:
    """Backfill missing independent gas-reference structure files."""
    _prepare_bond_gas_reference_assets(site, lateral_class)
    wrote = False
    for filename, atoms in (
        (
            "state_c_gas_reference.extxyz",
            getattr(lateral_class, "atoms_c_gas_reference", None),
        ),
        (
            "gas_molecule.extxyz",
            getattr(lateral_class, "atoms_gas_molecule", None),
        ),
    ):
        path = folder / filename
        if isinstance(atoms, Atoms) and not path.is_file():
            _atomic_extxyz(path, _safe_atoms_copy(atoms))
            wrote = True
    return wrote


def _write_missing_bond_result_assets(
    folder: Path,
    site,
    lateral_class,
) -> bool:
    """Backfill every currently available bond-result structure."""
    wrote = _write_missing_bond_gas_reference_assets(
        folder,
        site,
        lateral_class,
    )
    for filename, attribute in (
        ("state_ab_initial.extxyz", "atoms_ab_initial"),
        ("state_c_initial.extxyz", "atoms_c_initial"),
        ("state_ab.extxyz", "atoms_ab"),
        ("state_c.extxyz", "atoms_c"),
        ("ts.extxyz", "atoms_ts"),
    ):
        atoms = getattr(lateral_class, attribute, None)
        path = folder / filename
        if isinstance(atoms, Atoms) and not path.is_file():
            _atomic_extxyz(path, _safe_atoms_copy(atoms))
            wrote = True
    for filename, attribute in (
        ("neb_path_initial.extxyz", "atoms_neb_path_initial"),
        ("neb_path.extxyz", "atoms_neb_path"),
    ):
        images = getattr(lateral_class, attribute, None)
        path = folder / filename
        if images and not path.is_file():
            _atomic_extxyz(
                path,
                [_safe_atoms_copy(image) for image in images],
            )
            wrote = True
    if _write_neb_refinement_assets(folder, lateral_class):
        wrote = True
    return wrote


def _write_neb_refinement_assets(folder: Path, lateral_class) -> bool:
    """Write the optimized endpoints selected for a stalled-band restart."""
    wrote = False
    for filename, attribute in (
        ("neb_refinement_initial.extxyz", "atoms_neb_refinement_initial"),
        ("neb_refinement_final.extxyz", "atoms_neb_refinement_final"),
    ):
        atoms = getattr(lateral_class, attribute, None)
        path = folder / filename
        if isinstance(atoms, Atoms) and not path.is_file():
            _atomic_extxyz(path, _safe_atoms_copy(atoms))
            wrote = True
    return wrote


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

    * ``*_initial.extxyz``  — endpoint structures before relaxation.
    * endpoint ``.extxyz`` files — relaxed structures behind the energies.
    * ``neb_path_initial.extxyz`` — optional interpolated diffusion/bond band.
    * ``neb_path.extxyz`` — optional optimized diffusion/bond band.
    * ``reaction.json``     — description + energies + ΔE / barrier / rate
      / fired-event count.

    The species sub-level prevents ``iso0_lat0`` collisions when two
    different adsorbates share the same iso/lateral index counters.

    The folder is materialised lazily — it is written the **first time**
    a reaction with that key is fired, then re-used by every subsequent
    firing of either direction.  Event rows are appended immediately, while
    ``reaction.json`` statistics are batched and atomically flushed at a
    checkpoint boundary or when the writer closes.  The .extxyz files are
    written once. Optimized structures retain cached single-point ``energy``
    and ``forces`` fields; initial or otherwise unevaluated structures remain
    geometry-only.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        reactions_filename: str = REACTIONS_FILENAME,
        reactions_dir: str = REACTIONS_DIR,
        calculator_meta: dict[str, Any] | None = None,
        append: bool = False,
        run_id: str | None = None,
        checkpoint_step: int | None = None,
        event_recovery: EventLogRecovery | None = None,
    ):
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        self.reactions_root = self.output_dir / reactions_dir
        ensure_directory(self.reactions_root)
        self.invalid_diffusion_root = (
            self.output_dir / DIAGNOSTICS_DIR / INVALID_DIFFUSION_DIR
        )
        self.invalid_adsorption_root = (
            self.output_dir / DIAGNOSTICS_DIR / INVALID_ADSORPTION_DIR
        )
        self.invalid_bond_root = (
            self.output_dir / DIAGNOSTICS_DIR / INVALID_BOND_DIR
        )

        self._jsonl_path: Path = self.output_dir / reactions_filename
        self._calc_meta: dict[str, Any] = dict(calculator_meta or {})
        self.run_id = None if run_id is None else str(run_id)
        self._n_written: int = 0

        # Per (sub, species, iso, lat) bookkeeping for reaction.json files.
        self._folder_meta: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._folder_paths: dict[tuple[str, str, int, int], Path] = {}
        self._discovery_steps: dict[tuple[str, str, int, int], int] = {}
        self._pending_payloads: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._reaction_definitions: dict[str, dict[str, Any]] = {}
        self._reconciled_bond_keys: set[tuple[str, str, int, int]] = set()
        self._index: ReactionIndexWriter | None = None
        if append and checkpoint_step is not None:
            _quarantine_uncommitted_reaction_folders(
                self.output_dir,
                self.reactions_root,
                checkpoint_step=int(checkpoint_step),
            )

        file_existed = self._jsonl_path.is_file()
        mode = "a" if append else "w"
        self._fp: TextIO | None = cast(
            TextIO,
            self._jsonl_path.open(mode, encoding="utf-8"),
        )
        # New/truncated event files and newly appended rows must be synced
        # before their byte prefix can be published in a checkpoint.
        self._events_dirty = not append or not file_existed
        self._event_directory_dirty = not file_existed
        if append:
            self._restore_existing_state(event_recovery=event_recovery)
        self._index = ReactionIndexWriter(
            self.reactions_root / REACTION_INDEX_FILENAME,
            run_id=self.run_id,
            initial_entries=self._reaction_definitions.values(),
        )

    def _restore_existing_state(
        self,
        *,
        event_recovery: EventLogRecovery | None = None,
    ) -> None:
        """Restore counters for an append-mode checkpoint continuation."""
        restored_payloads: dict[
            tuple[str, str, int, int],
            dict[str, Any],
        ] = {}
        metadata_paths = [
            *((path, None) for path in self.reactions_root.glob("*/*/*/reaction.json")),
            *(
                (path, "diffusion_invalid")
                for path in self.invalid_diffusion_root.glob("*/*/reaction.json")
            ),
            *(
                (path, "bond_invalid")
                for path in self.invalid_bond_root.glob("*/*/reaction.json")
            ),
        ]
        for path, diagnostic_invalid in metadata_paths:
            try:
                if diagnostic_invalid:
                    diagnostic_root = (
                        self.invalid_diffusion_root
                        if diagnostic_invalid == "diffusion_invalid"
                        else self.invalid_bond_root
                    )
                    relative = path.relative_to(diagnostic_root)
                    species = relative.parts[0]
                    sub = diagnostic_invalid
                else:
                    relative = path.relative_to(self.reactions_root)
                    sub, species = relative.parts[:2]
                payload = json.loads(path.read_text(encoding="utf-8"))
                restored_sub = (
                    str(diagnostic_invalid)
                    if diagnostic_invalid
                    else (
                        f"{sub}_invalid"
                        if sub in {"diffusion", "bond"}
                        and payload.get("valid") is False
                        else str(sub)
                    )
                )
                key = (
                    restored_sub,
                    str(species),
                    int(payload["iso_class"]),
                    int(payload["lateral_class"]),
                )
                self._folder_meta[key] = {
                    "count": 0,
                    "first_step": None,
                    "last_step": None,
                }
                self._folder_paths[key] = path.parent
                restored_payloads[key] = payload
                discovery_step = payload.get("discovery_step")
                self._discovery_steps[key] = (
                    0 if discovery_step is None else int(discovery_step)
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                _log.warning("Ignoring unreadable reaction metadata during resume: %s", path)

        last_event_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        rate_bases_by_key: dict[tuple[str, str, int, int], set[str]] = {}
        if event_recovery is not None:
            self._n_written = int(event_recovery.count)
            for key, recovered in event_recovery.reaction_states.items():
                meta = self._folder_meta.get(key)
                if meta is None:
                    event = recovered.last_event or {}
                    _log.warning(
                        "Recovered event refers to a missing reaction folder: %s",
                        event.get("reaction_dir"),
                    )
                    continue
                meta.update({
                    "count": int(recovered.count),
                    "first_step": recovered.first_step,
                    "last_step": recovered.last_step,
                })
                rate_bases_by_key[key] = set(recovered.rate_energy_bases)
                if recovered.last_event is not None:
                    last_event_by_key[key] = recovered.last_event
        else:
            try:
                with self._jsonl_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            key = (
                                _kind_subdir(str(event["kind"])),
                                _smiles_to_dirname(str(event["reactant_smiles"])),
                                int(event["iso_class"]),
                                int(event["lateral_class"]),
                            )
                        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                            _log.warning(
                                "Ignoring invalid event line %d while restoring reaction stats",
                                line_number,
                            )
                            continue
                        self._n_written += 1
                        meta = self._folder_meta.get(key)
                        if meta is None:
                            _log.warning(
                                "Event line %d refers to a missing reaction folder: %s",
                                line_number,
                                event.get("reaction_dir"),
                            )
                            continue
                        step = int(event["step"])
                        meta["count"] += 1
                        if meta["first_step"] is None:
                            meta["first_step"] = step
                        meta["last_step"] = step
                        basis = event.get("rate_energy_basis")
                        if basis:
                            rate_bases_by_key.setdefault(key, set()).add(str(basis))
                        last_event_by_key[key] = event
            except OSError:
                self._n_written = 0

        # A crash can leave some reaction.json files ahead of the checkpoint
        # even after events.jsonl is truncated.  Rebuild their statistics and
        # last-event fields from the reconciled source of truth.
        for key in self._folder_paths:
            payload = restored_payloads.get(key)
            if payload is None:
                continue
            recovered_stats = dict(self._folder_meta[key])
            recovered_event = last_event_by_key.get(key)
            recovered_last_event = None
            recovered_description = payload.get("description")
            recovered_rate_bases = sorted(rate_bases_by_key.get(key, set()))
            if recovered_event is None:
                recovered_last_event = None
            else:
                if recovered_event.get("description") is not None:
                    recovered_description = recovered_event["description"]
                recovered_last_event = {
                    "kind": str(recovered_event["kind"]),
                    "delta_e_ev": recovered_event.get(
                        "rate_delta_ev",
                        recovered_event.get("delta_e_ev"),
                    ),
                    "barrier_ev": recovered_event.get(
                        "rate_barrier_ev",
                        recovered_event.get("barrier_ev"),
                    ),
                    "rate_hz": recovered_event.get("rate_hz"),
                    "step": int(recovered_event["step"]),
                }
                if key[0] in {"diffusion", "bond"}:
                    recovered_last_event["direction"] = recovered_event.get("direction")
            changed = (
                payload.get("stats") != recovered_stats
                or payload.get("last_event") != recovered_last_event
                or payload.get("description") != recovered_description
                or payload.get("rate_energy_bases", []) != recovered_rate_bases
            )
            payload["stats"] = recovered_stats
            payload["last_event"] = recovered_last_event
            payload["description"] = recovered_description
            payload["rate_energy_bases"] = recovered_rate_bases
            if changed:
                self._pending_payloads[key] = payload
            definition = reaction_definition_from_document(
                payload,
                folder=self._folder_paths[key].relative_to(self.output_dir),
                run_id=self.run_id,
            )
            self._reaction_definitions[str(definition["reaction_id"])] = definition

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

    @property
    def n_valid_reactions(self) -> int:
        if self._index is not None:
            return self._index.n_valid
        return sum(key[0] != "diffusion_invalid" for key in self._folder_meta)

    @property
    def n_invalid_reactions(self) -> int:
        if self._index is not None:
            return self._index.n_invalid
        return sum(key[0] == "diffusion_invalid" for key in self._folder_meta)

    @property
    def reaction_index_path(self) -> Path:
        return self.reactions_root / REACTION_INDEX_FILENAME

    @property
    def reaction_definitions(self) -> tuple[dict[str, Any], ...]:
        """Return snapshots of all persisted definitions for summary seeding."""
        definitions = (
            self._index.entries.values()
            if self._index is not None
            else self._reaction_definitions.values()
        )
        return tuple(dict(definition) for definition in definitions)

    # ------------------------------------------------------------------
    def _ensure_reaction_folder(self, reaction, *, discovery_step: int) -> Path:
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
        lc = reaction.lateral_class

        if sub == "bond":
            _prepare_bond_gas_reference_assets(reaction.site, lc)

        if key in self._folder_meta:
            self._folder_paths.setdefault(key, folder)
            if sub == "bond":
                _write_missing_bond_result_assets(
                    folder,
                    reaction.site,
                    lc,
                )
            elif sub == "diffusion":
                _write_neb_refinement_assets(folder, lc)
            return folder

        ensure_directory(folder)

        if sub == "diffusion":
            atoms_a_initial = getattr(lc, "atoms_a_initial", None)
            atoms_b_initial = getattr(lc, "atoms_b_initial", None)
            atoms_a  = getattr(lc, "atoms_a",  None)
            atoms_b  = getattr(lc, "atoms_b",  None)
            atoms_ts = getattr(lc, "atoms_ts", None)
            if atoms_a_initial is not None:
                _atomic_extxyz(
                    folder / "state_a_initial.extxyz",
                    _safe_atoms_copy(atoms_a_initial),
                )
            if atoms_b_initial is not None:
                _atomic_extxyz(
                    folder / "state_b_initial.extxyz",
                    _safe_atoms_copy(atoms_b_initial),
                )
            if atoms_a is not None:
                _atomic_extxyz(folder / "state_a.extxyz", _safe_atoms_copy(atoms_a))
            else:
                _log.warning(
                    "ReactionWriter: diffusion lateral_class iso=%d lat=%d "
                    "has no atoms_a — state_a.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_b is not None:
                _atomic_extxyz(folder / "state_b.extxyz", _safe_atoms_copy(atoms_b))
            else:
                _log.warning(
                    "ReactionWriter: diffusion lateral_class iso=%d lat=%d "
                    "has no atoms_b — state_b.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_ts is not None:
                _atomic_extxyz(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts))
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
                _atomic_extxyz(
                    folder / "neb_path.extxyz",
                    [_safe_atoms_copy(im) for im in atoms_neb_path],
                )
            atoms_neb_path_initial = getattr(
                lc,
                "atoms_neb_path_initial",
                None,
            )
            if atoms_neb_path_initial:
                _atomic_extxyz(
                    folder / "neb_path_initial.extxyz",
                    [_safe_atoms_copy(im) for im in atoms_neb_path_initial],
                )
            _write_neb_refinement_assets(folder, lc)
        elif sub == "bond":
            atoms_ab_initial = getattr(lc, "atoms_ab_initial", None)
            atoms_c_initial = getattr(lc, "atoms_c_initial", None)
            atoms_ab = getattr(lc, "atoms_ab", None)
            atoms_c  = getattr(lc, "atoms_c",  None)
            atoms_ts = getattr(lc, "atoms_ts", None)
            atoms_c_gas_reference = getattr(
                lc, "atoms_c_gas_reference", None
            )
            atoms_gas_molecule = getattr(lc, "atoms_gas_molecule", None)
            if atoms_ab_initial is not None:
                _atomic_extxyz(
                    folder / "state_ab_initial.extxyz",
                    _safe_atoms_copy(atoms_ab_initial),
                )
            if atoms_c_initial is not None:
                _atomic_extxyz(
                    folder / "state_c_initial.extxyz",
                    _safe_atoms_copy(atoms_c_initial),
                )
            if atoms_ab is not None:
                _atomic_extxyz(folder / "state_ab.extxyz", _safe_atoms_copy(atoms_ab))
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_ab — state_ab.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_c is not None:
                _atomic_extxyz(folder / "state_c.extxyz", _safe_atoms_copy(atoms_c))
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_c — state_c.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_c_gas_reference is not None:
                _atomic_extxyz(
                    folder / "state_c_gas_reference.extxyz",
                    _safe_atoms_copy(atoms_c_gas_reference),
                )
            if atoms_gas_molecule is not None:
                _atomic_extxyz(
                    folder / "gas_molecule.extxyz",
                    _safe_atoms_copy(atoms_gas_molecule),
                )
            if atoms_ts is not None:
                _atomic_extxyz(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts))
            else:
                _log.warning(
                    "ReactionWriter: bond lateral_class iso=%d lat=%d "
                    "has no atoms_ts — ts.extxyz will not be written.",
                    iso, lat,
                )
            atoms_neb_path = getattr(lc, "atoms_neb_path", None)
            if atoms_neb_path:
                _atomic_extxyz(
                    folder / "neb_path.extxyz",
                    [_safe_atoms_copy(im) for im in atoms_neb_path],
                )
            atoms_neb_path_initial = getattr(
                lc,
                "atoms_neb_path_initial",
                None,
            )
            if atoms_neb_path_initial:
                _atomic_extxyz(
                    folder / "neb_path_initial.extxyz",
                    [_safe_atoms_copy(im) for im in atoms_neb_path_initial],
                )
            _write_neb_refinement_assets(folder, lc)
        else:
            # Stamped onto the lateral class by check_site_stability().
            atoms_occ_initial = getattr(lc, "atoms_occupied_initial", None)
            atoms_unocc_initial = getattr(lc, "atoms_unoccupied_initial", None)
            atoms_occ   = getattr(lc, "atoms_occupied",   None)
            atoms_unocc = getattr(lc, "atoms_unoccupied", None)
            if atoms_occ_initial is not None:
                _atomic_extxyz(
                    folder / "occupied_initial.extxyz",
                    _safe_atoms_copy(atoms_occ_initial),
                )
            if atoms_unocc_initial is not None:
                _atomic_extxyz(
                    folder / "unoccupied_initial.extxyz",
                    _safe_atoms_copy(atoms_unocc_initial),
                )
            if atoms_occ is not None:
                _atomic_extxyz(
                    folder / "occupied.extxyz",
                    _safe_atoms_copy(atoms_occ),
                )
            else:
                _log.warning(
                    "ReactionWriter: lateral_class iso=%d lat=%d has no "
                    "atoms_occupied — occupied.extxyz will not be written.",
                    iso, lat,
                )
            if atoms_unocc is not None:
                _atomic_extxyz(
                    folder / "unoccupied.extxyz",
                    _safe_atoms_copy(atoms_unocc),
                )
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
        self._folder_paths[key] = folder
        existing_discovery_step = None
        metadata_path = folder / "reaction.json"
        if metadata_path.is_file():
            existing_discovery_step = _read_discovery_step(metadata_path)
        self._discovery_steps[key] = (
            int(discovery_step)
            if existing_discovery_step is None
            else existing_discovery_step
        )
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
        defer: bool = False,
    ) -> dict[str, Any]:
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
        payload = build_reaction_payload(
            reaction,
            subdir=sub,
            iso_class=iso,
            lateral_class=lat,
            discovery_step=self._discovery_steps[(sub, species, iso, lat)],
            smiles=smiles,
            description=_reaction_description(reaction, smiles),
            step=step,
            fired=fired,
            stats=meta,
            calculator_meta=self._calc_meta,
            run_id=self.run_id,
            gas_energies=gas_energies,
            gas_free_energies=gas_free_energies,
        )
        key = (sub, species, iso, lat)
        if defer:
            self._pending_payloads[key] = payload
        else:
            _atomic_json(folder / "reaction.json", payload)
        definition = reaction_definition_from_document(
            payload,
            folder=folder.relative_to(self.output_dir),
            run_id=self.run_id,
        )
        reaction_id = str(definition["reaction_id"])
        self._reaction_definitions[reaction_id] = definition
        if self._index is not None:
            self._index.register(definition)
        return payload

    def _queue_event_metadata(
        self,
        event: dict[str, Any],
    ) -> None:
        """Update only dynamic reaction.json fields after a firing."""
        iso = int(event["iso_class"])
        lat = int(event["lateral_class"])
        sub = _kind_subdir(str(event["kind"]))
        species = _smiles_to_dirname(str(event["reactant_smiles"]))
        key = (sub, species, iso, lat)
        meta = self._folder_meta[key]
        step = int(event["step"])
        meta["count"] += 1
        if meta["first_step"] is None:
            meta["first_step"] = step
        meta["last_step"] = step

        payload = self._pending_payloads.get(key)
        if payload is None:
            path = self._folder_paths[key] / "reaction.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload["stats"] = dict(meta)
        rate_energy_bases = {
            str(value)
            for value in payload.get("rate_energy_bases", [])
            if value
        }
        if event.get("rate_energy_basis"):
            rate_energy_bases.add(str(event["rate_energy_basis"]))
        payload["rate_energy_bases"] = sorted(rate_energy_bases)
        if event.get("description") is not None:
            payload["description"] = event["description"]
        last_event = {
            "kind": str(event["kind"]),
            "delta_e_ev": event.get("rate_delta_ev", event.get("delta_e_ev")),
            "barrier_ev": event.get(
                "rate_barrier_ev",
                event.get("barrier_ev"),
            ),
            "rate_hz": event.get("rate_hz"),
            "step": step,
        }
        if sub in {"diffusion", "bond"}:
            last_event["direction"] = event.get("direction")
        payload["last_event"] = last_event
        self._pending_payloads[key] = payload
        if self._index is not None:
            self._index.update_stats(
                str(event["reaction_id"]),
                count=int(meta["count"]),
                first_step=meta["first_step"],
                last_step=meta["last_step"],
                rate_energy_basis=event.get("rate_energy_basis"),
            )

    def _refresh_bond_structure_document(
        self,
        folder: Path,
        reaction,
        key: tuple[str, str, int, int],
        *,
        step: int,
        gas_energies: dict[str, float] | None,
        gas_free_energies: dict[str, float] | None,
    ) -> None:
        """Rebuild static bond metadata after late asset reconciliation."""
        metadata_path = folder / "reaction.json"
        existing = self._pending_payloads.get(key)
        pending = existing is not None
        if existing is None:
            if not metadata_path.is_file():
                return
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))

        iso = int(reaction.site.iso_class)
        lat = int(reaction.lateral_class.lateral_class)
        smiles = _reaction_smiles(reaction)
        refreshed = build_reaction_payload(
            reaction,
            subdir="bond",
            iso_class=iso,
            lateral_class=lat,
            discovery_step=self._discovery_steps[key],
            smiles=smiles,
            description=_reaction_description(reaction, smiles),
            step=int(step),
            fired=False,
            stats=self._folder_meta[key],
            calculator_meta=self._calc_meta,
            run_id=self.run_id,
            gas_energies=gas_energies,
            gas_free_energies=gas_free_energies,
        )
        if existing.get("last_event") is not None:
            refreshed["last_event"] = existing["last_event"]
        if existing.get("rate_energy_bases"):
            refreshed["rate_energy_bases"] = existing["rate_energy_bases"]

        if refreshed == existing:
            return
        if pending:
            self._pending_payloads[key] = refreshed
        else:
            _atomic_json(metadata_path, refreshed)
        definition = reaction_definition_from_document(
            refreshed,
            folder=folder.relative_to(self.output_dir),
            run_id=self.run_id,
        )
        reaction_id = str(definition["reaction_id"])
        self._reaction_definitions[reaction_id] = definition
        if self._index is not None:
            self._index.register(definition)

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

        Idempotent: the first call materialises the folder and writes
        ``occupied.extxyz`` / ``unoccupied.extxyz`` and an initial
        ``reaction.json`` (with ``stats.count = 0``). Subsequent bond calls
        reconcile any structures that became available after the folder was
        first registered, including across checkpoint resumes.

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
            folder = self._folder_paths.get(
                key,
                self.reactions_root
                / sub
                / species
                / _kind_folder_name(sub, iso, lat),
            )
            if sub == "bond":
                wrote_assets = _write_missing_bond_result_assets(
                    folder,
                    reaction.site,
                    reaction.lateral_class,
                )
                if wrote_assets or key not in self._reconciled_bond_keys:
                    self._refresh_bond_structure_document(
                        folder,
                        reaction,
                        key,
                        step=int(step),
                        gas_energies=gas_energies,
                        gas_free_energies=gas_free_energies,
                    )
                    self._reconciled_bond_keys.add(key)
            return folder
        folder = self._ensure_reaction_folder(
            reaction,
            discovery_step=int(step),
        )
        self._write_reaction_json(
            folder, reaction, step, gas_energies, gas_free_energies, fired=False,
        )
        if sub == "bond":
            self._reconciled_bond_keys.add(key)
        return folder

    # ------------------------------------------------------------------
    def write_invalid_adsorption(self, site, lc, *, step: int = 0) -> Path:
        """Write last-known structures for a failed adsorption lateral class."""
        iso = int(site.iso_class)
        lat = int(lc.lateral_class)
        smiles = str(getattr(site, "reactant", ""))
        species = _smiles_to_dirname(smiles) if smiles else "unknown"
        key = ("adsorption_invalid", species, iso, lat)
        folder = (
            self.invalid_adsorption_root
            / species
            / f"ads_iso{iso}_lat{lat}"
        )
        if key in self._folder_meta:
            _write_neb_refinement_assets(folder, lc)
            return self._folder_paths.get(key, folder)

        ensure_directory(folder)
        structure_specs = (
            (
                "occupied_initial",
                "occupied_initial.extxyz",
                getattr(lc, "atoms_occupied_initial", None),
            ),
            (
                "occupied",
                "occupied.extxyz",
                getattr(lc, "atoms_occupied", None),
            ),
            (
                "unoccupied_initial",
                "unoccupied_initial.extxyz",
                getattr(lc, "atoms_unoccupied_initial", None),
            ),
            (
                "unoccupied",
                "unoccupied.extxyz",
                getattr(lc, "atoms_unoccupied", None),
            ),
        )
        atom_assets: dict[str, str | None] = {}
        for key_name, filename, atoms in structure_specs:
            if atoms is not None:
                _atomic_extxyz(folder / filename, _safe_atoms_copy(atoms))
                atom_assets[key_name] = filename
            else:
                atom_assets[key_name] = None

        payload = {
            "artifact_type": "autokmc-invalid-adsorption-lateral-diagnostic",
            "schema_version": "1",
            "kind": "adsorption",
            "discovery_step": int(step),
            "iso_class": iso,
            "lateral_class": lat,
            "reactant_smiles": smiles,
            "invalid_reason": getattr(lc, "invalid_reason", None),
            "structures": atom_assets,
            "energies_ev": {
                "occupied": (
                    None
                    if getattr(lc, "energy_occupied", None) is None
                    else float(lc.energy_occupied)
                ),
                "unoccupied": (
                    None
                    if getattr(lc, "energy_unoccupied", None) is None
                    else float(lc.energy_unoccupied)
                ),
            },
            "calculator": dict(self._calc_meta),
            "run_id": self.run_id,
        }
        _atomic_json(folder / "diagnostic.json", payload)
        self._folder_meta[key] = {
            "count": 0,
            "first_step": None,
            "last_step": None,
        }
        self._folder_paths[key] = folder
        self._discovery_steps[key] = int(step)
        return folder

    # ------------------------------------------------------------------
    def write_invalid_diffusion(self, ds, lc, *, step: int = 0) -> Path:
        """Write an on-disk record for a diffusion lateral class that failed NEB.

        Creates
        ``diagnostics/invalid_diffusion/<species>/diff_iso{X}_lat{Y}/`` so
        failed candidates do not appear in the authoritative reaction network.
        A compact invalid entry is still included in ``reactions/index.jsonl``.
        Any partial atoms stored on *lc* remain available for post-mortem
        inspection.

        Idempotent — a second call for the same ``(iso, lat, species)`` is
        a no-op.
        """
        iso     = int(ds.iso_class)
        lat     = int(lc.lateral_class)
        smiles  = getattr(ds, "reactant", "")
        species = _smiles_to_dirname(smiles) if smiles else "unknown"
        key     = ("diffusion_invalid", species, iso, lat)
        folder = (
            self.invalid_diffusion_root
            / species
            / _diffusion_folder_name(iso, lat)
        )

        if key in self._folder_meta:
            return self._folder_paths.get(key, folder)

        ensure_directory(folder)

        atoms_a  = getattr(lc, "atoms_a",  None)
        atoms_b  = getattr(lc, "atoms_b",  None)
        atoms_ts = getattr(lc, "atoms_ts", None)
        atoms_a_initial = getattr(lc, "atoms_a_initial", None)
        atoms_b_initial = getattr(lc, "atoms_b_initial", None)
        if atoms_a_initial is not None:
            _atomic_extxyz(
                folder / "state_a_initial.extxyz",
                _safe_atoms_copy(atoms_a_initial),
            )
        if atoms_b_initial is not None:
            _atomic_extxyz(
                folder / "state_b_initial.extxyz",
                _safe_atoms_copy(atoms_b_initial),
            )
        if atoms_a is not None:
            _atomic_extxyz(folder / "state_a.extxyz", _safe_atoms_copy(atoms_a))
        if atoms_b is not None:
            _atomic_extxyz(folder / "state_b.extxyz", _safe_atoms_copy(atoms_b))
        if atoms_ts is not None:
            _atomic_extxyz(folder / "ts.extxyz", _safe_atoms_copy(atoms_ts))
        atoms_neb_path_initial = getattr(lc, "atoms_neb_path_initial", None)
        if atoms_neb_path_initial:
            _atomic_extxyz(
                folder / "neb_path_initial.extxyz",
                [_safe_atoms_copy(image) for image in atoms_neb_path_initial],
            )
        atoms_neb_path = getattr(lc, "atoms_neb_path", None)
        if atoms_neb_path:
            _atomic_extxyz(
                folder / "neb_path.extxyz",
                [_safe_atoms_copy(image) for image in atoms_neb_path],
            )
        _write_neb_refinement_assets(folder, lc)

        metadata_path = folder / "reaction.json"
        existing_discovery_step = (
            _read_discovery_step(metadata_path)
            if metadata_path.is_file()
            else None
        )
        discovery_step = (
            int(step)
            if existing_discovery_step is None
            else existing_discovery_step
        )
        failure_reason = getattr(lc, "last_failure_reason", None)
        composite = getattr(lc, "direct_event_status", None) == "composite"
        invalid_reason = (
            getattr(lc, "invalid_reason", None)
            or failure_reason
            or getattr(lc, "direct_event_reason", None)
        )
        stable = getattr(lc, "stable", None)
        numerical_failure = bool(
            stable is None and failure_reason
        )
        payload = {
            "artifact_type": REACTION_DOCUMENT_ARTIFACT_TYPE,
            "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
            "kind":            "diffusion",
            "discovery_step":  discovery_step,
            "iso_class":       iso,
            "lateral_class":   lat,
            "reactant_smiles": smiles,
            "valid":           False,
            "stable":          stable,
            "invalid_reason":  invalid_reason,
            "diagnostic_status": (
                "composite_direct_event"
                if composite
                else ("numerical_failure" if numerical_failure else "invalid")
            ),
            "retryable": False,
            "automatic_retry": False,
            "kind_directions": ["a_to_b", "b_to_a"],
            "description": (
                "Composite diffusion candidate removed from KMC"
                if composite
                else "Invalid diffusion candidate"
                if invalid_reason is None
                else (
                    "Diffusion candidate evaluation stopped after a "
                    "numerical failure (automatic retry disabled): "
                    f"{invalid_reason}"
                    if numerical_failure
                    else f"Invalid diffusion candidate: {invalid_reason}"
                )
            ),
            "template": {"species": smiles},
            "gas_product": False,
            "neb_intermediate_refinement": getattr(
                lc,
                "neb_intermediate_refinement",
                None,
            ),
            "neb_intermediate_refinement_history": getattr(
                lc,
                "neb_intermediate_refinement_history",
                [],
            ),
            "direct_event_status": getattr(
                lc,
                "direct_event_status",
                None,
            ),
            "direct_event_reason": getattr(
                lc,
                "direct_event_reason",
                None,
            ),
            "direct_event_certificate": getattr(
                lc,
                "direct_event_certificate",
                None,
            ),
            "direct_event_network_signature": getattr(
                lc,
                "direct_event_network_signature",
                None,
            ),
            "rate_energy_bases": [],
            "stats": {"count": 0, "first_step": None, "last_step": None},
            "energies_ev": {
                "state_a":   None if getattr(lc, "energy_a",  None) is None else float(lc.energy_a),
                "state_b":   None if getattr(lc, "energy_b",  None) is None else float(lc.energy_b),
                "transition": None if getattr(lc, "energy_ts", None) is None else float(lc.energy_ts),
            },
            "atoms": {
                "state_a_initial": (
                    "state_a_initial.extxyz"
                    if atoms_a_initial is not None
                    else None
                ),
                "state_b_initial": (
                    "state_b_initial.extxyz"
                    if atoms_b_initial is not None
                    else None
                ),
                "state_a": (
                    "state_a.extxyz" if atoms_a is not None else None
                ),
                "state_b": (
                    "state_b.extxyz" if atoms_b is not None else None
                ),
                "transition": (
                    "ts.extxyz" if atoms_ts is not None else None
                ),
                "neb_refinement_initial": (
                    "neb_refinement_initial.extxyz"
                    if getattr(lc, "atoms_neb_refinement_initial", None)
                    is not None
                    else None
                ),
                "neb_refinement_final": (
                    "neb_refinement_final.extxyz"
                    if getattr(lc, "atoms_neb_refinement_final", None)
                    is not None
                    else None
                ),
                "neb_path_initial": (
                    "neb_path_initial.extxyz"
                    if atoms_neb_path_initial
                    else None
                ),
                "neb_path": (
                    "neb_path.extxyz" if atoms_neb_path else None
                ),
            },
            "calculator": dict(self._calc_meta),
        }
        payload["run_id"] = self.run_id
        payload["reaction_id"] = stable_reaction_id(
            "diffusion",
            smiles,
            iso,
            lat,
        )
        _atomic_json(metadata_path, payload)

        self._folder_meta[key] = {"count": 0, "first_step": None, "last_step": None}
        self._folder_paths[key] = folder
        self._discovery_steps[key] = discovery_step
        definition = reaction_definition_from_document(
            payload,
            folder=folder.relative_to(self.output_dir),
            run_id=self.run_id,
        )
        reaction_id = str(definition["reaction_id"])
        self._reaction_definitions[reaction_id] = definition
        if self._index is not None:
            self._index.register(definition)
        _log.info(
            "ReactionWriter: wrote invalid diffusion folder "
            "species=%s iso=%d lat=%d  reason=%s",
            species, iso, lat, invalid_reason,
        )
        return folder

    # ------------------------------------------------------------------
    def write_invalid_bond(self, brs, lc, *, step: int = 0) -> Path:
        """Write endpoint and NEB diagnostics for an invalid bond candidate."""
        iso = int(brs.iso_class)
        lat = int(lc.lateral_class)
        template = brs.template
        smiles = (
            f"{template.smiles_a}+{template.smiles_b}"
            f"↔{template.smiles_c}"
        )
        species = _smiles_to_dirname(smiles) if smiles else "unknown"
        key = ("bond_invalid", species, iso, lat)
        folder = (
            self.invalid_bond_root
            / species
            / _kind_folder_name("bond", iso, lat)
        )
        ensure_directory(folder)
        _prepare_bond_gas_reference_assets(brs, lc)
        structure_specs = (
            ("state_ab_initial", "state_ab_initial.extxyz", getattr(lc, "atoms_ab_initial", None)),
            ("state_c_initial", "state_c_initial.extxyz", getattr(lc, "atoms_c_initial", None)),
            ("state_ab", "state_ab.extxyz", getattr(lc, "atoms_ab", None)),
            ("state_c", "state_c.extxyz", getattr(lc, "atoms_c", None)),
            (
                "state_c_gas_reference",
                "state_c_gas_reference.extxyz",
                getattr(lc, "atoms_c_gas_reference", None),
            ),
            (
                "gas_molecule",
                "gas_molecule.extxyz",
                getattr(lc, "atoms_gas_molecule", None),
            ),
            ("transition", "ts.extxyz", getattr(lc, "atoms_ts", None)),
            (
                "neb_refinement_initial",
                "neb_refinement_initial.extxyz",
                getattr(lc, "atoms_neb_refinement_initial", None),
            ),
            (
                "neb_refinement_final",
                "neb_refinement_final.extxyz",
                getattr(lc, "atoms_neb_refinement_final", None),
            ),
        )
        path_specs = (
            (
                "neb_path_initial",
                "neb_path_initial.extxyz",
                "atoms_neb_path_initial",
            ),
            ("neb_path", "neb_path.extxyz", "atoms_neb_path"),
        )
        if key in self._folder_meta:
            missing_structure = any(
                atoms is not None and not (folder / filename).is_file()
                for _, filename, atoms in structure_specs
            )
            missing_path = any(
                getattr(lc, attribute, None)
                and not (folder / filename).is_file()
                for _, filename, attribute in path_specs
            )
            if not missing_structure and not missing_path:
                return self._folder_paths.get(key, folder)

        atom_assets: dict[str, str | None] = {}
        for key_name, filename, atoms in structure_specs:
            path = folder / filename
            if atoms is not None and not path.is_file():
                _atomic_extxyz(path, _safe_atoms_copy(atoms))
            atom_assets[key_name] = filename if path.is_file() else None

        for key_name, filename, attribute in path_specs:
            images = getattr(lc, attribute, None)
            path = folder / filename
            if images and not path.is_file():
                _atomic_extxyz(
                    path,
                    [_safe_atoms_copy(image) for image in images],
                )
            atom_assets[key_name] = filename if path.is_file() else None

        metadata_path = folder / "reaction.json"
        existing_discovery_step = (
            _read_discovery_step(metadata_path)
            if metadata_path.is_file()
            else None
        )
        discovery_step = (
            int(step)
            if existing_discovery_step is None
            else existing_discovery_step
        )
        failure_reason = getattr(lc, "last_failure_reason", None)
        composite = getattr(lc, "direct_event_status", None) == "composite"
        invalid_reason = (
            getattr(lc, "invalid_reason", None)
            or failure_reason
            or getattr(lc, "direct_event_reason", None)
        )
        stable = getattr(lc, "stable", None)
        numerical_failure = bool(
            stable is None and failure_reason
        )
        payload = {
            "artifact_type": REACTION_DOCUMENT_ARTIFACT_TYPE,
            "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
            "kind": "bond",
            "discovery_step": discovery_step,
            "iso_class": iso,
            "lateral_class": lat,
            "reactant_smiles": smiles,
            "valid": False,
            "stable": stable,
            "invalid_reason": invalid_reason,
            "diagnostic_status": (
                "composite_direct_event"
                if composite
                else ("numerical_failure" if numerical_failure else "invalid")
            ),
            "retryable": False,
            "automatic_retry": False,
            "kind_directions": ["couple", "dissoc"],
            "description": (
                "Composite bond candidate removed from KMC"
                if composite
                else "Invalid bond candidate"
                if invalid_reason is None
                else (
                    "Bond candidate evaluation stopped after a numerical "
                    "failure (automatic retry disabled): "
                    f"{invalid_reason}"
                    if numerical_failure
                    else f"Invalid bond candidate: {invalid_reason}"
                )
            ),
            "template": {
                "smiles_a": template.smiles_a,
                "smiles_b": template.smiles_b,
                "smiles_c": template.smiles_c,
                "bond_type": getattr(template, "bond_type", None),
                "source": getattr(template, "source", None),
            },
            "gas_product": bool(getattr(brs, "gas_product", False)),
            "neb_intermediate_refinement": getattr(
                lc,
                "neb_intermediate_refinement",
                None,
            ),
            "neb_intermediate_refinement_history": getattr(
                lc,
                "neb_intermediate_refinement_history",
                [],
            ),
            "direct_event_status": getattr(
                lc,
                "direct_event_status",
                None,
            ),
            "direct_event_reason": getattr(
                lc,
                "direct_event_reason",
                None,
            ),
            "direct_event_certificate": getattr(
                lc,
                "direct_event_certificate",
                None,
            ),
            "direct_event_network_signature": getattr(
                lc,
                "direct_event_network_signature",
                None,
            ),
            "rate_energy_bases": [],
            "stats": {"count": 0, "first_step": None, "last_step": None},
            "energies_ev": {
                "state_ab": (
                    None
                    if getattr(lc, "energy_ab", None) is None
                    else float(lc.energy_ab)
                ),
                "state_c": (
                    None
                    if getattr(lc, "energy_c", None) is None
                    else float(lc.energy_c)
                ),
                "state_c_gas_reference": (
                    None
                    if getattr(lc, "energy_c_gas_reference", None) is None
                    else float(lc.energy_c_gas_reference)
                ),
                "gas_molecule": (
                    None
                    if getattr(getattr(brs, "gas_reactant", None), "energy", None)
                    is None
                    else float(brs.gas_reactant.energy)
                ),
                "transition_raw": (
                    None
                    if getattr(lc, "energy_ts", None) is None
                    else float(lc.energy_ts)
                ),
                "transition_eff": None,
            },
            "atoms": atom_assets,
            "calculator": dict(self._calc_meta),
            "run_id": self.run_id,
        }
        payload["reaction_id"] = stable_reaction_id(
            "bond",
            smiles,
            iso,
            lat,
        )
        _atomic_json(metadata_path, payload)

        self._folder_meta[key] = {
            "count": 0,
            "first_step": None,
            "last_step": None,
        }
        self._folder_paths[key] = folder
        self._discovery_steps[key] = discovery_step
        definition = reaction_definition_from_document(
            payload,
            folder=folder.relative_to(self.output_dir),
            run_id=self.run_id,
        )
        reaction_id = str(definition["reaction_id"])
        self._reaction_definitions[reaction_id] = definition
        if self._index is not None:
            self._index.register(definition)
        _log.info(
            "ReactionWriter: wrote invalid bond folder "
            "species=%s iso=%d lat=%d reason=%s",
            species,
            iso,
            lat,
            invalid_reason,
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
        transition: dict[str, Any] | None = None,
        # Backwards-compat — ignored under the new per-reaction-folder layout.
        atoms_initial: Atoms | None = None,
        atoms_final:   Atoms | None = None,
    ) -> ReactionRecord:
        """Persist one KMC event (lazy folder creation + JSONL append)."""
        if self._fp is None:
            raise RuntimeError("ReactionWriter has been closed")

        transition = dict(transition or reaction_transition(reaction))

        smiles = _reaction_smiles(reaction)
        folder = self.ensure_reaction(
            reaction,
            step=step,
            gas_energies=gas_energies,
            gas_free_energies=gas_free_energies,
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
        reaction_id = stable_reaction_id(
            str(reaction.kind),
            str(smiles),
            int(reaction.site.iso_class),
            int(reaction.lateral_class.lateral_class),
        )

        rec = ReactionRecord(
            schema_version  = EVENT_SCHEMA_VERSION,
            event_id        = stable_event_id(self.run_id, int(step)),
            reaction_id     = reaction_id,
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
            inputs          = list(transition["inputs"]),
            outputs         = list(transition["outputs"]),
            template        = transition.get("template"),
            gas_product     = bool(transition.get("gas_product", False)),
            run_id          = self.run_id,
            direction       = getattr(reaction, "direction", None),
            delta_g_ev      = delta_g_ev,
            barrier_g_ev    = barrier_g_ev,
            rate_energy_basis = rate_energy_basis,
            rate_delta_ev     = rate_delta_ev,
            rate_barrier_ev   = rate_barrier_ev,
        )

        event_payload = rec.to_jsonable(include_static=False)
        self._append_event(event_payload)
        self._n_written += 1
        # Static structures, energetics, vibration arrays, and calculator
        # metadata were built when the folder was discovered.  Only counters
        # and the last fired direction change per event.
        self._queue_event_metadata(rec.to_jsonable(include_static=True))
        return rec

    # ------------------------------------------------------------------
    @instrument("persistence.event_append")
    def _append_event(self, payload: dict[str, Any]) -> None:
        """Buffer one event row until the next durable checkpoint boundary."""
        if self._fp is None:
            raise RuntimeError("ReactionWriter has been closed")
        self._fp.write(json.dumps(payload) + "\n")
        self._events_dirty = True

    @instrument("persistence.metadata_flush")
    def flush_reaction_stats(self) -> None:
        """Atomically publish all batched reaction.json updates."""
        for key, payload in list(self._pending_payloads.items()):
            folder = self._folder_paths.get(key)
            if folder is None:
                continue
            _atomic_json(folder / "reaction.json", payload)
            self._pending_payloads.pop(key, None)

    @instrument("persistence.event_sync")
    def sync_for_checkpoint(self) -> EventLogCommit:
        """Durably sync events and batched metadata before a checkpoint."""
        if self._fp is None:
            raise RuntimeError("ReactionWriter has been closed")
        if self._events_dirty:
            self._fp.flush()
            os.fsync(self._fp.fileno())
            if self._event_directory_dirty:
                fsync_directory(self._jsonl_path.parent)
                self._event_directory_dirty = False
            self._events_dirty = False
        if self._pending_payloads:
            self.flush_reaction_stats()
        if self._index is not None:
            self._index.sync()
        return EventLogCommit(
            count=self._n_written,
            offset=os.fstat(self._fp.fileno()).st_size,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._fp is not None:
            try:
                self.sync_for_checkpoint()
            finally:
                self._fp.close()
                self._fp = None
        if self._index is not None:
            self._index.close()
            self._index = None

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, *exc):  # pragma: no cover
        self.close()
