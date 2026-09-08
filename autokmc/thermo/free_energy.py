"""
autokmc.thermo.free_energy
===================
Vibrational analysis + thermochemistry helpers built on top of ASE.

Two regimes:

* **Gas phase** — :func:`compute_gas_thermo` runs an ASE
  :class:`~ase.vibrations.Vibrations` calculation on the relaxed gas-phase
  reactant, derives the Gibbs free-energy correction from
  :class:`ase.thermochemistry.IdealGasThermo` at ``(T, p)``.
* **Surface (adsorbate / TS)** — :func:`compute_harmonic_thermo` runs
  vibrations jointly on all adsorbate atoms present in each state. Slab
  atoms do not enter the vibrational manifold. The Helmholtz / Gibbs correction comes from
  :class:`ase.thermochemistry.HarmonicThermo`.

Adsorption rate convention: the user-facing ``ΔG_ads`` and barrier are
computed at the **standard state** (1 bar reference).  The adsorption
rate is then multiplied by the reactant partial pressure (in bar) in the
KMC engine, keeping the energy outputs canonical.

All API functions return plain dicts of floats / float-lists so they can
be stored on lateral-class dataclasses and serialised to JSON without
extra glue.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms

from autokmc.core.pbc import has_real_cell
from autokmc.io.calculation_cache import (
    _scientific_atom_arrays as _calculation_atom_arrays,
    calculator_identity,
)
from autokmc.io.calculators import (
    CalculatorPool,
    acquire_calculator,
    calculator_batch_active,
)
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


# Conversion: 1 bar ≈ 100 000 Pa.
_BAR_PA: float = 1.0e5
_VIBRATION_LOCKS_GUARD = threading.Lock()
_VIBRATION_THREAD_LOCKS: dict[str, tuple[threading.Lock, int]] = {}
VIBRATIONAL_VALIDATION_VERSION = 1
SURFACE_VIBRATION_SUBSYSTEM = "all_adsorbates_v1"


class VibrationalStabilityError(ValueError):
    """A calculated Hessian has the wrong number of unstable directions."""

    def __init__(
        self,
        *,
        label: str,
        stationary_point: str,
        imaginary_ev: list[float],
        significant_imaginary_ev: list[float],
        tolerance_ev: float,
        expected: tuple[int, int],
    ) -> None:
        self.label = label
        self.stationary_point = stationary_point
        self.imaginary_ev = imaginary_ev
        self.significant_imaginary_ev = significant_imaginary_ev
        self.tolerance_ev = tolerance_ev
        count = len(significant_imaginary_ev)
        expected_text = (
            str(expected[0]) if expected[0] == expected[1]
            else f"{expected[0]} to {expected[1]}"
        )
        super().__init__(
            f"{label}: {stationary_point} has {count} significant imaginary "
            f"vibrational mode(s), expected {expected_text}; "
            f"imaginary energies={significant_imaginary_ev!r} eV "
            f"exceed tolerance {tolerance_ev:g} eV"
        )


# ---------------------------------------------------------------------------
# Config / options
# ---------------------------------------------------------------------------

@dataclass
class FreeEnergyOptions:
    """Runtime knobs for the free-energy / vibrational machinery.

    Mirrors :class:`autokmc.io.config.FreeEnergyCfg` but lives here so the
    energetics modules need not import the config layer.

    Attributes
    ----------
    enabled : bool
        Master switch.  When ``False`` every helper is a no-op and the
        downstream rate code falls back to electronic ΔE.
    pressure_bar : float
        Default reactant partial pressure in bar.  Per-reactant overrides
        live in ``ReactantCfg.partial_pressure_bar`` and the
        ``partial_pressures`` dict threaded through the KMC engine.
    vibration_displacement : float
        Finite-difference step (Å) handed to ASE
        :class:`~ase.vibrations.Vibrations`.
    vibration_nfree : int
        Number of displacements per atom direction (2 → central, 4 → 5pt).
    include_ts_vibrations : bool
        If ``True``, run a second harmonic vibrational analysis on the
        NEB / bond TS structure (drops the principal imaginary mode).  If
        ``False``, the TS gets the averaged endpoint free-energy correction.
    min_frequency_ev : float
        Modes with ``|E_vib| < min_frequency_ev`` are treated as imaginary /
        spurious and dropped from the harmonic partition function.  The
        raw values are still persisted under ``imaginary_ev``.
    imaginary_mode_tolerance_ev : float
        Imaginary mode energies above this magnitude invalidate a minimum.
        A bond transition state must have exactly one significant imaginary
        mode; diffusion permits zero or one to preserve endpoint-like maxima.
        This validation runs before the independent real-mode cutoff.
    symmetry_tolerance : float
        Cartesian tolerance in Angstrom passed to pymatgen's molecular
        point-group analyzer when the gas rotational symmetry number is not
        supplied explicitly.
    cache_dir : str | None
        Directory where ASE writes the per-displacement ``.json`` cache.
        ``None`` → an ephemeral dir under the OS temp area is used and
        deleted after the analysis.
    """
    enabled                 : bool         = True
    pressure_bar            : float        = 1.0
    vibration_displacement  : float        = 0.01
    vibration_nfree         : int          = 2
    include_ts_vibrations   : bool         = True
    min_frequency_ev        : float        = 0.0015
    symmetry_tolerance      : float        = 0.3
    default_spin            : float        = 0.0
    default_geometry        : str          = "auto"   # "auto" | "linear" | "nonlinear" | "monatomic"
    cache_dir               : str | None   = None
    #: Modes above this imaginary-energy magnitude invalidate a minimum.
    #: This is independent of the real-mode partition-function cutoff.
    imaginary_mode_tolerance_ev: float     = 0.01


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def vibrational_validation_parameters(options: FreeEnergyOptions) -> dict[str, Any]:
    """Fingerprint the vibrational subsystem and its acceptance policy."""
    return {
        "vibrational_validation_version": VIBRATIONAL_VALIDATION_VERSION,
        "surface_vibration_subsystem": SURFACE_VIBRATION_SUBSYSTEM,
        "imaginary_mode_tolerance_ev": float(getattr(
            options, "imaginary_mode_tolerance_ev",
            FreeEnergyOptions.imaginary_mode_tolerance_ev,
        )),
    }


def _validate_vibrational_stability(
    energies_ev: Sequence[complex] | Sequence[float],
    *,
    tolerance_ev: float,
    stationary_point: str,
    label: str,
) -> None:
    """Validate raw modes before any partition-function filtering.

    A positive soft mode must never be mistaken for an imaginary mode, even
    when ``min_frequency_ev`` excludes it from the thermochemical correction.
    Diffusion retains its explicit endpoint-like maximum policy; only that
    channel permits a transition-state spectrum with no unstable direction.
    """
    expected_by_kind = {
        "minimum": (0, 0),
        "transition_state": (1, 1),
        "diffusion_transition_state": (0, 1),
    }
    if stationary_point not in expected_by_kind:
        raise ValueError(f"Unknown stationary_point {stationary_point!r}")
    tolerance = float(tolerance_ev)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("imaginary_mode_tolerance_ev must be finite and non-negative")
    imaginary: list[float] = []
    for energy in energies_ev:
        value = complex(energy)
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{label}: non-finite vibrational energy {energy!r}")
        if abs(value.imag) > abs(value.real):
            imaginary.append(float(abs(value.imag)))
        elif value.real < 0.0:
            imaginary.append(float(abs(value.real)))
    significant = [energy for energy in imaginary if energy > tolerance]
    expected = expected_by_kind[stationary_point]
    if not expected[0] <= len(significant) <= expected[1]:
        raise VibrationalStabilityError(
            label=label,
            stationary_point=stationary_point,
            imaginary_ev=imaginary,
            significant_imaginary_ev=significant,
            tolerance_ev=tolerance,
            expected=expected,
        )

def _ensure_calc(atoms: Atoms, calculator) -> None:
    if atoms.calc is None and calculator is not None:
        atoms.calc = calculator


def _normalise_harmonic_pbc(atoms: Atoms) -> None:
    """Avoid mixed PBC when calculator-backed vibrations run on slab systems."""
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if pbc.any() and has_real_cell(atoms.get_cell()):
        atoms.set_pbc(True)


@contextmanager
def _vibration_cache_lock(cache_dir: Path, label: str):
    """Serialize one content-addressed vibration cache across threads/processes.

    ASE represents an in-progress displacement with an empty JSON file.
    Deleting empty files before a run can therefore remove another process's
    active lock. Remove abandoned empty files only after acquiring this
    whole-workflow lock. A separate advisory lock protects the complete same-label
    workflow, while the small in-process registry covers platforms where file
    locks are process-scoped.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (cache_dir / f".vib_{label}.lock").resolve()
    key = str(lock_path)
    with _VIBRATION_LOCKS_GUARD:
        entry = _VIBRATION_THREAD_LOCKS.get(key)
        if entry is None:
            thread_lock = threading.Lock()
            _VIBRATION_THREAD_LOCKS[key] = (thread_lock, 1)
        else:
            thread_lock, users = entry
            _VIBRATION_THREAD_LOCKS[key] = (thread_lock, users + 1)

    thread_lock.acquire()
    try:
        with lock_path.open("a+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - non-POSIX fallback
                yield
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        thread_lock.release()
        with _VIBRATION_LOCKS_GUARD:
            registered_lock, users = _VIBRATION_THREAD_LOCKS[key]
            if users == 1:
                del _VIBRATION_THREAD_LOCKS[key]
            else:
                _VIBRATION_THREAD_LOCKS[key] = (
                    registered_lock,
                    users - 1,
                )


def _split_real_imag_ev(
    energies_ev: Sequence[complex] | Sequence[float],
    *,
    min_frequency_ev: float,
) -> tuple[list[float], list[float]]:
    """Split ASE Vibrations mode energies into real and imaginary eV buckets.

    Modes with ``|E| < min_frequency_ev`` are treated as spurious imaginary
    contributions and routed into the ``imag_ev`` bucket regardless of
    sign (small soft modes blow up entropy estimates).
    """
    real_ev: list[float] = []
    imag_ev: list[float] = []
    for e in energies_ev:
        e_complex = complex(e)
        # ASE convention: imaginary energies are stored as 1j * |e|.
        if abs(e_complex.imag) > abs(e_complex.real):
            imag_ev.append(float(abs(e_complex.imag)))
            continue
        e_real = float(e_complex.real)
        if abs(e_real) < float(min_frequency_ev):
            imag_ev.append(float(abs(e_real)))
        else:
            if e_real <= 0.0:
                imag_ev.append(float(abs(e_real)))
            else:
                real_ev.append(float(e_real))
    return real_ev, imag_ev


def _cache_context(
    options: FreeEnergyOptions,
    cache_dir: str | Path | None,
):
    if cache_dir is not None:
        return nullcontext(Path(cache_dir))
    if options.cache_dir:
        return nullcontext(Path(options.cache_dir))
    return tempfile.TemporaryDirectory(prefix="autokmc_vib_")


def _cache_path(cache_root: str | Path) -> Path:
    return Path(cache_root)


def _atoms_geometry(atoms: Atoms, default_geometry: str) -> str:
    """Infer ``IdealGasThermo`` ``geometry`` (linear / nonlinear / monatomic)."""
    if default_geometry != "auto":
        return default_geometry
    n = len(atoms)
    if n == 1:
        return "monatomic"
    if n == 2:
        return "linear"
    # Three+ atoms — linear iff every atom lies on a single line.
    pos = atoms.get_positions()
    v0  = pos[1] - pos[0]
    n0  = float(np.linalg.norm(v0))
    if n0 < 1e-8:
        return "nonlinear"
    v0 /= n0
    for i in range(2, n):
        vi = pos[i] - pos[0]
        ni = float(np.linalg.norm(vi))
        if ni < 1e-8:
            continue
        # Cross product magnitude → 0 ⇒ collinear.
        cross_norm = float(np.linalg.norm(np.cross(v0, vi / ni)))
        if cross_norm > 1e-3:
            return "nonlinear"
    return "linear"


def _infer_rotational_symmetry_number(
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[int, str]:
    """Infer ``(rotational symmetry number, point group)`` from coordinates.

    Multi-atom molecules are classified with pymatgen's
    :class:`~pymatgen.symmetry.analyzer.PointGroupAnalyzer`.  A monatomic gas
    has no rotational contribution, so its symmetry number is one; handling
    it directly also avoids a pymatgen ``PointGroupAnalyzer`` edge case.
    """
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Gas-phase symmetry tolerance must be finite and positive")
    if len(atoms) == 0:
        raise ValueError("Cannot infer gas-phase symmetry for an empty structure")

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError("Cannot infer gas-phase symmetry from non-finite coordinates")
    if len(atoms) == 1:
        return 1, "K_h"

    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        molecule = Molecule(atoms.get_chemical_symbols(), positions)
        analyzer = PointGroupAnalyzer(molecule, tolerance=tol)
        symmetry_number = int(analyzer.get_rotational_symmetry_number())
        point_group = str(analyzer.sch_symbol)
        # Pymatgen's elemental symmetry operations can exchange isotopes.
        # Count only proper rotations that also preserve the nuclear masses.
        from scipy.optimize import linear_sum_assignment

        coordinates = np.asarray(analyzer.centered_mol.cart_coords)
        masses = atoms.get_masses()
        same_nuclei = (
            (atoms.numbers[:, None] == atoms.numbers[None, :])
            & np.isclose(masses[:, None], masses[None, :], rtol=0.0, atol=1e-8)
        )

        def preserves_nuclei(transformed):
            distances = np.linalg.norm(
                transformed[:, None, :] - coordinates[None, :, :], axis=2,
            )
            left, right = linear_sum_assignment(
                np.where(same_nuclei, distances, np.inf),
            )
            return bool(np.all(distances[left, right] <= tol))

        if point_group in {"D*h", "C*v"}:
            # A linear molecule has symmetry number two exactly when end-for-
            # end reversal preserves its nuclei. Some pymatgen versions omit
            # that proper half-turn from the finite operation list.
            symmetry_number = 2 if preserves_nuclei(-coordinates) else 1
            point_group = "D*h" if symmetry_number == 2 else "C*v"
        else:
            symmetry_number = sum(
                1 for operation in analyzer.get_symmetry_operations()
                if np.isclose(np.linalg.det(operation.rotation_matrix), 1.0, atol=1e-4)
                and preserves_nuclei(operation.operate_multi(coordinates))
            )
    except Exception as exc:
        raise ValueError(
            "Could not infer the gas-phase rotational symmetry number from "
            "the molecular coordinates. Supply reactants[].symmetry_number "
            "as an explicit override."
        ) from exc

    if symmetry_number < 1:
        raise ValueError(
            "Pymatgen returned an invalid gas-phase rotational symmetry "
            f"number ({symmetry_number}) for point group {point_group!r}"
        )
    return symmetry_number, point_group


def _vibrate(
    atoms: Atoms,
    indices: Sequence[int] | None,
    *,
    options: FreeEnergyOptions,
    cache_dir: Path,
    label: str,
) -> tuple[list[float], list[float], list[complex]]:
    """Run ASE :class:`~ase.vibrations.Vibrations`.

    Returns ``(real_ev, imag_ev, raw_energies_ev)`` where ``raw_energies_ev``
    is the unfiltered list of mode energies (some may be complex) suitable
    for handing to ASE's :class:`~ase.thermochemistry.HarmonicThermo`.
    """
    from ase.vibrations import Vibrations

    cache_dir.mkdir(parents=True, exist_ok=True)
    name = str(cache_dir / f"vib_{label}")

    # Strip any FixAtoms / FixCartesianMomentum constraints that were set
    # for the ML relaxation step.  These are not needed for finite-difference
    # vibrational analysis (we already restrict displaced atoms via `indices`)
    # and can cause ASE's internal adjust_forces() to raise an IndexError if
    # the constraint index array was sized for a different Atoms object.
    if atoms.constraints:
        atoms.set_constraint([])

    vib = Vibrations(
        atoms,
        indices      = list(indices) if indices is not None else None,
        name         = name,
        delta        = float(options.vibration_displacement),
        nfree        = int(options.vibration_nfree),
    )
    # The outer cache lock excludes live writers. A failed displacement can
    # leave an empty ASE lock file, which must be retried rather than reused.
    vib.clean(empty_files=True)
    vib.run()
    energies = list(vib.get_energies())
    real_ev, imag_ev = _split_real_imag_ev(
        energies, min_frequency_ev=options.min_frequency_ev,
    )
    return real_ev, imag_ev, energies


def _vibration_cache_label(
    atoms: Atoms,
    indices: Sequence[int] | None,
    *,
    options: FreeEnergyOptions,
    label: str,
    calculator,
) -> str:
    """Return a content-addressed label for restart-safe displacement reuse."""
    calculator_payload = (
        None if calculator is None else calculator_identity(calculator)
    )
    payload = {
        "schema": 2,
        "symbols": atoms.get_chemical_symbols(),
        "positions_A": np.asarray(atoms.positions, dtype=float).round(10).tolist(),
        "cell_A": np.asarray(atoms.cell.array, dtype=float).round(10).tolist(),
        "pbc": [bool(value) for value in atoms.pbc],
        # Keep this in lockstep with calculation-cache geometry identity:
        # charges, magnetic moments, tags, and arbitrary calculator-relevant
        # ASE arrays can all change forces at identical coordinates.
        "atom_arrays": _calculation_atom_arrays(atoms),
        "indices": None if indices is None else [int(index) for index in indices],
        "delta_A": float(options.vibration_displacement),
        "nfree": int(options.vibration_nfree),
        "calculator": calculator_payload,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{label}_{digest}"


def _vibrate_parallel(
    atoms: Atoms,
    indices: Sequence[int] | None,
    *,
    calculator: CalculatorPool,
    options: FreeEnergyOptions,
    cache_dir: Path,
    label: str,
) -> tuple[list[float], list[float], list[complex]]:
    """Evaluate independent finite-difference displacements across a pool."""
    from ase.vibrations import Vibrations

    cache_dir.mkdir(parents=True, exist_ok=True)
    if atoms.constraints:
        atoms.set_constraint([])
    vibration = Vibrations(
        atoms,
        indices=list(indices) if indices is not None else None,
        name=str(cache_dir / f"vib_{label}"),
        delta=float(options.vibration_displacement),
        nfree=int(options.vibration_nfree),
    )
    if not vibration.cache.writable:
        raise RuntimeError(
            "Cannot run vibration calculation because its cache is not writable"
        )
    vibration._check_old_pickles()
    vibration.clean(empty_files=True)

    def _calculate(displacement, displaced: Atoms) -> None:
        with vibration.cache.lock(displacement.name) as handle:
            if handle is None:
                return
            with calculator.acquire() as concrete:
                displaced.calc = concrete
                try:
                    result = {
                        "forces": np.asarray(
                            concrete.get_forces(displaced),
                            dtype=float,
                        )
                    }
                    if getattr(vibration, "ir", False):
                        result["dipole"] = concrete.get_dipole_moment(displaced)
                finally:
                    displaced.calc = None
            handle.save(result)

    max_workers = max(
        1,
        min(
            len(calculator),
            int(getattr(calculator, "max_workers", len(calculator)) or len(calculator)),
        ),
    )
    jobs = list(vibration.iterdisplace(inplace=False))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                copy_context().run,
                _calculate,
                displacement,
                displaced,
            )
            for displacement, displaced in jobs
        ]
        for future in futures:
            future.result()

    energies = list(vibration.get_energies())
    real_ev, imag_ev = _split_real_imag_ev(
        energies,
        min_frequency_ev=options.min_frequency_ev,
    )
    return real_ev, imag_ev, energies


def _run_vibrations(
    atoms: Atoms,
    indices: Sequence[int] | None,
    *,
    calculator,
    options: FreeEnergyOptions,
    cache_dir: Path,
    label: str,
    purpose: str,
    persistent_cache: bool = True,
) -> tuple[list[float], list[float], list[complex]]:
    """Choose serial or calculator-pool finite-difference execution."""
    cache_label = (
        _vibration_cache_label(
            atoms,
            indices,
            options=options,
            label=label,
            calculator=calculator,
        )
        if persistent_cache
        else label
    )
    with _vibration_cache_lock(cache_dir, cache_label):
        if (
            isinstance(calculator, CalculatorPool)
            and len(calculator) > 1
            and int(
                getattr(calculator, "max_workers", len(calculator))
                or len(calculator)
            ) > 1
            and not calculator_batch_active()
        ):
            atoms.calc = None
            return _vibrate_parallel(
                atoms,
                indices,
                calculator=calculator,
                options=options,
                cache_dir=cache_dir,
                label=cache_label,
            )

        with acquire_calculator(calculator, purpose=purpose) as concrete:
            _ensure_calc(atoms, concrete)
            return _vibrate(
                atoms,
                indices=indices,
                options=options,
                cache_dir=cache_dir,
                label=cache_label,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gas_thermo(
    atoms: Atoms,
    *,
    energy_ev: float,
    temperature_k: float,
    pressure_bar: float,
    calculator                = None,
    options: FreeEnergyOptions | None = None,
    symmetry_number: int | None = None,
    spin: float | None        = None,
    geometry: str | None      = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute gas-phase Gibbs free energy via ASE :class:`IdealGasThermo`.

    The vibrational analysis runs on every atom of *atoms*.  When
    ``symmetry_number`` is omitted, the rotational symmetry number is inferred
    from the final gas-phase coordinates with pymatgen.  An explicit value is
    retained as an override.  Spin still defaults to S=0 and should be
    overridden for species such as O₂ (S=1).

    The Gibbs free energy is **always** evaluated at the standard-state
    pressure of 1 bar, regardless of the *pressure_bar* argument.  This
    prevents ``log(0)`` errors when a species has ``partial_pressure_bar=0``
    (e.g. on-surface-only leaf species).  The partial-pressure effect on
    the adsorption rate is applied as a multiplicative factor in the KMC
    engine (see :func:`autokmc.reactions.adsorption._energetics_cached`), keeping
    all persisted ΔG / barrier values canonical at the 1 bar standard state.

    Returns a dict with keys:
    ``g_corr_ev``, ``g_total_ev``, ``zpe_ev``, ``entropy_ev_per_k``,
    ``frequencies_ev``, ``imaginary_ev``, ``geometry``,
    ``symmetry_number``, ``symmetry_number_source``, ``point_group``,
    ``symmetry_tolerance``, ``spin``, ``temperature_k``, ``pressure_bar``,
    ``pressure_pa``.

    When ``options.enabled`` is ``False`` (or the gas-phase atom count is
    zero), the function returns a no-op dict where all corrections are
    zero so callers may add the result unconditionally.
    """
    options = options or FreeEnergyOptions()
    if not options.enabled or len(atoms) == 0:
        return {
            "enabled":               False,
            "g_corr_ev":             0.0,
            "g_total_ev":            float(energy_ev),
            "zpe_ev":                0.0,
            "entropy_ev_per_k":      0.0,
            "frequencies_ev":        [],
            "imaginary_ev":          [],
            "geometry":              None,
            "symmetry_number":       None,
            "symmetry_number_source": None,
            "point_group":           None,
            "symmetry_tolerance":    None,
            "spin":                  None,
            "temperature_k":         float(temperature_k),
            "pressure_bar":          float(pressure_bar),   # caller's partial pressure (metadata)
            "pressure_pa":           1.0 * _BAR_PA,         # standard-state reference
        }

    from ase.thermochemistry import IdealGasThermo

    calculator = calculator if calculator is not None else atoms.calc
    snap = atoms.copy()
    snap.set_pbc(False)
    if calculator is not None:
        snap.calc = None

    symmetry_tolerance = float(options.symmetry_tolerance)
    if symmetry_number is None:
        sym, point_group = _infer_rotational_symmetry_number(
            snap,
            tolerance=symmetry_tolerance,
        )
        symmetry_number_source = "inferred"
    else:
        if isinstance(symmetry_number, bool) or int(symmetry_number) != symmetry_number:
            raise ValueError("Gas-phase symmetry number must be an integer")
        sym = int(symmetry_number)
        if sym < 1:
            raise ValueError("Gas-phase symmetry number must be at least one")
        point_group = None
        symmetry_number_source = "explicit"
    spin_ = float(spin if spin is not None else options.default_spin)
    geom  = geometry or _atoms_geometry(snap, options.default_geometry)

    label = f"gas_{snap.get_chemical_formula(empirical=False)}"

    with _cache_context(options, cache_dir) as cache_root_raw:
        cache_root = _cache_path(cache_root_raw)
        real_ev, imag_ev, raw_energies = _run_vibrations(
            snap,
            None,
            calculator=calculator,
            options=options,
            cache_dir=cache_root,
            label=label,
            purpose="gas-phase thermochemistry",
            persistent_cache=(
                cache_dir is not None or options.cache_dir is not None
            ),
        )

    _validate_vibrational_stability(
        raw_energies,
        tolerance_ev=options.imaginary_mode_tolerance_ev,
        stationary_point="minimum",
        label=label,
    )

    # IdealGasThermo wants vibrational energies in eV (real, positive).
    # Use `real_ev` which has already been filtered by `_split_real_imag_ev`.
    # The raw `raw_energies` list
    # contains spurious near-zero / imaginary modes that would inflate the
    # vibrational partition function and make the computed entropy diverge.
    # IdealGasThermo selects the appropriate number of modes (subtracting
    # translations / rotations) based on `geometry`, so pass all real
    # modes sorted descending.
    vib_energies_ev = np.asarray(sorted(real_ev, reverse=True), dtype=float)

    thermo = IdealGasThermo(
        vib_energies     = vib_energies_ev,
        geometry         = geom,
        atoms            = snap,
        symmetrynumber   = sym,
        spin             = spin_,
        potentialenergy  = float(energy_ev),
    )
    # Always evaluate at the standard-state reference pressure of 1 bar.
    # The caller's partial pressure (which may be 0 for on-surface species)
    # must NOT be passed here — log(0) would cause a divide-by-zero.
    # The per-reactant partial pressure multiplies the KMC adsorption rate
    # separately in kmc_adsorption._energetics_cached.
    _standard_pa = 1.0 * _BAR_PA
    g_total = float(thermo.get_gibbs_energy(
        temperature=float(temperature_k),
        pressure=_standard_pa,
        verbose=False,
    ))
    zpe = float(thermo.get_ZPE_correction())
    s   = float(thermo.get_entropy(
        temperature=float(temperature_k),
        pressure=_standard_pa,
        verbose=False,
    ))

    g_corr = g_total - float(energy_ev)

    return {
        "enabled":               True,
        "g_corr_ev":             float(g_corr),
        "g_total_ev":            float(g_total),
        "zpe_ev":                float(zpe),
        "entropy_ev_per_k":      float(s),
        "frequencies_ev":        list(real_ev),
        "imaginary_ev":          list(imag_ev),
        "geometry":              geom,
        "symmetry_number":       sym,
        "symmetry_number_source": symmetry_number_source,
        "point_group":           point_group,
        "symmetry_tolerance":    symmetry_tolerance,
        "spin":                  spin_,
        "temperature_k":         float(temperature_k),
        "pressure_bar":          float(pressure_bar),   # caller's partial pressure (metadata only)
        "pressure_pa":           _standard_pa,          # standard-state pressure used in computation
    }


def compute_harmonic_thermo(
    atoms: Atoms,
    vib_indices: Iterable[int],
    *,
    energy_ev: float,
    temperature_k: float,
    calculator               = None,
    options: FreeEnergyOptions | None = None,
    cache_dir: str | Path | None = None,
    label: str = "harm",
    drop_imaginary: bool     = True,
    stationary_point: str = "minimum",
) -> dict[str, Any]:
    """Compute Helmholtz/Gibbs correction via ASE :class:`HarmonicThermo`.

    Only the atoms in *vib_indices* are displaced. Surface callers include
    all adsorbate atoms present in each state together, retaining couplings
    between molecules; slab atoms contribute zero by construction.

    Parameters
    ----------
    atoms : Atoms
        Relaxed structure (must already be evaluated by *calculator*; if
        not, the calculator is attached and used for the displaced
        single-points).
    vib_indices : iterable[int]
        Atom indices to displace — all adsorbates for surface states.
    energy_ev : float
        The relaxed-structure electronic potential energy (cached on the
        lateral-class dataclass).  Used as the ZPE-anchor reference.
    drop_imaginary : bool
        ``True`` (default) drops allowed imaginary modes from the partition
        function. Validation always runs before this filtering.
    stationary_point : str
        ``"minimum"`` requires no significant imaginary modes;
        ``"transition_state"`` requires exactly one. The explicit
        ``"diffusion_transition_state"`` policy allows zero or one to
        retain endpoint-like diffusion maxima. Significance uses
        ``options.imaginary_mode_tolerance_ev`` independently of the
        real-mode thermochemistry cutoff.

    Returns
    -------
    dict
        Same shape as :func:`compute_gas_thermo` but with the
        gas-specific fields nulled out and ``g_corr_ev`` containing the
        harmonic Helmholtz correction at *temperature_k*.  Add
        ``g_corr_ev`` to the relaxed electronic ``energy_ev`` to recover
        a Gibbs-like free energy at the surface (PV-work negligible for
        condensed-phase species).
    """
    options = options or FreeEnergyOptions()
    indices = list(vib_indices)
    if not options.enabled or not indices:
        return {
            "enabled":          False,
            "g_corr_ev":        0.0,
            "g_total_ev":       float(energy_ev),
            "zpe_ev":           0.0,
            "entropy_ev_per_k": 0.0,
            "frequencies_ev":   [],
            "imaginary_ev":     [],
            "vib_indices":      indices,
            "temperature_k":    float(temperature_k),
        }
    n_atoms = len(atoms)
    invalid = [idx for idx in indices if idx < 0 or idx >= n_atoms]
    if invalid:
        raise ValueError(
            "vib_indices contains atom indices outside the structure: "
            f"{invalid!r} for {n_atoms} atoms"
        )

    from ase.thermochemistry import HarmonicThermo

    calculator = calculator if calculator is not None else atoms.calc
    snap = atoms.copy()
    _normalise_harmonic_pbc(snap)
    if calculator is not None:
        snap.calc = None
    with _cache_context(options, cache_dir) as cache_root_raw:
        cache_root = _cache_path(cache_root_raw)
        real_ev, imag_ev, raw_energies = _run_vibrations(
            snap,
            indices,
            calculator=calculator,
            options=options,
            cache_dir=cache_root,
            label=label,
            purpose="harmonic thermochemistry",
            persistent_cache=(
                cache_dir is not None or options.cache_dir is not None
            ),
        )

        _validate_vibrational_stability(
            raw_energies,
            tolerance_ev=options.imaginary_mode_tolerance_ev,
            stationary_point=stationary_point,
            label=label,
        )

        # HarmonicThermo wants real, positive energies in eV.
        vib_energies_ev: list[float] = []
        for e in raw_energies:
            ec = complex(e)
            if abs(ec.imag) > abs(ec.real):
                if drop_imaginary:
                    continue
                vib_energies_ev.append(float(abs(ec.imag)))
                continue
            e_real = float(ec.real)
            if e_real <= 0.0 or abs(e_real) < options.min_frequency_ev:
                if drop_imaginary:
                    continue
            vib_energies_ev.append(float(abs(e_real)))

        vib_arr = np.asarray(sorted(vib_energies_ev, reverse=True), dtype=float)
        if vib_arr.size == 0:
            # No modes survived — return a zero correction rather than letting
            # HarmonicThermo blow up.
            return {
                "enabled":          True,
                "g_corr_ev":        0.0,
                "g_total_ev":       float(energy_ev),
                "zpe_ev":           0.0,
                "entropy_ev_per_k": 0.0,
                "frequencies_ev":   list(real_ev),
                "imaginary_ev":     list(imag_ev),
                "vib_indices":      indices,
                "temperature_k":    float(temperature_k),
            }

        thermo = HarmonicThermo(
            vib_energies    = vib_arr,
            potentialenergy = float(energy_ev),
        )
        g_total = float(thermo.get_helmholtz_energy(
            temperature=float(temperature_k), verbose=False,
        ))
        zpe = float(thermo.get_ZPE_correction())
        s   = float(thermo.get_entropy(
            temperature=float(temperature_k), verbose=False,
        ))
        g_corr = g_total - float(energy_ev)

    return {
        "enabled":          True,
        "g_corr_ev":        float(g_corr),
        "g_total_ev":       float(g_total),
        "zpe_ev":           float(zpe),
        "entropy_ev_per_k": float(s),
        "frequencies_ev":   list(real_ev),
        "imaginary_ev":     list(imag_ev),
        "vib_indices":      indices,
        "temperature_k":    float(temperature_k),
    }


__all__ = [
    "FreeEnergyOptions",
    "VibrationalStabilityError",
    "compute_gas_thermo",
    "compute_harmonic_thermo",
]
