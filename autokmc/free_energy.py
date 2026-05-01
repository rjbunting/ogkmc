"""
autokmc.free_energy
===================
Vibrational analysis + thermochemistry helpers built on top of ASE.

Two regimes:

* **Gas phase** — :func:`compute_gas_thermo` runs an ASE
  :class:`~ase.vibrations.Vibrations` calculation on the relaxed gas-phase
  reactant, derives the Gibbs free-energy correction from
  :class:`ase.thermochemistry.IdealGasThermo` at ``(T, p)``.
* **Surface (adsorbate / TS)** — :func:`compute_harmonic_thermo` runs
  vibrations only on a subset of atom indices (the *reactive* species) so
  that frozen slab atoms and frozen lateral-shell adsorbates do not enter
  the vibrational manifold.  The Helmholtz / Gibbs correction comes from
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

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms, units

from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# Conversion: 1 bar ≈ 100 000 Pa.
_BAR_PA: float = 1.0e5


# ---------------------------------------------------------------------------
# Config / options
# ---------------------------------------------------------------------------

@dataclass
class FreeEnergyOptions:
    """Runtime knobs for the free-energy / vibrational machinery.

    Mirrors :class:`autokmc.config.FreeEnergyCfg` but lives here so the
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
        ``False``, the TS gets only an averaged endpoint ZPE correction.
    min_frequency_cm : float
        Modes with ``|ν| < min_frequency_cm`` are treated as imaginary /
        spurious and dropped from the harmonic partition function.  The
        raw values are still persisted under ``imaginary_cm``.
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
    min_frequency_cm        : float        = 12.0
    default_symmetry_number : int          = 1
    default_spin            : float        = 0.0
    default_geometry        : str          = "auto"   # "auto" | "linear" | "nonlinear" | "monatomic"
    cache_dir               : str | None   = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_calc(atoms: Atoms, calculator) -> None:
    if atoms.calc is None and calculator is not None:
        atoms.calc = calculator


def _split_real_imag(
    energies_ev: Sequence[complex] | Sequence[float],
    *,
    min_frequency_cm: float,
) -> tuple[list[float], list[float]]:
    """Split ASE Vibrations energies (eV, may be complex) into real & imag cm⁻¹.

    Modes with ``|ν| < min_frequency_cm`` are treated as spurious imaginary
    contributions and routed into the ``imag_cm`` bucket regardless of
    sign (small soft modes blow up entropy estimates).
    """
    real_cm: list[float] = []
    imag_cm: list[float] = []
    for e in energies_ev:
        e_complex = complex(e)
        # ASE convention: imaginary energies are stored as 1j * |e|.
        if abs(e_complex.imag) > abs(e_complex.real):
            nu_cm = abs(e_complex.imag) / units.invcm
            imag_cm.append(float(nu_cm))
            continue
        nu_cm = e_complex.real / units.invcm
        if abs(nu_cm) < float(min_frequency_cm):
            imag_cm.append(float(abs(nu_cm)))
        else:
            real_cm.append(float(nu_cm))
    return real_cm, imag_cm


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


def _vibrate(
    atoms: Atoms,
    indices: Sequence[int] | None,
    *,
    options: FreeEnergyOptions,
    cache_dir: Path,
    label: str,
) -> tuple[list[float], list[float], list[complex]]:
    """Run ASE :class:`~ase.vibrations.Vibrations`.

    Returns ``(real_cm, imag_cm, raw_energies_ev)`` where ``raw_energies_ev``
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
    # Always recompute — caches mix poorly when the underlying calculator
    # state changes or the number of atoms differs between lateral classes.
    # clean(empty_files=False) removes ALL cached displacement files, not
    # just the empty ones, preventing stale force arrays from a previous
    # lateral class (different n_atoms) from poisoning the Hessian assembly.
    vib.clean(empty_files=False)
    vib.run()
    energies = list(vib.get_energies())
    real_cm, imag_cm = _split_real_imag(
        energies, min_frequency_cm=options.min_frequency_cm,
    )
    return real_cm, imag_cm, energies


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

    The vibrational analysis runs on every atom of *atoms*.  Spin and
    symmetry default to safe values (σ=1, S=0) but should be overridden
    via the per-reactant config for diatomics like H₂ (σ=2) or O₂ (S=1).

    Returns a dict with keys:
    ``g_corr_ev``, ``g_total_ev``, ``zpe_ev``, ``entropy_ev_per_k``,
    ``frequencies_cm``, ``imaginary_cm``, ``geometry``,
    ``symmetry_number``, ``spin``, ``temperature_k``, ``pressure_bar``,
    ``pressure_pa``, ``partial_pressure_bar``.

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
            "frequencies_cm":        [],
            "imaginary_cm":          [],
            "geometry":              None,
            "symmetry_number":       None,
            "spin":                  None,
            "temperature_k":         float(temperature_k),
            "pressure_bar":          float(pressure_bar),
            "pressure_pa":           float(pressure_bar) * _BAR_PA,
        }

    from ase.thermochemistry import IdealGasThermo

    snap = atoms.copy()
    _ensure_calc(snap, calculator)

    sym  = int(symmetry_number if symmetry_number is not None
               else options.default_symmetry_number)
    spin_ = float(spin if spin is not None else options.default_spin)
    geom  = geometry or _atoms_geometry(snap, options.default_geometry)

    cache_root = Path(cache_dir) if cache_dir is not None else (
        Path(options.cache_dir) if options.cache_dir else
        Path("./_autokmc_vib_cache")
    )
    label = f"gas_{snap.get_chemical_formula(empirical=False)}"

    real_cm, imag_cm, raw_energies = _vibrate(
        snap, indices=None, options=options,
        cache_dir=cache_root, label=label,
    )

    # IdealGasThermo wants vibrational energies in eV (real, positive).
    # For monatomic / linear / nonlinear: ASE drops the right number of
    # translational/rotational modes automatically based on `geometry`.
    vib_energies_ev = np.asarray(
        [abs(complex(e).real) for e in raw_energies if abs(complex(e).imag) <= abs(complex(e).real)],
        dtype=float,
    )
    # Filter the small / spurious modes consistent with the cm⁻¹ split.
    keep_mask = (vib_energies_ev * 1e7) >= 0  # placeholder — IdealGasThermo
    # IdealGasThermo selects the appropriate number of modes itself, so
    # pass the raw real energies sorted descending.
    vib_energies_ev = np.sort(vib_energies_ev)[::-1]

    thermo = IdealGasThermo(
        vib_energies     = vib_energies_ev,
        geometry         = geom,
        atoms            = snap,
        symmetrynumber   = sym,
        spin             = spin_,
        potentialenergy  = float(energy_ev),
    )
    pressure_pa = float(pressure_bar) * _BAR_PA
    g_total = float(thermo.get_gibbs_energy(
        temperature=float(temperature_k),
        pressure=pressure_pa,
        verbose=False,
    ))
    zpe = float(thermo.get_ZPE_correction())
    s   = float(thermo.get_entropy(
        temperature=float(temperature_k),
        pressure=pressure_pa,
        verbose=False,
    ))

    g_corr = g_total - float(energy_ev)

    # If a temporary cache was used, do not persist it.
    if options.cache_dir is None and cache_dir is None:
        try:
            shutil.rmtree(cache_root, ignore_errors=True)
        except Exception:
            pass

    return {
        "enabled":               True,
        "g_corr_ev":             float(g_corr),
        "g_total_ev":            float(g_total),
        "zpe_ev":                float(zpe),
        "entropy_ev_per_k":      float(s),
        "frequencies_cm":        list(real_cm),
        "imaginary_cm":          list(imag_cm),
        "geometry":              geom,
        "symmetry_number":       sym,
        "spin":                  spin_,
        "temperature_k":         float(temperature_k),
        "pressure_bar":          float(pressure_bar),
        "pressure_pa":           pressure_pa,
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
) -> dict[str, Any]:
    """Compute Helmholtz/Gibbs correction via ASE :class:`HarmonicThermo`.

    Only the atoms in *vib_indices* are displaced.  Frozen slab atoms and
    frozen lateral-shell adsorbates contribute zero by construction.

    Parameters
    ----------
    atoms : Atoms
        Relaxed structure (must already be evaluated by *calculator*; if
        not, the calculator is attached and used for the displaced
        single-points).
    vib_indices : iterable[int]
        Atom indices to displace — typically the *reactive* species.
    energy_ev : float
        The relaxed-structure electronic potential energy (cached on the
        lateral-class dataclass).  Used as the ZPE-anchor reference.
    drop_imaginary : bool
        ``True`` (default) drops imaginary modes from the partition
        function.  Set ``False`` for TS analysis where you want to keep
        all modes (the principal imaginary mode is reported separately).

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
            "frequencies_cm":   [],
            "imaginary_cm":     [],
            "vib_indices":      indices,
            "temperature_k":    float(temperature_k),
        }

    from ase.thermochemistry import HarmonicThermo

    snap = atoms.copy()
    _ensure_calc(snap, calculator)

    cache_root = Path(cache_dir) if cache_dir is not None else (
        Path(options.cache_dir) if options.cache_dir else
        Path("./_autokmc_vib_cache")
    )

    real_cm, imag_cm, raw_energies = _vibrate(
        snap, indices=indices, options=options,
        cache_dir=cache_root, label=label,
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
        nu_cm = abs(ec.real) / units.invcm
        if nu_cm < options.min_frequency_cm:
            if drop_imaginary:
                continue
        vib_energies_ev.append(float(abs(ec.real)))

    vib_arr = np.asarray(sorted(vib_energies_ev, reverse=True), dtype=float)
    if vib_arr.size == 0:
        # No modes survived — return a zero correction rather than letting
        # HarmonicThermo blow up.
        if options.cache_dir is None and cache_dir is None:
            try:
                shutil.rmtree(cache_root, ignore_errors=True)
            except Exception:
                pass
        return {
            "enabled":          True,
            "g_corr_ev":        0.0,
            "g_total_ev":       float(energy_ev),
            "zpe_ev":           0.0,
            "entropy_ev_per_k": 0.0,
            "frequencies_cm":   list(real_cm),
            "imaginary_cm":     list(imag_cm),
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

    if options.cache_dir is None and cache_dir is None:
        try:
            shutil.rmtree(cache_root, ignore_errors=True)
        except Exception:
            pass

    return {
        "enabled":          True,
        "g_corr_ev":        float(g_corr),
        "g_total_ev":       float(g_total),
        "zpe_ev":           float(zpe),
        "entropy_ev_per_k": float(s),
        "frequencies_cm":   list(real_cm),
        "imaginary_cm":     list(imag_cm),
        "vib_indices":      indices,
        "temperature_k":    float(temperature_k),
    }


__all__ = [
    "FreeEnergyOptions",
    "compute_gas_thermo",
    "compute_harmonic_thermo",
]

