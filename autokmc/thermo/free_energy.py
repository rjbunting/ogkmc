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

import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms

from autokmc.io.calculators import acquire_calculator
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


# Conversion: 1 bar ≈ 100 000 Pa.
_BAR_PA: float = 1.0e5


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
        ``False``, the TS gets only an averaged endpoint ZPE correction.
    min_frequency_ev : float
        Modes with ``|E_vib| < min_frequency_ev`` are treated as imaginary /
        spurious and dropped from the harmonic partition function.  The
        raw values are still persisted under ``imaginary_ev``.
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
    # Always recompute — caches mix poorly when the underlying calculator
    # state changes or the number of atoms differs between lateral classes.
    # clean(empty_files=False) removes ALL cached displacement files, not
    # just the empty ones, preventing stale force arrays from a previous
    # lateral class (different n_atoms) from poisoning the Hessian assembly.
    vib.clean(empty_files=False)
    vib.run()
    energies = list(vib.get_energies())
    real_ev, imag_ev = _split_real_imag_ev(
        energies, min_frequency_ev=options.min_frequency_ev,
    )
    return real_ev, imag_ev, energies


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
    ``symmetry_number``, ``spin``, ``temperature_k``, ``pressure_bar``,
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
            "spin":                  None,
            "temperature_k":         float(temperature_k),
            "pressure_bar":          float(pressure_bar),   # caller's partial pressure (metadata)
            "pressure_pa":           1.0 * _BAR_PA,         # standard-state reference
        }

    from ase.thermochemistry import IdealGasThermo

    snap = atoms.copy()
    if calculator is not None:
        snap.calc = None

    sym  = int(symmetry_number if symmetry_number is not None
               else options.default_symmetry_number)
    spin_ = float(spin if spin is not None else options.default_spin)
    geom  = geometry or _atoms_geometry(snap, options.default_geometry)

    label = f"gas_{snap.get_chemical_formula(empirical=False)}"

    with acquire_calculator(calculator, purpose="gas-phase thermochemistry") as calc:
        _ensure_calc(snap, calc)
        with _cache_context(options, cache_dir) as cache_root_raw:
            cache_root = _cache_path(cache_root_raw)
            real_ev, imag_ev, raw_energies = _vibrate(
                snap, indices=None, options=options,
                cache_dir=cache_root, label=label,
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

    snap = atoms.copy()
    if calculator is not None:
        snap.calc = None
    with acquire_calculator(calculator, purpose="harmonic thermochemistry") as calc:
        _ensure_calc(snap, calc)
        with _cache_context(options, cache_dir) as cache_root_raw:
            cache_root = _cache_path(cache_root_raw)
            real_ev, imag_ev, raw_energies = _vibrate(
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
    "compute_gas_thermo",
    "compute_harmonic_thermo",
]
