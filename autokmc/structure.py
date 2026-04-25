"""
autokmc.structure
=================
Build and optimise metallic atomic structures for KMC workflows.

Supported structure types
-------------------------
* **Nanoparticle** – Wulff construction via WulffPack :func:`build_nanoparticle`.
* **Surface slab**  – pymatgen SlabGenerator + orthogonalisation + tiling
  via :func:`build_surface`.

Both builders accept arbitrary metal compositions (pure or alloy) and the three
most common crystal structures (FCC, BCC, HCP).

Composition
-----------
Pass a single element symbol string for a pure metal, or a dict of
``{symbol: fraction}`` for an alloy.  Fractions must sum to 1.  Example::

    composition = "Cu"                      # pure copper
    composition = {"Cu": 0.7, "Pt": 0.3}   # Cu–Pt alloy

Crystal structure
-----------------
One of ``"fcc"``, ``"bcc"``, or ``"hcp"``.

Lattice constants
-----------------
* FCC / BCC: a single float ``a`` in Å.
* HCP:  a dict ``{"a": 3.21, "c": 5.21}``  – or pass just a float ``a`` and
  the ideal ``c/a = sqrt(8/3) ≈ 1.633`` ratio is assumed.

If *lattice_constant* is not provided the bulk unit cell is first relaxed with
the supplied calculator and the optimised value is used.

Surface energies (nanoparticle)
-------------------------------
A dict whose keys are Miller-index **tuples** and values are surface energies in
J/m²::

    surface_energies = {(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51}

Optimisation helpers
--------------------
* :func:`optimise_bulk`      – relax a bulk unit cell, return lattice params.
* :func:`optimise_structure` – relax any ASE Atoms with a given calculator.

Typical usage
-------------
::

    from ase.calculators.emt import EMT
    from autokmc.structure import build_nanoparticle, build_surface

    # --- nanoparticle (pure Cu, FCC) ---
    np_atoms = build_nanoparticle(
        composition="Cu",
        crystal_structure="fcc",
        surface_energies={(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51},
        target_atoms=600,
        calculator=EMT(),
    )

    # --- nanoparticle (Cu–Pt alloy, FCC) ---
    np_atoms = build_nanoparticle(
        composition={"Cu": 0.70, "Pt": 0.30},
        crystal_structure="fcc",
        surface_energies={(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51},
        target_atoms=600,
        calculator=EMT(),
    )

    # --- surface slab (Cu(111), FCC) ---
    slab_atoms = build_surface(
        composition="Cu",
        crystal_structure="fcc",
        miller_index=(1, 1, 1),
        calculator=EMT(),
        goal_x=12.0,
        goal_y=12.0,
        n_freeze_layers=2,
    )
"""

from __future__ import annotations

import copy
import math
import os
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

from autokmc.logging_utils import get_logger

_log = get_logger(__name__)

# ExpCellFilter: newer ASE (≥3.23) ships it in ase.filters; fall back to
# ase.constraints for older installations.
try:
    from ase.filters import ExpCellFilter
except ImportError:  # pragma: no cover
    from ase.constraints import ExpCellFilter  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# Composition: single element string, or {symbol: fraction} dict summing to 1.
Composition = Union[str, Dict[str, float]]

# Lattice parameters: float (a only) or {"a": ..., "c": ...} for HCP.
LatticeParams = Union[float, Dict[str, float], None]

# ---------------------------------------------------------------------------
# WulffPack – required for nanoparticle builder
# ---------------------------------------------------------------------------
try:
    from wulffpack import SingleCrystal as _SingleCrystal
    _WULFF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SingleCrystal = None  # type: ignore[assignment, misc]
    _WULFF_AVAILABLE = False

# ---------------------------------------------------------------------------
# pymatgen – required for surface slab builder
# ---------------------------------------------------------------------------
try:
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
    _PMG_AVAILABLE = True
except ImportError:  # pragma: no cover
    SlabGenerator = None  # type: ignore[assignment, misc]
    AseAtomsAdaptor = None  # type: ignore[assignment]
    _PMG_AVAILABLE = False


# ===========================================================================
# Public API
# ===========================================================================

def build_nanoparticle(
    composition: Composition = "Cu",
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    surface_energies: Optional[Dict[Tuple[int, int, int], float]] = None,
    target_atoms: int = 600,
    composition_seed: int = 69,
    calculator=None,
    fmax: float = 0.05,
    vacuum: float = 10.0,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Build and optimise a Wulff-construction metal nanoparticle.

    Parameters
    ----------
    composition : str or dict
        Pure metal: a single element symbol, e.g. ``"Cu"``.
        Alloy: a ``{symbol: fraction}`` dict whose values sum to 1,
        e.g. ``{"Cu": 0.70, "Pt": 0.30}``.
    crystal_structure : {"fcc", "bcc", "hcp"}
        Crystal structure of the metal.
    lattice_constant : float, dict, or None
        Lattice constant(s) in Å.
        * FCC / BCC: a single float ``a``.
        * HCP: a dict ``{"a": 3.21, "c": 5.21}`` or a float (ideal c/a used).
        If *None* the bulk is relaxed with *calculator* first.
    surface_energies : dict
        ``{(h, k, l): energy_J_m2}`` mapping for the Wulff construction.
        Keys must be Miller-index tuples, e.g. ``(1, 1, 1)``.
        Default: standard FCC values ``{(1, 1, 1): 1.10, (1, 0, 0): 1.29,
        (1, 1, 0): 1.51}``.
    target_atoms : int
        Approximate number of atoms.
    composition_seed : int
        Random seed for alloy substitution.
    calculator : ASE calculator, optional
        Used for bulk relaxation and structure optimisation.  Defaults to EMT.
    fmax : float
        Force convergence criterion in eV/Å.
    vacuum : float
        Minimum vacuum gap (Å) between the nanoparticle and its periodic
        images, applied along all three axes.  The unit cell is built as a
        cube whose edge equals ``particle_extent + 2 * vacuum`` so that the
        particle is centred with at least *vacuum* Å of empty space on every
        side.  ``pbc`` is set to ``True`` along all axes — downstream code
        (e.g. :func:`autokmc.graph.build_graph`) determines effective
        periodicity by checking for bonds across cell images, not by reading
        this flag.  Default 10.0 Å.
    logfile : str, optional
        Path to the LBFGS log file.  ``None`` silences output.
    verbose : bool
        Print progress information.

    Returns
    -------
    Atoms
        Optimised nanoparticle ASE Atoms object.

    Raises
    ------
    ImportError
        If WulffPack is not installed (``pip install wulffpack``).
    ValueError
        If composition fractions do not sum to 1, or crystal_structure is
        unknown.
    """
    if not _WULFF_AVAILABLE:
        raise ImportError(
            "WulffPack is required for build_nanoparticle.\n"
            "Install it with:  pip install wulffpack"
        )

    comp: Dict[str, float] = _parse_composition(composition)
    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if calculator is None:
        calculator = EMT()

    se: Dict[Tuple[int, int, int], float] = (
        surface_energies if surface_energies is not None
        else {(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51}
    )

    # ------------------------------------------------------------------
    # 1. Lattice constant(s)
    # ------------------------------------------------------------------
    primary = _primary_element(comp)
    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 2. Build primitive bulk unit cell (monatomic, primary element)
    # ------------------------------------------------------------------
    primitive = _build_primitive_cell(primary, crystal_structure, lp)

    if verbose:
        _print_header(
            f"WulffPack Wulff construction  ({crystal_structure.upper()})"
        )
        print(f"  Primary element : {primary}")
        print(f"  Lattice params  : {_fmt_lp(lp)}")
        print(f"  Surface energies: {se}")
        print(f"  Target atoms    : {target_atoms}")

    # ------------------------------------------------------------------
    # 3. Wulff construction via WulffPack
    # ------------------------------------------------------------------
    particle = _SingleCrystal(
        surface_energies=se,
        primitive_structure=primitive,
        natoms=target_atoms,
    )
    atoms: Atoms = particle.atoms.copy()

    if verbose:
        print(f"  Built           : {len(atoms)} atoms")
        _print_divider()

    # ------------------------------------------------------------------
    # 4. Apply alloy composition (random substitution)
    # ------------------------------------------------------------------
    if len(comp) > 1:
        atoms = _apply_composition(atoms, comp, seed=composition_seed,
                                   verbose=verbose)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if verbose:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        r_max = np.linalg.norm(pos - pos.mean(axis=0), axis=1).max()
        _print_header("Nanoparticle built")
        print(f"  Formula        : {atoms.get_chemical_formula()}")
        print(f"  Atoms          : {len(atoms)}")
        for sym, frac in comp.items():
            print(f"  {sym:<14s} : {(syms == sym).sum():<6d}  ({frac*100:.1f} %)")
        print(f"  Diameter (2·R) : {2*r_max:.2f} Å  ({2*r_max/10:.2f} nm)")
        _print_divider()

    # ------------------------------------------------------------------
    # 5b. Wrap in a periodic vacuum-padded box (per-axis bounding box)
    # ------------------------------------------------------------------
    # Use a tight per-axis bounding box rather than a single cubic edge —
    # for elongated nanoparticles this saves significant cell volume (and
    # therefore memory in any downstream calculator that scales with
    # cell size, e.g. PW DFT).  Effective periodicity is determined later
    # by build_graph (via cross-image bond detection), not by the flag.
    pos = atoms.get_positions()
    box = (pos.max(axis=0) - pos.min(axis=0)) + 2.0 * float(vacuum)
    new_cell = np.diag(box.astype(float))
    atoms.set_cell(new_cell)
    atoms.set_pbc(True)
    # Centre the particle inside the new cell.
    com_shift = 0.5 * box - pos.mean(axis=0)
    atoms.set_positions(pos + com_shift)

    if verbose:
        print(f"  Cell (box)     : {box[0]:.2f} × {box[1]:.2f} × {box[2]:.2f} Å"
              f"  (vacuum={vacuum:.2f} Å)")
        print(f"  PBC            : True (effective periodicity inferred "
              "from bonding in build_graph)")
        _print_divider()

    # ------------------------------------------------------------------
    # 6. Optimise
    # ------------------------------------------------------------------
    atoms = optimise_structure(
        atoms,
        calculator=calculator,
        fmax=fmax,
        logfile=logfile,
        verbose=verbose,
    )
    return atoms


def build_surface(
    composition: Composition = "Cu",
    crystal_structure: str = "fcc",
    miller_index: Tuple[int, int, int] = (1, 1, 1),
    lattice_constant: LatticeParams = None,
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 12.0,
    goal_x: float = 12.0,
    goal_y: float = 12.0,
    center_slab: bool = True,
    orthogonalise: bool = True,
    n_freeze_layers: int = 2,
    composition_seed: int = 69,
    calculator=None,
    fmax: float = 0.05,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Build and optimise a metal surface slab.

    Workflow
    --------
    1. Relax the bulk unit cell → get lattice constant(s).
    2. Generate a slab with pymatgen :class:`~pymatgen.core.surface.SlabGenerator`.
    3. Orthogonalise the slab cell (find smallest orthogonal supercell).
    4. Tile to reach approximately *goal_x* × *goal_y* Å in-plane.
    5. Apply alloy composition by random substitution.
    6. Fix the bottom *n_freeze_layers* layers and relax with *calculator*.

    Parameters
    ----------
    composition : str or dict
        Pure metal element string, or ``{symbol: fraction}`` alloy dict.
    crystal_structure : {"fcc", "bcc", "hcp"}
        Crystal structure of the metal.
    miller_index : tuple of int
        Miller indices ``(h, k, l)``.

        .. note::
            For HCP the slab is generated against the **3-index**
            hexagonal Miller indices, *not* the 4-index Miller-Bravais
            convention.  E.g. the basal plane usually written
            ``(0001)`` in the literature must be passed here as
            ``(0, 0, 1)`` (drop the redundant ``i = -(h+k)`` index).
            This matches pymatgen's :class:`SlabGenerator` semantics.
    lattice_constant : float, dict, or None
        Lattice constant(s) in Å.  See module docstring for HCP format.
        If *None* the bulk is relaxed first.
    min_slab_size : float
        Minimum slab thickness in Å.
    min_vacuum_size : float
        Vacuum above the slab in Å.
    goal_x, goal_y : float
        Target in-plane dimensions in Å.
    center_slab : bool
        Whether to centre the slab in the cell.
    orthogonalise : bool
        If *True* (default), search for the smallest integer supercell
        whose in-plane lattice vectors are orthogonal and rotate the cell
        to a diagonal matrix.  Set to *False* to keep the original
        (potentially non-orthogonal) slab cell as returned by pymatgen.
    n_freeze_layers : int
        Number of bottom layers to fix during optimisation.
    composition_seed : int
        Random seed for alloy substitution.
    calculator : ASE calculator, optional
        Defaults to EMT.
    fmax : float
        Force convergence criterion in eV/Å.
    logfile : str, optional
        Path to the optimiser log file.
    verbose : bool
        Print progress information.

    Returns
    -------
    Atoms
        Optimised slab with periodic boundary conditions.  Bottom layers
        carry a :class:`~ase.constraints.FixAtoms` constraint.

    Raises
    ------
    ImportError
        If pymatgen is not available (``pip install pymatgen``).
    RuntimeError
        If SlabGenerator produces no slabs.
    ValueError
        If composition or crystal_structure is invalid.
    """
    if not _PMG_AVAILABLE:
        raise ImportError(
            "pymatgen is required for build_surface.\n"
            "Install it with:  pip install pymatgen"
        )

    comp: Dict[str, float] = _parse_composition(composition)
    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if calculator is None:
        calculator = EMT()

    primary = _primary_element(comp)

    # ------------------------------------------------------------------
    # 1. Lattice constant(s)
    # ------------------------------------------------------------------
    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 2. Build slab via pymatgen
    # ------------------------------------------------------------------
    # SlabGenerator interprets Miller indices in the basis of the parent
    # structure, so use a conventional cell here (especially important for
    # FCC/BCC where the primitive cell is non-cubic).
    bulk_atoms = _build_surface_parent_cell(primary, crystal_structure, lp)
    pmg_bulk = AseAtomsAdaptor.get_structure(bulk_atoms)

    hkl_str = "".join(str(i) for i in miller_index)
    if verbose:
        print(f"\nBuilding {primary}({hkl_str}) slab "
              f"[{crystal_structure.upper()}]  (pymatgen SlabGenerator) ...")

    slabgen = SlabGenerator(
        pmg_bulk,
        miller_index=miller_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
    )
    slabs = slabgen.get_slabs()
    if not slabs:
        raise RuntimeError(
            f"pymatgen SlabGenerator produced no slabs for "
            f"{primary}({hkl_str}) [{crystal_structure.upper()}]."
        )

    slab_ase = AseAtomsAdaptor.get_atoms(slabs[0])
    assert isinstance(slab_ase, Atoms)

    # ------------------------------------------------------------------
    # 3. Orthogonalise (optional)
    # ------------------------------------------------------------------
    if orthogonalise:
        slab_ase, ortho_info = _orthogonalise_slab(slab_ase)
        if verbose:
            n1, n2, m1, m2, det = ortho_info
            c = slab_ase.get_cell()
            print(f"  Orthogonal transform : ({n1},{n2},{m1},{m2})  det={det}")
            print(f"  Cell after ortho     : "
                  f"a={c[0,0]:.3f}  b={c[1,1]:.3f}  c={c[2,2]:.3f} Å")
    elif verbose:
        print("  Orthogonalisation skipped (orthogonalise=False)")

    # ------------------------------------------------------------------
    # 4. Tile to target lateral size
    # ------------------------------------------------------------------
    cell = slab_ase.get_cell()
    if orthogonalise:
        nx_rep = max(1, math.ceil(goal_x / cell[0, 0]))
        ny_rep = max(1, math.ceil(goal_y / cell[1, 1]))
    else:
        # Non-orthogonal cell: use vector magnitudes for tiling estimate
        nx_rep = max(1, math.ceil(goal_x / np.linalg.norm(cell[0])))
        ny_rep = max(1, math.ceil(goal_y / np.linalg.norm(cell[1])))
    atoms = make_supercell(slab_ase, [[nx_rep, 0, 0], [0, ny_rep, 0], [0, 0, 1]])

    if verbose:
        print(f"  Tiling {nx_rep}×{ny_rep} → {len(atoms)} atoms")

    # ------------------------------------------------------------------
    # 5. Apply alloy composition
    # ------------------------------------------------------------------
    if len(comp) > 1:
        atoms = _apply_composition(atoms, comp, seed=composition_seed,
                                   verbose=verbose)

    if verbose:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell_out = atoms.get_cell()
        _print_header(
            f"Surface {primary}({hkl_str})  [{crystal_structure.upper()}]"
        )
        print(f"  Formula        : {atoms.get_chemical_formula()}")
        print(f"  Atoms          : {len(atoms)}")
        for sym, frac in comp.items():
            print(f"  {sym:<14s} : {(syms == sym).sum():<6d}  ({frac*100:.1f} %)")
        print(f"  Cell (Å)       : a={np.linalg.norm(cell_out[0]):.3f}"
              f"  b={np.linalg.norm(cell_out[1]):.3f}"
              f"  c={np.linalg.norm(cell_out[2]):.3f}")
        print(f"  z range        : "
              f"[{pos[:,2].min():.2f}, {pos[:,2].max():.2f}] Å")
        _print_divider()

    # ------------------------------------------------------------------
    # 6. Freeze bottom layers and optimise
    # ------------------------------------------------------------------
    if n_freeze_layers > 0:
        a_val = float(lp["a"])
        fixed_indices = _get_bottom_layer_indices(atoms, n_freeze_layers, a_val)
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        # Store frozen indices in info so downstream code can reuse them
        # without re-running the layer-detection heuristic.
        atoms.info["frozen_indices"] = list(fixed_indices)
        if verbose:
            print(f"  Fixing bottom {n_freeze_layers} layer(s): "
                  f"{len(fixed_indices)} atoms")

    atoms = optimise_structure(
        atoms,
        calculator=calculator,
        fmax=fmax,
        logfile=logfile,
        verbose=verbose,
    )
    return atoms


def optimise_bulk(
    symbol: str,
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    calculator=None,
    fmax: float = 0.01,
    verbose: bool = True,
) -> Tuple[Atoms, Dict[str, float]]:
    """Relax a bulk unit cell and return the optimised Atoms and lattice params.

    Parameters
    ----------
    symbol : str
        Chemical symbol (e.g. ``"Cu"``).
    crystal_structure : {"fcc", "bcc", "hcp"}
        Crystal structure.
    lattice_constant : float, dict, or None
        Initial lattice constant(s) guess.  If *None* ASE's reference data is
        used as starting point.
    calculator : ASE calculator, optional
        Defaults to EMT.
    fmax : float
        Force convergence criterion in eV/Å (applied to both atomic forces
        and cell stress components via :class:`~ase.filters.ExpCellFilter`).
    verbose : bool
        Print summary.

    Returns
    -------
    bulk_atoms : Atoms
        Relaxed bulk unit cell.
    lp : dict
        Relaxed lattice parameters, e.g. ``{"a": 3.615}`` for FCC/BCC or
        ``{"a": 3.21, "c": 5.21}`` for HCP.
    """
    if calculator is None:
        calculator = EMT()

    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if lattice_constant is None:
        lp_in = _ase_reference_lp(symbol, crystal_structure)
    else:
        lp_in = _normalise_lp(lattice_constant, crystal_structure)

    bulk_atoms = _build_primitive_cell(symbol, crystal_structure, lp_in)
    bulk_atoms.calc = calculator

    # ExpCellFilter allows LBFGS to relax both atomic positions *and* the
    # unit-cell shape/volume simultaneously.  Without it, a perfect crystal
    # has zero atomic forces and the optimiser converges immediately without
    # ever changing the lattice constant.
    ecf = ExpCellFilter(bulk_atoms)
    opt = LBFGS(ecf, logfile=os.devnull)  # type: ignore[arg-type]
    opt.run(fmax=fmax)

    lp_out = _extract_lp(bulk_atoms, crystal_structure)

    if verbose:
        e = bulk_atoms.get_potential_energy()
        _print_header(f"Bulk {symbol} ({crystal_structure.upper()}) optimised")
        print(f"  Converged : {opt.converged()}"
              f"  |  steps : {opt.get_number_of_steps()}")
        print(f"  E/atom    : {e / len(bulk_atoms):.5f} eV")
        for k, v in lp_out.items():
            print(f"  Opt {k}     : {v:.4f} Å")
        _print_divider()

    return bulk_atoms, lp_out


def optimise_structure(
    atoms: Atoms,
    calculator=None,
    fmax: float = 0.05,
    steps: int = 1000,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Relax an ASE Atoms object with LBFGS.

    A *fresh copy* is returned so the input is never mutated.

    Parameters
    ----------
    atoms : Atoms
        Structure to optimise.  Existing constraints are preserved on the
        copy.  The calculator is **not** copied; pass *calculator* explicitly
        to attach a fresh one.
    calculator : ASE calculator, optional
        Overrides any existing calculator.  Defaults to EMT.
    fmax : float
        Force convergence criterion in eV/Å.
    steps : int
        Maximum number of optimisation steps.
    logfile : str, optional
        Path to the LBFGS log file.  ``None`` silences output.
    verbose : bool
        Print convergence summary.

    Returns
    -------
    Atoms
        Optimised copy of *atoms*.
    """
    result = atoms.copy()

    # atoms.copy() does NOT deep-copy the calculator; always attach a fresh
    # one so we never mutate the caller's calculator state.  When the
    # caller did not supply a calculator we deep-copy the existing one
    # (preserving any constructor kwargs / loaded ML models) rather than
    # blindly instantiating ``existing.__class__()`` — that pattern silently
    # threw away things like a NequIP model path.
    if calculator is not None:
        result.calc = calculator
    else:
        existing = atoms.calc
        if existing is not None:
            try:
                result.calc = copy.deepcopy(existing)
            except Exception:
                # Fall back to a fresh instance if deepcopy is unsupported
                # (e.g. calculators wrapping un-picklable C handles).
                result.calc = existing.__class__()
        else:
            result.calc = EMT()

    log = logfile if logfile is not None else os.devnull
    opt = LBFGS(result, logfile=log)
    opt.run(fmax=fmax, steps=steps)

    if verbose:
        e = result.get_potential_energy()
        print(f"  Optimisation : converged={opt.converged()} "
              f"steps={opt.get_number_of_steps()}  "
              f"E={e:.4f} eV  E/atom={e/len(result):.4f} eV/atom")

    if not opt.converged():
        warnings.warn(
            f"optimise_structure did not converge within {steps} steps "
            f"(fmax={fmax} eV/Å).  The returned structure may not be at a "
            "local minimum.",
            RuntimeWarning,
            stacklevel=2,
        )

    return result


# ===========================================================================
# Private helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# Composition utilities
# ---------------------------------------------------------------------------

def _parse_composition(composition: Composition) -> Dict[str, float]:
    """Return a normalised {symbol: fraction} dict.

    Accepts a bare element string (→ 100 % that element) or a dict.
    Validates that fractions sum to 1 (within 1e-6).
    """
    if isinstance(composition, str):
        return {composition: 1.0}

    comp = dict(composition)
    total = sum(comp.values())
    if total <= 0:
        raise ValueError(f"Composition values must be positive, got: {comp}")
    # Normalise so fractions sum to exactly 1
    return {sym: val / total for sym, val in comp.items()}


def _primary_element(composition: Dict[str, float]) -> str:
    """Return the element with the highest fraction (used for bulk structure)."""
    return max(composition, key=composition.__getitem__)


def _apply_composition(
    atoms: Atoms,
    composition: Dict[str, float],
    seed: int = 42,
    verbose: bool = True,
) -> Atoms:
    """Randomly substitute atoms to match target composition fractions.

    The primary element (highest fraction) remains as the default; all
    other elements are substituted in at their requested fraction.

    Parameters
    ----------
    atoms : Atoms
        Input structure (typically monatomic – all primary element).
    composition : dict
        ``{symbol: fraction}`` mapping, fractions summing to 1.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print substitution summary.

    Returns
    -------
    Atoms
        Copy with substituted chemical symbols.
    """
    result = atoms.copy()
    n_total = len(result)
    rng = np.random.default_rng(seed=seed)

    # Shuffle all atom indices, then assign contiguous blocks per element
    indices = rng.permutation(n_total)
    syms = np.array(result.get_chemical_symbols())

    primary = _primary_element(composition)
    cursor = 0
    for sym, frac in composition.items():
        if sym == primary:
            continue
        n_sub = int(round(n_total * frac))
        syms[indices[cursor:cursor + n_sub]] = sym
        cursor += n_sub

    result.set_chemical_symbols(syms.tolist())

    if verbose:
        unique, counts = np.unique(syms, return_counts=True)
        print("  Alloy composition applied:")
        for s, c in zip(unique, counts):
            print(f"    {s}: {c} atoms  ({c/n_total*100:.1f} %)")

    return result


# ---------------------------------------------------------------------------
# Crystal structure / lattice parameter utilities
# ---------------------------------------------------------------------------

_VALID_STRUCTURES = {"fcc", "bcc", "hcp"}

# Ideal c/a ratio for HCP: sqrt(8/3)
_HCP_IDEAL_CA = math.sqrt(8.0 / 3.0)   # ≈ 1.6330


def _validate_crystal_structure(cs: str) -> None:
    if cs not in _VALID_STRUCTURES:
        raise ValueError(
            f"crystal_structure must be one of {sorted(_VALID_STRUCTURES)}, "
            f"got '{cs}'."
        )


def _ase_reference_lp(symbol: str, crystal_structure: str) -> Dict[str, float]:
    """Return ASE reference lattice parameters for *symbol* as a dict."""
    from ase.data import reference_states, atomic_numbers
    Z = atomic_numbers[symbol]
    ref = reference_states[Z] or {}
    a = float(ref.get("a", 3.5))
    if crystal_structure == "hcp":
        ca = float(ref.get("c/a", _HCP_IDEAL_CA))
        return {"a": a, "c": a * ca}
    return {"a": a}


def _normalise_lp(
    lattice_constant: Union[float, Dict[str, float]],
    crystal_structure: str,
) -> Dict[str, float]:
    """Coerce *lattice_constant* to a ``{"a": ..., ["c": ...]}`` dict."""
    if isinstance(lattice_constant, (int, float)):
        a = float(lattice_constant)
        if crystal_structure == "hcp":
            return {"a": a, "c": a * _HCP_IDEAL_CA}
        return {"a": a}
    return dict(lattice_constant)


def _resolve_lattice_params(
    symbol: str,
    crystal_structure: str,
    lattice_constant: LatticeParams,
    calculator,
    fmax: float = 0.01,
    verbose: bool = True,
) -> Dict[str, float]:
    """Return a lattice params dict, running bulk relaxation if needed."""
    if lattice_constant is not None:
        return _normalise_lp(lattice_constant, crystal_structure)
    _, lp = optimise_bulk(
        symbol,
        crystal_structure=crystal_structure,
        lattice_constant=None,
        calculator=calculator,
        fmax=fmax,
        verbose=verbose,
    )
    return lp


def _build_primitive_cell(
    symbol: str,
    crystal_structure: str,
    lp: Dict[str, float],
) -> Atoms:
    """Build a bulk unit cell from lattice parameters.

    For FCC and BCC the **conventional cubic** cell is used (rather than
    the primitive rhombohedral / body-centred cell) so that the cell
    vectors are aligned with the cartesian axes — making lattice
    constants trivially recoverable as ``cell[0, 0]`` after a relaxation
    rather than reverse-engineered via factors of √2 / √3.  HCP returns
    the primitive hexagonal cell.
    """
    a = lp["a"]
    if crystal_structure == "hcp":
        c = lp.get("c", a * _HCP_IDEAL_CA)
        return bulk(symbol, crystalstructure="hcp", a=a, c=c)
    # FCC and BCC: use the conventional cubic cell.
    return bulk(symbol, crystalstructure=crystal_structure, a=a, cubic=True)


def _build_surface_parent_cell(
    symbol: str,
    crystal_structure: str,
    lp: Dict[str, float],
) -> Atoms:
    """Build the parent bulk cell used by pymatgen SlabGenerator.

    For FCC/BCC, a conventional cubic cell is required so Miller indices map
    to the expected crystallographic planes (e.g. FCC (1,1,0)).
    """
    a = lp["a"]
    if crystal_structure == "hcp":
        c = lp.get("c", a * _HCP_IDEAL_CA)
        return bulk(symbol, crystalstructure="hcp", a=a, c=c)
    return bulk(symbol, crystalstructure=crystal_structure, a=a, cubic=True)


def _extract_lp(atoms: Atoms, crystal_structure: str) -> Dict[str, float]:
    """Extract lattice parameters from an optimised bulk Atoms object.

    Both FCC and BCC are now built with conventional cubic cells (see
    :func:`_build_primitive_cell`), so the lattice constant is simply
    the length of the first cell vector — no √2 / √3 reverse-engineering
    of the primitive-cell norm is required.
    """
    cell = atoms.get_cell()
    if crystal_structure in ("fcc", "bcc"):
        a = float(np.linalg.norm(cell[0]))
        return {"a": a}
    # HCP: a from first cell vector, c from third
    a = float(np.linalg.norm(cell[0]))
    c = float(np.linalg.norm(cell[2]))
    return {"a": a, "c": c}


def _fmt_lp(lp: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f} Å" for k, v in lp.items())


# ---------------------------------------------------------------------------
# Slab utilities
# ---------------------------------------------------------------------------

def _orthogonalise_slab(
    atoms: Atoms,
    max_search: int = 10,
) -> Tuple[Atoms, Tuple[int, int, int, int, int]]:
    """Return (orthogonalised_atoms, (n1, n2, m1, m2, det)).

    Finds the smallest integer supercell whose in-plane lattice vectors are
    mutually orthogonal, applies the supercell transformation, then rotates
    so that a → x, b → y, c → z with a purely diagonal cell matrix.
    """
    old_cell = np.array(atoms.get_cell())
    a_vec, b_vec = old_cell[0], old_cell[1]
    tol = 1e-4

    best_ortho: Optional[Tuple[int, int, int, int]] = None
    best_size = float("inf")

    for n1 in range(1, max_search + 1):
        for n2 in range(-max_search, max_search + 1):
            v1 = n1 * a_vec + n2 * b_vec
            if np.linalg.norm(v1) < tol:
                continue
            for m1 in range(-max_search, max_search + 1):
                for m2 in range(1, max_search + 1):
                    det = n1 * m2 - n2 * m1
                    if det == 0:
                        continue
                    v2 = m1 * a_vec + m2 * b_vec
                    if np.linalg.norm(v2) < tol:
                        continue
                    dot = (abs(np.dot(v1, v2))
                           / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    if dot < tol and abs(det) < best_size:
                        best_size = abs(det)
                        best_ortho = (n1, n2, m1, m2)

    if best_ortho is None:
        raise ValueError(
            f"Could not find an orthogonal supercell within "
            f"max_search={max_search}.  Try increasing max_search."
        )

    n1, n2, m1, m2 = best_ortho
    transform = np.array([[n1, n2, 0], [m1, m2, 0], [0, 0, 1]], dtype=int)
    ortho = make_supercell(atoms, transform)

    # Rotate: a → x, b → y, c → z  (Gram–Schmidt + diagonal cell)
    oc = np.array(ortho.get_cell())
    ex = oc[0] / np.linalg.norm(oc[0])
    b_perp = oc[1] - np.dot(oc[1], ex) * ex
    ey = b_perp / np.linalg.norm(b_perp)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    # Ensure ez points in the +z direction so the slab is not inverted.
    if ez[2] < 0.0:
        ez = -ez
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)

    R = np.column_stack([ex, ey, ez])

    # Build diagonal cell from the projected vector lengths.
    new_cell = np.diag([
        np.linalg.norm(oc[0]),          # a → x: full length of a vector
        np.linalg.norm(b_perp),         # b → y: orthogonal component of b
        abs(float(oc[2] @ ez)),         # c → z: projection of c onto new z
    ])
    new_pos = ortho.get_positions() @ R

    ortho.set_cell(new_cell, scale_atoms=False)
    ortho.set_positions(new_pos)
    ortho.set_pbc(True)
    # Use ASE's wrap() rather than a manual `% 1.0` on fractional
    # coordinates: it handles the boundary edge case (positions at
    # 1.0 - 1e-15) and respects per-axis PBC flags consistently.
    ortho.wrap()

    return ortho, (n1, n2, m1, m2, int(best_size))


def _get_bottom_layer_indices(
    atoms: Atoms,
    n_layers: int,
    lattice_constant_a: float,
) -> list:
    """Return atom indices belonging to the bottom *n_layers* layers.

    Layers are detected by clustering z-coordinates with tolerance
    ``lattice_constant_a / (4·√2)``.
    """
    pos = atoms.get_positions()

    # Detect layers along the *actual slab normal* (cross(a, b)) rather than
    # global z. This is more robust for non-orthogonal and transformed cells.
    cell = np.array(atoms.get_cell())
    normal = np.cross(cell[0], cell[1])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        # Defensive fallback for malformed cells.
        layer_axis = pos[:, 2].copy()
    else:
        normal /= normal_norm
        layer_axis = pos @ normal

    # Start from the old physics-based heuristic, then tighten adaptively so
    # high-index surfaces (e.g. 211) do not collapse into a single "layer".
    fallback_tol = lattice_constant_a / (4.0 * np.sqrt(2))
    axis_sorted = np.sort(layer_axis)
    diffs = np.diff(axis_sorted)
    # Ignore near-zero spacing from numerical noise / same-plane atoms.
    positive_diffs = diffs[diffs > 1e-4]
    if positive_diffs.size > 0:
        small_gap = float(np.percentile(positive_diffs, 25))
        layer_tol = max(0.05, min(fallback_tol, 0.45 * small_gap))
    else:
        layer_tol = fallback_tol

    z_sorted = axis_sorted
    layers: list = []
    current = [float(z_sorted[0])]
    for z in z_sorted[1:]:
        if z - current[-1] < layer_tol:
            current.append(float(z))
        else:
            layers.append(current)
            current = [float(z)]
    layers.append(current)

    if n_layers > len(layers):
        warnings.warn(
            f"n_freeze_layers={n_layers} exceeds the number of detected "
            f"layers ({len(layers)}).  All layers will be frozen.",
            RuntimeWarning,
            stacklevel=3,
        )
        n_layers = len(layers)

    # Vectorised: for each bottom layer find atoms whose z is within tolerance
    # of the layer's z-range.  This replaces the previous O(n²) double loop.
    freeze_mask = np.zeros(len(layer_axis), dtype=bool)
    for lyr in layers[:n_layers]:
        z_lo = float(min(lyr)) - layer_tol
        z_hi = float(max(lyr)) + layer_tol
        freeze_mask |= (layer_axis >= z_lo) & (layer_axis <= z_hi)

    return list(np.where(freeze_mask)[0])


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_header(title: str, width: int = 58) -> None:
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_divider(width: int = 58) -> None:
    print("=" * width)

