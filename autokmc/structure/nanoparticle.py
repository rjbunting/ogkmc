"""Nanoparticle construction via WulffPack."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import surface as ase_surface
from autokmc.core.constants import RANDOM_SEED
from autokmc.core.pbc import set_full_pbc_if_cell
from autokmc.io.calculators import CalculatorConfigError, acquire_calculator
from autokmc.structure.builders import (
    _apply_composition,
    _build_primitive_cell,
    _fmt_lp,
    _parse_composition,
    _primary_element,
    _print_divider,
    _print_header,
    _validate_crystal_structure,
)
from autokmc.structure.optimization import _resolve_lattice_params, optimise_structure
from autokmc.structure.types import Composition, LatticeParams

try:
    from wulffpack import SingleCrystal as _SingleCrystal
    _WULFF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SingleCrystal = None  # type: ignore[assignment, misc]
    _WULFF_AVAILABLE = False


def _normalise_miller_index(value) -> tuple[int, int, int]:
    if isinstance(value, str):
        stripped = value.strip().strip("()[]")
        parts = stripped.replace(",", " ").split()
        if len(parts) == 1 and len(parts[0]) == 3 and parts[0].isdigit():
            parts = list(parts[0])
        value = parts
    if len(value) != 3:
        raise ValueError(f"Miller index must have three integers, got {value!r}")
    return tuple(int(i) for i in value)


def normalise_surface_energies(
    surface_energies: Mapping | None,
) -> dict[tuple[int, int, int], float] | None:
    """Coerce YAML/TOML-friendly surface-energy maps to tuple-keyed Wulff maps."""
    if surface_energies is None:
        return None
    out: dict[tuple[int, int, int], float] = {}
    for key, value in dict(surface_energies).items():
        out[_normalise_miller_index(key)] = float(value)
    return out


def calculate_surface_energies(
    composition: Composition = "Cu",
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    *,
    facets: Iterable[tuple[int, int, int]] = ((1, 1, 1), (1, 0, 0), (1, 1, 0)),
    layers: int = 6,
    vacuum: float = 10.0,
    calculator=None,
    fmax: float = 0.05,
    max_steps: int = 300,
    verbose: bool = True,
) -> dict[tuple[int, int, int], float]:
    """Calculate relaxed slab surface energies for Wulff construction.

    The returned values are in eV/Angstrom^2.  WulffPack only needs relative
    facet energies, but absolute values are useful for auditability.
    """
    comp: Dict[str, float] = _parse_composition(composition)
    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)
    primary = _primary_element(comp)
    if calculator is None:
        raise CalculatorConfigError(
            "calculate_surface_energies requires an explicit calculator"
        )

    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )
    bulk_atoms = _build_primitive_cell(primary, crystal_structure, lp)
    with acquire_calculator(calculator, purpose="bulk surface-energy relaxation") as calc:
        bulk_relaxed = optimise_structure(
            bulk_atoms,
            calculator=calc,
            fmax=fmax,
            steps=max_steps,
            logfile=os.devnull,
            verbose=False,
        )
        e_bulk_per_atom = float(bulk_relaxed.get_potential_energy()) / len(bulk_relaxed)
        bulk_relaxed.calc = None

    out: dict[tuple[int, int, int], float] = {}
    for facet in facets:
        hkl = _normalise_miller_index(facet)
        slab = ase_surface(bulk_atoms, hkl, layers=int(layers), vacuum=float(vacuum))
        set_full_pbc_if_cell(slab)
        with acquire_calculator(calculator, purpose="slab surface-energy relaxation") as calc:
            slab_relaxed = optimise_structure(
                slab,
                calculator=calc,
                fmax=fmax,
                steps=max_steps,
                logfile=os.devnull,
                verbose=False,
            )
            e_slab = float(slab_relaxed.get_potential_energy())
            slab_relaxed.calc = None
        area = float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])))
        if area <= 0.0:
            raise ValueError(f"facet {hkl} produced a zero-area slab cell")
        gamma = (e_slab - len(slab_relaxed) * e_bulk_per_atom) / (2.0 * area)
        out[hkl] = float(gamma)

    if verbose:
        print("  Calculated surface energies (eV/Å²):")
        for hkl, gamma in out.items():
            print(f"    {hkl}: {gamma:.6f}")
    return out


def build_nanoparticle(
    composition: Composition = "Cu",
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    surface_energies: Optional[Dict[Tuple[int, int, int], float]] = None,
    target_atoms: int = 600,
    surface_energy_facets: Iterable[tuple[int, int, int]] = ((1, 1, 1), (1, 0, 0), (1, 1, 0)),
    surface_energy_layers: int = 6,
    surface_energy_vacuum: float = 10.0,
    surface_energy_fmax: float | None = None,
    surface_energy_max_steps: int | None = None,
    composition_seed: int = RANDOM_SEED,
    calculator=None,
    fmax: float = 0.05,
    max_steps: int = 1000,
    vacuum: float = 10.0,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Build and optimise a Wulff-construction metal nanoparticle."""
    if not _WULFF_AVAILABLE:
        raise ImportError(
            "WulffPack is required for build_nanoparticle.\n"
            "Install it with:  pip install wulffpack"
        )

    comp: Dict[str, float] = _parse_composition(composition)
    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if calculator is None:
        raise CalculatorConfigError(
            "build_nanoparticle requires an explicit calculator"
        )

    primary = _primary_element(comp)
    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )
    primitive = _build_primitive_cell(primary, crystal_structure, lp)
    se = normalise_surface_energies(surface_energies)
    if se is None:
        if verbose:
            print("  Surface energies: calculating relaxed slabs for Wulff construction")
        se = calculate_surface_energies(
            composition          = composition,
            crystal_structure    = crystal_structure,
            lattice_constant     = lp,
            facets               = surface_energy_facets,
            layers               = surface_energy_layers,
            vacuum               = surface_energy_vacuum,
            calculator           = calculator,
            fmax                 = fmax if surface_energy_fmax is None else surface_energy_fmax,
            max_steps            = max_steps if surface_energy_max_steps is None else surface_energy_max_steps,
            verbose              = verbose,
        )

    if verbose:
        _print_header(f"WulffPack Wulff construction  ({crystal_structure.upper()})")
        print(f"  Primary element : {primary}")
        print(f"  Lattice params  : {_fmt_lp(lp)}")
        print(f"  Surface energies: {se}")
        print(f"  Target atoms    : {target_atoms}")

    particle = _SingleCrystal(
        surface_energies=se,
        primitive_structure=primitive,
        natoms=target_atoms,
    )
    atoms: Atoms = particle.atoms.copy()

    if verbose:
        print(f"  Built           : {len(atoms)} atoms")
        _print_divider()

    if len(comp) > 1:
        atoms = _apply_composition(atoms, comp, seed=composition_seed, verbose=verbose)

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

    pos = atoms.get_positions()
    box = (pos.max(axis=0) - pos.min(axis=0)) + 2.0 * float(vacuum)
    new_cell = np.diag(box.astype(float))
    atoms.set_cell(new_cell)
    set_full_pbc_if_cell(atoms)
    com_shift = 0.5 * box - pos.mean(axis=0)
    atoms.set_positions(pos + com_shift)

    if verbose:
        print(
            f"  Cell (box)     : {box[0]:.2f} × {box[1]:.2f} × {box[2]:.2f} Å"
            f"  (vacuum={vacuum:.2f} Å)"
        )
        print("  PBC            : True (effective periodicity inferred from bonding in build_graph)")
        _print_divider()

    with acquire_calculator(calculator, purpose="nanoparticle relaxation") as calc:
        atoms = optimise_structure(
            atoms,
            calculator=calc,
            fmax=fmax,
            steps=max_steps,
            logfile=logfile,
            verbose=verbose,
        )
        atoms.calc = None
    return atoms


__all__ = [
    "build_nanoparticle",
    "calculate_surface_energies",
    "normalise_surface_energies",
]
