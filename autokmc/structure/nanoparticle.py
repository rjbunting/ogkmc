"""Nanoparticle construction via WulffPack."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from autokmc.core.constants import RANDOM_SEED
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


def build_nanoparticle(
    composition: Composition = "Cu",
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    surface_energies: Optional[Dict[Tuple[int, int, int], float]] = None,
    target_atoms: int = 600,
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
        calculator = EMT()

    se: Dict[Tuple[int, int, int], float] = (
        surface_energies if surface_energies is not None
        else {(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51}
    )

    primary = _primary_element(comp)
    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )
    primitive = _build_primitive_cell(primary, crystal_structure, lp)

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
    atoms.set_pbc(True)
    com_shift = 0.5 * box - pos.mean(axis=0)
    atoms.set_positions(pos + com_shift)

    if verbose:
        print(
            f"  Cell (box)     : {box[0]:.2f} × {box[1]:.2f} × {box[2]:.2f} Å"
            f"  (vacuum={vacuum:.2f} Å)"
        )
        print("  PBC            : True (effective periodicity inferred from bonding in build_graph)")
        _print_divider()

    atoms = optimise_structure(
        atoms,
        calculator=calculator,
        fmax=fmax,
        steps=max_steps,
        logfile=logfile,
        verbose=verbose,
    )
    return atoms


__all__ = ["build_nanoparticle"]
