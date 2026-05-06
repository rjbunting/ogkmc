"""Surface-slab construction and slab-specific helpers."""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from autokmc.core.constants import RANDOM_SEED
from autokmc.io.calculators import acquire_calculator
from autokmc.structure.builders import (
    _apply_composition,
    _build_surface_parent_cell,
    _parse_composition,
    _primary_element,
    _print_divider,
    _print_header,
    _validate_crystal_structure,
)
from autokmc.structure.optimization import _resolve_lattice_params, optimise_structure
from autokmc.structure.types import Composition, LatticeParams

try:
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
    _PMG_AVAILABLE = True
except ImportError:  # pragma: no cover
    SlabGenerator = None  # type: ignore[assignment, misc]
    AseAtomsAdaptor = None  # type: ignore[assignment]
    _PMG_AVAILABLE = False


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
    composition_seed: int = RANDOM_SEED,
    calculator=None,
    fmax: float = 0.05,
    max_steps: int = 1000,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Build and optimise a metal surface slab."""
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
    lp = _resolve_lattice_params(
        primary, crystal_structure, lattice_constant, calculator,
        fmax=fmax, verbose=verbose,
    )

    bulk_atoms = _build_surface_parent_cell(primary, crystal_structure, lp)
    pmg_bulk = AseAtomsAdaptor.get_structure(bulk_atoms)

    hkl_str = "".join(str(i) for i in miller_index)
    if verbose:
        print(f"\nBuilding {primary}({hkl_str}) slab [{crystal_structure.upper()}]  (pymatgen SlabGenerator) ...")

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

    if orthogonalise:
        slab_ase, ortho_info = _orthogonalise_slab(slab_ase)
        if verbose:
            n1, n2, m1, m2, det = ortho_info
            c = slab_ase.get_cell()
            print(f"  Orthogonal transform : ({n1},{n2},{m1},{m2})  det={det}")
            print(f"  Cell after ortho     : a={c[0,0]:.3f}  b={c[1,1]:.3f}  c={c[2,2]:.3f} Å")
    elif verbose:
        print("  Orthogonalisation skipped (orthogonalise=False)")

    cell = slab_ase.get_cell()
    if orthogonalise:
        nx_rep = max(1, math.ceil(goal_x / cell[0, 0]))
        ny_rep = max(1, math.ceil(goal_y / cell[1, 1]))
    else:
        nx_rep = max(1, math.ceil(goal_x / np.linalg.norm(cell[0])))
        ny_rep = max(1, math.ceil(goal_y / np.linalg.norm(cell[1])))
    atoms = make_supercell(slab_ase, [[nx_rep, 0, 0], [0, ny_rep, 0], [0, 0, 1]])

    if verbose:
        print(f"  Tiling {nx_rep}×{ny_rep} → {len(atoms)} atoms")

    if len(comp) > 1:
        atoms = _apply_composition(atoms, comp, seed=composition_seed, verbose=verbose)

    if verbose:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell_out = atoms.get_cell()
        _print_header(f"Surface {primary}({hkl_str})  [{crystal_structure.upper()}]")
        print(f"  Formula        : {atoms.get_chemical_formula()}")
        print(f"  Atoms          : {len(atoms)}")
        for sym, frac in comp.items():
            print(f"  {sym:<14s} : {(syms == sym).sum():<6d}  ({frac*100:.1f} %)")
        print(
            f"  Cell (Å)       : a={np.linalg.norm(cell_out[0]):.3f}"
            f"  b={np.linalg.norm(cell_out[1]):.3f}"
            f"  c={np.linalg.norm(cell_out[2]):.3f}"
        )
        print(f"  z range        : [{pos[:,2].min():.2f}, {pos[:,2].max():.2f}] Å")
        _print_divider()

    if n_freeze_layers > 0:
        fixed_indices = _get_bottom_layer_indices(atoms, n_freeze_layers)
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        atoms.info["frozen_indices"] = list(fixed_indices)
        if verbose:
            print(f"  Fixing bottom {n_freeze_layers} layer(s): {len(fixed_indices)} atoms")

    with acquire_calculator(calculator, purpose="surface relaxation") as calc:
        result = optimise_structure(
            atoms,
            calculator=calc,
            fmax=fmax,
            steps=max_steps,
            logfile=logfile,
            verbose=verbose,
        )
        result.calc = None
        return result


def _orthogonalise_slab(
    atoms: Atoms,
    max_search: int = 10,
) -> Tuple[Atoms, Tuple[int, int, int, int, int]]:
    """Return ``(orthogonalised_atoms, (n1, n2, m1, m2, det))``."""
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
                    dot = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if dot < tol and abs(det) < best_size:
                        best_size = abs(det)
                        best_ortho = (n1, n2, m1, m2)

    if best_ortho is None:
        raise ValueError(
            f"Could not find an orthogonal supercell within max_search={max_search}. "
            "Try increasing max_search."
        )

    n1, n2, m1, m2 = best_ortho
    transform = np.array([[n1, n2, 0], [m1, m2, 0], [0, 0, 1]], dtype=int)
    ortho = make_supercell(atoms, transform)

    oc = np.array(ortho.get_cell())
    ex = oc[0] / np.linalg.norm(oc[0])
    b_perp = oc[1] - np.dot(oc[1], ex) * ex
    ey = b_perp / np.linalg.norm(b_perp)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    if ez[2] < 0.0:
        ez = -ez
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)

    R = np.column_stack([ex, ey, ez])
    new_cell = np.diag([
        np.linalg.norm(oc[0]),
        np.linalg.norm(b_perp),
        abs(float(oc[2] @ ez)),
    ])
    new_pos = ortho.get_positions() @ R

    ortho.set_cell(new_cell, scale_atoms=False)
    ortho.set_positions(new_pos)
    ortho.set_pbc(True)
    ortho.wrap()

    return ortho, (n1, n2, m1, m2, int(best_size))


def _get_bottom_layer_indices(atoms: Atoms, n_layers: int) -> list:
    """Return atom indices belonging to the bottom *n_layers* layers."""
    from autokmc.structure.surface import find_surface_atoms_raycasting

    if n_layers <= 0:
        return []

    remaining_idx = np.arange(len(atoms), dtype=int)
    work = atoms.copy()
    work.calc = None
    work.set_constraint()

    frozen: list[int] = []
    for layer in range(n_layers):
        if len(work) == 0:
            warnings.warn(
                f"_get_bottom_layer_indices: ran out of atoms after {layer} layer(s); requested {n_layers}.",
                RuntimeWarning,
                stacklevel=3,
            )
            break

        mask, local_indices = find_surface_atoms_raycasting(work, which="bottom")
        if not local_indices.size:
            warnings.warn(
                f"_get_bottom_layer_indices: ray-casting found no bottom surface atoms at layer {layer + 1}/{n_layers}. Stopping.",
                RuntimeWarning,
                stacklevel=3,
            )
            break

        frozen.extend(int(i) for i in remaining_idx[local_indices])
        keep = np.ones(len(work), dtype=bool)
        keep[local_indices] = False
        remaining_idx = remaining_idx[keep]
        work = work[keep]

    return sorted(set(frozen))


__all__ = ["build_surface", "_orthogonalise_slab", "_get_bottom_layer_indices"]
