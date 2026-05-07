"""Periodic-boundary geometry helpers."""

from __future__ import annotations

import numpy as np
from ase.geometry import find_mic


def has_real_cell(cell) -> bool:
    """Return True when *cell* is a full-rank 3D lattice cell."""
    cell_arr = np.asarray(cell, dtype=float)
    if cell_arr.shape != (3, 3):
        return False
    return bool(np.linalg.matrix_rank(cell_arr) == 3)


def full_pbc_for_cell(cell) -> np.ndarray:
    """Use full PBC for structures with a real cell, otherwise no PBC."""
    if has_real_cell(cell):
        return np.ones(3, dtype=bool)
    return np.zeros(3, dtype=bool)


def graph_pbc_for_atoms(atoms) -> np.ndarray:
    """Return graph-level PBC for tagged structures.

    Adsorbate-only reactant graphs are gas-phase molecules.  They can carry a
    vacuum cell from ``Atoms.center(vacuum=...)`` for calculator compatibility,
    but they must remain non-periodic for ASE gas thermochemistry.
    """
    surface = atoms.arrays.get("surface")
    if surface is not None:
        surface_arr = np.asarray(surface, dtype=int)
        if surface_arr.size and np.all(surface_arr == 2):
            return np.zeros(3, dtype=bool)
    return full_pbc_for_cell(atoms.get_cell())


def set_full_pbc_if_cell(atoms):
    """Set ``atoms.pbc`` to T T T when ``atoms`` has a real cell."""
    if has_real_cell(atoms.get_cell()):
        atoms.set_pbc(True)
    return atoms


def minimum_image_vectors(vectors, cell, pbc) -> np.ndarray:
    """Return minimum-image Cartesian vectors for any ``(..., 3)`` array.

    Component-wise rounding in fractional coordinates is only guaranteed for
    orthorhombic cells. ASE's ``find_mic`` handles skewed/triclinic cells by
    searching the relevant neighbouring images.
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.size == 0:
        return arr.copy()
    pbc_arr = np.asarray(pbc, dtype=bool)
    if not pbc_arr.any():
        return arr.copy()

    shape = arr.shape
    flat = arr.reshape((-1, 3))
    mic, _lengths = find_mic(flat, np.asarray(cell, dtype=float), pbc=pbc_arr)
    return np.asarray(mic, dtype=float).reshape(shape)


def minimum_image_distances(vectors, cell, pbc) -> np.ndarray:
    """Return minimum-image lengths for any ``(..., 3)`` vector array."""
    mic = minimum_image_vectors(vectors, cell, pbc)
    return np.linalg.norm(mic, axis=-1)


def wrap_positions_into_cell(
    positions,
    cell,
    pbc,
    *,
    reference=None,
) -> np.ndarray:
    """Wrap Cartesian positions into the primary periodic cell.

    If *reference* is supplied, all positions are translated by the same
    lattice vector that wraps the reference point. This preserves molecular
    geometry for adsorbates. Without *reference*, each position is wrapped
    independently.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return pos.copy()
    pbc_arr = np.asarray(pbc, dtype=bool)
    if not pbc_arr.any():
        return pos.copy()

    cell_arr = np.asarray(cell, dtype=float)
    cell_inv = np.linalg.inv(cell_arr)

    if reference is not None:
        ref_frac = np.asarray(reference, dtype=float) @ cell_inv
        shift_frac = np.zeros(3, dtype=float)
        for ax in range(3):
            if pbc_arr[ax]:
                shift_frac[ax] = np.floor(ref_frac[ax])
        return pos - shift_frac @ cell_arr

    frac = pos @ cell_inv
    wrapped = frac.copy()
    for ax in range(3):
        if pbc_arr[ax]:
            wrapped[..., ax] = wrapped[..., ax] - np.floor(wrapped[..., ax])
    return wrapped @ cell_arr


__all__ = [
    "full_pbc_for_cell",
    "graph_pbc_for_atoms",
    "has_real_cell",
    "minimum_image_distances",
    "minimum_image_vectors",
    "set_full_pbc_if_cell",
    "wrap_positions_into_cell",
]
