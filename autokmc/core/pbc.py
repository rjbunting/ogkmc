"""Periodic-boundary geometry helpers."""

from __future__ import annotations

import numpy as np
from ase.geometry import find_mic


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
    "minimum_image_distances",
    "minimum_image_vectors",
    "wrap_positions_into_cell",
]
