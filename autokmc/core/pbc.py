"""Periodic-boundary geometry helpers."""

from __future__ import annotations

from itertools import product

import numpy as np
from ase.geometry import find_mic


def slab_outward_normal(graph, position) -> np.ndarray:
    """Return the exposed face's normal for a slab aligned with Cartesian z.

    Keep whole-slab bounds in graph metadata so local ego graphs use the same
    face as the original structure. Hand-built graphs infer those bounds.
    """
    side = graph.graph.get("surface_side")
    if side == "bottom":
        sign = -1.0
    elif side == "top":
        sign = 1.0
    else:
        bounds = graph.graph.get("slab_z_bounds")
        if bounds is None:
            heights = [
                float(data["position"][2]) for _, data in graph.nodes(data=True)
                if data.get("type") in {"bulk", "surface"} and "position" in data
            ]
            bounds = (min(heights), max(heights)) if heights else (0.0, 0.0)
        sign = -1.0 if float(position[2]) < 0.5 * (bounds[0] + bounds[1]) else 1.0
    return np.array([0.0, 0.0, sign])


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


def periodic_image_offsets(cell, pbc, cutoff: float) -> np.ndarray:
    """Return a complete integer-image search box for wrapped points.

    The fixed ``{-1, 0, 1}`` image box is not complete for a skew,
    non-reduced lattice: a short Cartesian vector can require an integer
    coefficient whose magnitude exceeds one.  If both query and candidate
    positions are wrapped into the primary cell, their raw fractional
    difference is smaller than one along every periodic axis.  The reciprocal
    basis then bounds every image coefficient that can yield a Cartesian
    vector no longer than *cutoff*.

    The returned box is deliberately conservative by one boundary image.
    Callers must still apply their exact Cartesian/MIC distance criterion.
    """
    radius = float(cutoff)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("cutoff must be finite and non-negative")

    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    if cell_arr.shape != (3, 3) or pbc_arr.shape != (3,):
        raise ValueError("cell must be 3x3 and pbc must contain three axes")
    if not pbc_arr.any():
        return np.zeros((1, 3), dtype=int)

    try:
        cell_inv = np.linalg.inv(cell_arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("periodic image generation requires a full-rank cell") from exc

    reciprocal_norms = np.linalg.norm(cell_inv, axis=0)
    ranges = []
    for axis in range(3):
        if not pbc_arr[axis]:
            ranges.append(range(0, 1))
            continue
        bound = max(1, int(np.ceil(1.0 + radius * reciprocal_norms[axis])))
        ranges.append(range(-bound, bound + 1))
    return np.asarray(list(product(*ranges)), dtype=int)


def unwrap_positions_about_reference(
    positions,
    cell,
    pbc,
    *,
    reference=None,
) -> np.ndarray:
    """Map a compact group of positions into one image near *reference*.

    When *reference* is omitted, the first position is used.  This is intended
    for molecules, reacting fragments, and other local groups whose physical
    extent is smaller than the applicable minimum-image range.  It must not be
    used to unwrap an extended periodic structure.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return pos.copy()
    if pos.shape[-1] != 3:
        raise ValueError(
            "positions must have Cartesian coordinates along the final axis"
        )

    if reference is None:
        ref = pos.reshape((-1, 3))[0]
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (3,):
            raise ValueError("reference must be one Cartesian position")

    return ref + minimum_image_vectors(pos - ref, cell, pbc)


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
    "slab_outward_normal",
    "full_pbc_for_cell",
    "graph_pbc_for_atoms",
    "has_real_cell",
    "minimum_image_distances",
    "minimum_image_vectors",
    "periodic_image_offsets",
    "set_full_pbc_if_cell",
    "unwrap_positions_about_reference",
    "wrap_positions_into_cell",
]
