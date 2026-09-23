"""Utilities for identifying surface atoms in two geometries.

* :func:`find_surface_atoms_raycasting`
    Top-surface slab classification via atom-adaptive PBC-safe ray-casting.

* :func:`find_surface_atoms_convexhull`
    Nanoparticle surface classification via convex-hull signed distance.

* :func:`find_surface_atoms`
    Unified entry point: auto-detects geometry via :func:`has_pbc_connectivity`
    and dispatches to the appropriate method.  Returns a
    :class:`~autokmc.core.results.SurfaceClassification` (which is also iterable
    in the same shape as the legacy variable-length tuple, for backwards
    compatibility).

* :func:`tag_surface_atoms`
    Writes atom-type codes into ``atoms.arrays["surface"]`` (int8) so they
    are preserved as a column in ``.extxyz`` output.
    Encoding: 0 = bulk, 1 = surface, 2 = adsorbate.
"""

from __future__ import annotations

import logging
import numpy as np

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from scipy.spatial import ConvexHull

from autokmc.core.results import SurfaceClassification
from autokmc.core.constants import (
    NL_MULT_DEFAULT,
    RAYCAST_COVERAGE_THRESHOLD,
    RAYCAST_N_DISC_SAMPLE,
)
from autokmc.core.pbc import minimum_image_vectors, set_full_pbc_if_cell

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tag_surface_atoms(atoms: Atoms, surface_mask: np.ndarray) -> None:
    """Store per-atom type codes in ``atoms.arrays["surface"]``.

    Encoding (int8):

    * ``0`` – bulk
    * ``1`` – surface
    * ``2`` – adsorbate
    """
    atoms.arrays["surface"] = surface_mask.astype(np.int8)


def _pbc_connectivity_axes(
    atoms: Atoms,
    nl_mult: float = NL_MULT_DEFAULT,
) -> np.ndarray:
    """Return axes where at least one bonded pair crosses a periodic boundary.

    Builds an ASE :class:`~ase.neighborlist.NeighborList` and inspects the
    integer cell-image offsets for every neighbour.  A non-zero offset
    along a *periodic* axis means the bond crosses a periodic boundary
    image.

    Parameters
    ----------
    atoms : Atoms
    nl_mult : float
        Multiplier for :func:`~ase.neighborlist.natural_cutoffs`.  Defaults
        to :data:`autokmc.core.constants.NL_MULT_DEFAULT`.

    Structures with a real cell are normalised to full PBC before the
    neighbour-list is built.  The returned axes describe bonding connectivity,
    not the structure's stored PBC metadata.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs

    set_full_pbc_if_cell(atoms)
    pbc_axes = np.asarray(atoms.get_pbc(), dtype=bool)
    if not pbc_axes.any():
        return np.zeros(3, dtype=bool)

    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)

    connected = np.zeros(3, dtype=bool)
    for i in range(len(atoms)):
        _, offsets = nl.get_neighbors(i)
        if not len(offsets):
            continue
        # Only count cross-image bonds along axes marked periodic.
        masked = np.asarray(offsets) * pbc_axes[np.newaxis, :]
        connected |= masked.any(axis=0)
    return connected


def has_pbc_connectivity(atoms: Atoms,
                         nl_mult: float = NL_MULT_DEFAULT) -> bool:
    """Return ``True`` if at least one bonded pair crosses a periodic boundary.

    ``True`` indicates a periodic slab and dispatches to ray-casting surface
    classification. ``False`` indicates an isolated nanoparticle and
    dispatches to convex-hull classification.
    """
    return bool(_pbc_connectivity_axes(atoms, nl_mult=nl_mult).any())


def _surface_frame(
    cell: np.ndarray,
    pbc_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal in-plane vectors and an oriented slab normal."""
    cell_arr = np.asarray(cell, dtype=float)
    connected = list(np.flatnonzero(np.asarray(pbc_axes, dtype=bool)))
    if len(connected) >= 2:
        first, second = connected[:2]
    else:
        # Preserve the historical a/b surface convention for malformed,
        # one-dimensional, or directly-called ray-casting inputs.
        first, second = 0, 1

    # Copy before normalising: ``np.asarray(cell[first])`` is a view and an
    # in-place division would silently rescale the caller's lattice vector.
    tangent_a = np.array(cell_arr[first], dtype=float, copy=True)
    tangent_a_norm = float(np.linalg.norm(tangent_a))
    normal = np.cross(tangent_a, np.asarray(cell_arr[second], dtype=float))
    normal_norm = float(np.linalg.norm(normal))
    if tangent_a_norm <= 1.0e-12 or normal_norm <= 1.0e-12:
        raise ValueError("surface cell vectors must span a non-zero plane")
    tangent_a /= tangent_a_norm
    normal /= normal_norm

    remaining = [axis for axis in range(3) if axis not in (first, second)]
    if remaining:
        if float(np.dot(normal, cell_arr[remaining[0]])) < 0.0:
            normal = -normal
    elif normal[2] < 0.0:
        normal = -normal
    tangent_b = np.cross(normal, tangent_a)
    tangent_b /= float(np.linalg.norm(tangent_b))
    return tangent_a, tangent_b, normal


def align_periodic_slab_frame(
    atoms: Atoms,
    pbc_axes: np.ndarray,
    *,
    atol: float = 1.0e-10,
) -> dict[str, object]:
    """Rigidly align a detected slab normal with Cartesian +z in place.

    Unlike slab construction, this does not rebuild or orthogonalise the cell;
    it applies the same rigid Cartesian rotation to the supplied lattice and
    atom positions.  The transformation therefore keeps file-backed catalyst
    geometry and cell metrics unchanged while satisfying downstream
    adsorption and gas-endpoint +z conventions.
    """
    cell = np.asarray(atoms.get_cell(), dtype=float)
    tangent_a, tangent_b, normal = _surface_frame(cell, pbc_axes)
    rotated = not np.allclose(normal, [0.0, 0.0, 1.0], rtol=0.0, atol=atol)
    if rotated:
        rotation = np.column_stack((tangent_a, tangent_b, normal))
        atoms.set_positions(
            np.asarray(atoms.get_positions()) @ rotation,
            apply_constraint=False,
        )
        atoms.set_cell(cell @ rotation, scale_atoms=False)

    metadata: dict[str, object] = {
        "aligned_to_z": True,
        "rotation_applied": bool(rotated),
        "periodic_connectivity_axes": [
            int(axis) for axis in np.flatnonzero(np.asarray(pbc_axes, dtype=bool))
        ],
        "original_surface_normal": [float(value) for value in normal],
    }
    atoms.info["_autokmc_surface_frame"] = metadata
    return metadata


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def find_surface_atoms(
    atoms: Atoms,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
    # ray-casting kwargs
    surf_radius_factor: float = 1.0,
    which: str = "top",
    coverage_threshold: float = RAYCAST_COVERAGE_THRESHOLD,
    n_disc_sample: int = RAYCAST_N_DISC_SAMPLE,
    # convex-hull kwargs
    hull_tol_factor: float = 0.5,
    return_diagnostics: bool = False,
    # tagging
    tag_atoms: bool = False,
) -> SurfaceClassification:
    """Auto-detect geometry and classify surface atoms.

    Decision rule
    -------------
    1. Build a neighbour list and check PBC image offsets of every bond.
    2. Any cross-boundary bond → periodic slab → :func:`find_surface_atoms_raycasting`.
    3. No cross-boundary bonds → nanoparticle  → :func:`find_surface_atoms_convexhull`.

    Parameters
    ----------
    atoms : Atoms
    nl_mult : float
        Neighbour-list multiplier for the PBC-connectivity test.
    surf_radius_factor, which, coverage_threshold, n_disc_sample :
        Forwarded to :func:`find_surface_atoms_raycasting`.
    hull_tol_factor, return_diagnostics :
        Forwarded to :func:`find_surface_atoms_convexhull`.
    tag_atoms : bool
        If ``True``, set ``atoms.arrays["surface"]`` in-place before returning.

    Returns
    -------
    SurfaceClassification
        Dataclass exposing ``.mask``, ``.indices``, ``.method``, and
        (nanoparticle path only) ``.hull`` and optionally ``.diagnostics``.
        The dataclass is iterable in the same order as the legacy
        variable-length tuple, so ``mask, indices, method = ...`` and
        ``mask, indices, hull, method = ...`` continue to work.
    """
    pbc_axes = _pbc_connectivity_axes(atoms, nl_mult=nl_mult)

    if pbc_axes.any():
        surface_mask, surface_indices = find_surface_atoms_raycasting(
            atoms,
            surf_radius_factor=surf_radius_factor,
            which=which,
            coverage_threshold=coverage_threshold,
            n_disc_sample=n_disc_sample,
            pbc_axes=pbc_axes,
        )
        if tag_atoms:
            tag_surface_atoms(atoms, surface_mask)
        result = SurfaceClassification(
            mask=surface_mask, indices=surface_indices, method="raycasting",
        )
        _log.debug("find_surface_atoms: raycasting → %d/%d surface atoms",
                   int(surface_mask.sum()), len(atoms))
        return result

    # A nanoparticle uses convex-hull surface classification.
    raw = find_surface_atoms_convexhull(
        atoms,
        hull_tol_factor=hull_tol_factor,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        surface_mask, surface_indices, hull, diagnostics = raw
    else:
        surface_mask, surface_indices, hull = raw
        diagnostics = None
    if tag_atoms:
        tag_surface_atoms(atoms, surface_mask)
    # Stash the hull facet equations on atoms.info so that downstream
    # consumers (default_sites.find_sites_for_element, build_graph cache)
    # do not have to rebuild the same hull.  We store only the equations
    # array because the ConvexHull object itself doesn't survive
    # atoms.copy() / serialisation cleanly.
    atoms.info["_hull_equations"] = np.asarray(hull.equations, dtype=float)
    result = SurfaceClassification(
        mask=surface_mask, indices=surface_indices, method="convexhull",
        hull=hull, diagnostics=diagnostics,
    )
    _log.debug("find_surface_atoms: convexhull → %d/%d surface atoms",
               int(surface_mask.sum()), len(atoms))
    return result


# ---------------------------------------------------------------------------
# Method 1: Atom-adaptive PBC-safe ray-casting  (slabs)
# ---------------------------------------------------------------------------

def find_surface_atoms_raycasting(
    atoms: Atoms,
    *,
    surf_radius_factor: float = 1.0,
    which: str = "top",
    coverage_threshold: float = RAYCAST_COVERAGE_THRESHOLD,
    n_disc_sample: int = RAYCAST_N_DISC_SAMPLE,
    pbc_axes: np.ndarray | None = None,
):
    """Classify surface atoms by per-atom disc-coverage ray-casting.

    See module docstring for the algorithm.  Returns
    ``(mask, indices)``.
    """
    if which not in ("top", "bottom", "both"):
        raise ValueError(f"which must be 'top', 'bottom', or 'both', got {which!r}")

    set_full_pbc_if_cell(atoms)
    pos      = atoms.get_positions()          # (N, 3)
    cell     = np.array(atoms.get_cell())     # (3, 3)
    if pbc_axes is None:
        pbc_axes = _pbc_connectivity_axes(atoms)
    pbc_axes = np.asarray(pbc_axes, dtype=bool)
    tangent_a, tangent_b, normal = _surface_frame(cell, pbc_axes)
    N        = len(atoms)

    # Per-atom capture radius from covalent radii
    atomic_numbers = atoms.get_atomic_numbers()
    surf_radii = ASE_COVALENT_RADII[atomic_numbers] * surf_radius_factor  # (N,)

    # Build a unit-disc sample pattern (normalised to [-1, 1]²)
    g = np.linspace(-1.0, 1.0, n_disc_sample)
    gx, gy = np.meshgrid(g, g)
    unit_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)          # (n², 2)
    unit_pts = unit_pts[(unit_pts ** 2).sum(axis=1) <= 1.0]        # (K, 2) in disc
    K = len(unit_pts)

    offsets = (
        unit_pts[:, 0, np.newaxis] * tangent_a[np.newaxis, :]
        + unit_pts[:, 1, np.newaxis] * tangent_b[np.newaxis, :]
    )
    rays = (
        pos[:, np.newaxis, :]
        + surf_radii[:, np.newaxis, np.newaxis] * offsets[np.newaxis, :, :]
    ).reshape(-1, 3)
    heights = pos @ normal

    n_rays = len(rays)
    _CHUNK = 4096

    wins_top = np.zeros(N, dtype=np.int32)
    wins_bot = np.zeros(N, dtype=np.int32)

    for start in range(0, n_rays, _CHUNK):
        sl         = slice(start, start + _CHUNK)
        chunk = rays[sl]
        displacement = minimum_image_vectors(
            pos[np.newaxis, :, :] - chunk[:, np.newaxis, :],
            cell,
            pbc_axes,
        )
        normal_component = np.einsum("cni,i->cn", displacement, normal)
        tangent_displacement = (
            displacement - normal_component[:, :, np.newaxis] * normal
        )
        in_plane_distance = np.linalg.norm(tangent_displacement, axis=2)

        within  = in_plane_distance <= surf_radii[np.newaxis, :]   # (C, N)
        has_any = within.any(axis=1)                               # (C,)
        valid   = np.where(has_any)[0]                             # indices into chunk

        if which in ("top", "both") and valid.size:
            z_top    = np.where(within[valid], heights[np.newaxis, :], -np.inf)
            best_top = np.argmax(z_top, axis=1)                    # (valid,)
            np.add.at(wins_top, best_top, 1)

        if which in ("bottom", "both") and valid.size:
            z_bot    = np.where(within[valid], heights[np.newaxis, :], np.inf)
            best_bot = np.argmin(z_bot, axis=1)                    # (valid,)
            np.add.at(wins_bot, best_bot, 1)

    threshold_count = coverage_threshold * K

    if which == "top":
        surface_mask = wins_top >= threshold_count
    elif which == "bottom":
        surface_mask = wins_bot >= threshold_count
    else:  # "both"
        surface_mask = (wins_top >= threshold_count) | (wins_bot >= threshold_count)

    atoms.info["_autokmc_surface_side"] = which
    surface_indices = np.where(surface_mask)[0].astype(int)
    return surface_mask, surface_indices


# ---------------------------------------------------------------------------
# Method 2: Convex-hull signed distance  (nanoparticles)
# ---------------------------------------------------------------------------

def find_surface_atoms_convexhull(
    atoms: Atoms,
    *,
    hull_tol_factor: float = 0.5,
    return_diagnostics: bool = False,
):
    """Classify surface atoms via convex-hull signed distance.

    See module docstring; returns ``(mask, indices, hull[, diagnostics])``.
    """
    pos  = atoms.get_positions()
    N    = len(atoms)
    hull = ConvexHull(pos)

    eq          = hull.equations                                   # (F, 4)
    signed_dist = (eq[:, :3] @ pos.T + eq[:, 3:4]).max(axis=0)   # (N,)

    atomic_numbers = atoms.get_atomic_numbers()
    cov_radii      = ASE_COVALENT_RADII[atomic_numbers]           # (N,)
    hull_tol_base  = cov_radii * hull_tol_factor                   # (N,)

    normals       = eq[:, :3]                                      # (F, 3)
    signed_all    = eq[:, :3] @ pos.T + eq[:, 3:4]                # (F, N)
    nearest_facet = np.argmax(signed_all, axis=0)                  # (N,)

    K_NEAR         = min(5, len(normals) - 1)
    nn_dot         = normals @ normals.T                           # (F, F)
    np.fill_diagonal(nn_dot, -2.0)
    if K_NEAR < 1:
        # Degenerate hull with only one facet — no neighbour normals to
        # average; skip curvature adaptation and use a flat tolerance.
        curvature_norm = np.zeros(len(normals))
    else:
        knn_vals       = np.sort(nn_dot, axis=1)[:, -K_NEAR:]     # (F, K)
        curvature_f    = (1.0 - knn_vals).mean(axis=1)            # (F,)
        c_min, c_max   = curvature_f.min(), curvature_f.max()
        curvature_norm = (
            (curvature_f - c_min) / (c_max - c_min)
            if c_max > c_min else np.zeros_like(curvature_f)
        )
    hull_tol_per_atom = hull_tol_base * (1.0 + curvature_norm[nearest_facet])

    dist_surface_mask = signed_dist > -hull_tol_per_atom           # (N,)

    vertex_mask = np.zeros(N, dtype=bool)
    vertex_mask[hull.vertices] = True

    surface_mask    = dist_surface_mask | vertex_mask
    surface_indices = np.where(surface_mask)[0].astype(int)

    if return_diagnostics:
        diagnostics = {
            "signed_dist"       : signed_dist,
            "hull_tol_per_atom" : hull_tol_per_atom,
            "dist_surface_mask" : dist_surface_mask,
            "vertex_mask"       : vertex_mask,
        }
        return surface_mask, surface_indices, hull, diagnostics

    return surface_mask, surface_indices, hull
