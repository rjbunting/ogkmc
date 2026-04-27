"""
autokmc.surface
===============
Utilities for identifying surface atoms in two geometries:

* :func:`find_surface_atoms_raycasting`
    Top-surface slab classification via atom-adaptive PBC-safe ray-casting.

* :func:`find_surface_atoms_convexhull`
    Nanoparticle surface classification via convex-hull signed distance.

* :func:`find_surface_atoms`
    Unified entry point: auto-detects geometry via :func:`has_pbc_connectivity`
    and dispatches to the appropriate method.  Returns a
    :class:`~autokmc.results.SurfaceClassification` (which is also iterable
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

from autokmc.results import SurfaceClassification
from autokmc.constants import (
    NL_MULT_DEFAULT,
    RAYCAST_COVERAGE_THRESHOLD,
    RAYCAST_N_DISC_SAMPLE,
)

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


def has_pbc_connectivity(atoms: Atoms,
                         nl_mult: float = NL_MULT_DEFAULT) -> bool:
    """Return ``True`` if at least one bonded pair crosses a periodic boundary.

    Builds an ASE :class:`~ase.neighborlist.NeighborList` and inspects the
    integer cell-image offsets for every neighbour.  A non-zero offset
    along a *periodic* axis means the bond crosses a periodic boundary
    image.

    Parameters
    ----------
    atoms : Atoms
    nl_mult : float
        Multiplier for :func:`~ase.neighborlist.natural_cutoffs`.  Defaults
        to :data:`autokmc.constants.NL_MULT_DEFAULT`.

    Returns
    -------
    bool
        ``True``  → periodic slab   → use :func:`find_surface_atoms_raycasting`.
        ``False`` → nanoparticle    → use :func:`find_surface_atoms_convexhull`.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs

    pbc_axes = np.asarray(atoms.get_pbc(), dtype=bool)
    if not pbc_axes.any():
        return False

    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)

    for i in range(len(atoms)):
        _, offsets = nl.get_neighbors(i)
        if not len(offsets):
            continue
        # Only count cross-image bonds along axes the user actually marked
        # periodic.  This makes the test honest for partial-PBC cells.
        masked = np.asarray(offsets) * pbc_axes[np.newaxis, :]
        if masked.any():
            return True
    return False


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
    pbc = has_pbc_connectivity(atoms, nl_mult=nl_mult)

    if pbc:
        surface_mask, surface_indices = find_surface_atoms_raycasting(
            atoms,
            surf_radius_factor=surf_radius_factor,
            which=which,
            coverage_threshold=coverage_threshold,
            n_disc_sample=n_disc_sample,
        )
        if tag_atoms:
            tag_surface_atoms(atoms, surface_mask)
        result = SurfaceClassification(
            mask=surface_mask, indices=surface_indices, method="raycasting",
        )
        _log.debug("find_surface_atoms: raycasting → %d/%d surface atoms",
                   int(surface_mask.sum()), len(atoms))
        return result

    # ── nanoparticle (convex hull) ────────────────────────────────────────
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
):
    """Classify surface atoms by per-atom disc-coverage ray-casting.

    See module docstring for the algorithm.  Returns
    ``(mask, indices)``.
    """
    if which not in ("top", "bottom", "both"):
        raise ValueError(f"which must be 'top', 'bottom', or 'both', got {which!r}")

    pos      = atoms.get_positions()          # (N, 3)
    cell     = np.array(atoms.get_cell())     # (3, 3)
    cell_inv = np.linalg.inv(cell)            # full 3×3 inverse: handles
    #                                          # tilted slab cells correctly.
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

    # Rays placed in the xy plane around each atom.  We embed them in 3-D
    # at z=0 so the full 3×3 inverse can wrap them correctly even when the
    # cell has off-diagonal terms (e.g. a non-orthogonalised slab cell).
    rays_xy3 = np.zeros((N, K, 3), dtype=float)
    rays_xy3[..., 0] = (pos[:, np.newaxis, 0]
                        + surf_radii[:, np.newaxis] * unit_pts[np.newaxis, :, 0])
    rays_xy3[..., 1] = (pos[:, np.newaxis, 1]
                        + surf_radii[:, np.newaxis] * unit_pts[np.newaxis, :, 1])
    rays_xy3 = rays_xy3.reshape(-1, 3)                              # (N*K, 3)

    rays_frac3 = rays_xy3 @ cell_inv                                # (N*K, 3)
    atom_frac3 = pos @ cell_inv                                     # (N, 3)
    z_vals     = pos[:, 2]

    n_rays = len(rays_xy3)
    _CHUNK = 4096

    wins_top = np.zeros(N, dtype=np.int32)
    wins_bot = np.zeros(N, dtype=np.int32)

    for start in range(0, n_rays, _CHUNK):
        sl         = slice(start, start + _CHUNK)
        chunk_frac = rays_frac3[sl]                                 # (C, 3)

        # MIC wrap in *full* 3-D fractional space, then drop the z axis
        # back to Cartesian xy for the in-plane distance check.
        dfrac3 = atom_frac3[np.newaxis, :, :] - chunk_frac[:, np.newaxis, :]
        dfrac3 -= np.round(dfrac3)
        d_cart = dfrac3 @ cell                                      # (C, N, 3)
        dxy    = d_cart[..., :2]                                    # (C, N, 2)
        xy_dist = np.sqrt((dxy ** 2).sum(axis=2))                   # (C, N)

        within  = xy_dist <= surf_radii[np.newaxis, :]             # (C, N)
        has_any = within.any(axis=1)                               # (C,)
        valid   = np.where(has_any)[0]                             # indices into chunk

        if which in ("top", "both") and valid.size:
            z_top    = np.where(within[valid], z_vals[np.newaxis, :], -np.inf)
            best_top = np.argmax(z_top, axis=1)                    # (valid,)
            np.add.at(wins_top, best_top, 1)

        if which in ("bottom", "both") and valid.size:
            z_bot    = np.where(within[valid], z_vals[np.newaxis, :], np.inf)
            best_bot = np.argmin(z_bot, axis=1)                    # (valid,)
            np.add.at(wins_bot, best_bot, 1)

    threshold_count = coverage_threshold * K

    if which == "top":
        surface_mask = wins_top >= threshold_count
    elif which == "bottom":
        surface_mask = wins_bot >= threshold_count
    else:  # "both"
        surface_mask = (wins_top >= threshold_count) | (wins_bot >= threshold_count)

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
    knn_vals       = np.sort(nn_dot, axis=1)[:, -K_NEAR:]         # (F, K)
    curvature_f    = (1.0 - knn_vals).mean(axis=1)                # (F,)
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

