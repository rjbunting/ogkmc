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
    and dispatches to the appropriate method.

* :func:`tag_surface_atoms`
    Writes atom-type codes into ``atoms.arrays["surface"]`` (int8) so they
    are preserved as a column in ``.extxyz`` output.
    Encoding: 0 = bulk, 1 = surface, 2 = adsorbate.
"""

from __future__ import annotations

import numpy as np

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tag_surface_atoms(atoms: Atoms, surface_mask: np.ndarray) -> None:
    """Store per-atom type codes in ``atoms.arrays["surface"]``.

    Encoding (int8):

    * ``0`` – bulk
    * ``1`` – surface
    * ``2`` – adsorbate

    Written as ``int8`` so it appears as a plain integer column when the
    structure is saved with :func:`ase.io.write` to an ``.extxyz`` file.

    Parameters
    ----------
    atoms:
        The :class:`~ase.Atoms` object to annotate **in-place**.
    surface_mask:
        Boolean array of length ``len(atoms)``.  ``True`` → surface (1),
        ``False`` → bulk (0).  Pass an all-True mask to mark every atom
        as surface, or build the int8 array manually and assign it directly
        to ``atoms.arrays["surface"]`` for adsorbate (2) labelling.
    """
    atoms.arrays["surface"] = surface_mask.astype(np.int8)


def has_pbc_connectivity(atoms: Atoms, nl_mult: float = 1.1) -> bool:
    """Return ``True`` if at least one bonded pair crosses a periodic boundary.

    Builds an ASE :class:`~ase.neighborlist.NeighborList` and inspects the
    integer cell-image offsets for every neighbour.  A non-zero offset means
    the bond crosses a periodic boundary image.

    Parameters
    ----------
    atoms : Atoms
    nl_mult : float
        Multiplier for :func:`~ase.neighborlist.natural_cutoffs`.  Default 1.1.

    Returns
    -------
    bool
        ``True``  → periodic slab   → use :func:`find_surface_atoms_raycasting`.
        ``False`` → nanoparticle    → use :func:`find_surface_atoms_convexhull`.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs

    if not atoms.get_pbc().any():
        return False

    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)

    for i in range(len(atoms)):
        _, offsets = nl.get_neighbors(i)
        if offsets.any():
            return True
    return False


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def find_surface_atoms(
    atoms: Atoms,
    *,
    nl_mult: float = 1.0,
    # ray-casting kwargs
    surf_radius_factor: float = 1.0,
    which: str = "top",
    coverage_threshold: float = 0.7,
    n_disc_sample: int = 10,
    # convex-hull kwargs
    hull_tol_factor: float = 0.5,
    return_diagnostics: bool = False,
    # tagging
    tag_atoms: bool = False,
):
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
    surface_mask : np.ndarray[bool]
    surface_indices : np.ndarray[int]
    extra :
        Convex-hull path: ``(hull,)`` or ``(hull, diagnostics)`` depending on
        ``return_diagnostics``; ray-casting path: nothing extra.
    method : str
        ``"raycasting"`` or ``"convexhull"``.
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
        return surface_mask, surface_indices, "raycasting"

    else:
        result = find_surface_atoms_convexhull(
            atoms,
            hull_tol_factor=hull_tol_factor,
            return_diagnostics=return_diagnostics,
        )
        surface_mask = result[0]
        if tag_atoms:
            tag_surface_atoms(atoms, surface_mask)
        return (*result, "convexhull")


# ---------------------------------------------------------------------------
# Method 1: Atom-adaptive PBC-safe ray-casting  (slabs)
# ---------------------------------------------------------------------------

def find_surface_atoms_raycasting(
    atoms: Atoms,
    *,
    surf_radius_factor: float = 1.0,
    which: str = "top",
    coverage_threshold: float = 0.5,
    n_disc_sample: int = 5,
):
    """Classify surface atoms by per-atom disc-coverage ray-casting.

    For each atom a small grid of rays is sampled uniformly over its capture
    disc (a circle of radius ``surf_radius_factor * r_cov`` centred on its
    projected *xy* position).  The atom is classified as a surface atom if
    it wins at least ``coverage_threshold`` fraction of those rays — i.e. it
    is the highest-*z* atom within capture range of at least that fraction of
    the disc.

    This cleanly handles close-packed surfaces such as FCC(111): a top-layer
    atom wins nearly all its disc rays, while a sub-surface atom only wins
    rays that slip through the tiny inter-atom gaps, giving a coverage close
    to zero.

    Parameters
    ----------
    atoms : Atoms
        Slab with an orthogonal or non-orthogonal periodic cell.
    surf_radius_factor : float
        Per-atom capture radius = ``surf_radius_factor * r_cov(element)``.
        Default 1.0.
    which : {"top", "bottom", "both"}
        Which vacuum face to classify.  Default ``"top"``.
    coverage_threshold : float
        Minimum fraction of disc rays an atom must win to be labelled
        surface.  Default 0.5 (must win at least half its disc).
    n_disc_sample : int
        Side length of the square grid used to sample each atom's disc.
        A ``n_disc_sample × n_disc_sample`` grid is filtered to the
        inscribed circle, giving roughly ``0.78 * n_disc_sample²`` rays
        per atom.  Default 5 (≈ 19 rays/atom).

    Returns
    -------
    surface_mask : np.ndarray[bool], shape (N,)
    surface_indices : np.ndarray[int]
    """
    if which not in ("top", "bottom", "both"):
        raise ValueError(f"which must be 'top', 'bottom', or 'both', got {which!r}")

    pos      = atoms.get_positions()          # (N, 3)
    cell     = np.array(atoms.get_cell())     # (3, 3)
    cell_inv = np.linalg.inv(cell)
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

    # Each atom i contributes K rays: pos[i,:2] + surf_radii[i] * unit_pts
    # Shape: (N, K, 2) → (N*K, 2)
    rays_xy = (pos[:, np.newaxis, :2]
               + surf_radii[:, np.newaxis, np.newaxis] * unit_pts[np.newaxis, :, :])
    rays_xy    = rays_xy.reshape(-1, 2)                            # (N*K, 2)

    # Convert ray origins to fractional coordinates (for PBC wrapping)
    rays_frac  = rays_xy @ cell_inv[:2, :2]                        # (N*K, 2)

    # Fractional xy of all atoms
    atom_frac  = (pos @ cell_inv)[:, :2]                           # (N, 2)
    z_vals     = pos[:, 2]

    n_rays = len(rays_xy)

    # Internal chunk size: process rays in batches to bound peak memory.
    # 4096 rays × N atoms stays well under ~100 MB for typical slab sizes.
    _CHUNK = 4096

    # wins_top[i] / wins_bot[i] = number of disc rays atom i won
    wins_top = np.zeros(N, dtype=np.int32)
    wins_bot = np.zeros(N, dtype=np.int32)

    for start in range(0, n_rays, _CHUNK):
        sl         = slice(start, start + _CHUNK)
        chunk_frac = rays_frac[sl]                                 # (C, 2)

        # PBC-wrapped fractional delta → Cartesian xy distance  (C, N, 2)
        dfrac  = atom_frac[np.newaxis, :, :] - chunk_frac[:, np.newaxis, :]
        dfrac -= np.round(dfrac)
        dxy    = dfrac @ cell[:2, :2]                              # (C, N, 2)
        xy_dist = np.sqrt((dxy ** 2).sum(axis=2))                  # (C, N)

        # Atom j is a candidate for ray c if xy_dist[c,j] <= surf_radii[j]
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

    # Coverage fraction: wins / K  (rays that hit nothing don't count against)
    # We use K as the denominator — an atom that wins > coverage_threshold*K is surface.
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

    An atom is a surface atom if its signed distance to the convex hull
    satisfies ``signed_dist > -hull_tol_per_atom``.  The per-atom tolerance
    is ``hull_tol_factor * r_cov(element)`` further scaled by the local facet
    curvature, so larger atoms naturally receive a wider inclusion band and
    curved facets (corners/edges) are less strict than flat terraces.
    Hull vertices are always included regardless of the distance criterion.

    Parameters
    ----------
    atoms : Atoms
        Nanoparticle (finite, non-periodic structure).
    hull_tol_factor : float
        Per-atom tolerance = ``hull_tol_factor * r_cov(element)``.
        Default 0.5 (~0.64 Å for Cu).  Further scaled by local curvature
        (flat facet -> 1x, most curved -> 2x).
    return_diagnostics : bool
        If ``True``, append a ``diagnostics`` dict to the return value.
        Keys: ``"signed_dist"``, ``"hull_tol_per_atom"``,
        ``"dist_surface_mask"``, ``"vertex_mask"``.

    Returns
    -------
    surface_mask : np.ndarray[bool], shape (N,)
    surface_indices : np.ndarray[int]
    hull : ConvexHull
    diagnostics : dict  (only when ``return_diagnostics=True``)
    """
    pos  = atoms.get_positions()
    N    = len(atoms)
    hull = ConvexHull(pos)

    # Signed distance: max over facets of  n·p + d
    eq          = hull.equations                                   # (F, 4)
    signed_dist = (eq[:, :3] @ pos.T + eq[:, 3:4]).max(axis=0)   # (N,)

    # Per-atom base tolerance from covalent radii
    atomic_numbers = atoms.get_atomic_numbers()
    cov_radii      = ASE_COVALENT_RADII[atomic_numbers]           # (N,)
    hull_tol_base  = cov_radii * hull_tol_factor                   # (N,)

    # Adaptive scaling by local facet curvature
    # Curvature proxy: mean(1 - n_i · n_j) for K angularly-nearest facets.
    # 0 = flat/parallel neighbours → tight tolerance.
    # Up to 2 = perpendicular neighbours → loose tolerance.
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

    # Hull vertices are definitionally on the surface
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

