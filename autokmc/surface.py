"""
autokmc.surface
===============
Utilities for identifying surface atoms in two geometries:

* :func:`find_surface_atoms_raycasting`
    Top-surface slab classification via PBC-safe mesh ray-casting.

* :func:`find_surface_atoms_convexhull`
    Nanoparticle surface classification via convex-hull signed distance.

* :func:`find_surface_atoms`
    Unified entry point: auto-detects geometry via :func:`has_pbc_connectivity`
    and dispatches to the appropriate method.

* :func:`tag_surface_atoms`
    Writes the surface flag into ``atoms.arrays["surface"]`` (int8, 0/1)
    so it is preserved as a column in ``.extxyz`` output.
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
    """Store a per-atom surface flag in ``atoms.arrays["surface"]``.

    Written as ``int8`` (0 = bulk, 1 = surface) so it appears as a plain
    integer column when the structure is saved with :func:`ase.io.write`
    to an ``.extxyz`` file.

    Parameters
    ----------
    atoms:
        The :class:`~ase.Atoms` object to annotate **in-place**.
    surface_mask:
        Boolean array of length ``len(atoms)``.
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
    lattice_constant: float,
    *,
    nl_mult: float = 1.1,
    # ray-casting kwargs
    surf_radius_factor: float = 1.0,
    mesh_spacing_factor: float = 0.25,
    which: str = "top",
    chunk_size: int = 4096,
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
    lattice_constant : float
        FCC lattice constant *a* in Å (used for mesh spacing in ray-casting).
    nl_mult : float
        Neighbour-list multiplier for the PBC-connectivity test.
    surf_radius_factor, mesh_spacing_factor, which, chunk_size :
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
            atoms, lattice_constant,
            surf_radius_factor=surf_radius_factor,
            mesh_spacing_factor=mesh_spacing_factor,
            which=which,
            chunk_size=chunk_size,
        )
        if tag_atoms:
            tag_surface_atoms(atoms, surface_mask)
        return surface_mask, surface_indices, "raycasting"

    else:
        result = find_surface_atoms_convexhull(
            atoms, lattice_constant,
            hull_tol_factor=hull_tol_factor,
            return_diagnostics=return_diagnostics,
        )
        surface_mask = result[0]
        if tag_atoms:
            tag_surface_atoms(atoms, surface_mask)
        return (*result, "convexhull")


# ---------------------------------------------------------------------------
# Method 1: PBC-safe mesh ray-casting  (slabs)
# ---------------------------------------------------------------------------

def find_surface_atoms_raycasting(
    atoms: Atoms,
    lattice_constant: float,
    *,
    surf_radius_factor: float = 1.0,
    mesh_spacing_factor: float = 0.25,
    which: str = "top",
    chunk_size: int = 4096,
):
    """Classify surface atoms by PBC-safe mesh ray-casting.

    A fine 2-D grid of vertical rays is cast over the *xy* plane of the
    periodic cell.  For each ray, the atom with the extreme *z* coordinate
    whose PBC-wrapped xy-distance to the ray is within its covalent radius
    (scaled by ``surf_radius_factor``) is recorded as a surface atom.

    Using per-atom covalent radii makes the capture radius physically
    meaningful for multi-element systems (e.g. Cu–Pd alloys): a Pd atom
    gets a wider capture radius than a Cu atom.

    Parameters
    ----------
    atoms : Atoms
        Slab with an orthogonal or non-orthogonal periodic cell.
    lattice_constant : float
        Used **only** for mesh spacing (``a * mesh_spacing_factor``).
        The capture radius comes from covalent radii, not from this value.
    surf_radius_factor : float
        Per-atom xy capture radius = ``surf_radius_factor * r_cov(element)``.
        Default 1.0 (one full covalent radius).
    mesh_spacing_factor : float
        Mesh spacing = ``lattice_constant * mesh_spacing_factor``.
        Default 0.25.
    which : {"top", "bottom", "both"}
        Which vacuum face to classify.  Default ``"top"``.
    chunk_size : int
        Mesh rows per NumPy batch (memory–speed trade-off).  Default 4096.

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

    mesh_spacing = lattice_constant * mesh_spacing_factor

    # Per-atom capture radius from covalent radii
    atomic_numbers = atoms.get_atomic_numbers()
    surf_radii = ASE_COVALENT_RADII[atomic_numbers] * surf_radius_factor  # (N,)

    # 2-D mesh grid spanning the in-plane cell vectors
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    na    = max(1, int(np.ceil(a_len / mesh_spacing)))
    nb    = max(1, int(np.ceil(b_len / mesh_spacing)))
    fa    = np.linspace(0.0, 1.0, na, endpoint=False)
    fb    = np.linspace(0.0, 1.0, nb, endpoint=False)
    gfa, gfb = np.meshgrid(fa, fb, indexing="ij")
    # Cartesian xy of mesh points (ignore z component of cell vectors)
    mesh_xy = (gfa.ravel()[:, np.newaxis] * cell[0, :2]
               + gfb.ravel()[:, np.newaxis] * cell[1, :2])  # (N_mesh, 2)

    # Fractional xy of all atoms for PBC wrapping
    frac_xy = (pos @ cell_inv)[:, :2]   # (N, 2)

    n_mesh = len(mesh_xy)
    z_vals = pos[:, 2]
    surface_atom_set: set[int] = set()

    for start in range(0, n_mesh, chunk_size):
        chunk      = mesh_xy[start : start + chunk_size]          # (C, 2)
        chunk_frac = chunk @ cell_inv[:2, :2]                      # (C, 2)

        # PBC-wrapped fractional delta → Cartesian xy distance
        dfrac   = frac_xy[np.newaxis, :, :] - chunk_frac[:, np.newaxis, :]  # (C, N, 2)
        dfrac  -= np.round(dfrac)
        dxy     = dfrac @ cell[:2, :2]                             # (C, N, 2)
        xy_dist = np.sqrt((dxy ** 2).sum(axis=2))                  # (C, N)

        # Per-atom capture radius broadcast across mesh rows
        within  = xy_dist <= surf_radii[np.newaxis, :]             # (C, N)
        has_any = within.any(axis=1)                               # (C,)

        if which in ("top", "both"):
            z_top    = np.where(within, z_vals[np.newaxis, :], -np.inf)
            best_top = np.argmax(z_top, axis=1)                    # (C,)
            surface_atom_set.update(int(best_top[i]) for i in np.where(has_any)[0])

        if which in ("bottom", "both"):
            z_bot    = np.where(within, z_vals[np.newaxis, :], np.inf)
            best_bot = np.argmin(z_bot, axis=1)                    # (C,)
            surface_atom_set.update(int(best_bot[i]) for i in np.where(has_any)[0])

    surface_indices = np.array(sorted(surface_atom_set), dtype=int)
    surface_mask    = np.zeros(len(atoms), dtype=bool)
    surface_mask[surface_indices] = True
    return surface_mask, surface_indices


# ---------------------------------------------------------------------------
# Method 2: Convex-hull signed distance  (nanoparticles)
# ---------------------------------------------------------------------------

def find_surface_atoms_convexhull(
    atoms: Atoms,
    lattice_constant: float,
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
    lattice_constant : float
        Retained for API compatibility; not used internally.
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

    K_NEAR      = min(5, len(normals) - 1)
    nn_dot      = normals @ normals.T                              # (F, F)
    np.fill_diagonal(nn_dot, -2.0)
    knn_vals    = np.sort(nn_dot, axis=1)[:, -K_NEAR:]            # (F, K)
    curvature_f = (1.0 - knn_vals).mean(axis=1)                   # (F,)
    c_min, c_max = curvature_f.min(), curvature_f.max()
    curvature_norm = (
        (curvature_f - c_min) / (c_max - c_min)
        if c_max > c_min else np.zeros_like(curvature_f)
    )
    # hull_tol_base (N,) * scalar factor → (N,)
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

