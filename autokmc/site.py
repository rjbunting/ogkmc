"""
autokmc.site
============
Find all unique adsorption sites on a surface for a given adsorbate.

Algorithm
---------
**Single-atom adsorbate**

1. Cast a 3-D probe grid above the top surface layer.
2. For each grid point read the surface-atom connectivity directly from
   the grid (no post-hoc recheck).
3. Deduplicate by frozenset of connected surface-atom global indices.
4. Geometrically optimise each unique site.
5. Classify by ego-graph isomorphism.

**Multi-atom adsorbate**

For each atom *a* in the adsorbate molecule (each is tried as the *anchor*):

1. Place *a* at every point on the surface grid.
2. For each grid point sample *n_orientations* random orientations of the
   whole molecule (uniform SO(3) sampling) by rotating the molecule around *a*.
3. For each placement check which surface atoms are within bonding distance
   of **any** adsorbate atom.
4. Deduplicate by a frozenset of ``(adsorbate_local_idx, surface_global_idx)``
   pairs — this captures both which surface atoms bond *and* through which
   adsorbate atom.
5. Geometrically optimise each unique placement as a 6-DOF rigid body
   (3 translation + 3 rotation).
6. Classify by ego-graph isomorphism on the combined adsorbate + surface
   ego-subgraph.

Typical usage
-------------
::

    carbon = build_reactant("[C]", calculator=EMT())
    sites, site_graph = find_adsorption_sites(surface_graph, carbon)

    co = build_reactant("[C-]#[O+]", calculator=EMT())
    sites, site_graph = find_adsorption_sites(surface_graph, co,
                                               n_orientations=300)
"""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass
from typing import Any


import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from autokmc.reactants import Reactant


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROBE_NODE_BASE = -1     # single-atom sentinel; multi-atom uses negative ids
_SITE_LABELS = {1: "top", 2: "bridge", 3: "hollow"}


def _site_label(n: int) -> str:
    return _SITE_LABELS.get(n, f"{n}-fold")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdsorptionSite:
    """A unique adsorption site on the surface.

    Attributes
    ----------
    position : np.ndarray, shape (3,)  or  shape (N_ads, 3)
        For a single-atom adsorbate: Cartesian position of the atom (Å).
        For a multi-atom adsorbate: Cartesian positions of all adsorbate
        atoms in the optimised placement (Å).
    conn_global : frozenset
        * Single-atom: frozenset of surface atom global indices.
        * Multi-atom:  frozenset of ``(ads_local_idx, surf_global_idx)`` pairs.
    n_conn : int
        Total number of adsorbate–surface bonds.
    site_type : str
        ``"top"`` / ``"bridge"`` / ``"hollow"`` / ``"N-fold"`` based on the
        number of *distinct* surface atoms bonded to the adsorbate.
    iso_class : int
        Isomorphism class index.
    energy : float or None
        Adsorption energy (eV) = E(slab+ads) − E(slab).  Set only when
        a *calculator* and *slab* are passed to :func:`find_adsorption_sites`.
        All sites in the same iso-class share the same value (one
        representative is relaxed per class).
    converged : bool or None
        Whether the LBFGS relaxation converged for this iso-class.
    """
    position    : np.ndarray
    conn_global : frozenset
    n_conn      : int
    site_type   : str
    iso_class   : int
    energy      : float | None = None
    converged   : bool  | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pbc_dist_vec(probe: np.ndarray, surf_pos: np.ndarray,
                  cell: np.ndarray) -> np.ndarray:
    """PBC-wrapped distances (wrap x, y only) from *probe* (3,) to *surf_pos* (N,3)."""
    cell_inv = np.linalg.inv(cell)
    dv   = surf_pos - probe
    frac = dv @ cell_inv
    frac[:, :2] -= np.round(frac[:, :2])
    return np.sqrt(((frac @ cell) ** 2).sum(axis=1))


def _sample_so3(n: int, seed: int = 42) -> np.ndarray:
    """Return *n* uniformly distributed 3×3 rotation matrices (SO(3))."""
    return Rotation.random(n, random_state=seed).as_matrix()   # (n, 3, 3)


# ---------------------------------------------------------------------------
# WL-colour anchor deduplication
# ---------------------------------------------------------------------------

def _wl_anchor_indices(reactant: Reactant) -> list[int]:
    """Return one representative atom index per WL-colour equivalence class.

    Atoms that are symmetrically equivalent (same element environment at every
    shell radius) share the same WL colour.  Only one anchor per colour class
    needs to be sampled on the probe grid — all others are guaranteed to
    produce the same set of connectivity patterns by symmetry.

    Examples
    --------
    * O₂  (O=O)        → 1 anchor  (both O atoms are equivalent)
    * CO  ([C-]#[O+])  → 2 anchors (C and O are inequivalent)
    * CH₄ (C)          → 2 anchors (C and H are inequivalent)
    """
    graph = reactant.graph
    if graph.number_of_nodes() == 0:
        return []

    labels: dict = {n: d["element"] for n, d in graph.nodes(data=True)}
    n_iter = max(1, graph.number_of_nodes())
    for _ in range(n_iter):
        new_labels: dict = {}
        for n in graph.nodes():
            nbr = tuple(sorted(labels[u] for u in graph.neighbors(n)))
            new_labels[n] = (labels[n], nbr)
        labels = {n: str(v) for n, v in new_labels.items()}

    seen_colours: dict[str, int] = {}
    reps: list[int] = []
    for n in graph.nodes():
        col = labels[n]
        if col not in seen_colours:
            seen_colours[col] = n
            reps.append(n)
    return reps


# ---------------------------------------------------------------------------
# Convex-hull surface-exposure filter
# ---------------------------------------------------------------------------

def _hull_bondable_indices(ads_pos: np.ndarray) -> list[int]:
    """Return the adsorbate atom indices that can face the surface.

    An atom can only bond to the surface if it lies on the convex hull of
    the molecule — interior atoms (e.g. C in CH₄, the central atom of a
    tetrahedral molecule) are geometrically shielded by their neighbours
    and can never approach the surface directly.

    Since the grid approach tries all SO(3) orientations, **any** hull
    vertex can potentially face downward, so we expose all hull vertices
    rather than only bottom-facing faces.  Interior atoms (not on the hull)
    are excluded entirely.

    Degenerate cases (< 4 atoms, collinear, coplanar)
    --------------------------------------------------
    ``ConvexHull`` requires at least 4 non-coplanar points in 3D.  For
    molecules with fewer atoms, or those whose positions are degenerate,
    all atoms are returned as candidates (safe fallback).

    Parameters
    ----------
    ads_pos : (N, 3) ndarray
        Gas-phase Cartesian positions of the adsorbate atoms (Å).

    Returns
    -------
    bondable : list[int]
        Atom local indices that lie on the convex hull (or all indices for
        degenerate molecules).
    """
    from scipy.spatial import ConvexHull

    N = len(ads_pos)
    if N <= 1:
        return list(range(N))

    try:
        hull = ConvexHull(ads_pos)
        return sorted(set(int(i) for i in hull.vertices))
    except Exception:
        # Degenerate geometry (collinear / coplanar / too few points)
        return list(range(N))


# ---------------------------------------------------------------------------
# Steric clash helper
# ---------------------------------------------------------------------------

def _is_probe_clashing(probe: np.ndarray, exclude_lids: set,
                        surf_pos: np.ndarray, surf_rcov: np.ndarray,
                        r_cov_ads: float, clash_factor: float,
                        cell_inv: np.ndarray, cell: np.ndarray) -> bool:
    """Return True if *probe* is within ``clash_factor*(r_cov_ads+r_cov_s)``
    of any non-bonded surface atom.

    Parameters
    ----------
    probe        : (3,) proposed adsorbate position
    exclude_lids : local indices of the bonded atoms (not clash-checked)
    surf_pos     : (N, 3) all surface atom positions
    surf_rcov    : (N,) all surface atom covalent radii
    r_cov_ads    : adsorbate covalent radius
    clash_factor : fraction of combined covalent radii that counts as a clash
                   (e.g. bond_factor=1.1 means any atom that would actually
                   bond counts as a clash)
    cell_inv     : (3, 3) inverse unit cell matrix
    cell         : (3, 3) unit cell matrix
    """
    for k in range(len(surf_pos)):
        if k in exclude_lids:
            continue
        dv   = surf_pos[k] - probe
        frac = dv @ cell_inv
        frac[:2] -= np.round(frac[:2])
        dist = float(np.linalg.norm(frac @ cell))
        if dist < clash_factor * (r_cov_ads + float(surf_rcov[k])):
            return True
    return False


# ---------------------------------------------------------------------------
# Isomorphism helpers
# ---------------------------------------------------------------------------

def _build_ego_single(surface_graph: nx.Graph,
                      conn_global: frozenset[int],
                      ads_element: str) -> nx.Graph:
    """Ego-graph for a single-atom adsorbate probe."""
    shell1 = list(conn_global)
    shell2: set[int] = set()
    for g in shell1:
        if g in surface_graph:
            shell2.update(surface_graph.neighbors(g))
    sub = surface_graph.subgraph(set(shell1) | shell2).copy()
    sub.add_node(_PROBE_NODE_BASE, element=ads_element)
    for g in shell1:
        sub.add_edge(_PROBE_NODE_BASE, g)
    return sub


def _build_ego_multi(surface_graph: nx.Graph,
                     reactant: Reactant,
                     conn_pairs: frozenset[tuple[int, int]]) -> nx.Graph:
    """Ego-graph for a multi-atom adsorbate.

    Node ids for adsorbate atoms: ``-(local_idx + 1)``  (always negative).
    Surface atom nodes keep their global ids (positive).
    """
    # Negative ids for adsorbate atoms (avoids clash with surface atom ids)
    ads_id = lambda k: -(k + 1)

    surf_1st: set[int] = {sg for _, sg in conn_pairs}
    surf_2nd: set[int] = set()
    for sg in surf_1st:
        if sg in surface_graph:
            surf_2nd.update(surface_graph.neighbors(sg))

    sub = surface_graph.subgraph(surf_1st | surf_2nd).copy()

    # Add adsorbate atoms
    for ak, d in reactant.graph.nodes(data=True):
        sub.add_node(ads_id(ak), element=d["element"])

    # Internal adsorbate bonds
    for u, v in reactant.graph.edges():
        sub.add_edge(ads_id(u), ads_id(v))

    # Adsorbate–surface bonds
    for (ak, sg) in conn_pairs:
        sub.add_edge(ads_id(ak), sg)

    return sub


# ---------------------------------------------------------------------------
# Geometric optimisation
# ---------------------------------------------------------------------------

def _opt_single(probe0: np.ndarray, conn_pos: np.ndarray,
                bond_targets: np.ndarray, nonbond_pos: np.ndarray,
                cell: np.ndarray, z_lo: float, z_hi: float,
                lam: float = 5.0) -> np.ndarray:
    """Optimise a single-atom site (3 DOF: x, y, z)."""
    n_conn = len(conn_pos)

    def obj(p):
        d = _pbc_dist_vec(p, conn_pos, cell)
        loss = float(np.sum((d - bond_targets) ** 2))
        if n_conn < 3 and len(nonbond_pos):
            loss -= lam * float(_pbc_dist_vec(p, nonbond_pos, cell).min())
        return loss

    p0 = probe0.copy()
    p0[2] = np.clip(p0[2], z_lo, z_hi)
    res = minimize(obj, p0, method="L-BFGS-B",
                   bounds=[(None, None), (None, None), (z_lo, z_hi)],
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    return res.x


def _opt_multi(anchor0: np.ndarray, rotvec0: np.ndarray,
               anchor_idx: int, rel_pos: np.ndarray,
               ads_rcov: np.ndarray,
               conn_pairs: frozenset[tuple[int, int]],
               surf_pos: np.ndarray, surf_rcov: np.ndarray,
               surf_idx_to_local: dict[int, int],
               cell: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """Optimise a multi-atom rigid-body placement (6 DOF).

    Returns the optimised positions of all adsorbate atoms (N_ads, 3).
    """
    # Bond targets: r_cov_ads[ak] + r_cov_surf[sk]
    pairs = list(conn_pairs)
    tgt = np.array([ads_rcov[ak] + surf_rcov[surf_idx_to_local[sk]]
                    for ak, sk in pairs])
    surf_conn_pos = np.array([surf_pos[surf_idx_to_local[sk]] for _, sk in pairs])
    pair_ak = [ak for ak, _ in pairs]

    def obj(x):
        anchor = x[:3]
        rot    = Rotation.from_rotvec(x[3:]).as_matrix()
        ads_p  = anchor + (rot @ rel_pos.T).T          # (N_ads, 3)
        # bond-length residuals
        d = np.array([_pbc_dist_vec(ads_p[ak], surf_conn_pos[i:i+1], cell)[0]
                      for i, ak in enumerate(pair_ak)])
        loss = float(np.sum((d - tgt) ** 2))
        # soft penalty: adsorbate atoms must not go below surface
        below = np.maximum(0.0, z_lo - ads_p[:, 2])
        loss += 100.0 * float(np.sum(below ** 2))
        return loss

    x0 = np.zeros(6)
    x0[:3] = anchor0
    x0[3:] = rotvec0
    x0[2] = np.clip(x0[2], z_lo, z_hi)

    res = minimize(obj, x0, method="L-BFGS-B",
                   bounds=[(None, None), (None, None), (z_lo, z_hi),
                           (None, None), (None, None), (None, None)],
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    x_opt = res.x
    anchor_opt = x_opt[:3]
    rot_opt    = Rotation.from_rotvec(x_opt[3:]).as_matrix()
    return anchor_opt + (rot_opt @ rel_pos.T).T        # (N_ads, 3)


# ---------------------------------------------------------------------------
# Isomorphism pre-filter key
# ---------------------------------------------------------------------------

def _iso_prefilter_key(g: nx.Graph) -> tuple:
    """Cheap structural fingerprint used to skip full VF2 matching.

    Two graphs with different keys are definitely non-isomorphic.
    Graphs with the same key still need full VF2 verification.
    """
    elem_counts = tuple(sorted(
        (d["element"], deg)
        for _, d, deg in (
            (n, g.nodes[n], g.degree(n)) for n in g.nodes()
        )
    ))
    deg_seq = tuple(sorted(g.degree(n) for n in g.nodes()))
    return (g.number_of_nodes(), g.number_of_edges(), deg_seq, elem_counts)


# ---------------------------------------------------------------------------
# Adaptive probe grid
# ---------------------------------------------------------------------------

def _adaptive_probe_grid(
    surf_pos: np.ndarray,
    surf_rcov: np.ndarray,
    anchor_rcov_list: list[float],
    bond_factor: float,
    grid_spacing: float,
    surf_z_max: float,
    cell: np.ndarray,
) -> np.ndarray:
    """Generate a probe grid adapted to local covalent radii.

    Instead of a uniform rectangular grid, probe points are placed on
    hemispherical shells centred on each surface atom.  For every
    (surface atom *s*, anchor covalent radius *r_a*) pair two shells are
    sampled:

    * **ideal shell** at radius  ``r_a + r_s``   (the expected bond length)
    * **cutoff shell** at radius ``bond_factor * (r_a + r_s)``

    The angular spacing of each shell is chosen so that the arc-length
    between adjacent sample points is approximately *grid_spacing*.

    Points below ``surf_z_max + min(anchor_rcov) * 0.5`` are discarded.
    All surviving points are PBC-wrapped into the primary cell and
    spatially deduplicated with tolerance ``grid_spacing / 2`` using a
    ``cKDTree``.

    Parameters
    ----------
    surf_pos : (S, 3) ndarray
    surf_rcov : (S,) ndarray
    anchor_rcov_list : list of float
        Covalent radii of adsorbate atoms that may act as the grid anchor.
    bond_factor : float
    grid_spacing : float
        Target arc-length spacing between probe points (Å).
    surf_z_max : float
        Maximum z-coordinate of the top surface layer.
    cell : (3, 3) ndarray

    Returns
    -------
    grid_pts : (G, 3) ndarray
        Deduplicated probe positions PBC-wrapped into the primary cell.
    """
    from scipy.spatial import cKDTree

    cell_inv = np.linalg.inv(cell)
    z_min = surf_z_max + min(anchor_rcov_list) * 0.5

    raw: list[np.ndarray] = []

    for si in range(len(surf_pos)):
        ps  = surf_pos[si]
        rs  = float(surf_rcov[si])

        for r_a in anchor_rcov_list:
            r_a = float(r_a)
            d_ideal  = r_a + rs
            d_cutoff = bond_factor * d_ideal

            for radius in (d_ideal, d_cutoff):
                # Angular spacing: arc-length ~ grid_spacing
                n_theta = max(2, int(np.ceil((np.pi / 2) * radius / grid_spacing)))
                for i_t in range(n_theta + 1):
                    theta = (np.pi / 2) * i_t / n_theta   # 0 = top, π/2 = equator
                    rho   = radius * np.sin(theta)         # xy-plane offset
                    dz    = radius * np.cos(theta)
                    z     = ps[2] + dz
                    if z < z_min:
                        continue
                    n_phi = max(1, int(np.ceil(2 * np.pi * rho / grid_spacing))) \
                            if rho > 1e-6 else 1
                    for i_p in range(n_phi):
                        phi = 2 * np.pi * i_p / n_phi
                        pt = np.array([
                            ps[0] + rho * np.cos(phi),
                            ps[1] + rho * np.sin(phi),
                            z,
                        ])
                        # PBC-wrap xy into primary cell via fractional coords
                        frac = pt @ cell_inv
                        frac[:2] = frac[:2] % 1.0
                        raw.append(frac @ cell)

    if not raw:
        return np.empty((0, 3))

    pts = np.array(raw)

    # Spatial deduplication with tolerance grid_spacing / 2
    tol  = grid_spacing / 2.0
    tree = cKDTree(pts)
    used = np.zeros(len(pts), dtype=bool)
    keep: list[int] = []
    for i in range(len(pts)):
        if used[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], tol):
            used[j] = True

    return pts[keep]


# ---------------------------------------------------------------------------
# Single-atom site finding
# ---------------------------------------------------------------------------

def _find_sites_single(surface_graph: nx.Graph, reactant: Reactant,
                        bond_factor: float, grid_spacing: float,
                        verbose: bool,
                        clash_factor: float | None = None) -> tuple[list, list, list]:
    """Returns (unique_pts, unique_conns, iso_class_ids)."""
    cell = surface_graph.graph["cell"]
    surf_nodes   = [(n, d) for n, d in surface_graph.nodes(data=True)
                    if d["type"] == "surface"]
    surf_indices = np.array([n for n, d in surf_nodes], dtype=int)
    surf_pos     = np.array([d["position"] for n, d in surf_nodes])
    surf_rcov    = np.array([d["covalent_radius"] for n, d in surf_nodes])

    ads_data  = next(iter(reactant.graph.nodes(data=True)))[1]
    r_cov_ads = float(ads_data["covalent_radius"])
    ads_elem  = ads_data["element"]

    bond_cutoffs     = bond_factor * (r_cov_ads + surf_rcov)
    bond_targets_all = r_cov_ads + surf_rcov
    surf_z_max = surf_pos[:, 2].max()
    d_min = r_cov_ads * 0.5
    d_max = bond_factor * (r_cov_ads + surf_rcov.max())

    # ── Adaptive probe grid ────────────────────────────────────────────────
    # Points are placed on hemispherical shells centred on each surface atom
    # at radii tuned to (r_ads + r_surf) rather than a flat uniform grid.
    probe_pts = _adaptive_probe_grid(
        surf_pos, surf_rcov, [r_cov_ads],
        bond_factor, grid_spacing, surf_z_max, cell,
    )

    # Distance matrix between probe points and surface atoms (with PBC)
    cell_inv = np.linalg.inv(cell)
    dv    = surf_pos[None] - probe_pts[:, None]
    dfrac = dv @ cell_inv
    dfrac[:, :, :2] -= np.round(dfrac[:, :, :2])
    d_mat = np.sqrt(((dfrac @ cell) ** 2).sum(axis=2))

    # Keep only points within the bonding z-window
    keep = (d_mat.min(axis=1) >= d_min) & (d_mat.min(axis=1) <= d_max)
    probe_pts = probe_pts[keep]; d_mat = d_mat[keep]

    if verbose:
        print(f"  [single] probe pts after filter: {len(probe_pts)}")

    # Deduplicate — use sorted tuple as key (cheaper than frozenset hashing)
    seen: dict = {}
    unique_pts: list[np.ndarray] = []
    unique_conns: list[frozenset] = []
    n_clash_pruned = 0
    for i in range(len(probe_pts)):
        bonded = np.where(d_mat[i] <= bond_cutoffs)[0]
        if not len(bonded):
            continue
        key = tuple(sorted(int(surf_indices[k]) for k in bonded))
        if key in seen:
            continue

        # ── Steric clash filter ────────────────────────────────────────────
        # Reject probes where a non-bonded surface atom sits within
        # clash_factor*(r_cov_ads+r_cov_s) — i.e. it would actually bond,
        # meaning the connectivity set is incomplete / the site is blocked.
        if clash_factor is not None:
            if _is_probe_clashing(probe_pts[i], set(bonded.tolist()),
                                  surf_pos, surf_rcov, r_cov_ads,
                                  clash_factor, cell_inv, cell):
                n_clash_pruned += 1
                continue

        seen[key] = len(unique_pts)
        unique_pts.append(probe_pts[i].copy())
        unique_conns.append(frozenset(key))

    if verbose:
        print(f"  [single] unique connectivities: {len(unique_pts)}"
              + (f"  (clash pruned={n_clash_pruned})" if clash_factor is not None else ""))

    # Isomorphism — pre-filter with cheap structural key before running VF2
    node_match = isomorphism.categorical_node_match("element", "X")
    class_reps: list[nx.Graph] = []
    class_keys: list[tuple] = []   # pre-filter fingerprints
    iso_ids: list[int] = []
    for conn in unique_conns:
        ego = _build_ego_single(surface_graph, conn, ads_elem)
        fkey = _iso_prefilter_key(ego)
        assigned = False
        for cid, (rep, rkey) in enumerate(zip(class_reps, class_keys)):
            if fkey != rkey:
                continue
            if isomorphism.GraphMatcher(ego, rep, node_match=node_match).is_isomorphic():
                iso_ids.append(cid); assigned = True; break
        if not assigned:
            class_reps.append(ego)
            class_keys.append(fkey)
            iso_ids.append(len(class_reps) - 1)

    # Geometric optimisation — one representative per iso-class only
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}
    z_lo = float(surf_z_max) + d_min
    z_hi = float(surf_z_max) + d_max

    # Find the index of the first member of each iso-class
    iso_ids_arr = np.array(iso_ids)
    n_classes   = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0
    class_rep_idx = {cid: int(np.where(iso_ids_arr == cid)[0][0])
                     for cid in range(n_classes)}

    # Optimise one representative per class, then copy to all members
    class_opt_pts: dict[int, np.ndarray] = {}
    for cid, rep_i in class_rep_idx.items():
        pt0  = unique_pts[rep_i]
        conn = unique_conns[rep_i]
        lids       = [surf_idx_to_local[g] for g in sorted(conn)]
        conn_pos   = surf_pos[lids]
        bond_tgts  = bond_targets_all[lids]
        nb_lids    = [k for k in range(len(surf_indices)) if k not in lids]
        nonbond    = surf_pos[nb_lids] if nb_lids else np.empty((0, 3))
        class_opt_pts[cid] = _opt_single(pt0, conn_pos, bond_tgts, nonbond,
                                         cell, z_lo, z_hi)

    opt_pts = [class_opt_pts[cid] for cid in iso_ids]

    return opt_pts, unique_conns, iso_ids, class_reps, surf_rcov, surf_idx_to_local, bond_targets_all


# ---------------------------------------------------------------------------
# Multi-atom site finding
# ---------------------------------------------------------------------------

def _auto_n_orientations(ads_pos: np.ndarray, grid_spacing: float) -> int:
    """Compute SO(3) sample count consistent with *grid_spacing*.

    A rotation by angle δ displaces the furthest adsorbate atom (at radius
    *R_max* from the molecular centroid) by ``R_max × δ``.  Requiring
    ``δ ≤ grid_spacing / R_max`` to not miss any connectivity transition
    gives::

        N ≈ (π × R_max / grid_spacing)²

    This ties orientation density to the same length-scale as the position
    grid: larger molecules automatically receive more orientations, and
    tightening *grid_spacing* consistently increases both.
    Returns 1 for atomic / zero-extent adsorbates.
    """
    centroid = ads_pos.mean(axis=0)
    r_max = float(np.linalg.norm(ads_pos - centroid, axis=1).max())
    if r_max < 1e-6:
        return 1
    return max(1, int(np.ceil((np.pi * r_max / grid_spacing) ** 2)))


def _find_sites_multi(surface_graph: nx.Graph, reactant: Reactant,
                       bond_factor: float, grid_spacing: float,
                       n_orientations: int | None, verbose: bool,
                       clash_factor: float | None = None) -> tuple:
    """Returns (opt_positions_list, unique_conns, iso_ids, class_reps).

    Optimisations applied
    ---------------------
    1. **Vectorised orientation loop** – all *n_orientations* rotations for a
       given grid point and anchor atom are evaluated in a single batched
       NumPy operation rather than a Python loop.
    2. **KD-tree xy pre-filter** – grid points whose nearest surface atom
       (in the unwrapped xy plane) is farther than *d_max_global* are
       discarded before the rotation search begins.
    3. **Sorted-tuple connectivity key** – cheaper to construct and compare
       than ``frozenset`` for the deduplication dict.
    5. **Isomorphism pre-filter** – a cheap structural fingerprint
       (node/edge counts, degree sequence, element–degree pairs) is compared
       before invoking the full VF2 ``GraphMatcher``.
    """
    from scipy.spatial import cKDTree

    cell = surface_graph.graph["cell"]
    cell_inv = np.linalg.inv(cell)

    surf_nodes   = [(n, d) for n, d in surface_graph.nodes(data=True)
                    if d["type"] == "surface"]
    surf_indices = np.array([n for n, d in surf_nodes], dtype=int)
    surf_pos     = np.array([d["position"] for n, d in surf_nodes])
    surf_rcov    = np.array([d["covalent_radius"] for n, d in surf_nodes])

    ads_nodes = list(reactant.graph.nodes(data=True))
    ads_pos   = np.array([d["position"] for _, d in ads_nodes])   # (N_ads, 3)
    ads_rcov  = np.array([d["covalent_radius"] for _, d in ads_nodes])
    N_ads     = len(ads_pos)

    # Auto-compute orientations from molecule extent and grid_spacing if not given
    if n_orientations is None:
        n_orientations = _auto_n_orientations(ads_pos, grid_spacing)
        if verbose:
            centroid = ads_pos.mean(axis=0)
            r_max = float(np.linalg.norm(ads_pos - centroid, axis=1).max())
            print(f"  [multi] n_orientations auto={n_orientations} "
                  f"(R_max={r_max:.3f} Å, grid_spacing={grid_spacing:.3f} Å)")

    surf_z_max = surf_pos[:, 2].max()

    # Pre-compute per-(ads_atom, surf_atom) bonding cutoffs: (N_ads, S)
    cutoff_mat = bond_factor * (ads_rcov[:, None] + surf_rcov[None, :])  # (N_ads, S)
    max_cutoff = float(cutoff_mat.max())

    # z-window for anchor atoms
    d_min_global = ads_rcov.min() * 0.5
    d_max_global = bond_factor * (ads_rcov.max() + surf_rcov.max())
    z_lo = float(surf_z_max) + d_min_global
    z_hi = float(surf_z_max) + d_max_global

    # Sample SO(3) rotations once  →  (n_orient, 3, 3)
    rot_mats = _sample_so3(n_orientations)

    # ── Adaptive probe grid (replaces uniform xy+z grid) ──────────────────
    # One combined grid covers all anchor cov radii: for each surface atom and
    # each distinct anchor cov radius, shells at the ideal and cutoff bond
    # distances are sampled.  This ensures anchor positions are always placed
    # at physically meaningful distances regardless of which atom anchors.
    grid_pts = _adaptive_probe_grid(
        surf_pos, surf_rcov, list(ads_rcov),
        bond_factor, grid_spacing, surf_z_max, cell,
    )

    # ── Strategy 2: KD-tree xy pre-filter ─────────────────────────────────
    # (still applied on top of the adaptive grid to handle the tiled-image
    #  case and any residual out-of-range points)
    cell_x = float(cell[0, 0]); cell_y = float(cell[1, 1])
    tile_offsets = np.array([[dx * cell_x, dy * cell_y]
                              for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    surf_xy_tiled = (surf_pos[:, :2][:, None, :] + tile_offsets[None, :, :]).reshape(-1, 2)
    tree = cKDTree(surf_xy_tiled)
    dists_xy, _ = tree.query(grid_pts[:, :2])
    grid_pts = grid_pts[dists_xy <= max_cutoff]

    # ── 3D PBC-tiled KD-tree for per-point local neighbourhood queries ─────
    # For each grid point + anchor, only surface atoms within
    #   reach = max_extent_of_molecule_from_anchor + max_bond_cutoff
    # can possibly bond to any adsorbate atom regardless of orientation.
    # Pre-filtering to this local set avoids computing distances to the whole
    # surface (~S atoms) and replaces it with a much smaller local set (~L
    # atoms, L << S for small adsorbates on large slabs).
    pbc_z_offsets = np.array([[dx * cell_x, dy * cell_y, 0.0]
                               for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    surf_pos_tiled_3d = np.concatenate(
        [surf_pos + off for off in pbc_z_offsets], axis=0
    )  # (9*S, 3)
    surf_local_from_tiled = np.tile(np.arange(len(surf_pos)), 9)  # (9*S,)
    tree_3d = cKDTree(surf_pos_tiled_3d)

    if verbose:
        print(f"  [multi] grid pts: {len(grid_pts)} (after xy filter),  "
              f"orientations: {n_orientations},  anchors: {N_ads}")

    # ── Convex-hull pre-filter ────────────────────────────────────────────
    # Interior atoms (e.g. C in CH₄) are shielded by their neighbours and
    # can never reach the surface directly regardless of orientation.
    # Only hull vertices are kept as anchor candidates.
    hull_indices = set(_hull_bondable_indices(ads_pos))

    # ── WL anchor deduplication ───────────────────────────────────────────
    # Of the hull vertices, only one representative per WL-colour class
    # needs to be sampled — symmetrically equivalent atoms (e.g. both O in
    # O₂, or all H in CH₄) produce identical connectivity patterns.
    wl_reps = set(_wl_anchor_indices(reactant))
    anchor_indices = [i for i in range(N_ads)
                      if i in hull_indices and i in wl_reps]

    # Safety: if the intersection is empty (degenerate molecule) fall back
    # to WL reps alone so we always sample something.
    if not anchor_indices:
        anchor_indices = sorted(wl_reps)

    if verbose:
        print(f"  [multi] anchor candidates: {anchor_indices} "
              f"(hull={sorted(hull_indices)}, WL-reps={sorted(wl_reps)}, "
              f"final={len(anchor_indices)}/{N_ads})")

    # ── Adsorbate automorphism deduplication ──────────────────────────────
    # Pre-compute all graph automorphisms (element-preserving bijections of
    # the adsorbate graph onto itself).  When a new connectivity pattern is
    # accepted, all permutations of equivalent atoms are immediately
    # registered in `seen` so that redundant orientations are never added.
    _ads_nm  = isomorphism.categorical_node_match("element", "X")
    _gm_auto = isomorphism.GraphMatcher(
        reactant.graph, reactant.graph, node_match=_ads_nm
    )
    automorphisms: list[dict] = list(_gm_auto.isomorphisms_iter())

    def _symmetric_conn_keys(conn_list: list[tuple[int, int]]) -> list[tuple]:
        """Return all automorphism-permuted variants of a connectivity list."""
        keys: set[tuple] = set()
        for perm in automorphisms:
            keys.add(tuple(sorted((perm[ak], sg) for ak, sg in conn_list)))
        return list(keys)

    # ── Deduplication state ───────────────────────────────────────────────
    seen: dict[tuple, int] = {}
    unique_conns:      list[frozenset]    = []
    unique_anchors:    list[int]          = []
    unique_anchor_pos: list[np.ndarray]  = []
    unique_rotvec:     list[np.ndarray]  = []

    n_clash_pruned = 0

    # ── Vectorised orientation loop ───────────────────────────────────────
    for anchor_idx in anchor_indices:
        rel_pos = ads_pos - ads_pos[anchor_idx]   # (N_ads, 3)

        # Maximum distance any adsorbate atom can be from the anchor grid point,
        # regardless of orientation.  Any surface atom beyond this + max_bond_cutoff
        # is guaranteed to be out of reach for every orientation at every grid point.
        d_mol_max    = float(np.linalg.norm(rel_pos, axis=1).max())
        reach_radius = d_mol_max + float(cutoff_mat.max())

        for gpt in grid_pts:
            # ── Local surface atom pre-filter ─────────────────────────────
            # Only surface atoms within `reach_radius` of this grid point can
            # bond to any adsorbate atom at any orientation.  This reduces the
            # per-point distance tensor from (n_orient, N_ads, S) to
            # (n_orient, N_ads, L) where L is typically much smaller than S.
            tiled_near = tree_3d.query_ball_point(gpt, reach_radius)
            if not tiled_near:
                continue
            local_sl     = np.unique(surf_local_from_tiled[tiled_near])  # original local indices
            l_surf_pos   = surf_pos[local_sl]          # (L, 3)
            l_surf_rcov  = surf_rcov[local_sl]         # (L,)
            l_cutoff_mat = cutoff_mat[:, local_sl]     # (N_ads, L)
            l_surf_idx   = surf_indices[local_sl]      # global atom indices (L,)

            # placed: (n_orient, N_ads, 3)
            placed = gpt + np.einsum("oij,aj->oai", rot_mats, rel_pos)

            # PBC distances — only against local surface atoms → (n_orient, N_ads, L, 3)
            dv   = l_surf_pos[None, None, :, :] - placed[:, :, None, :]
            frac = dv @ cell_inv
            frac[..., :2] -= np.round(frac[..., :2])
            d_all = np.sqrt(((frac @ cell) ** 2).sum(axis=-1))  # (n_orient, N_ads, L)

            # Bonded mask: (n_orient, N_ads, L)
            bonded_mask = d_all <= l_cutoff_mat[None, :, :]

            # Any orientation that has at least one bond is interesting
            has_bond    = bonded_mask.any(axis=(1, 2))  # (n_orient,)
            interesting = np.where(has_bond)[0]

            for oi in interesting:
                    # Build connectivity set — sl_idx indexes into local_sl
                    ak_idx, sl_idx = np.where(bonded_mask[oi])

                    # ── Steric clash filter ────────────────────────────────
                    # Uses the local surface arrays; atoms outside reach_radius
                    # cannot clash because they cannot bond either.
                    if clash_factor is not None:
                        clashing = False
                        bonded_surf_per_ak: dict[int, set] = {}
                        for bi in range(len(ak_idx)):
                            bonded_surf_per_ak.setdefault(int(ak_idx[bi]), set()).add(int(sl_idx[bi]))
                        for ak_k, sl_set in bonded_surf_per_ak.items():
                            probe_ak = placed[oi, ak_k]
                            r_a = float(ads_rcov[ak_k])
                            if _is_probe_clashing(probe_ak, sl_set, l_surf_pos,
                                                  l_surf_rcov, r_a, clash_factor,
                                                  cell_inv, cell):
                                clashing = True
                                break
                        if clashing:
                            n_clash_pruned += 1
                            continue

                    # Map local sl_idx → global surface atom indices
                    conn_pairs = tuple(sorted(
                        (int(ak_idx[i]), int(l_surf_idx[sl_idx[i]]))
                        for i in range(len(ak_idx))
                    ))
                    if not conn_pairs:
                        continue
                    if conn_pairs in seen:
                        continue
                    # New connectivity: register all automorphism-permuted
                    # variants so equivalent orientations are never duplicated.
                    idx = len(unique_conns)
                    for sym_key in _symmetric_conn_keys(list(conn_pairs)):
                        seen[sym_key] = idx
                    unique_conns.append(frozenset(conn_pairs))
                    unique_anchors.append(anchor_idx)
                    unique_anchor_pos.append(gpt.copy())
                    unique_rotvec.append(
                        Rotation.from_matrix(rot_mats[oi]).as_rotvec()
                    )

    if verbose:
        print(f"  [multi] unique connectivities: {len(unique_conns)}"
              + (f"  (clash pruned={n_clash_pruned})" if clash_factor is not None else ""))

    # ── Strategy 5: isomorphism pre-filter ────────────────────────────────
    node_match = isomorphism.categorical_node_match("element", "X")
    class_reps:  list[nx.Graph] = []
    class_keys:  list[tuple]    = []
    iso_ids: list[int] = []
    for conn in unique_conns:
        ego  = _build_ego_multi(surface_graph, reactant, conn)
        fkey = _iso_prefilter_key(ego)
        assigned = False
        for cid, (rep, rkey) in enumerate(zip(class_reps, class_keys)):
            if fkey != rkey:
                continue
            if isomorphism.GraphMatcher(ego, rep, node_match=node_match).is_isomorphic():
                iso_ids.append(cid); assigned = True; break
        if not assigned:
            class_reps.append(ego)
            class_keys.append(fkey)
            iso_ids.append(len(class_reps) - 1)

    # Geometric optimisation (6 DOF) — one representative per iso-class only
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}

    iso_ids_arr = np.array(iso_ids)
    n_classes   = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0
    class_rep_idx = {cid: int(np.where(iso_ids_arr == cid)[0][0])
                     for cid in range(n_classes)}

    class_opt_pos: dict[int, np.ndarray] = {}
    for cid, rep_i in class_rep_idx.items():
        conn       = unique_conns[rep_i]
        anchor_idx = unique_anchors[rep_i]
        anchor0    = unique_anchor_pos[rep_i]
        rv0        = unique_rotvec[rep_i]
        rel_pos    = ads_pos - ads_pos[anchor_idx]
        class_opt_pos[cid] = _opt_multi(
            anchor0, rv0, anchor_idx, rel_pos, ads_rcov,
            conn, surf_pos, surf_rcov, surf_idx_to_local,
            cell, z_lo, z_hi,
        )

    opt_positions = [class_opt_pos[cid] for cid in iso_ids]

    return opt_positions, unique_conns, iso_ids, class_reps


# ---------------------------------------------------------------------------
# EMT relaxation helpers
# ---------------------------------------------------------------------------

def _get_frozen_indices(slab, n_freeze_layers: int) -> list[int]:
    """Return atom indices belonging to the bottom *n_freeze_layers* layers.

    Resolution order
    ----------------
    1. ``slab.info["frozen_indices"]`` – set by :func:`~autokmc.structure.build_surface`
       at construction time; used as-is when present.
    2. Existing :class:`~ase.constraints.FixAtoms` constraint on *slab* –
       used when the slab was built externally but still carries the constraint.
    3. Z-coordinate clustering heuristic (original fallback).
    """
    from ase.constraints import FixAtoms

    # 1. Preferred: indices stored in info dict
    if "frozen_indices" in slab.info:
        return list(slab.info["frozen_indices"])

    # 2. Read from existing FixAtoms constraint
    for c in slab.constraints:
        if isinstance(c, FixAtoms):
            return list(c.index)

    # 3. Fallback: z-coordinate clustering
    pos    = slab.get_positions()
    z_vals = pos[:, 2]
    z_sort = np.sort(z_vals)

    # Estimate NN spacing as the smallest gap > 0.3 Å between sorted z values
    diffs = np.diff(z_sort)
    nonzero = diffs[diffs > 0.3]
    tol = float(nonzero.min()) / 4.0 if len(nonzero) else 0.4

    layers: list[list[float]] = []
    current: list[float] = [z_sort[0]]
    for z in z_sort[1:]:
        if z - current[-1] < tol:
            current.append(z)
        else:
            layers.append(current)
            current = [z]
    layers.append(current)

    freeze_set: set[int] = set()
    for lyr in layers[:n_freeze_layers]:
        for z in lyr:
            for i, zi in enumerate(z_vals):
                if abs(zi - z) < tol:
                    freeze_set.add(i)
    return list(freeze_set)


def _relax_per_iso_class(
    slab,
    ads_element: str,
    unique_pts: list[np.ndarray],
    unique_conns: list[frozenset],
    iso_ids: list[int],
    surface_indices: np.ndarray,
    cell: np.ndarray,
    calculator,
    n_freeze_layers: int,
    bond_cutoff: float,
    fmax: float,
    steps: int,
    verbose: bool,
    debug_dir: str | None = None,
    reactant=None,
    bond_factor: float = 1.1,
) -> dict[int, dict]:
    """Relax one representative per iso-class.

    Returns
    -------
    iso_results : dict[int, dict]
        Keys are iso-class ids.  Each value has:
        ``energy`` (float), ``converged`` (bool), ``valid`` (bool).
        Invalid classes (connectivity changed during relaxation) are
        included with ``valid=False`` and ``energy=None``.
    """
    from ase import Atom
    from ase.optimize import LBFGS
    from ase.constraints import FixAtoms

    iso_ids_arr = np.array(iso_ids)
    n_iso_classes = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0

    # Clean slab energy (fresh calculator copy to avoid state pollution)
    slab_clean = slab.copy()
    slab_clean.calc = copy.deepcopy(calculator)
    e_slab = float(slab_clean.get_potential_energy())
    del slab_clean

    frozen_idx = _get_frozen_indices(slab, n_freeze_layers)
    cell_x = float(cell[0, 0])
    cell_y = float(cell[1, 1])

    if verbose:
        print(f"\n  EMT relaxation: 1 representative per iso-class "
              f"({n_iso_classes} classes, {len(frozen_idx)} atoms frozen)")
        print("  " + "=" * 56)

    iso_results: dict[int, dict] = {}

    for cid in range(n_iso_classes):
        members = np.where(iso_ids_arr == cid)[0]
        if len(members) == 0:
            continue
        site_idx = int(members[0])
        opt_pos  = unique_pts[site_idx]
        conn_set = unique_conns[site_idx]

        # Build slab + adsorbate
        slab_ads = slab.copy()
        slab_ads.calc = copy.deepcopy(calculator)
        slab_ads.append(Atom(ads_element, position=opt_pos))
        ads_idx = len(slab_ads) - 1
        slab_ads.set_constraint(FixAtoms(indices=frozen_idx))

        opt = LBFGS(slab_ads, logfile=os.devnull)
        opt.run(fmax=fmax, steps=steps)

        converged = bool(opt.converged())
        final_pos = slab_ads.get_positions()
        ads_pos_final = final_pos[ads_idx]

        # Connectivity validation: check bonds didn't change after relaxation
        dv = slab.get_positions()[surface_indices] - ads_pos_final
        dv[:, 0] -= np.round(dv[:, 0] / cell_x) * cell_x
        dv[:, 1] -= np.round(dv[:, 1] / cell_y) * cell_y
        d_surf = np.sqrt((dv ** 2).sum(axis=1))
        conn_actual   = frozenset(int(surface_indices[k])
                                  for k in np.where(d_surf <= bond_cutoff)[0])
        conn_intended = frozenset(int(g) for g in conn_set)
        valid = conn_actual == conn_intended

        # ── adsorbate internal-connectivity check ─────────────────────────
        # Rebuild bond graph from relaxed distances; compare to original topology.
        if valid and reactant is not None and len(reactant.atoms) > 1:
            from ase.data import covalent_radii as _cov_rad
            from ase.data import atomic_numbers as _anum
            n_ads = len(reactant.atoms)
            ads_positions = final_pos[-n_ads:]
            ads_rcov = np.array([
                _cov_rad[_anum[sym]]
                for sym in reactant.atoms.get_chemical_symbols()
            ])
            # Build relaxed bond graph
            relaxed_bonds: set[tuple[int, int]] = set()
            for ai in range(n_ads):
                for aj in range(ai + 1, n_ads):
                    d = np.linalg.norm(ads_positions[ai] - ads_positions[aj])
                    cutoff = bond_factor * (ads_rcov[ai] + ads_rcov[aj])
                    if d <= cutoff:
                        relaxed_bonds.add((ai, aj))
            # Expected bonds from original reactant graph
            expected_bonds = {(min(u, v), max(u, v)) for u, v in reactant.graph.edges()}
            if relaxed_bonds != expected_bonds:
                valid = False
                if verbose:
                    print(f"    adsorbate connectivity changed: "
                          f"expected={expected_bonds}  got={relaxed_bonds}")
        # ─────────────────────────────────────────────────────────────────

        adsorption_e = float(slab_ads.get_potential_energy()) - e_slab if valid else None

        if verbose:
            status = "OK  " if valid else "FAIL"
            n_i = len(conn_intended); n_a = len(conn_actual)
            print(f"  iso-class {cid:2d}  [{status}]  "
                  f"converged={converged}  "
                  f"intended={n_i}-fold  actual={n_a}-fold"
                  + (f"  E_ads={adsorption_e:.4f} eV" if adsorption_e is not None else ""))

        iso_results[cid] = dict(energy=adsorption_e, converged=converged, valid=valid)

    if verbose:
        n_valid = sum(1 for r in iso_results.values() if r["valid"])
        print("  " + "=" * 56)
        print(f"  Valid classes : {n_valid} / {n_iso_classes}")

    return iso_results


def _relax_per_iso_class_multi(
    slab,
    reactant: Reactant,
    opt_positions: list[np.ndarray],
    unique_conns: list[frozenset],
    iso_ids: list[int],
    surface_indices: np.ndarray,
    cell: np.ndarray,
    calculator,
    n_freeze_layers: int,
    bond_factor: float,
    fmax: float,
    steps: int,
    verbose: bool,
    debug_dir: str | None = None,
    internal_bond_factor: float = 1.3,
) -> dict[int, dict]:
    """Relax one representative per iso-class for a **multi-atom** adsorbate.

    After relaxation validates:
    1. Adsorbate–surface connectivity matches the intended set.
    2. Adsorbate internal bond graph is unchanged (molecule did not break apart).
    """
    from ase import Atoms as AseAtoms
    from ase.optimize import LBFGS
    from ase.constraints import FixAtoms
    from ase.data import covalent_radii as _cov_rad
    from ase.data import atomic_numbers as _anum

    iso_ids_arr   = np.array(iso_ids)
    n_iso_classes = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0
    n_ads         = len(reactant.atoms)
    ads_symbols   = reactant.atoms.get_chemical_symbols()
    ads_rcov      = np.array([_cov_rad[_anum[s]] for s in ads_symbols])

    # Expected internal bonds (from RDKit topology)
    expected_bonds = {(min(u, v), max(u, v)) for u, v in reactant.graph.edges()}

    # Per-atom covalent radii of surface atoms — looked up from slab symbols
    from ase.data import covalent_radii as _cov_rad_surf
    from ase.data import atomic_numbers as _anum_surf
    slab_symbols = slab.get_chemical_symbols()
    surf_rcov_check = np.array([
        _cov_rad_surf[_anum_surf[slab_symbols[i]]] for i in surface_indices
    ])
    cell_x = float(cell[0, 0])
    cell_y = float(cell[1, 1])

    # Clean slab energy
    slab_clean = slab.copy()
    slab_clean.calc = copy.deepcopy(calculator)
    e_slab = float(slab_clean.get_potential_energy())
    del slab_clean

    frozen_idx = _get_frozen_indices(slab, n_freeze_layers)

    if verbose:
        print(f"\n  Calculator relaxation: 1 representative per iso-class "
              f"({n_iso_classes} classes, {n_ads}-atom adsorbate, "
              f"{len(frozen_idx)} atoms frozen)")
        print("  " + "=" * 56)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    iso_results: dict[int, dict] = {}

    for cid in range(n_iso_classes):
        members = np.where(iso_ids_arr == cid)[0]
        if len(members) == 0:
            continue
        site_idx   = int(members[0])
        ads_pos0   = opt_positions[site_idx]   # (N_ads, 3)
        conn_pairs = unique_conns[site_idx]    # frozenset of (ak, sg)

        # Build slab + adsorbate
        slab_ads = slab.copy()
        slab_ads.calc = copy.deepcopy(calculator)
        for k in range(n_ads):
            slab_ads.append(ads_symbols[k])
            slab_ads.positions[-1] = ads_pos0[k]
        slab_ads.set_constraint(FixAtoms(indices=frozen_idx))

        if debug_dir is not None:
            from ase.io import write as _ase_write
            _ase_write(os.path.join(debug_dir, f"class_{cid:02d}_initial.extxyz"), slab_ads)

        logfile = os.path.join(debug_dir, f"class_{cid:02d}.log") \
            if debug_dir is not None else os.devnull
        traj    = os.path.join(debug_dir, f"class_{cid:02d}.traj") \
            if debug_dir is not None else None

        opt = LBFGS(slab_ads, logfile=logfile,
                    trajectory=traj if traj else None)
        opt.run(fmax=fmax, steps=steps)

        converged  = bool(opt.converged())
        final_pos  = slab_ads.get_positions()
        ads_final  = final_pos[-n_ads:]   # (N_ads, 3)

        if debug_dir is not None:
            from ase.io import write as _ase_write
            _ase_write(os.path.join(debug_dir, f"class_{cid:02d}_final.extxyz"), slab_ads)

        # ── 1. Adsorbate–surface connectivity check ───────────────────────
        surf_pos_all = slab.get_positions()
        conn_actual: set[tuple[int, int]] = set()
        for ak in range(n_ads):
            dv = surf_pos_all[surface_indices] - ads_final[ak]
            dv[:, 0] -= np.round(dv[:, 0] / cell_x) * cell_x
            dv[:, 1] -= np.round(dv[:, 1] / cell_y) * cell_y
            d = np.sqrt((dv ** 2).sum(axis=1))
            cutoffs_k = bond_factor * (ads_rcov[ak] + surf_rcov_check)
            for k in np.where(d <= cutoffs_k)[0]:
                conn_actual.add((int(ak), int(surface_indices[k])))

        conn_intended = {(ak, sg) for ak, sg in conn_pairs}
        valid = frozenset(conn_actual) == frozenset(conn_intended)

        # ── 2. Adsorbate internal-connectivity check ──────────────────────
        if valid and n_ads > 1:
            relaxed_bonds: set[tuple[int, int]] = set()
            for ai in range(n_ads):
                for aj in range(ai + 1, n_ads):
                    d = float(np.linalg.norm(ads_final[ai] - ads_final[aj]))
                    cutoff = internal_bond_factor * (ads_rcov[ai] + ads_rcov[aj])
                    if d <= cutoff:
                        relaxed_bonds.add((ai, aj))
            if relaxed_bonds != expected_bonds:
                valid = False
                if verbose:
                    print(f"    iso-class {cid:2d}: adsorbate broke apart — "
                          f"expected bonds {expected_bonds}, got {relaxed_bonds}")

        adsorption_e = float(slab_ads.get_potential_energy()) - e_slab if valid else None

        if verbose:
            status = "OK  " if valid else "FAIL"
            n_i = len({sg for _, sg in conn_intended})
            n_a = len({sg for _, sg in conn_actual})
            print(f"  iso-class {cid:2d}  [{status}]  converged={converged}  "
                  f"surf-bonds intended={n_i}  actual={n_a}"
                  + (f"  E_ads={adsorption_e:.4f} eV" if adsorption_e is not None else ""))

        iso_results[cid] = dict(energy=adsorption_e, converged=converged, valid=valid)

    if verbose:
        n_valid = sum(1 for r in iso_results.values() if r["valid"])
        print("  " + "=" * 56)
        print(f"  Valid classes : {n_valid} / {n_iso_classes}")

    return iso_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_adsorption_sites(
    surface_graph: nx.Graph,
    reactant: Reactant,
    *,
    bond_factor: float = 1.1,
    grid_spacing: float = 0.4,
    n_orientations: int | None = None,
    clash_factor: float | None = None,
    internal_bond_factor: float = 1.3,
    calculator: Any = None,
    slab: Any = None,
    n_freeze_layers: int = 2,
    fmax: float = 0.05,
    steps: int = 500,
    debug_dir: str | None = "debug",
    verbose: bool = False,
) -> tuple[list[AdsorptionSite], nx.Graph]:
    """Find all unique adsorption sites for an adsorbate on a surface.

    Parameters
    ----------
    surface_graph : nx.Graph
        Built by :func:`~autokmc.graph.build_graph`.  Must have ``"cell"``
        and ``"pbc"`` graph attributes and node attributes ``type``,
        ``position``, ``covalent_radius``.
    reactant : Reactant
        Adsorbate from :func:`~autokmc.reactants.build_reactant`.
    bond_factor : float
        Bonding cutoff multiplier.  Default 1.1.
    grid_spacing : float
        Surface grid spacing (Å).  Default 0.4 Å.
    n_orientations : int or None
        Number of random SO(3) orientations for multi-atom adsorbates.
        ``None`` (default) auto-computes the count from *grid_spacing* and
        the molecule's geometric extent: ``N = ceil((π × R_max / grid_spacing)²)``,
        where *R_max* is the furthest atom distance from the molecular centroid.
        Pass an explicit integer to override.
    clash_factor : float or None
        Steric clash threshold as a fraction of the combined covalent radii
        of the adsorbate atom and each non-bonded surface atom.  When any
        non-bonded surface atom is within
        ``clash_factor × (r_cov_ads + r_cov_s)`` of the probe position the
        site is rejected.

        * **Single-atom**: applied per grid point before deduplication.
        * **Multi-atom**: applied per adsorbate atom per orientation before
          deduplication.

        ``None`` (default) resolves to *bond_factor* at runtime — i.e. the
        standard bonding cutoff.  This is the physically correct default:
        a site is blocked if a non-bonded atom would actually bond to the
        adsorbate at the estimated probe position.  Pass an explicit float
        to tighten (< bond_factor) or loosen (> bond_factor) the filter.
        Pass ``0.0`` to disable it entirely.
    internal_bond_factor : float
        Bond-length multiplier used when checking whether the adsorbate's
        internal bond topology is preserved after relaxation.  Default
        ``1.3``, intentionally larger than *bond_factor* (1.1) because
        molecular bonds genuinely stretch on adsorption (e.g. O₂ on
        Cu(111) can reach ~1.45 Å vs. gas-phase 1.21 Å).  Increase toward
        1.5 for very soft bonds; decrease toward 1.1 to require near
        gas-phase bond lengths.
    calculator : ASE calculator or None
        When provided (together with *slab*), one representative per
        iso-class is structurally relaxed with this calculator.
        Adsorption energies are stored on the returned sites.
    slab : ASE Atoms or None
        The relaxed slab object.  Required when *calculator* is given.
    n_freeze_layers : int
        Number of bottom layers to fix during structural relaxation.
        Default 2.
    fmax : float
        Force convergence criterion for LBFGS.  Default 0.05 eV/Å.
    steps : int
        Maximum LBFGS steps per relaxation.  Default 500.
    verbose : bool
        Print progress.  Default ``False``.
    debug_dir : str or None
        When *calculator* is given, write ``class_XX_initial.extxyz``,
        ``class_XX_final.extxyz``, ``.traj`` and ``.log`` files here
        (matching workflow_surface.py convention).  Default ``"debug"``.
        Set to ``None`` to suppress all file output.

    Returns
    -------
    sites : list[AdsorptionSite]
    site_graph : nx.Graph
        Site-adjacency graph.
    """
    n_ads = len(reactant.atoms)
    cell  = surface_graph.graph["cell"]

    # Resolve clash_factor: None → use bond_factor (the bonding cutoff itself).
    # A site is physically blocked when a non-bonded surface atom is close
    # enough to actually bond, so bond_factor is the correct default threshold.
    effective_clash: float | None = bond_factor if clash_factor is None else clash_factor
    # Allow the caller to pass 0.0 to disable the filter entirely
    if effective_clash == 0.0:
        effective_clash = None

    if verbose:
        cf_str = (f"{effective_clash:.2f}" + (" (= bond_factor)" if clash_factor is None else "")
                  if effective_clash is not None else "disabled")
        print(f"Finding sites for '{reactant.smiles}'  "
              f"({'single' if n_ads == 1 else 'multi'}-atom path,  "
              f"clash_factor={cf_str})")

    if n_ads == 1:
        (opt_pts, unique_conns, iso_ids,
         class_reps, surf_rcov, surf_idx_to_local,
         bond_targets_all) = _find_sites_single(
            surface_graph, reactant, bond_factor, grid_spacing, verbose,
            clash_factor=effective_clash)

        # ── optional EMT relaxation (one rep per iso-class) ──────────────────
        iso_results: dict[int, dict] = {}
        if calculator is not None and slab is not None:
            # Gather surface indices from graph for connectivity re-check
            surf_nodes    = [(n, d) for n, d in surface_graph.nodes(data=True)
                             if d["type"] == "surface"]
            surf_indices  = np.array([n for n, d in surf_nodes], dtype=int)
            surf_rcov_arr = np.array([d["covalent_radius"] for n, d in surf_nodes])
            ads_data      = next(iter(reactant.graph.nodes(data=True)))[1]
            r_cov_ads     = float(ads_data["covalent_radius"])
            bond_cutoff   = bond_factor * (r_cov_ads + surf_rcov_arr.mean())

            iso_results = _relax_per_iso_class(
                slab, ads_data["element"],
                opt_pts, unique_conns, iso_ids,
                surf_indices, cell, calculator,
                n_freeze_layers, bond_cutoff, fmax, steps, verbose,
                debug_dir=debug_dir,
                reactant=reactant,
                bond_factor=bond_factor,
            )

        # ── build AdsorptionSite list, filtering invalid iso-classes ─────────
        sites: list[AdsorptionSite] = []
        for opt, conn, iso_cid in zip(opt_pts, unique_conns, iso_ids):
            # If relaxation was run and this class failed validation → skip
            if iso_results and not iso_results.get(iso_cid, {}).get("valid", True):
                continue
            n_surf = len(conn)
            res    = iso_results.get(iso_cid, {})
            sites.append(AdsorptionSite(
                position    = opt,
                conn_global = conn,
                n_conn      = n_surf,
                site_type   = _site_label(n_surf),
                iso_class   = iso_cid,
                energy      = res.get("energy"),
                converged   = res.get("converged"),
            ))

    else:
        opt_positions, unique_conns, iso_ids, class_reps = _find_sites_multi(
            surface_graph, reactant, bond_factor, grid_spacing,
            n_orientations, verbose, clash_factor=effective_clash)

        # ── optional calculator relaxation (one rep per iso-class) ───────────
        iso_results_multi: dict[int, dict] = {}
        if calculator is not None and slab is not None:
            surf_nodes   = [(n, d) for n, d in surface_graph.nodes(data=True)
                            if d["type"] == "surface"]
            surf_indices = np.array([n for n, d in surf_nodes], dtype=int)
            iso_results_multi = _relax_per_iso_class_multi(
                slab, reactant,
                opt_positions, unique_conns, iso_ids,
                surf_indices, cell, calculator,
                n_freeze_layers, bond_factor, fmax, steps, verbose,
                debug_dir=debug_dir,
                internal_bond_factor=internal_bond_factor,
            )

        sites = []
        for opt_ads, conn, iso_cid in zip(opt_positions, unique_conns, iso_ids):
            if iso_results_multi and not iso_results_multi.get(iso_cid, {}).get("valid", True):
                continue
            surf_atoms = {sg for _, sg in conn}
            n_surf = len(surf_atoms)
            res = iso_results_multi.get(iso_cid, {})
            sites.append(AdsorptionSite(
                position    = opt_ads,        # (N_ads, 3)
                conn_global = conn,
                n_conn      = n_surf,
                site_type   = _site_label(n_surf),
                iso_class   = iso_cid,
                energy      = res.get("energy"),
                converged   = res.get("converged"),
            ))

    # Site-adjacency graph
    site_graph = nx.Graph()
    site_graph.graph["cell"]      = cell
    site_graph.graph["pbc"]       = surface_graph.graph["pbc"]
    site_graph.graph["adsorbate"] = reactant.smiles

    for i, s in enumerate(sites):
        # Shared surface atoms for adjacency: extract surf indices from conn
        if n_ads == 1:
            surf_set = s.conn_global
        else:
            surf_set = frozenset(sg for _, sg in s.conn_global)

        site_graph.add_node(
            i,
            position    = s.position.copy(),
            conn_global = s.conn_global,
            n_conn      = s.n_conn,
            site_type   = s.site_type,
            iso_class   = s.iso_class,
            _surf_set   = surf_set,   # internal, for edge construction
        )

    for na, nb in itertools.combinations(list(site_graph.nodes), 2):
        if site_graph.nodes[na]["_surf_set"] & site_graph.nodes[nb]["_surf_set"]:
            site_graph.add_edge(na, nb)

    if verbose:
        present_ids = sorted({s.iso_class for s in sites})
        print(f"\nSite classification")
        print("=" * 56)
        for cid in present_ids:
            members = [s for s in sites if s.iso_class == cid]
            e_str = ""
            if members[0].energy is not None:
                e_str = f"  E_ads={members[0].energy:.4f} eV  converged={members[0].converged}"
            print(f"  iso-class {cid:2d}  {_site_label(members[0].n_conn):8s}"
                  f"  {len(members):3d} sites{e_str}")
        print("=" * 56)
        print(f"  Total unique sites : {len(sites)}")
        print(f"  Adjacency edges    : {site_graph.number_of_edges()}")

    return sites, site_graph

