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
# Single-atom site finding
# ---------------------------------------------------------------------------

def _find_sites_single(surface_graph: nx.Graph, reactant: Reactant,
                        bond_factor: float, grid_spacing: float,
                        verbose: bool) -> tuple[list, list, list]:
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

    # Build grid
    a_len = np.linalg.norm(cell[0]); b_len = np.linalg.norm(cell[1])
    fa = np.linspace(0, 1, max(1, int(np.ceil(a_len / grid_spacing))), endpoint=False)
    fb = np.linspace(0, 1, max(1, int(np.ceil(b_len / grid_spacing))), endpoint=False)
    gz = np.arange(surf_z_max + d_min, surf_z_max + d_max + grid_spacing, grid_spacing)
    gfa, gfb, ggz = np.meshgrid(fa, fb, gz, indexing="ij")
    probe_xy  = gfa.ravel()[:, None] * cell[0, :2] + gfb.ravel()[:, None] * cell[1, :2]
    probe_pts = np.hstack([probe_xy, ggz.ravel()[:, None]])

    # Distance matrix
    cell_inv = np.linalg.inv(cell)
    dv = surf_pos[None] - probe_pts[:, None]
    dfrac = dv @ cell_inv
    dfrac[:, :, :2] -= np.round(dfrac[:, :, :2])
    d_mat = np.sqrt(((dfrac @ cell) ** 2).sum(axis=2))

    keep = (d_mat.min(axis=1) >= d_min) & (d_mat.min(axis=1) <= d_max)
    probe_pts = probe_pts[keep]; d_mat = d_mat[keep]

    if verbose:
        print(f"  [single] probe pts after filter: {len(probe_pts)}")

    # Deduplicate
    seen: dict = {}
    unique_pts: list[np.ndarray] = []
    unique_conns: list[frozenset] = []
    for i in range(len(probe_pts)):
        bonded = np.where(d_mat[i] <= bond_cutoffs)[0]
        if not len(bonded):
            continue
        key = frozenset(int(surf_indices[k]) for k in bonded)
        if key not in seen:
            seen[key] = len(unique_pts)
            unique_pts.append(probe_pts[i].copy())
            unique_conns.append(key)

    if verbose:
        print(f"  [single] unique connectivities: {len(unique_pts)}")

    # Isomorphism
    node_match = isomorphism.categorical_node_match("element", "X")
    class_reps: list[nx.Graph] = []
    iso_ids: list[int] = []
    for conn in unique_conns:
        ego = _build_ego_single(surface_graph, conn, ads_elem)
        assigned = False
        for cid, rep in enumerate(class_reps):
            if isomorphism.GraphMatcher(ego, rep, node_match=node_match).is_isomorphic():
                iso_ids.append(cid); assigned = True; break
        if not assigned:
            class_reps.append(ego); iso_ids.append(len(class_reps) - 1)

    # Geometric optimisation
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}
    z_lo = float(surf_z_max) + d_min
    z_hi = float(surf_z_max) + d_max
    opt_pts: list[np.ndarray] = []
    for pt0, conn in zip(unique_pts, unique_conns):
        lids = [surf_idx_to_local[g] for g in sorted(conn)]
        conn_pos   = surf_pos[lids]
        bond_tgts  = bond_targets_all[lids]
        nb_lids    = [k for k in range(len(surf_indices)) if k not in lids]
        nonbond    = surf_pos[nb_lids] if nb_lids else np.empty((0, 3))
        opt_pts.append(_opt_single(pt0, conn_pos, bond_tgts, nonbond,
                                   cell, z_lo, z_hi))

    return opt_pts, unique_conns, iso_ids, class_reps, surf_rcov, surf_idx_to_local, bond_targets_all


# ---------------------------------------------------------------------------
# Multi-atom site finding
# ---------------------------------------------------------------------------

def _find_sites_multi(surface_graph: nx.Graph, reactant: Reactant,
                       bond_factor: float, grid_spacing: float,
                       n_orientations: int, verbose: bool) -> tuple:
    """Returns (opt_positions_list, unique_conns, iso_ids, class_reps)."""
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

    surf_z_max = surf_pos[:, 2].max()

    # Pre-compute per-(ads_atom, surf_atom) bonding cutoffs: (N_ads, S)
    cutoff_mat = bond_factor * (ads_rcov[:, None] + surf_rcov[None, :])  # (N_ads, S)

    # z-window for anchor atoms
    d_min_global = ads_rcov.min() * 0.5
    d_max_global = bond_factor * (ads_rcov.max() + surf_rcov.max())
    z_lo = float(surf_z_max) + d_min_global
    z_hi = float(surf_z_max) + d_max_global

    # Sample SO(3) rotations once
    rot_mats = _sample_so3(n_orientations)  # (n_orient, 3, 3)

    # Grid for anchor placement
    a_len = np.linalg.norm(cell[0]); b_len = np.linalg.norm(cell[1])
    fa = np.linspace(0, 1, max(1, int(np.ceil(a_len / grid_spacing))), endpoint=False)
    fb = np.linspace(0, 1, max(1, int(np.ceil(b_len / grid_spacing))), endpoint=False)
    gz = np.arange(surf_z_max + d_min_global, surf_z_max + d_max_global + grid_spacing,
                   grid_spacing)
    gfa, gfb, ggz = np.meshgrid(fa, fb, gz, indexing="ij")
    grid_xy  = gfa.ravel()[:, None] * cell[0, :2] + gfb.ravel()[:, None] * cell[1, :2]
    grid_pts = np.hstack([grid_xy, ggz.ravel()[:, None]])   # (G, 3)

    if verbose:
        print(f"  [multi] grid pts: {len(grid_pts)},  "
              f"orientations: {n_orientations},  anchors: {N_ads}")

    seen: dict = {}
    unique_conns:  list[frozenset] = []
    unique_anchors: list[int]      = []   # which ads atom was anchor
    unique_anchor_pos: list[np.ndarray] = []
    unique_rotvec:    list[np.ndarray]  = []

    # For each anchor atom in the adsorbate
    for anchor_idx in range(N_ads):
        # Relative positions of all ads atoms w.r.t. anchor
        rel_pos = ads_pos - ads_pos[anchor_idx]   # (N_ads, 3)

        for gpt in grid_pts:
            for rot in rot_mats:
                # Place adsorbate: anchor at gpt, rest rotated
                placed = gpt + (rot @ rel_pos.T).T   # (N_ads, 3)

                # For each ads atom, compute PBC distances to all surf atoms
                conn_set: set[tuple[int, int]] = set()
                for ak in range(N_ads):
                    # PBC wrap
                    dv   = surf_pos - placed[ak]
                    frac = dv @ cell_inv
                    frac[:, :2] -= np.round(frac[:, :2])
                    d_ak = np.sqrt(((frac @ cell) ** 2).sum(axis=1))   # (S,)
                    bonded_local = np.where(d_ak <= cutoff_mat[ak])[0]
                    for bl in bonded_local:
                        conn_set.add((int(ak), int(surf_indices[bl])))

                if not conn_set:
                    continue

                key = frozenset(conn_set)
                if key not in seen:
                    seen[key] = len(unique_conns)
                    unique_conns.append(key)
                    unique_anchors.append(anchor_idx)
                    unique_anchor_pos.append(gpt.copy())
                    # Initial rotation as rotvec
                    unique_rotvec.append(Rotation.from_matrix(rot).as_rotvec())

    if verbose:
        print(f"  [multi] unique connectivities: {len(unique_conns)}")

    # Isomorphism classification
    node_match = isomorphism.categorical_node_match("element", "X")
    class_reps: list[nx.Graph] = []
    iso_ids: list[int] = []
    for conn in unique_conns:
        ego = _build_ego_multi(surface_graph, reactant, conn)
        assigned = False
        for cid, rep in enumerate(class_reps):
            if isomorphism.GraphMatcher(ego, rep, node_match=node_match).is_isomorphic():
                iso_ids.append(cid); assigned = True; break
        if not assigned:
            class_reps.append(ego); iso_ids.append(len(class_reps) - 1)

    # Geometric optimisation (6 DOF per site)
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}
    opt_positions: list[np.ndarray] = []
    for conn, anchor_idx, anchor0, rv0 in zip(
            unique_conns, unique_anchors, unique_anchor_pos, unique_rotvec):
        rel_pos = ads_pos - ads_pos[anchor_idx]
        opt_ads = _opt_multi(anchor0, rv0, anchor_idx, rel_pos, ads_rcov,
                             conn, surf_pos, surf_rcov, surf_idx_to_local,
                             cell, z_lo, z_hi)
        opt_positions.append(opt_ads)   # (N_ads, 3)

    return opt_positions, unique_conns, iso_ids, class_reps


# ---------------------------------------------------------------------------
# EMT relaxation helpers
# ---------------------------------------------------------------------------

def _get_frozen_indices(slab, n_freeze_layers: int) -> list[int]:
    """Return atom indices belonging to the bottom *n_freeze_layers* layers.

    Layers are identified by clustering z-coordinates (tolerance = NN spacing / 4).
    """
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

    # Per-atom covalent radii of surface atoms — used via bond_factor per-pair in the check below
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
            cutoff = bond_factor * (ads_rcov[ak] + ads_rcov.mean())
            for k in np.where(d <= cutoff)[0]:
                conn_actual.add((int(ak), int(surface_indices[k])))

        conn_intended = {(ak, sg) for ak, sg in conn_pairs}
        valid = frozenset(conn_actual) == frozenset(conn_intended)

        # ── 2. Adsorbate internal-connectivity check ──────────────────────
        if valid and n_ads > 1:
            relaxed_bonds: set[tuple[int, int]] = set()
            for ai in range(n_ads):
                for aj in range(ai + 1, n_ads):
                    d = float(np.linalg.norm(ads_final[ai] - ads_final[aj]))
                    cutoff = bond_factor * (ads_rcov[ai] + ads_rcov[aj])
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
    n_orientations: int = 200,
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
    n_orientations : int
        Number of random SO(3) orientations for multi-atom adsorbates.
        Default 200.
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

    if verbose:
        print(f"Finding sites for '{reactant.smiles}'  "
              f"({'single' if n_ads == 1 else 'multi'}-atom path)")

    if n_ads == 1:
        (opt_pts, unique_conns, iso_ids,
         class_reps, surf_rcov, surf_idx_to_local,
         bond_targets_all) = _find_sites_single(
            surface_graph, reactant, bond_factor, grid_spacing, verbose)

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
            n_orientations, verbose)

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

