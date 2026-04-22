"""
autokmc.site
============
Find all unique adsorption sites on a surface for a given adsorbate.

Algorithm
---------
**Single-atom adsorbate**

1. Build an adsorbate-specific co-bonding graph on the surface atoms where
   two surface atoms are connected if an adsorbate atom with the given
   covalent radius could simultaneously bond to both
   (``d(s1,s2) ≤ bond_factor × (2·r_cov_ads + r_cov_s1 + r_cov_s2)``).
2. Enumerate all k-cliques (k = 1…k_max) of this graph.  Each clique
   directly defines a candidate bonded surface-atom set (top, bridge,
   hollow, …) without any spatial grid.
3. Geometrically optimise each unique connectivity (3 DOF: x, y, z).
4. Classify by ego-graph isomorphism.

**Multi-atom adsorbate**

For each atom *a* in the adsorbate molecule (each is tried as the *anchor*):

1. Build an anchor-specific co-bonding graph and enumerate k-cliques
   (same as single-atom) for the anchor contact set.
2. For each anchor clique, compute the rigid-body distance of every
   non-anchor adsorbate atom from the anchor (gas-phase geometry).
   Surface atoms within reach for some molecular orientation are found
   via the triangle-inequality bound; their co-bonding sub-cliques
   (including the empty set) are enumerated.
3. Form the Cartesian product of anchor clique × non-anchor sub-cliques
   to obtain all unique connectivity patterns
   ``frozenset{(ads_local_idx, surf_global_idx)}``.
4. Compute an initial rigid-body orientation with the Kabsch algorithm
   (aligns bonded non-anchor atoms toward their surface contacts).
5. Geometrically optimise each unique pattern as a 6-DOF rigid body.
6. Classify by ego-graph isomorphism on the combined adsorbate + surface
   ego-subgraph.

Typical usage
-------------
::

    carbon = build_reactant("[C]", calculator=EMT())
    sites, site_graph = find_adsorption_sites(surface_graph, carbon)

    co = build_reactant("[C-]#[O+]", calculator=EMT())
    sites, site_graph = find_adsorption_sites(surface_graph, co, k_max=3)
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


def _pbc_dist_scalar(p: np.ndarray, q: np.ndarray,
                     cell_inv: np.ndarray, cell: np.ndarray) -> float:
    """PBC-wrapped distance (x, y wrapped) between two points."""
    dv   = q - p
    frac = dv @ cell_inv
    frac[:2] -= np.round(frac[:2])
    return float(np.linalg.norm(frac @ cell))




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
# Single-atom site finding — graph / clique enumeration
# ---------------------------------------------------------------------------

def _find_sites_single_graph(
    surface_graph: nx.Graph,
    reactant: Reactant,
    bond_factor: float,
    k_max: int,
    verbose: bool,
) -> tuple:

    """Find single-atom adsorption sites by k-clique enumeration.

    Instead of casting a spatial probe grid, all cliques of size 1..k_max
    in the surface-atom subgraph are enumerated directly.  Each clique
    uniquely defines a connectivity set (bonded surface atoms) without any
    grid search:

    * size 1 → top sites      (single surface atom)
    * size 2 → bridge sites   (every edge in the surface graph)
    * size 3 → hollow sites   (every triangle)
    * size k → k-fold sites

    The surface graph edges encode which surface atom pairs are within
    covalent bonding distance, so any k atoms that form a k-clique are
    guaranteed to be mutually close enough to share a single adsorbate
    bonding partner above them.

    Returns the same tuple as :func:`_find_sites_single` so the rest of
    :func:`find_adsorption_sites` is unchanged.
    """
    cell     = surface_graph.graph["cell"]
    surf_nodes   = [(n, d) for n, d in surface_graph.nodes(data=True)
                    if d["type"] == "surface"]
    surf_indices = np.array([n for n, d in surf_nodes], dtype=int)
    surf_pos     = np.array([d["position"] for n, d in surf_nodes])
    surf_rcov    = np.array([d["covalent_radius"] for n, d in surf_nodes])
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}

    ads_data         = next(iter(reactant.graph.nodes(data=True)))[1]
    r_cov_ads        = float(ads_data["covalent_radius"])
    ads_elem         = ads_data["element"]
    bond_targets_all = r_cov_ads + surf_rcov

    surf_z_max = float(surf_pos[:, 2].max())
    d_min      = r_cov_ads * 0.5
    d_max      = bond_factor * (r_cov_ads + float(surf_rcov.max()))
    z_lo       = surf_z_max + d_min
    z_hi       = surf_z_max + d_max

    # ── Adsorbate-specific co-bonding graph ───────────────────────────────
    # Two surface atoms can share an adsorbate bond if the adsorbate's
    # bonding spheres (radius = bond_factor × (r_cov_ads + r_cov_si))
    # centred on each atom overlap — i.e. the inter-atom distance is at most
    # the sum of the two radii:
    #
    #   d(s1, s2) ≤ bond_factor × (r_cov_ads + r_cov_s1)
    #             + bond_factor × (r_cov_ads + r_cov_s2)
    #             = bond_factor × (2·r_cov_ads + r_cov_s1 + r_cov_s2)
    #
    # This is always >= the surface-surface bond cutoff (r_cov_s1 + r_cov_s2)
    # so it correctly captures bridge/hollow sites accessible only to large
    # adsorbates that the surface-surface graph would miss.
    co_bond_graph = nx.Graph()
    co_bond_graph.add_nodes_from(int(n) for n, _ in surf_nodes)
    cell_inv = np.linalg.inv(cell)
    for i in range(len(surf_indices)):
        for j in range(i + 1, len(surf_indices)):
            cutoff = bond_factor * (2.0 * r_cov_ads + float(surf_rcov[i]) + float(surf_rcov[j]))
            dv   = surf_pos[j] - surf_pos[i]
            frac = dv @ cell_inv
            frac[:2] -= np.round(frac[:2])
            dist = float(np.linalg.norm(frac @ cell))
            if dist <= cutoff:
                co_bond_graph.add_edge(int(surf_indices[i]), int(surf_indices[j]))

    # ── Enumerate all cliques of size 1 … k_max ───────────────────────────
    # nx.enumerate_all_cliques yields cliques in non-decreasing size order.
    unique_conns: list[frozenset]   = []
    unique_pts:   list[np.ndarray]  = []
    seen:         set[frozenset]    = set()

    for clique in nx.enumerate_all_cliques(co_bond_graph):
        if len(clique) > k_max:
            break                         # safe: yielded in size order
        key = frozenset(clique)
        if key in seen:
            continue
        seen.add(key)
        unique_conns.append(key)

        # Initial probe position: centroid of clique atoms + estimated height
        lids     = [surf_idx_to_local[g] for g in clique]
        centroid = surf_pos[lids].mean(axis=0).copy()
        height   = r_cov_ads + float(surf_rcov[lids].mean())
        centroid[2] = surf_z_max + height
        unique_pts.append(centroid)

    if verbose:
        by_size = {}
        for c in unique_conns:
            by_size.setdefault(len(c), 0)
            by_size[len(c)] += 1
        size_str = "  ".join(f"k={k}: {v}" for k, v in sorted(by_size.items()))
        print(f"  [single-graph] co-bond graph: {co_bond_graph.number_of_nodes()} nodes, "
              f"{co_bond_graph.number_of_edges()} edges  "
              f"(r_cov_ads={r_cov_ads:.3f} Å, bond_factor={bond_factor})")
        print(f"  [single-graph] clique candidates: {len(unique_pts)}  "
              f"({size_str})")

    # ── Isomorphism classification (identical to grid path) ───────────────
    node_match  = isomorphism.categorical_node_match("element", "X")
    class_reps: list[nx.Graph] = []
    class_keys: list[tuple]    = []
    iso_ids:    list[int]      = []
    for conn in unique_conns:
        ego  = _build_ego_single(surface_graph, conn, ads_elem)
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

    if verbose:
        print(f"  [single-graph] iso-classes: {len(class_reps)}")

    # ── Geometric optimisation — one representative per iso-class ─────────
    iso_ids_arr   = np.array(iso_ids)
    n_classes     = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0
    class_rep_idx = {cid: int(np.where(iso_ids_arr == cid)[0][0])
                     for cid in range(n_classes)}

    class_opt_pts: dict[int, np.ndarray] = {}
    for cid, rep_i in class_rep_idx.items():
        pt0  = unique_pts[rep_i]
        conn = unique_conns[rep_i]
        lids      = [surf_idx_to_local[g] for g in sorted(conn)]
        conn_pos  = surf_pos[lids]
        bond_tgts = bond_targets_all[lids]
        nb_lids   = [k for k in range(len(surf_indices)) if k not in set(lids)]
        nonbond   = surf_pos[nb_lids] if nb_lids else np.empty((0, 3))
        class_opt_pts[cid] = _opt_single(pt0, conn_pos, bond_tgts, nonbond,
                                         cell, z_lo, z_hi)

    opt_pts = [class_opt_pts[cid] for cid in iso_ids]

    return (opt_pts, unique_conns, iso_ids, class_reps,
            surf_rcov, surf_idx_to_local, bond_targets_all)


# ---------------------------------------------------------------------------
# Multi-atom site finding — graph / clique enumeration
# ---------------------------------------------------------------------------

def _find_sites_multi_graph(
    surface_graph: nx.Graph,
    reactant: Reactant,
    bond_factor: float,
    k_max: int,
    verbose: bool,
) -> tuple:
    """Find multi-atom adsorption sites by graph-based enumeration.

    Algorithm
    ---------
    For each adsorbate atom tried as the *anchor*:

    1. Build an anchor-specific co-bonding graph on the surface atoms and
       enumerate all cliques of size 1..k_max — exactly as in the single-atom
       path — to get candidate anchor contact sets.

    2. For each anchor clique (initial anchor position = centroid + height):

       a. For every non-anchor adsorbate atom ``a_j``, compute its
          rigid-body distance from the anchor in the gas-phase geometry
          (``d_mol_j = ||pos_j - pos_anchor||``).  Surface atoms within
          ``d_mol_j + bond_factor*(r_cov_j + r_cov_s)`` of the anchor form
          ``a_j``'s *reachable set* ``R_j`` — the atoms that *could* bond to
          ``a_j`` for some molecular orientation.

       b. Build a co-bonding graph on ``R_j`` (same adsorbate-aware edge
          criterion as the anchor graph) and enumerate sub-cliques of size
          0..k_max.  The empty set (no surface bond) is always included.

       c. Take the Cartesian product of sub-clique options for all
          non-anchor atoms.  Each combination together with the anchor
          clique defines a unique connectivity pattern
          ``frozenset{(ads_local_idx, surf_global_idx)}``.

    3. Deduplicate by connectivity frozenset.

    4. Compute an initial rigid-body orientation via the *orthogonal
       Procrustes* (Kabsch) algorithm: align non-anchor atoms that have
       non-empty surface contacts toward the centroid of those contacts.
       Atoms with no surface contact do not constrain the rotation.

    5. Isomorphism classification and 6-DOF geometric optimisation —
       identical to the existing multi-atom path.

    Returns the same tuple as :func:`_find_sites_multi`.
    """
    cell     = surface_graph.graph["cell"]
    cell_inv = np.linalg.inv(cell)

    surf_nodes   = [(n, d) for n, d in surface_graph.nodes(data=True)
                    if d["type"] == "surface"]
    surf_indices = np.array([n for n, d in surf_nodes], dtype=int)
    surf_pos     = np.array([d["position"] for n, d in surf_nodes])
    surf_rcov    = np.array([d["covalent_radius"] for n, d in surf_nodes])
    surf_idx_to_local = {int(surf_indices[k]): k for k in range(len(surf_indices))}

    ads_nodes = list(reactant.graph.nodes(data=True))
    ads_pos   = np.array([d["position"] for _, d in ads_nodes])
    ads_rcov  = np.array([d["covalent_radius"] for _, d in ads_nodes])
    N_ads     = len(ads_pos)

    surf_z_max   = float(surf_pos[:, 2].max())
    d_min_global = float(ads_rcov.min()) * 0.5
    d_max_global = bond_factor * (float(ads_rcov.max()) + float(surf_rcov.max()))
    z_lo         = surf_z_max + d_min_global
    z_hi         = surf_z_max + d_max_global

    # ── Helper: co-bonding graph for adsorbate atom with r_cov_a ─────────
    def _make_co_bond_graph(r_cov_a: float, local_ids: list) -> nx.Graph:
        """Adsorbate-aware co-bonding graph on a subset of surface atoms."""
        g = nx.Graph()
        g.add_nodes_from(int(surf_indices[li]) for li in local_ids)
        for ii in range(len(local_ids)):
            for jj in range(ii + 1, len(local_ids)):
                li, lj = local_ids[ii], local_ids[jj]
                cutoff = bond_factor * (2.0 * r_cov_a
                                        + float(surf_rcov[li])
                                        + float(surf_rcov[lj]))
                dist = _pbc_dist_scalar(surf_pos[li], surf_pos[lj], cell_inv, cell)
                if dist <= cutoff:
                    g.add_edge(int(surf_indices[li]), int(surf_indices[lj]))
        return g

    # ── Helper: Kabsch / orthogonal Procrustes rotation ──────────────────
    def _kabsch(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Rotation matrix R minimising ||R @ src[i] - tgt[i]||^2 (sum)."""
        H = src.T @ tgt
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T  # (3, 3)

    # ── Main enumeration ──────────────────────────────────────────────────
    seen:              dict[tuple, int]    = {}
    unique_conns:      list[frozenset]     = []
    unique_anchors:    list[int]           = []
    unique_anchor_pos: list[np.ndarray]   = []
    unique_rotvec:     list[np.ndarray]   = []

    all_local_ids = list(range(len(surf_indices)))

    # ── Anchor deduplication via WL colour refinement ─────────────────────
    # Atoms that are automorphically equivalent in the adsorbate graph will
    # always produce the same set of surface connectivity patterns regardless
    # of which one is chosen as the anchor.  We identify these equivalence
    # classes with Weisfeiler-Lehman (WL) label propagation and only try one
    # representative anchor per class, avoiding redundant enumeration.
    #
    # Number of WL iterations = graph diameter (longest shortest path between
    # any two nodes).  After `diameter` rounds each node's label encodes the
    # structure of its entire connected component, so no further refinement
    # is possible.  For a single-atom adsorbate the diameter is 0; we use
    # at least 1 iteration so the element label is always set.
    #
    # WL iteration: label(v) = hash(element(v), sorted(label(u) for u in N(v)))
    def _wl_anchor_representatives(graph: nx.Graph) -> list[int]:
        """Return one representative anchor index per WL equivalence class."""
        n_iter = nx.diameter(graph) if graph.number_of_nodes() > 1 else 1
        labels = {n: d["element"] for n, d in graph.nodes(data=True)}
        for _ in range(n_iter):
            new_labels = {}
            for n in graph.nodes():
                nbr_labels = tuple(sorted(labels[u] for u in graph.neighbors(n)))
                new_labels[n] = (labels[n], nbr_labels)
            labels = {n: str(v) for n, v in new_labels.items()}
        # One representative per unique label (preserving node order)
        seen_labels: dict[str, int] = {}
        for n in graph.nodes():
            lbl = labels[n]
            if lbl not in seen_labels:
                seen_labels[lbl] = n
        return list(seen_labels.values())

    anchor_representatives = _wl_anchor_representatives(reactant.graph)

    if verbose:
        print(f"  [multi-graph] adsorbate has {N_ads} atoms, "
              f"{len(anchor_representatives)} distinct anchor class(es): "
              f"indices {anchor_representatives}")

    for anchor_idx in anchor_representatives:
        r_cov_a = float(ads_rcov[anchor_idx])
        non_anchor_idxs = [j for j in range(N_ads) if j != anchor_idx]

        # ── 1. Anchor co-bonding graph + clique enumeration ───────────────
        anchor_cbg = _make_co_bond_graph(r_cov_a, all_local_ids)

        for anchor_clique in nx.enumerate_all_cliques(anchor_cbg):
            if len(anchor_clique) > k_max:
                break

            # Initial anchor position: centroid of clique atoms + height
            lids_anchor = [surf_idx_to_local[g] for g in anchor_clique]
            pos_anchor  = surf_pos[lids_anchor].mean(axis=0).copy()
            h_anchor    = r_cov_a + float(surf_rcov[lids_anchor].mean())
            pos_anchor[2] = surf_z_max + h_anchor

            # ── 2a–b. Non-anchor reachable sets + sub-clique options ──────
            # For each non-anchor adsorbate atom collect a list of possible
            # surface contact frozensets (including the empty set).
            non_anchor_options: list[list] = []   # one list per non-anchor atom
            for j in non_anchor_idxs:
                r_cov_j = float(ads_rcov[j])
                # Rigid-body distance from anchor to a_j in gas-phase geometry
                d_mol_j = float(np.linalg.norm(ads_pos[j] - ads_pos[anchor_idx]))

                # Surface atoms reachable by a_j (triangle-inequality upper bound)
                reach_lids = [
                    li for li in all_local_ids
                    if _pbc_dist_scalar(pos_anchor, surf_pos[li], cell_inv, cell)
                       <= d_mol_j + bond_factor * (r_cov_j + float(surf_rcov[li]))
                ]

                # Always include "no surface bond" for this atom
                options_j: list[frozenset] = [frozenset()]
                if reach_lids:
                    cbg_j = _make_co_bond_graph(r_cov_j, reach_lids)
                    for sub_clique in nx.enumerate_all_cliques(cbg_j):
                        if len(sub_clique) > k_max:
                            break
                        options_j.append(frozenset(sub_clique))

                non_anchor_options.append(options_j)

            # ── 2c. Cartesian product → unique connectivity patterns ───────
            for combo in itertools.product(*non_anchor_options):
                # Full connectivity: (ads_local_idx, surf_global_idx) pairs
                conn_list = (
                    [(anchor_idx, int(s)) for s in sorted(anchor_clique)]
                    + [(j, int(s))
                       for j, clique_j in zip(non_anchor_idxs, combo)
                       for s in sorted(clique_j)]
                )
                conn_key = tuple(sorted(conn_list))
                if conn_key in seen:
                    continue
                seen[conn_key] = len(unique_conns)

                unique_conns.append(frozenset(conn_key))
                unique_anchors.append(anchor_idx)
                unique_anchor_pos.append(pos_anchor.copy())

                # ── 4. Kabsch initial rotation ────────────────────────────
                # Align non-anchor atoms that have surface contacts toward the
                # centroid of those contacts.
                rel_pos = ads_pos - ads_pos[anchor_idx]  # (N_ads, 3) molecular frame
                src_pts, tgt_pts = [], []
                for j, clique_j in zip(non_anchor_idxs, combo):
                    if not clique_j:
                        continue
                    lids_j = [surf_idx_to_local[g] for g in clique_j]
                    tgt_j  = surf_pos[lids_j].mean(axis=0) - pos_anchor
                    # Ensure the target direction has a positive z component
                    # (the non-anchor should point up, not into the slab)
                    if tgt_j[2] < 0:
                        tgt_j[2] = abs(tgt_j[2])
                    src_pts.append(rel_pos[j])
                    tgt_pts.append(tgt_j)

                if src_pts:
                    R_init  = _kabsch(np.array(src_pts), np.array(tgt_pts))
                    rotvec0 = Rotation.from_matrix(R_init).as_rotvec()
                else:
                    rotvec0 = np.zeros(3)

                unique_rotvec.append(rotvec0)

    if verbose:
        print(f"  [multi-graph] unique connectivity patterns: {len(unique_conns)}")

    # ── Isomorphism classification ────────────────────────────────────────
    node_match = isomorphism.categorical_node_match("element", "X")
    class_reps:  list[nx.Graph] = []
    class_keys:  list[tuple]    = []
    iso_ids:     list[int]      = []
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

    if verbose:
        print(f"  [multi-graph] iso-classes: {len(class_reps)}")

    # ── Geometric optimisation (6 DOF) — one representative per iso-class ─
    iso_ids_arr   = np.array(iso_ids)
    n_classes     = int(iso_ids_arr.max()) + 1 if len(iso_ids_arr) else 0
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
# Relaxation helpers
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
    k_max: int = 4,
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
    k_max : int
        Maximum clique size to enumerate.  Maps directly to site
        coordination: 1 = top, 2 = bridge, 3 = hollow, 4 = 4-fold, etc.
        Default 4.
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
        ``class_XX_final.extxyz``, ``.traj`` and ``.log`` files here.
        Default ``"debug"``.  Set to ``None`` to suppress all file output.

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
              f"({'single' if n_ads == 1 else 'multi'}-atom path,  k_max={k_max})")

    if n_ads == 1:
        (opt_pts, unique_conns, iso_ids,
         class_reps, surf_rcov, surf_idx_to_local,
         bond_targets_all) = _find_sites_single_graph(
            surface_graph, reactant, bond_factor, k_max, verbose)

        # ── optional calculator relaxation (one rep per iso-class) ───────────
        iso_results: dict[int, dict] = {}
        if calculator is not None and slab is not None:
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

        sites: list[AdsorptionSite] = []
        for opt, conn, iso_cid in zip(opt_pts, unique_conns, iso_ids):
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
        opt_positions, unique_conns, iso_ids, class_reps = _find_sites_multi_graph(
            surface_graph, reactant, bond_factor, k_max, verbose)

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
                position    = opt_ads,
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

