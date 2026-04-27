"""
autokmc.find_anchors
====================
Enumerate, classify and geometrically optimise the adsorption anchor sites on
a surface graph.

An *anchor site* is a set of surface atoms (a k-clique in the co-bonding graph)
that an adsorbate atom of a given element can simultaneously bond to.  k=1 is a
top site, k=2 a bridge, k=3 a hollow, etc.

The full pipeline (one external entry point):

.. code-block:: python

    anchor_sites = find_anchor_sites(G, "C", verbose=True)

This internally does:

1. **Build the co-bonding graph** (:func:`_build_co_bond_graph`).
   Two surface atoms are connected when an adsorbate of the given covalent
   radius can simultaneously bind both::

       d(i, j) ≤ co_factor * (2·r_cov_ads + r_cov_i + r_cov_j)

2. **Enumerate all cliques** of size 1 … k_max.
   For nanoparticles, cliques whose centroid lies inside the convex hull are
   dropped as wrap-around artefacts.

3. **Reduce by graph isomorphism** (:func:`_reduce_by_isomorphism`).
   Two cliques belong to the same :class:`AnchorSite` when their n-shell
   ego-subgraphs (BFS around the clique atoms, skipping invisible node types)
   are graph-isomorphic with element-label matching.

4. **Optimise the representative position** (:func:`_optimise_position`).
   Calculator-free L-BFGS-B / SLSQP minimisation of::

       E = Σ_i (|p − pos_i| − d_ideal_i)²  +  w · Σ_j 1/(d_j² + ε)

   where i iterates over bonded clique atoms and j over nearby non-bonded
   surface atoms.

5. **Propagate to every member** via Kabsch ego-alignment
   (:func:`_kabsch_align_ego`).  The representative's optimised position is
   rigidly transformed onto each member's local frame; this is much cheaper
   than re-optimising every clique and guarantees geometric consistency.

6. **Materialise invisible anchor nodes** on *G*.
   One node per raw clique (``type="anchor"``), linked to its surface atoms by
   ``anchor_bond=True`` edges.  These nodes are *invisible* to the BFS inside
   :func:`_build_ego_graph` so they never short-circuit the ego-subgraph used
   for iso-class comparison.

Storage
-------
Results are stored in ``G.graph["anchor_sites"][element]``: a flat list of
:class:`AnchorSite` objects, one per iso-class.  The equivalent raw-clique
mapping is at ``G.graph["raw_cliques"][element]``.

Public API
----------
* :class:`AnchorSite`           — one iso-class of k-fold adsorption sites.
* :func:`find_anchor_sites`     — full pipeline; returns list[AnchorSite].
* :func:`k_max_for_element`     — fast k_max query (no site enumeration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from scipy.spatial import cKDTree

from ase.data import (
    covalent_radii as _ASE_RCOV,
    atomic_numbers as _ASE_AN,
)

from autokmc.logging_utils import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuneable defaults — single source of truth in :mod:`autokmc.constants`.
# Re-exported here so callers can ``from autokmc.find_anchors import CO_FACTOR``
# and so the function-default kwargs stay readable.  Mutating these
# module-level values does **not** affect existing function-default kwargs
# (those snapshot at def-time) — pass an explicit kwarg to override.
# ---------------------------------------------------------------------------

from autokmc.constants import (
    CO_FACTOR,
    OPT_FACTOR,
    REPULSION_WEIGHT,
    SITE_REPULSION_CUTOFF as REPULSION_CUTOFF,
    N_SHELLS_DEFAULT as N_SHELLS,
    HULL_TOL,
    KABSCH_MAX_MAPPINGS,
)

# ---------------------------------------------------------------------------
# Module-level lookup tables
# ---------------------------------------------------------------------------

#: Human-readable coordination labels used in verbose / log output.
COORD_LABELS: dict[int, str] = {1: "top", 2: "bridge", 3: "hollow"}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class AnchorSite:
    """One isomorphism class of k-fold surface anchor sites.

    Attributes
    ----------
    element : str
        Chemical symbol of the adsorbate atom.
    k : int
        Coordination number: 1 = top, 2 = bridge, 3 = hollow, …
    iso_class : int
        0-based index within the k group.
    n_shells : int
        Ego-graph depth used to define this iso-class.
    representative : frozenset[int]
        Surface-atom node ids of the *representative* clique (the first
        member encountered; used to seed Kabsch propagation).
    members : list[frozenset[int]]
        All cliques that belong to this iso-class (representative first).
    centroid : np.ndarray, shape (3,)
        MIC-aware mean position of the representative clique atoms.
    ego_graph : nx.Graph
        The n-shell ego-subgraph of the representative used for matching.
    position : np.ndarray | None
        Optimised adsorbate position (Å) for the *representative* clique;
        ``None`` until :func:`find_anchor_sites` completes.
    node_ids : list[int]
        Anchor node ids materialised on *G* – one per member, parallel to
        :attr:`members`.  Index with the same offset to get the graph node
        for a given member.
    """
    element      : str
    k            : int
    iso_class    : int
    n_shells     : int
    representative: frozenset[int]
    members      : list[frozenset[int]] = field(default_factory=list)
    centroid     : Any                  = None
    ego_graph    : Any                  = None
    position     : Any                  = None
    node_ids     : list[int]            = field(default_factory=list)


# ---------------------------------------------------------------------------
# MIC / cell helpers
# ---------------------------------------------------------------------------

def _get_cell(G: nx.Graph) -> tuple[np.ndarray, np.ndarray | None,
                                    np.ndarray, bool]:
    """Return ``(cell, cell_inv_or_None, pbc, use_mic)`` from *G*."""
    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)
    use_mic = bool(pbc.any())
    cell_inv: np.ndarray | None = None
    if use_mic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            use_mic = False
    return cell, cell_inv, pbc, use_mic


def _circular_centroid(
    positions: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> np.ndarray:
    """Circular-mean MIC-robust centroid of *positions* (N×3)."""
    if not use_mic or cell_inv is None or not pbc.any():
        return positions.mean(axis=0)
    frac = positions @ cell_inv
    out = np.empty(3)
    for ax in range(3):
        if pbc[ax]:
            theta = 2.0 * np.pi * frac[:, ax]
            ang = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
            if ang < 0.0:
                ang += 2.0 * np.pi
            out[ax] = ang / (2.0 * np.pi)
        else:
            out[ax] = frac[:, ax].mean()
    return out @ cell


def _clique_centroid(
    G: nx.Graph,
    clique: frozenset,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> np.ndarray:
    positions = np.array([G.nodes[n]["position"] for n in clique], dtype=float)
    return _circular_centroid(positions, cell, cell_inv, pbc, use_mic)


def _mic_distances(
    p: np.ndarray,
    ref: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """MIC distances from point *p* to each row of *ref* (N×3)."""
    dv = p[np.newaxis, :] - ref
    frac = dv @ cell_inv
    for ax in range(3):
        if pbc[ax]:
            frac[:, ax] -= np.round(frac[:, ax])
    return np.linalg.norm(frac @ cell, axis=1)


def _mic_unwrap(
    G: nx.Graph,
    nodes,
    anchor: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> dict[int, np.ndarray]:
    """Return {node: position} with positions unwrapped into the same image as *anchor*."""
    out: dict[int, np.ndarray] = {}
    for n in nodes:
        p = np.asarray(G.nodes[n]["position"], dtype=float)
        if use_mic and cell_inv is not None:
            dv = p - anchor
            frac = dv @ cell_inv
            for ax in range(3):
                if pbc[ax]:
                    frac[ax] -= np.round(frac[ax])
            p = anchor + frac @ cell
        out[int(n)] = p
    return out


# ---------------------------------------------------------------------------
# Ego-graph (BFS ignoring invisible nodes)
# ---------------------------------------------------------------------------

def _build_ego_graph(
    G: nx.Graph,
    clique: frozenset,
    n_shells: int,
) -> nx.Graph:
    """Return the n-shell BFS ego-subgraph of *G* rooted at *clique*.

    **Invisible** node types (skipped during BFS, excluded from the returned
    subgraph):

    * ``type == "anchor"``          — anchor bookkeeping nodes; would
      short-circuit between every clique that touches the same surface atom.
    * ``type == "adsorbate"`` **and** ``occupied == False`` — unoccupied
      adsorbate placeholder; becomes visible once ``occupied=True``.
    """
    frontier: set = set(clique)
    visited:  set = set(clique)
    for _ in range(n_shells):
        nxt: set = set()
        for n in frontier:
            for nb in G.neighbors(n):
                d = G.nodes[nb]
                t = d.get("type")
                if t == "anchor":
                    continue
                if t == "adsorbate" and not d.get("occupied", False):
                    continue
                nxt.add(nb)
        frontier = nxt - visited
        visited |= frontier
    return G.subgraph(visited).copy()


# ---------------------------------------------------------------------------
# Structural fingerprint (cheap pre-filter before full isomorphism test)
# ---------------------------------------------------------------------------

def _fingerprint(g: nx.Graph) -> tuple:
    """Cheap graph fingerprint — unequal keys → guaranteed non-isomorphic."""
    elem_deg = tuple(sorted(
        (d["element"], g.degree(n))
        for n, d in g.nodes(data=True)
    ))
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(sorted(g.degree(n) for n in g.nodes())),
        elem_deg,
    )


# ---------------------------------------------------------------------------
# Co-bonding graph
# ---------------------------------------------------------------------------

def _build_co_bond_graph(
    G: nx.Graph,
    r_cov_ads: float,
    co_factor: float = CO_FACTOR,
) -> nx.Graph:
    """Build the adsorbate co-bonding graph on surface atoms.

    Two surface atoms i, j are connected iff::

        d(i, j) ≤ co_factor × (2·r_cov_ads + r_cov_i + r_cov_j)

    Uses a :class:`scipy.spatial.cKDTree` (orthogonal cells use native
    ``boxsize``; non-orthogonal cells tile ±1 periodic images).
    """
    surf_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d["type"] == "surface"]
    if not surf_nodes:
        return nx.Graph()

    ids      = [n for n, _ in surf_nodes]
    pos      = np.array([d["position"] for _, d in surf_nodes], dtype=float)
    r_cov_s  = np.array([d["covalent_radius"] for _, d in surf_nodes], dtype=float)

    cell, cell_inv, pbc, use_mic = _get_cell(G)
    r_max = float(r_cov_s.max())
    r_query = co_factor * (2.0 * r_cov_ads + 2.0 * r_max)

    cbg = nx.Graph()
    cbg.add_nodes_from((n, dict(G.nodes[n])) for n in ids)

    is_ortho = use_mic and np.allclose(cell - np.diag(np.diag(cell)), 0.0)

    if is_ortho:
        boxsize = np.where(pbc, np.diag(cell), 0.0)
        pos_kd  = pos.copy()
        for ax in range(3):
            if pbc[ax] and boxsize[ax] > 0:
                pos_kd[:, ax] = np.mod(pos_kd[:, ax], boxsize[ax])
        tree  = cKDTree(pos_kd, boxsize=np.where(boxsize > 0, boxsize, 0.0))
        pairs = tree.query_pairs(r=r_query, output_type="ndarray")

    elif use_mic:
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx and not pbc[0]: continue
                    if dy and not pbc[1]: continue
                    if dz and not pbc[2]: continue
                    offsets.append(np.array([dx, dy, dz], dtype=int))
        tiled_pos: list[np.ndarray] = []
        tiled_idx: list[int]        = []
        for off in offsets:
            tiled_pos.append(pos + off @ cell)
            tiled_idx.extend(range(len(pos)))
        tree = cKDTree(np.concatenate(tiled_pos))
        raw: set[tuple[int, int]] = set()
        for i, p in enumerate(pos):
            for hit in tree.query_ball_point(p, r=r_query):
                j = tiled_idx[hit]
                if j == i:
                    continue
                raw.add((min(i, j), max(i, j)))
        pairs = (np.array(sorted(raw), dtype=int) if raw
                 else np.empty((0, 2), dtype=int))

    else:
        tree  = cKDTree(pos)
        pairs = tree.query_pairs(r=r_query, output_type="ndarray")

    if len(pairs) == 0:
        return cbg

    # Apply exact per-pair cutoff (kd-tree used a conservative upper bound).
    p_i = pos[pairs[:, 0]]
    p_j = pos[pairs[:, 1]]
    dv  = p_j - p_i
    if use_mic and cell_inv is not None:
        frac = dv @ cell_inv
        for ax in range(3):
            if pbc[ax]:
                frac[:, ax] -= np.round(frac[:, ax])
        dv = frac @ cell
    dist    = np.linalg.norm(dv, axis=1)
    cutoff  = co_factor * (2.0 * r_cov_ads
                           + r_cov_s[pairs[:, 0]]
                           + r_cov_s[pairs[:, 1]])
    for ii, jj in pairs[dist <= cutoff]:
        cbg.add_edge(ids[int(ii)], ids[int(jj)])

    return cbg


# ---------------------------------------------------------------------------
# Kabsch alignment
# ---------------------------------------------------------------------------

def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal proper-rotation alignment mapping rows of *src* → *dst*.

    Returns ``(R, t)`` such that ``src @ R.T + t ≈ dst``.
    """
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    H  = (src - sc).T @ (dst - dc)
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dc - R @ sc
    return R, t


def _kabsch_align_ego(
    G: nx.Graph,
    rep_clique,
    mem_clique,
    n_shells: int,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
    rmsd_tol: float = 1e-4,
    max_mappings: int = KABSCH_MAX_MAPPINGS,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rigid transform mapping the representative ego frame onto a member ego frame.

    Builds the ego-graph of each clique, tags seed membership, finds the
    best isomorphism by iterating all consistent mappings and picking the one
    with the lowest Kabsch RMSD.  At most *max_mappings* automorphisms are
    tried (defaults to :data:`autokmc.constants.KABSCH_MAX_MAPPINGS`) so that
    pathological high-symmetry egos cannot blow up the runtime.

    Returns ``(R, t)`` or ``(None, None)`` if the egos are not isomorphic.
    Apply as::

        p_member = p_rep @ R.T + t
    """
    rep_seed = frozenset(int(n) for n in rep_clique)
    mem_seed = frozenset(int(n) for n in mem_clique)
    if not rep_seed or not mem_seed:
        return None, None

    rep_ego = _build_ego_graph(G, rep_seed, n_shells)
    mem_ego = _build_ego_graph(G, mem_seed, n_shells)

    # Tag seed membership so the mapping is forced to send seed → seed.
    # NOTE: we mutate ``_seed`` on the *copies* returned by
    # :func:`_build_ego_graph` (which calls ``G.subgraph(...).copy()``).
    # The parent graph *G* is therefore never touched; the egos themselves
    # are throw-away locals and are not cached anywhere, so the stale
    # ``_seed`` attribute cannot leak across calls.
    for n in rep_ego.nodes:
        rep_ego.nodes[n]["_seed"] = (n in rep_seed)
    for n in mem_ego.nodes:
        mem_ego.nodes[n]["_seed"] = (n in mem_seed)

    node_match = isomorphism.categorical_node_match(
        ["element", "_seed"], ["X", False]
    )
    matcher = isomorphism.GraphMatcher(rep_ego, mem_ego, node_match=node_match)
    if not matcher.is_isomorphic():
        return None, None

    rep_c  = _clique_centroid(G, rep_seed, cell, cell_inv, pbc, use_mic)
    mem_c  = _clique_centroid(G, mem_seed, cell, cell_inv, pbc, use_mic)
    rep_p  = _mic_unwrap(G, list(rep_ego.nodes), rep_c, cell, cell_inv, pbc, use_mic)
    mem_p  = _mic_unwrap(G, list(mem_ego.nodes), mem_c, cell, cell_inv, pbc, use_mic)

    nodes  = list(rep_ego.nodes)
    src    = np.array([rep_p[n] for n in nodes], dtype=float)

    best_R: np.ndarray | None = None
    best_t: np.ndarray | None = None
    best_rmsd = np.inf
    for n_tried, mapping in enumerate(matcher.isomorphisms_iter()):
        if n_tried >= max_mappings:
            break
        dst  = np.array([mem_p[mapping[n]] for n in nodes], dtype=float)
        R, t = _kabsch(src, dst)
        res  = src @ R.T + t - dst
        rmsd = float(np.sqrt(np.mean(np.einsum("ij,ij->i", res, res))))
        if rmsd < best_rmsd:
            best_R, best_t, best_rmsd = R, t, rmsd
            if rmsd <= rmsd_tol:
                break

    return best_R, best_t


# ---------------------------------------------------------------------------
# Position optimisation (calculator-free)
# ---------------------------------------------------------------------------

def _outward_normal(G: nx.Graph, centroid: np.ndarray) -> np.ndarray:
    """Unit outward direction: from the surface geometric centre to *centroid*."""
    pos = np.array(
        [d["position"] for _, d in G.nodes(data=True) if d.get("type") == "surface"],
        dtype=float,
    )
    n = centroid - (pos.mean(axis=0) if len(pos) else centroid)
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])


def _optimise_position(
    G: nx.Graph,
    clique: frozenset,
    r_cov_ads: float,
    *,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    n_shells: int = N_SHELLS,
    repulsion_cutoff: float | None = REPULSION_CUTOFF,
) -> np.ndarray:
    """Calculator-free geometric optimisation of the adsorbate position.

    Minimises::

        E(p) = Σ_i (|p − pos_i|_MIC − d*_i)²  +  w · Σ_j 1/(d_j² + ε)

    where i loops over bonded atoms in *clique*, j over nearby non-bonded
    surface atoms (restricted to the n-shell ego and within *repulsion_cutoff*).

    Periodic slabs use L-BFGS-B with a hard z-floor at the highest bonded-atom
    z coordinate (valid for the orthogonalised slabs produced by
    :mod:`autokmc.structure` where the surface normal is aligned with +z).
    Nanoparticles use SLSQP constrained to the outward half-space.
    """
    from scipy.optimize import minimize

    cell, cell_inv, pbc, use_mic = _get_cell(G)

    bonded     = list(clique)
    b_pos      = np.array([G.nodes[n]["position"] for n in bonded], dtype=float)
    b_rcov     = np.array([G.nodes[n]["covalent_radius"] for n in bonded], dtype=float)
    d_ideal    = opt_factor * (r_cov_ads + b_rcov)
    centroid   = _circular_centroid(b_pos, cell, cell_inv, pbc, use_mic)

    # Non-bonded surface atoms for the repulsion term.
    clique_set = set(clique)
    if n_shells > 0:
        ego   = _build_ego_graph(G, frozenset(clique), n_shells)
        cands = [n for n, d in ego.nodes(data=True)
                 if d.get("type") == "surface" and n not in clique_set]
    else:
        cands = [n for n, d in G.nodes(data=True)
                 if d.get("type") == "surface" and n not in clique_set]
    nb_pos = np.array([G.nodes[n]["position"] for n in cands], dtype=float) \
             if cands else np.empty((0, 3), dtype=float)

    if repulsion_cutoff is not None and len(nb_pos):
        if use_mic and cell_inv is not None:
            d2c = _mic_distances(centroid, nb_pos, cell, cell_inv, pbc)
        else:
            d2c = np.linalg.norm(nb_pos - centroid, axis=1)
        nb_pos = nb_pos[d2c <= repulsion_cutoff]

    def obj(p: np.ndarray) -> float:
        if use_mic and cell_inv is not None:
            dists = _mic_distances(p, b_pos, cell, cell_inv, pbc)
        else:
            dists = np.linalg.norm(p[np.newaxis, :] - b_pos, axis=1)
        E = float(np.sum((dists - d_ideal) ** 2))
        if repulsion_weight > 0.0 and len(nb_pos):
            if use_mic and cell_inv is not None:
                nb = _mic_distances(p, nb_pos, cell, cell_inv, pbc)
            else:
                nb = np.linalg.norm(p[np.newaxis, :] - nb_pos, axis=1)
            E += repulsion_weight * float(np.sum(1.0 / (nb ** 2 + 1e-12)))
        return E

    if use_mic:
        # Slab: start above the highest bonded atom along +z (valid because
        # :func:`autokmc.structure._orthogonalise_slab` guarantees the
        # surface normal is aligned with the cartesian z-axis).
        if cell_inv is not None:
            dv_b = b_pos - b_pos[0]
            frac = dv_b @ cell_inv
            for ax in range(3):
                if pbc[ax]:
                    frac[:, ax] -= np.round(frac[:, ax])
            mic_rel = frac @ cell
        else:
            mic_rel = b_pos - b_pos[0]
        lat_d = np.linalg.norm(mic_rel[:, :2] - mic_rel[:, :2].mean(0), axis=1)
        h     = np.sqrt(np.maximum(0.0, d_ideal ** 2 - lat_d ** 2))
        z0    = float(b_pos[:, 2].max()) + float(h.mean())
        x0    = np.array([centroid[0], centroid[1], z0])
        res   = minimize(obj, x0, method="L-BFGS-B",
                         bounds=[(None, None), (None, None),
                                 (float(b_pos[:, 2].max()), None)])
    else:
        # Nanoparticle: constrained to the outward half-space.
        n_out  = _outward_normal(G, centroid)
        stand  = max(float(d_ideal.mean()), 0.5)
        x0     = centroid + stand * n_out
        _c, _n = centroid.copy(), n_out.copy()
        res    = minimize(
            obj, x0, method="SLSQP",
            constraints={"type": "ineq",
                         "fun": lambda p: float(np.dot(p - _c, _n))},
            options={"ftol": 1e-9, "maxiter": 500},
        )
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------------------
# Graph-side anchor node management
# ---------------------------------------------------------------------------

def _next_node_id(G: nx.Graph) -> int:
    """Smallest integer node id strictly greater than all existing ids.

    Accepts both Python ``int`` and ``numpy.integer`` ids so callers
    that produce ids from numpy ranges (``np.arange``) interoperate
    cleanly with callers using plain ``int``.
    """
    if not G.nodes:
        return 0
    return int(max(
        int(n) for n in G.nodes if isinstance(n, (int, np.integer))
    )) + 1


def _remove_anchor_nodes(G: nx.Graph, element: str) -> None:
    """Drop all anchor nodes for *element* and their edges from *G*."""
    stale = [n for n, d in G.nodes(data=True)
             if d.get("type") == "anchor" and d.get("element") == element]
    if stale:
        G.remove_nodes_from(stale)


def _add_anchor_node(
    G: nx.Graph,
    *,
    element: str,
    r_cov_ads: float,
    clique: frozenset,
    position: np.ndarray,
    k: int,
    iso_class: int | None = None,
    n_shells: int | None = None,
    ego_graph: nx.Graph | None = None,
) -> int:
    """Add one anchor node to *G* at *position*; wire it to its clique atoms.

    Returns the new node id.
    """
    nid = _next_node_id(G)
    G.add_node(
        nid,
        element         = element,
        position        = np.asarray(position, dtype=float).copy(),
        index           = nid,
        type            = "anchor",
        covalent_radius = float(r_cov_ads),
        clique          = clique,
        k               = int(k),
        iso_class       = iso_class,
        n_shells        = n_shells,
        ego_subgraph    = ego_graph,
        optimised       = False,
    )
    for s in clique:
        if s not in G:
            continue
        d = float(np.linalg.norm(
            np.asarray(G.nodes[s]["position"], dtype=float) - position
        ))
        G.add_edge(nid, s, distance=d, offset=(0, 0, 0), anchor_bond=True)
    return nid


# ---------------------------------------------------------------------------
# Clique enumeration helpers
# ---------------------------------------------------------------------------

def _enumerate_cliques(
    G: nx.Graph,
    element: str,
    r_cov_ads: float,
    co_factor: float = CO_FACTOR,
) -> dict[int, list[frozenset]]:
    """Return ``{k: [frozenset_of_node_ids, …]}`` for every clique size 1…k_max.

    Two geometric filters drop spurious wrap-around / sub-surface cliques:

    * **Nanoparticles** — the convex hull of the *surface* atoms is used as
      the boundary of the particle.  A clique whose MIC-aware centroid sits
      strictly inside the hull (signed distance below :data:`HULL_TOL`) is
      a wrap-around artefact (e.g. an "anchor site" buried at the centre
      of a periodic image of the NP) and is dropped.
    * **Slabs** — a clique whose centroid sits below the lowest surface
      atom along the local outward normal is similarly buried beneath the
      surface and dropped.  The outward normal is just ``+z`` for the
      orthogonalised slabs that :mod:`autokmc.structure` produces.
    """
    cbg = _build_co_bond_graph(G, r_cov_ads, co_factor)
    if cbg.number_of_nodes() == 0:
        return {1: []}

    k_max = max((len(c) for c in nx.find_cliques(cbg)), default=1)
    cell, cell_inv, pbc, use_mic = _get_cell(G)

    # ------------------------------------------------------------------
    # Geometric "is this clique buried?" filter.  Built once per call.
    # ------------------------------------------------------------------
    surf_pos = np.array(
        [d["position"] for _, d in G.nodes(data=True)
         if d.get("type") == "surface"],
        dtype=float,
    )

    hull_eq: np.ndarray | None = None
    z_floor: float | None      = None

    if not use_mic:
        # ── Nanoparticle: convex hull of surface atoms ────────────────
        hull_eq = G.graph.get("hull_equations")
        if hull_eq is None and len(surf_pos) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hull_eq = np.asarray(
                    ConvexHull(surf_pos).equations, dtype=float,
                )
                G.graph["hull_equations"] = hull_eq
            except Exception:
                hull_eq = None
    else:
        # ── Slab: drop cliques whose centroid is below the surface ────
        # All builders orthogonalise the slab cell (surface ‖ xy plane,
        # outward normal = +z), so a simple z-floor is sufficient and
        # cheap.  ``HULL_TOL`` is reused as the (negative) Å tolerance
        # below the lowest surface atom that we still accept.
        if len(surf_pos):
            z_floor = float(surf_pos[:, 2].min()) + HULL_TOL

    sites: dict[int, list[frozenset]] = {k: [] for k in range(1, k_max + 1)}
    seen: set[frozenset] = set()
    for clique in nx.enumerate_all_cliques(cbg):
        k = len(clique)
        if k > k_max:
            break
        key = frozenset(clique)
        if key in seen:
            continue
        seen.add(key)

        if hull_eq is not None or z_floor is not None:
            c = _clique_centroid(G, key, cell, cell_inv, pbc, use_mic)
            if hull_eq is not None:
                if float(np.max(hull_eq[:, :3] @ c + hull_eq[:, 3])) < HULL_TOL:
                    continue   # buried inside the NP hull
            elif z_floor is not None and c[2] < z_floor:
                continue       # buried beneath the slab surface

        sites[k].append(key)

    return {k: v for k, v in sites.items() if v}


# ---------------------------------------------------------------------------
# Reduction to iso-classes
# ---------------------------------------------------------------------------

def _reduce_by_isomorphism(
    G: nx.Graph,
    sites_by_k: dict[int, list[frozenset]],
    n_shells: int = N_SHELLS,
) -> dict[int, list[AnchorSite]]:
    """Group every clique into an :class:`AnchorSite` iso-class.

    Two cliques belong to the same iso-class iff their n-shell ego-subgraphs
    are graph-isomorphic under element-label matching.
    """
    node_match = isomorphism.categorical_node_match("element", "X")
    unique: dict[int, list[AnchorSite]] = {}

    cell, cell_inv, pbc, use_mic = _get_cell(G)

    for k, cliques in sorted(sites_by_k.items()):
        rep_egos:    list[nx.Graph] = []
        rep_fkeys:   list[tuple]    = []
        assignment:  list[int]      = []

        for clq in cliques:
            ego  = _build_ego_graph(G, clq, n_shells)
            fkey = _fingerprint(ego)
            placed = False
            for cid, (rep_ego, rfkey) in enumerate(zip(rep_egos, rep_fkeys)):
                if fkey != rfkey:
                    continue
                if isomorphism.GraphMatcher(ego, rep_ego,
                                            node_match=node_match).is_isomorphic():
                    assignment.append(cid)
                    placed = True
                    break
            if not placed:
                rep_egos.append(ego)
                rep_fkeys.append(fkey)
                assignment.append(len(rep_egos) - 1)

        classes: list[AnchorSite] = []
        for cid in range(len(rep_egos)):
            member_idxs = [i for i, a in enumerate(assignment) if a == cid]
            members     = [cliques[i] for i in member_idxs]
            rep         = members[0]
            pos_arr     = np.array([G.nodes[n]["position"] for n in rep], dtype=float)
            centroid    = _circular_centroid(pos_arr, cell, cell_inv, pbc, use_mic)
            classes.append(AnchorSite(
                element       = "",           # filled in by caller
                k             = k,
                iso_class     = cid,
                n_shells      = n_shells,
                representative= rep,
                members       = members,
                centroid      = centroid,
                ego_graph     = rep_egos[cid],
            ))

        label = COORD_LABELS.get(k, f"{k}-fold")
        _log.debug("  k=%d %-8s  %d cliques → %d iso-classes",
                   k, label, len(cliques), len(classes))
        unique[k] = classes

    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def k_max_for_element(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
) -> int:
    """Return the largest clique size (k_max) achievable by *element* on *G*.

    Does not enumerate all sites or modify *G*.
    """
    if element not in _ASE_AN:
        raise KeyError(f"Unknown element '{element}'.")
    r_cov = float(_ASE_RCOV[_ASE_AN[element]])
    cbg   = _build_co_bond_graph(G, r_cov, co_factor)
    return max((len(c) for c in nx.find_cliques(cbg)), default=1) \
           if cbg.number_of_nodes() else 1


def find_anchor_sites(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    repulsion_cutoff: float | None = REPULSION_CUTOFF,
    n_shells: int = N_SHELLS,
    k_max: int | None = None,
    verbose: bool = False,
) -> list[AnchorSite]:
    """Enumerate, classify and geometrically optimise all anchor sites for
    *element* on the surface graph *G*.

    Parameters
    ----------
    G : nx.Graph
        Connectivity graph from ``build_graph``.  Surface atoms must be tagged
        ``type="surface"``.  Graph-level keys ``"cell"`` and ``"pbc"`` must
        exist (both are set by ``build_graph``).
    element : str
        Chemical symbol of the adsorbate atom, e.g. ``"C"``, ``"O"``.
    co_factor : float
        Co-bonding cutoff scale.  Default 0.90.
    opt_factor : float
        Ideal bond-length scale for geometric optimisation.  Default 0.85.
    repulsion_weight : float
        Weight on the soft non-bonded repulsion term.  Default 0.1.
    repulsion_cutoff : float or None
        Spatial cutoff (Å) on which non-bonded atoms enter the repulsion sum.
        ``None`` disables the cutoff (slower but exact).  Default 6.0 Å.
    n_shells : int
        Ego-graph depth for iso-class discrimination.  1 distinguishes fcc vs
        hcp hollows on (111); 0 collapses them.  Default 1.
    k_max : int or None
        Optional hard cap on clique size.  ``None`` (default) keeps the
        natural ``k_max`` from the co-bonding graph.  Set explicitly to
        suppress runaway clique enumeration on very dense surfaces (see
        ``todo.MD``).
    verbose : bool
        Print a per-k summary table.

    Returns
    -------
    list[AnchorSite]
        One entry per iso-class (representative of a group of symmetry-
        equivalent cliques).  Also stored at
        ``G.graph["anchor_sites"][element]``.  The raw per-k clique lists
        are at ``G.graph["raw_cliques"][element]``.

    Side effects
    ------------
    * All previously materialised anchor nodes for *element* are removed from
      *G* and replaced with fresh ones.
    * One anchor node per raw clique is added to *G* (``type="anchor"``,
      invisible to the ego-graph BFS).  Node attributes include ``iso_class``,
      ``n_shells``, ``ego_subgraph``, ``optimised``, ``position``.
    """
    if element not in _ASE_AN:
        raise KeyError(f"Unknown element '{element}'.")

    r_cov = float(_ASE_RCOV[_ASE_AN[element]])

    if verbose:
        print(f"\nfind_anchor_sites: element={element!r}  "
              f"r_cov={r_cov:.4f} Å  n_shells={n_shells}  "
              f"co_factor={co_factor}  opt_factor={opt_factor}")

    # Remove stale anchor nodes from a previous call.
    _remove_anchor_nodes(G, element)

    # ── Step 1: enumerate raw cliques ─────────────────────────────────────
    sites_by_k = _enumerate_cliques(G, element, r_cov, co_factor)
    if k_max is not None:
        # Drop oversized cliques up-front (todo.MD: clique blowup).
        sites_by_k = {k: v for k, v in sites_by_k.items() if k <= k_max}
        if not sites_by_k:
            sites_by_k = {1: []}
    n_raw = sum(len(v) for v in sites_by_k.values())
    _log.debug("find_anchor_sites: %r  %d raw cliques  k_max=%d",
               element, n_raw, max(sites_by_k) if sites_by_k else 0)

    # ── Step 2: reduce by isomorphism ─────────────────────────────────────
    unique_by_k = _reduce_by_isomorphism(G, sites_by_k, n_shells=n_shells)
    n_iso = sum(len(v) for v in unique_by_k.values())

    # ── Step 3: build clique → (k, index) reverse map ─────────────────────
    clique_to_loc: dict[frozenset, tuple[int, int]] = {}
    for k, cliques in sites_by_k.items():
        for idx, clq in enumerate(cliques):
            clique_to_loc[clq] = (k, idx)

    # Positions array (parallel to sites_by_k) filled during propagation.
    positions: dict[int, list[np.ndarray | None]] = {
        k: [None] * len(v) for k, v in sites_by_k.items()
    }

    cell, cell_inv, pbc, use_mic = _get_cell(G)


    # ── Step 4 + 5: optimise representative, propagate to members ─────────
    all_sites: list[AnchorSite] = []
    for k, classes in sorted(unique_by_k.items()):
        n_prop     = 0
        n_fallback = 0
        for iso in classes:
            iso.element = element

            # Optimise the representative.
            p_rep = _optimise_position(
                G, iso.representative, r_cov,
                opt_factor=opt_factor,
                repulsion_weight=repulsion_weight,
                n_shells=n_shells,
                repulsion_cutoff=repulsion_cutoff,
            )
            iso.position = p_rep

            # Write representative into positions array.
            k_r, idx_r = clique_to_loc[iso.representative]
            positions[k_r][idx_r] = p_rep

            # Propagate to every other member via Kabsch ego-alignment.
            for member in iso.members:
                if member == iso.representative:
                    continue
                R, t = _kabsch_align_ego(
                    G, iso.representative, member, n_shells,
                    cell, cell_inv, pbc, use_mic,
                )
                k_m, idx_m = clique_to_loc[member]
                if R is not None and t is not None:
                    p_member = p_rep @ R.T + t
                    n_prop += 1
                else:
                    # Fallback: independently optimise this member.
                    p_member = _optimise_position(
                        G, member, r_cov,
                        opt_factor=opt_factor,
                        repulsion_weight=repulsion_weight,
                        n_shells=n_shells,
                        repulsion_cutoff=repulsion_cutoff,
                    )
                    n_fallback += 1
                positions[k_m][idx_m] = p_member

        label = COORD_LABELS.get(k, f"{k}-fold")
        _log.debug(
            "  k=%d %-8s  %d iso-classes  "
            "(%d propagated, %d fallback re-opt)",
            k, label, len(classes), n_prop, n_fallback,
        )
        all_sites.extend(classes)

    # ── Step 6: materialise anchor nodes on G ─────────────────────────────
    # Build a clique → position lookup from the just-filled arrays.
    clique_to_pos: dict[frozenset, np.ndarray] = {}
    for k, cliques in sites_by_k.items():
        for idx, clq in enumerate(cliques):
            p = positions[k][idx]
            if p is not None:
                clique_to_pos[clq] = p

    # Build a clique → AnchorSite lookup for iso_class stamping.
    clique_to_iso: dict[frozenset, AnchorSite] = {}
    for iso in all_sites:
        for member in iso.members:
            clique_to_iso[member] = iso

    # node_ids_by_clique maps each raw clique to its materialised node id.
    node_ids_by_clique: dict[frozenset, int] = {}
    for k, cliques in sorted(sites_by_k.items()):
        for clq in cliques:
            p     = clique_to_pos.get(clq)
            if p is None:
                # Emergency fallback: use clique centroid.
                p = _clique_centroid(G, clq, cell, cell_inv, pbc, use_mic)
            iso   = clique_to_iso.get(clq)
            nid   = _add_anchor_node(
                G,
                element   = element,
                r_cov_ads = r_cov,
                clique    = clq,
                position  = p,
                k         = k,
                iso_class = int(iso.iso_class) if iso else None,
                n_shells  = n_shells,
                ego_graph = iso.ego_graph if iso else None,
            )
            G.nodes[nid]["optimised"] = True
            node_ids_by_clique[clq] = nid

    # Attach node_ids lists to each AnchorSite (parallel to .members).
    for iso in all_sites:
        iso.node_ids = [node_ids_by_clique[m] for m in iso.members
                        if m in node_ids_by_clique]

    # ── Persist to G.graph ────────────────────────────────────────────────
    G.graph.setdefault("anchor_sites", {})[element]  = all_sites
    G.graph.setdefault("raw_cliques",  {})[element]  = sites_by_k

    if verbose:
        print(f"  {'k':>3}  {'type':<10}  {'raw':>6}  {'unique':>6}")
        print(f"  {'-'*32}")
        for k, classes in sorted(unique_by_k.items()):
            label = COORD_LABELS.get(k, f"{k}-fold")
            raw   = len(sites_by_k.get(k, []))
            print(f"  {k:>3}  {label:<10}  {raw:>6}  {len(classes):>6}")
        print(f"  {'-'*32}")
        print(f"  {'':>3}  {'total':<10}  {n_raw:>6}  {n_iso:>6}")

    _log.debug(
        "find_anchor_sites: %r done — %d raw cliques, %d iso-classes",
        element, n_raw, n_iso,
    )
    return all_sites

