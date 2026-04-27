"""
autokmc.find_adsorbate_sites
============================
N-atom adsorbate site enumeration on a surface graph, built cleanly on top of
:mod:`autokmc.find_anchors` and :mod:`autokmc.reactants`.

Where :func:`~autokmc.find_anchors.find_anchor_sites` finds all single-atom
adsorption *anchor* sites for one element, this module finds all geometrically
feasible placements of a multi-atom *molecule* (a
:class:`~autokmc.reactants.Reactant`).  A placement is stored as one surface
clique per reactant atom rather than a single node.

Strategy
--------
1. **Pick anchors.**  Only atoms in ``reactant.anchor_atoms`` (convex-hull-
   exposed, sterically unhindered) may bond to a surface clique.  Every other
   atom floats; its position is reconstructed by rigid alignment of the
   gas-phase geometry.

2. **Choose shell depth from molecular reach.**  Default
   ``n_shells_anchor = max(1, ceil(reach / d_nn))`` where *reach* is the
   largest intramolecular anchor-pair distance and *d_nn* ≈ 2.5 Å is a
   typical metal nearest-neighbour distance.

3. **Backtracking chain placement.**  For every orbit-canonicalised non-empty
   subset of anchors:

   * The first anchor is placed at every unique ``AnchorSite`` iso-class for
     its element (seeds from :func:`~autokmc.find_anchors.find_anchor_sites`).
   * Subsequent anchors are placed at any *raw* anchor-node position of their
     element whose MIC distance to every already-placed anchor matches the
     intramolecular distance within ``bond_tolerance``.

4. **Surface-connectivity guard.**  The bonded cliques of every placement must
   be mutually reachable through ``type=="surface"`` edges; placements whose
   cliques are too far apart (> ``max_pair_shells`` hops) are dropped.

5. **Reduce by isomorphism.**  Group placements by isomorphism of the union-
   of-cliques ego-subgraph (element-label matched;
   :func:`~autokmc.find_anchors._build_ego_graph`).

6. **Auto-grow** ``n_shells_anchor`` if any placement reaches further than the
   iso-class ego could see (up to ``max_shell_retries`` extra passes).

7. **Materialise** one adsorbate-site node per reactant atom per member on *G*
   (``type="adsorbate"``, ``occupied=False``).

8. **Optional rigid-body refinement.**
   :func:`optimise_adsorbate_site_positions` does a calculator-free L-BFGS-B
   refinement of each iso-class representative's 6 rigid-body DOF.

Storage
-------
Results are stored in ``G.graph["adsorbate_sites"][smiles]``: a flat list of
:class:`AdsorbateSite` objects.

Public API
----------
* :class:`AdsorbateSite`                    — one iso-class of molecule placements.
* :func:`find_adsorbate_sites`              — universal N-atom enumerator.
* :func:`optimise_adsorbate_site_positions` — rigid-body refinement.
* :func:`push_member_positions_to_graph`    — write refined positions back to G.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.find_anchors import (
    find_anchor_sites,
    _build_ego_graph,
    _kabsch_align_ego,
    _get_cell,
    _kabsch,
    AnchorSite,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable defaults — single source of truth in :mod:`autokmc.constants`.
# Re-exported here for backwards compatibility and readability of the
# function-default kwargs.  See ``constants.py`` for full docstrings.
# ---------------------------------------------------------------------------

from autokmc.constants import (
    BOND_TOLERANCE,
    NN_DISTANCE,
    MAX_PAIR_SHELLS,
    N_SHELLS_DEFAULT,
    CO_FACTOR,
    OPT_FACTOR,
    REPULSION_WEIGHT,
    CONTACT_FACTOR,
    STANDOFF_FACTOR,
    N_ADSORBATE_RESTARTS as N_RESTARTS,
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class AdsorbateSite:
    """One isomorphism class of N-atom molecule placements on a surface.

    The iso-class deduplication uses the n-shell ego-subgraph around the
    union of bonded surface-atom cliques (built by
    :func:`~autokmc.find_anchors._build_ego_graph`).

    Attributes
    ----------
    reactant : str
        SMILES of the reactant; key into the surrounding ``Reactant``
        registry from which gas-phase energy / structure can be retrieved.
    n_atoms : int
        Number of atoms in the reactant (length of ``atom_cliques`` and
        ``positions``).
    atom_cliques : list[frozenset[int] | None]
        Surface clique each reactant atom binds to, in reactant atom-index
        order.  ``None`` means "not bonded to the surface in this placement"
        (e.g. the dangling end of a physisorbed diatomic).
    positions : np.ndarray, shape (n_atoms, 3)
        Cartesian positions for the iso-class **representative** adsorbate
        atoms (Å).  Member positions live on *G* via ``member_node_ids``;
        they are not duplicated here.
    iso_class : int
        0-based index within the per-reactant iso-class list.
    members : list[list[frozenset | None]]
        All raw ``atom_cliques`` tuples folded into this iso-class (the
        first entry is the representative).
    member_node_ids : list[list[int]]
        Per-member graph node ids for the materialised adsorbate-site nodes
        (``len == n_atoms`` each, in reactant atom-index order).  Use to
        look up an iso-class member directly on the graph::

            [G.nodes[n] for n in member_node_ids[k]]

    ego_graph : nx.Graph | None
        The ``n_shells_pair`` ego-subgraph (built around the union of
        bonded cliques *plus* occupied neighbour chains) that defines this
        iso-class.
    """

    reactant        : str
    n_atoms         : int
    atom_cliques    : list
    positions       : Any
    iso_class       : int
    members         : list    = field(default_factory=list)
    member_node_ids : list    = field(default_factory=list)
    ego_graph       : Any     = None

    # ------------------------------------------------------------------
    # Occupancy helpers
    # ------------------------------------------------------------------

    def _member_is_occupied(self, G: nx.Graph, node_ids: list[int]) -> bool:
        for nid in node_ids:
            if nid in G and G.nodes[nid].get("occupied", False):
                return True
        return False

    def occupied_member_indices(self, G: nx.Graph) -> list[int]:
        """Indices into :attr:`members` flagged ``occupied=True`` on *G*."""
        return [k for k, nids in enumerate(self.member_node_ids)
                if self._member_is_occupied(G, nids)]

    def unoccupied_member_indices(self, G: nx.Graph) -> list[int]:
        """Indices into :attr:`members` flagged ``occupied=False`` on *G*."""
        return [k for k, nids in enumerate(self.member_node_ids)
                if not self._member_is_occupied(G, nids)]

    def occupied_member_node_ids(self, G: nx.Graph) -> list[list[int]]:
        """Node-id groups for currently-occupied members."""
        return [self.member_node_ids[k]
                for k in self.occupied_member_indices(G)]

    def unoccupied_member_node_ids(self, G: nx.Graph) -> list[list[int]]:
        """Node-id groups for currently-unoccupied members."""
        return [self.member_node_ids[k]
                for k in self.unoccupied_member_indices(G)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _suggested_n_shells(reach: float, nn_distance: float = NN_DISTANCE) -> int:
    """Heuristic shell depth from molecular geometric reach."""
    return max(1, int(np.ceil(reach / nn_distance)))


def _mic_distance(
    p: np.ndarray,
    q: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> float:
    """MIC distance between two Cartesian points."""
    dv = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    if use_mic and cell_inv is not None:
        frac = dv @ cell_inv
        for i in range(3):
            if pbc[i]:
                frac[i] -= np.round(frac[i])
        dv = frac @ cell
    return float(np.linalg.norm(dv))


def _outward_normal_at(
    G: nx.Graph, p: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Unit outward direction at *p*: +z for slabs, radial-out for NPs."""
    if pbc.any():
        return np.array([0.0, 0.0, 1.0])
    surf_pos = np.array(
        [d["position"] for _, d in G.nodes(data=True)
         if d.get("type") == "surface"],
        dtype=float,
    )
    if not len(surf_pos):
        return np.array([0.0, 0.0, 1.0])
    n = p - surf_pos.mean(axis=0)
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])


def _full_adsorbate_positions(
    reactant,
    bonded_indices: list[int],
    bonded_positions: np.ndarray,
    G: nx.Graph,
    pbc: np.ndarray,
) -> np.ndarray:
    """Reconstruct Cartesian positions for every reactant atom.

    * Two or more bonded anchors → Kabsch rigid alignment.
    * One bonded anchor → translation + outward-normal orientation.
    """
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    n_atoms   = react_pos.shape[0]

    if len(bonded_indices) >= 2:
        R, t = _kabsch(react_pos[bonded_indices], bonded_positions)
        return react_pos @ R.T + t

    # Single bonded anchor: translate; rotate unbonded atoms outward.
    i0       = bonded_indices[0]
    p_target = bonded_positions[0]
    rel      = react_pos - react_pos[i0]
    n_out    = _outward_normal_at(G, p_target, pbc)

    if n_atoms > 1:
        other  = np.delete(np.arange(n_atoms), i0)
        c      = rel[other].mean(axis=0)
        norm   = float(np.linalg.norm(c))
        if norm > 1e-8:
            v    = c / norm
            axis = np.cross(v, n_out)
            s    = float(np.linalg.norm(axis))
            cos  = float(np.dot(v, n_out))
            if s > 1e-8:
                axis /= s
                K = np.array([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])
                rel = rel @ (np.eye(3) + s * K + (1.0 - cos) * (K @ K)).T
            elif cos < 0.0:
                rel = -rel
    return rel + p_target


# ---------------------------------------------------------------------------
# Reactant molecule helpers
# ---------------------------------------------------------------------------

def _orbit_id_of(reactant) -> dict[int, tuple[str, int]]:
    """Map each reactant atom index → ``(element, orbit_idx)``."""
    out: dict[int, tuple[str, int]] = {}
    for elem, orbits in reactant.unique_nodes.items():
        for k, orb in enumerate(orbits):
            for i in orb:
                out[int(i)] = (elem, k)
    return out


def _canonical_subset_key(
    subset: tuple[int, ...],
    orbit_id: dict[int, tuple[str, int]],
) -> tuple:
    """Orbit-multiset signature — equal keys → symmetry-equivalent subsets."""
    return tuple(sorted(orbit_id[i] for i in subset))


def _fingerprint(g: nx.Graph) -> tuple:
    """Cheap graph fingerprint — unequal → guaranteed non-isomorphic."""
    elem_deg = tuple(sorted(
        (d.get("element", "X"), g.degree(n))
        for n, d in g.nodes(data=True)
    ))
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(sorted(g.degree(n) for n in g.nodes())),
        elem_deg,
    )


def _placement_signature(atom_cliques: list) -> tuple:
    """Hashable canonical key for a raw placement."""
    return tuple(None if c is None else frozenset(c) for c in atom_cliques)


# ---------------------------------------------------------------------------
# Surface graph helpers
# ---------------------------------------------------------------------------

def _surface_subgraph(G: nx.Graph) -> nx.Graph:
    """Induced subgraph on ``type == "surface"`` nodes."""
    return G.subgraph(
        [n for n, d in G.nodes(data=True) if d.get("type") == "surface"]
    )


def _get_surface_apsp(G: nx.Graph, *, cutoff: int = MAX_PAIR_SHELLS) -> dict:
    """Surface-only all-pairs shortest paths ≤ *cutoff* hops.

    Cached in ``G.graph["surface_apsp"]`` as
    ``{"_cutoff": cutoff, "data": {u: {v: dist}}}``.
    """
    cached = G.graph.get("surface_apsp")
    if isinstance(cached, dict) and cached.get("_cutoff") == cutoff:
        return cached["data"]
    G_surf = _surface_subgraph(G)
    data: dict[int, dict[int, int]] = {}
    for u in G_surf.nodes:
        data[u] = dict(
            nx.single_source_shortest_path_length(G_surf, u, cutoff=cutoff)
        )
    G.graph["surface_apsp"] = {"_cutoff": cutoff, "data": data}
    return data


def _shortest_path_between_cliques(
    clique_a,
    clique_b,
    apsp: dict,
) -> int:
    """Minimum surface-graph hops between the closest nodes of two cliques."""
    if not clique_a or not clique_b:
        return 10 ** 6
    if set(clique_a) & set(clique_b):
        return 0
    target = set(clique_b)
    best   = 10 ** 6
    for a in clique_a:
        a_dists = apsp.get(a, {})
        for b in target:
            d = a_dists.get(b)
            if d is not None and d < best:
                best = d
    return best


def _min_ego_depth_for_connectivity(
    bonded_cliques: list,
    apsp: dict,
) -> int:
    """Smallest ego depth needed so every pair of *bonded_cliques* is connected.

    For two cliques with closest-node hop distance L, a connecting path lies
    within the ``floor(L/2)``-shell ego of their union.
    """
    if len(bonded_cliques) < 2:
        return 0
    max_d = 0
    for i in range(len(bonded_cliques)):
        for j in range(i + 1, len(bonded_cliques)):
            L = _shortest_path_between_cliques(
                bonded_cliques[i], bonded_cliques[j], apsp
            )
            if L >= 10 ** 6:
                return 10 ** 6
            d = L // 2
            if d > max_d:
                max_d = d
    return max_d


def _required_n_shells(
    G: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    apsp: dict,
) -> int:
    """Largest depth needed so the iso ego contains every bonded clique pair."""
    needed = 0
    for ms in adsorbate_sites:
        bonded = [c for c in ms.atom_cliques if c is not None]
        if len(bonded) < 2:
            continue
        first = bonded[0]
        for other in bonded[1:]:
            d = _shortest_path_between_cliques(first, other, apsp)
            if d != 10 ** 6 and d > needed:
                needed = d
    return needed


# ---------------------------------------------------------------------------
# Anchor-site management
# ---------------------------------------------------------------------------

class _AnchorKDTree:
    """KD-tree over anchor positions with MIC-aware annulus queries.

    Built once per (element, pass) and queried during chain placement to
    replace the O(N) scan over every anchor position with an O(log N + h)
    range query, where *h* is the number of anchors lying within
    ``[D - bond_tolerance, D + bond_tolerance]`` of the query point.

    Three internal layouts depending on cell shape:

    * **orthogonal periodic** — :class:`scipy.spatial.cKDTree` with native
      ``boxsize`` so distances returned by the query are already the
      minimum-image distances.
    * **non-orthogonal periodic** — positions tiled across ±1 cell images
      along every periodic axis (up to 27 copies); duplicates are mapped
      back to original indices on output.
    * **non-periodic (nanoparticle)** — plain kd-tree.
    """

    def __init__(
        self,
        positions: np.ndarray,
        cliques: list,
        cell: np.ndarray,
        cell_inv: np.ndarray | None,
        pbc: np.ndarray,
        use_mic: bool,
    ) -> None:
        from scipy.spatial import cKDTree

        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.cliques   = list(cliques)
        self.pbc       = pbc
        self.use_mic   = use_mic
        self._is_ortho = False
        self._boxsize: np.ndarray | None = None
        self._tile_index: np.ndarray | None = None

        if len(self.positions) == 0:
            self._tree = None
            return

        is_ortho = (
            use_mic and np.allclose(cell - np.diag(np.diag(cell)), 0.0)
        )

        if is_ortho:
            boxsize = np.where(pbc, np.diag(cell), 0.0).astype(float)
            wrap = self.positions.copy()
            for ax in range(3):
                if pbc[ax] and boxsize[ax] > 0:
                    wrap[:, ax] = np.mod(wrap[:, ax], boxsize[ax])
            self._tree     = cKDTree(
                wrap, boxsize=np.where(boxsize > 0, boxsize, 0.0)
            )
            self._is_ortho = True
            self._boxsize  = boxsize
        elif use_mic:
            offsets: list[np.ndarray] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx and not pbc[0]:
                            continue
                        if dy and not pbc[1]:
                            continue
                        if dz and not pbc[2]:
                            continue
                        offsets.append(np.array([dx, dy, dz], dtype=int))
            tiled: list[np.ndarray] = []
            tile_idx: list[int]     = []
            for off in offsets:
                tiled.append(self.positions + off @ cell)
                tile_idx.extend(range(len(self.positions)))
            self._tree       = cKDTree(np.concatenate(tiled))
            self._tile_index = np.asarray(tile_idx, dtype=int)
        else:
            self._tree = cKDTree(self.positions)

    def query_annulus(
        self,
        p_query: np.ndarray,
        r_min: float,
        r_max: float,
    ) -> list[int]:
        """Return indices into :attr:`positions` whose MIC distance to
        *p_query* lies in ``[r_min, r_max]``.

        Negative ``r_min`` is treated as zero.
        """
        if self._tree is None:
            return []
        r_min = max(0.0, float(r_min))
        r_max = float(r_max)

        if self._is_ortho and self._boxsize is not None:
            q = np.asarray(p_query, dtype=float).copy()
            for ax in range(3):
                if self.pbc[ax] and self._boxsize[ax] > 0:
                    q[ax] = q[ax] % self._boxsize[ax]
            hits = self._tree.query_ball_point(q, r=r_max)
            if not hits:
                return []
            data = self._tree.data[np.asarray(hits, dtype=int)]
            d = np.linalg.norm(data - q, axis=1)
            keep = d >= r_min
            return [int(hits[i]) for i in np.where(keep)[0]]

        if self._tile_index is not None:
            hits = self._tree.query_ball_point(p_query, r=r_max)
            if not hits:
                return []
            hits = np.asarray(hits, dtype=int)
            d    = np.linalg.norm(self._tree.data[hits] - p_query, axis=1)
            keep = (d >= r_min) & (d <= r_max)
            seen: set[int] = set()
            out: list[int] = []
            for h, k in zip(hits, keep):
                if not k:
                    continue
                orig = int(self._tile_index[h])
                if orig in seen:
                    continue
                seen.add(orig)
                out.append(orig)
            return out

        hits = self._tree.query_ball_point(p_query, r=r_max)
        if not hits:
            return []
        hits = np.asarray(hits, dtype=int)
        d    = np.linalg.norm(self.positions[hits] - p_query, axis=1)
        keep = (d >= r_min) & (d <= r_max)
        return [int(hits[i]) for i in np.where(keep)[0]]


def _ensure_anchor_sites(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    n_shells: int = N_SHELLS_DEFAULT,
    verbose: bool = False,
) -> None:
    """Lazily run :func:`~autokmc.find_anchors.find_anchor_sites` if needed."""
    if element not in G.graph.get("anchor_sites", {}):
        find_anchor_sites(
            G, element,
            co_factor=co_factor,
            opt_factor=opt_factor,
            repulsion_weight=repulsion_weight,
            n_shells=n_shells,
            verbose=verbose,
        )


def _all_raw_sites_with_positions(
    G: nx.Graph, element: str
) -> list[tuple[frozenset, np.ndarray]]:
    """Return ``[(clique, position)]`` for every anchor node of *element* on *G*.

    Anchor nodes are materialised by
    :func:`~autokmc.find_anchors.find_anchor_sites` — one per raw clique,
    each carrying a Kabsch-propagated position.
    """
    out: list[tuple[frozenset, np.ndarray]] = []
    for _n, d in G.nodes(data=True):
        if d.get("type") == "anchor" and d.get("element") == element:
            clq = d.get("clique")
            pos = d.get("position")
            if clq is not None and pos is not None:
                out.append((frozenset(clq), np.asarray(pos, dtype=float)))
    return out


def _clique_position_index(
    G: nx.Graph, elements: list[str]
) -> dict[str, dict[frozenset, np.ndarray]]:
    """``element → {clique: position}`` from every anchor node in *G*.

    Cached on ``G.graph["_clique_position_index_cache"]`` keyed by the
    set of (anchor-node-id, element) pairs.  Adding or removing anchor
    nodes (e.g. by re-running :func:`find_anchor_sites`) automatically
    invalidates the cache because the key changes.
    """
    elem_set = frozenset(elements)
    # Build a structural key that changes whenever anchor nodes change.
    anchor_key = tuple(sorted(
        (int(n), d.get("element"))
        for n, d in G.nodes(data=True)
        if d.get("type") == "anchor"
    ))
    cache = G.graph.get("_clique_position_index_cache")
    if (isinstance(cache, dict)
            and cache.get("_anchor_key") == anchor_key
            and elem_set.issubset(cache.get("_elements", frozenset()))):
        # Sub-restrict the cached dict to the requested elements.
        return {el: cache["data"].get(el, {}) for el in elem_set}

    out: dict[str, dict[frozenset, np.ndarray]] = {}
    for el in elem_set:
        idx: dict[frozenset, np.ndarray] = {}
        for _n, d in G.nodes(data=True):
            if d.get("type") == "anchor" and d.get("element") == el:
                clq = d.get("clique")
                pos = d.get("position")
                if clq is not None and pos is not None:
                    idx[frozenset(clq)] = np.asarray(pos, dtype=float)
        out[el] = idx

    # Merge into the cache (preserving any prior elements computed).
    if isinstance(cache, dict) and cache.get("_anchor_key") == anchor_key:
        cache["data"].update(out)
        cache["_elements"] = cache.get("_elements", frozenset()) | elem_set
    else:
        G.graph["_clique_position_index_cache"] = {
            "_anchor_key": anchor_key,
            "_elements":   elem_set,
            "data":        dict(out),
        }
    return out


# ---------------------------------------------------------------------------
# Iso-class reduction
# ---------------------------------------------------------------------------

def _try_merge_or_new(
    adsorbate_sites: list[AdsorbateSite],
    *,
    reactant_smiles: str,
    atom_cliques: list,
    positions: np.ndarray,
    ego_graph: nx.Graph,
    node_match,
    seen_signatures: set,
) -> None:
    """Merge into an existing iso-class if isomorphic, else append a new one.

    Exact duplicate placements (same surface cliques in the same atom slots)
    are dropped via *seen_signatures*.
    """
    sig = _placement_signature(atom_cliques)
    if sig in seen_signatures:
        return
    seen_signatures.add(sig)

    bonded_pattern = tuple(c is None for c in atom_cliques)
    fkey = _fingerprint(ego_graph)
    pos_arr = np.asarray(positions, dtype=float)

    for ms in adsorbate_sites:
        if tuple(c is None for c in ms.atom_cliques) != bonded_pattern:
            continue
        if ms.ego_graph is None:
            continue
        if _fingerprint(ms.ego_graph) != fkey:
            continue
        if isomorphism.GraphMatcher(
            ego_graph, ms.ego_graph, node_match=node_match
        ).is_isomorphic():
            ms.members.append(list(atom_cliques))
            return

    adsorbate_sites.append(AdsorbateSite(
        reactant     = reactant_smiles,
        n_atoms      = len(atom_cliques),
        atom_cliques = list(atom_cliques),
        positions    = pos_arr,
        iso_class    = len(adsorbate_sites),
        members      = [list(atom_cliques)],
        ego_graph    = ego_graph,
    ))


# ---------------------------------------------------------------------------
# Graph materialisation
# ---------------------------------------------------------------------------

def _next_node_id(G: nx.Graph) -> int:
    """Smallest integer node id strictly greater than all existing ids."""
    if not G.nodes:
        return 0
    return int(max(int(n) for n in G.nodes if isinstance(n, (int, np.integer)))) + 1


def _remove_adsorbate_nodes(G: nx.Graph, smiles: str) -> None:
    """Drop all adsorbate-site nodes carrying ``reactant == smiles``."""
    stale = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "adsorbate" and d.get("reactant") == smiles
    ]
    if stale:
        G.remove_nodes_from(stale)


def _member_positions(
    reactant,
    atom_cliques: list,
    pos_index: dict[str, dict[frozenset, np.ndarray]],
    elements: list[str],
    G: nx.Graph,
    pbc: np.ndarray,
) -> np.ndarray | None:
    """Reconstruct ``(n_atoms, 3)`` Cartesians for one member.

    Returns ``None`` if any bonded clique has no cached position.
    """
    bonded: list[int] = []
    bonded_pos: list[np.ndarray] = []
    for i, clq in enumerate(atom_cliques):
        if clq is None:
            continue
        p = pos_index.get(elements[i], {}).get(frozenset(clq))
        if p is None:
            return None
        bonded.append(i)
        bonded_pos.append(p)
    if not bonded:
        return None
    return _full_adsorbate_positions(
        reactant, bonded, np.array(bonded_pos, dtype=float), G, pbc
    )


def _materialise_adsorbate_nodes(
    G: nx.Graph,
    reactant,
    adsorbate_sites: list[AdsorbateSite],
) -> None:
    """Create one connected adsorbate-site subgraph per member placement on *G*.

    For each :class:`AdsorbateSite` and each member, ``n_atoms`` nodes are
    added with::

        type           = "adsorbate"
        occupied       = False          ← invisible to ego-graph BFS
        reactant       = reactant.smiles
        iso_class      = ms.iso_class
        reactant_index = atom index within the reactant
        element        = atom element symbol
        clique         = frozenset of bonded surface atom ids (or None)
        is_bonded      = clique is not None
        siblings       = tuple of the other node ids in the same placement

    Intramolecular edges mirror ``reactant.graph`` topology
    (``intra_adsorbate=True``).  Bonded atoms receive ``anchor_bond=True``
    edges to every surface atom in their clique.

    Per-member node ids are pushed onto ``ms.member_node_ids``.
    """
    smiles      = reactant.smiles
    react_sym   = reactant.atoms.get_chemical_symbols()
    react_radii = (
        [float(d.get("covalent_radius", 0.0))
         for _, d in reactant.graph.nodes(data=True)]
        if reactant.graph is not None
        else [0.0] * len(react_sym)
    )
    intra_edges = (
        list(reactant.graph.edges())
        if reactant.graph is not None else []
    )

    _remove_adsorbate_nodes(G, smiles)

    pos_index = _clique_position_index(G, react_sym)
    pbc = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)

    for ms in adsorbate_sites:
        ms.member_node_ids = []
        rep_positions = np.asarray(ms.positions, dtype=float)

        for atom_cliques in ms.members:
            # Geometry for this member (falls back to representative if any
            # bonded clique is missing from the position index).
            positions = _member_positions(
                reactant, atom_cliques, pos_index, react_sym, G, pbc
            )
            if positions is None:
                positions = rep_positions
            positions = np.asarray(positions, dtype=float)

            # Allocate a contiguous block of new node ids.
            base     = _next_node_id(G)
            node_ids = [base + i for i in range(len(react_sym))]

            # Add one node per reactant atom.
            for i, nid in enumerate(node_ids):
                clq = atom_cliques[i]
                G.add_node(
                    nid,
                    element         = react_sym[i],
                    position        = positions[i].copy(),
                    index           = nid,
                    type            = "adsorbate",
                    occupied        = False,
                    covalent_radius = react_radii[i],
                    reactant        = smiles,
                    iso_class       = int(ms.iso_class),
                    reactant_index  = int(i),
                    clique          = (frozenset(clq) if clq is not None else None),
                    k               = (len(clq) if clq is not None else 0),
                    is_bonded       = (clq is not None),
                    siblings        = tuple(n for n in node_ids if n != nid),
                    optimised       = False,
                )

            # Intramolecular edges (mirror reactant.graph topology).
            for u, v in intra_edges:
                if int(u) >= len(node_ids) or int(v) >= len(node_ids):
                    continue
                a, b = node_ids[int(u)], node_ids[int(v)]
                d = float(np.linalg.norm(
                    positions[int(u)] - positions[int(v)]
                ))
                G.add_edge(a, b, distance=d, offset=(0, 0, 0),
                           intra_adsorbate=True)

            # Surface attachment edges for bonded atoms.
            for i, nid in enumerate(node_ids):
                clq = atom_cliques[i]
                if clq is None:
                    continue
                for surf_id in clq:
                    if surf_id not in G:
                        continue
                    d = float(np.linalg.norm(
                        np.asarray(G.nodes[surf_id]["position"], dtype=float)
                        - positions[i]
                    ))
                    G.add_edge(nid, surf_id, distance=d, offset=(0, 0, 0),
                               anchor_bond=True)

            ms.member_node_ids.append(node_ids)


# ---------------------------------------------------------------------------
# push_member_positions_to_graph
# ---------------------------------------------------------------------------

def push_member_positions_to_graph(
    G: nx.Graph,
    adsorbate_site: AdsorbateSite,
    member_index: int,
    positions: np.ndarray,
) -> None:
    """Write *positions* for one :class:`AdsorbateSite` member into *G*.

    Updates ``G.nodes[nid]["position"]`` for every node in
    ``adsorbate_site.member_node_ids[member_index]`` and refreshes all
    incident edge distances (``intra_adsorbate`` and ``anchor_bond``).

    Parameters
    ----------
    G : nx.Graph
    adsorbate_site : AdsorbateSite
    member_index : int
        Index into ``adsorbate_site.member_node_ids``.
    positions : np.ndarray, shape (n_atoms, 3)
    """
    if member_index >= len(adsorbate_site.member_node_ids):
        return
    node_ids = adsorbate_site.member_node_ids[member_index]
    pos_arr  = np.asarray(positions, dtype=float)
    if pos_arr.shape[0] != len(node_ids):
        raise ValueError(
            f"positions has {pos_arr.shape[0]} rows but member "
            f"{member_index} has {len(node_ids)} nodes."
        )
    for i, nid in enumerate(node_ids):
        if nid not in G:
            continue
        G.nodes[nid]["position"]  = pos_arr[i].copy()
        G.nodes[nid]["optimised"] = True
    # Refresh all incident edge distances.
    for nid_a in node_ids:
        if nid_a not in G:
            continue
        p_a = np.asarray(G.nodes[nid_a]["position"], dtype=float)
        for nid_b in G.neighbors(nid_a):
            if nid_b not in G:
                continue
            p_b = np.asarray(G.nodes[nid_b]["position"], dtype=float)
            G.edges[nid_a, nid_b]["distance"] = float(np.linalg.norm(p_a - p_b))


# ---------------------------------------------------------------------------
# Public API — find_adsorbate_sites
# ---------------------------------------------------------------------------

def find_adsorbate_sites(
    G: nx.Graph,
    reactant,
    *,
    bond_tolerance: float = BOND_TOLERANCE,
    n_shells_anchor: int | None = None,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    include_partial: bool = True,
    require_anchors: bool = True,
    auto_grow_shells: bool = True,
    max_shell_retries: int = 3,
    require_surface_connected: bool = True,
    max_pair_shells: int = MAX_PAIR_SHELLS,
    verbose: bool = False,
) -> list[AdsorbateSite]:
    """Universal N-atom adsorbate site enumerator (N ≥ 2).

    Parameters
    ----------
    G : nx.Graph
        Surface graph from :func:`autokmc.graph.build_graph`.  Anchor sites
        for every anchor element are lazily computed if not already on *G*.
    reactant : :class:`autokmc.reactants.Reactant`
        Gas-phase molecule.  ``len(reactant.atoms) >= 2`` is required.
    bond_tolerance : float
        Allowed Å deviation between any surface anchor-pair distance and the
        corresponding intramolecular distance.  Default 0.4 Å.
    n_shells_anchor : int or None
        Shell depth for reducing single-atom sites into iso-classes (used to
        seed the *first* anchor of each subset).  ``None`` picks
        ``max(1, ceil(reach / 2.5 Å))`` automatically.
    n_shells_pair : int
        Minimum ego-graph depth for iso-class discrimination of adsorbate
        placements.  Default 1.
    co_factor, opt_factor, repulsion_weight :
        Forwarded to :func:`~autokmc.find_anchors.find_anchor_sites` if
        anchor sites have not yet been computed for an element.
    include_partial : bool
        Enumerate all non-empty anchor subsets (including singletons).
        Set to ``False`` to try only the full-anchor subset.  Default True.
    require_anchors : bool
        Raise ``ValueError`` when the reactant has no anchor atoms.
    auto_grow_shells : bool
        Re-enumerate at a larger ``n_shells_anchor`` if any emitted placement
        spans more hops than the current ego depth (up to *max_shell_retries*).
    max_shell_retries : int
        Maximum grow-retry attempts.  Default 3.
    require_surface_connected : bool
        Drop placements whose bonded cliques cannot reach each other through
        surface edges within *max_pair_shells* hops.  Default True.
    max_pair_shells : int
        Hard cap on the per-placement connectivity check.  Default 10.
    verbose : bool
        Print per-step progress to stdout.

    Returns
    -------
    list[AdsorbateSite]
        One entry per iso-class.  Also stored at
        ``G.graph["adsorbate_sites"][reactant.smiles]``.
    """
    n_atoms = len(reactant.atoms)
    if n_atoms < 2:
        raise ValueError(
            f"find_adsorbate_sites requires ≥ 2 atoms, got {n_atoms}."
        )

    anchors = sorted(int(i) for i in reactant.anchor_atoms)
    if not anchors:
        if require_anchors:
            raise ValueError("Reactant has no anchor atoms.")
        return []

    # Remove stale adsorbate nodes from a previous call on the same SMILES.
    _remove_adsorbate_nodes(G, reactant.smiles)

    elements: dict[int, str] = {
        i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)
    }
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    # (N, N) intramolecular distance matrix.
    D = np.linalg.norm(
        react_pos[:, None, :] - react_pos[None, :, :], axis=-1
    )

    # Heuristic shell depth from molecular reach.
    if n_shells_anchor is None:
        if len(anchors) >= 2:
            reach = float(max(D[i, j] for i in anchors for j in anchors if i < j))
        else:
            reach = float(D[anchors[0]].max())
        n_shells_eff = _suggested_n_shells(reach)
    else:
        n_shells_eff = int(n_shells_anchor)

    cell, cell_inv, pbc, use_mic = _get_cell(G)
    orbit_id   = _orbit_id_of(reactant)
    node_match = isomorphism.categorical_node_match("element", "X")
    apsp = (
        _get_surface_apsp(G, cutoff=max_pair_shells)
        if require_surface_connected else None
    )

    if verbose:
        print(
            f"\nfind_adsorbate_sites: smiles={reactant.smiles!r}  "
            f"N={n_atoms}  anchors={anchors}  "
            f"n_shells_anchor={n_shells_eff}  n_shells_pair={n_shells_pair}  "
            f"tol={bond_tolerance} Å"
        )

    # ── Orbit-canonicalised anchor subsets ───────────────────────────────
    if include_partial:
        all_subsets: list[tuple[int, ...]] = []
        for r in range(1, len(anchors) + 1):
            all_subsets.extend(combinations(anchors, r))
    else:
        all_subsets = [tuple(anchors)]

    seen_subset_keys: set = set()
    canonical_subsets: list[tuple[int, ...]] = []
    for sub in all_subsets:
        key = _canonical_subset_key(sub, orbit_id)
        if key not in seen_subset_keys:
            seen_subset_keys.add(key)
            canonical_subsets.append(sub)

    # ── Enumeration pass ─────────────────────────────────────────────────
    def _run_pass(depth: int) -> list[AdsorbateSite]:
        anchor_elements = {elements[i] for i in anchors}
        for el in anchor_elements:
            _ensure_anchor_sites(
                G, el,
                co_factor=co_factor,
                opt_factor=opt_factor,
                repulsion_weight=repulsion_weight,
                n_shells=depth,
                verbose=verbose,
            )

        iso_by_elem: dict[str, list[AnchorSite]] = {
            el: G.graph["anchor_sites"][el] for el in anchor_elements
        }
        raw_by_elem: dict[str, list[tuple[frozenset, np.ndarray]]] = {
            el: _all_raw_sites_with_positions(G, el) for el in anchor_elements
        }

        # Build one kd-tree per anchor element so the inner chain-placement
        # loop becomes O(log N + h) instead of O(N) per step.  The tree
        # handles MIC for orthogonal periodic cells natively (boxsize),
        # tiles ±1 images for non-orthogonal periodic cells, and falls
        # back to a plain tree for nanoparticles.
        kdtree_by_elem: dict[str, _AnchorKDTree] = {}
        for el, raws in raw_by_elem.items():
            if raws:
                positions = np.array([p for _c, p in raws], dtype=float)
                cliques   = [c for c, _p in raws]
            else:
                positions = np.empty((0, 3), dtype=float)
                cliques   = []
            kdtree_by_elem[el] = _AnchorKDTree(
                positions, cliques, cell, cell_inv, pbc, use_mic,
            )

        adsorbate_sites: list[AdsorbateSite] = []
        seen_signatures: set = set()
        rejected_disconnected = 0

        def _emit(
            bonded: list[int],
            assigned_pos: dict[int, np.ndarray],
            assigned_clique: dict[int, frozenset],
        ) -> None:
            nonlocal rejected_disconnected
            atom_cliques: list = [None] * n_atoms
            for i in bonded:
                atom_cliques[i] = assigned_clique[i]

            # Surface-connectivity guard.
            if apsp is not None:
                bonded_cliques = [c for c in atom_cliques if c is not None]
                needed = _min_ego_depth_for_connectivity(bonded_cliques, apsp)
                if needed >= 10 ** 6 or needed > max_pair_shells:
                    rejected_disconnected += 1
                    return

            bonded_positions = np.array(
                [assigned_pos[i] for i in bonded], dtype=float
            )
            positions = _full_adsorbate_positions(
                reactant, bonded, bonded_positions, G, pbc
            )

            # Ego subgraph: n_shells_pair-shell BFS from union of bonded cliques.
            union: set[int] = set()
            for c in atom_cliques:
                if c is not None:
                    union |= set(c)
            ego = _build_ego_graph(G, frozenset(union), n_shells_pair)

            _try_merge_or_new(
                adsorbate_sites,
                reactant_smiles = reactant.smiles,
                atom_cliques    = atom_cliques,
                positions       = positions,
                ego_graph       = ego,
                node_match      = node_match,
                seen_signatures = seen_signatures,
            )

        def _recurse(
            bonded: list[int],
            idx: int,
            assigned_pos: dict[int, np.ndarray],
            assigned_clique: dict[int, frozenset],
            last_atom: int,
        ) -> None:
            if idx == len(bonded):
                _emit(bonded, assigned_pos, assigned_clique)
                return
            next_atom = bonded[idx]
            next_el   = elements[next_atom]
            kd        = kdtree_by_elem[next_el]

            # KD-tree prefilter: query the annulus around the most recently
            # placed anchor only.  All other distance constraints are then
            # verified explicitly on the (small) candidate set.
            p_anchor = assigned_pos[last_atom]
            target   = float(D[last_atom, next_atom])
            r_min    = target - bond_tolerance
            r_max    = target + bond_tolerance
            candidates = kd.query_annulus(p_anchor, r_min, r_max)

            for cand_idx in candidates:
                clique = kd.cliques[cand_idx]
                pos    = kd.positions[cand_idx]

                # No two adsorbate atoms may share the same surface clique.
                if any(clique == c for c in assigned_clique.values()):
                    continue

                # Verify remaining distance constraints (every prior anchor
                # other than ``last_atom``, which the kd-tree already
                # checked).
                ok = True
                for prev_idx, p_prev in assigned_pos.items():
                    if prev_idx == last_atom:
                        continue
                    target_prev = float(D[prev_idx, next_atom])
                    d = _mic_distance(p_prev, pos, cell, cell_inv, pbc, use_mic)
                    if abs(d - target_prev) > bond_tolerance:
                        ok = False
                        break
                if not ok:
                    continue

                assigned_pos[next_atom]    = pos
                assigned_clique[next_atom] = clique
                _recurse(bonded, idx + 1, assigned_pos, assigned_clique,
                         last_atom=next_atom)
                del assigned_pos[next_atom]
                del assigned_clique[next_atom]

        if verbose:
            print(
                f"  [depth={depth}]  {len(canonical_subsets)} canonical subsets "
                f"(from {len(all_subsets)} raw)"
            )

        for sub in canonical_subsets:
            bonded   = sorted(sub)
            first    = bonded[0]
            first_el = elements[first]
            for iso_first in iso_by_elem[first_el]:
                if iso_first.position is None:
                    continue
                assigned_pos    = {first: np.asarray(iso_first.position, float)}
                assigned_clique = {first: iso_first.representative}
                _recurse(bonded, 1, assigned_pos, assigned_clique,
                         last_atom=first)

        if verbose and rejected_disconnected:
            print(
                f"  [depth={depth}]  dropped {rejected_disconnected} "
                f"disconnected placement(s)  (max_pair_shells={max_pair_shells})"
            )

        return adsorbate_sites

    # ── Retry loop: auto-grow n_shells_anchor ────────────────────────────
    adsorbate_sites = _run_pass(n_shells_eff)
    retries = 0
    while auto_grow_shells and apsp is not None and retries < max_shell_retries:
        needed = _required_n_shells(G, adsorbate_sites, apsp)
        if needed <= n_shells_eff:
            break
        if verbose:
            print(
                f"  ⚠  bonded cliques span {needed} hops but iso depth was "
                f"{n_shells_eff} — growing to {needed} and re-enumerating"
            )
        n_shells_eff    = needed
        adsorbate_sites = _run_pass(n_shells_eff)
        retries += 1

    if verbose:
        n_full = sum(
            1 for ms in adsorbate_sites
            if all(ms.atom_cliques[i] is not None for i in anchors)
        )
        n_partial     = len(adsorbate_sites) - n_full
        total_members = sum(len(ms.members) for ms in adsorbate_sites)
        print(
            f"  → {len(adsorbate_sites)} iso-classes  "
            f"({n_full} full-anchor + {n_partial} partial)  "
            f"{total_members} placements total"
        )

    # ── Materialise nodes, then persist to G.graph ───────────────────────
    _materialise_adsorbate_nodes(G, reactant, adsorbate_sites)
    G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = adsorbate_sites

    _log.debug(
        "find_adsorbate_sites: %r done — %d iso-classes, %d total placements",
        reactant.smiles,
        len(adsorbate_sites),
        sum(len(ms.members) for ms in adsorbate_sites),
    )
    return adsorbate_sites


# ---------------------------------------------------------------------------
# Rigid-body refinement
# ---------------------------------------------------------------------------

def _rotation_from_axis_angle(rotvec: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix from a Rodrigues axis-angle vector."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _surface_atoms_array(
    G: nx.Graph,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return ``(positions[N,3], r_cov[N], node_ids[N])`` for surface nodes.

    Cached on ``G.graph["_surface_atoms_array_cache"]`` keyed by the
    sorted tuple of surface node ids so that repeated calls within one
    optimisation pass don't rebuild the (potentially huge) array.
    Invalidate by clearing the cache or by adding/removing surface nodes
    (the key changes automatically).
    """
    from ase.data import covalent_radii as _RC, atomic_numbers as _AN

    surf_ids = tuple(sorted(
        n for n, d in G.nodes(data=True) if d.get("type") == "surface"
    ))
    cache = G.graph.get("_surface_atoms_array_cache")
    if isinstance(cache, dict) and cache.get("_key") == surf_ids:
        return cache["pos"], cache["r_cov"], list(cache["ids"])

    pos:   list = []
    r_cov: list = []
    for n in surf_ids:
        d = G.nodes[n]
        pos.append(d["position"])
        r_cov.append(float(_RC[_AN[d["element"]]]))
    pos_arr   = np.asarray(pos,   dtype=float).reshape(-1, 3)
    r_cov_arr = np.asarray(r_cov, dtype=float)
    G.graph["_surface_atoms_array_cache"] = {
        "_key": surf_ids,
        "pos":  pos_arr,
        "r_cov": r_cov_arr,
        "ids":  list(surf_ids),
    }
    return pos_arr, r_cov_arr, list(surf_ids)


def optimise_adsorbate_site_positions(
    G: nx.Graph,
    smiles: str,
    reactant,
    *,
    restraint_weight: float = 10.0,
    repulsion_weight: float = 1.0,
    contact_factor: float = CONTACT_FACTOR,
    standoff_factor: float = STANDOFF_FACTOR,
    n_restarts: int = N_RESTARTS,
    try_flip: bool = True,
    max_iter: int = 100,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    verbose: bool = False,
) -> list[AdsorbateSite]:
    """Rigid-body refinement of every :attr:`AdsorbateSite.positions` for
    ``G.graph["adsorbate_sites"][smiles]``.

    Treats each placement as 6 rigid-body DOF (3 translation + 3 axis-angle
    rotation) applied to the gas-phase reactant geometry.  Minimises::

        E(R,t) = restraint_weight × Σ_{bonded i}  ‖p_i − p*_i‖²
               + repulsion_weight × Σ_{adsorbate a, surface s}
                                       max(0, R_min − d_as)²

    where ``p*_i`` is the clique centroid lifted by
    ``standoff_factor × (r_cov_a + ⟨r_cov_s⟩)`` along the local outward
    normal and ``R_min = contact_factor × (r_cov_a + r_cov_s)``.

    Multi-start: ``n_restarts`` rotational kicks about the local outward
    normal.  When ``try_flip=True`` (default) each kick is also tried with
    a 180° in-plane flip (essential for adsorbates with unbonded atoms that
    must point away from the surface).

    After refining the representative, Kabsch ego-alignment propagates the
    new geometry to every other member via
    :func:`~autokmc.find_anchors._kabsch_align_ego`.

    Parameters
    ----------
    G : nx.Graph
    smiles : str
        Key into ``G.graph["adsorbate_sites"]``.
    reactant : :class:`autokmc.reactants.Reactant`
    restraint_weight, repulsion_weight, contact_factor, standoff_factor :
        Objective-function weights / scales.
    n_restarts : int
        Rotational restarts per placement.  Default 6.
    try_flip : bool
        Also perform 180° in-plane flips.  Default True.
    max_iter : int
        L-BFGS-B iteration cap per restart.  Default 100.
    n_shells_pair : int
        Ego depth for Kabsch propagation to members.  Default 1.
    verbose : bool

    Returns
    -------
    list[AdsorbateSite]
        The same list at ``G.graph["adsorbate_sites"][smiles]``;
        :attr:`AdsorbateSite.positions` updated in place.

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    KeyError
        If *smiles* is not in ``G.graph["adsorbate_sites"]``.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "optimise_adsorbate_site_positions requires scipy "
            "(pip install scipy)."
        ) from e

    ads_dict = G.graph.get("adsorbate_sites", {})
    if smiles not in ads_dict:
        raise KeyError(
            f"No adsorbate sites for SMILES {smiles!r}; "
            "call find_adsorbate_sites first."
        )
    adsorbate_sites: list[AdsorbateSite] = ads_dict[smiles]
    if not adsorbate_sites:
        return adsorbate_sites

    from ase.data import covalent_radii as _RC, atomic_numbers as _AN
    q_ref    = np.asarray(reactant.atoms.get_positions(), dtype=float)
    n_atoms  = q_ref.shape[0]
    ads_rcov = np.array([
        float(_RC[_AN[reactant.graph.nodes[i]["element"]]])
        for i in range(n_atoms)
    ])
    q_centroid = q_ref.mean(axis=0)
    q_centred  = q_ref - q_centroid

    surf_pos, surf_r, surf_ids = _surface_atoms_array(G)
    surf_id_to_row = {nid: i for i, nid in enumerate(surf_ids)}
    elements = {i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)}
    site_idx = _clique_position_index(G, list(elements.values()))
    cell, cell_inv, pbc, use_mic = _get_cell(G)

    def _refine(ms: AdsorbateSite) -> tuple[np.ndarray, float, float, int]:
        bonded_idx = [i for i, c in enumerate(ms.atom_cliques) if c is not None]

        targets: list[np.ndarray] = []
        exclude_surf: set = set()
        for i in bonded_idx:
            el     = elements[i]
            clique = ms.atom_cliques[i]
            base   = site_idx.get(el, {}).get(
                frozenset(clique),
                np.asarray(ms.positions[i], dtype=float),
            )
            if standoff_factor > 0.0:
                rows    = [surf_id_to_row[int(s)] for s in clique
                           if int(s) in surf_id_to_row]
                r_s_avg = float(np.mean(surf_r[rows])) if rows else 0.0
                standoff = standoff_factor * (ads_rcov[i] + r_s_avg)
                n_hat    = _outward_normal_at(G, base, pbc)
                targets.append(base + standoff * n_hat)
            else:
                targets.append(base)
            exclude_surf.update(int(s) for s in clique)

        target_arr = np.array(targets, dtype=float).reshape(-1, 3)
        keep       = np.array([n not in exclude_surf for n in surf_ids], dtype=bool)
        free_pos   = surf_pos[keep]
        free_r     = surf_r[keep]

        cur_pos      = np.asarray(ms.positions, dtype=float)
        cur_centroid = cur_pos.mean(axis=0)
        R0, _        = _kabsch(q_ref, cur_pos)
        n_hat        = _outward_normal_at(G, cur_centroid, pbc)

        if try_flip:
            e1 = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(e1, n_hat))) > 0.9:
                e1 = np.array([0.0, 1.0, 0.0])
            flip_ax  = e1 - float(np.dot(e1, n_hat)) * n_hat
            flip_ax /= float(np.linalg.norm(flip_ax))
            R_flip   = _rotation_from_axis_angle(flip_ax * np.pi)
        else:
            R_flip = None

        def _make_pose(R_base):
            def _pose(x):
                return q_centred @ (R_base @ _rotation_from_axis_angle(x[3:])).T \
                       + (cur_centroid + x[:3])
            return _pose

        def _make_energy(pose_fn):
            def _energy(x):
                p = pose_fn(x)
                E = 0.0
                if bonded_idx:
                    disp = p[bonded_idx] - target_arr
                    E += restraint_weight * float(np.einsum("ij,ij->", disp, disp))
                if free_pos.size:
                    for ai in range(n_atoms):
                        dv = free_pos - p[ai]
                        if use_mic and cell_inv is not None:
                            frac = dv @ cell_inv
                            for ax in range(3):
                                if pbc[ax]:
                                    frac[:, ax] -= np.round(frac[:, ax])
                            dv = frac @ cell
                        d2    = np.einsum("ij,ij->i", dv, dv)
                        R_min = contact_factor * (ads_rcov[ai] + free_r)
                        delta = R_min - np.sqrt(d2 + 1e-12)
                        pos_c = np.maximum(delta, 0.0)
                        E += repulsion_weight * float(np.dot(pos_c, pos_c))
                return E
            return _energy

        E0 = _make_energy(_make_pose(R0))(np.zeros(6))

        n_starts  = max(1, int(n_restarts))
        bases     = []
        for k in range(n_starts):
            angle  = 2.0 * np.pi * k / n_starts
            R_kick = (_rotation_from_axis_angle(n_hat * angle)
                      if angle != 0.0 else np.eye(3))
            R_b = R_kick @ R0
            bases.append(R_b)
            if R_flip is not None:
                bases.append(R_flip @ R_b)

        best_E, best_pose, best_k = np.inf, None, 0
        for k, R_base in enumerate(bases):
            energy_k = _make_energy(_make_pose(R_base))
            try:
                res = minimize(energy_k, np.zeros(6), method="L-BFGS-B",
                               options={"maxiter": max_iter, "ftol": 1e-7})
                E_k = float(res.fun)
                p_k = _make_pose(R_base)(res.x)
            except Exception:
                continue
            if E_k < best_E:
                best_E, best_pose, best_k = E_k, p_k, k

        if best_pose is None:
            return cur_pos, E0, E0, 0
        return best_pose, E0, best_E, best_k

    if verbose:
        print(
            f"optimise_adsorbate_site_positions: smiles={smiles!r}  "
            f"placements={len(adsorbate_sites)}  "
            f"restraint={restraint_weight}  repulsion={repulsion_weight}  "
            f"contact={contact_factor}  standoff={standoff_factor}  "
            f"restarts={n_restarts}  flip={try_flip}"
        )

    for ms in adsorbate_sites:
        try:
            new_pos, E0, Ef, best_idx = _refine(ms)
        except Exception as exc:
            if verbose:
                print(f"  iso-class {ms.iso_class}: refinement failed ({exc!r})")
            continue

        rms = float(np.sqrt(np.mean(
            np.sum((new_pos - np.asarray(ms.positions)) ** 2, axis=1)
        )))
        ms.positions = new_pos

        # Push representative positions and Kabsch-propagate to other members.
        n_propagated = 0
        if ms.member_node_ids:
            push_member_positions_to_graph(G, ms, 0, new_pos)

            rep_seed: frozenset = frozenset(
                int(n)
                for c in ms.atom_cliques if c is not None
                for n in c
            )
            if rep_seed:
                cell_arr, cell_inv_arr, pbc_arr, use_mic_arr = _get_cell(G)
                for m_idx in range(1, len(ms.members)):
                    mem_seed: frozenset = frozenset(
                        int(n)
                        for c in ms.members[m_idx] if c is not None
                        for n in c
                    )
                    if not mem_seed:
                        continue
                    R, t = _kabsch_align_ego(
                        G,
                        rep_seed, mem_seed, n_shells_pair,
                        cell_arr, cell_inv_arr, pbc_arr, use_mic_arr,
                    )
                    if R is None or t is None:
                        continue
                    push_member_positions_to_graph(G, ms, m_idx, new_pos @ R.T + t)
                    n_propagated += 1

        if verbose:
            print(
                f"  iso-class {ms.iso_class:3d}:  "
                f"E {E0:9.4f} → {Ef:9.4f}  ΔRMS {rms:6.3f} Å  "
                f"best_restart={best_idx}  "
                f"propagated {n_propagated}/{max(0, len(ms.members) - 1)} members"
            )

    return adsorbate_sites

