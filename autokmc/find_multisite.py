"""
autokmc.find_multisite
======================
N-atom adsorbate site enumeration on a surface graph.

Where ``default_sites`` finds all single-atom adsorption sites for one
*element*, this module finds all geometrically feasible placements of a
multi-atom *molecule* (a :class:`~autokmc.reactants.Reactant`).  A
placement is stored as a *subgraph* — one surface clique per molecular
atom — rather than a single node.

The same code path handles diatomics, triatomics and arbitrary N-atom
adsorbates: there is **one** universal enumerator,
:func:`find_multisites`.  ``find_multisites_for_reactant`` is kept as a
thin back-compat wrapper.

Strategy
--------
1. **Pick anchors.**  Only atoms in ``reactant.anchor_atoms`` (convex-
   hull-exposed; sterically unhindered) are eligible to bond to a
   surface clique.  Every other atom floats and is positioned by rigid
   alignment of the gas-phase reactant geometry.

2. **Choose shell depth from molecular reach.**  Default
   ``n_shells_anchor = max(1, ceil(reach / d_nn))`` where ``reach`` is
   the largest intramolecular anchor-pair distance (or the molecule's
   diameter for single-anchor reactants) and ``d_nn ≈ 2.5 Å`` is a
   typical metal nearest-neighbour distance.

3. **Backtracking chain placement.**  For every non-empty subset of
   anchors (orbit-canonicalised so symmetry-equivalent subsets are
   enumerated only once):

   * The first anchor is placed at every unique iso-class site of its
     element at depth ``n_shells_anchor``.
   * Subsequent anchors are placed at any *raw* single-atom site of
     their element whose MIC distance to **every** previously-placed
     anchor matches the intramolecular distance within
     ``bond_tolerance``.

4. **Surface-connectivity guard.**  The bonded cliques of every
   placement must be mutually reachable through ``type=="surface"``
   edges.  The per-placement ego depth used to build the stored *site
   graph* is automatically grown until those cliques are connected
   (capped by ``max_pair_shells``); placements that cannot be connected
   are dropped.

5. **Reduce by isomorphism.**  Group placements by isomorphism of the
   union-of-cliques ego subgraph (with element-label matching).
   Orbit-equivalent placements collapse automatically.

6. **Auto-grow ``n_shells_anchor`` if needed.**  If any emitted
   placement has bonded cliques further apart than the iso-class ego
   graph could see, the enumeration is repeated at a larger depth.

7. **Optional rigid-body refinement.**  :func:`optimise_multisite_positions`
   does a calculator-free L-BFGS-B refinement of each placement's 6
   rigid-body DOF, restraining bonded atoms toward their site targets
   while penalising adsorbate ↔ surface clipping.

Storage
-------
Results are written to ``G.graph["multisites"][reactant.smiles]`` as a
``list[MultiSite]``.  The single-atom data in
``G.graph["sites"]`` / ``["unique_sites"]`` / ``["site_positions"]`` is
**never mutated**.

Public API
----------
* :class:`MultiSite`                       -- one iso-class of multi-atom placements
* :func:`find_multisites`                  -- universal N-atom enumerator (N>=2)
* :func:`find_multisites_for_reactant`     -- back-compat alias
* :func:`optimise_multisite_positions`     -- rigid-body refinement of MultiSite.positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.default_sites import (
    find_sites_for_element,
    reduce_sites_by_isomorphism,
    optimise_site_positions,
    propagate_positions_to_iso_classes,
    _build_clique_ego,
    _iso_prefilter_key,
)
from autokmc.cache import get_cache
from autokmc.constants import (
    BOND_TOLERANCE,
    CO_FACTOR,
    CONTACT_FACTOR,
    MAX_PAIR_SHELLS,
    NN_DISTANCE,
    N_SHELLS_DEFAULT,
    OPT_FACTOR,
    REPULSION_WEIGHT,
)
from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiSite:
    """One isomorphism class of multi-atom adsorbate placements.

    Attributes
    ----------
    smiles : str
        SMILES of the reactant this placement was generated from.
    n_atoms : int
        Number of atoms in the reactant.
    atom_cliques : list[frozenset[int] | None]
        Surface clique that each reactant atom binds to, in reactant
        atom-index order.  ``None`` means that atom is **not** bonded to
        the surface in this placement (e.g. the dangling end of a
        physisorbed diatomic).
    positions : np.ndarray, shape (n_atoms, 3)
        Geometric Cartesian positions for each adsorbate atom (Å).
        Bonded atoms use the ``IsoClass.position`` of their site;
        unbonded atoms are placed one bond-length outward from the
        surface (see :func:`_unbonded_position`).
    iso_class : int
        0-based index within the per-reactant placement list.
    members : list[list[frozenset | None]]
        All raw ``atom_cliques`` tuples that were folded into this
        iso-class (the first entry is the representative).
    ego_graph : nx.Graph | None
        ``n_shells_pair`` ego-subgraph used for isomorphism matching.
    """
    smiles       : str
    n_atoms      : int
    atom_cliques : list
    positions    : Any
    iso_class    : int
    members      : list   = field(default_factory=list)
    ego_graph    : Any    = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suggested_n_shells(bond_length: float,
                        nn_distance: float = NN_DISTANCE) -> int:
    """Heuristic shell depth from the molecule's geometric reach."""
    return max(1, int(np.ceil(bond_length / nn_distance)))


def _ensure_default_sites(
    G: nx.Graph,
    element: str,
    n_shells: int,
    *,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    verbose: bool = False,
) -> None:
    """Lazily run the ``default_sites`` pipeline for *element* on *G*.

    No-op for any stage already cached in ``G.graph``.
    """
    cache = get_cache(G)
    if element not in cache.sites:
        find_sites_for_element(G, element, co_factor=co_factor, verbose=verbose)
    needs_reduce = (element not in cache.unique_sites
                    or n_shells not in cache.unique_sites[element])
    if needs_reduce:
        reduce_sites_by_isomorphism(G, element, n_shells=n_shells, verbose=verbose)
    if element not in cache.site_positions:
        optimise_site_positions(
            G, element,
            opt_factor=opt_factor,
            repulsion_weight=repulsion_weight,
            verbose=verbose,
        )
    elif needs_reduce:
        # Positions were already computed at a previous shell depth; the
        # iso-classes we just created at the new depth carry
        # position=None.  Inject the cached site positions into them so
        # downstream code (which skips position=None classes) actually
        # sees them.  Without this, e.g. find_multisites for OCS — whose
        # reach forces n_shells_anchor=2 — would silently produce zero
        # placements after a previous depth-1 default_sites pass.
        propagate_positions_to_iso_classes(G, element)


def _resolve_cell(G: nx.Graph):
    """Return ``(cell, cell_inv, pbc, use_mic)`` for MIC distance work."""
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


def _mic_distance(p, q, cell, cell_inv, pbc, use_mic) -> float:
    """Minimum-image distance between two Cartesian points."""
    dv = np.asarray(q) - np.asarray(p)
    if use_mic:
        frac = dv @ cell_inv
        for i in range(3):
            if pbc[i]:
                frac[i] -= np.round(frac[i])
        dv = frac @ cell
    return float(np.linalg.norm(dv))


def _all_raw_sites_with_positions(
    G: nx.Graph, element: str
) -> list[tuple[frozenset, np.ndarray]]:
    """Flatten ``cache.sites[element]`` to ``[(clique, position), …]``."""
    cache = get_cache(G)
    sites     = cache.sites[element]
    positions = cache.site_positions[element]
    out: list[tuple[frozenset, np.ndarray]] = []
    for k in sorted(sites):
        for clique, pos in zip(sites[k], positions[k]):
            out.append((clique, np.asarray(pos, dtype=float)))
    return out


def _atoms_share_orbit(reactant, i: int, j: int) -> bool:
    """True if reactant atom indices *i* and *j* are in the same orbit."""
    for _elem, orbits in reactant.unique_nodes.items():
        for orb in orbits:
            if i in orb and j in orb:
                return True
    return False


def _outward_normal_at(G: nx.Graph, p: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Unit outward direction at point *p* (away from the surface).

    Slabs (any periodic axis): assume ``+z``.  Nanoparticles: direction
    from the geometric centre of all surface atoms to *p*.
    """
    if pbc.any():
        return np.array([0.0, 0.0, 1.0])
    surf_pos = np.array(
        [d["position"] for _, d in G.nodes(data=True) if d.get("type") == "surface"],
        dtype=float,
    )
    if not len(surf_pos):
        return np.array([0.0, 0.0, 1.0])
    n = p - surf_pos.mean(axis=0)
    norm = float(np.linalg.norm(n))
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _unbonded_position(
    G: nx.Graph, p_anchor: np.ndarray, bond_length: float, pbc: np.ndarray,
) -> np.ndarray:
    """Place an unbonded atom one bond-length outward from the anchor."""
    n_out = _outward_normal_at(G, p_anchor, pbc)
    return p_anchor + bond_length * n_out


def _placement_signature(atom_cliques) -> tuple:
    """Hashable canonical key for a per-atom-clique placement."""
    return tuple(c if c is None else frozenset(c) for c in atom_cliques)


def _try_merge_or_new(
    multisites: list[MultiSite],
    *,
    smiles: str,
    atom_cliques: list,
    positions: np.ndarray,
    ego_graph: nx.Graph,
    node_match,
    seen_signatures: set,
) -> None:
    """Append to an existing iso-class if isomorphic, else create one.

    ``seen_signatures`` is used to skip *exact* duplicate placements
    (same surface cliques in the same atom slots) — important when the
    union-of-cliques ego graph collapses ``(A, B)`` and ``(B, A)`` for
    same-element/orbit diatomics.
    """
    sig = _placement_signature(atom_cliques)
    if sig in seen_signatures:
        return
    seen_signatures.add(sig)

    bonded_pattern = tuple(c is None for c in atom_cliques)
    fkey = _iso_prefilter_key(ego_graph)

    for ms in multisites:
        if tuple(c is None for c in ms.atom_cliques) != bonded_pattern:
            continue
        if ms.ego_graph is None:
            continue
        if _iso_prefilter_key(ms.ego_graph) != fkey:
            continue
        if isomorphism.GraphMatcher(
            ego_graph, ms.ego_graph, node_match=node_match
        ).is_isomorphic():
            ms.members.append(list(atom_cliques))
            return

    multisites.append(MultiSite(
        smiles       = smiles,
        n_atoms      = len(atom_cliques),
        atom_cliques = list(atom_cliques),
        positions    = np.asarray(positions, dtype=float),
        iso_class    = len(multisites),
        members      = [list(atom_cliques)],
        ego_graph    = ego_graph,
    ))



# ---------------------------------------------------------------------------
# General N-atom enumerator
# ---------------------------------------------------------------------------

def _orbit_id_of(reactant) -> dict[int, tuple[str, int]]:
    """Map each reactant atom index to a stable orbit id ``(element, orb_idx)``.

    ``reactant.unique_nodes`` is ``{element: [orbit_set, …]}``; we flatten
    it so two atoms share an id iff they sit in the same intramolecular
    automorphism orbit.
    """
    out: dict[int, tuple[str, int]] = {}
    for elem, orbits in reactant.unique_nodes.items():
        for k, orb in enumerate(orbits):
            for i in orb:
                out[int(i)] = (elem, k)
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid alignment ``R, t`` mapping rows of *src* onto *dst*.

    Both arrays are ``(k, 3)``.  Returns ``R`` (3×3 proper rotation) and
    ``t`` (3,) such that ``src @ R.T + t ≈ dst``.
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - R @ src_c
    return R, t


def _full_adsorbate_positions(
    reactant,
    bonded_indices: list[int],
    bonded_positions: np.ndarray,
    G: nx.Graph,
    pbc: np.ndarray,
) -> np.ndarray:
    """Reconstruct Cartesian positions for every reactant atom.

    Strategy
    --------
    * ``len(bonded_indices) >= 2``: rigid Kabsch alignment of the
      gas-phase reactant geometry so the bonded atoms land on
      ``bonded_positions``.
    * ``len(bonded_indices) == 1``: translate the gas-phase geometry so
      the single bonded atom sits at its target, then add an outward
      offset to the *non-bonded* atoms (the reactant is already centred
      with the bonded atom at the surface; we lift the rest by their
      intramolecular separation along the surface normal).
    """
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    n_atoms   = react_pos.shape[0]

    if len(bonded_indices) >= 2:
        R, t = _kabsch(react_pos[bonded_indices], bonded_positions)
        return react_pos @ R.T + t

    # Single bonded anchor → pure translation; orient unbonded atoms outward.
    i0 = bonded_indices[0]
    p_target = bonded_positions[0]
    rel = react_pos - react_pos[i0]
    n_out = _outward_normal_at(G, p_target, pbc)

    # Rotate so that the centroid of the (other) atoms points along n_out.
    if n_atoms > 1:
        other = np.delete(np.arange(n_atoms), i0)
        c = rel[other].mean(axis=0)
        norm = float(np.linalg.norm(c))
        if norm > 1e-8:
            v = c / norm
            # Rodrigues rotation aligning v → n_out.
            axis = np.cross(v, n_out)
            s = float(np.linalg.norm(axis))
            cos = float(np.dot(v, n_out))
            if s > 1e-8:
                axis /= s
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R = np.eye(3) + s * K + (1.0 - cos) * (K @ K)
                rel = rel @ R.T
            elif cos < 0.0:
                rel = -rel  # antiparallel → flip
    return rel + p_target


def _canonical_subset_key(
    subset: tuple[int, ...], orbit_id: dict[int, tuple[str, int]]
) -> tuple:
    """Orbit-multiset signature used to skip equivalent anchor subsets."""
    return tuple(sorted(orbit_id[i] for i in subset))


def _clique_to_clique_max_hops(
    G: nx.Graph,
    clique_a,
    clique_b,
    *,
    apsp: dict | None = None,
) -> int:
    """Max graph-hops from *clique_a* required to reach every node of
    *clique_b* via the surface graph.

    Uses cached all-pairs shortest paths (``apsp``) when available — the
    typical path through :func:`find_multisites`, which calls
    :func:`_get_surface_apsp` once per enumeration pass.  Falls back to
    a multi-source BFS when no cache is provided.
    """
    target = set(clique_b)

    if apsp is not None:
        worst = 0
        for b in target:
            best_to_b = 10 ** 6
            for a in clique_a:
                d = apsp.get(a, {}).get(b)
                if d is None:
                    continue
                if d < best_to_b:
                    best_to_b = d
            if best_to_b >= 10 ** 6:
                return 10 ** 6
            if best_to_b > worst:
                worst = best_to_b
        return worst

    from collections import deque
    dist: dict = {n: 0 for n in clique_a}
    q: deque = deque(clique_a)
    while q:
        n = q.popleft()
        if target.issubset(dist):
            break
        for m in G.neighbors(n):
            if m not in dist:
                dist[m] = dist[n] + 1
                q.append(m)
    if any(b not in dist for b in target):
        return 10 ** 6
    return max(dist[b] for b in target)


def _required_n_shells(G: nx.Graph, multisites: list[MultiSite],
                       *, apsp: dict | None = None) -> int:
    """Largest depth needed for the iso-class ego graph to actually
    contain every other bonded clique seen in *multisites*.

    The first bonded reactant atom is the one whose clique seeds the
    ego subgraph used for iso-class discrimination; for every emitted
    placement we measure how far (in surface-graph hops) the *other*
    bonded cliques sit from that seed and return the max.
    """
    needed = 0
    for ms in multisites:
        bonded = [c for c in ms.atom_cliques if c is not None]
        if len(bonded) < 2:
            continue
        first = bonded[0]
        for other in bonded[1:]:
            d = _clique_to_clique_max_hops(G, first, other, apsp=apsp)
            if d > needed:
                needed = d
    return needed


def _surface_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the induced subgraph on nodes tagged ``type == "surface"``."""
    nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "surface"]
    return G.subgraph(nodes)


def _get_surface_apsp(G: nx.Graph, *, cutoff: int = MAX_PAIR_SHELLS) -> dict:
    """Return cached surface-only all-pairs shortest paths up to *cutoff*.

    Stored on :attr:`autokmc.cache.SiteCache.surface_apsp`.  The cache is
    invalidated implicitly whenever :func:`autokmc.cache.SiteCache.invalidate`
    is called (or whenever ``G.graph["autokmc"]`` is replaced).

    Parameters
    ----------
    G : nx.Graph
    cutoff : int
        Maximum BFS depth to compute.  Defaults to
        :data:`autokmc.constants.MAX_PAIR_SHELLS`.

    Returns
    -------
    dict[node, dict[node, int]]
        ``apsp[u][v]`` is the surface-graph hop distance from *u* to *v*
        (only entries with distance ≤ *cutoff* are populated).
    """
    cache = get_cache(G)
    apsp = cache.surface_apsp
    if isinstance(apsp, dict) and apsp.get("_cutoff") == cutoff:
        return apsp["data"]
    G_surf = _surface_subgraph(G)
    data: dict = {}
    for u in G_surf.nodes:
        # nx.single_source_shortest_path_length returns dict[node, distance]
        data[u] = dict(nx.single_source_shortest_path_length(G_surf, u, cutoff=cutoff))
    cache.surface_apsp = {"_cutoff": cutoff, "data": data}
    return data


def _bonded_cliques_surface_connected(
    G_surf: nx.Graph,
    bonded_cliques: list,
    *,
    apsp: dict | None = None,
) -> bool:
    """Are all *bonded_cliques* mutually reachable via surface-only edges?

    Uses the cached APSP table when available (constant-time lookups);
    otherwise falls back to a multi-source BFS through *G_surf*.
    """
    if len(bonded_cliques) < 2:
        return True

    if apsp is not None:
        first = bonded_cliques[0]
        if not first:
            return False
        for other in bonded_cliques[1:]:
            if not other:
                return False
            reachable = False
            for a in first:
                a_dists = apsp.get(a, {})
                if any(b in a_dists for b in other):
                    reachable = True
                    break
            if not reachable:
                return False
        return True

    from collections import deque
    seeds = set(bonded_cliques[0])
    targets = set()
    for c in bonded_cliques[1:]:
        targets |= set(c)
    if not seeds.issubset(G_surf):
        return False
    if not targets.issubset(G_surf):
        return False
    visited = set(seeds)
    q: deque = deque(seeds)
    while q and not targets.issubset(visited):
        n = q.popleft()
        for m in G_surf.neighbors(n):
            if m not in visited:
                visited.add(m)
                q.append(m)
    return targets.issubset(visited)


def _shortest_path_between_cliques(
    G: nx.Graph,
    clique_a,
    clique_b,
    *,
    apsp: dict | None = None,
) -> int:
    """Surface-graph hop distance between the two closest nodes of
    *clique_a* and *clique_b*.

    Uses the cached APSP table when available.
    """
    if not clique_a or not clique_b:
        return 10**6
    if set(clique_a) & set(clique_b):
        return 0

    if apsp is not None:
        target = set(clique_b)
        best = 10 ** 6
        for a in clique_a:
            a_dists = apsp.get(a, {})
            for b in target:
                d = a_dists.get(b)
                if d is None:
                    continue
                if d < best:
                    best = d
        return best

    from collections import deque
    target = set(clique_b)
    dist: dict = {n: 0 for n in clique_a}
    q: deque = deque(clique_a)
    while q:
        n = q.popleft()
        if n in target:
            return dist[n]
        for m in G.neighbors(n):
            if m not in dist:
                dist[m] = dist[n] + 1
                q.append(m)
    return 10**6


def _min_ego_depth_for_connectivity(
    G_surf: nx.Graph,
    bonded_cliques: list,
    *,
    apsp: dict | None = None,
) -> int:
    """Smallest ego-graph depth at which every pair of *bonded_cliques*
    is connected through surface-only edges.

    For two cliques whose closest surface-graph hop distance is *L*, the
    shortest connecting path lies entirely within the
    ``floor(L / 2)``-shell ego of their union.  Returns the max over all
    clique pairs.
    """
    if len(bonded_cliques) < 2:
        return 0
    # When using cached APSP we don't need to verify subset-of-G_surf
    # because non-surface seeds will simply have no entries in the table.
    if apsp is None and not all(set(c).issubset(G_surf) for c in bonded_cliques):
        return 10**6
    max_d = 0
    for i in range(len(bonded_cliques)):
        for j in range(i + 1, len(bonded_cliques)):
            L = _shortest_path_between_cliques(
                G_surf, bonded_cliques[i], bonded_cliques[j], apsp=apsp
            )
            if L >= 10**6:
                return 10**6
            d = L // 2
            if d > max_d:
                max_d = d
    return max_d


def find_multisites(
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
) -> list[MultiSite]:
    """Universal N-atom adsorbate site enumerator (handles N >= 2).

    Single code path for diatomics, triatomics and arbitrary multi-atom
    adsorbates.  The previous diatomic-only fast path has been retired —
    orbit-canonical anchor-subset enumeration plus union-of-cliques
    isomorphism deduplication subsumes its swap-mirror trick — and the
    "B floats" fallback of the diatomic version is now produced by the
    general singleton-anchor branch (``include_partial=True``).

    Only atoms in ``reactant.anchor_atoms`` (convex-hull-exposed atoms,
    i.e. atoms not sterically shielded by the rest of the molecule) are
    candidates for bonding to a surface clique.  Every other atom
    floats; its position is reconstructed by rigid alignment of the
    gas-phase reactant geometry to the chosen surface anchors.

    Algorithm
    ---------
    1. Read intramolecular pairwise distances ``D[i, j]`` from the
       relaxed reactant geometry.
    2. Pick a shell depth from the largest anchor-pair distance via
       :func:`_suggested_n_shells` (overridable with
       ``n_shells_anchor``).
    3. For each non-empty subset *S* of anchors (canonicalised by
       intramolecular orbit so symmetry-equivalent subsets are
       enumerated only once):

       * The first anchor in *S* (smallest reactant index) is placed at
         every unique iso-class site of its element.
       * Subsequent anchors are placed by enumerating raw single-atom
         sites of their element and accepting any position whose MIC
         distance to **every** already-placed anchor matches the
         intramolecular distance within ``bond_tolerance``.

    4. Reconstruct full-molecule positions, build the ``n_shells_pair``
       ego-subgraph of the union of bonded cliques, and reduce by
       isomorphism via :func:`_try_merge_or_new`.

    Parameters
    ----------
    G : nx.Graph
        Surface graph from :func:`autokmc.graph.build_graph`.
    reactant : :class:`autokmc.reactants.Reactant`
        Must satisfy ``len(reactant.atoms) >= 2``.
    bond_tolerance : float
        Allowed deviation in Å between any surface anchor-pair distance
        and the corresponding intramolecular distance.  Default 0.4 Å.
    n_shells_anchor : int or None
        Shell depth for reducing single-atom sites of every anchor
        element into iso-classes (used to seed the *first* anchor of
        each subset).  ``None`` (default) picks
        ``max(1, ceil(reach / 2.5 Å))``.
    n_shells_pair : int
        Minimum shell depth for the union-of-cliques ego subgraph used
        to merge equivalent placements.  Per-placement ego depth may be
        grown larger when ``require_surface_connected`` is on.
    co_factor, opt_factor, repulsion_weight :
        Forwarded to ``default_sites`` if it has not been run yet.
    include_partial : bool
        When True (default) enumerate every non-empty anchor subset
        (including singletons → physisorbed).  When False only the
        full-anchor subset is tried.
    require_anchors : bool
        When True (default) raise ``ValueError`` if the reactant has no
        anchor atoms (degenerate hulls fall back to "every atom is an
        anchor", so this should normally be satisfied).
    auto_grow_shells : bool
        When True (default), after enumeration we measure the largest
        surface-graph hop count between the *first* bonded clique of
        any placement and the other bonded cliques.  If that distance
        exceeds the current ``n_shells_anchor`` the iso-class ego
        graph used to deduplicate first-anchor sites was too small to
        even *see* the other anchors, so we re-run with a larger depth
        (up to ``max_shell_retries`` extra passes).  This guarantees
        that two first-anchor sites which differ only via atoms that
        another anchor reaches are not collapsed into one iso-class.
    max_shell_retries : int
        Maximum number of times to grow ``n_shells_anchor`` and retry
        when ``auto_grow_shells`` is enabled.  Default 3.
    require_surface_connected : bool
        When True (default), every placement's bonded cliques must be
        mutually reachable through ``type=="surface"`` edges.  Instead
        of rejecting placements whose anchors land far apart, the ego
        depth used to build the stored *site graph* is automatically
        grown per placement until those cliques are connected (cap:
        ``max_pair_shells``).  The chosen depth is at least
        ``n_shells_pair`` and at most ``max_pair_shells``; placements
        whose cliques cannot be connected within ``max_pair_shells``
        (e.g. genuinely disconnected nanoparticle facets) are dropped
        with a verbose count.
    max_pair_shells : int
        Hard cap on the per-placement adaptive ego depth used when
        ``require_surface_connected`` is on.  Default 10 (enough for any
        reasonable adsorbate; raise it for very long flexible chains).

    Returns
    -------
    list[MultiSite]
        Also stored in ``G.graph["multisites"][reactant.smiles]``.
    """
    n_atoms = len(reactant.atoms)
    if n_atoms < 2:
        raise ValueError(
            f"find_multisites needs >=2 atoms, got {n_atoms}."
        )

    anchors = sorted(int(i) for i in reactant.anchor_atoms)
    if not anchors:
        if require_anchors:
            raise ValueError("Reactant has no anchor atoms.")
        return []

    elements = {i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)}
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    D = np.linalg.norm(
        react_pos[:, None, :] - react_pos[None, :, :], axis=-1
    )  # (N, N) intramolecular distances

    # Heuristic shell depth from molecular reach.
    if n_shells_anchor is None:
        if len(anchors) >= 2:
            reach = max(D[i, j] for i in anchors for j in anchors if i < j)
        else:
            reach = float(D[anchors[0]].max())
        n_shells_anchor_eff = _suggested_n_shells(reach)
    else:
        n_shells_anchor_eff = int(n_shells_anchor)

    if verbose:
        print(
            f"find_multisites: smiles={reactant.smiles!r}  "
            f"N={n_atoms}  anchors={anchors}  "
            f"n_shells_anchor={n_shells_anchor_eff}  "
            f"n_shells_pair={n_shells_pair}  tol={bond_tolerance} Å"
        )

    # Cache default_sites for every unique anchor element.
    anchor_elements = {elements[i] for i in anchors}
    cell, cell_inv, pbc, use_mic = _resolve_cell(G)
    orbit_id = _orbit_id_of(reactant)
    node_match = isomorphism.categorical_node_match("element", "X")
    G_surf = _surface_subgraph(G) if require_surface_connected else None
    # Pre-computed surface-only all-pairs shortest paths (one BFS per
    # surface node, cached on the graph).  Replaces the per-pair BFS the
    # legacy connectivity helpers ran for every emitted placement.
    apsp = _get_surface_apsp(G, cutoff=max_pair_shells) \
        if require_surface_connected else None

    def _run_pass(depth: int) -> list[MultiSite]:
        """One enumeration pass at iso-class depth ``depth``."""
        for el in anchor_elements:
            _ensure_default_sites(
                G, el, depth,
                co_factor=co_factor, opt_factor=opt_factor,
                repulsion_weight=repulsion_weight, verbose=verbose,
            )

        raw_sites = {
            el: _all_raw_sites_with_positions(G, el)
            for el in anchor_elements
        }

        # Iso-class lookup per anchor element (used for the *first* atom).
        iso_classes_by_elem: dict[str, list] = {}
        for el in raw_sites:
            flat: list = []
            for _k, classes in sorted(
                G.graph["unique_sites"][el][depth].items()
            ):
                flat.extend(classes)
            iso_classes_by_elem[el] = flat

        multisites: list[MultiSite] = []
        seen_signatures: set = set()

        # Build the list of anchor subsets to enumerate.
        if include_partial:
            from itertools import combinations
            all_subsets = []
            for r in range(1, len(anchors) + 1):
                all_subsets.extend(combinations(anchors, r))
        else:
            all_subsets = [tuple(anchors)]

        # Skip subsets whose orbit signature we've already enumerated.
        seen_subset_keys: set = set()
        canonical_subsets: list[tuple[int, ...]] = []
        for sub in all_subsets:
            key = _canonical_subset_key(sub, orbit_id)
            if key in seen_subset_keys:
                continue
            seen_subset_keys.add(key)
            canonical_subsets.append(sub)

        if verbose:
            print(
                f"  [depth={depth}] enumerating {len(canonical_subsets)} canonical "
                f"anchor subsets (out of {len(all_subsets)} raw)"
            )

        # ── Backtracking chain placement ────────────────────────────────
        rejected_disconnected = 0

        def _emit(bonded: list[int], assigned_pos: dict[int, np.ndarray],
                  assigned_clique: dict[int, frozenset]) -> None:
            nonlocal rejected_disconnected
            atom_cliques: list = [None] * n_atoms
            for i in bonded:
                atom_cliques[i] = assigned_clique[i]

            # Surface-connectivity guard: drop placements whose bonded
            # cliques cannot reach each other through the surface graph
            # within max_pair_shells hops.  We do NOT grow the ego depth
            # used for the iso-class graph above n_shells_pair — the
            # iso-class ego is, by design, the n_shells_pair-shell
            # neighbourhood of all coordinated surface atoms (see the
            # `ego` construction below).
            if G_surf is not None:
                bonded_cliques = [c for c in atom_cliques if c is not None]
                needed = _min_ego_depth_for_connectivity(
                    G_surf, bonded_cliques, apsp=apsp
                )
                if needed >= 10**6 or needed > max_pair_shells:
                    rejected_disconnected += 1
                    return

            bonded_positions = np.array(
                [assigned_pos[i] for i in bonded], dtype=float
            )
            positions = _full_adsorbate_positions(
                reactant, bonded, bonded_positions, G, pbc
            )

            # Iso-class ego graph: the n_shells_pair-shell neighbourhood
            # of every surface atom that is coordinated to an adsorbate
            # atom (i.e. the union of all bonded cliques).  Default
            # n_shells_pair = 1 → "neighbours of all coordinated surface
            # atoms".
            union = set()
            for c in atom_cliques:
                if c is not None:
                    union |= set(c)
            ego = _build_clique_ego(G, frozenset(union), n_shells_pair)

            _try_merge_or_new(
                multisites,
                smiles          = reactant.smiles,
                atom_cliques    = atom_cliques,
                positions       = positions,
                ego_graph       = ego,
                node_match      = node_match,
                seen_signatures = seen_signatures,
            )

        def _recurse(bonded: list[int], idx: int,
                     assigned_pos: dict[int, np.ndarray],
                     assigned_clique: dict[int, frozenset]) -> None:
            if idx == len(bonded):
                _emit(bonded, assigned_pos, assigned_clique)
                return
            next_idx = bonded[idx]
            next_el  = elements[next_idx]
            for clique, pos in raw_sites[next_el]:
                # No two adsorbate atoms share the same surface clique.
                if any(clique == c for c in assigned_clique.values()):
                    continue
                ok = True
                for prev_idx, p_prev in assigned_pos.items():
                    target = float(D[prev_idx, next_idx])
                    d = _mic_distance(p_prev, pos, cell, cell_inv, pbc, use_mic)
                    if abs(d - target) > bond_tolerance:
                        ok = False
                        break
                if not ok:
                    continue
                assigned_pos[next_idx]    = pos
                assigned_clique[next_idx] = clique
                _recurse(bonded, idx + 1, assigned_pos, assigned_clique)
                del assigned_pos[next_idx]
                del assigned_clique[next_idx]

        for sub in canonical_subsets:
            bonded = sorted(sub)
            first = bonded[0]
            first_el = elements[first]
            n_first_classes = len(iso_classes_by_elem[first_el])
            if verbose:
                print(
                    f"    subset {bonded}: first={first}({first_el}) "
                    f"over {n_first_classes} iso-classes"
                )
            for iso_first in iso_classes_by_elem[first_el]:
                if iso_first.position is None:
                    continue
                assigned_pos    = {first: np.asarray(iso_first.position, float)}
                assigned_clique = {first: iso_first.representative}
                _recurse(bonded, 1, assigned_pos, assigned_clique)

        if verbose and rejected_disconnected:
            print(
                f"  [depth={depth}] rejected {rejected_disconnected} placement(s) "
                f"with bonded cliques unreachable within max_pair_shells="
                f"{max_pair_shells}"
            )

        return multisites

    # ── Retry loop: grow n_shells_anchor if any placement reaches further
    #    than the iso-class ego graph used to deduplicate first anchors. ─
    multisites: list[MultiSite] = _run_pass(n_shells_anchor_eff)
    retries = 0
    while auto_grow_shells and retries < max_shell_retries:
        needed = _required_n_shells(G, multisites, apsp=apsp)
        if needed <= n_shells_anchor_eff:
            break
        new_depth = needed
        if verbose:
            print(
                f"  ⚠  reach probe: bonded cliques span up to {needed} hops, "
                f"but iso-class depth was {n_shells_anchor_eff} — growing to "
                f"{new_depth} and re-enumerating"
            )
        n_shells_anchor_eff = new_depth
        multisites = _run_pass(n_shells_anchor_eff)
        retries += 1
    else:
        if auto_grow_shells and retries == max_shell_retries:
            needed = _required_n_shells(G, multisites, apsp=apsp)
            if needed > n_shells_anchor_eff and verbose:
                print(
                    f"  ⚠  hit max_shell_retries={max_shell_retries}; "
                    f"final depth {n_shells_anchor_eff} < required {needed} — "
                    "iso-class deduplication may still be coarser than ideal."
                )

    if verbose:
        print(
            f"  final n_shells_anchor = {n_shells_anchor_eff} "
            f"(after {retries} grow-retries)"
        )

    if verbose:
        n_full = sum(
            1 for ms in multisites
            if all((ms.atom_cliques[i] is not None) for i in anchors)
        )
        n_partial = len(multisites) - n_full
        total_members = sum(len(ms.members) for ms in multisites)
        print(
            f"  → {len(multisites)} unique multisite iso-classes "
            f"({n_full} full-anchor + {n_partial} partial), "
            f"{total_members} placements total"
        )

    G.graph.setdefault("multisites", {})[reactant.smiles] = multisites
    get_cache(G).multisites[reactant.smiles] = multisites
    return multisites


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

def find_multisites_for_reactant(*args, **kwargs):
    """Deprecated alias for :func:`find_multisites`.

    The N-atom and 2-atom enumerators have been merged into a single
    universal :func:`find_multisites`.  This shim is retained so that
    existing notebooks and call-sites continue to work; new code should
    use :func:`find_multisites` directly.
    """
    return find_multisites(*args, **kwargs)


# ---------------------------------------------------------------------------
# Rigid-body refinement of MultiSite.positions
# ---------------------------------------------------------------------------

def _rotation_from_axis_angle(rotvec: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a Rodrigues axis-angle vector.

    ``rotvec`` direction is the rotation axis; its magnitude is the
    rotation angle in radians.
    """
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _surface_atoms_array(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, list]:
    """Cache surface positions, covalent radii and node ids.

    Returns ``(positions[N,3], r_cov[N], node_ids[N])`` for every node
    tagged ``type == "surface"``.
    """
    pos:   list = []
    r_cov: list = []
    ids:   list = []
    from ase.data import covalent_radii as _RC, atomic_numbers as _AN
    for n, d in G.nodes(data=True):
        if d.get("type") != "surface":
            continue
        pos.append(d["position"])
        elem = d["element"]
        r_cov.append(float(_RC[_AN[elem]]))
        ids.append(n)
    return (
        np.asarray(pos, dtype=float).reshape(-1, 3),
        np.asarray(r_cov, dtype=float),
        ids,
    )


def optimise_multisite_positions(
    G: nx.Graph,
    smiles: str,
    reactant,
    *,
    restraint_weight: float = 10.0,
    repulsion_weight: float = 1.0,
    contact_factor: float = CONTACT_FACTOR,
    max_iter: int = 100,
    verbose: bool = False,
) -> list[MultiSite]:
    """Rigid-body refinement of every ``MultiSite.positions`` for
    ``G.graph["multisites"][smiles]``.

    Each placement is treated as 6 rigid-body DOF — 3 translation and 3
    axis-angle rotation — applied to the gas-phase reactant geometry
    in *reactant*.  L-BFGS-B minimises a calculator-free objective:

    .. code-block:: text

        E(R, t) = restraint_weight * Σ_{i ∈ bonded}  ||p_i − p_target_i||²
                + repulsion_weight * Σ_{a, s}        max(0, R_min − d_as)²

    where

    * ``p_i = R · q_i + t`` are the transformed adsorbate atom positions
      (``q_i`` = gas-phase reactant coordinate),
    * ``p_target_i`` = ``IsoClass.position`` of the surface site that
      anchor *i* is bonded to (read from the placement's first member's
      cliques + ``G.graph['site_positions']``),
    * ``R_min = contact_factor * (r_cov_a + r_cov_s)`` is the steric
      contact distance, and
    * the ``(a, s)`` sum runs over (adsorbate atom, surface atom) pairs
      with ``d_as < R_min``, **excluding** surface atoms that belong to
      any bonded clique of the placement (those are *supposed* to be in
      contact with their bonded adsorbate atom).

    The initial guess is taken from the existing ``MultiSite.positions``
    (i.e. the Kabsch / single-anchor fall-back from
    :func:`_full_adsorbate_positions`).  The optimiser only refines the
    geometry — it does **not** change ``atom_cliques``, ``ego_graph`` or
    iso-class membership.

    Strategy notes
    --------------
    Several refinement strategies are possible; this implementation uses
    the cheapest one — purely geometric, no calculator, no internal
    coordinates — which is appropriate for the autokmc setup pipeline.
    Stronger options (left for downstream code that has a calculator)
    include:

    * **ASE constrained relax** — embed the placement in the slab/NP
      ``Atoms``, freeze surface atoms via ``FixAtoms``, and run BFGS
      with the active calculator (NequIP, EMT, …).  Far more expensive
      but gives true PES minima.
    * **Internal-coordinate relax** — let bond lengths/angles/torsions
      vary instead of treating the molecule as rigid.  Useful for
      flexible chains where the gas-phase geometry is a poor proxy.
    * **Multi-start sampling** — re-run the rigid-body relaxation from
      several random initial rotations around the surface normal and
      keep the lowest-energy pose; combats local minima for asymmetric
      adsorbates on hollow / bridge sites.

    Parameters
    ----------
    G : nx.Graph
        Surface graph with multisites already enumerated.
    smiles : str
        Key into ``G.graph['multisites']``.
    reactant : :class:`autokmc.reactants.Reactant`
        Gas-phase reactant whose ``atoms.get_positions()`` defines the
        rigid-body reference geometry ``q_i``.  Must match *smiles*.
    restraint_weight : float
        Weight on the bonded-atom position restraint.  Default 10.0.
    repulsion_weight : float
        Weight on the soft surface clipping penalty.  Default 1.0.
    contact_factor : float
        Multiplier on covalent-radii sums when computing ``R_min``.
        Default 0.9 (matches ``opt_factor`` semantics in
        ``default_sites.optimise_site_positions``).
    max_iter : int
        L-BFGS-B iteration cap per placement.  Default 100.
    verbose : bool
        Print per-placement RMS displacements.

    Returns
    -------
    list[MultiSite]
        The same list stored in ``G.graph['multisites'][smiles]``;
        ``MultiSite.positions`` is updated in place.

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    KeyError
        If ``smiles`` is not present in ``G.graph['multisites']`` or if
        the bonded cliques cannot be located in ``G.graph['sites']``.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "optimise_multisite_positions requires scipy "
            "(install with `pip install scipy`)."
        ) from e

    if "multisites" not in G.graph or smiles not in G.graph["multisites"]:
        raise KeyError(
            f"No multisites enumerated for SMILES {smiles!r}; "
            "call find_multisites first."
        )

    multisites: list[MultiSite] = G.graph["multisites"][smiles]
    if not multisites:
        return multisites

    # ── Reference geometry & per-atom covalent radii ────────────────────
    from ase.data import covalent_radii as _RC, atomic_numbers as _AN
    q_ref = np.asarray(reactant.atoms.get_positions(), dtype=float)  # (N, 3)
    n_atoms = q_ref.shape[0]
    ads_r_cov = np.array([
        float(_RC[_AN[reactant.graph.nodes[i]["element"]]])
        for i in range(n_atoms)
    ])
    q_centroid = q_ref.mean(axis=0)
    q_centred  = q_ref - q_centroid

    # Surface atom cache (positions, radii, node ids).
    surf_pos, surf_r, surf_ids = _surface_atoms_array(G)
    surf_id_to_row = {nid: i for i, nid in enumerate(surf_ids)}

    # Index sites by clique to look up target positions for each placement.
    elements = {i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)}
    site_lookup: dict[str, dict[frozenset, np.ndarray]] = {}
    for el in set(elements.values()):
        if el not in G.graph.get("sites", {}):
            continue
        site_lookup[el] = {}
        for k, cliques in G.graph["sites"][el].items():
            for clique, pos in zip(cliques,
                                   G.graph["site_positions"][el][k]):
                site_lookup[el][clique] = np.asarray(pos, dtype=float)

    cell, cell_inv, pbc, use_mic = _resolve_cell(G)

    def _refine(ms: MultiSite) -> tuple[np.ndarray, float, float]:
        """Refine one placement; returns (new_positions, E0, E_final)."""
        bonded_idx = [i for i, c in enumerate(ms.atom_cliques) if c is not None]

        # Build target positions and exclusion set for the repulsion sum.
        targets: list[np.ndarray] = []
        exclude_surf: set = set()
        for i in bonded_idx:
            el = elements[i]
            clique = ms.atom_cliques[i]
            if el not in site_lookup or clique not in site_lookup[el]:
                # Fall back to current placement position; no restraint.
                targets.append(np.asarray(ms.positions[i], dtype=float))
            else:
                targets.append(site_lookup[el][clique])
            exclude_surf.update(int(n) for n in clique)
        target_arr = np.array(targets, dtype=float).reshape(-1, 3)

        # Pre-compute mask of "free" surface atoms (not in any bonded clique).
        if len(exclude_surf):
            keep = np.array([n not in exclude_surf for n in surf_ids], dtype=bool)
        else:
            keep = np.ones(len(surf_ids), dtype=bool)
        free_pos = surf_pos[keep]
        free_r   = surf_r[keep]

        # Initial rigid-body params: best-fit (R, t) of q_ref → ms.positions.
        # Since ms.positions came from Kabsch, we can simply start from
        # rotvec=0, t = ms.positions.mean − q_centroid.
        cur_pos = np.asarray(ms.positions, dtype=float)
        t0 = cur_pos.mean(axis=0) - q_centroid
        # Recover initial rotation from current positions vs reference.
        R0, t0_full = _kabsch(q_ref, cur_pos)
        # Encode R0 as axis-angle for the optimiser.
        cos_th = (np.trace(R0) - 1.0) / 2.0
        cos_th = float(np.clip(cos_th, -1.0, 1.0))
        theta0 = float(np.arccos(cos_th))
        if theta0 > 1e-8:
            axis0 = np.array([
                R0[2, 1] - R0[1, 2],
                R0[0, 2] - R0[2, 0],
                R0[1, 0] - R0[0, 1],
            ]) / (2.0 * np.sin(theta0))
            rotvec0 = axis0 * theta0
        else:
            rotvec0 = np.zeros(3)

        x0 = np.concatenate([t0_full + R0 @ q_centroid - q_centroid, rotvec0])
        # Re-derive translation so that initial pose exactly reproduces cur_pos.
        # Pose model:  p_i = (R0 + δR) @ (q_i - q_centroid) + (q_centroid + t)
        # At x0 we want p_i ≈ cur_pos[i], so set t such that:
        #   q_centroid + t = cur_pos.mean(0) - R0 @ q_centroid.mean(0)  (= cur_pos.mean(0))
        # i.e. t = cur_pos.mean(0) − q_centroid.
        x0[:3] = cur_pos.mean(axis=0) - q_centroid

        def _pose(x: np.ndarray) -> np.ndarray:
            t = x[:3]
            R = R0 @ _rotation_from_axis_angle(x[3:])
            return q_centred @ R.T + (q_centroid + t)

        def _energy(x: np.ndarray) -> float:
            p = _pose(x)
            E = 0.0
            # Restraint on bonded atoms.
            if bonded_idx:
                disp = p[bonded_idx] - target_arr
                E += restraint_weight * float(np.einsum("ij,ij->", disp, disp))
            # Soft repulsion against free surface atoms.
            if free_pos.size:
                # MIC-aware pairwise distances (small adsorbate ↔ surface).
                for ai in range(n_atoms):
                    pa = p[ai]
                    dv = free_pos - pa
                    if use_mic:
                        frac = dv @ cell_inv
                        for k in range(3):
                            if pbc[k]:
                                frac[:, k] -= np.round(frac[:, k])
                        dv = frac @ cell
                    d2 = np.einsum("ij,ij->i", dv, dv)
                    R_min = contact_factor * (ads_r_cov[ai] + free_r)
                    delta = R_min - np.sqrt(d2 + 1e-12)
                    pos = np.maximum(delta, 0.0)
                    E += repulsion_weight * float(np.dot(pos, pos))
            return E

        E0 = _energy(x0)
        res = minimize(_energy, x0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-7})
        new_pos = _pose(res.x)
        return new_pos, E0, float(res.fun)

    if verbose:
        print(
            f"optimise_multisite_positions: smiles={smiles!r}  "
            f"placements={len(multisites)}  "
            f"restraint_weight={restraint_weight} repulsion_weight={repulsion_weight} "
            f"contact_factor={contact_factor}"
        )

    for ms in multisites:
        try:
            new_pos, E0, Ef = _refine(ms)
        except Exception as exc:
            if verbose:
                print(f"  iso-class {ms.iso_class}: refinement failed ({exc!r})")
            continue
        rms = float(np.sqrt(np.mean(np.sum(
            (new_pos - np.asarray(ms.positions))**2, axis=1
        ))))
        ms.positions = new_pos
        if verbose:
            print(
                f"  iso-class {ms.iso_class:3d}: "
                f"E {E0:9.4f} → {Ef:9.4f}  RMS Δ {rms:6.3f} Å"
            )

    return multisites

