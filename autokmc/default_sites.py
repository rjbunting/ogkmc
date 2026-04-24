"""
autokmc.default_sites
=====================
Find all adsorption sites for an element on a surface graph by enumerating
every k-clique (k = 1 ... k_max) of the adsorbate-specific co-bonding graph,
then optionally reduce them to unique classes by graph isomorphism.

Workflow
--------
1. Resolve the covalent radius of the element from ASE data.
2. Build the co-bonding graph: d(i,j) <= bond_factor*(2*r_cov_ads+r_cov_i+r_cov_j).
3. k_max = size of the largest clique (maximum possible coordination).
4. Enumerate every clique of size 1...k_max.
5. Optionally reduce to unique iso-classes by comparing the n-shell
   ego-subgraph around each clique.  More shells = finer discrimination.
6. Optimise each adsorbate's Cartesian position with a calculator-free
   geometric objective (bond-length restraint + non-bonded soft repulsion
   from atoms inside the same n-shell ego graph).

Public API
----------
* find_sites_for_element      -- enumerate all raw sites
* reduce_sites_by_isomorphism -- group into iso-classes at a given shell depth
* optimise_site_positions     -- compute optimal adsorbate position per site
* k_max_for_element           -- k_max only (no site enumeration)
* k_max_for_radius            -- low-level k_max from a radius float
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

from autokmc.constants import (
    CO_FACTOR,
    HULL_TOL,
    OPT_FACTOR,
    REPULSION_WEIGHT,
    SITE_REPULSION_CUTOFF,
    N_SHELLS_DEFAULT,
)
from autokmc.cache import get_cache
from autokmc.logging_utils import get_logger, verbose_scope

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Iso-class record
# ---------------------------------------------------------------------------

@dataclass
class IsoClass:
    """One isomorphism class of k-fold surface sites at a given shell depth.

    Attributes
    ----------
    k : int
        Coordination number (1 = top, 2 = bridge, 3 = hollow, ...).
    iso_class : int
        Index within the k group (0-based).
    n_shells : int
        Number of neighbor shells used for isomorphism comparison.
    representative : frozenset[int]
        Global atom indices of one representative clique.
    members : list[frozenset[int]]
        All cliques in this iso-class.
    centroid : np.ndarray, shape (3,)
        Mean Cartesian position of the representative clique atoms (Å).
    ego_graph : nx.Graph
        The n-shell ego-subgraph of the representative (used for matching).
    position : np.ndarray | None
        Optimised adsorbate Cartesian position (Å); ``None`` until
        :func:`optimise_site_positions` has been run.
    """
    k              : int
    iso_class      : int
    n_shells       : int
    representative : frozenset
    members        : list = field(default_factory=list)
    centroid       : Any  = None
    ego_graph      : Any  = None
    position       : Any  = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_pbc_cell(G: nx.Graph) -> tuple[np.ndarray, np.ndarray | None,
                                            np.ndarray, bool]:
    """Return ``(cell, cell_inv_or_None, pbc_bool, use_mic)`` for *G*."""
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


def _build_co_bond_graph(
    surface_graph: nx.Graph,
    r_cov_ads: float,
    co_factor: float = CO_FACTOR,
) -> nx.Graph:
    """Build the adsorbate-specific co-bonding graph on surface atoms.

    Two surface atoms ``i, j`` are connected when an adsorbate of covalent
    radius *r_cov_ads* can simultaneously bind both:

    .. code-block:: text

        d(i, j) <= co_factor * (2*r_cov_ads + r_cov_i + r_cov_j)

    Implementation notes
    --------------------
    Uses :class:`scipy.spatial.cKDTree` to avoid the previous O(N²) Python
    loop.  For periodic cells with an orthogonal lattice the kd-tree is
    built with ``boxsize`` directly; for non-orthogonal lattices we tile
    the surface positions across ±1 image of every periodic axis and run
    a non-periodic query (the few-image strategy is cheap as long as the
    cutoff is small relative to the cell, which it always is here).
    """
    surf_nodes = [(n, d) for n, d in surface_graph.nodes(data=True)
                  if d["type"] == "surface"]
    if not surf_nodes:
        return nx.Graph()

    surf_list  = [n for n, _ in surf_nodes]
    surf_pos   = np.array([d["position"] for _, d in surf_nodes], dtype=float)
    surf_rcov  = np.array([d["covalent_radius"] for _, d in surf_nodes], dtype=float)

    cell, cell_inv, pbc, use_mic = _resolve_pbc_cell(surface_graph)

    # Maximum possible cutoff: every pair test uses
    #   co_factor * (2*r_cov_ads + r_cov_i + r_cov_j)
    # so an upper bound that is safe to use as the kd-tree query radius is
    # the value at the maximum r_cov on both sides.
    max_rcov = float(surf_rcov.max())
    r_query  = co_factor * (2.0 * r_cov_ads + 2.0 * max_rcov)

    cbg = nx.Graph()
    cbg.add_nodes_from((n, dict(surface_graph.nodes[n])) for n in surf_list)

    from scipy.spatial import cKDTree

    # Detect orthogonality so we can use cKDTree's native boxsize.
    is_ortho = use_mic and np.allclose(cell - np.diag(np.diag(cell)), 0.0)

    if is_ortho:
        boxsize = np.where(pbc, np.diag(cell), 0.0)
        # cKDTree requires boxsize > 0 along periodic axes; non-periodic
        # axes use 0 (which means "no wrap" in scipy ≥ 1.6).
        tree = cKDTree(surf_pos, boxsize=np.where(boxsize > 0, boxsize, 0.0))
        pairs = tree.query_pairs(r=r_query, output_type="ndarray")
    elif use_mic:
        # Non-orthogonal periodic cell: tile ±1 image of every periodic axis,
        # build a single tree, query in original positions.
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx and not pbc[0]: continue
                    if dy and not pbc[1]: continue
                    if dz and not pbc[2]: continue
                    offsets.append(np.array([dx, dy, dz], dtype=int))
        tiled_pos:  list[np.ndarray] = []
        tiled_idx:  list[int]        = []
        for off in offsets:
            tiled_pos.append(surf_pos + off @ cell)
            tiled_idx.extend(range(len(surf_pos)))
        tiled_pos_arr = np.concatenate(tiled_pos, axis=0)
        tree = cKDTree(tiled_pos_arr)
        # Query each *original* atom against the tiled tree; collect pairs
        # where the partner index (mod len(surf_pos)) is greater than the
        # query index, to dedupe (and skip self-image with offset 0).
        raw_pairs = set()
        for i, p in enumerate(surf_pos):
            for hit in tree.query_ball_point(p, r=r_query):
                j = tiled_idx[hit]
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                raw_pairs.add((a, b))
        pairs = np.array(sorted(raw_pairs), dtype=int) if raw_pairs \
                else np.empty((0, 2), dtype=int)
    else:
        tree = cKDTree(surf_pos)
        pairs = tree.query_pairs(r=r_query, output_type="ndarray")

    if len(pairs) == 0:
        return cbg

    # Now apply the *exact* per-pair cutoff (the kd-tree query used a
    # conservative upper bound).  Distances are computed MIC-aware.
    p_i = surf_pos[pairs[:, 0]]
    p_j = surf_pos[pairs[:, 1]]
    dv = p_j - p_i
    if use_mic and cell_inv is not None:
        frac = dv @ cell_inv
        for axis in range(3):
            if pbc[axis]:
                frac[:, axis] -= np.round(frac[:, axis])
        dv = frac @ cell
    dist = np.linalg.norm(dv, axis=1)
    cutoff = co_factor * (2.0 * r_cov_ads + surf_rcov[pairs[:, 0]]
                                          + surf_rcov[pairs[:, 1]])
    keep = dist <= cutoff

    for (ii, jj) in pairs[keep]:
        cbg.add_edge(surf_list[int(ii)], surf_list[int(jj)])

    return cbg


def _build_clique_ego(
    surface_graph: nx.Graph,
    clique: frozenset,
    n_shells: int,
) -> nx.Graph:
    """Return the n-shell ego-subgraph of *surface_graph* around *clique*.

    Expands outward shell by shell from the clique nodes through the full
    graph (all node types, so subsurface atoms contribute at n_shells>=2).
    """
    frontier: set[int] = set(clique)
    visited:  set[int] = set(clique)

    for _ in range(n_shells):
        next_shell: set[int] = set()
        for n in frontier:
            next_shell.update(surface_graph.neighbors(n))
        frontier = next_shell - visited
        visited |= frontier

    return surface_graph.subgraph(visited).copy()


def _circular_mean_centroid(
    positions: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> np.ndarray:
    """MIC-robust centroid of *positions* (rows of Cartesian coordinates).

    For non-periodic axes, returns the plain arithmetic mean.  For
    periodic axes, uses the standard circular-mean trick (average
    ``cos(2πf), sin(2πf)`` of the fractional coordinates and recover the
    mean angle with ``atan2``).  This is robust against atoms that
    straddle a periodic boundary — the previous "anchor at positions[0]"
    approach failed for cliques with >2 atoms near a corner.
    """
    if not use_mic or cell_inv is None or not pbc.any():
        return positions.mean(axis=0)

    frac = positions @ cell_inv                         # (N, 3)
    centroid_frac = np.empty(3)
    for axis in range(3):
        if pbc[axis]:
            theta = 2.0 * np.pi * frac[:, axis]
            mean_c = np.cos(theta).mean()
            mean_s = np.sin(theta).mean()
            ang = np.arctan2(mean_s, mean_c)
            if ang < 0.0:
                ang += 2.0 * np.pi
            centroid_frac[axis] = ang / (2.0 * np.pi)
        else:
            centroid_frac[axis] = frac[:, axis].mean()
    return centroid_frac @ cell


def _clique_centroid(
    G: nx.Graph,
    clique: frozenset,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> np.ndarray:
    """MIC-aware centroid of the atom positions in *clique*."""
    positions = np.array([G.nodes[n]["position"] for n in clique], dtype=float)
    return _circular_mean_centroid(positions, cell, cell_inv, pbc, use_mic)


def _iso_prefilter_key(g: nx.Graph) -> tuple:
    """Cheap structural fingerprint -- unequal keys guarantee non-isomorphism."""
    elem_deg = tuple(sorted(
        (d["element"], g.degree(n))
        for n, d in g.nodes(data=True)
    ))
    return (g.number_of_nodes(), g.number_of_edges(),
            tuple(sorted(g.degree(n) for n in g.nodes())),
            elem_deg)


def _outward_normal(G: nx.Graph, centroid: np.ndarray) -> np.ndarray:
    """Return the unit outward-normal for a nanoparticle site.

    Defined as the direction from the geometric centre of all surface atoms
    to the site *centroid*.  Falls back to ``[0, 0, 1]`` if the centroid
    coincides with the geometric centre (pathological case).
    """
    surf_pos = np.array(
        [d["position"] for _, d in G.nodes(data=True) if d.get("type") == "surface"],
        dtype=float,
    )
    nano_center = surf_pos.mean(axis=0) if len(surf_pos) else centroid
    n = centroid - nano_center
    norm = float(np.linalg.norm(n))
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _mic_distances(
    p: np.ndarray,
    ref_pos: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return minimum-image distances from each row of *ref_pos* to point *p*."""
    dv   = p[np.newaxis, :] - ref_pos
    frac = dv @ cell_inv
    for i in range(3):
        if pbc[i]:
            frac[:, i] -= np.round(frac[:, i])
    return np.linalg.norm(frac @ cell, axis=1)


def _optimize_site_position(
    G: nx.Graph,
    clique: frozenset,
    r_cov_ads: float,
    *,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    n_shells: int = N_SHELLS_DEFAULT,
    repulsion_cutoff: float | None = SITE_REPULSION_CUTOFF,
) -> np.ndarray:
    """Find the optimal Cartesian position for an adsorbate at *clique*.

    Objective
    ---------
    Bond term — penalises deviation from the ideal bond length to each
    bonded surface atom::

        d_ideal(i) = opt_factor * (r_cov_ads + r_cov_i)
        bond_term  = sum_i (|p - pos_i|_MIC - d_ideal_i)^2

    Repulsion term — soft ``1/r²`` repulsion from non-bonded surface atoms,
    pushing the adsorbate away from neighbours it is *not* bonded to.

    The set of non-bonded atoms is restricted to the n-shell ego graph
    around *clique* (parameter ``n_shells``), and further filtered to
    surface atoms within ``repulsion_cutoff`` Å of the clique centroid.
    Both filters dramatically cut the per-eval cost compared with summing
    over every surface atom in the system, while leaving the local
    geometry unaffected (the ``1/r²`` term dies off rapidly).

    Constraints
    -----------
    * Periodic slabs: ``z >= z_floor`` of the bonded atoms (L-BFGS-B).
    * Nanoparticles: the adsorbate is constrained to lie on the outward
      side of the clique centroid (SLSQP).
    """
    from scipy.optimize import minimize

    cell, cell_inv, pbc, use_mic = _resolve_pbc_cell(G)

    bonded      = list(clique)
    bonded_pos  = np.array([G.nodes[n]["position"]        for n in bonded])
    bonded_rcov = np.array([G.nodes[n]["covalent_radius"] for n in bonded])
    ideal_dists = opt_factor * (r_cov_ads + bonded_rcov)

    # ------------------------------------------------------------------
    # Restrict the non-bonded repulsion atoms to (a) the n-shell ego of
    # the clique, and (b) atoms within `repulsion_cutoff` of its centroid.
    # The inner ego graph is what makes the repulsion "based on n_shell of
    # the clique" — the same notion of locality the iso-class step uses.
    # ------------------------------------------------------------------
    clique_set = set(clique)
    if n_shells > 0:
        ego = _build_clique_ego(G, frozenset(clique), n_shells)
        candidate_nodes = (n for n, d in ego.nodes(data=True)
                           if d.get("type") == "surface" and n not in clique_set)
    else:
        candidate_nodes = (n for n, d in G.nodes(data=True)
                           if d.get("type") == "surface" and n not in clique_set)
    nb_pos_list = [G.nodes[n]["position"] for n in candidate_nodes]
    nb_pos = np.array(nb_pos_list, dtype=float) if nb_pos_list \
             else np.empty((0, 3), dtype=float)

    # MIC-aware centroid of the bonded atoms, used both for the cutoff
    # filter and as the optimiser's starting guess.
    centroid = _circular_mean_centroid(bonded_pos, cell, cell_inv, pbc, use_mic)

    if repulsion_cutoff is not None and len(nb_pos):
        if use_mic and cell_inv is not None:
            d_to_centroid = _mic_distances(centroid, nb_pos, cell, cell_inv, pbc)
        else:
            d_to_centroid = np.linalg.norm(nb_pos - centroid, axis=1)
        nb_pos = nb_pos[d_to_centroid <= repulsion_cutoff]

    # MIC-aware bonded relative positions for the slab z-floor heuristic.
    if use_mic and cell_inv is not None:
        dv_bonded = bonded_pos - bonded_pos[0]
        frac_dv   = dv_bonded @ cell_inv
        for i in range(3):
            if pbc[i]:
                frac_dv[:, i] -= np.round(frac_dv[:, i])
        mic_rel = frac_dv @ cell
    else:
        mic_rel = bonded_pos - bonded_pos[0]

    def objective(p: np.ndarray) -> float:
        if use_mic and cell_inv is not None:
            dists = _mic_distances(p, bonded_pos, cell, cell_inv, pbc)
        else:
            dists = np.linalg.norm(p[np.newaxis, :] - bonded_pos, axis=1)
        bond_term = float(np.sum((dists - ideal_dists) ** 2))
        if repulsion_weight > 0.0 and len(nb_pos):
            if use_mic and cell_inv is not None:
                nb_dists = _mic_distances(p, nb_pos, cell, cell_inv, pbc)
            else:
                nb_dists = np.linalg.norm(p[np.newaxis, :] - nb_pos, axis=1)
            repulsion = float(np.sum(1.0 / (nb_dists ** 2 + 1e-12)))
            return bond_term + repulsion_weight * repulsion
        return bond_term

    if use_mic:
        lateral_dists = np.linalg.norm(
            mic_rel[:, :2] - mic_rel[:, :2].mean(axis=0), axis=1)
        h_per_atom = np.sqrt(np.maximum(0.0, ideal_dists ** 2 - lateral_dists ** 2))
        z_floor = float(bonded_pos[:, 2].max())
        z0      = z_floor + float(h_per_atom.mean())
        x0      = np.array([centroid[0], centroid[1], z0])
        bounds  = [(None, None), (None, None), (z_floor, None)]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    else:
        n_out     = _outward_normal(G, centroid)
        stand_off = float(ideal_dists.mean())
        x0        = centroid + max(stand_off, 0.5) * n_out
        _c = centroid.copy()
        _n = n_out.copy()
        res = minimize(
            objective, x0,
            method="SLSQP",
            constraints={"type": "ineq",
                         "fun": lambda p: float(np.dot(p - _c, _n))},
            options={"ftol": 1e-9, "maxiter": 500},
        )
    return res.x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def k_max_for_radius(
    surface_graph: nx.Graph,
    r_cov_ads: float,
    co_factor: float = CO_FACTOR,
) -> int:
    """Return ``k_max`` -- the largest clique of the co-bonding graph."""
    cbg = _build_co_bond_graph(surface_graph, r_cov_ads, co_factor)
    if cbg.number_of_nodes() == 0:
        return 1
    return max((len(c) for c in nx.find_cliques(cbg)), default=1)


def k_max_for_element(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
    verbose: bool = False,
) -> int:
    """Find ``k_max`` for *element* and cache it in ``G.graph['k_max'][element]``."""
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")
    cache = get_cache(G)

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])
    k_max = k_max_for_radius(G, r_cov, co_factor=co_factor)

    with verbose_scope(_log, verbose):
        _log.debug("k_max_for_element: %r r_cov=%.4f Å k_max=%d",
                   element, r_cov, k_max)

    cache.k_max[element] = k_max
    return k_max


def find_sites_for_element(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
    verbose: bool = False,
) -> dict[int, list[frozenset]]:
    """Find all adsorption sites for *element* and cache them on *G*.

    Stored at ``G.graph['sites'][element]`` (and on the typed
    :class:`~autokmc.cache.SiteCache`).  Each site is a frozenset of
    surface-atom global indices.
    """
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")
    cache = get_cache(G)

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])

    cbg = _build_co_bond_graph(G, r_cov, co_factor)
    cell, cell_inv, pbc, use_mic = _resolve_pbc_cell(G)

    if cbg.number_of_nodes() == 0:
        k_max = 1
    else:
        k_max = max((len(c) for c in nx.find_cliques(cbg)), default=1)

    with verbose_scope(_log, verbose):
        _log.debug("find_sites_for_element: %r r_cov=%.4f Å k_max=%d co_factor=%g",
                   element, r_cov, k_max, co_factor)

        # Re-use cached hull facet equations if find_surface_atoms (or a
        # previous call) computed them; otherwise build for nanoparticles.
        hull_equations: np.ndarray | None = None
        if not use_mic:
            hull_equations = cache.hull
            if hull_equations is None:
                try:
                    from scipy.spatial import ConvexHull
                    all_pos = np.array([d["position"]
                                        for _, d in G.nodes(data=True)])
                    _hull = ConvexHull(all_pos)
                    hull_equations = np.asarray(_hull.equations, dtype=float)
                    cache.hull = hull_equations
                except Exception:
                    hull_equations = None

        sites: dict[int, list[frozenset]] = {k: [] for k in range(1, k_max + 1)}
        seen:  set[frozenset] = set()

        for clique in nx.enumerate_all_cliques(cbg):
            k = len(clique)
            if k > k_max:
                break
            key = frozenset(clique)
            if key in seen:
                continue
            if hull_equations is not None:
                c = _clique_centroid(G, key, cell, cell_inv, pbc, use_mic)
                max_sd = float(np.max(
                    hull_equations[:, :3] @ c + hull_equations[:, 3]
                ))
                if max_sd < HULL_TOL:
                    seen.add(key)
                    continue
            seen.add(key)
            sites[k].append(key)

        sites = {k: v for k, v in sites.items() if v}
        if sites:
            k_max = max(sites)
        else:
            k_max = 1
            sites = {1: []}

        _LABELS = {1: "top", 2: "bridge", 3: "hollow"}
        total = sum(len(v) for v in sites.values())
        for k, cliques in sorted(sites.items()):
            label = _LABELS.get(k, f"{k}-fold")
            _log.debug("  k=%d %-8s %4d sites", k, label, len(cliques))
        _log.debug("  total : %d sites", total)

    cache.k_max[element] = k_max
    cache.sites[element] = sites
    return sites


def reduce_sites_by_isomorphism(
    G: nx.Graph,
    element: str,
    *,
    n_shells: int = N_SHELLS_DEFAULT,
    verbose: bool = False,
) -> dict[int, list[IsoClass]]:
    """Group all sites for *element* into iso-classes using an n-shell ego-graph.

    Two cliques are placed in the same iso-class when their n-shell
    ego-subgraphs are graph-isomorphic with element-label matching.
    """
    cache = get_cache(G)
    if element not in cache.sites:
        raise KeyError(
            f"No sites found for '{element}'. "
            "Call find_sites_for_element(G, element) first."
        )

    sites_by_k: dict[int, list[frozenset]] = cache.sites[element]

    surf_pos = {n: d["position"] for n, d in G.nodes(data=True)
                if d.get("type") == "surface"}

    node_match = isomorphism.categorical_node_match("element", "X")
    unique: dict[int, list[IsoClass]] = {}
    _LABELS = {1: "top", 2: "bridge", 3: "hollow"}

    with verbose_scope(_log, verbose):
        _log.debug("reduce_sites_by_isomorphism: %r n_shells=%d",
                   element, n_shells)
        _log.debug("  %3s  %-8s  %6s  %6s", "k", "type", "raw", "unique")
        _log.debug("  " + "-" * 30)

        for k, cliques in sorted(sites_by_k.items()):
            class_reps: list[nx.Graph] = []
            class_keys: list[tuple]    = []
            iso_ids:    list[int]      = []

            for clique in cliques:
                ego  = _build_clique_ego(G, clique, n_shells)
                fkey = _iso_prefilter_key(ego)
                assigned = False
                for cid, (rep, rkey) in enumerate(zip(class_reps, class_keys)):
                    if fkey != rkey:
                        continue
                    if isomorphism.GraphMatcher(ego, rep,
                                                node_match=node_match).is_isomorphic():
                        iso_ids.append(cid)
                        assigned = True
                        break
                if not assigned:
                    class_reps.append(ego)
                    class_keys.append(fkey)
                    iso_ids.append(len(class_reps) - 1)

            classes: list[IsoClass] = []
            for cid in range(len(class_reps)):
                members  = [cliques[i] for i, iso in enumerate(iso_ids) if iso == cid]
                rep      = members[0]
                pos_arr  = np.array([surf_pos[n] for n in rep if n in surf_pos])
                centroid = pos_arr.mean(axis=0) if len(pos_arr) else None
                classes.append(IsoClass(
                    k              = k,
                    iso_class      = cid,
                    n_shells       = n_shells,
                    representative = rep,
                    members        = members,
                    centroid       = centroid,
                    ego_graph      = class_reps[cid],
                ))
            unique[k] = classes

            label = _LABELS.get(k, f"{k}-fold")
            _log.debug("  %3d  %-8s  %6d  %6d", k, label, len(cliques), len(classes))

        total_raw    = sum(len(v) for v in sites_by_k.values())
        total_unique = sum(len(v) for v in unique.values())
        _log.debug("  " + "-" * 30)
        _log.debug("       %-8s  %6d  %6d", "total", total_raw, total_unique)

    cache.unique_sites.setdefault(element, {})[n_shells] = unique
    return unique


def optimise_site_positions(
    G: nx.Graph,
    element: str,
    *,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    n_shells: int = N_SHELLS_DEFAULT,
    repulsion_cutoff: float | None = SITE_REPULSION_CUTOFF,
    verbose: bool = False,
) -> dict[int, list[np.ndarray]]:
    """Compute the optimal adsorbate position for every enumerated site.

    Stored at ``G.graph['site_positions'][element]``.

    If :func:`reduce_sites_by_isomorphism` has already been called, the
    cached :class:`IsoClass` records for every shell depth are *replaced*
    with new instances carrying the freshly-optimised position
    (``dataclasses.replace`` rather than mutating ``IsoClass.position`` in
    place — this makes the data flow explicit and avoids surprising
    aliasing between the position-and-iso-class stages).
    """
    cache = get_cache(G)
    if element not in cache.sites:
        raise KeyError(
            f"No sites found for '{element}'. "
            "Call find_sites_for_element(G, element) first."
        )
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])
    sites_by_k: dict[int, list[frozenset]] = cache.sites[element]

    with verbose_scope(_log, verbose):
        _log.debug(
            "optimise_site_positions: %r r_cov=%.4f Å opt_factor=%g "
            "repulsion_weight=%g n_shells=%d cutoff=%s",
            element, r_cov, opt_factor, repulsion_weight, n_shells,
            repulsion_cutoff,
        )

        positions: dict[int, list[np.ndarray]] = {}
        _LABELS = {1: "top", 2: "bridge", 3: "hollow"}

        for k, cliques in sorted(sites_by_k.items()):
            pos_list: list[np.ndarray] = []
            for clique in cliques:
                p = _optimize_site_position(
                    G, clique, r_cov,
                    opt_factor=opt_factor,
                    repulsion_weight=repulsion_weight,
                    n_shells=n_shells,
                    repulsion_cutoff=repulsion_cutoff,
                )
                pos_list.append(p)
            positions[k] = pos_list
            label = _LABELS.get(k, f"{k}-fold")
            _log.debug("  k=%d %-8s %4d positions optimised",
                       k, label, len(cliques))

    cache.site_positions[element] = positions

    # ------------------------------------------------------------------
    # Replace cached IsoClass records with new instances that include the
    # freshly-computed position.  We construct fresh dataclass instances
    # via dataclasses.replace rather than mutating in place — see the
    # docstring rationale above.
    # ------------------------------------------------------------------
    if element in cache.unique_sites:
        for n_shells_cached, unique in cache.unique_sites[element].items():
            new_unique: dict[int, list[IsoClass]] = {}
            for k, classes in unique.items():
                k_cliques = sites_by_k.get(k, [])
                new_classes: list[IsoClass] = []
                for iso in classes:
                    try:
                        idx = k_cliques.index(iso.representative)
                        new_classes.append(replace(iso, position=positions[k][idx]))
                    except ValueError:
                        new_classes.append(iso)
                new_unique[k] = new_classes
            cache.unique_sites[element][n_shells_cached] = new_unique

    return positions

