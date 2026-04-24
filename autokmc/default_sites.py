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

Public API
----------
* find_sites_for_element      -- enumerate all raw sites
* reduce_sites_by_isomorphism -- group into iso-classes at a given shell depth
* optimise_site_positions     -- compute optimal adsorbate position per site
* k_max_for_element           -- k_max only (no site enumeration)
* k_max_for_radius            -- low-level k_max from a radius float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS


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
        Mean Cartesian position of the representative clique atoms (A).
    ego_graph : nx.Graph
        The n-shell ego-subgraph of the representative (used for matching).
    """
    k              : int
    iso_class      : int
    n_shells       : int
    representative : frozenset
    members        : list = field(default_factory=list)
    centroid       : Any  = None
    ego_graph      : Any  = None
    position       : Any  = None   # optimised adsorbate Cartesian position (Å)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_co_bond_graph(
    surface_graph: nx.Graph,
    r_cov_ads: float,
    co_factor: float = 0.95,
) -> nx.Graph:
    """Build the adsorbate-specific co-bonding graph on surface atoms.

    Two surface atoms i, j are connected when an adsorbate with covalent
    radius r_cov_ads can simultaneously bond to both:
        d(i, j) <= co_factor * (2*r_cov_ads + r_cov_i + r_cov_j)
    """
    surf_nodes = [(n, d) for n, d in surface_graph.nodes(data=True)
                  if d["type"] == "surface"]
    surf_list  = [n for n, _ in surf_nodes]
    surf_pos   = {n: d["position"]        for n, d in surf_nodes}
    surf_rcov  = {n: d["covalent_radius"] for n, d in surf_nodes}

    cell = np.array(surface_graph.graph["cell"], dtype=float)
    pbc  = np.asarray(surface_graph.graph.get("pbc", [True, True, False]), dtype=bool)

    # For non-periodic structures (nanoparticles) the cell may be a zero
    # matrix or otherwise singular.  Fall back to plain Euclidean distances.
    use_mic = pbc.any()
    if use_mic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            use_mic = False

    cbg = nx.Graph()
    cbg.add_nodes_from((n, dict(surface_graph.nodes[n])) for n in surf_list)

    for ii in range(len(surf_list)):
        for jj in range(ii + 1, len(surf_list)):
            ni, nj = surf_list[ii], surf_list[jj]
            cutoff = co_factor * (2.0 * r_cov_ads + surf_rcov[ni] + surf_rcov[nj])
            dv = surf_pos[nj] - surf_pos[ni]
            if use_mic:
                frac = dv @ cell_inv
                for i in range(3):
                    if pbc[i]:
                        frac[i] -= np.round(frac[i])
                dist = float(np.linalg.norm(frac @ cell))
            else:
                dist = float(np.linalg.norm(dv))
            if dist <= cutoff:
                cbg.add_edge(ni, nj)

    return cbg


def _build_clique_ego(
    surface_graph: nx.Graph,
    clique: frozenset,
    n_shells: int,
) -> nx.Graph:
    """Return the n-shell ego-subgraph of surface_graph around clique.

    Expands outward shell by shell from the clique nodes through the full
    graph (all node types, so subsurface atoms contribute at n_shells >= 2).

    Parameters
    ----------
    surface_graph : nx.Graph
        Full atom graph from build_graph.
    clique : frozenset[int]
        Seed nodes (global atom indices).
    n_shells : int
        Number of neighbor shells to expand.  1 = direct neighbors only,
        2 = neighbors of neighbors, etc.
    """
    frontier = set(clique)
    visited  = set(clique)

    for _ in range(n_shells):
        next_shell: set[int] = set()
        for n in frontier:
                if n in surface_graph:
                    next_shell.update(surface_graph.neighbors(n))
        frontier = next_shell - visited
        visited |= frontier

    return surface_graph.subgraph(visited).copy()


def _clique_centroid(
    G: nx.Graph,
    clique: frozenset,
    cell: np.ndarray,
    cell_inv: np.ndarray,
    pbc: np.ndarray,
    use_mic: bool,
) -> np.ndarray:
    """MIC-aware centroid of the atom positions in *clique*."""
    positions = np.array([G.nodes[n]["position"] for n in clique])
    ref = positions[0]
    dv = positions - ref
    if use_mic:
        frac = dv @ cell_inv
        for i in range(3):
            if pbc[i]:
                frac[:, i] -= np.round(frac[:, i])
        dv = frac @ cell
    return ref + dv.mean(axis=0)


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
    to the site *centroid*.  Falls back to [0, 0, 1] if the centroid
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
    """Return minimum-image distances from each row of ref_pos to point p.

    Parameters
    ----------
    p : np.ndarray, shape (3,)
        Query point (Cartesian, Å).
    ref_pos : np.ndarray, shape (N, 3)
        Reference positions (Cartesian, Å).
    cell : np.ndarray, shape (3, 3)
        Lattice matrix (rows = lattice vectors).
    cell_inv : np.ndarray, shape (3, 3)
        Inverse of cell.
    pbc : np.ndarray, shape (3,) bool
        Periodic boundary flags for each lattice direction.

    Returns
    -------
    dists : np.ndarray, shape (N,)
    """
    dv   = p[np.newaxis, :] - ref_pos          # (N, 3) Cartesian
    frac = dv @ cell_inv                        # (N, 3) fractional
    for i in range(3):
        if pbc[i]:
            frac[:, i] -= np.round(frac[:, i])
    return np.linalg.norm(frac @ cell, axis=1)


def _optimize_site_position(
    G: nx.Graph,
    clique: frozenset,
    r_cov_ads: float,
    *,
    opt_factor: float = 0.85,
    repulsion_weight: float = 0.1,
) -> np.ndarray:
    """Find the optimal Cartesian position for an adsorbate at a given site.

    The objective function has two terms:

    *Bond term* – penalises deviation from the ideal bond length to each
    bonded surface atom::

        d_ideal(i) = opt_factor * (r_cov_ads + r_cov_i)
        bond_term  = sum_i (|p - pos_i|_MIC - d_ideal_i)^2

    *Repulsion term* – soft repulsion from non-bonded surface atoms,
    pushing the adsorbate away from neighbours it is NOT bonded to::

        repulsion = sum_j 1 / |p - pos_j|_MIC^2   (j not in clique)

    For periodic slabs the adsorbate is constrained to ``z >= z_floor``
    (the maximum z of the bonded atoms) and optimised with L-BFGS-B.
    For non-periodic nanoparticles the adsorbate is constrained to lie on
    the outward side of the clique centroid (``dot(p - centroid, n_out) >= 0``
    where ``n_out`` points from the nanoparticle centre to the site) and
    optimised with SLSQP.

    Parameters
    ----------
    G : nx.Graph
        Full atom graph (must contain surface nodes with ``position`` and
        ``covalent_radius`` attributes, and graph-level ``"cell"``/``"pbc"``).
    clique : frozenset[int]
        Global atom indices of the bonded surface atoms.
    r_cov_ads : float
        Covalent radius of the adsorbate (Å).
    opt_factor : float
        Scales the ideal bond length for geometric position optimisation.
        Default 0.85.
    repulsion_weight : float
        Relative weight of the non-bonded repulsion term.  Default 0.1.

    Returns
    -------
    position : np.ndarray, shape (3,)
        Optimised Cartesian coordinates for the adsorbate (Å).
    """
    from scipy.optimize import minimize

    cell     = np.array(G.graph["cell"],       dtype=float)
    pbc      = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)

    # For non-periodic structures the cell may be singular; fall back to
    # plain Euclidean distances (equivalent to MIC with no wrapping).
    use_mic = pbc.any()
    if use_mic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            use_mic = False

    bonded      = list(clique)
    bonded_pos  = np.array([G.nodes[n]["position"]        for n in bonded])
    bonded_rcov = np.array([G.nodes[n]["covalent_radius"] for n in bonded])
    ideal_dists = opt_factor * (r_cov_ads + bonded_rcov)

    clique_set = set(clique)
    nb_pos_list = [
        d["position"]
        for n, d in G.nodes(data=True)
        if d.get("type") == "surface" and n not in clique_set
    ]
    nb_pos = np.array(nb_pos_list) if nb_pos_list else np.empty((0, 3))

    # ------------------------------------------------------------------ #
    # MIC-aware centroid of the bonded atoms                             #
    # ------------------------------------------------------------------ #
    ref = bonded_pos[0]
    if use_mic:
        dv_bonded = bonded_pos - ref
        frac_dv   = dv_bonded @ cell_inv
        for i in range(3):
            if pbc[i]:
                frac_dv[:, i] -= np.round(frac_dv[:, i])
        mic_rel = frac_dv @ cell
    else:
        mic_rel = bonded_pos - ref
    centroid = ref + mic_rel.mean(axis=0)

    # ------------------------------------------------------------------ #
    # Objective                                                           #
    # ------------------------------------------------------------------ #
    def objective(p: np.ndarray) -> float:
        if use_mic:
            dists = _mic_distances(p, bonded_pos, cell, cell_inv, pbc)
        else:
            dists = np.linalg.norm(p[np.newaxis, :] - bonded_pos, axis=1)
        bond_term = float(np.sum((dists - ideal_dists) ** 2))
        if repulsion_weight > 0.0 and len(nb_pos):
            if use_mic:
                nb_dists = _mic_distances(p, nb_pos, cell, cell_inv, pbc)
            else:
                nb_dists = np.linalg.norm(p[np.newaxis, :] - nb_pos, axis=1)
            repulsion = float(np.sum(1.0 / (nb_dists ** 2)))
            return bond_term + repulsion_weight * repulsion
        return bond_term

    if use_mic:
        # -------------------------------------------------------------- #
        # Periodic slab: constrain z >= z_floor (L-BFGS-B)               #
        # -------------------------------------------------------------- #
        lateral_dists = np.linalg.norm(
            mic_rel[:, :2] - mic_rel[:, :2].mean(axis=0), axis=1)
        h_per_atom = np.sqrt(np.maximum(0.0, ideal_dists ** 2 - lateral_dists ** 2))
        z_floor = float(bonded_pos[:, 2].max())
        z0      = z_floor + float(h_per_atom.mean())
        x0      = np.array([centroid[0], centroid[1], z0])
        bounds  = [(None, None), (None, None), (z_floor, None)]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    else:
        # -------------------------------------------------------------- #
        # Nanoparticle: constrain outward from nanoparticle centre (SLSQP)#
        # -------------------------------------------------------------- #
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
    co_factor: float = 0.95,
) -> int:
    """Return k_max -- the largest clique of the co-bonding graph."""
    cbg = _build_co_bond_graph(surface_graph, r_cov_ads, co_factor)
    if cbg.number_of_nodes() == 0:
        return 1
    return max((len(c) for c in nx.find_cliques(cbg)), default=1)


def k_max_for_element(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = 0.95,
    verbose: bool = False,
) -> int:
    """Find k_max for element and cache it in G.graph['k_max'][element]."""
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])
    k_max = k_max_for_radius(G, r_cov, co_factor=co_factor)

    if verbose:
        print(f"k_max_for_element: '{element}'  r_cov={r_cov:.4f} A  k_max={k_max}")

    G.graph.setdefault("k_max", {})[element] = k_max
    return k_max


def find_sites_for_element(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = 0.95,
    verbose: bool = False,
) -> dict[int, list[frozenset]]:
    """Find all adsorption sites for element and store in G.graph['sites'][element].

    Enumerates every clique of size 1...k_max in the adsorbate co-bonding graph.
    Each clique is a frozenset of surface-atom global indices.

    Returns
    -------
    sites : dict[int, list[frozenset[int]]]
        Keyed by coordination number k.
        Also stored in G.graph['sites'][element].
    """
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])

    # ── Build co-bonding graph ────────────────────────────────────────────
    cbg = _build_co_bond_graph(G, r_cov, co_factor)

    # ── Resolve cell / pbc (needed for centroid MIC and hull check) ───────
    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)
    use_mic = pbc.any()
    cell_inv: np.ndarray | None = None
    if use_mic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            use_mic = False

    # ── k_max from largest clique ─────────────────────────────────────────
    if cbg.number_of_nodes() == 0:
        k_max = 1
    else:
        k_max = max((len(c) for c in nx.find_cliques(cbg)), default=1)

    if verbose:
        print(f"find_sites_for_element: '{element}'  "
              f"r_cov={r_cov:.4f} A  k_max={k_max}  "
              f"co_factor={co_factor}")

    # ── Enumerate all cliques of size 1 … k_max ───────────────────────────
    sites: dict[int, list[frozenset]] = {k: [] for k in range(1, k_max + 1)}
    seen:  set[frozenset] = set()

    # ── Pre-compute convex hull for nanoparticles (non-periodic) ─────────
    # Cliques whose centroid lies inside the hull are spurious wrap-around
    # sites and must be discarded before any downstream processing.
    hull_equations = None
    if not use_mic:
        try:
            from scipy.spatial import ConvexHull
            all_pos = np.array([d["position"] for _, d in G.nodes(data=True)])
            _hull = ConvexHull(all_pos)
            hull_equations = _hull.equations   # shape (nfacets, 4): [nx,ny,nz,d]
        except Exception:
            hull_equations = None   # fall back: no filtering

    # Threshold: centroid may sit up to 1 Å inside the hull (legitimate for
    # hollow facets where the geometric centre is slightly recessed).  Deeper
    # than this means the clique wraps around the interior of the particle.
    _HULL_TOL = -0.2   # Å  (negative = inward from hull surface)

    for clique in nx.enumerate_all_cliques(cbg):
        k = len(clique)
        if k > k_max:
            break
        key = frozenset(clique)
        if key not in seen:
            # Hull check for nanoparticles
            if hull_equations is not None:
                c = _clique_centroid(G, key, cell,
                                     cell_inv if cell_inv is not None else cell,
                                     pbc, use_mic)
                # max signed distance: > 0 → outside hull, < 0 → inside hull
                max_sd = float(np.max(hull_equations[:, :3] @ c + hull_equations[:, 3]))
                if max_sd < _HULL_TOL:
                    seen.add(key)   # mark seen so sub-cliques skip it too
                    continue
            seen.add(key)
            sites[k].append(key)

    # Drop k-levels that were entirely filtered out, and recompute k_max.
    sites = {k: v for k, v in sites.items() if v}
    if sites:
        k_max = max(sites)
    else:
        k_max = 1
        sites = {1: []}

    if verbose:
        _LABELS = {1: "top", 2: "bridge", 3: "hollow"}
        total = sum(len(v) for v in sites.values())
        for k, cliques in sorted(sites.items()):
            label = _LABELS.get(k, f"{k}-fold")
            print(f"  k={k}  {label:8s}  {len(cliques):4d} sites")
        print(f"  total : {total} sites")

    # ── Store in graph ────────────────────────────────────────────────────
    G.graph.setdefault("k_max", {})[element] = k_max
    G.graph.setdefault("sites", {})[element] = sites

    return sites


def reduce_sites_by_isomorphism(
    G: nx.Graph,
    element: str,
    *,
    n_shells: int = 1,
    verbose: bool = False,
) -> dict[int, list[IsoClass]]:
    """Group all sites for element into iso-classes using an n-shell ego-graph.

    For each coordination number k, two sites (cliques) are placed in the
    same iso-class when their n-shell ego-subgraphs are graph-isomorphic with
    element-label matching.

    Shell depth controls discrimination:
      n_shells=0  -- direct bonding neighbors only.  Coarse: fcc and hcp
                     hollows on Cu(111) look identical.
      n_shells=1  -- adds the second shell.  Fine: fcc vs hcp hollows are
                     now distinguished by the subsurface atom beneath the fcc
                     site.
      n_shells=2+ -- deeper shells; useful for step-edge or defect sites.

    Results are stored independently for every depth:
      G.graph['unique_sites'][element][1]   -- coarse
      G.graph['unique_sites'][element][2]   -- standard
      G.graph['unique_sites'][element][3]   -- fine

    Parameters
    ----------
    G : nx.Graph
        find_sites_for_element must have been called first.
    element : str
    n_shells : int
        Default 1.
    verbose : bool
        Print a raw vs unique count table.

    Returns
    -------
    unique : dict[int, list[IsoClass]]
        Stored in G.graph['unique_sites'][element][n_shells].

    Raises
    ------
    KeyError
        If sites have not been enumerated yet.
    """
    if "sites" not in G.graph or element not in G.graph["sites"]:
        raise KeyError(
            f"No sites found for '{element}'. "
            "Call find_sites_for_element(G, element) first."
        )

    sites_by_k: dict[int, list[frozenset]] = G.graph["sites"][element]

    surf_pos = {n: d["position"] for n, d in G.nodes(data=True)
                if d.get("type") == "surface"}

    node_match = isomorphism.categorical_node_match("element", "X")
    unique: dict[int, list[IsoClass]] = {}
    _LABELS = {1: "top", 2: "bridge", 3: "hollow"}

    if verbose:
        print(f"reduce_sites_by_isomorphism: '{element}'  n_shells={n_shells}")
        print(f"  {'k':>3}  {'type':8s}  {'raw':>6}  {'unique':>6}")
        print("  " + "-" * 30)

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

        if verbose:
            label = _LABELS.get(k, f"{k}-fold")
            print(f"  {k:>3}  {label:8s}  {len(cliques):>6}  {len(classes):>6}")

    if verbose:
        total_raw    = sum(len(v) for v in sites_by_k.values())
        total_unique = sum(len(v) for v in unique.values())
        print("  " + "-" * 30)
        print(f"  {'':>3}  {'total':8s}  {total_raw:>6}  {total_unique:>6}")

    G.graph.setdefault("unique_sites", {}).setdefault(element, {})[n_shells] = unique
    return unique


def optimise_site_positions(
    G: nx.Graph,
    element: str,
    *,
    opt_factor: float = 0.85,
    repulsion_weight: float = 0.1,
    verbose: bool = False,
) -> dict[int, list[np.ndarray]]:
    """Compute the optimal adsorbate position for every enumerated site.

    For each raw site (clique) stored in ``G.graph['sites'][element]``,
    runs a local geometry optimisation that:

    * Places the adsorbate at the ideal bond-length distance from each
      bonded surface atom (``opt_factor * (r_cov_ads + r_cov_i)``).
    * Maximises the distance from non-bonded surface atoms via a soft
      ``1/r²`` repulsion term (important for top and bridge sites where
      there is lateral freedom).

    Results are stored in ``G.graph['site_positions'][element]`` as a
    ``dict[int, list[np.ndarray]]`` (keyed by coordination number *k*).

    If ``reduce_sites_by_isomorphism`` has already been called, the
    optimised position of each representative clique is also written into
    the corresponding ``IsoClass.position`` field for every shell depth.

    Parameters
    ----------
    G : nx.Graph
        ``find_sites_for_element`` must have been called first.
    element : str
    opt_factor : float
        Passed to :func:`_optimize_site_position`.  Default 0.85.
    repulsion_weight : float
        Weight of the non-bonded repulsion term.  Default 0.1.
    verbose : bool
        Print a per-k summary of how many positions were optimised.

    Returns
    -------
    positions : dict[int, list[np.ndarray]]
        Also stored in ``G.graph['site_positions'][element]``.

    Raises
    ------
    KeyError
        If ``find_sites_for_element`` has not been called yet.
    """
    if "sites" not in G.graph or element not in G.graph["sites"]:
        raise KeyError(
            f"No sites found for '{element}'. "
            "Call find_sites_for_element(G, element) first."
        )

    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")

    r_cov = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])
    sites_by_k: dict[int, list[frozenset]] = G.graph["sites"][element]

    if verbose:
        print(f"optimise_site_positions: '{element}'  "
              f"r_cov={r_cov:.4f} Å  opt_factor={opt_factor}  "
              f"repulsion_weight={repulsion_weight}")

    positions: dict[int, list[np.ndarray]] = {}
    _LABELS = {1: "top", 2: "bridge", 3: "hollow"}

    for k, cliques in sorted(sites_by_k.items()):
        pos_list: list[np.ndarray] = []
        for clique in cliques:
            p = _optimize_site_position(
                G, clique, r_cov,
                opt_factor=opt_factor,
                repulsion_weight=repulsion_weight,
            )
            pos_list.append(p)
        positions[k] = pos_list

        if verbose:
            label = _LABELS.get(k, f"{k}-fold")
            print(f"  k={k}  {label:8s}  {len(cliques):4d} positions optimised")

    G.graph.setdefault("site_positions", {})[element] = positions

    # ------------------------------------------------------------------
    # Propagate to IsoClass.position for every shell depth already stored
    # ------------------------------------------------------------------
    if "unique_sites" in G.graph and element in G.graph["unique_sites"]:
        for _n_shells, unique in G.graph["unique_sites"][element].items():
            for k, classes in unique.items():
                k_cliques = sites_by_k[k]
                for iso in classes:
                    try:
                        idx = k_cliques.index(iso.representative)
                        iso.position = positions[k][idx]
                    except ValueError:
                        pass  # representative not found – skip gracefully

    return positions

