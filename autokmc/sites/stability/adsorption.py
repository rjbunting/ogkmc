"""
autokmc.sites.stability.adsorption
=============================
On-the-fly lateral-interaction classifier and ML-potential stability checker
for materialised adsorbate sites.

**Lateral classification** (:func:`check_adsorbate_site_lateral`)
------------------------------------------------------------------
When a specific adsorbate-site member on the graph has occupied neighbours,
the local environment — surface topology plus the pattern of occupied
neighbouring adsorbate nodes — constitutes a *lateral interaction* that
modifies the site's effective adsorption energy or barrier.

Because the combinatorics of all possible neighbour occupancies are enormous,
lateral classes are enumerated **lazily** rather than upfront:

1. Call :func:`check_adsorbate_site_lateral` for any specific member of an
   :class:`~autokmc.sites.adsorbate.AdsorbateSite`.
2. The function builds the *lateral ego-graph* around that member's bonded
   surface clique — a BFS through surface nodes (to depth ``n_shells``) with
   occupied adsorbate nodes included as labelled leaves.
3. The ego-graph is compared by isomorphism to every
   :class:`~autokmc.sites.adsorbate.AdsorbateSiteLateral` already stored
   on the parent :class:`~autokmc.sites.adsorbate.AdsorbateSite`.
4. If a matching lateral class is found, ``member_index`` is appended to its
   ``members`` list.  Otherwise a new ``AdsorbateSiteLateral`` is created and
   appended to ``adsorbate_site.lateral_classes``.
5. The matching (or newly created)
   :class:`~autokmc.sites.adsorbate.AdsorbateSiteLateral` is returned.

**Stability checking** (:func:`check_site_stability`)
------------------------------------------------------
Given a lateral class (from :func:`check_adsorbate_site_lateral`), build two
ASE :class:`~ase.Atoms` objects from the graph:

* **occupied** — full slab + relevant occupied lateral-neighbour adsorbates
  + the site itself.
* **unoccupied** — the same slab + lateral neighbours, *without* the site.

Both are relaxed with an ML potential via
:func:`~autokmc.structure.optimise_structure`.  After each relaxation the
atom-connectivity graph (ASE :class:`~ase.neighborlist.NeighborList`) is
compared against the pre-relaxation connectivity; if any bond appears or
disappears, an appropriate :class:`SiteStabilityError` subclass is raised.
Energies are stored on the :class:`~autokmc.sites.adsorbate.AdsorbateSiteLateral`
object and returned as ``(E_occupied, E_unoccupied)``.

Lateral ego-graph conventions
------------------------------
* **BFS seed** — the union of all bonded surface-atom cliques for the member
  being checked.
* **BFS frontier** — only ``type == "surface"`` nodes are traversed.  Anchor
  bookkeeping nodes are always skipped.
* **Occupied adsorbate leaves** — after the BFS, any occupied
  ``type == "adsorbate"`` node adjacent to *any* surface node in the BFS set
  is added as a leaf (not traversed further).  This captures the nearest
  occupied adsorbate neighbours without recursively nesting their environments.
* **Self inclusion** — the adsorbate-site's own nodes are included as leaves
  and stamped ``occupied=True`` in the ego-graph copy, so the isomorphism
  match is consistent whether the site is physically occupied or not.

Node-match semantics for isomorphism
-------------------------------------
* ``type == "surface"``   : must share ``element``.
* ``type == "adsorbate"`` : must share ``element``, ``iso_class``, and
  ``reactant`` (SMILES).

Public API
----------
* :class:`SiteStabilityError`        — base error for stability failures.
* :class:`SurfaceConnectivityError`  — surface bonds changed after relaxation.
* :class:`AdsorbateDissociationError`— adsorbate broke apart after relaxation.
* :class:`OptimisationFailedError`   — LBFGS did not converge.
* :func:`check_adsorbate_site_lateral` — classify the lateral environment of
  one specific member; updates ``adsorbate_site.lateral_classes`` in place.
* :func:`check_site_stability`       — relax occupied / unoccupied structures,
  check connectivity, store and return energies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs

from autokmc.io.calculators import acquire_calculator
from autokmc.io.calculation_cache import (
    apply_cached_states,
    calculation_cache_key,
    calculator_identity,
    load_calculation_record,
    make_calculation_record,
    state_payload,
    write_calculation_record,
)
from autokmc.io.reaction_graph import normalise_reaction_graph
from autokmc.species.smiles import smiles_to_dirname
from autokmc.core.pbc import full_pbc_for_cell
from autokmc.sites.adsorbate import AdsorbateSite, AdsorbateSiteLateral
from autokmc.core.constants import NL_MULT_DEFAULT, LATERAL_SHELLS_DEFAULT
from autokmc.utils.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stability-check error hierarchy
# ---------------------------------------------------------------------------

class SiteStabilityError(Exception):
    """Base class for all site-stability failures."""


class SurfaceConnectivityError(SiteStabilityError):
    """Surface bond topology changed after ML relaxation."""


class AdsorbateDissociationError(SiteStabilityError):
    """Adsorbate connectivity changed (bond broke or formed) after ML relaxation."""


class OptimisationFailedError(SiteStabilityError):
    """LBFGS relaxation did not converge within the allowed number of steps."""


# ---------------------------------------------------------------------------
# Lateral ego-graph builder
# ---------------------------------------------------------------------------

def _surface_bfs_shells(
    G: nx.Graph,
    seed_clique: frozenset,
    n_shells: int,
) -> frozenset:
    """Cached n-shell surface-only BFS expansion of *seed_clique*.

    Surface topology is invariant during a KMC simulation, so the BFS
    through ``type == "surface"`` nodes can be computed once and reused.
    Cached per (seed_clique, n_shells) on
    ``G.graph["_surface_shells_cache"]`` (see suggestion.MD #5).

    Returns
    -------
    frozenset[int]
        Every surface node id reachable from any node in *seed_clique* in
        ≤ ``n_shells`` hops, traversing only ``type == "surface"`` edges.
    """
    cache: dict = G.graph.setdefault("_surface_shells_cache", {})
    key = (seed_clique, int(n_shells))
    hit = cache.get(key)
    if hit is not None:
        return hit

    frontier: set = set(seed_clique)
    visited:  set = set(frontier)
    for _ in range(int(n_shells)):
        nxt: set = set()
        for n in frontier:
            if G.nodes[n].get("type") != "surface":
                continue
            for nb in G.neighbors(n):
                if nb in visited:
                    continue
                d = G.nodes[nb]
                if d.get("type") == "surface":
                    nxt.add(nb)
                # anchor / unoccupied-adsorbate / other types: skipped.
        frontier = nxt - visited
        visited |= frontier

    out = frozenset(visited)
    cache[key] = out
    return out


def _build_lateral_ego_graph(
    G: nx.Graph,
    seed_clique: frozenset,
    n_shells: int,
    *,
    self_node_ids: frozenset | None = None,
    ignore_occupied_neighbours: bool = False,
) -> nx.Graph:
    """Build an n-shell ego-subgraph for lateral-interaction matching.

    Traverses only ``type == "surface"`` nodes (anchor nodes are always
    skipped).  After the BFS is complete, every occupied
    ``type == "adsorbate"`` node that is adjacent to at least one surface node
    in the BFS set — and is not in *self_node_ids* — is included as a leaf.

    The static surface-only BFS is delegated to
    :func:`_surface_bfs_shells` so the result is cached across every call
    that shares the same ``(seed_clique, n_shells)`` (see suggestion.MD #5).
    Only the per-call adsorbate-leaf collection is recomputed, since
    occupancy changes step to step.

    Parameters
    ----------
    G : nx.Graph
        The full surface + adsorbate graph.
    seed_clique : frozenset[int]
        Surface-atom node ids that form the BFS root (i.e. the union of all
        bonded surface cliques for the member being checked).
    n_shells : int
        BFS depth through surface nodes.
    self_node_ids : frozenset[int] | None
        Node ids of the adsorbate member being checked.  These are excluded
        from the returned graph so the site does not appear in its own
        environment.
    ignore_occupied_neighbours : bool
        When ``True``, neighbouring occupied adsorbate nodes are **not**
        collected as leaves.  Only the site's own *self_node_ids* are added.
        This collapses all members to a single "bare" lateral class (lat0),
        effectively disabling lateral interactions.  Default ``False``.

    Returns
    -------
    nx.Graph
        Induced subgraph copy containing the BFS surface nodes plus any
        occupied adsorbate neighbours.  Node attributes are preserved from *G*.
    """
    self_ids: frozenset = frozenset(self_node_ids) if self_node_ids else frozenset()

    # Static surface BFS (cached, invariant during the KMC loop).
    visited_full = _surface_bfs_shells(G, seed_clique, n_shells)
    visited: set = set(visited_full) - self_ids

    # ── Collect adsorbate leaves adjacent to the BFS surface set ────────────
    # Two categories are included:
    #   1. Genuinely occupied adsorbate nodes (neighbours of the BFS surface)
    #      — skipped when ``ignore_occupied_neighbours=True``.
    #   2. The site's own nodes (self_ids) — treated as occupied regardless of
    #      their current ``occupied`` flag on G, because we are evaluating the
    #      environment *as if* this site were occupied.
    ads_leaves: set = set()
    for n in visited:
        for nb in G.neighbors(n):
            if nb in visited:
                continue
            d = G.nodes[nb]
            if d.get("type") != "adsorbate":
                continue
            if nb in self_ids:
                ads_leaves.add(nb)
            elif not ignore_occupied_neighbours and d.get("occupied", False):
                ads_leaves.add(nb)

    result = G.subgraph(visited | ads_leaves).copy()

    # Stamp the site's own nodes as occupied in the copy so the iso-match
    # sees them exactly like any other occupied adsorbate leaf.
    for nid in self_ids:
        if nid in result.nodes:
            result.nodes[nid]["occupied"] = True

    return result


# ---------------------------------------------------------------------------
# Fingerprint (cheap pre-filter before full isomorphism test)
# ---------------------------------------------------------------------------

def _lateral_fingerprint(g: nx.Graph) -> tuple:
    """Cheap fingerprint for a lateral ego-graph.

    Unequal fingerprints guarantee non-isomorphism.  Node signatures include:

    * node type
    * element
    * iso_class  (adsorbate nodes only, else ``-1``)
    * reactant SMILES (adsorbate nodes only, else empty string)
    * graph degree
    """
    node_sigs = tuple(sorted(
        (
            d.get("type",      "X"),
            d.get("element",   "X"),
            int(d.get("iso_class", -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",  "")) if d.get("type") == "adsorbate" else "",
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (g.number_of_nodes(), g.number_of_edges(), node_sigs)


# ---------------------------------------------------------------------------
# Node-match predicate for GraphMatcher
# ---------------------------------------------------------------------------

def _lateral_node_match(d1: dict, d2: dict) -> bool:
    """Return ``True`` iff two nodes are compatible for lateral iso matching.

    * ``type == "surface"``   → must share ``element``.
    * ``type == "adsorbate"`` → must share ``element``, ``iso_class``, and
      ``reactant``.
    """
    if d1.get("type") != d2.get("type"):
        return False
    if d1.get("element") != d2.get("element"):
        return False
    if d1.get("type") == "adsorbate":
        if d1.get("iso_class") != d2.get("iso_class"):
            return False
        if d1.get("reactant") != d2.get("reactant"):
            return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_adsorbate_site_lateral(
    G: nx.Graph,
    adsorbate_site: AdsorbateSite,
    member_index: int,
    *,
    n_shells: int | None = None,
    ignore_lateral: bool = False,
) -> AdsorbateSiteLateral:
    """Classify the lateral-interaction environment of one specific member.

    Builds the lateral ego-graph around ``member_index`` (surface topology
    out to ``n_shells`` shells, plus occupied adsorbate leaves), compares it
    against every :class:`~autokmc.sites.adsorbate.AdsorbateSiteLateral`
    already recorded on *adsorbate_site*, and either appends the member to a
    matching lateral class or creates a new one.

    The result is written back onto ``adsorbate_site.lateral_classes`` in
    place **and** returned.

    Parameters
    ----------
    G : nx.Graph
        The full surface + adsorbate graph (from
        :func:`~autokmc.core.graph.build_graph` after
        :func:`~autokmc.sites.adsorbate.find_adsorbate_sites` and with
        at least some ``occupied=True`` adsorbate nodes set by the caller).
    adsorbate_site : AdsorbateSite
        Parent iso-class whose ``lateral_classes`` list will be updated.
    member_index : int
        Index into ``adsorbate_site.member_node_ids`` for the specific
        placement to classify.
    n_shells : int | None
        BFS depth for the lateral ego-graph.  ``None`` (default) uses
        :data:`~autokmc.core.constants.LATERAL_SHELLS_DEFAULT` (``0``), meaning
        only adsorbates that bond to the **same surface atoms** as the member
        are counted as lateral neighbours (clique-sharing criterion).  Pass
        ``1`` to also include adsorbates on first-nearest-neighbour surface
        atoms, etc.
    ignore_lateral : bool
        When ``True``, neighbouring occupied adsorbate nodes are excluded from
        the ego-graph so every member always maps to the single "bare" lat0.
        Effectively disables lateral interactions for this site.  Default
        ``False``.

    Returns
    -------
    AdsorbateSiteLateral
        The lateral class this member belongs to (existing or newly created).
        ``adsorbate_site.lateral_classes`` is updated in place.

    Raises
    ------
    IndexError
        If *member_index* is out of range for *adsorbate_site*.
    ValueError
        If the member has no bonded surface atoms on *G* (seed clique is
        empty), which prevents building a meaningful ego-graph.

    Notes
    -----
    Calling this function repeatedly on the same *member_index* is safe: the
    second call checks whether ``member_index`` is already recorded in the
    matching lateral class before appending.

    The number of lateral classes grows monotonically as new occupancy
    patterns are encountered.  There is no automatic reset; to clear all
    lateral classes for a given site call
    ``adsorbate_site.lateral_classes.clear()``.
    """
    if member_index < 0 or member_index >= len(adsorbate_site.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — AdsorbateSite "
            f"iso_class={adsorbate_site.iso_class} has "
            f"{len(adsorbate_site.member_node_ids)} member(s)."
        )

    # ── Determine BFS depth ───────────────────────────────────────────────
    depth: int = LATERAL_SHELLS_DEFAULT if n_shells is None else int(n_shells)

    # ── Derive seed clique and self node ids ──────────────────────────────
    node_ids: list[int] = adsorbate_site.member_node_ids[member_index]

    seed_clique: frozenset = frozenset(
        surf_id
        for nid in node_ids
        if nid in G and G.nodes[nid].get("is_bonded", False)
        for surf_id in (G.nodes[nid].get("clique") or [])
    )

    if not seed_clique:
        raise ValueError(
            f"Member {member_index} of AdsorbateSite iso_class="
            f"{adsorbate_site.iso_class} has no bonded surface atoms on G — "
            "cannot build a lateral ego-graph.  Ensure the member's adsorbate "
            "nodes are present on G and have 'is_bonded=True'."
        )

    self_node_ids: frozenset = frozenset(nid for nid in node_ids if nid in G)

    # ── Build lateral ego-graph ───────────────────────────────────────────
    ego = _build_lateral_ego_graph(
        G, seed_clique, depth, self_node_ids=self_node_ids,
        ignore_occupied_neighbours=ignore_lateral,
    )

    fkey = _lateral_fingerprint(ego)

    # ── Compare against existing lateral classes ──────────────────────────
    # Lateral classes are bucketed by their cached fingerprint on the parent
    # site so we only run the GraphMatcher on collisions instead of scanning
    # every existing class (suggestion.MD #5).  The fingerprint cache lives
    # on a per-site dict-of-list; ``lc._fingerprint`` is set at creation and
    # never recomputed.
    fp_index: dict = getattr(adsorbate_site, "_lateral_fp_index", None)
    if fp_index is None:
        fp_index = {}
        adsorbate_site._lateral_fp_index = fp_index  # type: ignore[attr-defined]

    def _drop_from_other_classes(new_lc=None) -> None:
        for other in adsorbate_site.lateral_classes:
            if other is new_lc:
                continue
            if member_index in other.members:
                other.members.remove(member_index)

    for lc in fp_index.get(fkey, ()):
        if lc.n_shells != depth or lc.ego_graph is None:
            continue
        gm = isomorphism.GraphMatcher(
            ego, lc.ego_graph,
            node_match=_lateral_node_match,
        )
        if gm.is_isomorphic():
            _drop_from_other_classes(new_lc=lc)
            if member_index not in lc.members:
                lc.members.append(member_index)
            _log.debug(
                "check_adsorbate_site_lateral: iso_class=%d member=%d "
                "→ existing lateral_class=%d",
                adsorbate_site.iso_class, member_index, lc.lateral_class,
            )
            return lc

    # ── No match — create a new lateral class ──────────��─────────────────
    _drop_from_other_classes(new_lc=None)
    new_lc = AdsorbateSiteLateral(
        lateral_class = len(adsorbate_site.lateral_classes),
        ego_graph     = ego,
        n_shells      = depth,
        members       = [member_index],
    )
    new_lc._fingerprint = fkey  # type: ignore[attr-defined]
    adsorbate_site.lateral_classes.append(new_lc)
    fp_index.setdefault(fkey, []).append(new_lc)

    _log.debug(
        "check_adsorbate_site_lateral: iso_class=%d member=%d "
        "→ new lateral_class=%d  (total lateral classes=%d)",
        adsorbate_site.iso_class, member_index,
        new_lc.lateral_class, len(adsorbate_site.lateral_classes),
    )
    return new_lc


# ---------------------------------------------------------------------------
# Stability-check helpers
# ---------------------------------------------------------------------------

def _expand_to_full_placement(G: nx.Graph, seed_node_ids: set[int]) -> set[int]:
    """Expand a set of adsorbate node ids to every atom in the same placement.

    Uses the ``siblings`` attribute materialised on each adsorbate node by
    :func:`~autokmc.sites.adsorbate._materialise_adsorbate_nodes` so
    that, e.g., if only the bonded C of a CO placement is given, the dangling
    O is automatically included.
    """
    result: set[int] = set()
    for nid in seed_node_ids:
        if nid not in G:
            continue
        result.add(nid)
        for s in G.nodes[nid].get("siblings", ()):
            if int(s) in G:
                result.add(int(s))
    return result


def _build_stability_atoms(
    G: nx.Graph,
    lateral_class: AdsorbateSiteLateral,
    self_node_ids: frozenset,
    *,
    include_self: bool,
    frozen_indices: list[int] | None = None,
) -> tuple[Atoms, int, int]:
    """Build an ASE Atoms object for an ML-potential stability calculation.

    Atom ordering
    -------------
    1. All slab atoms (``type == "bulk"`` or ``"surface"``), sorted by their
       original ASE ``index`` attribute so the order matches the slab from
       which the graph was built.
    2. Lateral-neighbour adsorbate atoms — all atoms of every occupied-neighbour
       placement found in ``lateral_class.ego_graph``, expanded to full
       placements via :func:`_expand_to_full_placement`.
    3. [``include_self=True`` only] The atoms of the site being checked.

    Parameters
    ----------
    G : nx.Graph
    lateral_class : AdsorbateSiteLateral
        Provides the occupied-neighbour node ids via its ``ego_graph``.
    self_node_ids : frozenset[int]
        Node ids of the member being checked (included as group 3 or excluded).
    include_self : bool
        Include the site's own atoms (occupied state) or not (unoccupied).
    frozen_indices : list[int] | None
        Atom indices within the **slab** group (0-based, same ordering as the
        slab atoms in the returned object) to freeze with
        :class:`~ase.constraints.FixAtoms`.  ``None`` → no constraints.

    Returns
    -------
    atoms : Atoms
    n_slab : int
        Number of slab atoms at the start of *atoms*.
    n_lat : int
        Number of lateral-neighbour adsorbate atoms (``len(lat_nodes)``),
        occupying indices ``n_slab … n_slab+n_lat-1`` in *atoms*.
    n_self : int
        Number of site-own adsorbate atoms actually placed (``len(self_nodes)``
        — nodes in *self_node_ids* that are present in *G* at call time),
        occupying indices ``n_slab+n_lat … n_slab+n_lat+n_self-1`` in *atoms*.
    """
    # ── 1. Slab atoms ────────────────────────────────────────────────────
    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True)
         if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )

    # ── 2. Lateral-neighbour adsorbate atoms ─────────────────────────────
    lat_seed: set[int] = set()
    if lateral_class.ego_graph is not None:
        for n, d in lateral_class.ego_graph.nodes(data=True):
            if d.get("type") == "adsorbate" and n not in self_node_ids:
                lat_seed.add(n)
    lat_nodes: list[int] = sorted(_expand_to_full_placement(G, lat_seed))

    # ── 3. Self atoms (conditionally) ────────────────────────────────────
    self_nodes: list[int] = (
        sorted(nid for nid in self_node_ids if nid in G)
        if include_self else []
    )

    all_node_ids = slab_nodes + lat_nodes + self_nodes
    n_slab = len(slab_nodes)
    n_lat  = len(lat_nodes)
    n_self = len(self_nodes)

    symbols   = [G.nodes[n]["element"]  for n in all_node_ids]
    positions = [G.nodes[n]["position"] for n in all_node_ids]

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = full_pbc_for_cell(cell)

    atoms = Atoms(
        symbols   = symbols,
        positions = positions,
        cell      = cell,
        pbc       = pbc,
    )

    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))

    return atoms, n_slab, n_lat, n_self


def _bond_set(
    atoms: Atoms,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
    relevant_indices: set[int] | None = None,
) -> set[frozenset]:
    """Return the set of bonded atom-index pairs from an ASE NeighborList.

    Uses the same ``natural_cutoffs`` scheme as :func:`~autokmc.core.graph.build_graph`
    so the connectivity judgement is consistent with the graph that was built
    from the original slab.

    Parameters
    ----------
    relevant_indices : set[int] | None
        When supplied, only bonds where **at least one** endpoint is in
        *relevant_indices* are returned.  This is the suggestion.MD #6
        optimisation: the bond-topology stability check only cares about
        bonds touching the adsorbate region (slab-internal bonds practically
        never break under a stable potential), so the per-call work drops
        from O(N_atoms × ⟨coord⟩) to O(|relevant_indices| × ⟨coord⟩).
        ``None`` falls back to the legacy full-graph behaviour.
    """
    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    bonds: set[frozenset] = set()
    if relevant_indices is None:
        for i in range(len(atoms)):
            for j in nl.get_neighbors(i)[0]:
                bonds.add(frozenset((int(i), int(j))))
    else:
        for i in range(len(atoms)):
            i_relevant = i in relevant_indices
            for j in nl.get_neighbors(i)[0]:
                j_int = int(j)
                if i_relevant or j_int in relevant_indices:
                    bonds.add(frozenset((int(i), j_int)))
    return bonds


def _check_intended_coordination_stable(
    atoms_opt: Atoms,
    G: nx.Graph,
    self_node_ids,
    n_slab: int,
    n_lat: int,
    nl_mult: float,
    *,
    self_node_order: list[int] | None = None,
) -> None:
    """Raise :class:`AdsorbateDissociationError` if any self adsorbate atom lost
    its intended surface-clique bond after ML relaxation.

    The *atoms_opt* atom ordering is ``[slab | lat_neighbours | self]``.
    Self atoms are at ASE indices ``n_slab+n_lat … n_slab+n_lat+len(self_nodes)-1``.

    ``G.nodes[nid]["clique"]`` gives the frozenset of intended surface G-node ids
    for each adsorbate node.  The slab atoms in *atoms_opt* are sorted by
    ``G.nodes[n].get("index", n)`` so we can map G-node ids → ASE indices.

    Parameters
    ----------
    self_node_order : list[int] | None
        Explicit ordering of the self-block node ids matching the ASE atom
        order in *atoms_opt*.  Required when the self block was not built
        in sorted-by-node-id order — e.g. the diffusion endpoint builder
        orders the migrating molecule by SMILES ``reactant_index`` so that
        atoms align across A and B for the NEB.  When ``None`` (default,
        adsorption path) the function falls back to
        ``sorted(self_node_ids)``, matching :func:`_build_stability_atoms`.
    """
    # Map slab G-node id → ASE index in atoms_opt.
    slab_nodes_sorted = sorted(
        (n for n, d in G.nodes(data=True) if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )
    node_to_ase = {int(nid): i for i, nid in enumerate(slab_nodes_sorted)}

    # Self nodes in the same order as the caller laid them out in atoms_opt.
    if self_node_order is not None:
        self_nid_list = [int(nid) for nid in self_node_order if nid in G]
    else:
        # Default: matches _build_stability_atoms (sorted by node id).
        self_nid_list = sorted(int(nid) for nid in self_node_ids if nid in G)

    cutoffs = natural_cutoffs(atoms_opt, mult=nl_mult)
    nl      = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms_opt)

    missing_info: list[str] = []
    for j, nid in enumerate(self_nid_list):
        clq = G.nodes[nid].get("clique")
        if not clq:
            continue
        ase_self   = n_slab + n_lat + j
        neighbours = {int(k) for k in nl.get_neighbors(ase_self)[0]}
        for surf_nid in clq:
            ase_surf = node_to_ase.get(int(surf_nid))
            if ase_surf is None:
                continue
            if ase_surf not in neighbours:
                missing_info.append(
                    f"node {nid} → surf_node {surf_nid} "
                    f"(ASE ads={ase_self}, surf={ase_surf})"
                )

    if missing_info:
        detail = "  ".join(missing_info[:3]) + (
            "…" if len(missing_info) > 3 else ""
        )
        raise AdsorbateDissociationError(
            f"[occupied] Adsorbate lost intended surface bond after relaxation. "
            + detail
        )


def _check_connectivity_stable(
    atoms_before: Atoms,
    atoms_after:  Atoms,
    n_slab:       int,
    n_ads:        int,
    state_label:  str,
    nl_mult:      float,
    *,
    relevant_indices: set[int] | None = None,
    n_lat: int = 0,
) -> None:
    """Raise a :class:`SiteStabilityError` subclass if bond topology changed.

    Parameters
    ----------
    atoms_before, atoms_after : Atoms
        Structures before and after ML relaxation.
    n_slab : int
        Number of slab atoms (indices ``0 … n_slab-1``).
    n_ads : int
        Number of adsorbate atoms total (indices ``n_slab … n_slab+n_ads-1``).
        This covers **both** lateral-neighbour atoms (first ``n_lat``) and the
        site's own atoms (last ``n_ads - n_lat``).
    state_label : str
        ``"occupied"`` or ``"unoccupied"`` — used only for the error message.
    nl_mult : float
        Cutoff multiplier forwarded to :func:`_bond_set`.
    relevant_indices : set[int] | None
        Forwarded to :func:`_bond_set`.  When set, only bonds touching one
        of these indices are compared.  See suggestion.MD #6.
    n_lat : int
        Number of lateral-neighbour adsorbate atoms at the start of the
        adsorbate block (indices ``n_slab … n_slab+n_lat-1``).  Used to
        produce a more informative error message distinguishing whether a
        bond change involved the lateral neighbours or the site under test.
        Default ``0`` (no distinction made).

    Raises
    ------
    AdsorbateDissociationError
        A bond involving at least one adsorbate atom appeared or disappeared.
        The message identifies whether the affected atom belongs to a lateral
        neighbour or to the site being checked.

    Notes
    -----
    Pure slab–slab bond changes are **intentionally ignored**.  ML potentials
    (e.g. NequIP, MACE) naturally relax the top surface layer during endpoint
    optimisation, causing metal–metal bond topology to flicker as atoms move
    within the cutoff tolerance.  These slab-internal changes are physically
    irrelevant to adsorbate stability.  The ``relevant_indices`` parameter
    (set to adsorbate atom indices by all callers) is the designed mechanism
    to restrict the bond set to adsorbate-touching bonds only.
    """
    before = _bond_set(atoms_before, nl_mult=nl_mult,
                       relevant_indices=relevant_indices)
    after  = _bond_set(atoms_after,  nl_mult=nl_mult,
                       relevant_indices=relevant_indices)

    added   = after  - before
    removed = before - after
    changed = added | removed

    ads_set  = set(range(n_slab, n_slab + n_ads))
    lat_set  = set(range(n_slab, n_slab + n_lat))         # lateral neighbours
    site_set = set(range(n_slab + n_lat, n_slab + n_ads)) # site's own atoms

    # Only classify bond changes that involve at least one adsorbate atom.
    # Pure slab-slab bonds are intentionally excluded: ML potentials naturally
    # relax the top surface layer, causing Cu-Cu (or other metal) bond
    # topology to flicker as atoms move within the cutoff tolerance.  These
    # slab-internal changes are physically irrelevant to adsorbate stability
    # and must not cause a site to be marked invalid.  The `relevant_indices`
    # parameter (normally set to adsorbate indices by all callers) is the
    # designed mechanism to restrict the bond set; a separate slab-only scan
    # is explicitly NOT performed here.
    ads_changes = [b for b in changed if b & ads_set]     # any in adsorbate

    if not ads_changes:
        return

    if ads_changes:
        # Distinguish whether the bond change involves a lateral-neighbour
        # adsorbate or the site under test — important for debugging which
        # occupied neighbour is destabilising the configuration.
        lat_changes  = [b for b in ads_changes if b & lat_set  and not (b & site_set)]
        site_changes = [b for b in ads_changes if b & site_set]
        mixed        = [b for b in ads_changes
                        if b not in lat_changes and b not in site_changes]

        parts: list[str] = []
        if site_changes:
            pairs = ", ".join(f"{{{min(b)},{max(b)}}}" for b in site_changes[:3])
            parts.append(f"site-under-test bonds: {pairs}"
                         + ("…" if len(site_changes) > 3 else ""))
        if lat_changes:
            pairs = ", ".join(f"{{{min(b)},{max(b)}}}" for b in lat_changes[:3])
            parts.append(f"lateral-neighbour bonds: {pairs}"
                         + ("…" if len(lat_changes) > 3 else ""))
        if mixed:
            pairs = ", ".join(f"{{{min(b)},{max(b)}}}" for b in mixed[:3])
            parts.append(f"cross-group bonds: {pairs}"
                         + ("…" if len(mixed) > 3 else ""))

        raise AdsorbateDissociationError(
            f"[{state_label}] Adsorbate connectivity changed after relaxation. "
            + "  ".join(parts)
        )


# ---------------------------------------------------------------------------
# Public API — check_site_stability
# ---------------------------------------------------------------------------

def check_site_stability(
    G: nx.Graph,
    adsorbate_site: AdsorbateSite,
    member_index: int,
    lateral_class: AdsorbateSiteLateral,
    calculator,
    *,
    frozen_indices: list[int] | None = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    nl_mult: float = NL_MULT_DEFAULT,
    verbose: bool = False,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
) -> tuple[float, float]:
    """Relax the occupied and unoccupied structures and check for stability.

    Builds two ASE :class:`~ase.Atoms` objects from the graph:

    * **occupied**   — full slab + lateral-neighbour adsorbates + this site.
    * **unoccupied** — same slab + lateral neighbours, site absent.

    Each is relaxed with a calculator acquired from *calculator* via
    :func:`~autokmc.structure.optimise_structure` (LBFGS).  Before and after
    each relaxation the ASE :class:`~ase.neighborlist.NeighborList` bond
    topology is compared; changes raise a :class:`SiteStabilityError`
    subclass.

    On success the potential energies are stored on *lateral_class* and
    returned as ``(E_occupied, E_unoccupied)``.

    Parameters
    ----------
    G : nx.Graph
        The full surface + adsorbate graph.
    adsorbate_site : AdsorbateSite
        Parent iso-class of the member being checked.
    member_index : int
        Index into ``adsorbate_site.member_node_ids``.
    lateral_class : AdsorbateSiteLateral
        The lateral-interaction class returned by
        :func:`check_adsorbate_site_lateral` for this member.
    calculator
        Any ASE-compatible ML/empirical potential or CalculatorPool.  One
        concrete calculator is acquired for each relaxation; calculator
        instances are never deep-copied.
    frozen_indices : list[int] | None
        Indices into the **slab** portion of the constructed Atoms (0-based,
        same ordering as bulk/surface nodes sorted by their original ASE atom
        ``index``).  Passed to :class:`~ase.constraints.FixAtoms`.
        ``None`` → no frozen atoms.  The slab ordering matches the original
        ASE Atoms object used to build the graph, so the ``frozen_indices``
        from ``atoms.info["frozen_indices"]`` can be passed directly.
    fmax : float
        Force convergence threshold (eV/Å).  Default 0.05.
    max_steps : int
        Maximum LBFGS steps.  Default 200.
    nl_mult : float
        Neighborlist cutoff multiplier for the connectivity stability check.
        Default :data:`~autokmc.core.constants.NL_MULT_DEFAULT`.
    verbose : bool
        Print per-step progress.

    Returns
    -------
    tuple[float, float]
        ``(E_occupied, E_unoccupied)`` in eV.

    Raises
    ------
    IndexError
        *member_index* out of range.
    OptimisationFailedError
        LBFGS did not converge for either the occupied or unoccupied structure.
    SurfaceConnectivityError
        A slab bond changed during either relaxation.
    AdsorbateDissociationError
        An adsorbate bond changed during either relaxation.
    """
    from autokmc.structure import optimise_structure  # local import avoids circular

    if member_index < 0 or member_index >= len(adsorbate_site.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — AdsorbateSite "
            f"iso_class={adsorbate_site.iso_class} has "
            f"{len(adsorbate_site.member_node_ids)} member(s)."
        )

    self_node_ids: frozenset = frozenset(
        nid for nid in adsorbate_site.member_node_ids[member_index]
        if nid in G
    )
    cache_kind = "adsorption"
    cache_key: str | None = None
    cache_graph: nx.Graph | None = None
    cache_parameters = {
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "nl_mult": float(nl_mult),
        "n_shells": int(lateral_class.n_shells),
        "free_energy_enabled": bool(
            free_energy_options is not None
            and getattr(free_energy_options, "enabled", False)
        ),
        "temperature_k": (
            None if free_energy_temperature_k is None
            else float(free_energy_temperature_k)
        ),
        "calculator": calculator_identity(calculator),
    }
    if free_energy_options is not None:
        cache_parameters["free_energy"] = {
            "vibration_displacement": float(free_energy_options.vibration_displacement),
            "vibration_nfree": int(free_energy_options.vibration_nfree),
            "include_ts_vibrations": bool(free_energy_options.include_ts_vibrations),
            "min_frequency_ev": float(free_energy_options.min_frequency_ev),
            "symmetry_tolerance": float(free_energy_options.symmetry_tolerance),
            "default_spin": float(free_energy_options.default_spin),
            "default_geometry": str(free_energy_options.default_geometry),
        }
    if calculation_cache_root is not None:
        try:
            cache_graph = normalise_reaction_graph(
                lateral_class.ego_graph,
                endpoint_node_ids=self_node_ids,
                endpoint_role="site",
            )
            cache_graph.graph["n_shells"] = int(lateral_class.n_shells)
            atoms_occ_init, _, _, _ = _build_stability_atoms(
                G, lateral_class, self_node_ids,
                include_self=True,
                frozen_indices=frozen_indices,
            )
            atoms_unocc_init, _, _, _ = _build_stability_atoms(
                G, lateral_class, self_node_ids,
                include_self=False,
                frozen_indices=frozen_indices,
            )
            cache_inputs = {
                "occupied_initial": atoms_occ_init,
                "unoccupied_initial": atoms_unocc_init,
            }
            cache_identity = {
                "kind": cache_kind,
                "reactant_smiles": adsorbate_site.reactant,
                "iso_class": int(adsorbate_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
            }
            cache_key = calculation_cache_key(
                kind=cache_kind,
                identity=cache_identity,
                parameters=cache_parameters,
                inputs=cache_inputs,
            )
            cached = load_calculation_record(
                calculation_cache_root,
                cache_kind,
                cache_key,
                reaction_graph=cache_graph,
                operation=cache_identity,
                parameters=cache_parameters,
            )
            if cached is not None and apply_cached_states(
                lateral_class,
                cached,
                {
                    "occupied": ("energy_occupied", "atoms_occupied"),
                    "unoccupied": ("energy_unoccupied", "atoms_unoccupied"),
                },
            ):
                if verbose:
                    print(
                        f"  [cache] adsorption iso={adsorbate_site.iso_class} "
                        f"lat={lateral_class.lateral_class}: loaded "
                        "occupied/unoccupied relaxations"
                    )
                return float(lateral_class.energy_occupied), float(lateral_class.energy_unoccupied)
        except Exception as exc:
            _log.debug(
                "check_site_stability: calculation cache lookup failed "
                "(iso=%d lat=%d): %s",
                adsorbate_site.iso_class,
                lateral_class.lateral_class,
                exc,
            )

    def _relax_and_check(include_self: bool) -> tuple:
        state = "occupied" if include_self else "unoccupied"

        # _build_stability_atoms returns n_lat and n_self directly from the
        # node-list lengths so we never have to re-derive them from
        # len(self_node_ids), which can differ if any node id is absent from G
        # at call time (e.g. after _materialise_adsorbate_nodes re-indexes).
        atoms_init, n_slab, n_lat, n_self_actual = _build_stability_atoms(
            G, lateral_class, self_node_ids,
            include_self    = include_self,
            frozen_indices  = frozen_indices,
        )

        n_ads = n_lat + n_self_actual

        # ads_indices covers BOTH the lateral-neighbour adsorbate atoms
        # (indices n_slab … n_slab+n_lat-1) AND the site's own atoms
        # (indices n_slab+n_lat … n_slab+n_ads-1).
        #
        # This means the connectivity check after relaxation enforces:
        #   • The site's bonds to the surface are unchanged           (stability of site)
        #   • The lateral neighbours' bonds to the surface are unchanged (they must
        #     remain coordinated to the same surface atoms as before relaxation)
        #   • Intramolecular bonds of all adsorbates are unchanged   (no dissociation)
        #
        # If a lateral neighbour desorbs, migrates to a different clique, or
        # dissociates during the ML relaxation, the bond-set comparison detects
        # the change and raises AdsorbateDissociationError, marking this lateral
        # class as unstable.  The error message identifies whether the site or a
        # lateral neighbour caused the instability (see _check_connectivity_stable).
        ads_indices: set[int] = set(range(n_slab, n_slab + n_ads))

        bonds_before = _bond_set(atoms_init, nl_mult=nl_mult,
                                 relevant_indices=ads_indices)

        if verbose:
            print(
                f"  [{state}]  atoms={len(atoms_init)}  "
                f"(slab={n_slab}, lat_neighbours={n_lat}, "
                f"site={n_ads - n_lat})  "
                f"bonds_before={len(bonds_before)}"
            )

        with acquire_calculator(
            calculator, purpose=f"adsorption {state} relaxation"
        ) as calc:
            atoms_opt = optimise_structure(
                atoms_init,
                calculator = calc,
                fmax       = fmax,
                steps      = max_steps,
                verbose    = verbose,
            )

            # Convergence guard — optimise_structure issues a RuntimeWarning but
            # we want to raise an actionable error for the stability workflow.
            # Check forces directly on the returned structure.
            forces = atoms_opt.get_forces()
            if frozen_indices:
                free_mask = np.ones(len(atoms_opt), dtype=bool)
                free_mask[list(frozen_indices)] = False
                max_force = float(np.linalg.norm(forces[free_mask], axis=1).max())
            else:
                max_force = float(np.linalg.norm(forces, axis=1).max())

            if max_force > fmax:
                raise OptimisationFailedError(
                    f"[{state}] LBFGS did not converge: "
                    f"max|F| = {max_force:.4f} eV/Å after {max_steps} "
                    f"steps (fmax={fmax} eV/Å)."
                )

            energy = float(atoms_opt.get_potential_energy())
            atoms_opt.set_pbc(atoms_init.get_pbc())

            _check_connectivity_stable(
                atoms_init, atoms_opt, n_slab, n_ads, state, nl_mult,
                relevant_indices=ads_indices,
                n_lat=n_lat,
            )

            # ── Intended-coordination check (occupied state only) ─────────────
            # Verify each self-adsorbate atom is still bonded to its intended
            # surface clique in the relaxed structure.  The bonds_before/after
            # comparison above only catches changes relative to the *initial*
            # placement; if the initial placement already lacks the intended bond
            # (e.g. after Kabsch propagation moved the anchor too far), the
            # bonds_before==bonds_after test passes trivially.  This check uses
            # the clique stored on the graph node as the ground truth.
            if include_self:
                _check_intended_coordination_stable(
                    atoms_opt, G, self_node_ids,
                    n_slab, n_lat, nl_mult,
                )

            if verbose:
                bonds_after = _bond_set(atoms_opt, nl_mult=nl_mult,
                                        relevant_indices=ads_indices)
                print(
                    f"  [{state}]  E={energy:.4f} eV  "
                    f"bonds_after={len(bonds_after)}  "
                    f"max|F|={max_force:.4f} eV/Å  ✓ stable"
                )
            atoms_opt.calc = None

        # Return n_slab, n_lat and n_self alongside the energy and relaxed atoms
        # so the free-energy section can compute vib_idx_occ from the SAME
        # _build_stability_atoms invocation that produced atoms_opt.
        # n_self is len(self_nodes) — the adsorbate atoms actually placed in
        # atoms_opt, NOT len(self_node_ids) which can disagree if any node id
        # was absent from G at call time.
        return energy, atoms_opt, n_slab, n_lat, n_self_actual

    E_occ,   atoms_occ,   _n_slab_occ,   _n_lat_occ,   _n_self_occ   = _relax_and_check(include_self=True)
    E_unocc, atoms_unocc, _n_slab_unocc, _n_lat_unocc, _n_self_unocc = _relax_and_check(include_self=False)

    lateral_class.energy_occupied   = E_occ
    lateral_class.energy_unoccupied = E_unocc
    # Persisted later by autokmc.io.persistence.ReactionWriter as
    # reactions/iso{N}_lat{M}/{occupied,unoccupied}.extxyz.
    lateral_class.atoms_occupied    = atoms_occ
    lateral_class.atoms_unoccupied  = atoms_unocc
    lateral_class.stable            = True

    # ── Optional harmonic thermochemistry on the relaxed states ─────────
    # Vibrate ONLY the reactive species (the site's own atoms).  Frozen
    # slab atoms and frozen lateral-shell adsorbates contribute zero by
    # construction.  The Atoms ordering is [slab | lat_neighbours | self]
    # — the self-adsorbate atoms occupy the tail of the array.
    if (free_energy_options is not None
            and getattr(free_energy_options, "enabled", False)
            and free_energy_temperature_k is not None):
        from autokmc.thermo.free_energy import compute_harmonic_thermo

        # Use the n_slab / n_lat / n_self values captured inside
        # _relax_and_check from the SAME _build_stability_atoms call that
        # produced atoms_occ / atoms_unocc.  n_self is the count of adsorbate
        # atoms ACTUALLY placed in atoms_occ (len(self_nodes)), NOT
        # len(self_node_ids), which can be larger if any node id was absent
        # from G when _build_stability_atoms ran.  Using the actual placed
        # count prevents vib_idx_occ from ever pointing past the end of
        # atoms_occ.
        n_slab_occ   = _n_slab_occ
        n_lat_occ    = _n_lat_occ
        n_self_occ   = _n_self_occ       # atoms actually in atoms_occ tail
        n_slab_unocc = _n_slab_unocc
        n_lat_unocc  = _n_lat_unocc

        # Vibrate ONLY the adsorbate atoms (the tail of atoms_occ).
        # The Atoms ordering is [slab | lat_neighbours | self]; self atoms
        # occupy exactly the last n_self_occ indices.
        vib_idx_occ = list(range(
            n_slab_occ + n_lat_occ,
            n_slab_occ + n_lat_occ + n_self_occ,
        ))
        # Unoccupied state has no reactive species — nothing vibrates,
        # so the harmonic correction is zero by construction.  We still
        # call the helper to populate the bookkeeping fields with zeros
        # so downstream code can rely on them.
        vib_idx_unocc: list[int] = []

        from pathlib import Path as _Path
        cache_dir_root = (
            _Path(vib_cache_root) if vib_cache_root is not None else None
        )
        per_lat_dir = (
            cache_dir_root / f"ads_{smiles_to_dirname(adsorbate_site.reactant)}" /
            f"iso{adsorbate_site.iso_class}_lat{lateral_class.lateral_class}"
            if cache_dir_root is not None else None
        )

        # Strip FixAtoms constraints from the vibration copies.  The
        # constraints were applied inside _build_stability_atoms to freeze
        # bottom-layer slab atoms during ML relaxation; they are NOT needed
        # here because vib_idx_occ already limits displacements to the
        # adsorbate atoms only.  Keeping them can cause ASE's internal
        # constraint-adjustment code (adjust_forces / adjust_positions) to
        # fail with an IndexError when the frozen-atoms index array is applied
        # to force arrays whose leading dimension differs from what was set up
        # during earlier relaxation calls (e.g. different n_ads between calls).
        atoms_occ_vib = atoms_occ.copy()
        atoms_occ_vib.set_constraint([])   # remove all constraints

        _log.debug(
            "check_site_stability: harmonic thermo for OCCUPIED "
            "state (iso=%d, lat=%d) — n_atoms=%d  vib_idx=%s",
            adsorbate_site.iso_class, lateral_class.lateral_class,
            len(atoms_occ_vib), vib_idx_occ,
        )

        occ_thermo = compute_harmonic_thermo(
            atoms_occ_vib, vib_idx_occ,
            energy_ev     = float(E_occ),
            temperature_k = float(free_energy_temperature_k),
            calculator    = calculator,
            options       = free_energy_options,
            cache_dir     = (str(per_lat_dir) if per_lat_dir is not None else None),
            label         = "occupied",
            drop_imaginary= True,
        )

        atoms_unocc_vib = atoms_unocc.copy()
        atoms_unocc_vib.set_constraint([])  # remove all constraints

        unocc_thermo = compute_harmonic_thermo(
            atoms_unocc_vib, vib_idx_unocc,
            energy_ev     = float(E_unocc),
            temperature_k = float(free_energy_temperature_k),
            calculator    = calculator,
            options       = free_energy_options,
            cache_dir     = (str(per_lat_dir) if per_lat_dir is not None else None),
            label         = "unoccupied",
            drop_imaginary= True,
        )

        if occ_thermo is not None:
            lateral_class.g_correction_occupied   = occ_thermo["g_corr_ev"]
            lateral_class.g_occupied              = occ_thermo["g_total_ev"]
            lateral_class.zpe_occupied            = occ_thermo["zpe_ev"]
            lateral_class.entropy_occupied        = occ_thermo["entropy_ev_per_k"]
            lateral_class.frequencies_occupied_ev = occ_thermo["frequencies_ev"]
            lateral_class.imaginary_occupied_ev   = occ_thermo["imaginary_ev"]
            lateral_class.vib_indices_occupied    = occ_thermo["vib_indices"]
        if unocc_thermo is not None:
            lateral_class.g_correction_unoccupied   = unocc_thermo["g_corr_ev"]
            lateral_class.g_unoccupied              = unocc_thermo["g_total_ev"]
            lateral_class.zpe_unoccupied            = unocc_thermo["zpe_ev"]
            lateral_class.entropy_unoccupied        = unocc_thermo["entropy_ev_per_k"]
            lateral_class.frequencies_unoccupied_ev = unocc_thermo["frequencies_ev"]
            lateral_class.imaginary_unoccupied_ev   = unocc_thermo["imaginary_ev"]
            lateral_class.vib_indices_unoccupied    = unocc_thermo["vib_indices"]

    _log.debug(
        "check_site_stability: iso_class=%d member=%d lateral_class=%d "
        "E_occ=%.4f eV  E_unocc=%.4f eV",
        adsorbate_site.iso_class, member_index,
        lateral_class.lateral_class, E_occ, E_unocc,
    )
    if calculation_cache_root is not None and cache_key is not None:
        occupied_props = {
            name: getattr(lateral_class, name, None)
            for name in (
                "g_correction_occupied",
                "g_occupied",
                "zpe_occupied",
                "entropy_occupied",
                "frequencies_occupied_ev",
                "imaginary_occupied_ev",
                "vib_indices_occupied",
            )
        }
        unoccupied_props = {
            name: getattr(lateral_class, name, None)
            for name in (
                "g_correction_unoccupied",
                "g_unoccupied",
                "zpe_unoccupied",
                "entropy_unoccupied",
                "frequencies_unoccupied_ev",
                "imaginary_unoccupied_ev",
                "vib_indices_unoccupied",
            )
        }
        record = make_calculation_record(
            kind=cache_kind,
            cache_key=cache_key,
            operation={
                "label": f"adsorption:{adsorbate_site.reactant}",
                "reactant_smiles": adsorbate_site.reactant,
                "iso_class": int(adsorbate_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
                "temperature_k": cache_parameters["temperature_k"],
            },
            parameters=cache_parameters,
            inputs={
                "reactant_smiles": adsorbate_site.reactant,
                "iso_class": int(adsorbate_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
            },
            states={
                "occupied": state_payload(
                    atoms_occ, energy_ev=E_occ, properties=occupied_props,
                ),
                "unoccupied": state_payload(
                    atoms_unocc, energy_ev=E_unocc, properties=unoccupied_props,
                ),
            },
            reaction_graph=cache_graph,
        )
        write_calculation_record(
            calculation_cache_root, cache_kind, cache_key, record,
        )
    return E_occ, E_unocc
