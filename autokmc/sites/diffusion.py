"""
autokmc.sites.diffusion
============================
Enumerate diffusion (hop) site-pairs on top of materialised
:class:`~autokmc.sites.adsorbate.AdsorbateSite`'s.

A *diffusion site* is an iso-class of unordered pairs of adsorbate placements
that share

* the **same SMILES** (the molecule that hops), and
* a **surface-graph link** between their bonded surface cliques: the minimum
  hop distance through ``type == "surface"`` edges between any atom of A's
  bonded-clique union and any atom of B's bonded-clique union must be at
  most ``max_hops`` (default :data:`~autokmc.core.constants.DIFFUSION_MAX_HOPS`
  = 0, i.e. "share at least one surface atom").

Pairs are deduplicated into iso-classes by graph-isomorphism of the union
ego-graph (the surface-only ``n_shells_pair``-shell BFS around the union of
both endpoints' bonded cliques, with the two adsorbate placements stamped on
as labelled occupied leaves — consistent with the adsorption lateral
ego-graph convention).

There is **no ML pruning** at this stage — the underlying
:class:`AdsorbateSite`'s have already been pruned by
:func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`, so every
endpoint is known stable.  Lateral-environment classification and the NEB
transition-state search are deferred to
:mod:`autokmc.sites.stability.diffusion` and run lazily during the KMC loop —
mirroring the on-the-fly lateral-class workflow of
:mod:`autokmc.sites.stability.adsorption`.

Storage on the graph
--------------------
``G.graph["diffusion_sites"][smiles]``
    The list of :class:`DiffusionSite` returned by
    :func:`find_diffusion_sites`.

``G.graph["diffusion_clique_to_members"]``
    ``dict[frozenset, list[(DiffusionSite, member_index)]]`` — for the KMC
    incremental update path.  Both endpoints' bonded cliques are registered
    so that toggling either side reaches every diffusion event touching it.

``G.graph["diffusion_surface_node_to_members"]``
    ``dict[int, list[(DiffusionSite, member_index)]]`` — surface-atom →
    diffusion members mapping for the lateral-shell expansion in
    :mod:`autokmc.kmc.engine`.

Public API
----------
* :class:`DiffusionLateral`         — one lateral-interaction class for a pair.
* :class:`DiffusionSite`            — one iso-class of hop pairs.
* :func:`find_diffusion_sites`      — universal pair enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.sites.adsorbate import (
    AdsorbateSite,
)
from autokmc.sites.identity import SiteId, site_identifier
from autokmc.sites.stability.adsorption import _surface_bfs_shells
from autokmc.core.constants import (
    DIFFUSION_MAX_HOPS,
    DIFFUSION_PRUNE_BY_ADS_PAIR,
    N_SHELLS_DEFAULT,
)
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiffusionLateral:
    """One lateral-interaction environment of a :class:`DiffusionSite`.

    Mirrors :class:`autokmc.sites.adsorbate.AdsorbateSiteLateral` for
    the diffusion case.  Energies and relaxed atoms are filled in lazily by
    :func:`autokmc.sites.stability.diffusion.check_diffusion_stability` once the
    NEB has run for the first member to land in this lateral class.

    Attributes
    ----------
    lateral_class : int
        0-based index within the parent :class:`DiffusionSite`'s
        ``lateral_classes`` list.
    ego_graph : nx.Graph | None
        The lateral ego-graph (surface BFS + occupied-adsorbate leaves)
        used for iso-class matching.  See
        :func:`autokmc.sites.stability.diffusion.check_diffusion_site_lateral`.
    n_shells : int
        BFS depth used to build :attr:`ego_graph`.
    members : list[int]
        Indices into the parent :class:`DiffusionSite`'s ``members`` whose
        current local environment is isomorphic to this lateral class.
    energy_a, energy_b : float | None
        Potential energies (eV) of the two relaxed endpoints — *A occupied,
        B empty* and *A empty, B occupied* respectively.  Filled by
        :func:`autokmc.sites.stability.diffusion.check_diffusion_stability`.
    energy_ts : float | None
        Potential energy (eV) of the highest NEB image (the climbing-image
        saddle when ``climb=True``).
    atoms_a, atoms_b, atoms_ts : Atoms | None
        Relaxed ASE atoms snapshots persisted by
        :class:`autokmc.io.persistence.ReactionWriter`.
    atoms_a_initial, atoms_b_initial : Atoms | None
        Pre-optimization endpoint structures supplied to the relaxations.
    atoms_neb_path_initial : list[Atoms] | None
        Interpolated NEB band before any NEB optimization.
    atoms_neb_path : list[Atoms] | None
        Full NEB band (endpoints + intermediate images) — optional, only
        kept when ``persist_neb_path=True``.
    stable : bool | None
        ``True`` when both endpoint relaxations and the NEB converged
        without changing surface / adsorbate connectivity; ``False`` on
        any stability failure; ``None`` until the check has run.
    invalid_reason : str | None
        Human-readable explanation of why this lateral class is invalid
        (set when ``stable=False``).  ``None`` when ``stable`` is ``True``
        or not yet evaluated.
    last_failure_reason : str | None
        Most recent retryable numerical failure.  This is diagnostic state;
        :attr:`stable` remains ``None`` so a later evaluation can retry it.
    """
    lateral_class : int
    ego_graph     : Any              = None
    n_shells      : int              = 0
    members       : list[int]        = field(default_factory=list)
    energy_a      : float | None     = None
    energy_b      : float | None     = None
    energy_ts     : float | None     = None
    atoms_a       : Any              = None
    atoms_b       : Any              = None
    atoms_ts      : Any              = None
    atoms_a_initial: Any             = None
    atoms_b_initial: Any             = None
    atoms_neb_path_initial: Any      = None
    atoms_neb_path: Any              = None
    neb_path_energies: list[float] | None = None
    neb_n_images: int | None       = None
    neb_n_frames: int | None       = None
    neb_max_endpoint_displacement: float | None = None
    neb_target_image_spacing: float | None = None
    neb_estimated_image_spacing: float | None = None
    neb_image_count_limited_by: str | None = None
    neb_convergence_mode: str | None = None
    neb_convergence_fmax: float | None = None
    neb_converged_low_barrier: bool | None = None
    neb_low_barrier_fmax: float | None = None
    neb_low_barrier_threshold: float | None = None
    neb_low_barrier_stage: str | None = None
    stable        : bool | None      = None
    invalid_reason: str | None       = None
    last_failure_reason: str | None  = None
    # The free-energy module populates these vibrational fields.
    g_correction_a   : float | None = None
    g_correction_b   : float | None = None
    g_correction_ts  : float | None = None
    g_a              : float | None = None
    g_b              : float | None = None
    g_ts             : float | None = None
    zpe_a            : float | None = None
    zpe_b            : float | None = None
    zpe_ts           : float | None = None
    entropy_a        : float | None = None
    entropy_b        : float | None = None
    entropy_ts       : float | None = None
    frequencies_a_ev : list = field(default_factory=list)
    frequencies_b_ev : list = field(default_factory=list)
    frequencies_ts_ev: list = field(default_factory=list)
    imaginary_a_ev   : list = field(default_factory=list)
    imaginary_b_ev   : list = field(default_factory=list)
    imaginary_ts_ev  : list = field(default_factory=list)
    vib_indices_a    : list = field(default_factory=list)
    vib_indices_b    : list = field(default_factory=list)
    vib_indices_ts   : list = field(default_factory=list)
    if TYPE_CHECKING:
        _fingerprint : tuple = field(init=False, repr=False, compare=False)
        _rate_cache : dict = field(init=False, repr=False, compare=False)


@dataclass
class DiffusionSite:
    """One isomorphism class of adsorbate hop pairs.

    Each *member* is a concrete unordered pair of (parent
    :class:`AdsorbateSite`, member index) handles describing the two
    physical placements involved in the hop.

    Attributes
    ----------
    reactant : str
        SMILES of the migrating molecule.  Both endpoints share this SMILES.
    iso_class : int
        0-based index in the per-SMILES list of diffusion iso-classes.
    members : list[tuple[AdsorbateSite, int, AdsorbateSite, int]]
        ``(site_a, member_index_a, site_b, member_index_b)`` for every
        physical pair folded into this iso-class.  The first entry is the
        iso-class **representative**.  Endpoint A is canonically the smaller
        of the two ``(iso_class, member_index)`` keys.
    member_node_ids : list[tuple[list[int], list[int]]]
        ``(A_node_ids, B_node_ids)`` per member — convenience handles into
        the live graph.
    ego_graph : nx.Graph | None
        Surface-only ``n_shells_pair_settled``-shell ego-graph around the
        union of both endpoints' bonded cliques (with the two placements
        included as labelled adsorbate leaves).  Used for iso-class
        matching by :func:`find_diffusion_sites`.
    n_shells_pair_settled : int
        Ego depth at which this iso-class was matched.
    lateral_classes : list[DiffusionLateral]
        Lazily-populated lateral-interaction classes — see
        :func:`autokmc.sites.stability.diffusion.check_diffusion_site_lateral`.
    """
    reactant              : str
    iso_class             : int
    members               : list[tuple] = field(default_factory=list)
    member_node_ids       : list[tuple] = field(default_factory=list)
    ego_graph             : Any = None
    n_shells_pair_settled : int = 0
    lateral_classes       : list[DiffusionLateral] = field(default_factory=list)
    #: Stable KMC identity, assigned lazily once member nodes are available.
    site_id                : str = field(default="", compare=False)
    # Lazily attached so older checkpoints and manual instances retain the
    # established ``hasattr``-based initialisation path.
    if TYPE_CHECKING:
        _lateral_fp_index : dict[tuple, list[DiffusionLateral]] = field(
            init=False, repr=False, compare=False,
        )
        _member_lc : dict[int, DiffusionLateral] = field(
            init=False, repr=False, compare=False,
        )
        applicable_reactions : list[Any] = field(
            init=False, repr=False, compare=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _member_clique_union(site: AdsorbateSite, m_idx: int) -> frozenset[int]:
    """Union of bonded-surface cliques for ``site.members[m_idx]``.

    Reads from the cached per-member tuple ``site._member_cliques`` populated
    by :func:`~autokmc.sites.adsorbate.find_adsorbate_sites`.
    """
    member_cliques = getattr(site, "_member_cliques", None)
    if member_cliques is None or m_idx >= len(member_cliques):
        return frozenset()
    out: set[int] = set()
    for clq in member_cliques[m_idx]:
        out |= set(clq)
    return frozenset(out)


def _surface_node_index_for_placements(
    flat: list[tuple[AdsorbateSite, int, frozenset[int]]],
) -> dict[int, list[int]]:
    """Map surface atom id to placement indexes touching that atom."""
    index: dict[int, list[int]] = {}
    for i, (_, _, clq_union) in enumerate(flat):
        for surf_id in clq_union:
            index.setdefault(int(surf_id), []).append(i)
    return index


def _nearby_placement_indices(
    G: nx.Graph,
    surface_index: dict[int, list[int]],
    seed_clique: frozenset[int],
    max_hops: int,
) -> list[int]:
    """Return placement indexes whose clique intersects the local surface shell."""
    if not seed_clique:
        return []
    shell = _surface_bfs_shells(G, seed_clique, max(0, int(max_hops)))
    seen: set[int] = set()
    for surf_id in shell:
        for idx in surface_index.get(int(surf_id), ()):
            seen.add(int(idx))
    return sorted(seen)


def rebuild_diffusion_reverse_indexes(
    G: nx.Graph,
    diffusion_sites_by_smiles: dict[str, list[DiffusionSite]] | None = None,
) -> None:
    """Rebuild diffusion reverse indexes from the active diffusion site store."""
    sites_by_smiles = (
        diffusion_sites_by_smiles
        if diffusion_sites_by_smiles is not None
        else (G.graph.get("diffusion_sites", {}) or {})
    )
    diff_clique_to_members: dict = {}
    diff_surface_to_members: dict = {}

    for diffusion_sites in sites_by_smiles.values():
        for ds in diffusion_sites:
            for m_idx, _ in enumerate(ds.member_node_ids):
                a_site, a_m, b_site, b_m = ds.members[m_idx]
                a_member_cliques = getattr(a_site, "_member_cliques", None)
                b_member_cliques = getattr(b_site, "_member_cliques", None)
                cliques: list[frozenset] = []
                if a_member_cliques is not None and a_m < len(a_member_cliques):
                    cliques.extend(a_member_cliques[a_m])
                if b_member_cliques is not None and b_m < len(b_member_cliques):
                    cliques.extend(b_member_cliques[b_m])
                for clq in cliques:
                    diff_clique_to_members.setdefault(clq, []).append(
                        (ds, m_idx)
                    )
                    for surf_id in clq:
                        diff_surface_to_members.setdefault(
                            int(surf_id), [],
                        ).append((ds, m_idx))

    G.graph["diffusion_clique_to_members"] = diff_clique_to_members
    G.graph["diffusion_surface_node_to_members"] = diff_surface_to_members


def _build_pair_ego_graph(
    G: nx.Graph,
    a_node_ids: list[int],
    b_node_ids: list[int],
    a_clique_union: frozenset[int],
    b_clique_union: frozenset[int],
    n_shells_a: int,
    n_shells_b: int,
) -> nx.Graph:
    """Build the iso-class ego-graph for a diffusion pair.

    Uses **surface-only BFS** — consistent with the adsorption lateral
    ego-graph (:func:`autokmc.sites.stability.adsorption._build_lateral_ego_graph`).
    Two cached surface-only BFS expansions are run (one per endpoint's
    bonded-clique union, at that endpoint's own iso-class shell depth) and
    their results are unioned.

    After the BFS, any *other* occupied adsorbate nodes adjacent to the
    surface set are collected as leaves (but not traversed further).

    Both endpoint placements are then added as labelled occupied leaves with
    ``endpoint_role="endpoint"`` so the iso-match treats A↔B symmetrically
    and endpoints cannot be confused with third-party adsorbate neighbours.

    G is **not mutated** — no temporary occupancy changes are made.
    """
    all_endpoint_nids = [nid for nid in (*a_node_ids, *b_node_ids) if nid in G]
    endpoint_ids: frozenset = frozenset(all_endpoint_nids)

    # 1. Surface-only BFS (cached) from each endpoint's bonded-clique union.
    visited_a = _surface_bfs_shells(G, a_clique_union, n_shells_a)
    visited_b = _surface_bfs_shells(G, b_clique_union, n_shells_b)
    visited: set = (set(visited_a) | set(visited_b)) - endpoint_ids

    # 2. Collect *other* occupied adsorbate leaves adjacent to the BFS set;
    #    endpoints are added explicitly afterwards with their role label.
    ads_leaves: set = set()
    for n in visited:
        for nb in G.neighbors(n):
            if nb in visited or nb in endpoint_ids:
                continue
            d = G.nodes[nb]
            if d.get("type") != "adsorbate":
                continue
            if d.get("occupied", False):
                ads_leaves.add(nb)

    result = G.subgraph(visited | ads_leaves).copy()

    # 3. Add endpoint nodes as labelled occupied leaves with endpoint_role.
    for nid in all_endpoint_nids:
        if nid not in G:
            continue
        d = G.nodes[nid]
        if nid not in result:
            result.add_node(
                nid,
                element        = d.get("element"),
                type           = d.get("type", "adsorbate"),
                iso_class      = int(d.get("iso_class", -1)),
                reactant       = str(d.get("reactant",  "")),
                reactant_index = int(d.get("reactant_index", -1)),
                occupied       = True,
                endpoint_role  = "endpoint",
            )
        else:
            result.nodes[nid]["occupied"]      = True
            result.nodes[nid]["endpoint_role"] = "endpoint"
        # Restore intramolecular edges.
        for sib in d.get("siblings", ()):
            sib = int(sib)
            if sib in result and not result.has_edge(nid, sib):
                result.add_edge(nid, sib, intra_adsorbate=True)
        # Restore anchor bonds to bonded surface atoms.
        clq = d.get("clique")
        if clq is not None:
            for surf_id in clq:
                if surf_id in result and not result.has_edge(nid, surf_id):
                    result.add_edge(nid, surf_id, anchor_bond=True)

    return result


def _pair_node_match(d1: dict, d2: dict) -> bool:
    """Node-match predicate for diffusion-pair iso classification.

    * ``type == "surface"``   — must share ``element``.
    * ``type == "adsorbate"`` — must share ``element``, ``iso_class``,
      ``reactant``, ``reactant_index`` *and* ``endpoint_role`` (so an
      endpoint never maps onto a third-party occupied adsorbate that happens
      to share the SMILES, and symmetry-inequivalent atoms of the same element
      within a multi-atom adsorbate are not interchanged).
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
        if d1.get("reactant_index") != d2.get("reactant_index"):
            return False
        if d1.get("endpoint_role") != d2.get("endpoint_role"):
            return False
    return True


def _pair_fingerprint(g: nx.Graph) -> tuple:
    """Cheap fingerprint to bucket ego-graphs before the full GraphMatcher.

    Mirrors :func:`_pair_node_match`: ``endpoint_role`` is only included
    for adsorbate nodes (it is unset on surface nodes and would otherwise
    inject a constant signature field).
    """
    sigs = tuple(sorted(
        (
            d.get("type",    "X"),
            d.get("element", "X"),
            int(d.get("iso_class",      -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",       "")) if d.get("type") == "adsorbate" else "",
            int(d.get("reactant_index", -1)) if d.get("type") == "adsorbate" else -1,
            (d.get("endpoint_role", "") or "") if d.get("type") == "adsorbate" else "",
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (g.number_of_nodes(), g.number_of_edges(), sigs)



def _prune_one_per_adsorption_pair(
    diffusion_sites: list[DiffusionSite],
    *,
    verbose: bool = False,
    smiles: str = "",
) -> list[DiffusionSite]:
    """Keep the single smallest-ego :class:`DiffusionSite` per adsorption-pair.

    For every unordered pair of adsorption iso-classes ``(ads_iso_a,
    ads_iso_b)`` that appears in *diffusion_sites*, retains only the
    :class:`DiffusionSite` whose ego-graph has the fewest nodes + edges
    (i.e. the most direct / geometrically closest hop path between that pair
    of site types).  All other iso-classes for the same adsorption pair are
    discarded.

    Parameters
    ----------
    diffusion_sites :
        The full list produced by the graph-isomorphism deduplication loop.
    verbose :
        Print one line per pruned pair when ``True``.
    smiles :
        SMILES string for the verbose prefix.

    Returns
    -------
    list[DiffusionSite]
        Surviving sites in their original insertion order.
    """
    # Group by unordered adsorption-iso-class pair (representative = first member).
    groups: dict[tuple[int, int], list[DiffusionSite]] = {}
    for ds in diffusion_sites:
        if not ds.members:
            continue
        site_a, _, site_b, _ = ds.members[0]
        ic_a = int(site_a.iso_class)
        ic_b = int(site_b.iso_class)
        pair_key = (min(ic_a, ic_b), max(ic_a, ic_b))
        groups.setdefault(pair_key, []).append(ds)

    def _ego_size(ds: DiffusionSite) -> int:
        g = ds.ego_graph
        if g is None:
            return 0
        return g.number_of_nodes() + g.number_of_edges()

    kept_ids: set[SiteId] = set()
    for pair_key, candidates in groups.items():
        best = min(candidates, key=_ego_size)
        kept_ids.add(site_identifier(best))
        if verbose and len(candidates) > 1:
            discarded = [c for c in candidates if c is not best]
            print(
                f"  prune_one_per_adsorption_pair[{smiles!r}]: "
                f"pair {pair_key}: kept iso {best.iso_class} "
                f"(ego size {_ego_size(best)}), "
                f"discarded iso classes {[c.iso_class for c in discarded]}"
            )

    # Preserve original insertion order of surviving sites.
    return [ds for ds in diffusion_sites if site_identifier(ds) in kept_ids]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_diffusion_sites(
    G: nx.Graph,
    adsorbate_sites: Iterable[AdsorbateSite],
    *,
    max_hops: int = DIFFUSION_MAX_HOPS,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    prune_by_adsorption_pair: bool = DIFFUSION_PRUNE_BY_ADS_PAIR,
    verbose: bool = False,
) -> dict[str, list[DiffusionSite]]:
    """Enumerate diffusion site-pairs for every SMILES present in *adsorbate_sites*.

    Parameters
    ----------
    G : nx.Graph
        Surface + adsorbate graph (must already contain the materialised
        adsorbate-site nodes from
        :func:`~autokmc.sites.adsorbate.find_adsorbate_sites`).
    adsorbate_sites : iterable of AdsorbateSite
        The (typically stable) adsorbate iso-classes to consider.  Pairs are
        only formed between members that share the same ``site.reactant``
        SMILES.
    max_hops : int
        Maximum surface-graph hop distance allowed between A's and B's
        bonded-clique union.  Default
        :data:`~autokmc.core.constants.DIFFUSION_MAX_HOPS` (= 1).
    n_shells_pair : int
        Surface-only BFS depth used for iso-class deduplication.
    prune_by_adsorption_pair : bool
        When ``True`` (default), for every unordered pair of adsorption
        iso-classes keep only the single :class:`DiffusionSite` whose
        ego-graph is smallest (fewest nodes + edges), i.e. the most direct
        hop path.  Set to ``False`` to retain all crystallographically-
        distinct hop directions between the same pair of site types.
    verbose : bool

    Returns
    -------
    dict[str, list[DiffusionSite]]
        ``{smiles: [DiffusionSite, …]}``.  Also stored on
        ``G.graph["diffusion_sites"]``.  Reverse indexes
        ``G.graph["diffusion_clique_to_members"]`` and
        ``G.graph["diffusion_surface_node_to_members"]`` are populated for
        the KMC incremental-update path.
    """
    sites_by_smiles: dict[str, list[AdsorbateSite]] = {}
    for s in adsorbate_sites:
        sites_by_smiles.setdefault(s.reactant, []).append(s)

    out: dict[str, list[DiffusionSite]] = {}

    for smiles, sites in sites_by_smiles.items():
        # Flatten members to a single canonical-ordered index list so the
        # i < j enumeration below avoids double-counting (and self-pairs
        # at i == j are excluded by the strict <).
        flat: list[tuple[AdsorbateSite, int, frozenset[int]]] = []
        for site in sites:
            for m_idx in range(len(site.member_node_ids)):
                clq_union = _member_clique_union(site, m_idx)
                if not clq_union:
                    continue
                flat.append((site, m_idx, clq_union))

        # Stable canonical key: (iso_class, m_idx) — well-defined across
        # different AdsorbateSite objects (every site has a unique iso_class).
        flat.sort(key=lambda t: (int(t[0].iso_class), int(t[1])))
        surface_index = _surface_node_index_for_placements(flat)

        diffusion_sites: list[DiffusionSite] = []
        # Bucket discovered iso-classes by fingerprint for cheap pre-filter.
        fp_index: dict[tuple, list[DiffusionSite]] = {}

        n_pairs_considered = 0
        n_pairs_kept       = 0

        for i in range(len(flat)):
            site_a, m_a, clq_a = flat[i]
            for j in _nearby_placement_indices(
                G, surface_index, clq_a, int(max_hops),
            ):
                if j <= i:
                    continue
                site_b, m_b, clq_b = flat[j]
                n_pairs_considered += 1

                # The bounded shell lookup already enforces the hop limit.
                if clq_a == clq_b:
                    continue

                a_nids = list(site_a.member_node_ids[m_a])
                b_nids = list(site_b.member_node_ids[m_b])

                # Per-endpoint BFS depth — at least the depth used to define
                # the adsorbate iso-class, lower-bounded by n_shells_pair.
                ns_a = max(int(getattr(site_a, "n_shells_settled", 0) or 0),
                           int(n_shells_pair))
                ns_b = max(int(getattr(site_b, "n_shells_settled", 0) or 0),
                           int(n_shells_pair))

                ego = _build_pair_ego_graph(
                    G, a_nids, b_nids, clq_a, clq_b, ns_a, ns_b,
                )
                fkey = _pair_fingerprint(ego)

                # Try to merge into an existing iso-class.
                merged = False
                for ds in fp_index.get(fkey, ()):
                    if ds.ego_graph is None:
                        continue
                    gm = isomorphism.GraphMatcher(
                        ego, ds.ego_graph, node_match=_pair_node_match,
                    )
                    if gm.is_isomorphic():
                        ds.members.append((site_a, m_a, site_b, m_b))
                        ds.member_node_ids.append((a_nids, b_nids))
                        # Keep the maximum settled shell depth so that lateral
                        # classification always uses the deepest ego seen so far.
                        ds.n_shells_pair_settled = max(
                            ds.n_shells_pair_settled, max(ns_a, ns_b)
                        )
                        merged = True
                        break

                if not merged:
                    ds = DiffusionSite(
                        reactant              = smiles,
                        iso_class             = len(diffusion_sites),
                        members               = [(site_a, m_a, site_b, m_b)],
                        member_node_ids       = [(a_nids, b_nids)],
                        ego_graph             = ego,
                        n_shells_pair_settled = max(ns_a, ns_b),
                    )
                    diffusion_sites.append(ds)
                    fp_index.setdefault(fkey, []).append(ds)

                n_pairs_kept += 1

        # If requested, keep one diffusion site for each adsorption pair.
        if prune_by_adsorption_pair:
            diffusion_sites = _prune_one_per_adsorption_pair(
                diffusion_sites, verbose=verbose, smiles=smiles,
            )

        # Finally, renumber the remaining iso-classes in sequence. Different
        # crystallographic directions and hop distances can have different
        # barriers even when they connect the same adsorption-site types. The
        # graph-isomorphism check above removes only true equivalents.
        for new_idx, ds in enumerate(diffusion_sites):
            ds.iso_class = new_idx

        if verbose:
            print(
                f"  {len(diffusion_sites)} diffusion iso-class(es) found "
                f"({n_pairs_kept} pair(s) kept from {n_pairs_considered} considered)"
            )

        if verbose:
            n_members = sum(len(ds.members) for ds in diffusion_sites)
            print(
                f"find_diffusion_sites[{smiles!r}]: "
                f"{n_pairs_considered} candidate pair(s) considered, "
                f"{n_pairs_kept} kept (max_hops={max_hops}) → "
                f"{len(diffusion_sites)} iso-class(es), "
                f"{n_members} placement(s)"
            )

        out[smiles] = diffusion_sites

    G.graph["diffusion_sites"] = out
    rebuild_diffusion_reverse_indexes(G, out)

    _log.debug(
        "find_diffusion_sites: %d SMILES processed, totals=%s",
        len(out),
        {k: len(v) for k, v in out.items()},
    )
    return out
