"""
autokmc.find_diffusion_sites
============================
Enumerate diffusion (hop) site-pairs on top of materialised
:class:`~autokmc.find_adsorbate_sites.AdsorbateSite`'s.

A *diffusion site* is an iso-class of unordered pairs of adsorbate placements
that share

* the **same SMILES** (the molecule that hops), and
* a **surface-graph link** between their bonded surface cliques: the minimum
  hop distance through ``type == "surface"`` edges between any atom of A's
  bonded-clique union and any atom of B's bonded-clique union must be at
  most ``max_hops`` (default :data:`~autokmc.constants.DIFFUSION_MAX_HOPS`
  = 1, i.e. "share a surface atom OR are first-neighbour surface atoms").

Pairs are deduplicated into iso-classes by graph-isomorphism of the union
ego-graph (the surface-only ``n_shells_pair``-shell BFS around the union of
both endpoints' bonded cliques, with the two adsorbate placements stamped on
as labelled leaves).

There is **no ML pruning** at this stage — the underlying
:class:`AdsorbateSite`'s have already been pruned by
:func:`autokmc.find_adsorbate_sites.prune_unstable_adsorbate_sites`, so every
endpoint is known stable.  Lateral-environment classification and the NEB
transition-state search are deferred to
:mod:`autokmc.check_diffusion_sites` and run lazily during the KMC loop —
mirroring the on-the-fly lateral-class workflow of
:mod:`autokmc.check_adsorbate_sites`.

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
    :mod:`autokmc.kmc_simulation`.

Public API
----------
* :class:`DiffusionLateral`         — one lateral-interaction class for a pair.
* :class:`DiffusionSite`            — one iso-class of hop pairs.
* :func:`find_diffusion_sites`      — universal pair enumerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.find_adsorbate_sites import (
    AdsorbateSite,
    _get_surface_apsp,
    _shortest_path_between_cliques,
)
from autokmc.find_anchors import _build_ego_graph
from autokmc.constants import (
    DIFFUSION_MAX_HOPS,
    N_SHELLS_DEFAULT,
    MAX_PAIR_SHELLS,
)
from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiffusionLateral:
    """One lateral-interaction environment of a :class:`DiffusionSite`.

    Mirrors :class:`autokmc.find_adsorbate_sites.AdsorbateSiteLateral` for
    the diffusion case.  Energies and relaxed atoms are filled in lazily by
    :func:`autokmc.check_diffusion_sites.check_diffusion_stability` once the
    NEB has run for the first member to land in this lateral class.

    Attributes
    ----------
    lateral_class : int
        0-based index within the parent :class:`DiffusionSite`'s
        ``lateral_classes`` list.
    ego_graph : nx.Graph | None
        The lateral ego-graph (surface BFS + occupied-adsorbate leaves)
        used for iso-class matching.  See
        :func:`autokmc.check_diffusion_sites.check_diffusion_site_lateral`.
    n_shells : int
        BFS depth used to build :attr:`ego_graph`.
    members : list[int]
        Indices into the parent :class:`DiffusionSite`'s ``members`` whose
        current local environment is isomorphic to this lateral class.
    energy_a, energy_b : float | None
        Potential energies (eV) of the two relaxed endpoints — *A occupied,
        B empty* and *A empty, B occupied* respectively.  Filled by
        :func:`autokmc.check_diffusion_sites.check_diffusion_stability`.
    energy_ts : float | None
        Potential energy (eV) of the highest NEB image (the climbing-image
        saddle when ``climb=True``).
    atoms_a, atoms_b, atoms_ts : Atoms | None
        Relaxed ASE atoms snapshots persisted by
        :class:`autokmc.persistence.ReactionWriter`.
    atoms_neb_path : list[Atoms] | None
        Full NEB band (endpoints + intermediate images) — optional, only
        kept when ``persist_neb_path=True``.
    stable : bool | None
        ``True`` when both endpoint relaxations and the NEB converged
        without changing surface / adsorbate connectivity; ``False`` on
        any stability failure; ``None`` until the check has run.
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
    atoms_neb_path: Any              = None
    stable        : bool | None      = None


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
        :func:`autokmc.check_diffusion_sites.check_diffusion_site_lateral`.
    """
    reactant              : str
    iso_class             : int
    members               : list[tuple] = field(default_factory=list)
    member_node_ids       : list[tuple] = field(default_factory=list)
    ego_graph             : Any = None
    n_shells_pair_settled : int = 0
    lateral_classes       : list[DiffusionLateral] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _member_clique_union(site: AdsorbateSite, m_idx: int) -> frozenset[int]:
    """Union of bonded-surface cliques for ``site.members[m_idx]``.

    Reads from the cached per-member tuple ``site._member_cliques`` populated
    by :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`.
    """
    member_cliques = getattr(site, "_member_cliques", None)
    if member_cliques is None or m_idx >= len(member_cliques):
        return frozenset()
    out: set[int] = set()
    for clq in member_cliques[m_idx]:
        out |= set(clq)
    return frozenset(out)


def _build_pair_ego_graph(
    G: nx.Graph,
    a_node_ids: list[int],
    b_node_ids: list[int],
    a_clique_union: frozenset[int],
    b_clique_union: frozenset[int],
    n_shells_pair: int,
) -> nx.Graph:
    """Build the iso-class ego-graph for a diffusion pair.

    The graph is the surface-only BFS around ``a_clique_union ∪
    b_clique_union`` out to ``n_shells_pair`` hops, with both adsorbate
    placements added as labelled leaves carrying ``element``,
    ``iso_class`` and ``reactant`` from *G*.

    The two endpoints are stamped with a synthetic ``endpoint_role`` value of
    ``"endpoint"`` (rather than e.g. ``"a"`` / ``"b"``) so that the iso-match
    is **symmetric** under swapping A↔B — which is exactly what we want, since
    a hop is intrinsically reversible.
    """
    seed = a_clique_union | b_clique_union
    ego = _build_ego_graph(G, frozenset(seed), n_shells_pair).copy()

    # Add both placements as leaves (with their intramolecular structure
    # so the iso-match captures multi-atom shapes).
    for endpoint_ids in (a_node_ids, b_node_ids):
        for nid in endpoint_ids:
            if nid not in G:
                continue
            d = G.nodes[nid]
            if nid not in ego:
                ego.add_node(
                    nid,
                    element        = d.get("element"),
                    type           = d.get("type", "adsorbate"),
                    iso_class      = int(d.get("iso_class", -1)),
                    reactant       = str(d.get("reactant",  "")),
                    reactant_index = int(d.get("reactant_index", -1)),
                    endpoint_role  = "endpoint",
                )
            else:
                ego.nodes[nid]["endpoint_role"] = "endpoint"
            # Intra-molecular edges between siblings in the placement.
            for sib in d.get("siblings", ()):
                sib = int(sib)
                if sib in ego and not ego.has_edge(nid, sib):
                    ego.add_edge(nid, sib, intra_adsorbate=True)
            # Anchor bonds to surface atoms of the BFS set.
            clq = d.get("clique")
            if clq is not None:
                for surf_id in clq:
                    if surf_id in ego and not ego.has_edge(nid, surf_id):
                        ego.add_edge(nid, surf_id, anchor_bond=True)
    return ego


def _pair_node_match(d1: dict, d2: dict) -> bool:
    """Node-match predicate for diffusion-pair iso classification.

    * ``type == "surface"``   — must share ``element``.
    * ``type == "adsorbate"`` — must share ``element``, ``iso_class``,
      ``reactant`` *and* ``endpoint_role`` (so an endpoint never maps onto
      a third-party occupied adsorbate that happens to share the SMILES).
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
        if d1.get("endpoint_role") != d2.get("endpoint_role"):
            return False
    return True


def _pair_fingerprint(g: nx.Graph) -> tuple:
    """Cheap fingerprint to bucket ego-graphs before the full GraphMatcher."""
    sigs = tuple(sorted(
        (
            d.get("type",    "X"),
            d.get("element", "X"),
            int(d.get("iso_class", -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",  "")) if d.get("type") == "adsorbate" else "",
            d.get("endpoint_role", "") or "",
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (g.number_of_nodes(), g.number_of_edges(), sigs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_diffusion_sites(
    G: nx.Graph,
    adsorbate_sites: Iterable[AdsorbateSite],
    *,
    max_hops: int = DIFFUSION_MAX_HOPS,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    surface_apsp_cutoff: int = MAX_PAIR_SHELLS,
    verbose: bool = False,
) -> dict[str, list[DiffusionSite]]:
    """Enumerate diffusion site-pairs for every SMILES present in *adsorbate_sites*.

    Parameters
    ----------
    G : nx.Graph
        Surface + adsorbate graph (must already contain the materialised
        adsorbate-site nodes from
        :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`).
    adsorbate_sites : iterable of AdsorbateSite
        The (typically stable) adsorbate iso-classes to consider.  Pairs are
        only formed between members that share the same ``site.reactant``
        SMILES.
    max_hops : int
        Maximum surface-graph hop distance allowed between A's and B's
        bonded-clique union.  Default
        :data:`~autokmc.constants.DIFFUSION_MAX_HOPS` (= 1).
    n_shells_pair : int
        Surface-only BFS depth used for iso-class deduplication.
    surface_apsp_cutoff : int
        Cutoff handed to :func:`autokmc.find_adsorbate_sites._get_surface_apsp`
        for the cached APSP table.  Must be ≥ ``max_hops``.
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

    apsp = _get_surface_apsp(
        G, cutoff=max(int(max_hops), int(surface_apsp_cutoff)),
    )

    out: dict[str, list[DiffusionSite]] = {}
    diff_clique_to_members: dict = G.graph.setdefault(
        "diffusion_clique_to_members", {}
    )
    diff_surface_to_members: dict = G.graph.setdefault(
        "diffusion_surface_node_to_members", {}
    )

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

        diffusion_sites: list[DiffusionSite] = []
        # Bucket discovered iso-classes by fingerprint for cheap pre-filter.
        fp_index: dict[tuple, list[DiffusionSite]] = {}

        n_pairs_considered = 0
        n_pairs_kept       = 0

        for i in range(len(flat)):
            site_a, m_a, clq_a = flat[i]
            for j in range(i + 1, len(flat)):
                site_b, m_b, clq_b = flat[j]
                n_pairs_considered += 1

                # Reject pairs whose cliques are too far apart, or that
                # share *all* their surface atoms (same physical site).
                hop = _shortest_path_between_cliques(clq_a, clq_b, apsp)
                if hop > int(max_hops):
                    continue
                if clq_a == clq_b:
                    continue

                a_nids = list(site_a.member_node_ids[m_a])
                b_nids = list(site_b.member_node_ids[m_b])

                ego = _build_pair_ego_graph(
                    G, a_nids, b_nids, clq_a, clq_b, n_shells_pair,
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
                        merged = True
                        break

                if not merged:
                    ds = DiffusionSite(
                        reactant              = smiles,
                        iso_class             = len(diffusion_sites),
                        members               = [(site_a, m_a, site_b, m_b)],
                        member_node_ids       = [(a_nids, b_nids)],
                        ego_graph             = ego,
                        n_shells_pair_settled = int(n_shells_pair),
                    )
                    diffusion_sites.append(ds)
                    fp_index.setdefault(fkey, []).append(ds)

                n_pairs_kept += 1

        # ── Reverse indexes for the KMC incremental update path ──────────
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

    _log.debug(
        "find_diffusion_sites: %d SMILES processed, totals=%s",
        len(out),
        {k: len(v) for k, v in out.items()},
    )
    return out

