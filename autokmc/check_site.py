"""
autokmc.check_site
==================
Lookup of an adsorbate's lateral-interaction subgraph against the
recorded library of optimised placements.

Use case
--------
At KMC runtime the surface is occupied: a placement that was optimised
in isolation now finds itself next to other adsorbates, so the lateral
subgraph of its anchor cliques on the *occupied* snapshot may include
neighbouring molecules (via the chain-following closure in
:func:`autokmc.opt_site.lateral_neighbour_atoms`).

This module answers:

    "Given the lateral subgraph around this placement on the current
    surface, do I already have a recorded site whose lateral subgraph
    is element-isomorphic to it?"

If yes, the recorded site can be reused (its energy / forces / propagation
are valid for this exact local environment).  If no, the local environment
is novel and a fresh optimisation is needed.

Public API
----------
* :func:`find_matching_record` — main entry point; takes either a
  pre-built subgraph or ``(G_snapshot, anchor_nodes)`` and returns the
  first matching :class:`MatchRecord` (or ``None``).
* :func:`iter_recorded_subgraphs` — generator over every recorded
  subgraph in :class:`~autokmc.cache.SiteCache.adsorbate_sites`, yielding
  ``(record, subgraph)``.  Useful if you want to enumerate matches or
  build your own lookup index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal

import networkx as nx
from networkx.algorithms import isomorphism as _nx_iso
from networkx.algorithms.isomorphism import categorical_node_match

from autokmc.cache import get_cache, SiteCache
from autokmc.constants import N_SHELLS_DEFAULT
from autokmc.default_sites import _iso_prefilter_key
from autokmc.logging_utils import get_logger
from autokmc.opt_site import lateral_neighbour_subgraph

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

SubgraphSource = Literal[
    "member_subgraph",          # stable ms.member_subgraphs[member_index]
    "initial_lateral_subgraph", # ms.initial_lateral_subgraph
    "relaxed_lateral_subgraph", # ms.relaxed_lateral_subgraph
]


@dataclass
class MatchRecord:
    """A single matching recorded subgraph."""
    smiles        : str
    iso_class     : int
    member_index  : int | None    # only meaningful for ``member_subgraph`` source
    source        : SubgraphSource
    stable        : bool
    adsorbate_site: Any           # the matching AdsorbateSite object
    subgraph      : nx.Graph      # the recorded subgraph that matched
    node_mapping  : dict[int, int] | None = None  # query -> recorded node ids

    @property
    def multisite(self) -> Any:
        """Backward-compatible alias for older exploratory notebooks."""
        return self.adsorbate_site


# ---------------------------------------------------------------------------
# Enumeration of recorded subgraphs
# ---------------------------------------------------------------------------

def iter_recorded_subgraphs(
    cache: SiteCache,
    *,
    smiles: str | None = None,
    sources: Iterable[SubgraphSource] = (
        "member_subgraph",
        "initial_lateral_subgraph",
        "relaxed_lateral_subgraph",
    ),
    only_stable: bool = False,
) -> Iterator[tuple[MatchRecord, nx.Graph]]:
    """Yield ``(record, subgraph)`` for every recorded lateral subgraph.

    Parameters
    ----------
    cache : SiteCache
        Source of recorded adsorbate sites (``cache.adsorbate_sites``).
    smiles : str, optional
        If given, restrict to ``cache.adsorbate_sites[smiles]``.  Otherwise
        iterate over every SMILES key.
    sources : iterable of {"member_subgraph", "initial_lateral_subgraph",
                          "relaxed_lateral_subgraph"}
        Which attributes to pull subgraphs from.  Default: all three.
    only_stable : bool
        Skip adsorbate sites whose ``stable`` attribute is False.  Default
        False — unstable placements are useful too (they tell you a
        novel-looking subgraph actually corresponds to a *known* unstable
        site, not a brand-new one).
    """
    sources = set(sources)
    smiles_keys = [smiles] if smiles is not None else list(cache.adsorbate_sites.keys())

    for sm in smiles_keys:
        for ms in cache.adsorbate_sites.get(sm, []):
            stable = bool(getattr(ms, "stable", False))
            if only_stable and not stable:
                continue

            if "member_subgraph" in sources:
                mlist = getattr(ms, "member_subgraphs", None) or []
                for k, sg in enumerate(mlist):
                    if sg is None or sg.number_of_nodes() == 0:
                        continue
                    yield (MatchRecord(
                        smiles=sm, iso_class=int(ms.iso_class),
                        member_index=k, source="member_subgraph",
                        stable=stable, adsorbate_site=ms, subgraph=sg,
                    ), sg)

            if "initial_lateral_subgraph" in sources:
                sg = getattr(ms, "initial_lateral_subgraph", None)
                if isinstance(sg, nx.Graph) and sg.number_of_nodes() > 0:
                    yield (MatchRecord(
                        smiles=sm, iso_class=int(ms.iso_class),
                        member_index=None, source="initial_lateral_subgraph",
                        stable=stable, adsorbate_site=ms, subgraph=sg,
                    ), sg)

            if "relaxed_lateral_subgraph" in sources:
                sg = getattr(ms, "relaxed_lateral_subgraph", None)
                if isinstance(sg, nx.Graph) and sg.number_of_nodes() > 0:
                    yield (MatchRecord(
                        smiles=sm, iso_class=int(ms.iso_class),
                        member_index=None, source="relaxed_lateral_subgraph",
                        stable=stable, adsorbate_site=ms, subgraph=sg,
                    ), sg)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _query_subgraph(
    query: nx.Graph | tuple[nx.Graph, Iterable[int]],
    *,
    n_shells: int = N_SHELLS_DEFAULT,
    follow_adsorbate_chains: bool = True,
) -> nx.Graph:
    """Coerce *query* into a stand-alone subgraph.

    Accepts either a ready-built ``nx.Graph`` or a pair
    ``(G, anchor_nodes)`` — the latter is materialised via
    :func:`autokmc.opt_site.lateral_neighbour_subgraph`.
    """
    if isinstance(query, nx.Graph):
        return query
    G, anchor_nodes = query
    return lateral_neighbour_subgraph(
        G, anchor_nodes,
        n_shells=n_shells,
        follow_adsorbate_chains=follow_adsorbate_chains,
        include_anchors=True,
    )


def find_matching_record(
    query: nx.Graph | tuple[nx.Graph, Iterable[int]],
    cache_or_graph: SiteCache | nx.Graph,
    *,
    smiles: str | None = None,
    sources: Iterable[SubgraphSource] = (
        "member_subgraph",
        "initial_lateral_subgraph",
        "relaxed_lateral_subgraph",
    ),
    only_stable: bool = False,
    return_all: bool = False,
    n_shells: int = N_SHELLS_DEFAULT,
    follow_adsorbate_chains: bool = True,
) -> MatchRecord | list[MatchRecord] | None:
    """Look up *query* against the recorded lateral subgraphs.

    Parameters
    ----------
    query : nx.Graph or (G, anchor_nodes)
        Either a pre-built lateral subgraph (e.g.
        ``ms.relaxed_lateral_subgraph`` from a fresh optimisation, or a
        subgraph extracted from an occupied KMC snapshot), or a pair
        ``(G_snapshot, anchor_nodes)`` from which the lateral subgraph
        will be built via :func:`lateral_neighbour_subgraph` (using the
        same chain-following defaults as the optimiser).
    cache_or_graph : SiteCache or nx.Graph
        Either a :class:`SiteCache` directly, or a graph carrying one
        (``get_cache(G)`` will be called).
    smiles : str, optional
        Restrict search to ``cache.adsorbate_sites[smiles]``.
    sources : iterable
        Which subgraph sources to consider.  See :func:`iter_recorded_subgraphs`.
    only_stable : bool
        Skip unstable adsorbate sites (default ``False``).
    return_all : bool
        If True, return *every* matching record (in iteration order)
        instead of stopping at the first.  Default ``False``.
    n_shells, follow_adsorbate_chains : int, bool
        Forwarded to :func:`lateral_neighbour_subgraph` when *query* is
        a ``(G, anchor_nodes)`` pair.

    Returns
    -------
    MatchRecord | list[MatchRecord] | None
        * ``return_all=False`` (default): first match, else ``None``.
        * ``return_all=True``: list of all matches (possibly empty).

    Notes
    -----
    Matching is element-only graph isomorphism (``node_match`` on
    ``element``), with the same two-tier prefilter used elsewhere in
    the package (:func:`autokmc.default_sites._iso_prefilter_key` first,
    full :class:`~networkx.algorithms.isomorphism.GraphMatcher` only on
    survivors).  Edge attributes are ignored.

    A successful match populates ``record.node_mapping`` with the
    *query → recorded* node correspondence reported by ``GraphMatcher``.
    """
    # Resolve cache.
    if isinstance(cache_or_graph, nx.Graph):
        cache = get_cache(cache_or_graph)
    else:
        cache = cache_or_graph

    qg = _query_subgraph(
        query, n_shells=n_shells,
        follow_adsorbate_chains=follow_adsorbate_chains,
    )
    if qg.number_of_nodes() == 0:
        return [] if return_all else None

    try:
        qkey = _iso_prefilter_key(qg)
    except Exception:
        qkey = None

    nm = categorical_node_match("element", "X")
    matches: list[MatchRecord] = []

    for record, sg in iter_recorded_subgraphs(
        cache, smiles=smiles, sources=sources, only_stable=only_stable,
    ):
        if qkey is not None:
            try:
                if _iso_prefilter_key(sg) != qkey:
                    continue
            except Exception:
                continue
        gm = _nx_iso.GraphMatcher(qg, sg, node_match=nm)
        if not gm.is_isomorphic():
            continue
        record.node_mapping = {int(k): int(v) for k, v in gm.mapping.items()}
        if not return_all:
            _log.debug(
                "find_matching_record: matched %s/iso_class=%d source=%s",
                record.smiles, record.iso_class, record.source,
            )
            return record
        matches.append(record)

    return matches if return_all else None


