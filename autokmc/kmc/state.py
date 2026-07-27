"""Graph occupancy and affected-member lookup helpers for KMC."""

from __future__ import annotations

import networkx as nx

from autokmc.core.graph_state import (
    BOND_SURFACE_NODE_TO_MEMBERS,
    DIFFUSION_SURFACE_NODE_TO_MEMBERS,
    OCCUPIED_BY_CLIQUE,
    SURFACE_NODE_TO_MEMBERS,
    get_bond_surface_node_to_members,
    get_clique_to_members,
    get_diffusion_surface_node_to_members,
    get_n_occupied,
    get_occupied_by_clique,
    get_surface_node_to_members,
    set_n_occupied,
)
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.identity import (
    SiteId,
    SiteMemberId,
    member_identifier,
    site_identifier,
)
from autokmc.sites.stability.adsorption import _surface_bfs_shells


def _set_member_occupied(
    G: nx.Graph,
    site: AdsorbateSite,
    member_index: int,
    value: bool,
) -> None:
    """Toggle one adsorbate member and update graph-level occupancy indexes."""
    occupied_by_clique = (
        get_occupied_by_clique(G)
        if OCCUPIED_BY_CLIQUE in G.graph
        else None
    )
    if not hasattr(site, "_n_occupied"):
        site._n_occupied = 0

    new_state = bool(value)
    was_occupied = any(
        nid in G and G.nodes[nid].get("occupied", False)
        for nid in site.member_node_ids[member_index]
    )

    for nid in site.member_node_ids[member_index]:
        if nid not in G:
            continue
        G.nodes[nid]["occupied"] = new_state
        if occupied_by_clique is not None:
            clique = G.nodes[nid].get("clique")
            if clique is not None:
                bucket = occupied_by_clique.setdefault(clique, set())
                if new_state:
                    bucket.add(nid)
                else:
                    bucket.discard(nid)

    if was_occupied != new_state:
        delta = 1 if new_state else -1
        site._n_occupied = max(0, int(site._n_occupied) + delta)
        set_n_occupied(G, get_n_occupied(G) + delta)


def _affected_surface_cliques(
    G: nx.Graph,
    site: AdsorbateSite,
    member_index: int,
) -> set:
    """Return surface cliques bonded by a member."""
    member_cliques = getattr(site, "_member_cliques", None)
    if member_cliques is not None:
        return set(member_cliques[member_index])

    out: set = set()
    for nid in site.member_node_ids[member_index]:
        if nid not in G:
            continue
        clique = G.nodes[nid].get("clique")
        if clique is not None:
            out.add(clique)
    return out


def _affected_members_for_cliques(
    G: nx.Graph,
    affected_cliques: set,
) -> list[tuple[AdsorbateSite, int]]:
    """Return unique adsorbate ``(site, member)`` pairs touching cliques."""
    clique_to_members = get_clique_to_members(G)
    if not clique_to_members or not affected_cliques:
        return []
    seen: set[SiteMemberId] = set()
    out: list[tuple[AdsorbateSite, int]] = []
    for clique in affected_cliques:
        for site, m_idx in clique_to_members.get(clique, ()):
            key = member_identifier(site, m_idx)
            if key in seen:
                continue
            seen.add(key)
            out.append((site, m_idx))
    return out


def _members_from_surface_index(
    G: nx.Graph,
    surface_node_to_members: dict,
    affected_cliques: set,
    active_site_ids: set[SiteId] | None,
    max_n_shells: int,
) -> list:
    if not surface_node_to_members or not affected_cliques:
        return []

    seed = frozenset(surface for clique in affected_cliques for surface in clique)
    expanded: frozenset = _surface_bfs_shells(G, seed, max_n_shells)

    seen: set[SiteMemberId] = set()
    out = []
    for surface_id in expanded:
        for site, m_idx in surface_node_to_members.get(surface_id, ()):
            if (
                active_site_ids is not None
                and site_identifier(site) not in active_site_ids
            ):
                continue
            key = member_identifier(site, m_idx)
            if key in seen:
                continue
            seen.add(key)
            out.append((site, m_idx))
    return out


def _lateral_shell_members(
    G: nx.Graph,
    affected_cliques: set,
    active_site_ids: set[SiteId] | None,
    max_n_shells: int,
) -> list[tuple[AdsorbateSite, int]] | None:
    """Return adsorbate members whose lateral environment may have changed."""
    if SURFACE_NODE_TO_MEMBERS not in G.graph:
        return None
    return _members_from_surface_index(
        G,
        get_surface_node_to_members(G),
        affected_cliques,
        active_site_ids,
        max_n_shells,
    )


def _diffusion_lateral_shell_members(
    G: nx.Graph,
    affected_cliques: set,
    active_diffusion_ids: set[SiteId] | None,
    max_n_shells: int,
) -> list[tuple[DiffusionSite, int]] | None:
    """Diffusion analogue of :func:`_lateral_shell_members`."""
    if DIFFUSION_SURFACE_NODE_TO_MEMBERS not in G.graph:
        return None
    return _members_from_surface_index(
        G,
        get_diffusion_surface_node_to_members(G),
        affected_cliques,
        active_diffusion_ids,
        max_n_shells,
    )


def _bond_lateral_shell_members(
    G: nx.Graph,
    affected_cliques: set,
    active_bond_ids: set[SiteId] | None,
    max_n_shells: int,
) -> list[tuple[BondReactionSite, int]] | None:
    """Bond-reaction analogue of :func:`_lateral_shell_members`."""
    if BOND_SURFACE_NODE_TO_MEMBERS not in G.graph:
        return None
    return _members_from_surface_index(
        G,
        get_bond_surface_node_to_members(G),
        affected_cliques,
        active_bond_ids,
        max_n_shells,
    )


__all__ = [
    "_set_member_occupied",
    "_affected_surface_cliques",
    "_affected_members_for_cliques",
    "_lateral_shell_members",
    "_diffusion_lateral_shell_members",
    "_bond_lateral_shell_members",
]
