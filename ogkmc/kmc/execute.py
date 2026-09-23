"""Reaction execution and graph mutation."""

from __future__ import annotations

import networkx as nx

from ogkmc.core.graph_state import (
    OCCUPIED_BY_CLIQUE,
    get_occupied_by_clique,
)
from ogkmc.kmc.state import _affected_surface_cliques, _set_member_occupied
from ogkmc.sites.bond import BondReactionSite
from ogkmc.sites.diffusion import DiffusionSite


def _member_occupied(G: nx.Graph, site, member_index: int) -> bool:
    return any(
        nid in G and G.nodes[nid].get("occupied", False)
        for nid in site.member_node_ids[member_index]
    )


def _member_cliques(G: nx.Graph, site, member_index: int) -> tuple[frozenset, ...]:
    cached = getattr(site, "_member_cliques", None)
    if cached is not None and member_index < len(cached):
        return tuple(cached[member_index])

    out: list[frozenset] = []
    for nid in site.member_node_ids[member_index]:
        if nid not in G:
            continue
        clique = G.nodes[nid].get("clique")
        if clique is not None:
            out.append(clique)
    return tuple(out)


def _member_blocked_by_other(G: nx.Graph, site, member_index: int) -> bool:
    cliques = _member_cliques(G, site, member_index)
    if not cliques:
        return False

    member_ids = frozenset(site.member_node_ids[member_index])
    occupied_by_clique = (
        get_occupied_by_clique(G)
        if OCCUPIED_BY_CLIQUE in G.graph
        else None
    )
    if occupied_by_clique is not None:
        for clique in cliques:
            occupied = occupied_by_clique.get(clique)
            if occupied and not occupied.issubset(member_ids):
                return True
        return False

    target = set(cliques)
    for nid, data in G.nodes(data=True):
        if nid in member_ids:
            continue
        if data.get("type") != "adsorbate":
            continue
        if not data.get("occupied", False):
            continue
        clique = data.get("clique")
        if clique is not None and clique in target:
            return True
    return False


def _members_share_exact_clique(
    G: nx.Graph,
    site_a,
    m_a: int,
    site_b,
    m_b: int,
) -> bool:
    return bool(
        set(_member_cliques(G, site_a, m_a))
        & set(_member_cliques(G, site_b, m_b))
    )


def execute_reaction(G: nx.Graph, reaction) -> set:
    """Apply *reaction* in place and return touched surface cliques."""
    if getattr(reaction, "kind", None) == "diffusion":
        ds: DiffusionSite = reaction.site
        site_a, m_a, site_b, m_b = ds.members[reaction.member_index]
        direction = getattr(reaction, "direction", None)
        if direction == "a_to_b":
            src_site, src_m = site_a, m_a
            tgt_site, tgt_m = site_b, m_b
        elif direction == "b_to_a":
            src_site, src_m = site_b, m_b
            tgt_site, tgt_m = site_a, m_a
        else:
            raise ValueError(f"unknown diffusion direction {direction!r}")
        if not _member_occupied(G, src_site, src_m):
            raise ValueError("stale diffusion reaction: source member is not occupied")
        if _member_occupied(G, tgt_site, tgt_m):
            raise ValueError("stale diffusion reaction: target member is already occupied")
        cliques = (
            _affected_surface_cliques(G, src_site, src_m)
            | _affected_surface_cliques(G, tgt_site, tgt_m)
        )
        _set_member_occupied(G, src_site, src_m, False)
        _set_member_occupied(G, tgt_site, tgt_m, True)
        return cliques

    if getattr(reaction, "kind", None) == "bond":
        brs: BondReactionSite = reaction.site
        gas_product = bool(getattr(brs, "gas_product", False))
        site_a, m_a, site_b, m_b, site_c, m_c = brs.members[reaction.member_index]
        direction = getattr(reaction, "direction", None)
        if direction not in {"couple", "dissoc"}:
            raise ValueError(f"unknown bond direction {direction!r}")
        if _members_share_exact_clique(G, site_a, m_a, site_b, m_b):
            raise ValueError(
                "stale bond reaction: A and B target members share a surface clique"
            )
        cliques = (
            _affected_surface_cliques(G, site_a, m_a)
            | _affected_surface_cliques(G, site_b, m_b)
        )
        if not gas_product and site_c is not None:
            cliques |= _affected_surface_cliques(G, site_c, m_c)
        if direction == "couple":
            if not _member_occupied(G, site_a, m_a):
                raise ValueError("stale bond reaction: A member is not occupied")
            if not _member_occupied(G, site_b, m_b):
                raise ValueError("stale bond reaction: B member is not occupied")
            if (not gas_product) and _member_occupied(G, site_c, m_c):
                raise ValueError("stale bond reaction: C member is already occupied")
            _set_member_occupied(G, site_a, m_a, False)
            _set_member_occupied(G, site_b, m_b, False)
            if not gas_product:
                _set_member_occupied(G, site_c, m_c, True)
        else:
            if (not gas_product) and not _member_occupied(G, site_c, m_c):
                raise ValueError("stale bond reaction: C member is not occupied")
            if _member_occupied(G, site_a, m_a):
                raise ValueError("stale bond reaction: A member is already occupied")
            if _member_occupied(G, site_b, m_b):
                raise ValueError("stale bond reaction: B member is already occupied")
            if not gas_product:
                _set_member_occupied(G, site_c, m_c, False)
            _set_member_occupied(G, site_a, m_a, True)
            _set_member_occupied(G, site_b, m_b, True)
        return cliques

    if reaction.kind not in {"adsorption", "desorption"}:
        raise ValueError(f"unknown reaction kind {reaction.kind!r}")
    new_state = reaction.kind == "adsorption"
    current_state = _member_occupied(G, reaction.site, reaction.member_index)
    if new_state:
        if current_state:
            raise ValueError("stale adsorption reaction: member is already occupied")
        if _member_blocked_by_other(G, reaction.site, reaction.member_index):
            raise ValueError(
                "stale adsorption reaction: member is clique-blocked by an occupied adsorbate"
            )
    elif not current_state:
        raise ValueError("stale desorption reaction: member is not occupied")
    cliques = _affected_surface_cliques(G, reaction.site, reaction.member_index)
    _set_member_occupied(G, reaction.site, reaction.member_index, new_state)
    return cliques


__all__ = ["execute_reaction"]
