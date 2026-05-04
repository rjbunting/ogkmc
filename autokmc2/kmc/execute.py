"""Reaction execution and graph mutation."""

from __future__ import annotations

import networkx as nx

from autokmc2.kmc.state import _affected_surface_cliques, _set_member_occupied
from autokmc2.sites.bond import BondReactionSite
from autokmc2.sites.diffusion import DiffusionSite


def _member_occupied(G: nx.Graph, site, member_index: int) -> bool:
    return any(
        nid in G and G.nodes[nid].get("occupied", False)
        for nid in site.member_node_ids[member_index]
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
        site_a, m_a, site_b, m_b, site_c, m_c = brs.members[reaction.member_index]
        direction = getattr(reaction, "direction", None)
        if direction not in {"couple", "dissoc"}:
            raise ValueError(f"unknown bond direction {direction!r}")
        cliques = (
            _affected_surface_cliques(G, site_a, m_a)
            | _affected_surface_cliques(G, site_b, m_b)
            | _affected_surface_cliques(G, site_c, m_c)
        )
        if direction == "couple":
            if not _member_occupied(G, site_a, m_a):
                raise ValueError("stale bond reaction: A member is not occupied")
            if not _member_occupied(G, site_b, m_b):
                raise ValueError("stale bond reaction: B member is not occupied")
            if _member_occupied(G, site_c, m_c):
                raise ValueError("stale bond reaction: C member is already occupied")
            _set_member_occupied(G, site_a, m_a, False)
            _set_member_occupied(G, site_b, m_b, False)
            _set_member_occupied(G, site_c, m_c, True)
        else:
            if not _member_occupied(G, site_c, m_c):
                raise ValueError("stale bond reaction: C member is not occupied")
            if _member_occupied(G, site_a, m_a):
                raise ValueError("stale bond reaction: A member is already occupied")
            if _member_occupied(G, site_b, m_b):
                raise ValueError("stale bond reaction: B member is already occupied")
            _set_member_occupied(G, site_c, m_c, False)
            _set_member_occupied(G, site_a, m_a, True)
            _set_member_occupied(G, site_b, m_b, True)
        return cliques

    if reaction.kind not in {"adsorption", "desorption"}:
        raise ValueError(f"unknown reaction kind {reaction.kind!r}")
    new_state = reaction.kind == "adsorption"
    cliques = _affected_surface_cliques(G, reaction.site, reaction.member_index)
    _set_member_occupied(G, reaction.site, reaction.member_index, new_state)
    return cliques


__all__ = ["execute_reaction"]
