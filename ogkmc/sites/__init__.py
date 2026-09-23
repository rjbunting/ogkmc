"""Public surface-site enumeration API.

The implementation is split by site family:

* :mod:`ogkmc.sites.anchors` finds bare-surface anchor cliques.
* :mod:`ogkmc.sites.adsorbate` builds adsorption placements.
* :mod:`ogkmc.sites.diffusion` builds diffusion candidates.
* :mod:`ogkmc.sites.bond` builds bond-forming and bond-breaking candidates.

Keep private geometry, ego-graph, and lateral helpers in their owning modules
until they are promoted into real reusable abstractions. Thin modules that only
re-export underscore-prefixed helpers make the package harder to reason about.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnchorSite": "ogkmc.sites.anchors",
    "find_anchor_sites": "ogkmc.sites.anchors",
    "k_max_for_element": "ogkmc.sites.anchors",
    "AdsorbateSite": "ogkmc.sites.adsorbate",
    "AdsorbateSiteLateral": "ogkmc.sites.adsorbate",
    "find_adsorbate_sites": "ogkmc.sites.adsorbate",
    "optimise_adsorbate_site_positions": "ogkmc.sites.adsorbate",
    "prune_unstable_adsorbate_sites": "ogkmc.sites.adsorbate",
    "push_member_positions_to_graph": "ogkmc.sites.adsorbate",
    "rebuild_adsorbate_reverse_indexes": "ogkmc.sites.adsorbate",
    "DiffusionLateral": "ogkmc.sites.diffusion",
    "DiffusionSite": "ogkmc.sites.diffusion",
    "find_diffusion_sites": "ogkmc.sites.diffusion",
    "rebuild_diffusion_reverse_indexes": "ogkmc.sites.diffusion",
    "BondReactionLateral": "ogkmc.sites.bond",
    "BondReactionSite": "ogkmc.sites.bond",
    "BondReactionTemplate": "ogkmc.sites.bond",
    "derive_bond_templates": "ogkmc.sites.bond",
    "derive_coupling_templates": "ogkmc.sites.bond",
    "derive_dissociation_templates": "ogkmc.sites.bond",
    "find_bond_sites": "ogkmc.sites.bond",
    "prune_unstable_bond_sites": "ogkmc.sites.bond",
    "rebuild_bond_reverse_indexes": "ogkmc.sites.bond",
    "SiteId": "ogkmc.sites.identity",
    "MemberSignature": "ogkmc.sites.identity",
    "SiteMemberId": "ogkmc.sites.identity",
    "member_identifier": "ogkmc.sites.identity",
    "member_signature": "ogkmc.sites.identity",
    "site_identifier": "ogkmc.sites.identity",
}

__all__ = [
    "AnchorSite",
    "find_anchor_sites",
    "k_max_for_element",
    "AdsorbateSite",
    "AdsorbateSiteLateral",
    "find_adsorbate_sites",
    "optimise_adsorbate_site_positions",
    "prune_unstable_adsorbate_sites",
    "push_member_positions_to_graph",
    "rebuild_adsorbate_reverse_indexes",
    "DiffusionLateral",
    "DiffusionSite",
    "find_diffusion_sites",
    "rebuild_diffusion_reverse_indexes",
    "BondReactionLateral",
    "BondReactionSite",
    "BondReactionTemplate",
    "derive_bond_templates",
    "derive_coupling_templates",
    "derive_dissociation_templates",
    "find_bond_sites",
    "prune_unstable_bond_sites",
    "rebuild_bond_reverse_indexes",
    "SiteId",
    "MemberSignature",
    "SiteMemberId",
    "member_identifier",
    "member_signature",
    "site_identifier",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
