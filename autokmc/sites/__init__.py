"""Public surface-site enumeration API.

The implementation is split by site family:

* :mod:`autokmc.sites.anchors` finds bare-surface anchor cliques.
* :mod:`autokmc.sites.adsorbate` builds adsorption placements.
* :mod:`autokmc.sites.diffusion` builds diffusion candidates.
* :mod:`autokmc.sites.bond` builds bond-forming and bond-breaking candidates.

Keep private geometry, ego-graph, and lateral helpers in their owning modules
until they are promoted into real reusable abstractions. Thin modules that only
re-export underscore-prefixed helpers make the package harder to reason about.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnchorSite": "autokmc.sites.anchors",
    "find_anchor_sites": "autokmc.sites.anchors",
    "k_max_for_element": "autokmc.sites.anchors",
    "AdsorbateSite": "autokmc.sites.adsorbate",
    "AdsorbateSiteLateral": "autokmc.sites.adsorbate",
    "find_adsorbate_sites": "autokmc.sites.adsorbate",
    "optimise_adsorbate_site_positions": "autokmc.sites.adsorbate",
    "prune_unstable_adsorbate_sites": "autokmc.sites.adsorbate",
    "push_member_positions_to_graph": "autokmc.sites.adsorbate",
    "rebuild_adsorbate_reverse_indexes": "autokmc.sites.adsorbate",
    "DiffusionLateral": "autokmc.sites.diffusion",
    "DiffusionSite": "autokmc.sites.diffusion",
    "find_diffusion_sites": "autokmc.sites.diffusion",
    "rebuild_diffusion_reverse_indexes": "autokmc.sites.diffusion",
    "BondReactionLateral": "autokmc.sites.bond",
    "BondReactionSite": "autokmc.sites.bond",
    "BondReactionTemplate": "autokmc.sites.bond",
    "derive_bond_templates": "autokmc.sites.bond",
    "derive_coupling_templates": "autokmc.sites.bond",
    "derive_dissociation_templates": "autokmc.sites.bond",
    "find_bond_sites": "autokmc.sites.bond",
    "prune_unstable_bond_sites": "autokmc.sites.bond",
    "rebuild_bond_reverse_indexes": "autokmc.sites.bond",
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
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
