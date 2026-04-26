"""
autokmc.sites
=============
Unified site-discovery API.

This module is the **single canonical entry point** for everything
related to enumerating, classifying and refining adsorption sites on a
surface graph.  It collects the public surface of three internal
implementation modules:

* :mod:`autokmc.default_sites`  — single-atom site enumeration, k-clique
  iso-class reduction, calculator-free position optimisation, and the
  graph-anchor materialisation that backs every other stage.
* :mod:`autokmc.find_multisite` — multi-atom (molecular) adsorbate
  enumeration via orbit-canonical anchor-subset placement, surface-
  connectivity guarding, and rigid-body refinement.

Typical usage
-------------
::

    from autokmc.sites import (
        # single-atom anchor pipeline
        find_sites_for_element, reduce_sites_by_isomorphism,
        optimise_site_positions,
        # multi-atom adsorbate pipeline
        AdsorbateSite, find_adsorbate_sites,
        optimise_adsorbate_site_positions,
        push_member_positions_to_graph,
    )
"""

from __future__ import annotations

from autokmc.default_sites import (  # noqa: F401
    IsoClass,
    _build_clique_ego,
    _iso_prefilter_key,
    find_sites_for_element,
    k_max_for_element,
    k_max_for_radius,
    optimise_site_positions,
    propagate_positions_to_iso_classes,
    reduce_sites_by_isomorphism,
)
from autokmc.find_multisite import (  # noqa: F401
    AdsorbateSite,
    find_adsorbate_sites,
    optimise_adsorbate_site_positions,
    push_member_positions_to_graph,
)


__all__ = [
    # Single-atom
    "IsoClass",
    "find_sites_for_element",
    "k_max_for_element",
    "k_max_for_radius",
    "optimise_site_positions",
    "propagate_positions_to_iso_classes",
    "reduce_sites_by_isomorphism",
    "_build_clique_ego",
    "_iso_prefilter_key",
    # Multi-atom (preferred names)
    "AdsorbateSite",
    "find_adsorbate_sites",
    "optimise_adsorbate_site_positions",
    "push_member_positions_to_graph",
]

