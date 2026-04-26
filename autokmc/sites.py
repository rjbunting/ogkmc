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
* :mod:`autokmc.site_graph`     — *deprecated* legacy approach
  (anchor + co-face enumeration); not re-exported.  Kept as a stub for
  one release to avoid breaking pinned imports.

Naming
------
The pre-rename names ``MultiSite`` / ``find_multisites`` /
``optimise_multisite_positions`` / ``optimise_multisites_ml`` /
``seed_single_atom_multisites`` are kept as backward-compatible aliases
both here and in the implementation modules, but new code should prefer
the **adsorbate-site** spelling:

================================================  ===============================================
Legacy name                                       Preferred name
================================================  ===============================================
``MultiSite``                                     :class:`AdsorbateSite`
``find_multisites``                               :func:`find_adsorbate_sites`
``find_multisites_for_reactant``                  :func:`find_adsorbate_sites_for_reactant`
``optimise_multisite_positions``                  :func:`optimise_adsorbate_site_positions`
``optimise_multisites_ml``                        :func:`optimise_adsorbate_sites_ml`
``seed_single_atom_multisites``                   :func:`seed_single_atom_adsorbate_sites`
``cache.multisites`` / ``G.graph["multisites"]``  ``cache.adsorbate_sites`` / ``G.graph["adsorbate_sites"]``
``N_MULTISITE_RESTARTS``                          :data:`autokmc.constants.N_ADSORBATE_RESTARTS`
================================================  ===============================================

Both spellings continue to work — the cache field is a property alias,
the graph-level legacy keys are aliased to the same dict object in
:func:`autokmc.cache.get_cache`, and the legacy callable / class names
remain importable.

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

# ---------------------------------------------------------------------------
# Single-atom default sites (was: autokmc.default_sites)
# ---------------------------------------------------------------------------
from autokmc.default_sites import (  # noqa: F401
    IsoClass,
    find_sites_for_element,
    k_max_for_element,
    k_max_for_radius,
    optimise_site_positions,
    propagate_positions_to_iso_classes,
    reduce_sites_by_isomorphism,
)

# Internal helpers exposed for downstream modules that legitimately
# need the same iso-class machinery (e.g. opt_site.optimise_*_ml).
from autokmc.default_sites import (  # noqa: F401
    _build_clique_ego,
    _iso_prefilter_key,
)

# ---------------------------------------------------------------------------
# Multi-atom adsorbate sites (was: autokmc.find_multisite)
# ---------------------------------------------------------------------------
from autokmc.find_multisite import (  # noqa: F401
    # Preferred new-name public API
    AdsorbateSite,
    find_adsorbate_sites,
    find_adsorbate_sites_for_reactant,
    optimise_adsorbate_site_positions,
    push_member_positions_to_graph,
    # Legacy aliases (still importable here for back-compat)
    MultiSite,
    find_multisites,
    find_multisites_for_reactant,
    optimise_multisite_positions,
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
    # Multi-atom (preferred names)
    "AdsorbateSite",
    "find_adsorbate_sites",
    "find_adsorbate_sites_for_reactant",
    "optimise_adsorbate_site_positions",
    "push_member_positions_to_graph",
    # Multi-atom (legacy aliases)
    "MultiSite",
    "find_multisites",
    "find_multisites_for_reactant",
    "optimise_multisite_positions",
]

