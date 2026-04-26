"""
autokmc – Automated KMC workflow package.

Re-exports
----------
The most commonly used names are surfaced here so users can do
``from autokmc import build_graph, find_surface_atoms`` without
remembering the submodule layout.
"""

from autokmc import constants  # noqa: F401
from autokmc.cache import SiteCache, get_cache  # noqa: F401
from autokmc.graph import build_graph  # noqa: F401
from autokmc.logging_utils import get_logger  # noqa: F401
from autokmc.results import SurfaceClassification  # noqa: F401
from autokmc.surface import (  # noqa: F401
    find_surface_atoms,
    find_surface_atoms_convexhull,
    find_surface_atoms_raycasting,
    has_pbc_connectivity,
    tag_surface_atoms,
)
from autokmc.sites import (  # noqa: F401
    # Preferred new-name public API
    AdsorbateSite,
    find_adsorbate_sites,
    find_adsorbate_sites_for_reactant,
    optimise_adsorbate_site_positions,
    push_member_positions_to_graph,
    # Backward-compatible legacy aliases
    MultiSite,
    find_multisites,
    find_multisites_for_reactant,
    optimise_multisite_positions,
)
from autokmc.opt_site import (  # noqa: F401
    lateral_neighbour_atoms,
    lateral_neighbour_subgraph,
    # Preferred names
    optimise_adsorbate_sites_ml,
    seed_single_atom_adsorbate_sites,
    # Backward-compatible legacy aliases
    optimise_multisites_ml,
    seed_single_atom_multisites,
)
