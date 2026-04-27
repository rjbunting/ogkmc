"""autokmc — surface KMC input generation from atomic structures.

Public entry points (full pipeline, in order):

    structure.build_surface / structure.build_nanoparticle
        → surface.find_surface_atoms (writes atoms.arrays["surface"])
        → graph.build_graph          (returns nx.Graph)
        → reactants.build_reactant   (SMILES → Reactant)
        → find_anchors.find_anchor_sites
        → find_adsorbate_sites.find_adsorbate_sites
        → find_adsorbate_sites.optimise_adsorbate_site_positions
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export the most common entry points at package level.
from autokmc.structure import (
    build_nanoparticle,
    build_surface,
    optimise_bulk,
    optimise_structure,
)
from autokmc.surface import (
    find_surface_atoms,
    find_surface_atoms_convexhull,
    find_surface_atoms_raycasting,
    has_pbc_connectivity,
    tag_surface_atoms,
)
from autokmc.results import SurfaceClassification
from autokmc.graph import build_graph
from autokmc.reactants import (
    Reactant,
    build_reactant,
    find_anchor_atoms,
    find_unique_atoms,
)
from autokmc.find_anchors import (
    AnchorSite,
    find_anchor_sites,
    k_max_for_element,
)
from autokmc.find_adsorbate_sites import (
    AdsorbateSite,
    AdsorbateSiteLateral,
    find_adsorbate_sites,
    optimise_adsorbate_site_positions,
    push_member_positions_to_graph,
)
from autokmc.check_adsorbate_sites import (
    check_adsorbate_site_lateral,
    check_site_stability,
    SiteStabilityError,
    SurfaceConnectivityError,
    AdsorbateDissociationError,
    OptimisationFailedError,
)

__all__ = [
    "__version__",
    # structure
    "build_nanoparticle", "build_surface", "optimise_bulk", "optimise_structure",
    # surface
    "find_surface_atoms", "find_surface_atoms_convexhull",
    "find_surface_atoms_raycasting", "has_pbc_connectivity",
    "tag_surface_atoms", "SurfaceClassification",
    # graph
    "build_graph",
    # reactants
    "Reactant", "build_reactant", "find_anchor_atoms", "find_unique_atoms",
    # anchors
    "AnchorSite", "find_anchor_sites", "k_max_for_element",
    # adsorbate sites
    "AdsorbateSite", "AdsorbateSiteLateral", "find_adsorbate_sites",
    "optimise_adsorbate_site_positions", "push_member_positions_to_graph",
    # lateral interactions
    "check_adsorbate_site_lateral", "check_site_stability",
    "SiteStabilityError", "SurfaceConnectivityError",
    "AdsorbateDissociationError", "OptimisationFailedError",
]

