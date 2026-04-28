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
    prune_unstable_adsorbate_sites,
)
from autokmc.check_adsorbate_sites import (
    check_adsorbate_site_lateral,
    check_site_stability,
    SiteStabilityError,
    SurfaceConnectivityError,
    AdsorbateDissociationError,
    OptimisationFailedError,
)
from autokmc.kmc_adsorption import (
    AdsorptionReaction,
    Reaction,                       # alias of AdsorptionReaction (back-compat)
    KB_EV,
    H_EV_S,
    EA_MIN,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    is_clique_blocked,
    get_applicable_reactions,
    compute_all_reactions,
    gather_all_applicable_reactions,
    fast_reaction_for_member,
)
from autokmc.find_diffusion_sites import (
    DiffusionSite,
    DiffusionLateral,
    find_diffusion_sites,
)
from autokmc.check_diffusion_sites import (
    check_diffusion_site_lateral,
    check_diffusion_stability,
    DiffusionStabilityError,
    EndpointStabilityError,
    NEBNotConvergedError,
    TransitionStateInvalidError,
)
from autokmc.kmc_diffusion import (
    DiffusionReaction,
    is_diffusion_applicable,
    get_applicable_diffusions,
    compute_all_diffusions,
    fast_diffusion_for_member,
)
from autokmc.kmc_simulation import (
    total_rate,
    sample_tau,
    choose_reaction,
    execute_reaction,
    run_kmc_steps,
)
from autokmc.persistence import (
    ReactionRecord,
    ReactionWriter,
    TrajectoryWriter,
    ReactionSummary,
    atoms_from_graph,
    make_run_meta,
)
from autokmc.config import (
    RunConfig,
    OutputCfg,
    StructureCfg,
    ReactantCfg,
    CalculatorCfg,
    AdsorbateSitesCfg,
    KMCCfg,
    DiffusionCfg,
    ConfigError,
    load_config,
    build_calculator,
    calculator_meta,
)
from autokmc.cli import run_from_config

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
    "prune_unstable_adsorbate_sites",
    # lateral interactions
    "check_adsorbate_site_lateral", "check_site_stability",
    "SiteStabilityError", "SurfaceConnectivityError",
    "AdsorbateDissociationError", "OptimisationFailedError",
    # KMC reactions (adsorption / desorption)
    "AdsorptionReaction", "Reaction",
    "KB_EV", "H_EV_S", "EA_MIN", "DEFAULT_TRANSMISSION_COEFFICIENT",
    "is_clique_blocked", "get_applicable_reactions",
    "compute_all_reactions", "gather_all_applicable_reactions",
    "fast_reaction_for_member",
    # Diffusion (NEB)
    "DiffusionSite", "DiffusionLateral", "find_diffusion_sites",
    "check_diffusion_site_lateral", "check_diffusion_stability",
    "DiffusionStabilityError", "EndpointStabilityError",
    "NEBNotConvergedError", "TransitionStateInvalidError",
    "DiffusionReaction", "is_diffusion_applicable",
    "get_applicable_diffusions", "compute_all_diffusions",
    "fast_diffusion_for_member",
    # KMC simulation
    "total_rate", "sample_tau", "choose_reaction",
    "execute_reaction", "run_kmc_steps",
    # persistence
    "ReactionRecord", "ReactionWriter", "TrajectoryWriter",
    "ReactionSummary", "atoms_from_graph", "make_run_meta",
    # config / CLI
    "RunConfig", "OutputCfg", "StructureCfg", "ReactantCfg",
    "CalculatorCfg", "AdsorbateSitesCfg", "KMCCfg", "DiffusionCfg",
    "ConfigError",
    "load_config", "build_calculator", "calculator_meta",
    "run_from_config",
]

