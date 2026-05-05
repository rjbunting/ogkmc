"""Reaction models, templates, applicability, and rate construction."""

from __future__ import annotations

from autokmc.reactions.adsorption import (
    AdsorptionReaction,
    Reaction,
    compute_all_reactions,
    fast_reaction_for_member,
    gather_all_applicable_reactions,
    get_applicable_reactions,
    is_clique_blocked,
)
from autokmc.reactions.bond import (
    BondReaction,
    compute_all_bond_reactions,
    fast_bond_reaction_for_member,
    get_applicable_bond_reactions,
    is_bond_applicable,
)
from autokmc.reactions.diffusion import (
    DiffusionReaction,
    compute_all_diffusions,
    fast_diffusion_for_member,
    get_applicable_diffusions,
    is_diffusion_applicable,
)
from autokmc.reactions.models import ReactionLike, ReactionSnapshot
from autokmc.reactions.rates import (
    DEFAULT_TRANSMISSION_COEFFICIENT,
    EA_MIN,
    H_EV_S,
    KB_EV,
)
from autokmc.reactions.templates import (
    BondReactionTemplate,
    derive_bond_templates,
    derive_coupling_templates,
    derive_dissociation_templates,
)

__all__ = [
    "AdsorptionReaction",
    "Reaction",
    "is_clique_blocked",
    "get_applicable_reactions",
    "compute_all_reactions",
    "gather_all_applicable_reactions",
    "fast_reaction_for_member",
    "DiffusionReaction",
    "is_diffusion_applicable",
    "get_applicable_diffusions",
    "compute_all_diffusions",
    "fast_diffusion_for_member",
    "BondReaction",
    "is_bond_applicable",
    "get_applicable_bond_reactions",
    "compute_all_bond_reactions",
    "fast_bond_reaction_for_member",
    "ReactionLike",
    "ReactionSnapshot",
    "KB_EV",
    "H_EV_S",
    "DEFAULT_TRANSMISSION_COEFFICIENT",
    "EA_MIN",
    "BondReactionTemplate",
    "derive_dissociation_templates",
    "derive_coupling_templates",
    "derive_bond_templates",
]
