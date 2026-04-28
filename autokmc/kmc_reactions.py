"""
autokmc.kmc_reactions  (deprecated — use :mod:`autokmc.kmc_adsorption`)
=====================================================================
Backwards-compatibility shim.  All adsorption/desorption KMC machinery has
moved to :mod:`autokmc.kmc_adsorption` (with the dataclass renamed from
``Reaction`` to :class:`AdsorptionReaction`) so that the diffusion machinery
in :mod:`autokmc.kmc_diffusion` lives at the same naming level.

External code should import from :mod:`autokmc.kmc_adsorption` going
forward.  ``Reaction`` is preserved here as an alias for
:class:`AdsorptionReaction`.
"""
from __future__ import annotations

from autokmc.kmc_adsorption import (  # noqa: F401
    AdsorptionReaction,
    Reaction,
    KB_EV,
    H_EV_S,
    EA_MIN,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    is_clique_blocked,
    get_applicable_reactions,
    compute_all_reactions,
    gather_all_applicable_reactions,
    fast_reaction_for_member,
    _energetics,
    _energetics_cached,
    _eyring_prefactor,
    _build_gas_energy_lookup,
    _site_is_occupied,
)

__all__ = [
    "AdsorptionReaction", "Reaction",
    "KB_EV", "H_EV_S", "EA_MIN", "DEFAULT_TRANSMISSION_COEFFICIENT",
    "is_clique_blocked", "get_applicable_reactions",
    "compute_all_reactions", "gather_all_applicable_reactions",
    "fast_reaction_for_member",
]
