"""Public stability and lateral-classification API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AdsorbateDissociationError": "autokmc.sites.stability.adsorption",
    "OptimisationFailedError": "autokmc.sites.stability.adsorption",
    "SiteStabilityError": "autokmc.sites.stability.adsorption",
    "SurfaceConnectivityError": "autokmc.sites.stability.adsorption",
    "check_adsorbate_site_lateral": "autokmc.sites.stability.adsorption",
    "check_site_stability": "autokmc.sites.stability.adsorption",
    "DiffusionStabilityError": "autokmc.sites.stability.diffusion",
    "EndpointStabilityError": "autokmc.sites.stability.diffusion",
    "NEBNotConvergedError": "autokmc.sites.stability.diffusion",
    "TransitionStateInvalidError": "autokmc.sites.stability.diffusion",
    "check_diffusion_site_lateral": "autokmc.sites.stability.diffusion",
    "check_diffusion_stability": "autokmc.sites.stability.diffusion",
    "BondEndpointStabilityError": "autokmc.sites.stability.bond",
    "BondNEBNotConvergedError": "autokmc.sites.stability.bond",
    "BondStabilityError": "autokmc.sites.stability.bond",
    "BondTransitionStateInvalidError": "autokmc.sites.stability.bond",
    "check_bond_site_lateral": "autokmc.sites.stability.bond",
    "check_bond_site_stability": "autokmc.sites.stability.bond",
    "NEBRunResult": "autokmc.sites.stability.neb",
    "make_neb_band": "autokmc.sites.stability.neb",
    "neb_optimizer_logfile": "autokmc.sites.stability.neb",
    "run_neb": "autokmc.sites.stability.neb",
}

__all__ = [
    "AdsorbateDissociationError",
    "OptimisationFailedError",
    "SiteStabilityError",
    "SurfaceConnectivityError",
    "check_adsorbate_site_lateral",
    "check_site_stability",
    "DiffusionStabilityError",
    "EndpointStabilityError",
    "NEBNotConvergedError",
    "TransitionStateInvalidError",
    "check_diffusion_site_lateral",
    "check_diffusion_stability",
    "BondEndpointStabilityError",
    "BondNEBNotConvergedError",
    "BondStabilityError",
    "BondTransitionStateInvalidError",
    "check_bond_site_lateral",
    "check_bond_site_stability",
    "NEBRunResult",
    "make_neb_band",
    "neb_optimizer_logfile",
    "run_neb",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
