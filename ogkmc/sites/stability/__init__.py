"""Public stability and lateral-classification API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AdsorbateDissociationError": "ogkmc.sites.stability.adsorption",
    "OptimisationFailedError": "ogkmc.sites.stability.adsorption",
    "SiteStabilityError": "ogkmc.sites.stability.adsorption",
    "SurfaceConnectivityError": "ogkmc.sites.stability.adsorption",
    "check_adsorbate_site_lateral": "ogkmc.sites.stability.adsorption",
    "check_site_stability": "ogkmc.sites.stability.adsorption",
    "DiffusionStabilityError": "ogkmc.sites.stability.diffusion",
    "EndpointStabilityError": "ogkmc.sites.stability.diffusion",
    "NEBNotConvergedError": "ogkmc.sites.stability.diffusion",
    "TransitionStateInvalidError": "ogkmc.sites.stability.diffusion",
    "check_diffusion_site_lateral": "ogkmc.sites.stability.diffusion",
    "check_diffusion_stability": "ogkmc.sites.stability.diffusion",
    "get_diffusion_bare_lateral": "ogkmc.sites.stability.diffusion",
    "BondEndpointStabilityError": "ogkmc.sites.stability.bond",
    "BondNEBNotConvergedError": "ogkmc.sites.stability.bond",
    "BondStabilityError": "ogkmc.sites.stability.bond",
    "BondTransitionStateInvalidError": "ogkmc.sites.stability.bond",
    "check_bond_site_lateral": "ogkmc.sites.stability.bond",
    "check_bond_site_stability": "ogkmc.sites.stability.bond",
    "get_bond_bare_lateral": "ogkmc.sites.stability.bond",
    "NEBRunResult": "ogkmc.sites.stability.neb",
    "make_neb_band": "ogkmc.sites.stability.neb",
    "neb_optimizer_logfile": "ogkmc.sites.stability.neb",
    "project_neb_path": "ogkmc.sites.stability.neb",
    "run_neb": "ogkmc.sites.stability.neb",
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
    "get_diffusion_bare_lateral",
    "BondEndpointStabilityError",
    "BondNEBNotConvergedError",
    "BondStabilityError",
    "BondTransitionStateInvalidError",
    "check_bond_site_lateral",
    "check_bond_site_stability",
    "get_bond_bare_lateral",
    "NEBRunResult",
    "make_neb_band",
    "neb_optimizer_logfile",
    "project_neb_path",
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
