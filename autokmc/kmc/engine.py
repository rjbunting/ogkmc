"""Backward-compatible public entry point for kinetic Monte Carlo runs."""

from __future__ import annotations

import random
from typing import Iterable

import networkx as nx
import numpy as np

from autokmc.kmc.execute import execute_reaction
from autokmc.kmc.expansion import expand_bond_sites_after_event
from autokmc.kmc.models import (
    BondChannelOptions,
    DiffusionChannelOptions,
    KMCChannels,
    KMCFunctions,
    KMCObservers,
    KMCResumeState,
    KMCRunRequest,
    KMCRunResult,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.core.constants import LATERAL_SHELLS_DEFAULT
from autokmc.kmc.recompute import recompute_affected_sites
from autokmc.kmc.restart import (
    capture_rng_state,
    final_occupancy_by_species,
    reactants_for_checkpoint,
    restore_rng_state,
)
from autokmc.kmc.sampling import choose_reaction, sample_tau, total_rate
from autokmc.kmc.session import KMCSession
from autokmc.reactions.adsorption import compute_all_reactions
from autokmc.reactions.rates import DEFAULT_TRANSMISSION_COEFFICIENT
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.species.reactant import Reactant

# Historic private helpers remain importable for downstream users and tests.
_capture_rng_state = capture_rng_state
_restore_rng_state = restore_rng_state
_reactants_for_checkpoint = reactants_for_checkpoint
_final_occupancy_by_species = final_occupancy_by_species
_recompute_affected_sites = recompute_affected_sites


def default_kmc_functions() -> KMCFunctions:
    """Return the standard kernels, preserving historic monkeypatch seams."""
    return KMCFunctions(
        compute_adsorption=compute_all_reactions,
        recompute_affected=_recompute_affected_sites,
        expand_bond_network=expand_bond_sites_after_event,
    )


def run_kmc(
    request: KMCRunRequest,
    *,
    functions: KMCFunctions | None = None,
) -> KMCRunResult:
    """Execute one typed KMC request and return its typed result."""
    return KMCSession(
        request=request,
        functions=functions or default_kmc_functions(),
    ).run()


def run_kmc_steps(
    G: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    calculator,
    reactants: Reactant | Iterable[Reactant] | dict,
    *,
    temperature: float,
    n_steps: int,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    rng: random.Random | np.random.Generator | int | None = None,
    log_every: int = 100,
    progress: bool | None = None,
    verbose: bool = True,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
    bond_sites: list[BondReactionSite] | None = None,
    bond_kwargs: dict | None = None,
    bond_growth_kwargs: dict | None = None,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    reaction_writer=None,
    trajectory_writer=None,
    summary_collector=None,
    checkpoint_writer=None,
    initial_step: int = 0,
    initial_time_s: float = 0.0,
    initial_history: list | None = None,
    initial_reaction_counts: dict[str, int] | None = None,
    initial_rng_state: dict | None = None,
) -> dict:
    """Run up to ``n_steps`` BKL/Gillespie events in place on ``G``.

    The signature and returned mapping are retained for compatibility.  The
    implementation is delegated to :class:`~autokmc.kmc.session.KMCSession`,
    whose typed inputs separate scientific settings, channels, persistence,
    and checkpoint state.
    """
    diffusion_values = dict(diffusion_kwargs or {})
    bond_values = dict(bond_kwargs or {})
    resolved_cache_root = calculation_cache_root
    if resolved_cache_root is None:
        resolved_cache_root = diffusion_values.pop("calculation_cache_root", None)
    else:
        diffusion_values.pop("calculation_cache_root", None)
    if resolved_cache_root is None:
        resolved_cache_root = bond_values.pop("calculation_cache_root", None)
    else:
        bond_values.pop("calculation_cache_root", None)
    for reserved in ("free_energy_options", "vib_cache_root"):
        diffusion_values.pop(reserved, None)
        bond_values.pop(reserved, None)

    request = KMCRunRequest(
        system=KMCSystem(
            graph=G,
            adsorbate_sites=adsorbate_sites,
            calculator=calculator,
            reactants=reactants,
        ),
        settings=KMCSettings(
            temperature=temperature,
            n_steps=n_steps,
            transmission_coefficient=transmission_coefficient,
            frozen_indices=frozen_indices,
            fmax=fmax,
            max_steps=max_steps,
            log_every=log_every,
            progress=verbose if progress is None else progress,
            verbose=verbose,
            lateral_interactions=lateral_interactions,
            lateral_shells=lateral_shells,
        ),
        channels=KMCChannels(
            diffusion_sites=list(diffusion_sites or []),
            diffusion_options=DiffusionChannelOptions.from_mapping(
                diffusion_values
            ),
            bond_sites=list(bond_sites or []),
            bond_options=BondChannelOptions.from_mapping(bond_values),
            bond_growth_kwargs=bond_growth_kwargs,
        ),
        thermochemistry=KMCThermochemistry(
            free_energy_options=free_energy_options,
            vib_cache_root=vib_cache_root,
            calculation_cache_root=resolved_cache_root,
        ),
        observers=KMCObservers(
            reaction_writer=reaction_writer,
            trajectory_writer=trajectory_writer,
            summary_collector=summary_collector,
            checkpoint_writer=checkpoint_writer,
        ),
        resume=KMCResumeState(
            step=initial_step,
            time_s=initial_time_s,
            history=list(initial_history or []),
            reaction_counts=dict(initial_reaction_counts or {}),
            rng_state=initial_rng_state,
        ),
        rng=rng,
    )
    return run_kmc(request, functions=default_kmc_functions()).to_legacy_dict()


# Stable reference used by the configured workflow to detect an explicitly
# replaced legacy entry point.  Normal execution constructs ``KMCSession``
# directly; a downstream monkeypatch/subclass hook continues to be honoured.
_RUN_KMC_STEPS_COMPAT_ADAPTER = run_kmc_steps


__all__ = [
    "choose_reaction",
    "default_kmc_functions",
    "execute_reaction",
    "run_kmc",
    "run_kmc_steps",
    "sample_tau",
    "total_rate",
]
