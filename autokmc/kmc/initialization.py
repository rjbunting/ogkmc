"""Initial reaction sweeps and mutable-state construction for KMC."""

from __future__ import annotations

from dataclasses import replace

from autokmc.core.graph_state import N_OCCUPIED, set_n_occupied
from autokmc.io.event_log import EventHistory
from autokmc.kmc.index import _ReactionIndex
from autokmc.kmc.models import (
    KMCChannels,
    KMCResumeState,
    KMCRuntime,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.kmc.restart import normalise_rng
from autokmc.reactions.adsorption import (
    _build_gas_energy_lookup,
    _build_gas_g_lookup,
    _build_partial_pressure_lookup,
)
from autokmc.reactions.bond import compute_all_bond_reactions
from autokmc.reactions.diffusion import compute_all_diffusions
from autokmc.reactions.rates import H_EV_S, KB_EV
from autokmc.utils.telemetry import increment, set_gauge


def normalise_channels(
    channels: KMCChannels,
    thermochemistry: KMCThermochemistry,
) -> tuple[KMCChannels, KMCThermochemistry]:
    """Copy caller-owned site containers while retaining typed options."""
    cache_root = thermochemistry.calculation_cache_root
    if cache_root is None:
        cache_root = channels._legacy_diffusion_reserved.get(
            "calculation_cache_root"
        )
    if cache_root is None:
        cache_root = channels._legacy_bond_reserved.get("calculation_cache_root")
    if cache_root is None:
        cache_root = channels._legacy_bond_growth_reserved.get(
            "calculation_cache_root"
        )
    cache_lookup_enabled = thermochemistry.calculation_cache_lookup_enabled
    if cache_lookup_enabled is None:
        for reserved_values in (
            channels._legacy_diffusion_reserved,
            channels._legacy_bond_reserved,
            channels._legacy_bond_growth_reserved,
        ):
            if "calculation_cache_lookup_enabled" in reserved_values:
                cache_lookup_enabled = bool(
                    reserved_values["calculation_cache_lookup_enabled"]
                )
                break
    if cache_lookup_enabled is None:
        cache_lookup_enabled = False

    free_energy_options = thermochemistry.free_energy_options
    for reserved_values in (
        channels._legacy_diffusion_reserved,
        channels._legacy_bond_reserved,
        channels._legacy_bond_growth_reserved,
    ):
        if free_energy_options is None:
            free_energy_options = reserved_values.get("free_energy_options")
    vib_cache_root = thermochemistry.vib_cache_root
    for reserved_values in (
        channels._legacy_diffusion_reserved,
        channels._legacy_bond_reserved,
        channels._legacy_bond_growth_reserved,
    ):
        if vib_cache_root is None:
            vib_cache_root = reserved_values.get("vib_cache_root")
    free_energy_temperature_k = thermochemistry.free_energy_temperature_k
    if free_energy_temperature_k is None:
        legacy_temperature = channels._legacy_bond_growth_reserved.get(
            "free_energy_temperature_k"
        )
        if legacy_temperature is not None:
            free_energy_temperature_k = float(legacy_temperature)

    normalised = KMCChannels(
        diffusion_sites=list(channels.diffusion_sites),
        diffusion_options=channels.diffusion_options,
        bond_sites=list(channels.bond_sites),
        bond_options=channels.bond_options,
        bond_growth_options=channels.bond_growth_options,
    )
    # Preserve the sparse legacy mapping view for callers that inspect it;
    # scientific execution uses the fully typed option dataclasses above.
    normalised._diffusion_mapping_view = channels.diffusion_kwargs
    normalised._bond_mapping_view = channels.bond_kwargs
    normalised._bond_growth_mapping_view = channels.bond_growth_kwargs
    return normalised, replace(
        thermochemistry,
        calculation_cache_root=cache_root,
        calculation_cache_lookup_enabled=cache_lookup_enabled,
        free_energy_options=free_energy_options,
        vib_cache_root=vib_cache_root,
        free_energy_temperature_k=free_energy_temperature_k,
    )


def _print_initial_header(
    system: KMCSystem,
    settings: KMCSettings,
) -> None:
    if not settings.verbose:
        return
    print("\n[KMC] ═══════════════════════════════════════════════════════")
    print("[KMC]  KMC initialisation")
    print(
        f"[KMC]  T = {settings.temperature} K   "
        f"κ = {settings.transmission_coefficient}   "
        f"ν_Eyring = "
        f"{settings.transmission_coefficient * KB_EV * settings.temperature / H_EV_S:.3e} Hz"
    )
    print(
        f"[KMC]  max steps = {settings.n_steps}   "
        f"lateral interactions = {settings.lateral_interactions}"
    )
    print("[KMC] ─────────────────────────────────────────────────────────")
    n_members_ads = sum(len(site.member_node_ids) for site in system.adsorbate_sites)
    print(
        "[KMC]  Adsorption/desorption channel"
        f"  │  {len(system.adsorbate_sites)} iso-class(es)"
        f"  │  {n_members_ads} total member(s)"
    )


def initialise_runtime(
    system: KMCSystem,
    settings: KMCSettings,
    channels: KMCChannels,
    thermochemistry: KMCThermochemistry,
    resume: KMCResumeState,
    *,
    rng,
    compute_adsorption,
) -> KMCRuntime:
    """Run each enabled channel's initial sweep and build its rate index."""
    graph = system.graph
    gas_energies = _build_gas_energy_lookup(system.reactants)
    gas_free_energies = _build_gas_g_lookup(system.reactants)
    partial_pressures = _build_partial_pressure_lookup(system.reactants)
    random_source = normalise_rng(
        rng,
        dict(resume.rng_state) if resume.rng_state is not None else None,
    )

    _print_initial_header(system, settings)

    compute_adsorption(
        graph,
        system.adsorbate_sites,
        system.calculator,
        gas_energies,
        temperature=settings.temperature,
        transmission_coefficient=settings.transmission_coefficient,
        frozen_indices=settings.frozen_indices,
        fmax=settings.fmax,
        max_steps=settings.max_steps,
        optimizer=settings.optimizer,
        verbose=settings.verbose,
        lateral_interactions=settings.lateral_interactions,
        lateral_shells=settings.lateral_shells,
        gas_g=gas_free_energies,
        partial_pressures=partial_pressures,
        free_energy_options=thermochemistry.free_energy_options,
        vib_cache_root=thermochemistry.vib_cache_root,
        calculation_cache_root=thermochemistry.calculation_cache_root,
        calculation_cache_lookup_enabled=bool(
            thermochemistry.calculation_cache_lookup_enabled
        ),
    )
    if settings.verbose:
        n_reactions = sum(
            len(getattr(site, "applicable_reactions", None) or [])
            for site in system.adsorbate_sites
        )
        print(f"[KMC]    → {n_reactions} applicable reaction(s) after initial sweep")

    if settings.verbose:
        if channels.diffusion_sites:
            n_members = sum(
                len(site.member_node_ids) for site in channels.diffusion_sites
            )
            print(
                "[KMC]  Diffusion channel"
                f"  │  {len(channels.diffusion_sites)} iso-class(es)"
                f"  │  {n_members} total member(s)"
                "  │  NEB lazily per new lateral class"
            )
        else:
            print(
                "[KMC]  Diffusion channel  │  disabled "
                "(no DiffusionSite(s) supplied)"
            )
    if channels.diffusion_sites:
        compute_all_diffusions(
            graph,
            channels.diffusion_sites,
            system.calculator,
            temperature=settings.temperature,
            transmission_coefficient=settings.transmission_coefficient,
            frozen_indices=settings.frozen_indices,
            verbose=settings.verbose,
            lateral_interactions=settings.lateral_interactions,
            lateral_shells=settings.lateral_shells,
            free_energy_options=thermochemistry.free_energy_options,
            vib_cache_root=thermochemistry.vib_cache_root,
            calculation_cache_root=thermochemistry.calculation_cache_root,
            calculation_cache_lookup_enabled=bool(
                thermochemistry.calculation_cache_lookup_enabled
            ),
            **channels.diffusion_options.to_kwargs(),
        )
        if settings.verbose:
            n_reactions = sum(
                len(getattr(site, "applicable_reactions", None) or [])
                for site in channels.diffusion_sites
            )
            print(f"[KMC]    → {n_reactions} applicable diffusion(s) after initial sweep")

    if settings.verbose:
        if channels.bond_sites:
            n_members = sum(len(site.member_node_ids) for site in channels.bond_sites)
            print(
                "[KMC]  Bond-reaction channel"
                f"  │  {len(channels.bond_sites)} iso-class(es)"
                f"  │  {n_members} total member(s)"
                "  │  NEB lazily per new lateral class"
            )
        else:
            print(
                "[KMC]  Bond-reaction channel  │  disabled "
                "(no BondReactionSite(s) supplied)"
            )
    if channels.bond_sites:
        compute_all_bond_reactions(
            graph,
            channels.bond_sites,
            system.calculator,
            temperature=settings.temperature,
            transmission_coefficient=settings.transmission_coefficient,
            frozen_indices=settings.frozen_indices,
            verbose=settings.verbose,
            lateral_interactions=settings.lateral_interactions,
            lateral_shells=settings.lateral_shells,
            calculation_cache_root=thermochemistry.calculation_cache_root,
            calculation_cache_lookup_enabled=bool(
                thermochemistry.calculation_cache_lookup_enabled
            ),
            free_energy_options=thermochemistry.free_energy_options,
            vib_cache_root=thermochemistry.vib_cache_root,
            **channels.bond_options.to_kwargs(),
        )
        if settings.verbose:
            n_reactions = sum(
                len(getattr(site, "applicable_reactions", None) or [])
                for site in channels.bond_sites
            )
            print(
                f"[KMC]    → {n_reactions} applicable bond reaction(s) "
                "after initial sweep"
            )

    reaction_index = _ReactionIndex(
        system.adsorbate_sites,
        channels.diffusion_sites,
        channels.bond_sites,
    )
    initial_sites = (
        *system.adsorbate_sites,
        *channels.diffusion_sites,
        *channels.bond_sites,
    )
    reaction_index.install_sites(initial_sites)
    increment("kmc.index.initial_builds")
    increment("kmc.index.initial_leaves", reaction_index.n_total)
    set_gauge("kmc.index.leaves", reaction_index.n_total)

    if settings.verbose:
        n_active = sum(reaction is not None for reaction in reaction_index.reactions)
        print(
            "[KMC]  Segment-tree index built"
            f"  │  {reaction_index.n_total} leaves"
            f"  │  {n_active} active reaction(s)"
            f"  │  Q₀ = {reaction_index.total_rate():.3e} Hz"
        )
        print("[KMC] ═══════════════════════════════════════════════════════\n")

    if N_OCCUPIED not in graph.graph:
        set_n_occupied(
            graph,
            sum(
            1
            for site in system.adsorbate_sites
            for node_ids in site.member_node_ids
            if any(
                node_id in graph and graph.nodes[node_id].get("occupied", False)
                for node_id in node_ids
            )
            ),
        )
    for site in system.adsorbate_sites:
        if not hasattr(site, "_n_occupied"):
            site._n_occupied = sum(
                1
                for node_ids in site.member_node_ids
                if any(
                    node_id in graph and graph.nodes[node_id].get("occupied", False)
                    for node_id in node_ids
                )
            )

    reaction_counts = {
        "adsorption": 0,
        "desorption": 0,
        "diffusion": 0,
        "bond": 0,
        "bond_couple": 0,
        "bond_dissoc": 0,
    }
    reaction_counts.update(
        {str(key): int(value) for key, value in (resume.reaction_counts or {}).items()}
    )
    resume_history = resume.history
    runtime_history = (
        resume_history
        if isinstance(resume_history, EventHistory)
        else list(resume_history or [])
    )
    return KMCRuntime(
        rng=random_source,
        gas_energies=gas_energies,
        gas_free_energies=gas_free_energies,
        partial_pressures=partial_pressures,
        reaction_index=reaction_index,
        history=runtime_history,
        reaction_counts=reaction_counts,
        current_time_s=float(resume.time_s or 0.0),
        start_step=int(resume.step or 0),
    )


__all__ = ["initialise_runtime", "normalise_channels"]
