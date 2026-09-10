"""Incremental lateral-state and rate recomputation after a KMC event."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import copy_context
from functools import partial
from typing import Any, TypeAlias

import networkx as nx

from autokmc.io.calculators import CalculatorPool
from autokmc.core.constants import LATERAL_SHELLS_DEFAULT
from autokmc.kmc.index import _ReactionIndex
from autokmc.kmc.state import (
    _affected_surface_cliques,
    _bond_lateral_shell_members,
    _diffusion_lateral_shell_members,
    _lateral_shell_members,
)
from autokmc.reactions.adsorption import (
    AdsorptionReaction,
    get_applicable_reaction_for_member,
)
from autokmc.reactions.bond import (
    BondReaction,
    get_applicable_bond_reaction_for_member,
)
from autokmc.reactions.diffusion import (
    DiffusionReaction,
    get_applicable_diffusion_for_member,
)
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.identity import (
    SiteId,
    SiteMemberId,
    member_identifier,
    site_identifier,
)
from autokmc.sites.stability.adsorption import _surface_bfs_shells
from autokmc.utils.telemetry import increment, instrument, set_gauge
from autokmc.utils.optimizers import DEFAULT_OPTIMIZER

IndexedReaction: TypeAlias = AdsorptionReaction | DiffusionReaction | BondReaction
_RESERVED_CHANNEL_KWARGS = (
    "calculation_cache_root",
    "calculation_cache_lookup_enabled",
    "free_energy_options",
    "vib_cache_root",
)


def _print_new_state_batch(
    *,
    adsorption: int,
    diffusion: int,
    bond: int,
    verbose: bool,
) -> None:
    if not verbose or (adsorption + diffusion + bond) == 0:
        return
    parts: list[str] = []
    if adsorption:
        parts.append(f"{adsorption} adsorption stability state(s)")
    if diffusion:
        parts.append(f"{diffusion} diffusion NEB state(s)")
    if bond:
        parts.append(f"{bond} bond-reaction NEB state(s)")
    print(
        "[KMC]  New state batch discovered: "
        + ", ".join(parts)
        + " evaluated."
    )


def _member_surface_nodes(
    graph: nx.Graph,
    cliques: set,
) -> set[int]:
    return {
        int(surface)
        for clique in cliques
        for surface in clique
        if surface in graph
    }


def _fallback_adsorption_members(
    graph: nx.Graph,
    sites: list[AdsorbateSite],
    affected_cliques: set,
    max_n_shells: int,
    active_ids: set[SiteId] | None,
) -> list[tuple[AdsorbateSite, int]]:
    if not sites:
        return []
    seed = frozenset(surface for clique in affected_cliques for surface in clique)
    expanded = _surface_bfs_shells(graph, seed, max_n_shells)
    result: list[tuple[AdsorbateSite, int]] = []
    for site in sites:
        if active_ids is not None and site_identifier(site) not in active_ids:
            continue
        for member_index in range(len(site.member_node_ids)):
            member_cliques = _affected_surface_cliques(
                graph,
                site,
                member_index,
            )
            if _member_surface_nodes(graph, member_cliques) & expanded:
                result.append((site, member_index))
    return result


def _fallback_diffusion_members(
    graph: nx.Graph,
    sites: list[DiffusionSite],
    affected_cliques: set,
    max_n_shells: int,
    active_ids: set[SiteId] | None,
) -> list[tuple[DiffusionSite, int]]:
    if not sites:
        return []
    seed = frozenset(surface for clique in affected_cliques for surface in clique)
    expanded = _surface_bfs_shells(graph, seed, max_n_shells)
    result: list[tuple[DiffusionSite, int]] = []
    for site in sites:
        if active_ids is not None and site_identifier(site) not in active_ids:
            continue
        for member_index in range(len(site.member_node_ids)):
            site_a, member_a, site_b, member_b = site.members[member_index]
            cliques = (
                _affected_surface_cliques(graph, site_a, member_a)
                | _affected_surface_cliques(graph, site_b, member_b)
            )
            if _member_surface_nodes(graph, cliques) & expanded:
                result.append((site, member_index))
    return result


def _fallback_bond_members(
    graph: nx.Graph,
    sites: list[BondReactionSite],
    affected_cliques: set,
    max_n_shells: int,
    active_ids: set[SiteId] | None,
) -> list[tuple[BondReactionSite, int]]:
    if not sites:
        return []
    seed = frozenset(surface for clique in affected_cliques for surface in clique)
    expanded = _surface_bfs_shells(graph, seed, max_n_shells)
    result: list[tuple[BondReactionSite, int]] = []
    for site in sites:
        if active_ids is not None and site_identifier(site) not in active_ids:
            continue
        for member_index in range(len(site.member_node_ids)):
            cliques_a, cliques_b, cliques_c = site._member_cliques[member_index]
            cliques = set((*cliques_a, *cliques_b, *cliques_c))
            if _member_surface_nodes(graph, cliques) & expanded:
                result.append((site, member_index))
    return result


def _unique_members(members: list[tuple[Any, int]]) -> list[tuple[Any, int]]:
    seen: set[SiteMemberId] = set()
    result: list[tuple[Any, int]] = []
    for site, member_index in members:
        identifier = member_identifier(site, member_index)
        if identifier in seen:
            continue
        seen.add(identifier)
        result.append((site, int(member_index)))
    return result


def _members_by_site(
    members: list[tuple[Any, int]],
) -> list[tuple[Any, list[int]]]:
    grouped: dict[SiteId, tuple[Any, list[int]]] = {}
    for site, member_index in members:
        identifier = site_identifier(site)
        entry = grouped.get(identifier)
        if entry is None:
            entry = (site, [])
            grouped[identifier] = entry
        entry[1].append(int(member_index))
    return list(grouped.values())


MemberEvaluator = Callable[[Any, int], IndexedReaction | None]
MemberResult: TypeAlias = tuple[Any, int, IndexedReaction | None]


def _evaluate_member_batch(
    channel: str,
    site: Any,
    member_indices: list[int],
    evaluator: MemberEvaluator,
) -> tuple[str, list[MemberResult], int]:
    results: list[MemberResult] = []
    new_states = 0
    for member_index in member_indices:
        before = len(site.lateral_classes)
        reaction = evaluator(site, member_index)
        new_states += max(0, len(site.lateral_classes) - before)
        results.append((site, member_index, reaction))

    refreshed = set(member_indices)
    cached = [
        candidate
        for candidate in (getattr(site, "applicable_reactions", None) or [])
        if int(candidate.member_index) not in refreshed
    ]
    cached.extend(
        reaction
        for _, _, reaction in results
        if reaction is not None
    )
    cached.sort(key=lambda candidate: int(candidate.member_index))
    site.applicable_reactions = cached
    return channel, results, new_states


def _run_member_jobs(
    jobs: list[Callable[[], tuple[str, list[MemberResult], int]]],
    calculator,
    *,
    allow_parallel: bool,
) -> list[tuple[str, list[MemberResult], int]]:
    if (
        allow_parallel
        and isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and len(jobs) > 1
    ):
        futures = [
            calculator.submit(copy_context().run, job)
            for job in jobs
        ]
        return calculator.gather(futures)
    return [job() for job in jobs]


@instrument("kmc.recompute")
def recompute_affected_sites(
    graph: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    affected_cliques: set,
    calculator,
    gas_energies: dict[str, float],
    *,
    temperature: float,
    transmission_coefficient: float,
    frozen_indices: list[int] | None,
    fmax: float,
    max_steps: int,
    verbose: bool,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict | None = None,
    max_n_shells: int = 1,
    rxn_index: _ReactionIndex | None = None,
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
    bond_sites: list[BondReactionSite] | None = None,
    bond_kwargs: dict | None = None,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    gas_g: dict[str, float] | None = None,
    partial_pressures: dict[str, float] | None = None,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
) -> tuple[list[IndexedReaction], list[DiffusionSite]]:
    """Refresh every reaction whose lateral environment may have changed."""
    increment("kmc.recompute.affected_cliques", len(affected_cliques))
    set_gauge("kmc.recompute.last_affected_cliques", len(affected_cliques))
    if not affected_cliques:
        return [], []

    adsorption_ids = (
        rxn_index._adsorbate_ids if rxn_index is not None else None
    )
    adsorption_members: list[tuple[AdsorbateSite, int]] | None
    adsorption_members = _lateral_shell_members(
        graph,
        affected_cliques,
        adsorption_ids,
        max_n_shells,
    )
    if adsorption_members is None:
        adsorption_members = _fallback_adsorption_members(
            graph,
            adsorbate_sites,
            affected_cliques,
            max_n_shells,
            adsorption_ids,
        )
    adsorption_members = _unique_members(adsorption_members)

    diffusion_ids = (
        rxn_index._diffusion_ids if rxn_index is not None else None
    )
    diffusion_members: list[tuple[DiffusionSite, int]] | None
    diffusion_members = _diffusion_lateral_shell_members(
        graph,
        affected_cliques,
        diffusion_ids,
        max_n_shells,
    )
    if diffusion_members is None:
        diffusion_members = _fallback_diffusion_members(
            graph,
            list(diffusion_sites or []),
            affected_cliques,
            max_n_shells,
            diffusion_ids,
        )
    diffusion_members = _unique_members(diffusion_members)

    bond_ids = rxn_index._bond_ids if rxn_index is not None else None
    bond_members: list[tuple[BondReactionSite, int]] | None
    bond_members = _bond_lateral_shell_members(
        graph,
        affected_cliques,
        bond_ids,
        max_n_shells,
    )
    if bond_members is None:
        bond_members = _fallback_bond_members(
            graph,
            list(bond_sites or []),
            affected_cliques,
            max_n_shells,
            bond_ids,
        )
    bond_members = _unique_members(bond_members)

    n_affected_members = (
        len(adsorption_members)
        + len(diffusion_members)
        + len(bond_members)
    )
    set_gauge("kmc.recompute.last_affected_members", n_affected_members)
    increment("kmc.recompute.affected_members", n_affected_members)

    def evaluate_adsorption(
        site: AdsorbateSite,
        member_index: int,
    ) -> AdsorptionReaction | None:
        return get_applicable_reaction_for_member(
            graph,
            site,
            member_index,
            calculator,
            gas_energies,
            temperature=temperature,
            transmission_coefficient=transmission_coefficient,
            frozen_indices=frozen_indices,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            verbose=verbose,
            lateral_interactions=lateral_interactions,
            lateral_shells=lateral_shells,
            gas_g=gas_g,
            partial_pressures=partial_pressures,
            free_energy_options=free_energy_options,
            vib_cache_root=vib_cache_root,
            calculation_cache_root=calculation_cache_root,
            calculation_cache_lookup_enabled=calculation_cache_lookup_enabled,
            update_site_cache=False,
        )

    diffusion_options = dict(diffusion_kwargs or {})
    for key in _RESERVED_CHANNEL_KWARGS:
        diffusion_options.pop(key, None)

    def evaluate_diffusion(
        site: DiffusionSite,
        member_index: int,
    ) -> DiffusionReaction | None:
        return get_applicable_diffusion_for_member(
            graph,
            site,
            member_index,
            calculator,
            temperature=temperature,
            transmission_coefficient=transmission_coefficient,
            frozen_indices=frozen_indices,
            verbose=verbose,
            lateral_interactions=lateral_interactions,
            lateral_shells=lateral_shells,
            free_energy_options=free_energy_options,
            vib_cache_root=vib_cache_root,
            calculation_cache_root=calculation_cache_root,
            calculation_cache_lookup_enabled=calculation_cache_lookup_enabled,
            update_site_cache=False,
            **diffusion_options,
        )

    bond_options = dict(bond_kwargs or {})
    for key in _RESERVED_CHANNEL_KWARGS:
        bond_options.pop(key, None)

    def evaluate_bond(
        site: BondReactionSite,
        member_index: int,
    ) -> BondReaction | None:
        return get_applicable_bond_reaction_for_member(
            graph,
            site,
            member_index,
            calculator,
            temperature=temperature,
            transmission_coefficient=transmission_coefficient,
            frozen_indices=frozen_indices,
            lateral_interactions=lateral_interactions,
            lateral_shells=lateral_shells,
            verbose=verbose,
            calculation_cache_root=calculation_cache_root,
            calculation_cache_lookup_enabled=calculation_cache_lookup_enabled,
            free_energy_options=free_energy_options,
            vib_cache_root=vib_cache_root,
            update_site_cache=False,
            **bond_options,
        )

    adsorption_jobs: list[
        Callable[[], tuple[str, list[MemberResult], int]]
    ] = []
    adsorption_jobs.extend(
        partial(
            _evaluate_member_batch,
            "adsorption",
            site,
            member_indices,
            evaluate_adsorption,
        )
        for site, member_indices in _members_by_site(adsorption_members)
    )
    transition_jobs: list[
        Callable[[], tuple[str, list[MemberResult], int]]
    ] = []
    transition_jobs.extend(
        partial(
            _evaluate_member_batch,
            "diffusion",
            site,
            member_indices,
            evaluate_diffusion,
        )
        for site, member_indices in _members_by_site(diffusion_members)
    )
    transition_jobs.extend(
        partial(
            _evaluate_member_batch,
            "bond",
            site,
            member_indices,
            evaluate_bond,
        )
        for site, member_indices in _members_by_site(bond_members)
    )

    updated: list[IndexedReaction] = []
    new_states = {"adsorption": 0, "diffusion": 0, "bond": 0}
    affected_diffusion_sites: dict[SiteId, DiffusionSite] = {}
    # Independent adsorption batches use the calculator pool unless a new
    # state can launch internally-parallel vibrations.  Diffusion and bond
    # batches remain outer-serial because their NEB/vibration kernels schedule
    # across the same pool; nesting another pool-sized executor would
    # oversubscribe threads and GPUs.  All segment-tree writes stay serial.
    job_results = _run_member_jobs(
        adsorption_jobs,
        calculator,
        allow_parallel=not bool(
            getattr(free_energy_options, "enabled", False)
        ),
    )
    job_results.extend(
        _run_member_jobs(
            transition_jobs,
            calculator,
            allow_parallel=False,
        )
    )
    for channel, member_results, discovered in job_results:
        new_states[channel] += discovered
        for site, member_index, reaction in member_results:
            if channel == "diffusion":
                affected_diffusion_sites[site_identifier(site)] = site
            if rxn_index is not None:
                rxn_index.install(reaction, site, member_index)
            if reaction is not None:
                updated.append(reaction)

    _print_new_state_batch(
        adsorption=new_states["adsorption"],
        diffusion=new_states["diffusion"],
        bond=new_states["bond"],
        verbose=verbose,
    )
    return updated, list(affected_diffusion_sites.values())


__all__ = ["recompute_affected_sites"]
