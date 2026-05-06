"""
autokmc.kmc.engine
======================
Kinetic Monte Carlo (BKL / Gillespie) engine for autokmc.

This module mirrors the role of :mod:`disreax_kmc.simulation` but operates on
the on-the-fly :class:`~autokmc.reactions.adsorption.AdsorptionReaction` objects produced from
materialised :class:`~autokmc.sites.adsorbate.AdsorbateSite`'s.

The standard KMC loop is::

    1. Compute the list of all applicable Reactions and their rates.
    2. Total rate Q = Σ rᵢ.
    3. Sample τ from Exp(Q):     τ = ln(1/u) / Q,   u ~ U(0,1].
    4. Pick reaction i with probability rᵢ / Q.
    5. Execute the reaction (toggle occupied flag on G).
    6. Update only the affected sites (those touching the changed clique).
    7. Loop.

Public API
----------
* :func:`total_rate`
* :func:`sample_tau`
* :func:`choose_reaction`
* :func:`execute_reaction`
* :func:`run_kmc_steps`
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import networkx as nx

from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.bond import BondReactionSite
from autokmc.species.reactant import Reactant
from autokmc.reactions.adsorption import (
    AdsorptionReaction as Reaction,  # alias keeps existing type hints valid
    _build_gas_energy_lookup,
    _build_gas_g_lookup,
    _build_partial_pressure_lookup,
    compute_all_reactions,
    get_applicable_reactions,
    fast_reaction_for_member,
    is_clique_blocked,
)
from autokmc.reactions.rates import KB_EV, H_EV_S, DEFAULT_TRANSMISSION_COEFFICIENT
from autokmc.reactions.diffusion import (
    DiffusionReaction,
    compute_all_diffusions,
    get_applicable_diffusions,
    is_diffusion_applicable,
)
from autokmc.reactions.bond import (
    BondReaction,
    compute_all_bond_reactions,
    get_applicable_bond_reactions,
    is_bond_applicable,
)
from autokmc.kmc.expansion import expand_bond_sites_after_event
from autokmc.kmc.execute import execute_reaction
from autokmc.kmc.index import _ReactionIndex
from autokmc.kmc.sampling import choose_reaction, sample_tau, total_rate
from autokmc.kmc.state import (
    _affected_surface_cliques,
    _bond_lateral_shell_members,
    _diffusion_lateral_shell_members,
    _lateral_shell_members,
)
from autokmc.sites.stability.adsorption import (
    _surface_bfs_shells,
    check_adsorbate_site_lateral,
)
from autokmc.sites.stability.diffusion import check_diffusion_site_lateral
from autokmc.sites.stability.bond import check_bond_site_lateral
from autokmc.io.calculators import CalculatorPool
from autokmc.utils.logging import get_logger
from autokmc.core.constants import LATERAL_SHELLS_DEFAULT

_log = get_logger(__name__)


@contextmanager
def _acquire_calculator(calculator):
    if isinstance(calculator, CalculatorPool):
        with calculator.acquire() as calc:
            yield calc
    else:
        yield calculator


# Sampling, indexing, state mutation, and event execution live in focused
# sibling modules. The engine keeps only the orchestration and incremental
# recomputation loop.


def _final_occupancy_by_species(adsorbate_sites: list[AdsorbateSite]) -> dict[str, int]:
    """Return final occupancy keyed by species plus iso-class."""
    out: dict[str, int] = {}
    for site in adsorbate_sites:
        smiles = str(getattr(site, "reactant", "unknown") or "unknown")
        iso = int(getattr(site, "iso_class"))
        key = f"{smiles}:iso{iso}"
        out[key] = out.get(key, 0) + int(getattr(site, "_n_occupied", 0))
    return out


def _count_new_adsorption_states(
    G: nx.Graph,
    sites: Iterable[AdsorbateSite],
    *,
    lateral_interactions: bool,
) -> int:
    count = 0
    for site in sites:
        for m_idx in range(len(site.member_node_ids)):
            if is_clique_blocked(G, site, m_idx):
                continue
            before = len(site.lateral_classes)
            try:
                check_adsorbate_site_lateral(
                    G,
                    site,
                    m_idx,
                    ignore_lateral=not lateral_interactions,
                )
            except (ValueError, IndexError):
                continue
            count += max(0, len(site.lateral_classes) - before)
    return count


def _count_new_diffusion_states(
    G: nx.Graph,
    sites: Iterable[DiffusionSite],
    *,
    lateral_interactions: bool,
) -> int:
    count = 0
    for ds in sites:
        for m_idx in range(len(ds.member_node_ids)):
            applicable, direction = is_diffusion_applicable(G, ds, m_idx)
            if not applicable or direction is None:
                continue
            before = len(ds.lateral_classes)
            try:
                check_diffusion_site_lateral(
                    G,
                    ds,
                    m_idx,
                    ignore_lateral=not lateral_interactions,
                )
            except (ValueError, IndexError):
                continue
            count += max(0, len(ds.lateral_classes) - before)
    return count


def _count_new_bond_states(
    G: nx.Graph,
    sites: Iterable[BondReactionSite],
    *,
    lateral_interactions: bool,
) -> int:
    count = 0
    for brs in sites:
        for m_idx in range(len(brs.member_node_ids)):
            applicable, direction = is_bond_applicable(G, brs, m_idx)
            if not applicable or direction is None:
                continue
            before = len(brs.lateral_classes)
            try:
                check_bond_site_lateral(
                    G,
                    brs,
                    m_idx,
                    ignore_lateral=not lateral_interactions,
                )
            except (ValueError, IndexError):
                continue
            count += max(0, len(brs.lateral_classes) - before)
    return count


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
        + ". Running the required calculations now."
    )


def _recompute_affected_sites(
    G: nx.Graph,
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
    max_n_shells: int = 1,
    rxn_index: _ReactionIndex | None = None,
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
    bond_sites: list[BondReactionSite] | None = None,
    bond_kwargs: dict | None = None,
    lateral_interactions: bool = True,
    gas_g: dict[str, float] | None = None,
    partial_pressures: dict[str, float] | None = None,
    free_energy_options=None,
    vib_cache_root: str | None = None,
) -> tuple[list[Reaction | DiffusionReaction | BondReaction], list[DiffusionSite]]:
    """Recompute lateral classes and rates for every member in the lateral
    shell of the just-toggled member.

    **Why the full lateral shell, not just clique-touching members:**
    A toggle at surface clique C changes the occupancy leaf of every member
    whose lateral ego-graph reaches C.  That ego-graph extends up to
    ``n_shells_settled`` surface hops from the member's seed clique.
    Conversely, all members whose seed clique is within ``n_shells`` hops of
    C are potentially affected.  Using only clique-touching members (as the
    previous implementation did) misses members 1+ hops away, silently
    freezing their lateral class at a stale value.

    **Why the slow path (get_applicable_reactions) is always used here:**
    The fast path (``fast_reaction_for_member``) looks up a cached lateral
    class without calling ``check_adsorbate_site_lateral``.  After a
    neighbouring member is toggled the cached lateral class is stale —
    re-classification via ``check_adsorbate_site_lateral`` is required to
    detect whether the member maps to an existing or a new lateral class.
    When the new class has already been ML-relaxed the call is O(GraphMatcher);
    only truly novel occupancy patterns trigger an ML evaluation.
    """
    if not affected_cliques:
        return [], []

    active_site_ids: set[int] | None = (
        set(rxn_index._adsorbate_ids) if rxn_index is not None else None
    )

    # Find all laterally-affected members via the n_shells surface expansion.
    affected = _lateral_shell_members(
        G, affected_cliques, active_site_ids, max_n_shells,
    )

    if not affected:
        # Reverse index not yet built (e.g. called before find_adsorbate_sites
        # ran).  Fall back to scanning the active sites list and checking
        # clique-overlap only — corrected output is still better than nothing.
        updated_reactions: list[Reaction | DiffusionReaction | BondReaction] = []
        for site in adsorbate_sites:
            for m_idx in range(len(site.member_node_ids)):
                cliques_m = _affected_surface_cliques(G, site, m_idx)
                if cliques_m & affected_cliques:
                    with _acquire_calculator(calculator) as calc:
                        rxns = get_applicable_reactions(
                            G, site, calc, gas_energies,
                            temperature              = temperature,
                            transmission_coefficient = transmission_coefficient,
                            frozen_indices           = frozen_indices,
                            fmax                     = fmax,
                            max_steps                = max_steps,
                            verbose                  = verbose,
                            lateral_interactions     = lateral_interactions,
                            gas_g                    = gas_g,
                            partial_pressures        = partial_pressures,
                            free_energy_options      = free_energy_options,
                            vib_cache_root           = vib_cache_root,
                        )
                    updated_reactions.extend(rxns)
                    if rxn_index is not None:
                        rxn_index.install_site(site, rxns)
                    break
        return updated_reactions, []

    # Deduplicate at the site level — get_applicable_reactions refreshes
    # ALL members of a site in one call, so calling it once per site is
    # both correct and cheaper than one call per (site, member) pair.
    sites_to_update: dict[int, AdsorbateSite] = {}
    for site, _ in affected:
        sites_to_update[id(site)] = site

    ds_to_update: dict[int, DiffusionSite] = {}
    brs_to_update: dict[int, BondReactionSite] = {}

    # ── Diffusion sites: same lateral-shell expansion, separate index ─────
    if diffusion_sites:
        active_ds_ids: set[int] | None = (
            set(rxn_index._diffusion_ids) if rxn_index is not None else None
        )
        affected_ds = _diffusion_lateral_shell_members(
            G, affected_cliques, active_ds_ids, max_n_shells,
        )
        for ds, _ in affected_ds:
            ds_to_update[id(ds)] = ds

        # Fallback when the reverse index isn't built — recompute every
        # diffusion site whose endpoints fall within ``max_n_shells`` surface
        # hops of the affected cliques.  This mirrors the lateral-shell
        # expansion of the fast path so we don't miss neighbour members at
        # shell ≥ 1.
        if not ds_to_update:
            seed = frozenset(s for clq in affected_cliques for s in clq)
            expanded: frozenset = _surface_bfs_shells(G, seed, max_n_shells)
            for ds in diffusion_sites:
                for m_idx in range(len(ds.member_node_ids)):
                    site_a, m_a, site_b, m_b = ds.members[m_idx]
                    cliques_pair = (
                        _affected_surface_cliques(G, site_a, m_a)
                        | _affected_surface_cliques(G, site_b, m_b)
                    )
                    pair_surface = {s for clq in cliques_pair for s in clq}
                    if pair_surface & expanded:
                        ds_to_update[id(ds)] = ds
                        # Break after finding the first qualifying member: we
                        # only need to register `ds` once in `ds_to_update`.
                        # `get_applicable_diffusions` / `install_site` will
                        # then refresh ALL members of the site, so stopping
                        # early here is correct, not incomplete.
                        break

    # ── Bond reactions: same lateral-shell expansion, separate index ──────
    if bond_sites:
        active_brs_ids: set[int] | None = (
            set(rxn_index._bond_ids) if rxn_index is not None else None
        )
        affected_brs = _bond_lateral_shell_members(
            G, affected_cliques, active_brs_ids, max_n_shells,
        )
        for brs, _ in affected_brs:
            brs_to_update[id(brs)] = brs

        # Fallback when reverse index is empty: scan every bond site whose
        # placements fall within ``max_n_shells`` surface hops of the
        # affected cliques.
        if not brs_to_update:
            seed = frozenset(s for clq in affected_cliques for s in clq)
            expanded: frozenset = _surface_bfs_shells(G, seed, max_n_shells)
            for brs in bond_sites:
                hit = False
                for m_idx in range(len(brs.member_node_ids)):
                    cliques_a, cliques_b, cliques_c = brs._member_cliques[m_idx]
                    triple_surface = {
                        s
                        for clq in (*cliques_a, *cliques_b, *cliques_c)
                        for s in clq
                    }
                    if triple_surface & expanded:
                        hit = True
                        break
                if hit:
                    brs_to_update[id(brs)] = brs

    if verbose:
        new_adsorption_states = _count_new_adsorption_states(
            G,
            sites_to_update.values(),
            lateral_interactions=lateral_interactions,
        )
        new_diffusion_states = _count_new_diffusion_states(
            G,
            ds_to_update.values(),
            lateral_interactions=lateral_interactions,
        )
        new_bond_states = _count_new_bond_states(
            G,
            brs_to_update.values(),
            lateral_interactions=lateral_interactions,
        )
        _print_new_state_batch(
            adsorption=new_adsorption_states,
            diffusion=new_diffusion_states,
            bond=new_bond_states,
            verbose=verbose,
        )

    updated_reactions: list[Reaction | DiffusionReaction | BondReaction] = []
    for site in sites_to_update.values():
        with _acquire_calculator(calculator) as calc:
            rxns = get_applicable_reactions(
                G, site, calc, gas_energies,
                temperature              = temperature,
                transmission_coefficient = transmission_coefficient,
                frozen_indices           = frozen_indices,
                fmax                     = fmax,
                max_steps                = max_steps,
                verbose                  = verbose,
                lateral_interactions     = lateral_interactions,
                gas_g                    = gas_g,
                partial_pressures        = partial_pressures,
                free_energy_options      = free_energy_options,
                vib_cache_root           = vib_cache_root,
            )
        updated_reactions.extend(rxns)
        if rxn_index is not None:
            rxn_index.install_site(site, rxns)

    dkwargs = dict(diffusion_kwargs or {})
    for ds in ds_to_update.values():
        with _acquire_calculator(calculator) as calc:
            rxns = get_applicable_diffusions(
                G, ds, calc,
                temperature              = temperature,
                transmission_coefficient = transmission_coefficient,
                frozen_indices           = frozen_indices,
                verbose                  = verbose,
                lateral_interactions     = lateral_interactions,
                free_energy_options      = free_energy_options,
                vib_cache_root           = vib_cache_root,
                **dkwargs,
            )
        updated_reactions.extend(rxns)
        if rxn_index is not None:
            rxn_index.install_site(ds, rxns)

    bkwargs = dict(bond_kwargs or {})
    for brs in brs_to_update.values():
        with _acquire_calculator(calculator) as calc:
            rxns = get_applicable_bond_reactions(
                G, brs, calc,
                temperature              = temperature,
                transmission_coefficient = transmission_coefficient,
                frozen_indices           = frozen_indices,
                lateral_interactions     = lateral_interactions,
                verbose                  = verbose,
                **bkwargs,
            )
        updated_reactions.extend(rxns)
        if rxn_index is not None:
            rxn_index.install_site(brs, rxns)
    return updated_reactions, list(ds_to_update.values())



# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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
    log_every: int = 1,
    verbose: bool = True,
    lateral_interactions: bool = True,
    # ── Diffusion (NEB) channel ────────────────────────────────────────────
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
    # ── Bond-changing (A + B ⇌ C) channel ──────────────────────────────────
    bond_sites: list[BondReactionSite] | None = None,
    bond_kwargs: dict | None = None,
    bond_growth_kwargs: dict | None = None,
    # ── Free-energy / thermochemistry hooks ───────────────────────────────
    free_energy_options=None,
    vib_cache_root: str | None = None,
    # ── Optional persistence hooks (autokmc.io.persistence) ──────────────────
    reaction_writer=None,
    trajectory_writer=None,
    summary_collector=None,
    checkpoint_writer=None,
    initial_step: int = 0,
    initial_time_s: float = 0.0,
) -> dict:
    """Run a KMC simulation in place on *G* for up to ``n_steps`` events.

    Parameters
    ----------
    G : nx.Graph
        Surface + adsorbate graph (already populated by
        :func:`~autokmc.sites.adsorbate.find_adsorbate_sites`).
    adsorbate_sites : list[AdsorbateSite]
        Stable iso-classes returned by :func:`find_adsorbate_sites` (typically
        with ``prune_stable_only=True``).
    calculator
        ASE-compatible calculator used by
        :func:`~autokmc.sites.stability.adsorption.check_site_stability`.
    reactants
        :class:`~autokmc.species.reactant.Reactant`, iterable thereof, or
        ``{smiles: gas_energy_eV}`` mapping.  ``Reactant.energy`` is taken as
        the gas-phase reference for adsorption / desorption ΔE.
    temperature : float
        Simulation temperature in Kelvin.
    n_steps : int
        Maximum number of KMC events.
    transmission_coefficient : float
        Eyring transmission coefficient κ (dimensionless, default 1.0).
        The Eyring rate is ``κ · (k_B T / h) · exp(−Ea / kT)``.
    frozen_indices, fmax, max_steps :
        Forwarded to :func:`check_site_stability` for new lateral classes.
    rng :
        ``random.Random``, ``numpy.random.Generator``, an integer seed, or
        ``None`` (fresh ``Generator``).
    log_every : int
        Print a log line every N steps.  Set to 0/None to silence per-step output.
    verbose : bool
    lateral_interactions : bool
        When ``False``, neighbouring occupied adsorbate nodes are **excluded**
        from the lateral ego-graph for both adsorption and diffusion reactions.
        Every member therefore always maps to a single bare lat0, so only one
        ML relaxation is performed per iso-class (no coverage-dependent
        re-classification).  The lateral shell update is also skipped on every
        step — only clique-touching members are recomputed.  Default ``True``.
    diffusion_sites : list[DiffusionSite] | None
        Diffusion (hop) iso-classes from
        :func:`autokmc.sites.diffusion.find_diffusion_sites`.  When
        non-empty the diffusion channel is enabled: every applicable
        :class:`DiffusionReaction` is added to the segment-tree alongside
        the adsorption / desorption reactions.
    diffusion_kwargs : dict | None
        Keyword arguments forwarded to
        :func:`autokmc.reactions.diffusion.get_applicable_diffusions` (NEB knobs:
        ``fmax``, ``max_steps``, ``n_images``, ``climb``, ``spring_k``,
        ``interpolation``, ``persist_neb_path``).
    bond_sites : list[BondReactionSite] | None
        Bond-changing iso-classes from
        :func:`autokmc.sites.bond.find_bond_sites`.  When non-empty
        the bond channel is enabled: every applicable
        :class:`~autokmc.reactions.bond.BondReaction` (couple ``A+B→C`` and
        dissoc ``C→A+B``) is added to the segment-tree alongside
        adsorption / desorption / diffusion reactions.  CI-NEB barriers
        are computed lazily per (iso, lateral) class.
    bond_kwargs : dict | None
        Keyword arguments forwarded to
        :func:`autokmc.reactions.bond.get_applicable_bond_reactions` (NEB knobs,
        same names as *diffusion_kwargs*).
    bond_growth_kwargs : dict | None
        Keyword arguments forwarded to
        :func:`autokmc.kmc.expansion.expand_bond_sites_after_event`,
        called after every coupling event so newly-formed product
        species extend the bond network on the fly.
    reaction_writer : autokmc.io.persistence.ReactionWriter | None
        Optional writer.  When supplied, every executed event is persisted
        as one JSON line + sidecar XYZ snapshots of the pre/post Atoms.
    trajectory_writer : autokmc.io.persistence.TrajectoryWriter | None
        Optional writer.  Initial state plus every Nth state is written to
        an ASE ``.traj`` (cadence configured on the writer itself).
    summary_collector : autokmc.io.persistence.ReactionSummary | None
        Optional aggregator.  When supplied, ``.add()`` is called for each
        executed event so per-reaction-type statistics are available at the
        end of the run via ``summary_collector.to_dict()``.

    Returns
    -------
    dict
        Summary with keys: ``time``, ``steps_executed``, ``history``,
        ``reaction_counts``, ``final_occupancy``.
        ``history`` is a list of ``(step, time, kind, iso_class,
        member_index, lateral_class, delta_e, barrier, rate)`` tuples.
    """
    # ── RNG normalisation ─────────────────────────────────────────────────
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    gas_energies = _build_gas_energy_lookup(reactants)
    gas_g_lookup = _build_gas_g_lookup(reactants)
    pressures    = _build_partial_pressure_lookup(reactants)

    # ── Initial reaction list ─────────────────────────────────────────────
    if verbose:
        print(
            f"\n[KMC] ═══════════════════════════════════════════════════════"
        )
        print(
            f"[KMC]  KMC initialisation"
        )
        print(
            f"[KMC]  T = {temperature} K   "
            f"κ = {transmission_coefficient}   "
            f"ν_Eyring = {transmission_coefficient * KB_EV * temperature / H_EV_S:.3e} Hz"
        )
        print(
            f"[KMC]  max steps = {n_steps}   "
            f"lateral interactions = {lateral_interactions}"
        )
        print(
            f"[KMC] ─────────────────────────────────────────────────────────"
        )
        n_members_ads = sum(len(s.member_node_ids) for s in adsorbate_sites)
        print(
            f"[KMC]  Adsorption/desorption channel"
            f"  │  {len(adsorbate_sites)} iso-class(es)"
            f"  │  {n_members_ads} total member(s)"
        )

    compute_all_reactions(
        G, adsorbate_sites, calculator, gas_energies,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
        frozen_indices           = frozen_indices,
        fmax                     = fmax,
        max_steps                = max_steps,
        verbose                  = verbose,
        lateral_interactions     = lateral_interactions,
        gas_g                    = gas_g_lookup,
        partial_pressures        = pressures,
        free_energy_options      = free_energy_options,
        vib_cache_root           = vib_cache_root,
    )

    if verbose:
        _n_ads_rxns = sum(
            len(getattr(s, "applicable_reactions", None) or [])
            for s in adsorbate_sites
        )
        print(f"[KMC]    → {_n_ads_rxns} applicable reaction(s) after initial sweep")

    # ── Diffusion channel: initial NEB sweep ──────────────────────────────
    diffusion_sites = list(diffusion_sites or [])
    diffusion_kwargs = dict(diffusion_kwargs or {})
    if verbose:
        if diffusion_sites:
            n_diff_members = sum(len(ds.member_node_ids) for ds in diffusion_sites)
            print(
                f"[KMC]  Diffusion channel"
                f"  │  {len(diffusion_sites)} iso-class(es)"
                f"  │  {n_diff_members} total member(s)"
                f"  │  NEB lazily per new lateral class"
            )
        else:
            print(f"[KMC]  Diffusion channel  │  disabled (no DiffusionSite(s) supplied)")
    if diffusion_sites:
        compute_all_diffusions(
            G, diffusion_sites, calculator,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            verbose                  = verbose,
            lateral_interactions     = lateral_interactions,
            free_energy_options      = free_energy_options,
            vib_cache_root           = vib_cache_root,
            **diffusion_kwargs,
        )
        if verbose:
            _n_diff_rxns = sum(
                len(getattr(ds, "applicable_reactions", None) or [])
                for ds in diffusion_sites
            )
            print(f"[KMC]    → {_n_diff_rxns} applicable diffusion(s) after initial sweep")

    # ── Bond channel: initial NEB sweep ───────────────────────────────────
    bond_sites = list(bond_sites or [])
    bond_kwargs = dict(bond_kwargs or {})
    bond_growth_kwargs = dict(bond_growth_kwargs or {})
    if verbose:
        if bond_sites:
            n_bond_members = sum(len(brs.member_node_ids) for brs in bond_sites)
            print(
                f"[KMC]  Bond-reaction channel"
                f"  │  {len(bond_sites)} iso-class(es)"
                f"  │  {n_bond_members} total member(s)"
                f"  │  NEB lazily per new lateral class"
            )
        else:
            print(f"[KMC]  Bond-reaction channel  │  disabled (no BondReactionSite(s) supplied)")
    if bond_sites:
        compute_all_bond_reactions(
            G, bond_sites, calculator,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            verbose                  = verbose,
            lateral_interactions     = lateral_interactions,
            **bond_kwargs,
        )
        if verbose:
            _n_bond_rxns = sum(
                len(getattr(brs, "applicable_reactions", None) or [])
                for brs in bond_sites
            )
            print(f"[KMC]    → {_n_bond_rxns} applicable bond reaction(s) after initial sweep")

    # ── Build the segment-tree rate index (suggestion.MD #3) ─────────────
    # Each (site, member) pair gets a fixed leaf position so the per-step
    # cost of sampling a reaction and updating affected leaves is O(log R)
    # instead of the O(R) ``np.cumsum`` + ``np.searchsorted`` rebuild.
    rxn_index = _ReactionIndex(adsorbate_sites, diffusion_sites, bond_sites)
    for site in adsorbate_sites:
        rxns = getattr(site, "applicable_reactions", None) or []
        rxn_index.install_site(site, rxns)
    for ds in diffusion_sites:
        rxns = getattr(ds, "applicable_reactions", None) or []
        rxn_index.install_site(ds, rxns)
    for brs in bond_sites:
        rxns = getattr(brs, "applicable_reactions", None) or []
        rxn_index.install_site(brs, rxns)

    if verbose:
        _n_active = sum(1 for r in rxn_index.reactions if r is not None)
        print(
            f"[KMC]  Segment-tree index built"
            f"  │  {rxn_index.n_total} leaves"
            f"  │  {_n_active} active reaction(s)"
            f"  │  Q₀ = {rxn_index.total_rate():.3e} Hz"
        )
        print(f"[KMC] ═══════════════════════════════════════════════════════\n")

    def _persist_reactions(
        reactions: Iterable[Reaction | DiffusionReaction | BondReaction],
        *,
        step_for_discovery: int,
    ) -> None:
        """Materialise per-(iso, lat) folders for supplied applicable reactions."""
        if reaction_writer is None:
            return
        for rxn in reactions:
            if rxn is None:
                continue
            try:
                reaction_writer.ensure_reaction(
                    rxn,
                    step=step_for_discovery,
                    gas_energies=gas_energies,
                    gas_free_energies=gas_g_lookup,
                )
            except Exception as exc:  # pragma: no cover
                _log.warning("reaction_writer.ensure_reaction failed: %s", exc)

    def _persist_invalid_diffusion_sites(sites: Iterable[DiffusionSite]) -> None:
        """Write invalid diffusion records for supplied sites only."""
        if reaction_writer is None:
            return
        for ds in sites:
            for lc in ds.lateral_classes:
                if lc.stable is False:
                    try:
                        reaction_writer.write_invalid_diffusion(ds, lc)
                    except Exception as exc:  # pragma: no cover
                        _log.warning(
                            "reaction_writer.write_invalid_diffusion failed: %s", exc
                        )

    start_step = int(initial_step or 0)
    _persist_reactions(
        (rxn for rxn in rxn_index.reactions if rxn is not None),
        step_for_discovery=start_step,
    )
    _persist_invalid_diffusion_sites(diffusion_sites)

    # Initialise the graph-level occupancy counter (suggestion.MD #9).
    if "n_occupied" not in G.graph:
        G.graph["n_occupied"] = sum(
            1 for s in adsorbate_sites
            for nids in s.member_node_ids
            if any(nid in G and G.nodes[nid].get("occupied", False) for nid in nids)
        )
    # Per-site counters seeded from the current graph state.
    for s in adsorbate_sites:
        if not hasattr(s, "_n_occupied"):
            s._n_occupied = sum(  # type: ignore[attr-defined]
                1 for nids in s.member_node_ids
                if any(nid in G and G.nodes[nid].get("occupied", False) for nid in nids)
            )

    # max_n_shells for the lateral-shell expansion in _recompute_affected_sites.
    # This must match the BFS depth used by check_adsorbate_site_lateral so that
    # the incremental trigger radius is consistent with the lateral environment
    # actually being evaluated.  Both are driven by LATERAL_SHELLS_DEFAULT.
    # When lateral_interactions=False no lateral re-classification is needed —
    # set max_n_shells=0 so only clique-touching members are recomputed.
    max_n_shells: int = LATERAL_SHELLS_DEFAULT if lateral_interactions else 0

    history: list[tuple] = []
    reaction_counts: dict[str, int] = {
        "adsorption": 0, "desorption": 0, "diffusion": 0,
        "bond": 0, "bond_couple": 0, "bond_dissoc": 0,
    }
    current_time = float(initial_time_s or 0.0)
    steps_executed = 0

    # Optional: trajectory writer (extxyz append) needs an atoms snapshot.
    # The reaction writer does NOT — it pulls atoms straight from
    # ``reaction.lateral_class.atoms_{occupied,unoccupied}`` (the relaxed
    # structures stamped on by check_site_stability).
    if trajectory_writer is not None:
        try:
            from autokmc.io.atoms import atoms_from_graph
            trajectory_writer.maybe_write(atoms_from_graph(G), step=start_step)
        except Exception as exc:  # pragma: no cover
            _log.warning("trajectory_writer initial frame failed: %s", exc)

    for step in range(start_step + 1, start_step + int(n_steps) + 1):
        q_total = rxn_index.total_rate()
        if q_total <= 0.0:
            if verbose:
                print(f"[KMC] Step {step}: total rate = 0 — stopping.")
            break

        # Draw two independent uniforms: one for reaction selection, one for τ.
        if isinstance(rng, np.random.Generator):
            u_pick = float(rng.random())
            u_tau  = float(rng.random())
        else:
            u_pick = rng.random()
            u_tau  = rng.random()
        if u_tau <= 0.0:
            u_tau = float(np.nextafter(0.0, 1.0))

        chosen = rxn_index.sample(u_pick)
        if chosen is None:
            if verbose:
                print(f"[KMC] Step {step}: sampler returned None — stopping.")
            break

        tau = float(np.log(1.0 / u_tau) / q_total)
        current_time += tau

        affected = execute_reaction(G, chosen)
        steps_executed += 1
        reaction_counts[chosen.kind] = reaction_counts.get(chosen.kind, 0) + 1
        if chosen.kind == "bond":
            sub = "bond_" + getattr(chosen, "direction", "couple")
            reaction_counts[sub] = reaction_counts.get(sub, 0) + 1

        # Persist the event — the reaction writer materialises the per-
        # lateral-class folder lazily on first sighting and otherwise just
        # appends a row to events.jsonl.
        if reaction_writer is not None:
            try:
                reaction_writer.record(
                    step          = step,
                    time_s        = current_time,
                    tau_s         = tau,
                    reaction      = chosen,
                    gas_energies  = gas_energies,
                    gas_free_energies = gas_g_lookup,
                )
            except Exception as exc:  # pragma: no cover
                _log.warning("reaction_writer.record failed: %s", exc)

        if summary_collector is not None:
            try:
                summary_collector.add(chosen, step=step)
            except Exception as exc:  # pragma: no cover
                _log.warning("summary_collector.add failed: %s", exc)

        history.append((
            step,
            current_time,
            chosen.kind,
            chosen.site.iso_class,
            chosen.member_index,
            chosen.lateral_class.lateral_class,
            chosen.delta_e,
            chosen.barrier,
            chosen.rate,
        ))

        if verbose and log_every and (step % log_every == 0):
            # suggestion.MD #9: O(1) read instead of O(total members) scan.
            n_occ = int(G.graph.get("n_occupied", 0))
            # Build a concise reaction label that includes bond direction /
            # SMILES when available.
            _kind = chosen.kind
            if _kind == "bond":
                _dir  = getattr(chosen, "direction", "")
                _tmpl = getattr(getattr(chosen, "site", None), "template", None)
                if _tmpl is not None:
                    if _dir == "couple":
                        _rxn_label = (
                            f"bond/couple  "
                            f"{_tmpl.smiles_a}+{_tmpl.smiles_b}→{_tmpl.smiles_c}"
                        )
                    else:
                        _rxn_label = (
                            f"bond/dissoc  "
                            f"{_tmpl.smiles_c}→{_tmpl.smiles_a}+{_tmpl.smiles_b}"
                        )
                else:
                    _rxn_label = f"bond/{_dir}"
            elif _kind == "diffusion":
                _rxn_label = f"diffusion   "
            else:
                _rxn_label = f"{_kind:<11}"
            print(
                f"[KMC] step {step:>5}  t = {current_time:.4e} s  "
                f"τ = {tau:.3e} s  Q = {q_total:.3e} Hz  "
                f"{_rxn_label}  iso={chosen.site.iso_class} "
                f"m={chosen.member_index} lat={chosen.lateral_class.lateral_class} "
                f"ΔE={chosen.delta_e:+.3f} eV  Ea={chosen.barrier:.3f} eV  "
                f"k={chosen.rate:.2e} Hz  occ={n_occ}"
            )

        # Incremental update — re-classify lateral environments for every
        # member in the n_shells surface shell around the toggled clique.
        # This includes the toggled member itself (0 hops), clique-collision
        # neighbours (0 hops), and genuine lateral neighbours (1…n_shells).
        # The slow path (get_applicable_reactions) is always used so that
        # check_adsorbate_site_lateral is called and lateral classes are
        # correctly updated.
        reactions_to_persist, invalid_diffusion_sites_to_persist = _recompute_affected_sites(
            G, adsorbate_sites, affected, calculator, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            verbose                  = verbose,
            max_n_shells             = max_n_shells,
            rxn_index                = rxn_index,
            diffusion_sites          = diffusion_sites,
            diffusion_kwargs         = diffusion_kwargs,
            bond_sites               = bond_sites,
            bond_kwargs              = bond_kwargs,
            lateral_interactions     = lateral_interactions,
            gas_g                    = gas_g_lookup,
            partial_pressures        = pressures,
            free_energy_options      = free_energy_options,
            vib_cache_root           = vib_cache_root,
        )

        # Bond events (coupling A+B→C *or* dissociation C→A+B) may introduce
        # a species whose bond, diffusion, and adsorption channels have not
        # been enumerated yet.  Expand the network on the fly so subsequent
        # KMC steps include all new reactions.
        #
        # * Coupling: product C may be genuinely new.
        # * Dissociation: fragments A and B were pre-built as leaf nodes when
        #   C entered the registry, but their own coupling/dissociation
        #   templates were never derived.  expand_bond_sites_after_event now
        #   handles both directions; the idempotency guard on
        #   reg["expanded_species"] prevents redundant work.
        if chosen.kind == "bond":
            # Announce what just fired so the user can track the chemistry.
            if verbose:
                _tmpl = getattr(getattr(chosen, "site", None), "template", None)
                _dir  = getattr(chosen, "direction", "")
                if _tmpl is not None:
                    if _dir == "couple":
                        print(
                            f"[KMC]  ⚡ Bond coupling fired: "
                            f"{_tmpl.smiles_a} + {_tmpl.smiles_b} → {_tmpl.smiles_c}"
                            f"  (iso={chosen.site.iso_class} m={chosen.member_index})"
                        )
                        print(
                            f"[KMC]  Formed {_tmpl.smiles_c!r}; checking for "
                            "new adsorbate, diffusion, and bond reaction series."
                        )
                    else:
                        print(
                            f"[KMC]  ⚡ Bond dissociation fired: "
                            f"{_tmpl.smiles_c} → {_tmpl.smiles_a} + {_tmpl.smiles_b}"
                            f"  (iso={chosen.site.iso_class} m={chosen.member_index})"
                        )
                        print(
                            f"[KMC]  Formed fragments {_tmpl.smiles_a!r} and "
                            f"{_tmpl.smiles_b!r}; checking for new reaction series."
                        )

            # Snapshot existing site identities before expansion so we can
            # detect what gets added to the registry / graph.
            _snap_ads_ids  = {id(s)  for s in adsorbate_sites}
            _snap_diff_ids = {id(ds) for ds in diffusion_sites}
            # Also snapshot registry adsorbate site IDs so we can distinguish
            # "pre-built leaf species now being activated for the first time"
            # from "brand-new species discovered by this expansion run".
            _reg_pre = G.graph.get("bond_registry", {})
            _snap_reg_ids: set[int] = {
                id(s)
                for _sl in _reg_pre.get("adsorbate_sites", {}).values()
                for s in _sl
            }
            # Species produced by this specific event — used to activate
            # pre-built leaf sites only at the moment they are first produced.
            _ftmpl = getattr(getattr(chosen, "site", None), "template", None)
            _fdir  = getattr(chosen, "direction", None)
            if _ftmpl is not None and _fdir == "couple":
                _newly_produced: set[str] = {getattr(_ftmpl, "smiles_c", "")}
            elif _ftmpl is not None and _fdir == "dissoc":
                _newly_produced = {
                    getattr(_ftmpl, "smiles_a", ""),
                    getattr(_ftmpl, "smiles_b", ""),
                }
            else:
                _newly_produced = set()
            _newly_produced.discard("")

            # Build the kwargs dict for the on-the-fly expander.  Merge
            # bond_growth_kwargs first, then override with the params known
            # only inside run_kmc_steps (frozen_indices, verbose) so that
            # caller-supplied bond_growth_kwargs can never accidentally
            # shadow them or produce duplicate-keyword TypeErrors.
            _grow_kw = dict(bond_growth_kwargs or {})
            _grow_kw["verbose"] = verbose
            _grow_kw.setdefault("frozen_indices", frozen_indices)

            try:
                with _acquire_calculator(calculator) as calc:
                    new_brs = expand_bond_sites_after_event(
                        G, chosen,
                        calculator=calc,
                        **_grow_kw,
                    )
            except Exception as exc:  # pragma: no cover
                _log.warning("expand_bond_sites_after_event failed: %s", exc)
                new_brs = []

            # ── New adsorbate iso-classes from the registry ───────────────
            # find_adsorbate_sites (called inside _ensure_species_known)
            # merges new sites into G's reverse indices via setdefault, so
            # the lateral-shell expander will find them automatically.
            # We still need to add them to the segment-tree and to the
            # adsorbate_sites list so adsorption / desorption can be sampled.
            #
            # Activation rules (both conditions must hold: not yet active):
            #   (a) Brand-new species built during this expansion — their sites
            #       were not in the registry before → activate immediately.
            #   (b) Pre-built leaf species (e.g. OH) that appear in the
            #       registry since CLI setup but were NOT in kmc_initial_sites
            #       → activate only when the event that PRODUCED them fires
            #       (i.e. their SMILES matches _newly_produced).
            _reg = G.graph.get("bond_registry", {})
            _new_ads: list[AdsorbateSite] = []
            for _smi, _sl in _reg.get("adsorbate_sites", {}).items():
                for s in _sl:
                    if id(s) in _snap_ads_ids:
                        continue  # already active in the KMC segment tree
                    if id(s) not in _snap_reg_ids:
                        # Brand-new site, built by this expansion — activate.
                        _new_ads.append(s)
                    elif _smi in _newly_produced:
                        # Pre-built leaf site, first produced right now.
                        _new_ads.append(s)
            if _new_ads:
                if verbose:
                    _new_smi_ads = sorted({s.reactant for s in _new_ads})
                    print(
                        f"[KMC]  🆕 New adsorbate iso-class(es) discovered: "
                        f"{len(_new_ads)} class(es) across "
                        f"species {_new_smi_ads}"
                    )
                    for _nsmi in _new_smi_ads:
                        _nc = sum(1 for s in _new_ads if s.reactant == _nsmi)
                        _nm = sum(
                            len(s.member_node_ids)
                            for s in _new_ads if s.reactant == _nsmi
                        )
                        print(
                            f"[KMC]    {_nsmi!r:>12}  "
                            f"{_nc} iso-class(es)  │  {_nm} member(s)"
                        )
                # Extend gas-energy / pressure lookups so that
                # get_applicable_reactions can compute rates for the new
                # species.  Newly-formed species have partial_pressure=0
                # (set in _ensure_species_known) so they contribute only
                # desorption reactions (they are not supplied from the gas
                # phase).
                _reg_species = _reg.get("species", {})
                for _s in _new_ads:
                    _r = _reg_species.get(_s.reactant) if _s.reactant else None
                    if _r is not None:
                        _smi = getattr(_r, "smiles", _s.reactant)
                        if _smi not in gas_energies:
                            _e = getattr(_r, "energy", float("nan"))
                            if not np.isnan(_e):
                                gas_energies[_smi] = float(_e)
                        if _smi not in pressures:
                            pressures[_smi] = float(
                                getattr(_r, "partial_pressure_bar", 0.0)
                            )
                        if _smi not in gas_g_lookup:
                            _g = getattr(_r, "gibbs_energy", float("nan"))
                            if not np.isnan(_g):
                                gas_g_lookup[_smi] = float(_g)
                # Initial reaction sweep for the new adsorbate iso-classes.
                if verbose:
                    _n_new_ads_members = sum(
                        len(getattr(s, "member_node_ids", ())) for s in _new_ads
                    )
                    print(
                        f"[KMC]    Running adsorption/desorption stability "
                        f"sweep for {_n_new_ads_members} new placement(s)."
                    )
                compute_all_reactions(
                    G, _new_ads, calculator, gas_energies,
                    temperature              = temperature,
                    transmission_coefficient = transmission_coefficient,
                    frozen_indices           = frozen_indices,
                    fmax                     = fmax,
                    max_steps                = max_steps,
                    verbose                  = verbose,
                    lateral_interactions     = lateral_interactions,
                    gas_g                    = gas_g_lookup,
                    partial_pressures        = pressures,
                    free_energy_options      = free_energy_options,
                    vib_cache_root           = vib_cache_root,
                )
                if verbose:
                    _n_new_ads_rxns = sum(
                        len(getattr(s, "applicable_reactions", None) or [])
                        for s in _new_ads
                    )
                    print(
                        f"[KMC]    → {_n_new_ads_rxns} applicable "
                        f"adsorption/desorption reaction(s) for new species"
                    )
                adsorbate_sites.extend(_new_ads)
                reactions_to_persist.extend(
                    rxn
                    for s in _new_ads
                    for rxn in (getattr(s, "applicable_reactions", None) or [])
                )

            # ── New diffusion iso-classes from the graph ──────────────────
            # expand_bond_sites_for_new_species stores newly-found
            # DiffusionSites in G.graph["diffusion_sites"] (when
            # find_diffusion=True is passed via bond_growth_kwargs).
            _g_diff = G.graph.get("diffusion_sites", {})
            _new_diff: list[DiffusionSite] = [
                ds
                for _dlist in (
                    _g_diff.values() if isinstance(_g_diff, dict) else []
                )
                if isinstance(_dlist, list)
                for ds in _dlist
                if id(ds) not in _snap_diff_ids
            ]
            if _new_diff:
                if verbose:
                    print(
                        f"[KMC]  🆕 New diffusion iso-class(es) discovered: "
                        f"{len(_new_diff)} site-pair(s)"
                    )
                _dkwargs = dict(diffusion_kwargs or {})
                try:
                    if verbose:
                        _n_diff_members = sum(
                            len(getattr(ds, "member_node_ids", ())) for ds in _new_diff
                        )
                        print(
                            f"[KMC]    Running diffusion applicability and NEB "
                            f"sweep for {_n_diff_members} new hop member(s)."
                        )
                    compute_all_diffusions(
                        G, _new_diff, calculator,
                        temperature              = temperature,
                        transmission_coefficient = transmission_coefficient,
                        frozen_indices           = frozen_indices,
                        verbose                  = verbose,
                        lateral_interactions     = lateral_interactions,
                        free_energy_options      = free_energy_options,
                        vib_cache_root           = vib_cache_root,
                        **_dkwargs,
                    )
                except Exception as exc:  # pragma: no cover
                    _log.warning(
                        "compute_all_diffusions for new species failed: %s", exc
                    )
                if verbose:
                    _n_new_diff_rxns = sum(
                        len(getattr(ds, "applicable_reactions", None) or [])
                        for ds in _new_diff
                    )
                    print(
                        f"[KMC]    → {_n_new_diff_rxns} applicable "
                        f"diffusion reaction(s) for new species"
                    )
                diffusion_sites.extend(_new_diff)
                reactions_to_persist.extend(
                    rxn
                    for ds in _new_diff
                    for rxn in (getattr(ds, "applicable_reactions", None) or [])
                )
                invalid_diffusion_sites_to_persist.extend(_new_diff)

            # ── Append new bond-reaction iso-classes ──────────────────────
            for brs in new_brs:
                if id(brs) in rxn_index._bond_ids:
                    continue
                bond_sites.append(brs)

            if verbose and new_brs:
                print(
                    f"[KMC]  🆕 New bond-reaction iso-class(es) enumerated: "
                    f"{len(new_brs)} iso-class(es)  "
                    f"(total bond iso-classes now: {len(bond_sites)})"
                )
                for _brs in new_brs:
                    _t = _brs.template
                    print(
                        f"[KMC]    bond_iso {_brs.iso_class:>3}  "
                        f"{_t.smiles_a!r}+{_t.smiles_b!r}⇌{_t.smiles_c!r}"
                        f"  source={_t.source}"
                        f"  members={len(_brs.member_node_ids)}"
                    )

            # ── Rebuild the segment-tree if anything was added ────────────
            if new_brs or _new_ads or _new_diff:
                # Snapshot currently-live reactions before the index rebuild
                # so no rate state is lost for existing sites.
                live_rxns = [r for r in rxn_index.reactions if r is not None]
                rxn_index = _ReactionIndex(
                    adsorbate_sites, diffusion_sites, bond_sites,
                )
                for r in live_rxns:
                    rxn_index.install(r, r.site, r.member_index)
                # Install freshly-computed reactions for new sites.
                for _s in _new_ads:
                    rxn_index.install_site(
                        _s, getattr(_s, "applicable_reactions", []) or [],
                    )
                for _ds in _new_diff:
                    rxn_index.install_site(
                        _ds, getattr(_ds, "applicable_reactions", []) or [],
                    )
                # Sweep new bond sites for initial NEB / lateral rates.
                if new_brs:
                    if verbose:
                        _n_bond_members = sum(
                            len(getattr(brs, "member_node_ids", ())) for brs in new_brs
                        )
                        print(
                            f"[KMC]    Running bond reaction stability/NEB "
                            f"sweep for {_n_bond_members} new member(s)."
                        )
                    compute_all_bond_reactions(
                        G, new_brs, calculator,
                        temperature              = temperature,
                        transmission_coefficient = transmission_coefficient,
                        frozen_indices           = frozen_indices,
                        verbose                  = verbose,
                        lateral_interactions     = lateral_interactions,
                        **bond_kwargs,
                    )
                    for brs in new_brs:
                        rxn_index.install_site(
                            brs,
                            getattr(brs, "applicable_reactions", []) or [],
                        )
                    reactions_to_persist.extend(
                        rxn
                        for brs in new_brs
                        for rxn in (getattr(brs, "applicable_reactions", None) or [])
                    )

                if verbose:
                    _n_active_now = sum(
                        1 for r in rxn_index.reactions if r is not None
                    )
                    print(
                        f"[KMC]  Segment-tree rebuilt"
                        f"  │  {rxn_index.n_total} leaves"
                        f"  │  {_n_active_now} active reaction(s)"
                        f"  │  Q = {rxn_index.total_rate():.3e} Hz"
                        f"  │  channels: "
                        f"ads={len(adsorbate_sites)}"
                        f" diff={len(diffusion_sites)}"
                        f" bond={len(bond_sites)}"
                    )

        # Persist only the reactions touched by this event's recompute/expansion.
        _persist_reactions(reactions_to_persist, step_for_discovery=step)
        _persist_invalid_diffusion_sites(invalid_diffusion_sites_to_persist)

        # Periodic trajectory dump (cadence enforced inside the writer).
        # Writes one extended-XYZ frame to the trajectory_writer's output file.
        if trajectory_writer is not None:
            try:
                from autokmc.io.atoms import atoms_from_graph
                trajectory_writer.maybe_write(atoms_from_graph(G), step=step)
            except Exception as exc:  # pragma: no cover
                _log.warning("trajectory_writer.maybe_write failed: %s", exc)

        if checkpoint_writer is not None:
            try:
                checkpoint_writer.maybe_write(
                    step=step,
                    time_s=current_time,
                    graph=G,
                    adsorbate_sites=adsorbate_sites,
                    diffusion_sites=diffusion_sites,
                    bond_sites=bond_sites,
                    reactants=(
                        list(reactants.values())
                        if isinstance(reactants, dict)
                        else (
                            list(reactants)
                            if isinstance(reactants, Iterable)
                            and not isinstance(reactants, Reactant)
                            else [reactants]
                        )
                    ),
                    frozen_indices=frozen_indices,
                    history=history,
                    reaction_counts=reaction_counts,
                )
            except Exception as exc:  # pragma: no cover
                _log.warning("checkpoint_writer.maybe_write failed: %s", exc)

    # Close trajectory writer if we own a handle.
    if trajectory_writer is not None:
        try:
            trajectory_writer.close()
        except Exception:  # pragma: no cover
            pass

    if checkpoint_writer is not None and steps_executed > 0:
        final_step = start_step + steps_executed
        try:
            checkpoint_writer.maybe_write(
                step=final_step,
                force=True,
                time_s=current_time,
                graph=G,
                adsorbate_sites=adsorbate_sites,
                diffusion_sites=diffusion_sites,
                bond_sites=bond_sites,
                reactants=(
                    list(reactants.values())
                    if isinstance(reactants, dict)
                    else (
                        list(reactants)
                        if isinstance(reactants, Iterable)
                        and not isinstance(reactants, Reactant)
                        else [reactants]
                    )
                ),
                frozen_indices=frozen_indices,
                history=history,
                reaction_counts=reaction_counts,
            )
        except Exception as exc:  # pragma: no cover
            _log.warning("final checkpoint write failed: %s", exc)

    final_occupancy = _final_occupancy_by_species(adsorbate_sites)

    summary = {
        "time"            : current_time,
        "steps_executed"  : steps_executed,
        "history"         : history,
        "reaction_counts" : reaction_counts,
        "final_occupancy" : final_occupancy,
    }

    if verbose:
        print(f"\n[KMC] Done.  steps={steps_executed}  t={current_time:.4e} s")
        print(f"[KMC] Reaction counts: {reaction_counts}")
        print(f"[KMC] Final occupancy per species/iso-class: {final_occupancy}")

    return summary
