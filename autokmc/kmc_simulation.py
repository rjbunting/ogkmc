"""
autokmc.kmc_simulation
======================
Kinetic Monte Carlo (BKL / Gillespie) engine for autokmc.

This module mirrors the role of :mod:`disreax_kmc.simulation` but operates on
the on-the-fly :class:`~autokmc.kmc_reactions.Reaction` objects produced from
materialised :class:`~autokmc.find_adsorbate_sites.AdsorbateSite`'s.

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
from typing import Iterable

import numpy as np
import networkx as nx

from autokmc.find_adsorbate_sites import AdsorbateSite
from autokmc.reactants import Reactant
from autokmc.kmc_reactions import (
    Reaction,
    KB_EV,
    H_EV_S,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    _build_gas_energy_lookup,
    compute_all_reactions,
    get_applicable_reactions,
    fast_reaction_for_member,
)
from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Segment-tree rate sampler (suggestion.MD #3)
# ---------------------------------------------------------------------------

class _RateSegmentTree:
    """Sum-segment tree over a fixed-size rate vector.

    Each leaf holds a non-negative rate; internal nodes hold the sum of
    their subtree.  Supports

    * ``update(i, rate)`` — O(log n)
    * ``sample(u)``       — O(log n), returns leaf i with prob rate[i]/Q
    * ``total``           — O(1)

    Replaces the per-step ``np.cumsum`` + ``np.searchsorted`` over the full
    flat reaction list (which was O(R) per step even though only a handful
    of leaves change between steps).  See suggestion.MD #3.
    """
    __slots__ = ("_n", "_size", "_tree")

    def __init__(self, n: int):
        self._n = int(n)
        size = 1
        while size < max(1, self._n):
            size *= 2
        self._size = size
        # 1-indexed implicit binary tree; indices 1 … 2*size-1.
        self._tree = np.zeros(2 * size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def __len__(self) -> int:
        return self._n

    def update(self, i: int, value: float) -> None:
        """Set leaf *i* to *value* (clipped at 0) and propagate sums."""
        if i < 0 or i >= self._n:
            raise IndexError(f"leaf index {i} out of range [0, {self._n})")
        v = float(value)
        if v < 0.0:
            v = 0.0
        pos = self._size + i
        self._tree[pos] = v
        pos //= 2
        while pos:
            self._tree[pos] = self._tree[2 * pos] + self._tree[2 * pos + 1]
            pos //= 2

    def sample(self, u: float) -> int:
        """Return the leaf id whose prefix sum first exceeds ``u·total``."""
        total = self._tree[1]
        if total <= 0.0:
            return -1
        target = float(u) * float(total)
        pos = 1
        while pos < self._size:
            left = self._tree[2 * pos]
            if target < left:
                pos = 2 * pos
            else:
                target -= left
                pos = 2 * pos + 1
        return pos - self._size


class _ReactionIndex:
    """Flat (site, member) → leaf-id mapping plus the segment-tree of rates.

    Each ``AdsorbateSite`` gets a contiguous block of leaves of length
    ``len(site.member_node_ids)``.  ``self.reactions[leaf]`` holds the
    currently-active :class:`Reaction` for that (site, member) pair, or
    ``None`` when the member has no applicable reaction (clique-blocked,
    unstable, …).  The companion segment-tree keeps the rate column in
    sync so sampling and updates are both O(log R).
    """
    __slots__ = ("base", "n_total", "tree", "reactions", "site_order")

    def __init__(self, sites: list[AdsorbateSite]):
        self.site_order: list[AdsorbateSite] = list(sites)
        self.base: dict[int, int] = {}
        offset = 0
        for s in self.site_order:
            self.base[id(s)] = offset
            offset += len(s.member_node_ids)
        self.n_total = offset
        self.tree = _RateSegmentTree(self.n_total)
        self.reactions: list[Reaction | None] = [None] * self.n_total

    def leaf_id(self, site: AdsorbateSite, m_idx: int) -> int:
        return self.base[id(site)] + int(m_idx)

    def install(self, rxn: Reaction | None,
                site: AdsorbateSite, m_idx: int) -> None:
        i = self.leaf_id(site, m_idx)
        self.reactions[i] = rxn
        self.tree.update(i, rxn.rate if (rxn is not None and rxn.rate > 0.0) else 0.0)

    def install_site(self, site: AdsorbateSite,
                     reactions: list[Reaction]) -> None:
        """Refresh every leaf for *site* from a freshly-computed reaction list."""
        # Clear all leaves of the site first.
        b = self.base[id(site)]
        for k in range(len(site.member_node_ids)):
            self.reactions[b + k] = None
            self.tree.update(b + k, 0.0)
        # Then install whatever new reactions exist.
        for r in reactions:
            self.install(r, r.site, r.member_index)

    def total_rate(self) -> float:
        return self.tree.total

    def sample(self, u: float) -> Reaction | None:
        i = self.tree.sample(u)
        if i < 0:
            return None
        return self.reactions[i]


# ---------------------------------------------------------------------------
# Rate / sampling helpers
# ---------------------------------------------------------------------------

def total_rate(reactions: Iterable[Reaction]) -> float:
    """Sum of rates over a reaction iterable (negative rates clipped to 0)."""
    rates = np.fromiter(
        (r.rate if r.rate > 0.0 else 0.0 for r in reactions),
        dtype=np.float64,
    )
    return float(rates.sum()) if rates.size else 0.0


def sample_tau(q_total: float, rng: random.Random | np.random.Generator) -> float:
    """KMC time increment τ ~ Exp(q_total): ``τ = ln(1/u) / Q``."""
    if q_total <= 0.0:
        return float("inf")
    if isinstance(rng, np.random.Generator):
        u = float(rng.random())
    else:
        u = rng.random()
    if u <= 0.0:
        u = float(np.nextafter(0.0, 1.0))
    return float(np.log(1.0 / u) / q_total)


def choose_reaction(
    reactions: list[Reaction],
    rng: random.Random | np.random.Generator,
) -> tuple[Reaction | None, int | None, float]:
    """Pick one Reaction with probability ∝ rate.

    Returns ``(reaction, index, total_rate)``; all-None when no reaction is
    available.
    """
    if not reactions:
        return None, None, 0.0
    rates = np.fromiter(
        (r.rate if r.rate > 0.0 else 0.0 for r in reactions),
        dtype=np.float64,
    )
    total = float(rates.sum())
    if total <= 0.0:
        return None, None, 0.0

    if isinstance(rng, np.random.Generator):
        u = float(rng.random())
    else:
        u = rng.random()

    cum    = np.cumsum(rates)
    target = u * total
    idx    = int(np.searchsorted(cum, target))
    if idx >= len(reactions):
        idx = len(reactions) - 1
    return reactions[idx], idx, total


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _set_member_occupied(
    G: nx.Graph, site: AdsorbateSite, member_index: int, value: bool,
) -> None:
    """Toggle the occupancy of one member and update graph-level counters.

    Maintains:

    * ``G.graph["occupied_by_clique"]`` — per-clique set of currently-occupied
      adsorbate node ids (suggestion.MD #4 — used by ``is_clique_blocked``
      for an O(1) collision check).
    * ``G.graph["n_occupied"]``         — running total of occupied members
      (suggestion.MD #9 — turns the per-step KMC log line from O(N_members)
      into O(1)).
    * Per-site counter ``site._n_occupied`` (same purpose, per iso-class).
    """
    occupied_by_clique = G.graph.get("occupied_by_clique")
    if not hasattr(site, "_n_occupied"):
        site._n_occupied = 0  # type: ignore[attr-defined]

    new_state = bool(value)

    # The "is the member currently occupied?" judgement uses the same
    # any-node-set rule as ``AdsorbateSite._member_is_occupied`` so we stay
    # consistent with the rest of the code base.
    was_occupied = any(
        nid in G and G.nodes[nid].get("occupied", False)
        for nid in site.member_node_ids[member_index]
    )

    for nid in site.member_node_ids[member_index]:
        if nid not in G:
            continue
        G.nodes[nid]["occupied"] = new_state
        if occupied_by_clique is not None:
            clq = G.nodes[nid].get("clique")
            if clq is not None:  # already a frozenset (#8)
                bucket = occupied_by_clique.setdefault(clq, set())
                if new_state:
                    bucket.add(nid)
                else:
                    bucket.discard(nid)

    if was_occupied != new_state:
        delta = +1 if new_state else -1
        site._n_occupied = max(0, int(site._n_occupied) + delta)  # type: ignore[attr-defined]
        G.graph["n_occupied"] = max(0, int(G.graph.get("n_occupied", 0)) + delta)


def _affected_surface_cliques(
    G: nx.Graph, site: AdsorbateSite, member_index: int,
) -> set:
    """Surface cliques bonded by the executed member (used for incremental update).

    Returns a set of frozensets — uses the cached per-member tuple
    populated by :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`
    (suggestion.MD #4 / #8) when available, falling back to the legacy
    per-node scan otherwise.
    """
    member_cliques = getattr(site, "_member_cliques", None)
    if member_cliques is not None:
        return set(member_cliques[member_index])

    # Legacy fallback (no reverse index).
    out: set = set()
    for nid in site.member_node_ids[member_index]:
        if nid not in G:
            continue
        clq = G.nodes[nid].get("clique")
        if clq is not None:
            out.add(clq)  # already a frozenset (#8)
    return out


def execute_reaction(G: nx.Graph, reaction: Reaction) -> set:
    """Apply *reaction* in place by toggling the member's occupancy on *G*.

    Returns the set of surface cliques touched by the executed member —
    useful for the incremental rebuild in :func:`run_kmc_steps`.
    """
    new_state = (reaction.kind == "adsorption")
    cliques = _affected_surface_cliques(G, reaction.site, reaction.member_index)
    _set_member_occupied(G, reaction.site, reaction.member_index, new_state)
    return cliques


def _affected_members_for_cliques(
    G: nx.Graph,
    affected_cliques: set,
) -> list[tuple[AdsorbateSite, int]]:
    """Return the unique (site, member) pairs touching any of *affected_cliques*.

    Uses the ``G.graph["clique_to_members"]`` reverse index built in
    :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`
    (suggestion.MD #4) — O(affected_members) instead of O(sites × members).
    """
    clique_to_members: dict | None = G.graph.get("clique_to_members")
    if clique_to_members is None or not affected_cliques:
        return []
    seen: set[tuple[int, int]] = set()
    out: list[tuple[AdsorbateSite, int]] = []
    for clq in affected_cliques:
        for site, m_idx in clique_to_members.get(clq, ()):
            key = (id(site), int(m_idx))
            if key in seen:
                continue
            seen.add(key)
            out.append((site, m_idx))
    return out


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
    rxn_index: _ReactionIndex | None = None,
) -> None:
    """Recompute applicable reactions for every member touching one of
    *affected_cliques* and write the new rates into *rxn_index*.

    Uses the suggestion.MD #4 reverse index to find affected members in
    O(affected_members) instead of scanning every site.

    For each affected member we attempt the suggestion.MD #12 fast path:
    if the member already has a cached lateral class on the parent site
    (populated on a previous full ``get_applicable_reactions`` pass), we
    just look up the cached forward / reverse rate via
    :func:`~autokmc.kmc_reactions.fast_reaction_for_member` — no
    GraphMatcher, no ML calls.  This is correct because clique-collision
    members can never be lateral neighbours of the toggled member (they
    share the same surface clique, so the toggled member sits in their
    *seed* clique, not their lateral leaf set), so the lateral-class
    identity for clique-touching members is invariant under a flip.

    Members that fail the fast path (no cached lateral class yet, or no
    cached lateral class for this specific m_idx) fall back to the slow
    per-site ``get_applicable_reactions`` path so the cache is populated
    for subsequent flips.
    """
    if not affected_cliques:
        return

    affected = _affected_members_for_cliques(G, affected_cliques)
    if not affected:
        # No reverse index available — fall back to the legacy per-site loop.
        for site in adsorbate_sites:
            for m_idx in range(len(site.member_node_ids)):
                cliques_m = _affected_surface_cliques(G, site, m_idx)
                if cliques_m & affected_cliques:
                    rxns = get_applicable_reactions(
                        G, site, calculator, gas_energies,
                        temperature              = temperature,
                        transmission_coefficient = transmission_coefficient,
                        frozen_indices           = frozen_indices,
                        fmax                     = fmax,
                        max_steps                = max_steps,
                        verbose                  = verbose,
                    )
                    if rxn_index is not None:
                        rxn_index.install_site(site, rxns)
                    break
        return

    # Group affected members by site so that any site needing a slow-path
    # rebuild is only rebuilt once.
    sites_needing_slow_pass: dict[int, AdsorbateSite] = {}
    for site, m_idx in affected:
        member_lc: dict | None = getattr(site, "_member_lc", None)
        if member_lc is None or m_idx not in member_lc:
            sites_needing_slow_pass[id(site)] = site

    # Slow path first: rebuild every site that has even one cache-miss
    # member.  This refreshes ``site.applicable_reactions`` and
    # ``site._member_lc`` for *all* its members in one go, which keeps the
    # segment-tree consistent with the cached reaction list.
    for site in sites_needing_slow_pass.values():
        rxns = get_applicable_reactions(
            G, site, calculator, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            verbose                  = verbose,
        )
        if rxn_index is not None:
            rxn_index.install_site(site, rxns)

    # Fast path for the remaining (site, member) pairs.
    if rxn_index is None:
        return
    for site, m_idx in affected:
        if id(site) in sites_needing_slow_pass:
            continue  # already handled in the slow pass above
        new_rxn = fast_reaction_for_member(
            G, site, m_idx, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
        )
        rxn_index.install(new_rxn, site, m_idx)


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
) -> dict:
    """Run a KMC simulation in place on *G* for up to ``n_steps`` events.

    Parameters
    ----------
    G : nx.Graph
        Surface + adsorbate graph (already populated by
        :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`).
    adsorbate_sites : list[AdsorbateSite]
        Stable iso-classes returned by :func:`find_adsorbate_sites` (typically
        with ``prune_stable_only=True``).
    calculator
        ASE-compatible calculator used by
        :func:`~autokmc.check_adsorbate_sites.check_site_stability`.
    reactants
        :class:`~autokmc.reactants.Reactant`, iterable thereof, or
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

    # ── Initial reaction list ─────────────────────────────────────────────
    if verbose:
        print(
            f"[KMC] T = {temperature} K   "
            f"κ = {transmission_coefficient}   "
            f"ν_Eyring = {transmission_coefficient * KB_EV * temperature / H_EV_S:.3e} Hz   "
            f"max steps = {n_steps}"
        )
        print("[KMC] Building initial reaction list…")

    compute_all_reactions(
        G, adsorbate_sites, calculator, gas_energies,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
        frozen_indices           = frozen_indices,
        fmax                     = fmax,
        max_steps                = max_steps,
        verbose                  = False,
    )

    # ── Build the segment-tree rate index (suggestion.MD #3) ─────────────
    # Each (site, member) pair gets a fixed leaf position so the per-step
    # cost of sampling a reaction and updating affected leaves is O(log R)
    # instead of the O(R) ``np.cumsum`` + ``np.searchsorted`` rebuild.
    rxn_index = _ReactionIndex(adsorbate_sites)
    for site in adsorbate_sites:
        rxns = getattr(site, "applicable_reactions", None) or []
        rxn_index.install_site(site, rxns)

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

    history: list[tuple] = []
    reaction_counts: dict[str, int] = {"adsorption": 0, "desorption": 0}
    current_time = 0.0
    steps_executed = 0

    for step in range(1, int(n_steps) + 1):
        q_total = rxn_index.total_rate()
        if q_total <= 0.0:
            if verbose:
                print(f"[KMC] Step {step}: total rate = 0 — stopping.")
            break

        # Draw the uniform once and reuse for both τ and the sampler so
        # that the segment-tree path is fully O(log R) per step.
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
            print(
                f"[KMC] step {step:>5}  t = {current_time:.4e} s  "
                f"τ = {tau:.3e} s  Q = {q_total:.3e} Hz  "
                f"{chosen.kind:<11} iso={chosen.site.iso_class} "
                f"m={chosen.member_index} lat={chosen.lateral_class.lateral_class} "
                f"ΔE={chosen.delta_e:+.3f} eV  Ea={chosen.barrier:.3f} eV  "
                f"k={chosen.rate:.2e} Hz  occ={n_occ}"
            )

        # Incremental update — recompute only members touching the changed
        # cliques (suggestion.MD #4) and prefer the cached fast path
        # (suggestion.MD #12) over re-running GraphMatcher.
        # First refresh the toggled member itself.
        toggled_rxn = fast_reaction_for_member(
            G, chosen.site, chosen.member_index, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
        )
        rxn_index.install(toggled_rxn, chosen.site, chosen.member_index)

        # Then propagate to the rest of the affected (clique-collision) members.
        _recompute_affected_sites(
            G, adsorbate_sites, affected, calculator, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            verbose                  = False,
            rxn_index                = rxn_index,
        )

    final_occupancy = {
        site.iso_class: int(getattr(site, "_n_occupied", 0))
        for site in adsorbate_sites
    }

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
        print(f"[KMC] Final occupancy per iso-class: {final_occupancy}")

    return summary

