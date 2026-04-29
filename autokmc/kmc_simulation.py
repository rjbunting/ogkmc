"""
autokmc.kmc_simulation
======================
Kinetic Monte Carlo (BKL / Gillespie) engine for autokmc.

This module mirrors the role of :mod:`disreax_kmc.simulation` but operates on
the on-the-fly :class:`~autokmc.kmc_adsorption.AdsorptionReaction` objects produced from
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
from autokmc.find_diffusion_sites import DiffusionSite
from autokmc.reactants import Reactant
from autokmc.kmc_adsorption import (
    AdsorptionReaction as Reaction,  # alias keeps existing type hints valid
    KB_EV,
    H_EV_S,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    _build_gas_energy_lookup,
    compute_all_reactions,
    get_applicable_reactions,
    fast_reaction_for_member,
)
from autokmc.kmc_diffusion import (
    DiffusionReaction,
    compute_all_diffusions,
    get_applicable_diffusions,
)
from autokmc.check_adsorbate_sites import _surface_bfs_shells
from autokmc.logging_utils import get_logger
from autokmc.constants import LATERAL_SHELLS_DEFAULT

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

    Each ``AdsorbateSite`` and each ``DiffusionSite`` gets a contiguous
    block of leaves of length ``len(site.member_node_ids)``.
    ``self.reactions[leaf]`` holds the currently-active reaction for that
    (site, member) pair (an :class:`AdsorptionReaction` or a
    :class:`DiffusionReaction`), or ``None`` when no reaction is currently
    applicable (clique-blocked, unstable, …).  The companion segment-tree
    keeps the rate column in sync so sampling and updates are both O(log R).
    """
    __slots__ = ("base", "n_total", "tree", "reactions", "site_order",                 "diffusion_site_order",
                 "_adsorbate_ids", "_diffusion_ids")

    def __init__(
        self,
        sites: list[AdsorbateSite],
        diffusion_sites: list[DiffusionSite] | None = None,
    ):
        self.site_order: list[AdsorbateSite] = list(sites)
        self.diffusion_site_order: list[DiffusionSite] = list(diffusion_sites or [])
        self.base: dict[int, int] = {}
        self._adsorbate_ids: set[int] = set()
        self._diffusion_ids: set[int] = set()
        offset = 0
        for s in self.site_order:
            self.base[id(s)] = offset
            self._adsorbate_ids.add(id(s))
            offset += len(s.member_node_ids)
        for ds in self.diffusion_site_order:
            self.base[id(ds)] = offset
            self._diffusion_ids.add(id(ds))
            offset += len(ds.member_node_ids)
        self.n_total = offset
        self.tree = _RateSegmentTree(self.n_total)
        self.reactions: list[Reaction | DiffusionReaction | None] = (
            [None] * self.n_total
        )

    def leaf_id(self, site, m_idx: int) -> int:
        return self.base[id(site)] + int(m_idx)

    def install(self, rxn, site, m_idx: int) -> None:
        i = self.leaf_id(site, m_idx)
        self.reactions[i] = rxn
        self.tree.update(i, rxn.rate if (rxn is not None and rxn.rate > 0.0) else 0.0)

    def install_site(self, site, reactions: list) -> None:
        """Refresh every leaf for *site* from a freshly-computed reaction list.

        Works for both :class:`AdsorbateSite` and :class:`DiffusionSite`.
        """
        b = self.base[id(site)]
        for k in range(len(site.member_node_ids)):
            self.reactions[b + k] = None
            self.tree.update(b + k, 0.0)
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


def execute_reaction(G: nx.Graph, reaction) -> set:
    """Apply *reaction* in place by toggling member occupancies on *G*.

    Returns the set of surface cliques touched by the executed event —
    used by the incremental rebuild in :func:`run_kmc_steps`.

    Dispatch:

    * :class:`AdsorptionReaction` (``kind ∈ {"adsorption","desorption"}``)
      — toggles one member.
    * :class:`DiffusionReaction`  (``kind == "diffusion"``)
      — vacates the source endpoint and occupies the target endpoint
      (per ``reaction.direction``); returns the union of both endpoints'
      bonded cliques.
    """
    if getattr(reaction, "kind", None) == "diffusion":
        ds: DiffusionSite = reaction.site
        site_a, m_a, site_b, m_b = ds.members[reaction.member_index]
        if reaction.direction == "a_to_b":
            src_site, src_m = site_a, m_a
            tgt_site, tgt_m = site_b, m_b
        else:
            src_site, src_m = site_b, m_b
            tgt_site, tgt_m = site_a, m_a
        cliques: set = set()
        cliques |= _affected_surface_cliques(G, src_site, src_m)
        cliques |= _affected_surface_cliques(G, tgt_site, tgt_m)
        # Vacate first, then occupy — this keeps occupied_by_clique
        # consistent if the two endpoints happen to share a (sub-) clique.
        _set_member_occupied(G, src_site, src_m, False)
        _set_member_occupied(G, tgt_site, tgt_m, True)
        return cliques

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


def _lateral_shell_members(
    G: nx.Graph,
    affected_cliques: set,
    active_site_ids: set[int] | None,
    max_n_shells: int,
) -> list[tuple[AdsorbateSite, int]]:
    """Return every (site, member) pair whose lateral ego-graph may include
    any of the toggled adsorbate nodes.

    A member M is laterally affected by a toggle at clique C when the
    minimum surface-graph distance from any atom in C to any atom in M's
    seed clique is ≤ M's ``n_shells_settled``.  We conservatively expand C
    by ``max_n_shells`` BFS hops through surface nodes, then look up all
    members bonded to any surface node in the expanded set — these are
    exactly the members whose lateral ego-graph overlaps with the shell
    around C.

    Uses ``G.graph["surface_node_to_members"]`` built by
    :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites`.

    Parameters
    ----------
    G : nx.Graph
    affected_cliques : set[frozenset]
        Bonded surface cliques of the member that was just toggled.
    active_site_ids : set[int] | None
        ``id(site)`` for every site registered in the segment-tree index.
        Pruned sites from ``G.graph["surface_node_to_members"]`` are filtered
        out.  Pass ``None`` to skip filtering.
    max_n_shells : int
        BFS expansion depth — use the maximum ``n_shells_settled`` across
        all active adsorbate sites.

    Returns
    -------
    list[tuple[AdsorbateSite, int]]
        Deduplicated list of (site, member_index) pairs to re-evaluate.
    """
    surface_node_to_members: dict | None = G.graph.get("surface_node_to_members")
    if not surface_node_to_members or not affected_cliques:
        return []

    # Seed: the union of all surface atoms in the toggled member's cliques.
    seed = frozenset(s for clq in affected_cliques for s in clq)

    # Expand through surface-only BFS to capture all nodes within n_shells.
    # _surface_bfs_shells is cached per (seed, n_shells) so repeated calls
    # within the same step are free.
    expanded: frozenset = _surface_bfs_shells(G, seed, max_n_shells)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[AdsorbateSite, int]] = []
    for surf_id in expanded:
        for site, m_idx in surface_node_to_members.get(surf_id, ()):
            if active_site_ids is not None and id(site) not in active_site_ids:
                continue
            key = (id(site), int(m_idx))
            if key in seen:
                continue
            seen.add(key)
            out.append((site, m_idx))
    return out


def _diffusion_lateral_shell_members(
    G: nx.Graph,
    affected_cliques: set,
    active_diffusion_ids: set[int] | None,
    max_n_shells: int,
) -> list[tuple[DiffusionSite, int]]:
    """Diffusion analogue of :func:`_lateral_shell_members`.

    Walks ``G.graph["diffusion_surface_node_to_members"]`` after a
    surface-only BFS expansion of ``affected_cliques`` by ``max_n_shells``
    hops.
    """
    surface_node_to_members: dict | None = G.graph.get(
        "diffusion_surface_node_to_members"
    )
    if not surface_node_to_members or not affected_cliques:
        return []

    seed = frozenset(s for clq in affected_cliques for s in clq)
    expanded: frozenset = _surface_bfs_shells(G, seed, max_n_shells)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[DiffusionSite, int]] = []
    for surf_id in expanded:
        for ds, m_idx in surface_node_to_members.get(surf_id, ()):
            if active_diffusion_ids is not None and id(ds) not in active_diffusion_ids:
                continue
            key = (id(ds), int(m_idx))
            if key in seen:
                continue
            seen.add(key)
            out.append((ds, m_idx))
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
    max_n_shells: int = 1,
    rxn_index: _ReactionIndex | None = None,
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
) -> None:
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
        return

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

    # Deduplicate at the site level — get_applicable_reactions refreshes
    # ALL members of a site in one call, so calling it once per site is
    # both correct and cheaper than one call per (site, member) pair.
    sites_to_update: dict[int, AdsorbateSite] = {}
    for site, _ in affected:
        sites_to_update[id(site)] = site

    for site in sites_to_update.values():
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

    # ── Diffusion sites: same lateral-shell expansion, separate index ─────
    if diffusion_sites:
        active_ds_ids: set[int] | None = (
            set(rxn_index._diffusion_ids) if rxn_index is not None else None
        )
        affected_ds = _diffusion_lateral_shell_members(
            G, affected_cliques, active_ds_ids, max_n_shells,
        )
        ds_to_update: dict[int, DiffusionSite] = {}
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
                        break

        dkwargs = dict(diffusion_kwargs or {})
        for ds in (ds_to_update.values() if ds_to_update else ()):
            rxns = get_applicable_diffusions(
                G, ds, calculator,
                temperature              = temperature,
                transmission_coefficient = transmission_coefficient,
                frozen_indices           = frozen_indices,
                verbose                  = verbose,
                **dkwargs,
            )
            if rxn_index is not None:
                rxn_index.install_site(ds, rxns)



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
    # ── Diffusion (NEB) channel ────────────────────────────────────────────
    diffusion_sites: list[DiffusionSite] | None = None,
    diffusion_kwargs: dict | None = None,
    # ── Optional persistence hooks (autokmc.persistence) ─────────��────────
    reaction_writer=None,
    trajectory_writer=None,
    summary_collector=None,
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
    diffusion_sites : list[DiffusionSite] | None
        Diffusion (hop) iso-classes from
        :func:`autokmc.find_diffusion_sites.find_diffusion_sites`.  When
        non-empty the diffusion channel is enabled: every applicable
        :class:`DiffusionReaction` is added to the segment-tree alongside
        the adsorption / desorption reactions.
    diffusion_kwargs : dict | None
        Keyword arguments forwarded to
        :func:`autokmc.kmc_diffusion.get_applicable_diffusions` (NEB knobs:
        ``fmax``, ``max_steps``, ``n_images``, ``climb``, ``spring_k``,
        ``interpolation``, ``persist_neb_path``).
    reaction_writer : autokmc.persistence.ReactionWriter | None
        Optional writer.  When supplied, every executed event is persisted
        as one JSON line + sidecar XYZ snapshots of the pre/post Atoms.
    trajectory_writer : autokmc.persistence.TrajectoryWriter | None
        Optional writer.  Initial state plus every Nth state is written to
        an ASE ``.traj`` (cadence configured on the writer itself).
    summary_collector : autokmc.persistence.ReactionSummary | None
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

    # ── Diffusion channel: initial NEB sweep ──────────────────────────────
    diffusion_sites = list(diffusion_sites or [])
    diffusion_kwargs = dict(diffusion_kwargs or {})
    if diffusion_sites:
        if verbose:
            print(
                f"[KMC] Initial diffusion sweep over "
                f"{len(diffusion_sites)} DiffusionSite(s) "
                f"(NEB lazily per new lateral class)…"
            )
        compute_all_diffusions(
            G, diffusion_sites, calculator,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            verbose                  = False,
            **diffusion_kwargs,
        )

    # ── Build the segment-tree rate index (suggestion.MD #3) ─────────────
    # Each (site, member) pair gets a fixed leaf position so the per-step
    # cost of sampling a reaction and updating affected leaves is O(log R)
    # instead of the O(R) ``np.cumsum`` + ``np.searchsorted`` rebuild.
    rxn_index = _ReactionIndex(adsorbate_sites, diffusion_sites)
    for site in adsorbate_sites:
        rxns = getattr(site, "applicable_reactions", None) or []
        rxn_index.install_site(site, rxns)
    for ds in diffusion_sites:
        rxns = getattr(ds, "applicable_reactions", None) or []
        rxn_index.install_site(ds, rxns)

    def _persist_all_known_reactions(step_for_discovery: int) -> None:
        """Materialise per-(iso, lat) folders for every currently-known
        applicable reaction.  Idempotent — the writer short-circuits on the
        second sighting of any (iso, lat)."""
        if reaction_writer is None:
            return
        for rxn in rxn_index.reactions:
            if rxn is None:
                continue
            try:
                reaction_writer.ensure_reaction(
                    rxn, step=step_for_discovery, gas_energies=gas_energies,
                )
            except Exception as exc:  # pragma: no cover
                _log.warning("reaction_writer.ensure_reaction failed: %s", exc)

    _persist_all_known_reactions(step_for_discovery=0)

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
    max_n_shells: int = LATERAL_SHELLS_DEFAULT

    history: list[tuple] = []
    reaction_counts: dict[str, int] = {
        "adsorption": 0, "desorption": 0, "diffusion": 0,
    }
    current_time = 0.0
    steps_executed = 0

    # Optional: trajectory writer (extxyz append) needs an atoms snapshot.
    # The reaction writer does NOT — it pulls atoms straight from
    # ``reaction.lateral_class.atoms_{occupied,unoccupied}`` (the relaxed
    # structures stamped on by check_site_stability).
    if trajectory_writer is not None:
        try:
            from autokmc.persistence import atoms_from_graph
            trajectory_writer.maybe_write(atoms_from_graph(G), step=0)
        except Exception as exc:  # pragma: no cover
            _log.warning("trajectory_writer initial frame failed: %s", exc)

    for step in range(1, int(n_steps) + 1):
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
            print(
                f"[KMC] step {step:>5}  t = {current_time:.4e} s  "
                f"τ = {tau:.3e} s  Q = {q_total:.3e} Hz  "
                f"{chosen.kind:<11} iso={chosen.site.iso_class} "
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
        _recompute_affected_sites(
            G, adsorbate_sites, affected, calculator, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            verbose                  = False,
            max_n_shells             = max_n_shells,
            rxn_index                = rxn_index,
            diffusion_sites          = diffusion_sites,
            diffusion_kwargs         = diffusion_kwargs,
        )

        # Persist every newly-discovered (iso, lat) reaction surfaced by the
        # incremental rebuild.  ``ensure_reaction`` is a no-op once a folder
        # exists, so the cost after the first few steps is just dict lookups.
        _persist_all_known_reactions(step_for_discovery=step)

        # Periodic trajectory dump (cadence enforced inside the writer).
        # Writes one extended-XYZ frame to the trajectory_writer's output file.
        if trajectory_writer is not None:
            try:
                from autokmc.persistence import atoms_from_graph
                trajectory_writer.maybe_write(atoms_from_graph(G), step=step)
            except Exception as exc:  # pragma: no cover
                _log.warning("trajectory_writer.maybe_write failed: %s", exc)

    # Close trajectory writer if we own a handle.
    if trajectory_writer is not None:
        try:
            trajectory_writer.close()
        except Exception:  # pragma: no cover
            pass

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

