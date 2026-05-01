"""
autokmc.kmc_bond
================
Build KMC **bond-changing** reactions for a graph of materialised
:class:`~autokmc.find_bond_reactions.BondReactionSite`'s.

Each bond reaction is a reversible three-state event::

    A + B  ⇌  C

with three placements (``placement_A``, ``placement_B``, ``placement_C``)
sitting on the surface graph (see :mod:`autokmc.find_bond_reactions` for
the geometric constraints).

Applicability rule
------------------
For every member of every :class:`BondReactionSite`, exactly one of the
two firing directions may be active at any instant:

* **coupling** ``A + B → C`` — both *A* and *B* are currently occupied
  *and* *C* is empty.
* **dissociation** ``C → A + B`` — *C* is currently occupied *and* both
  *A* and *B* are empty.

In addition, the **clique-collision guard** rejects the event if any
*other* (i.e. third-party) occupied adsorbate already claims a surface
clique that the reaction needs to occupy when it fires:

* coupling: the cliques of *C* must not be claimed by anything other
  than *A* or *B* themselves (they will vacate as the reaction fires).
* dissociation: the cliques of *A* and *B* must not be claimed by
  anything other than *C* itself.

Energetics
----------
Identical scheme to :mod:`autokmc.kmc_diffusion`.  The first time a
member maps to a new lateral class the on-the-fly stability /  NEB
calculation populates ``lc.energy_ab``, ``lc.energy_c``, ``lc.energy_ts``
on the :class:`BondReactionLateral`.  The KMC rate uses an Eyring
prefactor with the standard ``EA_MIN`` floor::

    e_ts_eff = max(e_ts, max(e_ab, e_c) + EA_MIN)
    Ea(coupling)      = max(EA_MIN, e_ts_eff − e_ab)
    Ea(dissociation)  = max(EA_MIN, e_ts_eff − e_c)
    k                 = κ · (k_B T / h) · exp(−Ea / kT)

The *raw* TS energy is persisted unchanged by
:class:`autokmc.persistence.ReactionWriter`; the floor is only applied
when computing the rate consumed by the KMC loop.

.. note::
    The on-the-fly lateral classifier and the NEB / endpoint relaxation
    machinery for bond reactions are intentionally **not** built yet —
    this module exposes the data flow (applicability, energetics cache,
    Eyring rate) but defers the expensive geometry calls so that the
    user can plug them in incrementally.  ``get_applicable_bond_reactions``
    will silently skip any member whose lateral class has not been
    populated with energies yet.

Public API
----------
* :class:`BondReaction`                    — one applicable bond event.
* :func:`is_bond_applicable`               — XOR-occupancy + clique guard.
* :func:`get_applicable_bond_reactions`    — per-site enumeration.
* :func:`compute_all_bond_reactions`       — full enumeration over all sites.
* :func:`fast_bond_reaction_for_member`    — single-member fast path
  (assumes cached lateral class is still valid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx

from .find_bond_sites import BondReactionSite, BondReactionLateral
from .kmc_adsorption import EA_MIN, DEFAULT_TRANSMISSION_COEFFICIENT, _eyring_prefactor
from .check_bond_sites import (
    check_bond_site_lateral,
    check_bond_site_stability,
    BondStabilityError,
)
from .constants import (
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_N_IMAGES,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
    NL_MULT_DEFAULT,
)
from .logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# BondReaction dataclass
# ---------------------------------------------------------------------------

@dataclass
class BondReaction:
    """One applicable KMC bond-change event.

    Attributes
    ----------
    kind : str
        Always ``"bond"``.
    direction : str
        ``"couple"`` (A + B → C) or ``"dissoc"`` (C → A + B).  Determined
        by which of the two configurations is currently occupied.
    site : BondReactionSite
        Parent iso-class.
    member_index : int
        Index into ``site.members`` (a concrete triple of placements).
    lateral_class : BondReactionLateral
        Lateral environment under which the energetics were computed
        and cached.
    delta_e : float
        ``E_target − E_source`` (eV) for the firing direction.
    barrier : float
        KMC activation energy ``max(EA_MIN, e_ts_eff − E_source)`` (eV).
        The unfloored raw barrier is persisted in the reaction folder.
    rate : float
        Eyring rate ``κ · (k_B T / h) · exp(−barrier / kT)`` (Hz).
    """
    kind          : str
    direction     : str            # "couple" | "dissoc"
    site          : BondReactionSite
    member_index  : int
    lateral_class : BondReactionLateral
    delta_e       : float
    barrier       : float
    rate          : float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _placement_occupied(G: nx.Graph, node_ids: Iterable[int]) -> bool:
    """True iff any node in *node_ids* is currently flagged ``occupied``."""
    for nid in node_ids:
        if nid in G and G.nodes[nid].get("occupied", False):
            return True
    return False


def _cliques_blocked(
    G: nx.Graph,
    cliques: Iterable[frozenset],
    excluded_node_ids: frozenset,
) -> bool:
    """Clique-collision guard.

    Returns True if any clique in *cliques* is currently claimed by an
    occupied adsorbate node *other* than those in *excluded_node_ids*.
    Uses ``G.graph["occupied_by_clique"]`` for an O(k) check; falls back
    to a graph scan when the reverse index is missing.
    """
    occupied_by_clique = G.graph.get("occupied_by_clique")
    if occupied_by_clique is not None:
        for clq in cliques:
            occ_nodes = occupied_by_clique.get(clq)
            if occ_nodes and not occ_nodes.issubset(excluded_node_ids):
                return True
        return False

    # Fallback path — slow but functional.
    target_cliques = set(cliques)
    if not target_cliques:
        return False
    for n, d in G.nodes(data=True):
        if n in excluded_node_ids:
            continue
        if d.get("type") != "adsorbate":
            continue
        if not d.get("occupied", False):
            continue
        clq = d.get("clique")
        if clq is not None and clq in target_cliques:
            return True
    return False


def is_bond_applicable(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
) -> tuple[bool, str | None]:
    """Decide whether ``(brs, member_index)`` can fire right now.

    Returns ``(applicable, direction)`` with ``direction`` one of
    ``"couple"``, ``"dissoc"``, or ``None``.  Applicability requires:

    1. Exactly one of the two configurations *(A occupied AND B occupied
       AND C empty)* / *(C occupied AND A empty AND B empty)* is true.
    2. The cliques that the firing direction needs to occupy are not
       claimed by any *third-party* adsorbate.
    """
    # Read node IDs live from the AdsorbateSite objects so that we always see
    # the current graph-node ids, even after _materialise_adsorbate_nodes has
    # re-run for a species and invalidated the cached copies in
    # brs.member_node_ids.
    site_a, m_a, site_b, m_b, site_c, m_c = brs.members[member_index]
    a_nids = list(site_a.member_node_ids[m_a])
    b_nids = list(site_b.member_node_ids[m_b])
    c_nids = list(site_c.member_node_ids[m_c])
    a_occ = _placement_occupied(G, a_nids)
    b_occ = _placement_occupied(G, b_nids)
    c_occ = _placement_occupied(G, c_nids)

    couple_state = (a_occ and b_occ and not c_occ)
    dissoc_state = (c_occ and not a_occ and not b_occ)

    if couple_state == dissoc_state:
        # Either both true (impossible — A+B+C can't all be in the right state)
        # or neither — wrong occupancy pattern for this reaction.
        return False, None

    cliques_a, cliques_b, cliques_c = brs._member_cliques[member_index]
    a_set = frozenset(a_nids)
    b_set = frozenset(b_nids)
    c_set = frozenset(c_nids)

    if couple_state:
        # Direction: A + B → C.  C will become occupied — its cliques
        # must be free except for what A or B already (themselves) hold.
        if _cliques_blocked(G, cliques_c, a_set | b_set | c_set):
            return False, None
        return True, "couple"
    else:
        # Direction: C → A + B.  A and B will become occupied — their
        # cliques must be free except for what C already holds.
        if _cliques_blocked(G, cliques_a, a_set | b_set | c_set):
            return False, None
        if _cliques_blocked(G, cliques_b, a_set | b_set | c_set):
            return False, None
        return True, "dissoc"


# ---------------------------------------------------------------------------
# Energetics
# ---------------------------------------------------------------------------

def _bond_energetics_cached(
    lc: BondReactionLateral,
    direction: str,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float, float]:
    """Return ``(delta_e, barrier_kmc, rate)`` for one firing direction.

    Mirrors :func:`autokmc.kmc_diffusion._diffusion_energetics_cached`:
    the effective TS energy is raised to sit at least ``EA_MIN`` above
    the higher of the two endpoints so that detailed balance is preserved
    after the KMC floor is applied.
    """
    key = (
        round(float(temperature),              9),
        round(float(transmission_coefficient), 9),
        str(direction),
    )
    cache: dict | None = getattr(lc, "_rate_cache", None)
    if cache is None:
        cache = {}
        lc._rate_cache = cache  # type: ignore[attr-defined]
    hit = cache.get(key)
    if hit is not None:
        return hit

    e_ab = float(lc.energy_ab)   # type: ignore[arg-type]
    e_c  = float(lc.energy_c)    # type: ignore[arg-type]
    e_ts = float(lc.energy_ts)   # type: ignore[arg-type]

    e_ts_eff = max(e_ts, max(e_ab, e_c) + EA_MIN)
    prefactor, kT = _eyring_prefactor(temperature, transmission_coefficient)

    def _make(direction_: str) -> tuple[float, float, float]:
        if direction_ == "couple":
            de     = e_c  - e_ab
            ea_kmc = max(EA_MIN, e_ts_eff - e_ab)
        else:  # "dissoc"
            de     = e_ab - e_c
            ea_kmc = max(EA_MIN, e_ts_eff - e_c)
        rate = float(prefactor * np.exp(-ea_kmc / kT))
        return float(de), float(ea_kmc), rate

    out_couple = _make("couple")
    out_dissoc = _make("dissoc")
    cache[key[:-1] + ("couple",)] = out_couple
    cache[key[:-1] + ("dissoc",)] = out_dissoc

    return out_couple if direction == "couple" else out_dissoc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_applicable_bond_reactions(
    G: nx.Graph,
    brs: BondReactionSite,
    calculator=None,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = NEB_FMAX,
    max_steps: int = NEB_MAX_STEPS,
    n_images: int = NEB_N_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    lateral_interactions: bool = True,
    verbose: bool = False,
) -> list[BondReaction]:
    """Enumerate currently-applicable bond events for one BondReactionSite.

    Mirrors :func:`autokmc.kmc_diffusion.get_applicable_diffusions` —
    classifies each applicable member into a
    :class:`BondReactionLateral` and triggers a CI-NEB stability run the
    first time a new lateral class is seen.  When *calculator* is None
    the NEB step is skipped and members whose lateral class has not yet
    been populated with energies are silently skipped.
    """
    if not hasattr(brs, "_member_lc"):
        brs._member_lc = {}  # type: ignore[attr-defined]

    reactions: list[BondReaction] = []

    for m_idx in range(len(brs.member_node_ids)):
        applicable, direction = is_bond_applicable(G, brs, m_idx)
        if not applicable or direction is None:
            brs._member_lc.pop(m_idx, None)  # type: ignore[attr-defined]
            continue

        # 1. Lateral classification (cheap if seen before).
        try:
            lc = check_bond_site_lateral(
                G, brs, m_idx,
                ignore_lateral=not lateral_interactions,
            )
        except (ValueError, IndexError) as exc:
            if verbose:
                print(
                    f"  ⚠  bond_iso={brs.iso_class} m={m_idx}: "
                    f"lateral check skipped ({exc})"
                )
            continue

        # 2. NEB / endpoint relaxation (only for new lateral classes).
        if lc.stable is None:
            if calculator is None:
                # Defer — caller didn't supply a calculator, so we cannot
                # populate energies.  Skip silently as documented.
                continue
            try:
                check_bond_site_stability(
                    G, brs, m_idx, lc, calculator,
                    frozen_indices   = frozen_indices,
                    fmax             = fmax,
                    max_steps        = max_steps,
                    n_images         = n_images,
                    climb            = climb,
                    spring_k         = spring_k,
                    interpolation    = interpolation,
                    nl_mult          = nl_mult,
                    persist_neb_path = persist_neb_path,
                    verbose          = verbose,
                )
            except BondStabilityError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "bond_iso=%d m=%d lat=%d: %s — "
                    "marking as invalid (excluded from KMC)",
                    brs.iso_class, m_idx, lc.lateral_class, reason,
                )
                if verbose:
                    print(
                        f"  ⚠  bond_iso={brs.iso_class} m={m_idx} "
                        f"lat={lc.lateral_class}: {reason}\n"
                        f"     → marked as invalid (will not be admitted to KMC)"
                    )
                lc.stable         = False
                lc.invalid_reason = reason
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _log.error(
                    "bond_iso=%d m=%d lat=%d: unexpected error during "
                    "check_bond_site_stability — marking as invalid: %s",
                    brs.iso_class, m_idx, lc.lateral_class, reason,
                    exc_info=True,
                )
                if verbose:
                    print(
                        f"  ✗  bond_iso={brs.iso_class} m={m_idx} "
                        f"lat={lc.lateral_class}: unexpected error: {reason}\n"
                        f"     → marked as invalid (will not be admitted to KMC)"
                    )
                lc.stable         = False
                lc.invalid_reason = reason

        if not lc.stable:
            continue
        if lc.energy_ab is None or lc.energy_c is None or lc.energy_ts is None:
            continue

        brs._member_lc[m_idx] = lc  # type: ignore[attr-defined]

        delta_e, barrier, rate = _bond_energetics_cached(
            lc, direction,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
        )
        reactions.append(BondReaction(
            kind          = "bond",
            direction     = direction,
            site          = brs,
            member_index  = m_idx,
            lateral_class = lc,
            delta_e       = delta_e,
            barrier       = barrier,
            rate          = rate,
        ))

    brs.applicable_reactions = reactions  # type: ignore[attr-defined]
    return reactions


def compute_all_bond_reactions(
    G: nx.Graph,
    bond_reaction_sites: Iterable[BondReactionSite],
    calculator=None,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = NEB_FMAX,
    max_steps: int = NEB_MAX_STEPS,
    n_images: int = NEB_N_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    lateral_interactions: bool = True,
    verbose: bool = False,
) -> list[BondReaction]:
    """Compute applicable bond events for every site; return the flat list."""
    out: list[BondReaction] = []
    for brs in bond_reaction_sites:
        out.extend(get_applicable_bond_reactions(
            G, brs, calculator,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            n_images                 = n_images,
            climb                    = climb,
            spring_k                 = spring_k,
            interpolation            = interpolation,
            nl_mult                  = nl_mult,
            persist_neb_path         = persist_neb_path,
            lateral_interactions     = lateral_interactions,
            verbose                  = verbose,
        ))
    return out


def fast_bond_reaction_for_member(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> BondReaction | None:
    """Build the current BondReaction for one member from cached state.

    Cache-only fast path mirroring
    :func:`autokmc.kmc_diffusion.fast_diffusion_for_member`.
    """
    applicable, direction = is_bond_applicable(G, brs, member_index)
    if not applicable or direction is None:
        return None

    member_lc: dict | None = getattr(brs, "_member_lc", None)
    lc = member_lc.get(member_index) if member_lc is not None else None
    if lc is None or lc.stable is False:
        return None
    if lc.energy_ab is None or lc.energy_c is None or lc.energy_ts is None:
        return None

    delta_e, barrier, rate = _bond_energetics_cached(
        lc, direction,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
    )
    return BondReaction(
        kind          = "bond",
        direction     = direction,
        site          = brs,
        member_index  = member_index,
        lateral_class = lc,
        delta_e       = delta_e,
        barrier       = barrier,
        rate          = rate,
    )


__all__ = [
    "BondReaction",
    "is_bond_applicable",
    "get_applicable_bond_reactions",
    "compute_all_bond_reactions",
    "fast_bond_reaction_for_member",
]

