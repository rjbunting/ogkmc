"""
autokmc.reactions.diffusion
=====================
Build KMC diffusion (hop) reactions for a graph of materialised
:class:`~autokmc.sites.diffusion.DiffusionSite`'s.

Mirrors :mod:`autokmc.reactions.adsorption` for adsorption / desorption.  For
every member of every :class:`DiffusionSite`:

1. **Direction guard** — exactly one of the two endpoints (A, B) must be
   currently occupied.  If both or neither are occupied the hop is not
   applicable (covered by the adsorption/desorption channel instead).
2. **Target clique guard** — the target endpoint's bonded surface cliques
   must not already be claimed by a *different* occupied adsorbate.
3. **Lateral classification** — :func:`autokmc.sites.stability.diffusion.
   check_diffusion_site_lateral` classifies the local environment.
4. **NEB stability** — the first time a member maps to a new lateral class,
   :func:`autokmc.sites.stability.diffusion.check_diffusion_stability` relaxes
   both endpoints, runs a CI-NEB, and caches ``E_a``, ``E_b``, ``E_ts``,
   plus the relaxed atoms, on the :class:`DiffusionLateral`.
5. **Energetics** for the firing direction::

       ΔE        = E_target − E_source
       Ea_raw    = E_ts     − E_source
       Ea_kmc    = max(EA_MIN, Ea_raw)              # eV
       k         = κ · (k_B T / h) · exp(−Ea_kmc / kT)

   The *raw* TS energy (and ``Ea_raw``) is persisted unchanged in the
   reaction folder by :class:`autokmc.io.persistence.ReactionWriter`; the
   ``EA_MIN`` floor is only applied when computing the rate consumed by
   the KMC loop, exactly as documented on
   :data:`autokmc.reactions.adsorption.EA_MIN`.

Public API
----------
* :class:`DiffusionReaction`         — one applicable hop event (carries
  a ``direction`` field: ``"a_to_b"`` or ``"b_to_a"``).
* :func:`is_diffusion_applicable`    — XOR-occupancy + clique-collision check.
* :func:`get_applicable_diffusions`  — per-DiffusionSite enumeration.
* :func:`compute_all_diffusions`     — full enumeration over all sites.
* :func:`fast_diffusion_for_member`  — single-member fast path for the KMC
  inner loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx

from autokmc.io.calculators import CalculatorConfigError, CalculatorPool
from autokmc.sites.diffusion import DiffusionSite, DiffusionLateral
from autokmc.sites.stability.diffusion import (
    check_diffusion_site_lateral,
    check_diffusion_stability,
    DiffusionStabilityError,
)
from autokmc.reactions.rates import (
    EA_MIN,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    _eyring_prefactor,
)
from autokmc.core.constants import (
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_N_IMAGES,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
    NL_MULT_DEFAULT,
)
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# DiffusionReaction dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiffusionReaction:
    """One applicable KMC diffusion (hop) event.

    Attributes
    ----------
    kind : str
        Always ``"diffusion"``.
    direction : str
        ``"a_to_b"`` (A → B hop) or ``"b_to_a"`` (B → A hop).  Determined
        by which endpoint is currently occupied.
    site : DiffusionSite
        Parent diffusion iso-class.
    member_index : int
        Index into ``site.member_node_ids`` (a concrete pair of placements).
    lateral_class : DiffusionLateral
        Lateral environment under which the energetics were computed and
        cached.
    delta_e : float
        ``E_target − E_source`` (eV) — the *raw* energy difference between
        the two relaxed endpoints in the firing direction.
    barrier : float
        KMC activation energy ``max(EA_MIN, E_ts − E_source)`` (eV).  The
        unfloored raw barrier is persisted in the reaction folder.
    rate : float
        Eyring rate ``κ · (k_B T / h) · exp(−barrier / kT)`` (Hz).
    """
    kind          : str
    direction     : str            # "a_to_b" | "b_to_a"
    site          : DiffusionSite
    member_index  : int
    lateral_class : DiffusionLateral
    delta_e       : float
    barrier       : float
    rate          : float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _endpoint_occupied(G: nx.Graph, node_ids: list[int]) -> bool:
    """Return True if any of the placement's nodes is currently occupied.

    Mirrors :meth:`AdsorbateSite._member_is_occupied`.
    """
    for nid in node_ids:
        if nid in G and G.nodes[nid].get("occupied", False):
            return True
    return False


def _target_blocked(
    G: nx.Graph,
    target_site,
    target_m_idx: int,
    source_node_ids: list[int],
) -> bool:
    """Clique-collision guard on the *target* endpoint.

    The target endpoint is "blocked" if any *other* occupied adsorbate
    (i.e. not the source endpoint we're about to vacate) shares one of the
    target's bonded surface cliques.  Uses
    ``G.graph["occupied_by_clique"]`` for an O(k) check.
    """
    member_cliques = getattr(target_site, "_member_cliques", None)
    occupied_by_clique = G.graph.get("occupied_by_clique")
    # Set of node ids we *don't* count as conflicts (source vacates as the
    # hop fires, target's own current state is irrelevant — we already know
    # it is unoccupied because direction was determined by XOR).
    target_node_ids = frozenset(target_site.member_node_ids[target_m_idx])
    excluded: frozenset = target_node_ids | frozenset(source_node_ids)

    if member_cliques is not None and occupied_by_clique is not None:
        for clq in member_cliques[target_m_idx]:
            occ_nodes = occupied_by_clique.get(clq)
            if occ_nodes and not occ_nodes.issubset(excluded):
                return True
        return False

    target_cliques: set[frozenset] = set()
    if member_cliques is not None:
        target_cliques.update(member_cliques[target_m_idx])
    else:
        for nid in target_node_ids:
            if nid not in G:
                continue
            clq = G.nodes[nid].get("clique")
            if clq is not None:
                target_cliques.add(clq)
    if not target_cliques:
        return False

    for n, d in G.nodes(data=True):
        if n in excluded:
            continue
        if d.get("type") != "adsorbate":
            continue
        if not d.get("occupied", False):
            continue
        clq = d.get("clique")
        if clq is not None and clq in target_cliques:
            return True
    return False


def is_diffusion_applicable(
    G: nx.Graph, ds: DiffusionSite, member_index: int,
) -> tuple[bool, str | None]:
    """Decide whether the given (DiffusionSite, member) is applicable now.

    Returns ``(applicable, direction)``.  ``direction`` is one of
    ``"a_to_b"``, ``"b_to_a"``, or ``None`` (when not applicable).

    Applicability requires:

    1. Exactly one of A and B currently occupied (XOR).
    2. The target endpoint is not clique-blocked by another adsorbate.
    """
    a_nids, b_nids = ds.member_node_ids[member_index]
    a_occ = _endpoint_occupied(G, list(a_nids))
    b_occ = _endpoint_occupied(G, list(b_nids))
    if a_occ == b_occ:
        return False, None

    site_a, m_a, site_b, m_b = ds.members[member_index]
    if a_occ:
        # A → B hop.  Source = A, target = B.
        if _target_blocked(G, site_b, m_b, list(a_nids)):
            return False, None
        return True, "a_to_b"
    else:
        # B → A hop.
        if _target_blocked(G, site_a, m_a, list(b_nids)):
            return False, None
        return True, "b_to_a"


# ---------------------------------------------------------------------------
# Energetics
# ---------------------------------------------------------------------------

def _diffusion_energetics_cached(
    lc: DiffusionLateral,
    direction: str,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float, float]:
    """Return ``(delta_e, barrier_kmc, rate)`` for one hop direction.

    When the lateral class has free-energy fields populated
    (``g_a`` / ``g_b`` / ``g_ts``) those are used in place of the electronic
    energies so the rate is derived from ΔG.  Pressure does not enter the
    intramolecular hop rate (no gas-phase species changes between A and B).
    """
    if direction not in {"a_to_b", "b_to_a"}:
        raise ValueError(
            "unknown diffusion direction "
            f"{direction!r}; expected 'a_to_b' or 'b_to_a'"
        )
    use_g = (
        getattr(lc, "g_a",  None) is not None
        and getattr(lc, "g_b",  None) is not None
        and getattr(lc, "g_ts", None) is not None
    )
    key = (
        round(float(temperature),              9),
        round(float(transmission_coefficient), 9),
        str(direction),
        bool(use_g),
    )
    cache: dict | None = getattr(lc, "_rate_cache", None)
    if cache is None:
        cache = {}
        lc._rate_cache = cache  # type: ignore[attr-defined]
    hit = cache.get(key)
    if hit is not None:
        return hit

    if use_g:
        e_a  = float(lc.g_a)    # type: ignore[arg-type]
        e_b  = float(lc.g_b)    # type: ignore[arg-type]
        e_ts = float(lc.g_ts)   # type: ignore[arg-type]
    else:
        e_a  = float(lc.energy_a)   # type: ignore[arg-type]
        e_b  = float(lc.energy_b)   # type: ignore[arg-type]
        e_ts = float(lc.energy_ts)  # type: ignore[arg-type]

    # Raise the effective TS so it is at least EA_MIN above the higher
    # endpoint.  Deriving both barriers from the same e_ts_eff preserves
    # energy consistency: Ea_fwd − Ea_rev = E_b − E_a.
    e_ts_eff = max(e_ts, max(e_a, e_b) + EA_MIN)

    prefactor, kT = _eyring_prefactor(temperature, transmission_coefficient)

    def _make(direction_: str) -> tuple[float, float, float]:
        if direction_ == "a_to_b":
            de     = e_b - e_a
            ea_kmc = max(EA_MIN, e_ts_eff - e_a)   # safety floor
        else:
            de     = e_a - e_b
            ea_kmc = max(EA_MIN, e_ts_eff - e_b)   # safety floor
        rate = float(prefactor * np.exp(-ea_kmc / kT))
        return float(de), float(ea_kmc), rate

    out_fwd = _make("a_to_b")
    out_rev = _make("b_to_a")
    cache[key[:2] + ("a_to_b", bool(use_g))] = out_fwd
    cache[key[:2] + ("b_to_a", bool(use_g))] = out_rev

    return out_fwd if direction == "a_to_b" else out_rev


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_applicable_diffusions(
    G: nx.Graph,
    ds: DiffusionSite,
    calculator,
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
    verbose: bool = False,
    lateral_interactions: bool = True,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
) -> list[DiffusionReaction]:
    """Enumerate all currently-applicable hop events for one DiffusionSite.

    Parameters
    ----------
    lateral_interactions : bool
        When ``False``, third-party occupied adsorbate neighbours are excluded
        from the lateral ego-graph so every member maps to the single bare
        lat0.  Default ``True``.
    """
    if not hasattr(ds, "_member_lc"):
        ds._member_lc = {}  # type: ignore[attr-defined]

    reactions: list[DiffusionReaction] = []

    for m_idx in range(len(ds.member_node_ids)):
        applicable, direction = is_diffusion_applicable(G, ds, m_idx)
        if not applicable or direction is None:
            # Drop any stale lateral-class cache entry — applicability may
            # change again later, at which point the lateral environment
            # will be re-classified from scratch.
            ds._member_lc.pop(m_idx, None)  # type: ignore[attr-defined]
            continue

        # 1. Lateral classification (cheap if seen before).
        try:
            lc = check_diffusion_site_lateral(
                G, ds, m_idx,
                ignore_lateral=not lateral_interactions,
            )
        except (ValueError, IndexError) as exc:
            if verbose:
                print(
                    f"  ⚠  diff_iso={ds.iso_class} m={m_idx}: "
                    f"lateral check skipped ({exc})"
                )
            continue

        # 2. NEB / endpoint relaxation (only for new lateral classes).
        if lc.stable is None:
            try:
                check_diffusion_stability(
                    G, ds, m_idx, lc, calculator,
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
                    free_energy_options       = free_energy_options,
                    free_energy_temperature_k = float(temperature),
                    vib_cache_root            = vib_cache_root,
                    calculation_cache_root    = calculation_cache_root,
                )
            except DiffusionStabilityError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "diff_iso=%d m=%d lat=%d: %s — "
                    "marking as invalid (excluded from KMC)",
                    ds.iso_class, m_idx, lc.lateral_class, reason,
                )
                if verbose:
                    print(
                        f"  ⚠  diff_iso={ds.iso_class} m={m_idx} "
                        f"lat={lc.lateral_class}: {reason}\n"
                        f"     → marked as invalid (will not be admitted to KMC)"
                    )
                lc.stable         = False
                lc.invalid_reason = reason
            except CalculatorConfigError:
                raise
            except Exception as exc:
                # Catch-all for unexpected errors (calculator failures, NumPy
                # broadcast errors, etc.) that are not DiffusionStabilityError
                # subtypes.  Without this, lc.stable stays None and the same
                # expensive check is retried every KMC step.
                reason = f"{type(exc).__name__}: {exc}"
                _log.error(
                    "diff_iso=%d m=%d lat=%d: unexpected error during "
                    "check_diffusion_stability — marking as invalid: %s",
                    ds.iso_class, m_idx, lc.lateral_class, reason,
                    exc_info=True,
                )
                if verbose:
                    print(
                        f"  ✗  diff_iso={ds.iso_class} m={m_idx} "
                        f"lat={lc.lateral_class}: unexpected error: {reason}\n"
                        f"     → marked as invalid (will not be admitted to KMC)"
                    )
                lc.stable         = False
                lc.invalid_reason = reason

        if not lc.stable:
            continue
        if lc.energy_a is None or lc.energy_b is None or lc.energy_ts is None:
            continue

        ds._member_lc[m_idx] = lc  # type: ignore[attr-defined]

        delta_e, barrier, rate = _diffusion_energetics_cached(
            lc, direction,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
        )
        reactions.append(DiffusionReaction(
            kind          = "diffusion",
            direction     = direction,
            site          = ds,
            member_index  = m_idx,
            lateral_class = lc,
            delta_e       = delta_e,
            barrier       = barrier,
            rate          = rate,
        ))

    ds.applicable_reactions = reactions  # type: ignore[attr-defined]
    return reactions


def compute_all_diffusions(
    G: nx.Graph,
    diffusion_sites: list[DiffusionSite],
    calculator,
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
    verbose: bool = False,
    lateral_interactions: bool = True,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
) -> list[DiffusionReaction]:
    """Compute applicable hops for every DiffusionSite; return the flat list."""
    all_reactions: list[DiffusionReaction] = []
    if (
        isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and len(diffusion_sites) > 1
    ):
        def _one(ds: DiffusionSite) -> list[DiffusionReaction]:
            return get_applicable_diffusions(
                G, ds, calculator,
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
                verbose                  = verbose,
                lateral_interactions     = lateral_interactions,
                free_energy_options      = free_energy_options,
                vib_cache_root           = vib_cache_root,
                calculation_cache_root   = calculation_cache_root,
            )

        with ThreadPoolExecutor(max_workers=calculator.max_workers) as ex:
            for rxns in ex.map(_one, diffusion_sites):
                all_reactions.extend(rxns)
        return all_reactions

    for ds in diffusion_sites:
        rxns = get_applicable_diffusions(
            G, ds, calculator,
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
            verbose                  = verbose,
            lateral_interactions     = lateral_interactions,
            free_energy_options      = free_energy_options,
            vib_cache_root           = vib_cache_root,
            calculation_cache_root   = calculation_cache_root,
        )
        all_reactions.extend(rxns)
    return all_reactions


def fast_diffusion_for_member(
    G: nx.Graph,
    ds: DiffusionSite,
    member_index: int,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> DiffusionReaction | None:
    """Build the current DiffusionReaction for one member from cached state.

    .. warning::
       This is a *cache-only* fast path: it consults
       ``ds._member_lc[member_index]`` and assumes that cached
       :class:`DiffusionLateral` is still the correct classification for
       the current graph state.  If the member's lateral environment may
       have changed since the cache was populated (e.g. a neighbouring
       adsorbate was toggled), use :func:`get_applicable_diffusions`
       instead so the lateral class is re-evaluated.

       The KMC main loop always uses :func:`get_applicable_diffusions`
       through :func:`autokmc.kmc.engine._recompute_affected_sites`.
       This helper is provided for callers that maintain their own
       invalidation discipline.
    """
    applicable, direction = is_diffusion_applicable(G, ds, member_index)
    if not applicable or direction is None:
        return None

    member_lc: dict | None = getattr(ds, "_member_lc", None)
    lc = member_lc.get(member_index) if member_lc is not None else None
    if lc is None or not lc.stable:
        return None
    if lc.energy_a is None or lc.energy_b is None or lc.energy_ts is None:
        return None

    delta_e, barrier, rate = _diffusion_energetics_cached(
        lc, direction,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
    )
    return DiffusionReaction(
        kind          = "diffusion",
        direction     = direction,
        site          = ds,
        member_index  = member_index,
        lateral_class = lc,
        delta_e       = delta_e,
        barrier       = barrier,
        rate          = rate,
    )
