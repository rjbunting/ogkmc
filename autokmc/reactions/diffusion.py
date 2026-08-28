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

from contextvars import copy_context
from copy import copy
from dataclasses import dataclass
from typing import Any
import numpy as np
import networkx as nx
from ase import Atoms

from autokmc.io.calculators import (
    CalculatorConfigError,
    CalculatorPool,
    calculator_batch_active,
    calculator_batch_context,
)
from autokmc.sites.diffusion import DiffusionSite, DiffusionLateral
from autokmc.sites.stability.diffusion import (
    check_diffusion_site_lateral,
    check_diffusion_stability,
    DiffusionStabilityError,
    get_diffusion_bare_lateral,
    NEBNotConvergedError,
)
from autokmc.reactions.rates import (
    EA_MIN,
    DEFAULT_TRANSMISSION_COEFFICIENT,
    _eyring_prefactor,
)
from autokmc.core.constants import (
    NEB_BAND_EVAL,
    NEB_FMAX,
    NEB_IMAGE_SPACING,
    NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    NEB_INTERMEDIATE_MINIMUM_PROMINENCE,
    NEB_INTERMEDIATE_STAGNATION_STEPS,
    NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER,
    NEB_MAX_IMAGES,
    NEB_MAX_STEPS,
    NEB_MIN_IMAGES,
    NEB_N_IMAGES,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
    NEB_METHOD,
    NL_MULT_DEFAULT,
    LATERAL_SHELLS_DEFAULT,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER

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
    # Exclude the source because it becomes empty when the hop occurs. The
    # target is also excluded because the XOR check already showed it is empty.
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
        # In this direction, A is the source and B is the target.
        if _target_blocked(G, site_b, m_b, list(a_nids)):
            return False, None
        return True, "a_to_b"
    else:
        # Otherwise, B is the source and A is the target.
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
        lc._rate_cache = cache
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
    if not all(np.isfinite(value) for value in (e_a, e_b, e_ts)):
        raise ValueError(
            f"diffusion energies must be finite, got {(e_a, e_b, e_ts)!r}"
        )

    # First, place the effective transition state at least EA_MIN above the
    # higher endpoint. Both barriers then come from the same transition-state
    # energy, which preserves Ea_fwd - Ea_rev = E_b - E_a.
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
        if not all(np.isfinite(value) for value in (de, ea_kmc, rate)):
            raise ValueError("diffusion energetics produced non-finite values")
        return float(de), float(ea_kmc), rate

    out_fwd = _make("a_to_b")
    out_rev = _make("b_to_a")
    cache[key[:2] + ("a_to_b", bool(use_g))] = out_fwd
    cache[key[:2] + ("b_to_a", bool(use_g))] = out_rev

    return out_fwd if direction == "a_to_b" else out_rev


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _replace_cached_member_diffusion(
    site: DiffusionSite,
    member_index: int,
    reaction: DiffusionReaction | None,
) -> None:
    reactions = [
        candidate
        for candidate in (getattr(site, "applicable_reactions", None) or [])
        if int(candidate.member_index) != int(member_index)
    ]
    if reaction is not None:
        reactions.append(reaction)
    reactions.sort(key=lambda candidate: int(candidate.member_index))
    site.applicable_reactions = reactions


def _diffusion_seed_path(
    lateral_class: DiffusionLateral,
    *,
    n_images: int | None,
    current_member_index: int,
) -> tuple[list[Atoms] | None, int | None]:
    """Return a detached compatible bare band and its source member."""
    source_member = getattr(
        lateral_class,
        "_warm_start_member_index",
        None,
    )
    try:
        source_member = int(source_member)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if source_member != int(current_member_index):
        return None, None

    path = (
        getattr(lateral_class, "_warm_start_neb_path", None)
        or getattr(lateral_class, "atoms_neb_path", None)
    )
    try:
        images = list(path or [])
    except TypeError:
        return None, None
    if (
        (n_images is not None and len(images) != int(n_images) + 2)
        or not all(isinstance(image, Atoms) for image in images)
    ):
        return None, None
    detached = [image.copy() for image in images]
    for image in detached:
        image.calc = None
    lateral_class._warm_start_neb_path = [
        image.copy() for image in detached
    ]
    lateral_class._warm_start_member_index = source_member
    return detached, source_member


def get_applicable_diffusion_for_member(
    G: nx.Graph,
    ds: DiffusionSite,
    member_index: int,
    calculator,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = NEB_FMAX,
    max_steps: int = NEB_MAX_STEPS,
    n_images: int = NEB_N_IMAGES,
    image_spacing: float | None = NEB_IMAGE_SPACING,
    min_images: int = NEB_MIN_IMAGES,
    max_images: int = NEB_MAX_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER,
    neb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_climb_optimizer: str | None = None,
    neb_climb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_method: str = NEB_METHOD,
    neb_band_eval: str = NEB_BAND_EVAL,
    neb_geometry_guard_multiplier: float = (
        NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER
    ),
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS,
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    ),
    verbose: bool = False,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    update_site_cache: bool = True,
) -> DiffusionReaction | None:
    """Scientifically reclassify and evaluate one concrete diffusion member."""
    if not hasattr(ds, "_member_lc"):
        ds._member_lc = {}
    index = int(member_index)
    reaction: DiffusionReaction | None = None
    applicable, direction = is_diffusion_applicable(G, ds, index)
    if not applicable or direction is None:
        ds._member_lc.pop(index, None)
    else:
        stability_kwargs = {
            "frozen_indices": frozen_indices,
            "fmax": fmax,
            "max_steps": max_steps,
            "n_images": n_images,
            "image_spacing": image_spacing,
            "min_images": min_images,
            "max_images": max_images,
            "climb": climb,
            "spring_k": spring_k,
            "interpolation": interpolation,
            "nl_mult": nl_mult,
            "persist_neb_path": persist_neb_path,
            "optimizer": optimizer,
            "optimizer_kwargs": optimizer_kwargs,
            "neb_optimizer": neb_optimizer,
            "neb_optimizer_kwargs": neb_optimizer_kwargs,
            "neb_climb_optimizer": neb_climb_optimizer,
            "neb_climb_optimizer_kwargs": neb_climb_optimizer_kwargs,
            "neb_method": neb_method,
            "neb_band_eval": neb_band_eval,
            "neb_geometry_guard_multiplier": neb_geometry_guard_multiplier,
            "neb_intermediate_stagnation_steps": neb_intermediate_stagnation_steps,
            "neb_intermediate_energy_tolerance": neb_intermediate_energy_tolerance,
            "neb_intermediate_minimum_prominence": (
                neb_intermediate_minimum_prominence
            ),
            "verbose": verbose,
            "free_energy_options": free_energy_options,
            "free_energy_temperature_k": float(temperature),
            "vib_cache_root": vib_cache_root,
            "calculation_cache_root": calculation_cache_root,
            "calculation_cache_lookup_enabled": (
                calculation_cache_lookup_enabled
            ),
        }
        bare_lc: DiffusionLateral | None = None
        bare_seed_path: list[Atoms] | None = None
        bare_seed_member_index: int | None = None
        try:
            if lateral_interactions and calculator is not None:
                bare_lc = get_diffusion_bare_lateral(
                    G,
                    ds,
                    index,
                    n_shells=lateral_shells,
                )
            lc = check_diffusion_site_lateral(
                G,
                ds,
                index,
                n_shells=lateral_shells,
                ignore_lateral=not lateral_interactions,
            )
        except CalculatorConfigError:
            raise
        except (ValueError, IndexError) as exc:
            ds._member_lc.pop(index, None)
            if verbose:
                print(
                    f"  WARNING diff_iso={ds.iso_class} m={index}: "
                    f"lateral check skipped ({exc})"
                )
        else:
            if bare_lc is not None and lc.stable is None:
                bare_seed_path, bare_seed_member_index = (
                    _diffusion_seed_path(
                        bare_lc,
                        n_images=(None if image_spacing is not None else n_images),
                        current_member_index=index,
                    )
                )
                capture_lc = bare_lc
                preserve_bare_result = (
                    bare_lc.stable is True and bare_seed_path is None
                )
                if preserve_bare_result:
                    # The prior bare result is scientifically valid, but its
                    # path is missing or belongs to another concrete member.
                    # Capture a same-frame path without risking that result.
                    capture_lc = copy(bare_lc)
                    capture_lc.stable = None
                elif bare_lc.stable is not True:
                    bare_seed_path = None
                    bare_seed_member_index = None
                if (
                    capture_lc.stable is None
                    and not getattr(
                        capture_lc, "last_failure_reason", None,
                    )
                ):
                    try:
                        check_diffusion_stability(
                            G,
                            ds,
                            index,
                            capture_lc,
                            calculator,
                            capture_neb_path=True,
                            **stability_kwargs,
                        )
                    except NEBNotConvergedError as exc:
                        # This bare calculation is only an optional warm start
                        # for the current lateral event.  Keep the bare class
                        # undecided, latch the numerical failure against
                        # automatic retries, and let a distinct lateral class
                        # use interpolation.
                        reason = f"{type(exc).__name__}: {exc}"
                        if not preserve_bare_result:
                            bare_lc.last_failure_reason = reason
                        bare_seed_path = None
                        bare_seed_member_index = None
                        _log.warning(
                            "diff_iso=%d m=%d bare warm-start NEB did not "
                            "converge: %s; automatic retry is disabled and "
                            "a distinct lateral calculation will use %s "
                            "interpolation",
                            ds.iso_class,
                            index,
                            reason,
                            interpolation,
                        )
                    except DiffusionStabilityError as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        if not preserve_bare_result:
                            bare_lc.stable = False
                            bare_lc.invalid_reason = reason
                        bare_seed_path = None
                        bare_seed_member_index = None
                        _log.warning(
                            "diff_iso=%d m=%d bare warm-start failed: %s; "
                            "lateral calculation will use %s interpolation",
                            ds.iso_class,
                            index,
                            reason,
                            interpolation,
                        )
                    else:
                        bare_seed_path, bare_seed_member_index = (
                            _diffusion_seed_path(
                                capture_lc,
                                n_images=(
                                    None if image_spacing is not None else n_images
                                ),
                                current_member_index=index,
                            )
                        )
                        if preserve_bare_result and bare_seed_path is not None:
                            bare_lc._warm_start_neb_path = [
                                image.copy() for image in bare_seed_path
                            ]
                            bare_lc._warm_start_neb_energies = list(
                                getattr(
                                    capture_lc,
                                    "_warm_start_neb_energies",
                                    [],
                                )
                                or []
                            )
                            bare_lc._warm_start_member_index = (
                                bare_seed_member_index
                            )

            if (
                lc.stable is None
                and not getattr(lc, "last_failure_reason", None)
            ):
                try:
                    check_diffusion_stability(
                        G,
                        ds,
                        index,
                        lc,
                        calculator,
                        neb_seed_path=(
                            bare_seed_path
                            if lc is not bare_lc
                            else None
                        ),
                        neb_seed_member_index=bare_seed_member_index,
                        **stability_kwargs,
                    )
                except NEBNotConvergedError as exc:
                    # Preserve stable=None: a numerical search failure is
                    # not evidence that the event is impossible.  The stored
                    # failure reason is also a latch: this lateral class stays
                    # out of the rate index and is not tried again
                    # automatically during this run or after checkpoint
                    # resume.
                    reason = f"{type(exc).__name__}: {exc}"
                    lc.last_failure_reason = reason
                    _log.warning(
                        "diff_iso=%d m=%d lat=%d: %s — excluding this "
                        "reaction from KMC; the unresolved lateral class "
                        "will be retained for diagnostics without automatic "
                        "retry",
                        ds.iso_class,
                        index,
                        lc.lateral_class,
                        reason,
                    )
                    if verbose:
                        print(
                            f"  WARNING diff_iso={ds.iso_class} m={index} "
                            f"lat={lc.lateral_class}: {reason}\n"
                            "     → omitted from KMC and retained as an "
                            "unresolved diagnostic (automatic retry disabled)"
                        )
                except DiffusionStabilityError as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    _log.warning(
                        "diff_iso=%d m=%d lat=%d: %s — "
                        "marking as invalid (excluded from KMC)",
                        ds.iso_class,
                        index,
                        lc.lateral_class,
                        reason,
                    )
                    if verbose:
                        print(
                            f"  WARNING diff_iso={ds.iso_class} m={index} "
                            f"lat={lc.lateral_class}: {reason}\n"
                            "     → marked as invalid "
                            "(will not be admitted to KMC)"
                        )
                    lc.stable = False
                    lc.invalid_reason = reason
                except CalculatorConfigError:
                    raise

            if (
                lc.stable
                and lc.energy_a is not None
                and lc.energy_b is not None
                and lc.energy_ts is not None
            ):
                ds._member_lc[index] = lc
                delta_e, barrier, rate = _diffusion_energetics_cached(
                    lc,
                    direction,
                    temperature=temperature,
                    transmission_coefficient=transmission_coefficient,
                )
                reaction = DiffusionReaction(
                    kind="diffusion",
                    direction=direction,
                    site=ds,
                    member_index=index,
                    lateral_class=lc,
                    delta_e=delta_e,
                    barrier=barrier,
                    rate=rate,
                )
            else:
                ds._member_lc.pop(index, None)

    if update_site_cache:
        _replace_cached_member_diffusion(ds, index, reaction)
    return reaction


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
    image_spacing: float | None = NEB_IMAGE_SPACING,
    min_images: int = NEB_MIN_IMAGES,
    max_images: int = NEB_MAX_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER,
    neb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_climb_optimizer: str | None = None,
    neb_climb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_method: str = NEB_METHOD,
    neb_band_eval: str = NEB_BAND_EVAL,
    neb_geometry_guard_multiplier: float = (
        NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER
    ),
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS,
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    ),
    verbose: bool = False,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
) -> list[DiffusionReaction]:
    """Enumerate all currently-applicable hop events for one DiffusionSite.

    Parameters
    ----------
    lateral_interactions : bool
        When ``False``, third-party occupied adsorbate neighbours are excluded
        from the lateral ego-graph so every member maps to the single bare
        lat0.  Default ``True``.
    lateral_shells : int
        Number of surface-neighbour shells included in lateral
        classification.  Defaults to :data:`LATERAL_SHELLS_DEFAULT`.
    """
    reactions: list[DiffusionReaction] = []
    # Keep completed members visible while the initial sweep is in progress.
    # A later numerical NEB failure must not hide earlier valid reactions from
    # the emergency persistence pass.
    ds.applicable_reactions = reactions

    for m_idx in range(len(ds.member_node_ids)):
        try:
            reaction = get_applicable_diffusion_for_member(
                G,
                ds,
                m_idx,
                calculator,
                temperature=temperature,
                transmission_coefficient=transmission_coefficient,
                frozen_indices=frozen_indices,
                fmax=fmax,
                max_steps=max_steps,
                n_images=n_images,
                image_spacing=image_spacing,
                min_images=min_images,
                max_images=max_images,
                climb=climb,
                spring_k=spring_k,
                interpolation=interpolation,
                nl_mult=nl_mult,
                persist_neb_path=persist_neb_path,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                neb_optimizer=neb_optimizer,
                neb_optimizer_kwargs=neb_optimizer_kwargs,
                neb_climb_optimizer=neb_climb_optimizer,
                neb_climb_optimizer_kwargs=neb_climb_optimizer_kwargs,
                neb_method=neb_method,
                neb_band_eval=neb_band_eval,
                neb_geometry_guard_multiplier=neb_geometry_guard_multiplier,
                neb_intermediate_stagnation_steps=neb_intermediate_stagnation_steps,
                neb_intermediate_energy_tolerance=neb_intermediate_energy_tolerance,
                neb_intermediate_minimum_prominence=(
                    neb_intermediate_minimum_prominence
                ),
                verbose=verbose,
                lateral_interactions=lateral_interactions,
                lateral_shells=lateral_shells,
                free_energy_options=free_energy_options,
                vib_cache_root=vib_cache_root,
                calculation_cache_root=calculation_cache_root,
                calculation_cache_lookup_enabled=calculation_cache_lookup_enabled,
                update_site_cache=False,
            )
        except NEBNotConvergedError as exc:
            # Defensive boundary: the member entry point normally converts
            # this into ``None`` after recording diagnostics.  Keep a future
            # or alternate implementation from aborting the remaining sweep.
            _log.warning(
                "diff_iso=%d m=%d: numerical NEB failure escaped member "
                "evaluation (%s); continuing the KMC sweep",
                ds.iso_class,
                m_idx,
                exc,
            )
            reaction = None
        if reaction is not None:
            reactions.append(reaction)

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
    image_spacing: float | None = NEB_IMAGE_SPACING,
    min_images: int = NEB_MIN_IMAGES,
    max_images: int = NEB_MAX_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER,
    neb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_climb_optimizer: str | None = None,
    neb_climb_optimizer_kwargs: dict[str, Any] | None = None,
    neb_method: str = NEB_METHOD,
    neb_band_eval: str = NEB_BAND_EVAL,
    neb_geometry_guard_multiplier: float = (
        NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER
    ),
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS,
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    ),
    verbose: bool = False,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
) -> list[DiffusionReaction]:
    """Compute applicable hops for every DiffusionSite; return the flat list."""
    all_reactions: list[DiffusionReaction] = []
    if (
        isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and len(diffusion_sites) > 1
        and not calculator_batch_active()
    ):
        def _one(ds: DiffusionSite) -> list[DiffusionReaction]:
            with calculator_batch_context():
                return get_applicable_diffusions(
                    G, ds, calculator,
                    temperature              = temperature,
                    transmission_coefficient = transmission_coefficient,
                    frozen_indices           = frozen_indices,
                    fmax                     = fmax,
                    max_steps                = max_steps,
                    n_images                 = n_images,
                    image_spacing            = image_spacing,
                    min_images               = min_images,
                    max_images               = max_images,
                    climb                    = climb,
                    spring_k                 = spring_k,
                    interpolation            = interpolation,
                    nl_mult                  = nl_mult,
                    persist_neb_path         = persist_neb_path,
                    optimizer                = optimizer,
                    optimizer_kwargs         = optimizer_kwargs,
                    neb_optimizer            = neb_optimizer,
                    neb_optimizer_kwargs     = neb_optimizer_kwargs,
                    neb_climb_optimizer      = neb_climb_optimizer,
                    neb_climb_optimizer_kwargs = neb_climb_optimizer_kwargs,
                    neb_method                = neb_method,
                    neb_band_eval             = neb_band_eval,
                    neb_geometry_guard_multiplier = (
                        neb_geometry_guard_multiplier
                    ),
                    neb_intermediate_stagnation_steps = (
                        neb_intermediate_stagnation_steps
                    ),
                    neb_intermediate_energy_tolerance = (
                        neb_intermediate_energy_tolerance
                    ),
                    neb_intermediate_minimum_prominence = (
                        neb_intermediate_minimum_prominence
                    ),
                    verbose                  = verbose,
                    lateral_interactions     = lateral_interactions,
                    lateral_shells           = lateral_shells,
                    free_energy_options      = free_energy_options,
                    vib_cache_root           = vib_cache_root,
                    calculation_cache_root   = calculation_cache_root,
                    calculation_cache_lookup_enabled = (
                        calculation_cache_lookup_enabled
                    ),
                )

        futures = [
            calculator.submit(copy_context().run, _one, site)
            for site in diffusion_sites
        ]
        for reactions in calculator.gather(futures):
            all_reactions.extend(reactions)
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
            image_spacing            = image_spacing,
            min_images               = min_images,
            max_images               = max_images,
            climb                    = climb,
            spring_k                 = spring_k,
            interpolation            = interpolation,
            nl_mult                  = nl_mult,
            persist_neb_path         = persist_neb_path,
            optimizer                = optimizer,
            optimizer_kwargs         = optimizer_kwargs,
            neb_optimizer            = neb_optimizer,
            neb_optimizer_kwargs     = neb_optimizer_kwargs,
            neb_climb_optimizer      = neb_climb_optimizer,
            neb_climb_optimizer_kwargs = neb_climb_optimizer_kwargs,
            neb_method                = neb_method,
            neb_band_eval             = neb_band_eval,
            neb_geometry_guard_multiplier = neb_geometry_guard_multiplier,
            neb_intermediate_stagnation_steps = neb_intermediate_stagnation_steps,
            neb_intermediate_energy_tolerance = neb_intermediate_energy_tolerance,
            neb_intermediate_minimum_prominence = neb_intermediate_minimum_prominence,
            verbose                  = verbose,
            lateral_interactions     = lateral_interactions,
            lateral_shells           = lateral_shells,
            free_energy_options      = free_energy_options,
            vib_cache_root           = vib_cache_root,
            calculation_cache_root   = calculation_cache_root,
            calculation_cache_lookup_enabled = calculation_cache_lookup_enabled,
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

       The KMC main loop uses
       :func:`get_applicable_diffusion_for_member`, which reclassifies the
       affected member before consulting stability data.  This helper is
       provided only for callers that maintain their own invalidation
       discipline.
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
