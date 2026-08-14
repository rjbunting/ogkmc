"""
autokmc.reactions.bond
================
Build KMC **bond-changing** reactions for a graph of materialised
:class:`~autokmc.sites.bond.BondReactionSite`'s.

Each bond reaction is a reversible three-state event::

    A + B  ⇌  C

with three placements (``placement_A``, ``placement_B``, ``placement_C``)
sitting on the surface graph (see :mod:`autokmc.sites.bond` for
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
Identical scheme to :mod:`autokmc.reactions.diffusion`.  The first time a
member maps to a new lateral class the on-the-fly stability /  NEB
calculation populates ``lc.energy_ab``, ``lc.energy_c``, ``lc.energy_ts``
on the :class:`BondReactionLateral`.  The KMC rate uses an Eyring
prefactor with the standard ``EA_MIN`` floor::

    e_ts_eff = max(e_ts, max(e_ab, e_c) + EA_MIN)
    Ea(coupling)      = max(EA_MIN, e_ts_eff − e_ab)
    Ea(dissociation)  = max(EA_MIN, e_ts_eff − e_c)
    k                 = κ · (k_B T / h) · exp(−Ea / kT)

The *raw* TS energy is persisted unchanged by
:class:`autokmc.io.persistence.ReactionWriter`; the floor is only applied
when computing the rate consumed by the KMC loop.

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

from contextvars import copy_context
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import networkx as nx
from ase import Atoms

from autokmc.io.calculators import (
    CalculatorConfigError,
    CalculatorPool,
    calculator_batch_active,
    calculator_batch_context,
)
from autokmc.sites.bond import BondReactionSite, BondReactionLateral
from autokmc.reactions.rates import EA_MIN, DEFAULT_TRANSMISSION_COEFFICIENT, _eyring_prefactor
from autokmc.sites.stability.bond import (
    check_bond_site_lateral,
    check_bond_site_stability,
    BondNEBNotConvergedError,
    BondStabilityError,
    get_bond_bare_lateral,
)
from autokmc.core.constants import (
    NEB_BAND_EVAL,
    NEB_FMAX,
    NEB_IMAGE_SPACING,
    NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER,
    NEB_MAX_IMAGES,
    NEB_MAX_STEPS,
    NEB_MIN_IMAGES,
    NEB_N_IMAGES,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_METHOD,
    NEB_LOW_BARRIER_FMAX,
    BOND_NEB_INTERPOLATION,
    BOND_ATOM_MATCHING,
    BOND_MATCHING_TRIALS,
    BOND_GAS_PRECURSOR_DISTANCE,
    BOND_GAS_PRECURSOR_RELAX,
    NL_MULT_DEFAULT,
    LATERAL_SHELLS_DEFAULT,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER

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


def _placements_share_exact_clique(
    cliques_a: Iterable[frozenset],
    cliques_b: Iterable[frozenset],
) -> bool:
    """Return True when two placements cannot be occupied together."""
    return bool(set(cliques_a) & set(cliques_b))


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
    gas_product = bool(getattr(brs, "gas_product", False))
    site_a, m_a, site_b, m_b, site_c, m_c = brs.members[member_index]
    a_nids = list(site_a.member_node_ids[m_a])
    b_nids = list(site_b.member_node_ids[m_b])
    c_nids = (
        []
        if gas_product or site_c is None
        else list(site_c.member_node_ids[m_c])
    )
    a_occ = _placement_occupied(G, a_nids)
    b_occ = _placement_occupied(G, b_nids)
    c_occ = False if gas_product else _placement_occupied(G, c_nids)

    if gas_product:
        couple_state = a_occ and b_occ
        dissoc_state = (not a_occ and not b_occ)
    else:
        couple_state = (a_occ and b_occ and not c_occ)
        dissoc_state = (c_occ and not a_occ and not b_occ)

    if couple_state == dissoc_state:
        # Either both true (impossible — A+B+C can't all be in the right state)
        # or neither — wrong occupancy pattern for this reaction.
        return False, None

    cliques_a, cliques_b, cliques_c = brs._member_cliques[member_index]
    if (
        not cliques_a
        or not cliques_b
        or (not gas_product and not cliques_c)
        or _placements_share_exact_clique(cliques_a, cliques_b)
    ):
        return False, None
    a_set = frozenset(a_nids)
    b_set = frozenset(b_nids)
    c_set = frozenset(c_nids)

    if couple_state:
        # Direction: A + B → C.  C will become occupied — its cliques
        # must be free except for what A or B already (themselves) hold.
        if (not gas_product) and _cliques_blocked(G, cliques_c, a_set | b_set | c_set):
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

    Mirrors :func:`autokmc.reactions.diffusion._diffusion_energetics_cached`:
    the effective TS energy is raised to sit at least ``EA_MIN`` above
    the higher of the two endpoints so that detailed balance is preserved
    after the KMC floor is applied.
    """
    if direction not in {"couple", "dissoc"}:
        raise ValueError(
            "unknown bond direction "
            f"{direction!r}; expected 'couple' or 'dissoc'"
        )
    gas_pressure_bar = (
        float(getattr(lc, "gas_pressure_bar", 0.0) or 0.0)
        if getattr(lc, "gas_product", False) else 0.0
    )
    use_g = all(
        getattr(lc, name, None) is not None
        for name in ("g_ab", "g_c", "g_ts")
    )
    key = (
        round(float(temperature),              9),
        round(float(transmission_coefficient), 9),
        round(gas_pressure_bar, 12),
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
        e_ab = float(lc.g_ab)
        e_c = float(lc.g_c)
        e_ts = float(lc.g_ts)
    else:
        e_ab = float(lc.energy_ab)   # type: ignore[arg-type]
        e_c  = float(lc.energy_c)    # type: ignore[arg-type]
        e_ts = float(lc.energy_ts)   # type: ignore[arg-type]
    if not all(np.isfinite(value) for value in (e_ab, e_c, e_ts)):
        raise ValueError(f"bond energies must be finite, got {(e_ab, e_c, e_ts)!r}")

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
        if getattr(lc, "gas_product", False) and direction_ == "dissoc":
            rate *= max(0.0, gas_pressure_bar)
        if not all(np.isfinite(value) for value in (de, ea_kmc, rate)):
            raise ValueError("bond energetics produced non-finite values")
        return float(de), float(ea_kmc), rate

    out_couple = _make("couple")
    out_dissoc = _make("dissoc")
    cache[key[:3] + ("couple", bool(use_g))] = out_couple
    cache[key[:3] + ("dissoc", bool(use_g))] = out_dissoc

    return out_couple if direction == "couple" else out_dissoc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _replace_cached_member_bond_reaction(
    site: BondReactionSite,
    member_index: int,
    reaction: BondReaction | None,
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


def _bond_seed_path(
    lateral_class: BondReactionLateral,
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


def get_applicable_bond_reaction_for_member(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
    calculator=None,
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
    interpolation: str = BOND_NEB_INTERPOLATION,
    atom_matching: str = BOND_ATOM_MATCHING,
    matching_trials: int = BOND_MATCHING_TRIALS,
    gas_precursor_relax: bool = BOND_GAS_PRECURSOR_RELAX,
    gas_precursor_distance: float = BOND_GAS_PRECURSOR_DISTANCE,
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
    neb_low_barrier_fmax: float = NEB_LOW_BARRIER_FMAX,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    verbose: bool = False,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    free_energy_options=None,
    vib_cache_root: str | None = None,
    update_site_cache: bool = True,
) -> BondReaction | None:
    """Scientifically reclassify and evaluate one concrete bond member."""
    if not hasattr(brs, "_member_lc"):
        brs._member_lc = {}
    index = int(member_index)
    reaction: BondReaction | None = None
    applicable, direction = is_bond_applicable(G, brs, index)
    if not applicable or direction is None:
        brs._member_lc.pop(index, None)
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
            "atom_matching": atom_matching,
            "matching_trials": matching_trials,
            "gas_precursor_relax": gas_precursor_relax,
            "gas_precursor_distance": gas_precursor_distance,
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
            "neb_low_barrier_fmax": neb_low_barrier_fmax,
            "verbose": verbose,
            "calculation_cache_root": calculation_cache_root,
            "calculation_cache_lookup_enabled": (
                calculation_cache_lookup_enabled
            ),
            "free_energy_options": free_energy_options,
            "free_energy_temperature_k": float(temperature),
            "vib_cache_root": vib_cache_root,
        }
        bare_lc: BondReactionLateral | None = None
        bare_seed_path: list[Atoms] | None = None
        bare_seed_member_index: int | None = None
        try:
            if lateral_interactions and calculator is not None:
                bare_lc = get_bond_bare_lateral(
                    G,
                    brs,
                    index,
                    n_shells=lateral_shells,
                )
            lc = check_bond_site_lateral(
                G,
                brs,
                index,
                n_shells=lateral_shells,
                ignore_lateral=not lateral_interactions,
            )
        except CalculatorConfigError:
            raise
        except (ValueError, IndexError) as exc:
            brs._member_lc.pop(index, None)
            if verbose:
                print(
                    f"  ⚠  bond_iso={brs.iso_class} m={index}: "
                    f"lateral check skipped ({exc})"
                )
        else:
            if bare_lc is not None and lc.stable is None:
                bare_seed_path, bare_seed_member_index = _bond_seed_path(
                    bare_lc,
                    n_images=(None if image_spacing is not None else n_images),
                    current_member_index=index,
                )
                capture_lc = bare_lc
                preserve_bare_result = (
                    bare_lc.stable is True and bare_seed_path is None
                )
                if preserve_bare_result:
                    capture_lc = copy(bare_lc)
                    capture_lc.stable = None
                elif bare_lc.stable is not True:
                    bare_seed_path = None
                    bare_seed_member_index = None
                if capture_lc.stable is None:
                    try:
                        check_bond_site_stability(
                            G,
                            brs,
                            index,
                            capture_lc,
                            calculator,
                            capture_neb_path=True,
                            **stability_kwargs,
                        )
                    except BondNEBNotConvergedError as exc:
                        # This bare calculation is only an optional warm start
                        # for the current lateral event. Keep the bare class
                        # undecided and let the lateral NEB use interpolation.
                        bare_seed_path = None
                        bare_seed_member_index = None
                        _log.warning(
                            "bond_iso=%d m=%d bare warm-start NEB did not "
                            "converge: %s; lateral calculation will use %s "
                            "interpolation",
                            brs.iso_class,
                            index,
                            exc,
                            interpolation,
                        )
                    except BondStabilityError as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        if not preserve_bare_result:
                            bare_lc.stable = False
                            bare_lc.invalid_reason = reason
                        bare_seed_path = None
                        bare_seed_member_index = None
                        _log.warning(
                            "bond_iso=%d m=%d bare warm-start failed: %s; "
                            "lateral calculation will use %s interpolation",
                            brs.iso_class,
                            index,
                            reason,
                            interpolation,
                        )
                    else:
                        bare_seed_path, bare_seed_member_index = (
                            _bond_seed_path(
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

            if lc.stable is None and calculator is not None:
                lc.last_failure_reason = None
                try:
                    check_bond_site_stability(
                        G,
                        brs,
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
                except BondNEBNotConvergedError as exc:
                    # Preserve stable=None: a numerical search failure is
                    # retryable, not evidence that the event is impossible.
                    # Omit only this candidate from the current rate index;
                    # aborting the full initial sweep would prevent otherwise
                    # valid KMC reactions from executing at all.
                    reason = f"{type(exc).__name__}: {exc}"
                    lc.last_failure_reason = reason
                    _log.warning(
                        "bond_iso=%d m=%d lat=%d: %s — excluding this "
                        "reaction from the current KMC sweep; the lateral "
                        "class remains retryable",
                        brs.iso_class,
                        index,
                        lc.lateral_class,
                        reason,
                    )
                    if verbose:
                        print(
                            f"  ⚠  bond_iso={brs.iso_class} m={index} "
                            f"lat={lc.lateral_class}: {reason}\n"
                            "     → omitted from this KMC sweep "
                            "(will be retried when recomputed)"
                        )
                except BondStabilityError as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    _log.warning(
                        "bond_iso=%d m=%d lat=%d: %s — "
                        "marking as invalid (excluded from KMC)",
                        brs.iso_class,
                        index,
                        lc.lateral_class,
                        reason,
                    )
                    if verbose:
                        print(
                            f"  ⚠  bond_iso={brs.iso_class} m={index} "
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
                and lc.energy_ab is not None
                and lc.energy_c is not None
                and lc.energy_ts is not None
            ):
                brs._member_lc[index] = lc
                delta_e, barrier, rate = _bond_energetics_cached(
                    lc,
                    direction,
                    temperature=temperature,
                    transmission_coefficient=transmission_coefficient,
                )
                reaction = BondReaction(
                    kind="bond",
                    direction=direction,
                    site=brs,
                    member_index=index,
                    lateral_class=lc,
                    delta_e=delta_e,
                    barrier=barrier,
                    rate=rate,
                )
            else:
                brs._member_lc.pop(index, None)

    if update_site_cache:
        _replace_cached_member_bond_reaction(brs, index, reaction)
    return reaction


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
    image_spacing: float | None = NEB_IMAGE_SPACING,
    min_images: int = NEB_MIN_IMAGES,
    max_images: int = NEB_MAX_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = BOND_NEB_INTERPOLATION,
    atom_matching: str = BOND_ATOM_MATCHING,
    matching_trials: int = BOND_MATCHING_TRIALS,
    gas_precursor_relax: bool = BOND_GAS_PRECURSOR_RELAX,
    gas_precursor_distance: float = BOND_GAS_PRECURSOR_DISTANCE,
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
    neb_low_barrier_fmax: float = NEB_LOW_BARRIER_FMAX,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    verbose: bool = False,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    free_energy_options=None,
    vib_cache_root: str | None = None,
) -> list[BondReaction]:
    """Enumerate currently-applicable bond events for one BondReactionSite.

    Mirrors :func:`autokmc.reactions.diffusion.get_applicable_diffusions` —
    classifies each applicable member into a
    :class:`BondReactionLateral` and triggers a CI-NEB stability run the
    first time a new lateral class is seen.  When *calculator* is None
    the NEB step is skipped and members whose lateral class has not yet
    been populated with energies are silently skipped.  ``lateral_shells``
    controls how many surface-neighbour shells are included in that
    classification.
    """
    reactions: list[BondReaction] = []
    # Keep completed members visible while the initial sweep is in progress.
    # In particular, a CI-NEB failure for a later lateral class must not hide
    # valid reactions already obtained for this BondReactionSite.
    brs.applicable_reactions = reactions

    for m_idx in range(len(brs.member_node_ids)):
        try:
            reaction = get_applicable_bond_reaction_for_member(
                G,
                brs,
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
                atom_matching=atom_matching,
                matching_trials=matching_trials,
                gas_precursor_relax=gas_precursor_relax,
                gas_precursor_distance=gas_precursor_distance,
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
                neb_low_barrier_fmax=neb_low_barrier_fmax,
                lateral_interactions=lateral_interactions,
                lateral_shells=lateral_shells,
                verbose=verbose,
                calculation_cache_root=calculation_cache_root,
                calculation_cache_lookup_enabled=calculation_cache_lookup_enabled,
                free_energy_options=free_energy_options,
                vib_cache_root=vib_cache_root,
                update_site_cache=False,
            )
        except BondNEBNotConvergedError as exc:
            # Defensive boundary: the member entry point normally converts
            # this into ``None`` after recording diagnostics.  Keep a future
            # or alternate implementation from aborting the remaining sweep.
            _log.warning(
                "bond_iso=%d m=%d: numerical NEB failure escaped member "
                "evaluation (%s); continuing the KMC sweep",
                brs.iso_class,
                m_idx,
                exc,
            )
            reaction = None
        if reaction is not None:
            reactions.append(reaction)

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
    image_spacing: float | None = NEB_IMAGE_SPACING,
    min_images: int = NEB_MIN_IMAGES,
    max_images: int = NEB_MAX_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = BOND_NEB_INTERPOLATION,
    atom_matching: str = BOND_ATOM_MATCHING,
    matching_trials: int = BOND_MATCHING_TRIALS,
    gas_precursor_relax: bool = BOND_GAS_PRECURSOR_RELAX,
    gas_precursor_distance: float = BOND_GAS_PRECURSOR_DISTANCE,
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
    neb_low_barrier_fmax: float = NEB_LOW_BARRIER_FMAX,
    lateral_interactions: bool = True,
    lateral_shells: int = LATERAL_SHELLS_DEFAULT,
    verbose: bool = False,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    free_energy_options=None,
    vib_cache_root: str | None = None,
) -> list[BondReaction]:
    """Compute applicable bond events for every site; return the flat list."""
    out: list[BondReaction] = []
    sites = list(bond_reaction_sites)
    if (
        isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and len(sites) > 1
        and not calculator_batch_active()
    ):
        def _one(brs: BondReactionSite) -> list[BondReaction]:
            with calculator_batch_context():
                return get_applicable_bond_reactions(
                    G, brs, calculator,
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
                    atom_matching            = atom_matching,
                    matching_trials          = matching_trials,
                    gas_precursor_relax       = gas_precursor_relax,
                    gas_precursor_distance    = gas_precursor_distance,
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
                    neb_low_barrier_fmax  = neb_low_barrier_fmax,
                    lateral_interactions     = lateral_interactions,
                    lateral_shells           = lateral_shells,
                    verbose                  = verbose,
                    calculation_cache_root   = calculation_cache_root,
                    calculation_cache_lookup_enabled = (
                        calculation_cache_lookup_enabled
                    ),
                    free_energy_options      = free_energy_options,
                    vib_cache_root           = vib_cache_root,
                )

        futures = [
            calculator.submit(copy_context().run, _one, site)
            for site in sites
        ]
        for reactions in calculator.gather(futures):
            out.extend(reactions)
        return out

    for brs in sites:
        out.extend(get_applicable_bond_reactions(
            G, brs, calculator,
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
            atom_matching            = atom_matching,
            matching_trials          = matching_trials,
            gas_precursor_relax       = gas_precursor_relax,
            gas_precursor_distance    = gas_precursor_distance,
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
            neb_low_barrier_fmax     = neb_low_barrier_fmax,
            lateral_interactions     = lateral_interactions,
            lateral_shells           = lateral_shells,
            verbose                  = verbose,
            calculation_cache_root   = calculation_cache_root,
            calculation_cache_lookup_enabled = calculation_cache_lookup_enabled,
            free_energy_options      = free_energy_options,
            vib_cache_root           = vib_cache_root,
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
    :func:`autokmc.reactions.diffusion.fast_diffusion_for_member`.
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
    "get_applicable_bond_reaction_for_member",
    "get_applicable_bond_reactions",
    "compute_all_bond_reactions",
    "fast_bond_reaction_for_member",
]
