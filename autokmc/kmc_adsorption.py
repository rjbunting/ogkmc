"""
autokmc.kmc_adsorption
======================
Build KMC adsorption / desorption reactions for a graph of materialised
:class:`~autokmc.find_adsorbate_sites.AdsorbateSite`'s.

(Previously named ``autokmc.kmc_reactions``; renamed for clarity now that
diffusion reactions live in their own module.  A thin
:mod:`autokmc.kmc_reactions` shim is kept for backwards compatibility.)

This module mirrors the role of :mod:`disreax_kmc.reactions` — turning the
current state of the system graph into a flat list of *applicable* reaction
entries that the KMC engine can sample from.  It is, however, adapted to the
on-the-fly stability / lateral-class workflow of :mod:`autokmc`:

For every member of every :class:`AdsorbateSite`:

1. A cheap *clique-collision* guard discards members blocked by an occupied
   neighbour that bonds to the same surface clique
   (see :func:`is_clique_blocked`).
2. The lateral-interaction class is classified on the fly with
   :func:`autokmc.check_adsorbate_sites.check_adsorbate_site_lateral`.
3. If that lateral class has not been ML-relaxed yet, the energies of the
   occupied / unoccupied configurations are computed with
   :func:`autokmc.check_adsorbate_sites.check_site_stability` and cached on
   the :class:`AdsorbateSiteLateral`.
4. An :class:`AdsorptionReaction` is emitted:

   * Adsorption (member currently empty)::

         ΔE_ads = E_occ − E_unocc − E_gas
         Ea     = max(EA_MIN, ΔE_ads + EA_MIN)         # eV
         k      = ν · exp(−Ea / kT)

   * Desorption (member currently occupied)::

         ΔE_des = E_unocc + E_gas − E_occ
         Ea     = max(EA_MIN, ΔE_des + EA_MIN)         # eV
         k      = ν · exp(−Ea / kT)

   ``E_gas`` is the gas-phase reactant energy (``Reactant.energy``)
   supplied via a SMILES → energy mapping.

Public API
----------
* :class:`AdsorptionReaction`       — one applicable adsorption / desorption
  event.  ``Reaction`` is kept as an alias.
* :func:`is_clique_blocked`         — clique-collision guard.
* :func:`get_applicable_reactions`  — per-site enumeration.
* :func:`compute_all_reactions`     — full enumeration over all sites.
* :func:`gather_all_applicable_reactions` — flatten into a single list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx

from autokmc.find_adsorbate_sites import AdsorbateSite, AdsorbateSiteLateral
from autokmc.check_adsorbate_sites import (
    check_adsorbate_site_lateral,
    check_site_stability,
    SiteStabilityError,
)
from autokmc.reactants import Reactant
from autokmc.logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Boltzmann constant in eV / K.
KB_EV: float = 8.617_333_262e-5

#: Planck constant in eV · s.
H_EV_S: float = 4.135_667_696e-15

#: Default transmission coefficient κ for the Eyring equation (dimensionless).
DEFAULT_TRANSMISSION_COEFFICIENT: float = 1.0

#: Minimum activation barrier (eV).  Two distinct uses:
#:
#: * Adsorption / desorption: simple ``Ea = max(EA_MIN, ΔE + EA_MIN)`` model.
#: * Diffusion (NEB): the *raw* barrier ``E_ts − E_initial`` from the NEB
#:   band is **floored at EA_MIN** for the KMC rate evaluation, but the
#:   *actual* converged ``E_ts`` is always saved unchanged in the
#:   per-reaction ``reaction.json`` so the floor is auditable.
EA_MIN: float = 0.1


# ---------------------------------------------------------------------------
# AdsorptionReaction dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdsorptionReaction:
    """One applicable KMC adsorption / desorption event.

    Attributes
    ----------
    kind : str
        Either ``"adsorption"`` or ``"desorption"``.
    site : AdsorbateSite
        Parent iso-class.
    member_index : int
        Index into ``site.member_node_ids``.
    lateral_class : AdsorbateSiteLateral
        Lateral environment under which the reaction's energetics were
        computed and cached.
    delta_e : float
        Reaction energy ``E_final − E_initial`` (eV), **including** the
        gas-phase reactant energy.
    barrier : float
        Activation energy ``Ea = max(EA_MIN, ΔE + EA_MIN)`` (eV).
    rate : float
        Eyring rate ``κ · (k_B T / h) · exp(−Ea / kT)`` (Hz).
    """
    kind          : str
    site          : AdsorbateSite
    member_index  : int
    lateral_class : AdsorbateSiteLateral
    delta_e       : float
    barrier       : float
    rate          : float


#: Backwards-compat alias.  Older code (and the segment-tree leaves in
#: :mod:`autokmc.kmc_simulation`) refer to ``Reaction``; new code should
#: use :class:`AdsorptionReaction`.
Reaction = AdsorptionReaction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_gas_energy_lookup(
    reactants: Reactant | Iterable[Reactant] | dict | None,
) -> dict[str, float]:
    """Normalise *reactants* to a ``{smiles: gas_energy}`` mapping."""
    if reactants is None:
        return {}
    if isinstance(reactants, dict):
        return {str(k): float(v) for k, v in reactants.items()}
    if isinstance(reactants, Reactant):
        reactants = [reactants]
    out: dict[str, float] = {}
    for r in reactants:
        if not isinstance(r, Reactant):
            raise TypeError(
                f"reactants must be Reactant / iterable of Reactant / "
                f"dict[smiles, energy] — got {type(r).__name__}"
            )
        out[r.smiles] = float(r.energy)
    return out


def _site_is_occupied(G: nx.Graph, site: AdsorbateSite, member_index: int) -> bool:
    return site._member_is_occupied(G, site.member_node_ids[member_index])


def is_clique_blocked(
    G: nx.Graph, site: AdsorbateSite, member_index: int,
) -> bool:
    """Return True if any *other* occupied adsorbate shares an exact clique.

    Two adsorbate atoms that bind to exactly the same set of surface atoms
    cannot physically co-exist.  When the current member's bonding clique is
    already claimed by an occupied neighbour, the site is blocked and neither
    adsorption (for an empty member) nor lateral-class classification of the
    member should be attempted.

    Uses the ``G.graph["occupied_by_clique"]`` reverse index built by
    :func:`~autokmc.find_adsorbate_sites.find_adsorbate_sites` and maintained
    by :func:`~autokmc.kmc_simulation._set_member_occupied`.
    """
    member_cliques = getattr(site, "_member_cliques", None)
    occupied_by_clique = G.graph.get("occupied_by_clique")
    node_ids = site.member_node_ids[member_index]
    member_id_set = frozenset(node_ids)

    if member_cliques is not None and occupied_by_clique is not None:
        for clq in member_cliques[member_index]:
            occ_nodes = occupied_by_clique.get(clq)
            if occ_nodes and not occ_nodes.issubset(member_id_set):
                return True
        return False

    # ── Legacy fallback: full graph scan (only hit if reverse index not built)
    member_cliques_set: set[frozenset] = set()
    for nid in node_ids:
        if nid not in G:
            continue
        clq = G.nodes[nid].get("clique")
        if clq is not None:
            member_cliques_set.add(clq)

    if not member_cliques_set:
        return False

    for n, d in G.nodes(data=True):
        if n in member_id_set:
            continue
        if d.get("type") != "adsorbate":
            continue
        if not d.get("occupied", False):
            continue
        clq = d.get("clique")
        if clq is not None and clq in member_cliques_set:
            return True
    return False


# ---------------------------------------------------------------------------
# Reaction-energetics helpers
# ---------------------------------------------------------------------------

def _eyring_prefactor(
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float]:
    """Return ``(prefactor, kT)`` for the Eyring rate equation."""
    kT = KB_EV * float(temperature)
    prefactor = float(transmission_coefficient) * kT / H_EV_S
    return float(prefactor), float(kT)


def _energetics_cached(
    lc: AdsorbateSiteLateral,
    e_gas: float,
    occupied: bool,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float, float]:
    """Return ``(delta_e, barrier, rate)`` with caching of *both* directions."""
    key = (
        round(float(temperature),                9),
        round(float(transmission_coefficient),   9),
        round(float(e_gas),                      9),
        bool(occupied),
    )
    cache: dict | None = getattr(lc, "_rate_cache", None)
    if cache is None:
        cache = {}
        lc._rate_cache = cache  # type: ignore[attr-defined]
    hit = cache.get(key)
    if hit is not None:
        return hit

    e_occ   = float(lc.energy_occupied)    # type: ignore[arg-type]
    e_unocc = float(lc.energy_unoccupied)  # type: ignore[arg-type]

    if occupied:
        delta_e = e_unocc + float(e_gas) - e_occ
    else:
        delta_e = e_occ - (e_unocc + float(e_gas))

    barrier  = max(EA_MIN, delta_e + EA_MIN)
    prefactor, kT = _eyring_prefactor(temperature, transmission_coefficient)
    rate = float(prefactor * np.exp(-barrier / kT))

    out = (float(delta_e), float(barrier), rate)
    cache[key] = out

    other_key = key[:-1] + (not bool(occupied),)
    if other_key not in cache:
        other_occ = not bool(occupied)
        if other_occ:
            other_de = e_unocc + float(e_gas) - e_occ
        else:
            other_de = e_occ - (e_unocc + float(e_gas))
        other_barrier = max(EA_MIN, other_de + EA_MIN)
        other_rate    = float(prefactor * np.exp(-other_barrier / kT))
        cache[other_key] = (
            float(other_de), float(other_barrier), other_rate,
        )

    return out


def _energetics(
    lc: AdsorbateSiteLateral,
    e_gas: float,
    occupied: bool,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float, float]:
    """Backwards-compatible wrapper around :func:`_energetics_cached`."""
    return _energetics_cached(
        lc, e_gas, occupied,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_applicable_reactions(
    G: nx.Graph,
    site: AdsorbateSite,
    calculator,
    gas_energies: dict[str, float],
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    verbose: bool = False,
) -> list[AdsorptionReaction]:
    """Enumerate all applicable adsorption / desorption events for one site."""
    if site.reactant not in gas_energies:
        raise KeyError(
            f"No gas-phase energy supplied for reactant SMILES "
            f"{site.reactant!r}.  Pass it via the ``reactants`` argument."
        )
    e_gas = float(gas_energies[site.reactant])

    if not hasattr(site, "_member_lc"):
        site._member_lc = {}  # type: ignore[attr-defined]

    reactions: list[AdsorptionReaction] = []

    for m_idx in range(len(site.member_node_ids)):
        if is_clique_blocked(G, site, m_idx):
            if verbose:
                print(f"  ⛔ iso={site.iso_class} m={m_idx}: clique blocked")
            continue

        try:
            lc = check_adsorbate_site_lateral(G, site, m_idx)
        except (ValueError, IndexError) as exc:
            if verbose:
                print(
                    f"  ⚠  iso={site.iso_class} m={m_idx}: "
                    f"lateral check skipped ({exc})"
                )
            continue

        if lc.stable is None:
            try:
                check_site_stability(
                    G, site, m_idx, lc, calculator,
                    frozen_indices = frozen_indices,
                    fmax           = fmax,
                    max_steps      = max_steps,
                    verbose        = verbose,
                )
            except SiteStabilityError as exc:
                lc.stable = False
                if verbose:
                    print(
                        f"  ✗  iso={site.iso_class} m={m_idx} "
                        f"lat={lc.lateral_class}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue

        if not lc.stable:
            continue
        if lc.energy_occupied is None or lc.energy_unoccupied is None:
            continue

        site._member_lc[m_idx] = lc  # type: ignore[attr-defined]

        occ = _site_is_occupied(G, site, m_idx)
        kind = "desorption" if occ else "adsorption"
        delta_e, barrier, rate = _energetics_cached(
            lc, e_gas, occ,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
        )
        reactions.append(AdsorptionReaction(
            kind          = kind,
            site          = site,
            member_index  = m_idx,
            lateral_class = lc,
            delta_e       = delta_e,
            barrier       = barrier,
            rate          = rate,
        ))

    site.applicable_reactions = reactions  # type: ignore[attr-defined]
    return reactions


def compute_all_reactions(
    G: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    calculator,
    reactants: Reactant | Iterable[Reactant] | dict | None,
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
    frozen_indices: list[int] | None = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    verbose: bool = False,
) -> list[AdsorptionReaction]:
    """Compute applicable reactions for every site and return the flat list."""
    gas_energies = _build_gas_energy_lookup(reactants)

    all_reactions: list[AdsorptionReaction] = []
    for site in adsorbate_sites:
        rxns = get_applicable_reactions(
            G, site, calculator, gas_energies,
            temperature              = temperature,
            transmission_coefficient = transmission_coefficient,
            frozen_indices           = frozen_indices,
            fmax                     = fmax,
            max_steps                = max_steps,
            verbose                  = verbose,
        )
        all_reactions.extend(rxns)
    return all_reactions


def gather_all_applicable_reactions(
    adsorbate_sites: list[AdsorbateSite],
) -> list[AdsorptionReaction]:
    """Flatten every site's cached ``applicable_reactions`` into one list."""
    out: list[AdsorptionReaction] = []
    for site in adsorbate_sites:
        rxns = getattr(site, "applicable_reactions", None) or []
        out.extend(rxns)
    return out


# ---------------------------------------------------------------------------
# Fast single-member re-evaluation (KMC inner loop)
# ---------------------------------------------------------------------------

def fast_reaction_for_member(
    G: nx.Graph,
    site: AdsorbateSite,
    member_index: int,
    gas_energies: dict[str, float],
    *,
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> AdsorptionReaction | None:
    """Build the current AdsorptionReaction for one member from cached state."""
    if is_clique_blocked(G, site, member_index):
        return None

    member_lc: dict | None = getattr(site, "_member_lc", None)
    lc = member_lc.get(member_index) if member_lc is not None else None
    if lc is None or not lc.stable:
        return None
    if lc.energy_occupied is None or lc.energy_unoccupied is None:
        return None

    e_gas = gas_energies.get(site.reactant)
    if e_gas is None:
        return None

    occ = _site_is_occupied(G, site, member_index)
    kind = "desorption" if occ else "adsorption"
    delta_e, barrier, rate = _energetics_cached(
        lc, e_gas, occ,
        temperature              = temperature,
        transmission_coefficient = transmission_coefficient,
    )
    return AdsorptionReaction(
        kind          = kind,
        site          = site,
        member_index  = member_index,
        lateral_class = lc,
        delta_e       = delta_e,
        barrier       = barrier,
        rate          = rate,
    )

