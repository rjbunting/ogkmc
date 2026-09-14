"""
autokmc.kmc.expansion
=======================
On-the-fly growth of the bond-reaction network during the KMC loop.

When a coupling event ``A + B → C`` fires for the first time and produces
a species *C* that the package has not seen before, three things have to
happen before the next KMC step:

1. **Build a Reactant for C** — the gas-phase reference structure +
   energy used downstream.
2. **Find adsorbate sites for C** — so the geometric enumerator
   :func:`autokmc.sites.bond.find_bond_sites` has placements to
   match against.
3. **Derive new templates centred on C** — every dissociation
   ``C → X + Y`` and every coupling ``C + Z → W`` for each known species
   *Z*, then enumerate the corresponding bond-reaction iso-classes on
   the live graph.

This module owns a small **registry** stored on ``G.graph["bond_registry"]``
that tracks the cumulative state across KMC steps:

* ``species``         — ``{canonical_smiles: Reactant}``
* ``adsorbate_sites`` — ``{canonical_smiles: list[AdsorbateSite]}``
* ``templates``       — ``set[(smi_a, smi_b, smi_c)]``

The actual list of enumerated :class:`~autokmc.sites.bond.BondReactionSite`'s
continues to live at ``G.graph["bond_reaction_sites"]`` (with the reverse
index ``G.graph["bond_clique_to_members"]``) — extended by every call to
:func:`expand_bond_sites_for_new_species`.

Note
----
* "Leaf" species discovered as fragments / coupling products of *C* are
  given a Reactant + adsorbate sites but their **own** dissociation /
  coupling templates are NOT enumerated immediately.  Those expansions
  happen lazily when (and only when) a KMC event actually produces that
  species — exactly the same on-the-fly contract that drives this module.
* Deterministically invalid molecular definitions are recorded as
  permanently unavailable.  Calculator, optimisation, I/O, and other
  runtime failures are retried and then raised explicitly; they never mark
  a species as expanded or silently remove chemistry from the network.

Public API
----------
* :func:`initialise_bond_registry`        — bootstrap from the user's
  initial reactants / sites / templates.
* :func:`bond_species_known`              — has the package already
  built this SMILES?
* :func:`expand_bond_sites_for_new_species` — main on-the-fly entry
  point; safe to call from inside the KMC inner loop.
* :func:`expand_bond_sites_after_event`   — convenience hook that takes
  a fired :class:`~autokmc.reactions.bond.BondReaction` and triggers expansion
  for the C-side product when needed.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

import networkx as nx

from autokmc.core.graph_state import (
    BOND_CLIQUE_TO_MEMBERS,
    BOND_SURFACE_NODE_TO_MEMBERS,
    DIFFUSION_CLIQUE_TO_MEMBERS,
    DIFFUSION_SURFACE_NODE_TO_MEMBERS,
    get_bond_reaction_sites,
    get_bond_registry,
    get_diffusion_sites,
    set_bond_reaction_sites,
    set_bond_registry,
    set_diffusion_sites,
)
from autokmc.sites.adsorbate import AdsorbateSite, find_adsorbate_sites
from autokmc.sites.bond import (
    BondReactionTemplate,
    BondReactionSite,
    derive_dissociation_templates,
    derive_coupling_templates,
    find_bond_sites,
    prune_unstable_bond_sites,
    rebuild_bond_reverse_indexes,
    _prune_one_per_adsorption_triple,
    _canon_smiles,
)
from autokmc.sites.diffusion import (
    DiffusionSite,
    find_diffusion_sites,
    rebuild_diffusion_reverse_indexes,
)
from autokmc.sites.identity import SiteId, site_identifier
from autokmc.species.reactant import (
    Reactant,
    ReactantDefinitionError,
    build_reactant,
)
from autokmc.io.calculators import CalculatorConfigError
from autokmc.species.smiles import (
    canonical_atom_inventory_smiles, reactant_atom_inventory_smiles,
)
from autokmc.core.constants import (
    BOND_TOLERANCE,
    BOND_MAX_HOPS,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    BOND_PRUNE_WITH_CALCULATOR,
    BOND_GAS_LIFT_HEIGHT,
    CO_FACTOR,
    CONTACT_FACTOR,
    DIFFUSION_MAX_HOPS,
    HULL_TOL,
    KABSCH_MAX_MAPPINGS,
    MAX_PAIR_SHELLS,
    NL_MULT_DEFAULT,
    NN_DISTANCE,
    N_ADSORBATE_RESTARTS,
    N_SHELLS_DEFAULT,
    OPT_FACTOR,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
    RANDOM_SEED,
    REPULSION_WEIGHT,
    SITE_REPULSION_CUTOFF,
    STANDOFF_FACTOR,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_OPTIMIZER

_log = get_logger(__name__)
_T = TypeVar("_T")
_EXPANSION_MAX_ATTEMPTS = 3


class SpeciesExpansionError(RuntimeError):
    """A retryable runtime operation exhausted its expansion attempts."""


# ---------------------------------------------------------------------------
# Registry helpers — state lives on G.graph["bond_registry"]
# ---------------------------------------------------------------------------

def _registry(G: nx.Graph) -> dict:
    """Return the lazily-initialised on-graph registry dict."""
    reg = get_bond_registry(G)
    if not reg:
        reg = {}
        set_bond_registry(G, reg)
    # Back-fill registries created by earlier releases.
    reg.setdefault("species", {})
    reg.setdefault("adsorbate_sites", {})
    reg.setdefault("templates", set())
    reg.setdefault("expanded_species", set())
    reg.setdefault("expansion_failures", {})
    reg.setdefault("invalid_templates", [])
    return reg


def _failure_record(reg: dict, smi: str, stage: str) -> dict:
    """Return the persistent diagnostic record for one expansion stage."""
    by_species = reg["expansion_failures"].setdefault(smi, {})
    return by_species.setdefault(stage, {"attempts": 0})


def _retry_expansion_operation(
    reg: dict,
    smi: str,
    stage: str,
    operation: Callable[[], _T],
    *,
    max_attempts: int = _EXPANSION_MAX_ATTEMPTS,
) -> _T:
    """Run an expansion operation with bounded retries and diagnostics.

    Deterministic molecular-definition failures and configuration/dependency
    errors remain explicit and are not retried.  Other failures are retried
    because calculator services and filesystem-backed caches can fail
    transiently during a long run.
    """
    existing_record = (
        reg["expansion_failures"].get(smi, {}).get(stage)
    )
    prior_attempts = (
        int(existing_record.get("attempts", 0))
        if isinstance(existing_record, dict)
        else 0
    )
    for local_attempt in range(1, max_attempts + 1):
        try:
            result = operation()
        except ReactantDefinitionError as exc:
            record = _failure_record(reg, smi, stage)
            record.update(
                attempts=prior_attempts + local_attempt,
                status="permanent_invalid",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        except (CalculatorConfigError, ImportError) as exc:
            record = _failure_record(reg, smi, stage)
            record.update(
                attempts=prior_attempts + local_attempt,
                status="fatal",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        except Exception as exc:
            record = _failure_record(reg, smi, stage)
            record.update(
                attempts=prior_attempts + local_attempt,
                status=(
                    "retryable_failure"
                    if local_attempt < max_attempts
                    else "retry_exhausted"
                ),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if local_attempt < max_attempts:
                _log.warning(
                    "Runtime expansion stage %s for species %r failed "
                    "(attempt %d/%d): %s; retrying",
                    stage,
                    smi,
                    local_attempt,
                    max_attempts,
                    exc,
                )
                continue
            raise SpeciesExpansionError(
                f"runtime expansion stage {stage!r} for species {smi!r} "
                f"failed after {max_attempts} attempts: {exc}"
            ) from exc

        if local_attempt > 1 or existing_record is not None:
            record = _failure_record(reg, smi, stage)
            record.update(
                attempts=prior_attempts + local_attempt,
                status="recovered",
            )
        return result

    raise AssertionError("unreachable expansion retry state")


def initialise_bond_registry(
    G: nx.Graph,
    *,
    reactants: Reactant | Iterable[Reactant] | Mapping[str, Reactant],
    adsorbate_sites: Iterable[AdsorbateSite] | Mapping[str, Iterable[AdsorbateSite]] | None = None,
    templates: Iterable[BondReactionTemplate] | None = None,
    bond_sites: Iterable[BondReactionSite] | None = None,
    expanded_smiles: Iterable[str] | None = None,
) -> dict:
    """Bootstrap the on-graph registry from the user's initial state.

    Call this **once** after running the static
    :func:`autokmc.sites.bond.find_bond_sites` so subsequent calls to
    :func:`expand_bond_sites_for_new_species` see every species, site and
    template that the user has already built.

    Parameters
    ----------
    G : nx.Graph
        Live surface + adsorbate graph.
    reactants : Reactant | iterable[Reactant] | dict[smiles, Reactant]
        Initial set of known reactants.
    adsorbate_sites :
        Initial set of materialised :class:`AdsorbateSite`'s — either a
        flat iterable (regrouped here by ``site.reactant``) or a dict
        keyed by SMILES.
    templates :
        Initial set of :class:`BondReactionTemplate`'s already enumerated.
    bond_sites :
        Initial list of :class:`BondReactionSite`'s.  When supplied, the
        graph attribute ``G.graph["bond_reaction_sites"]`` is set to this
        list so future expansion appends to it.
    expanded_smiles : iterable[str] | None
        SMILES of species whose bond templates have already been fully
        derived and enumerated (i.e. they were the *subject* of
        :func:`~autokmc.reactions.templates.derive_bond_templates`, not merely
        auto-built as leaf / product nodes).  These species are marked in
        ``reg["expanded_species"]`` so that
        :func:`expand_bond_sites_for_new_species` does not attempt to
        re-derive their templates.  Typically the user-provided reactant
        SMILES.

    Returns
    -------
    dict
        The on-graph registry (also retrievable via
        ``G.graph["bond_registry"]``).
    """
    reg = _registry(G)

    # First, register the known reactants.
    if isinstance(reactants, Reactant):
        items: list[tuple[str, Reactant]] = [(reactants.smiles, reactants)]
    elif isinstance(reactants, Mapping):
        items = [(str(k), v) for k, v in reactants.items()]
    else:
        reactant_items = cast(Iterable[Reactant], reactants)
        items = [(reactant.smiles, reactant) for reactant in reactant_items]
    pending = dict(reg["species"])
    inventory_labels = {
        reactant_atom_inventory_smiles(reactant): label
        for label, reactant in pending.items() if isinstance(reactant, Reactant)
    }
    for smi, r in items:
        label = _canon_smiles(smi)
        previous = pending.get(label)
        if previous is not None and previous is not r:
            raise ValueError(f"Species label {label!r} is already registered; reuse its reactant")
        if isinstance(r, Reactant):
            inventory = reactant_atom_inventory_smiles(r)
            previous_label = inventory_labels.get(inventory)
            if previous_label is not None and previous_label != label:
                raise ValueError(
                    f"Species {label!r} duplicates the atom inventory of {previous_label!r}; "
                    "reuse the registered species"
                )
            inventory_labels[inventory] = label
        pending[label] = r
    reg["species"].update(pending)

    # Next, register their adsorbate sites.
    if adsorbate_sites is not None:
        if isinstance(adsorbate_sites, Mapping):
            for smi, sites in adsorbate_sites.items():
                reg["adsorbate_sites"][_canon_smiles(smi)] = list(sites)
        else:
            flat_sites = cast(Iterable[AdsorbateSite], adsorbate_sites)
            for s in flat_sites:
                reg["adsorbate_sites"].setdefault(
                    _canon_smiles(s.reactant), []
                ).append(s)

    # The reaction templates are registered after their species and sites.
    if templates is not None:
        for t in templates:
            reg["templates"].add((t.smiles_a, t.smiles_b, t.smiles_c))

    # Species expanded during initialization do not need to be derived again
    # during the KMC loop, so mark them here.
    if expanded_smiles is not None:
        for smi in expanded_smiles:
            cs = _canon_smiles(smi)
            if cs:
                reg["expanded_species"].add(cs)

    # Finally, register the bond sites. The graph remains the primary store for
    # compatibility with existing callers.
    if bond_sites is not None:
        set_bond_reaction_sites(G, bond_sites)

    return reg


def bond_species_known(G: nx.Graph, smiles: str) -> bool:
    """Return True iff *smiles* is already in the bond-reaction registry."""
    reg = get_bond_registry(G)
    if not reg:
        return False
    if _canon_smiles(smiles) in reg["species"]:
        return True
    inventory = canonical_atom_inventory_smiles(smiles)
    return any(
        isinstance(reactant, Reactant)
        and reactant_atom_inventory_smiles(reactant) == inventory
        for reactant in reg["species"].values()
    )


def _rebuild_bond_reverse_indexes(
    G: nx.Graph,
    bond_sites: Iterable[BondReactionSite],
) -> None:
    """Rebuild bond reverse indexes from the complete active site list."""
    rebuild_bond_reverse_indexes(G, bond_sites)


def _preserve_reverse_index(G: nx.Graph, key: str) -> dict | None:
    """Keep an existing index object before an enumerator replaces its key."""
    if key not in G.graph:
        return None
    index = G.graph.get(key)
    return index if isinstance(index, dict) else {}


def _append_diffusion_reverse_indexes(
    G: nx.Graph,
    clique_index: dict,
    surface_index: dict,
    sites: Iterable[DiffusionSite],
) -> None:
    for site in sites:
        for member_index, _ in enumerate(site.member_node_ids):
            site_a, member_a, site_b, member_b = site.members[member_index]
            cliques: list[frozenset] = []
            member_cliques_a = getattr(site_a, "_member_cliques", None)
            member_cliques_b = getattr(site_b, "_member_cliques", None)
            if (
                member_cliques_a is not None
                and member_a < len(member_cliques_a)
            ):
                cliques.extend(member_cliques_a[member_a])
            if (
                member_cliques_b is not None
                and member_b < len(member_cliques_b)
            ):
                cliques.extend(member_cliques_b[member_b])
            for clique in cliques:
                clique_index.setdefault(clique, []).append(
                    (site, member_index)
                )
                for surface_id in clique:
                    surface_index.setdefault(int(surface_id), []).append(
                        (site, member_index)
                    )


def _append_bond_reverse_indexes(
    clique_index: dict,
    surface_index: dict,
    sites: Iterable[BondReactionSite],
) -> None:
    for site in sites:
        for member_index, (cliques_a, cliques_b, cliques_c) in enumerate(
            site._member_cliques
        ):
            for clique in (*cliques_a, *cliques_b, *cliques_c):
                clique_index.setdefault(clique, []).append(
                    (site, member_index)
                )
                for surface_id in clique:
                    surface_index.setdefault(int(surface_id), []).append(
                        (site, member_index)
                    )


# ---------------------------------------------------------------------------
# Internal — build Reactant + AdsorbateSites for a single species
# ---------------------------------------------------------------------------

def _ensure_species_known(
    G: nx.Graph,
    smi: str,
    reg: dict,
    *,
    calculator,
    frozen_indices: list[int] | None,
    nl_mult: float,
    random_seed: int,
    adsorption_prune_fmax: float,
    adsorption_prune_max_steps: int,
    reactant_fmax: float,
    reactant_max_steps: int,
    optimizer: str,
    optimizer_kwargs: dict[str, Any] | None,
    anchor_k_max: int | None,
    adsorbate_bond_tolerance: float,
    adsorbate_n_shells_anchor: int | None,
    adsorbate_n_shells_pair: int,
    co_bond_factor: float,
    anchor_bond_factor: float,
    anchor_repulsion_weight: float,
    site_repulsion_cutoff: float | None,
    adsorbate_contact_factor: float,
    adsorbate_standoff_factor: float,
    adsorbate_rotational_restarts: int,
    typical_neighbor_distance: float,
    adsorbate_max_pair_shells: int,
    anchor_hull_tolerance: float,
    kabsch_max_mappings: int,
    add_hydrogens: bool,
    verbose: bool,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
) -> bool:
    """Build a Reactant + find adsorbate sites for *smi* if not already known.

    Returns ``True`` on success and ``False`` only for a deterministic,
    permanently invalid molecular definition.  Runtime failures are retried
    and then raised as :class:`SpeciesExpansionError`.
    """
    if smi in reg["species"]:
        existing = reg["species"][smi]
        if existing is None:
            status = (
                reg["expansion_failures"]
                .get(smi, {})
                .get("build_reactant", {})
                .get("status")
            )
            if status == "permanent_invalid":
                return False
            # Legacy checkpoints used ``None`` for every failure, including
            # transient backend errors.  Retry those records instead of
            # silently preserving an incomplete reaction network.
            del reg["species"][smi]
            reg["adsorbate_sites"].pop(smi, None)
        elif smi in reg["adsorbate_sites"]:
            return True

    if verbose:
        print(
            f"  [KMC] species {smi!r} is new to the registry; "
            "building gas reference and adsorbate placements"
        )

    r = reg["species"].get(smi)
    if r is None:
        try:
            # Newly-discovered species introduced during a KMC run should
            # default to zero partial pressure (they are produced on-surface
            # and are not assumed to be present in the gas phase unless the
            # user explicitly adds them to the config).
            r = _retry_expansion_operation(
                reg,
                smi,
                "build_reactant",
                lambda: build_reactant(
                    smi,
                    calculator=calculator,
                    add_hydrogens=add_hydrogens,
                    fmax=reactant_fmax,
                    steps=reactant_max_steps,
                    nl_mult=nl_mult,
                    random_seed=random_seed,
                    partial_pressure_bar=0.0,
                    free_energy_options=free_energy_options,
                    free_energy_temperature_k=free_energy_temperature_k,
                    vib_cache_root=vib_cache_root,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                ),
            )
        except ReactantDefinitionError as exc:
            _log.error(
                "Runtime expansion rejected species %r permanently: %s",
                smi,
                exc,
            )
            reg["species"][smi] = None
            reg["adsorbate_sites"][smi] = []
            return False
        reg["species"][smi] = r

    sites = _retry_expansion_operation(
        reg,
        smi,
        "find_adsorbate_sites",
        lambda: find_adsorbate_sites(
            G,
            r,
            bond_tolerance=adsorbate_bond_tolerance,
            n_shells_anchor=adsorbate_n_shells_anchor,
            n_shells_pair=adsorbate_n_shells_pair,
            anchor_k_max=anchor_k_max,
            co_factor=co_bond_factor,
            opt_factor=anchor_bond_factor,
            repulsion_weight=anchor_repulsion_weight,
            repulsion_cutoff=site_repulsion_cutoff,
            contact_factor=adsorbate_contact_factor,
            standoff_factor=adsorbate_standoff_factor,
            n_restarts=adsorbate_rotational_restarts,
            nn_distance=typical_neighbor_distance,
            max_pair_shells=adsorbate_max_pair_shells,
            hull_tolerance=anchor_hull_tolerance,
            kabsch_max_mappings=kabsch_max_mappings,
            nl_mult=nl_mult,
            prune_stable_only=True,
            calculator=calculator,
            frozen_indices=frozen_indices,
            prune_fmax=adsorption_prune_fmax,
            prune_max_steps=adsorption_prune_max_steps,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            verbose=verbose,
        )
    )
    reg["adsorbate_sites"][smi] = list(sites)

    if verbose:
        print(
            f"  + species {smi!r}: built Reactant + "
            f"{len(sites)} adsorbate iso-class(es)"
        )
    return True


# ---------------------------------------------------------------------------
# Diffusion discovery, independent of bond-template expansion
# ---------------------------------------------------------------------------

def _pending_diffusion_sites(
    G: nx.Graph, reg: dict, species: Iterable[str],
) -> list[AdsorbateSite]:
    """Find species whose ready adsorption placements have not been searched.

    The diffusion store retains a species key even when no legal hops exist.
    This records a completed search independently of bond-template expansion.
    Adsorption placements for a species are constructed together and stay
    immutable once indexed.
    """
    existing = get_diffusion_sites(G) or {}
    pending: list[AdsorbateSite] = []
    seen: set[SiteId] = set()
    for smi in sorted(set(species)):
        for site in reg.get("adsorbate_sites", {}).get(smi, []):
            if site.reactant in existing:
                continue
            identifier = site_identifier(site)
            if identifier not in seen:
                pending.append(site)
                seen.add(identifier)
    return pending


def _ensure_species_diffusion(
    G: nx.Graph,
    reg: dict,
    cs: str,
    species: Iterable[str],
    *,
    diffusion_max_hops: int,
    diffusion_n_shells_pair: int,
    diffusion_prune_by_ads_pair: bool | None,
    verbose: bool,
) -> None:
    """Discover missing hops and append them without replacing existing rates."""
    species = tuple(sorted(set(species)))
    new_ads_sites = _pending_diffusion_sites(G, reg, species)
    if not new_ads_sites:
        return
    if verbose:
        n_members = sum(len(s.member_node_ids) for s in new_ads_sites)
        print(
            f"  [KMC] diffusion expansion: {len(new_ads_sites)} "
            f"new adsorbate iso-class(es), {n_members} placement(s); "
            "enumerating hop permutations"
        )
    diff_kwargs: dict = dict(
        max_hops            = diffusion_max_hops,
        n_shells_pair       = diffusion_n_shells_pair,
        verbose             = verbose,
    )
    if diffusion_prune_by_ads_pair is not None:
        diff_kwargs["prune_by_adsorption_pair"] = diffusion_prune_by_ads_pair

    existing_diff: dict = dict(get_diffusion_sites(G) or {})
    existing_diff_cliques = _preserve_reverse_index(
        G,
        DIFFUSION_CLIQUE_TO_MEMBERS,
    )
    existing_diff_surfaces = _preserve_reverse_index(
        G,
        DIFFUSION_SURFACE_NODE_TO_MEMBERS,
    )

    def _enumerate_diffusion_sites() -> dict:
        try:
            return find_diffusion_sites(
                G,
                new_ads_sites,
                **diff_kwargs,
            )
        except Exception:
            # The enumerator replaces graph-level stores as it works.
            # Restore the pre-expansion state before retrying or
            # surfacing the terminal error.
            set_diffusion_sites(G, existing_diff)
            if existing_diff_cliques is None:
                G.graph.pop(DIFFUSION_CLIQUE_TO_MEMBERS, None)
            else:
                G.graph[DIFFUSION_CLIQUE_TO_MEMBERS] = (
                    existing_diff_cliques
                )
            if existing_diff_surfaces is None:
                G.graph.pop(DIFFUSION_SURFACE_NODE_TO_MEMBERS, None)
            else:
                G.graph[DIFFUSION_SURFACE_NODE_TO_MEMBERS] = (
                    existing_diff_surfaces
                )
            raise

    new_diff = _retry_expansion_operation(
        reg,
        cs,
        "find_diffusion_sites",
        _enumerate_diffusion_sites,
    )

    # ``find_diffusion_sites`` overwrites ``G.graph["diffusion_sites"]``
    # with whatever it just enumerated.  Merge with existing entries
    # so previously-discovered diffusion channels are preserved.
    merged: dict[str, list[DiffusionSite]] = {
        k: list(v) for k, v in existing_diff.items()
    }
    accepted_new_diffusion: list[DiffusionSite] = []
    for smi, sites in new_diff.items():
        merged.setdefault(smi, [])
        # Avoid duplicate DiffusionSite identity on re-entry.
        seen_ds = {site_identifier(x) for x in merged[smi]}
        for ds in sites:
            identifier = site_identifier(ds)
            if identifier not in seen_ds:
                merged[smi].append(ds)
                seen_ds.add(identifier)
                accepted_new_diffusion.append(ds)
    set_diffusion_sites(G, merged)
    if (
        existing_diff_cliques is not None
        and existing_diff_surfaces is not None
    ):
        _append_diffusion_reverse_indexes(
            G,
            existing_diff_cliques,
            existing_diff_surfaces,
            accepted_new_diffusion,
        )
        G.graph[DIFFUSION_CLIQUE_TO_MEMBERS] = existing_diff_cliques
        G.graph[DIFFUSION_SURFACE_NODE_TO_MEMBERS] = (
            existing_diff_surfaces
        )
    else:
        # Legacy graphs may not have reverse indexes yet; build the
        # complete pair once, after which expansions append to it.
        rebuild_diffusion_reverse_indexes(G, merged)
    if verbose:
        added = sum(len(v) for v in new_diff.values())
        print(
            f"  → diffusion: +{added} diffusion iso-class(es) across "
            f"species {list(species)}"
        )


def expand_bond_sites_for_new_species(
    G: nx.Graph,
    new_smiles: str,
    *,
    calculator,
    frozen_indices: list[int] | None = None,
    bond_max_hops: int = BOND_MAX_HOPS,
    nl_mult: float = NL_MULT_DEFAULT,
    random_seed: int = RANDOM_SEED,
    adsorption_prune_fmax: float = PRUNE_FMAX,
    adsorption_prune_max_steps: int = PRUNE_MAX_STEPS,
    bond_prune_fmax: float = PRUNE_FMAX,
    bond_prune_max_steps: int = PRUNE_MAX_STEPS,
    reactant_fmax: float = 0.05,
    reactant_max_steps: int = 500,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    anchor_k_max: int | None = None,
    adsorbate_bond_tolerance: float = BOND_TOLERANCE,
    adsorbate_n_shells_anchor: int | None = None,
    adsorbate_n_shells_pair: int = N_SHELLS_DEFAULT,
    co_bond_factor: float = CO_FACTOR,
    anchor_bond_factor: float = OPT_FACTOR,
    anchor_repulsion_weight: float = REPULSION_WEIGHT,
    site_repulsion_cutoff: float | None = SITE_REPULSION_CUTOFF,
    adsorbate_contact_factor: float = CONTACT_FACTOR,
    adsorbate_standoff_factor: float = STANDOFF_FACTOR,
    adsorbate_rotational_restarts: int = N_ADSORBATE_RESTARTS,
    typical_neighbor_distance: float = NN_DISTANCE,
    adsorbate_max_pair_shells: int = MAX_PAIR_SHELLS,
    anchor_hull_tolerance: float = HULL_TOL,
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    add_hydrogens: bool = True,
    include_homo_coupling: bool = True,
    include_dissociation: bool = True,
    include_coupling: bool = True,
    deduplicate_iso: bool = True,
    auto_build_leaf_species: bool = True,
    find_diffusion: bool = False,
    diffusion_max_hops: int = DIFFUSION_MAX_HOPS,
    diffusion_n_shells_pair: int = N_SHELLS_DEFAULT,
    diffusion_prune_by_ads_pair: bool | None = None,
    bond_pair_n_shells: int = BOND_PAIR_N_SHELLS,
    bond_prune_by_triple: bool = BOND_PRUNE_BY_TRIPLE,
    bond_prune_with_calculator: bool = BOND_PRUNE_WITH_CALCULATOR,
    gas_lift_height: float = BOND_GAS_LIFT_HEIGHT,
    verbose: bool = False,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
) -> list[BondReactionSite]:
    """Add a newly-formed species to the bond-reaction registry and expand.

    Idempotent: completed bond expansion is not repeated. When requested,
    missing diffusion channels are discovered even for an expanded species
    or a species with no new bond templates.

    Steps
    -----
    1. Canonicalise *new_smiles*; reuse already constructed species and sites.
    2. Build a :class:`~autokmc.species.reactant.Reactant` and find
       :class:`~autokmc.sites.adsorbate.AdsorbateSite`'s for it.
    3. Derive new :class:`BondReactionTemplate`'s centred on the new
       species:

       * **Dissociation** templates ``new_smiles → X + Y`` for every
         cleavable bond.
       * **Coupling** templates ``new_smiles + Z → W`` for every species
         *Z* already in the registry (including ``Z == new_smiles`` when
         *include_homo_coupling* is true).

    4. For every species *X*, *Y*, *W* referenced by the new templates
       that is **not** yet in the registry, build its Reactant + sites
       (a one-shot per species — see :func:`_ensure_species_known`).
    5. When enabled, discover diffusion for the new species and every ready
       species referenced by its templates, preserving existing channels.
    6. Run :func:`autokmc.sites.bond.find_bond_sites` for the new
       templates against the cumulative AdsorbateSite list, append the
       resulting :class:`BondReactionSite`'s to
       ``G.graph["bond_reaction_sites"]`` with continuous global
       ``iso_class`` numbering.

    Parameters
    ----------
    G : nx.Graph
        Live surface + adsorbate graph.
    new_smiles : str
        SMILES of the freshly-formed species.
    calculator
        ASE calculator handed to :func:`autokmc.species.reactant.build_reactant`
        and :func:`autokmc.sites.adsorbate.find_adsorbate_sites`.
    frozen_indices, nl_mult, adsorption_prune_fmax,
    adsorption_prune_max_steps, bond_prune_fmax, bond_prune_max_steps,
    reactant_fmax, reactant_max_steps, anchor_k_max,
    add_hydrogens
        Forwarded to the per-species reactant, adsorption-site enumeration,
        and bond-site pruning stages as applicable.
    bond_max_hops
        Forwarded to :func:`find_bond_sites`.
    bond_types, include_ring_bonds
        Forwarded to :func:`derive_dissociation_templates`.
    include_dissociation, include_coupling
        Enable the same reaction families selected by the run config.
    deduplicate_iso
        Forwarded to :func:`find_bond_sites`.
    auto_build_leaf_species
        Build template products/fragments that are not already registered.
        When false, templates requiring an unavailable leaf are skipped.
    include_homo_coupling : bool
        Include the homo-coupling template ``new_smiles + new_smiles → W``.
        Default ``True``.
    find_diffusion : bool
        Discover missing hops independently of whether new bond templates
        exist. A completed search with no legal hops is also remembered.
    verbose : bool

    Returns
    -------
    list[BondReactionSite]
        The newly enumerated bond-reaction iso-classes (also appended to
        ``G.graph["bond_reaction_sites"]``).
    """
    reg = _registry(G)

    def ensure_species_known(smi: str) -> bool:
        """Forward the typed expansion settings to the species builder."""
        return _ensure_species_known(
            G,
            smi,
            reg,
            calculator=calculator,
            frozen_indices=frozen_indices,
            nl_mult=nl_mult,
            random_seed=random_seed,
            adsorption_prune_fmax=adsorption_prune_fmax,
            adsorption_prune_max_steps=adsorption_prune_max_steps,
            reactant_fmax=reactant_fmax,
            reactant_max_steps=reactant_max_steps,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            anchor_k_max=anchor_k_max,
            adsorbate_bond_tolerance=adsorbate_bond_tolerance,
            adsorbate_n_shells_anchor=adsorbate_n_shells_anchor,
            adsorbate_n_shells_pair=adsorbate_n_shells_pair,
            co_bond_factor=co_bond_factor,
            anchor_bond_factor=anchor_bond_factor,
            anchor_repulsion_weight=anchor_repulsion_weight,
            site_repulsion_cutoff=site_repulsion_cutoff,
            adsorbate_contact_factor=adsorbate_contact_factor,
            adsorbate_standoff_factor=adsorbate_standoff_factor,
            adsorbate_rotational_restarts=adsorbate_rotational_restarts,
            typical_neighbor_distance=typical_neighbor_distance,
            adsorbate_max_pair_shells=adsorbate_max_pair_shells,
            anchor_hull_tolerance=anchor_hull_tolerance,
            kabsch_max_mappings=kabsch_max_mappings,
            add_hydrogens=add_hydrogens,
            verbose=verbose,
            free_energy_options=free_energy_options,
            free_energy_temperature_k=free_energy_temperature_k,
            vib_cache_root=vib_cache_root,
        )

    cs = _canon_smiles(new_smiles)
    if not cs:
        return []
    if cs in reg["expanded_species"]:
        if find_diffusion:
            _ensure_species_diffusion(
                G, reg, cs, [cs],
                diffusion_max_hops=diffusion_max_hops,
                diffusion_n_shells_pair=diffusion_n_shells_pair,
                diffusion_prune_by_ads_pair=diffusion_prune_by_ads_pair,
                verbose=verbose,
            )
        if verbose:
            print(f"  SKIPPED species {cs!r}: already expanded")
        return []

    # First, build the new reactant and its surface sites.
    built_ok = ensure_species_known(cs)
    if not built_ok:
        # Only deterministic molecular-definition failures reach this path.
        # They are permanently classified in ``expansion_failures`` and can
        # safely be excluded without hiding a transient backend problem.
        reg["expanded_species"].add(cs)
        return []

    # Next, derive every new template that contains this species.
    new_tpls: list[BondReactionTemplate] = []
    pending_template_keys: set[tuple[str, str, str]] = set()
    inventories = {
        label: reactant_atom_inventory_smiles(reactant)
        for label, reactant in reg["species"].items()
        if reactant is not None
    }

    # Begin with dissociation templates in which this species forms X and Y.
    if include_dissociation:
        for t in derive_dissociation_templates(
            cs,
            bond_types         = bond_types,
            include_ring_bonds = include_ring_bonds,
            add_hydrogens      = add_hydrogens,
            atom_inventory_smiles = inventories,
        ):
            key = (t.smiles_a, t.smiles_b, t.smiles_c)
            if key not in reg["templates"] and key not in pending_template_keys:
                pending_template_keys.add(key)
                new_tpls.append(t)

    # Then form coupling templates between this species and every known
    # species, including itself.
    if include_coupling:
        known_smiles = [
            smi
            for smi, reactant in reg["species"].items()
            if reactant is not None
        ]  # cs is now in here
        for z in known_smiles:
            if z == cs:
                if not include_homo_coupling:
                    continue
                pair_inputs = [cs]
                kwargs = dict(include_homo=True, include_hetero=False)
            else:
                pair_inputs = [cs, z]
                kwargs = dict(include_homo=False, include_hetero=True)
            for t in derive_coupling_templates(
                pair_inputs, atom_inventory_smiles=inventories, **kwargs,
            ):
                # Keep only templates that contain the species being expanded.
                if cs not in (t.smiles_a, t.smiles_b):
                    continue
                key = (t.smiles_a, t.smiles_b, t.smiles_c)
                if key not in reg["templates"] and key not in pending_template_keys:
                    pending_template_keys.add(key)
                    new_tpls.append(t)

    if not auto_build_leaf_species and new_tpls:
        available_species = {
            smi
            for smi, reactant in reg["species"].items()
            if reactant is not None
        }
        kept: list[BondReactionTemplate] = []
        for template in new_tpls:
            required = {template.smiles_a, template.smiles_b, template.smiles_c}
            missing = sorted(required - available_species)
            if missing:
                _log.warning(
                    "Skipping runtime bond template %s + %s <-> %s because "
                    "auto_build_leaf_species is false and species are missing: %s",
                    template.smiles_a,
                    template.smiles_b,
                    template.smiles_c,
                    missing,
                )
                continue
            kept.append(template)
        new_tpls = kept

    if not new_tpls:
        if verbose:
            print(
                f"  → species {cs!r}: no new templates generated"
            )

    if verbose and new_tpls:
        n_dissoc = sum(1 for t in new_tpls if t.source == "dissociation")
        n_couple = sum(1 for t in new_tpls if t.source == "coupling")
        print(
            f"  [KMC] species {cs!r}: building reaction series from "
            f"{len(new_tpls)} new template(s) "
            f"({n_dissoc} dissociation, {n_couple} coupling)"
        )

    # After the templates are built, materialize every species they reference.
    diffusion_species = {cs}
    unavailable_species: set[str] = set()
    checked_species: set[str] = set()
    for t in new_tpls:
        for smi in (t.smiles_a, t.smiles_b, t.smiles_c):
            diffusion_species.add(smi)
            if auto_build_leaf_species and smi not in checked_species:
                checked_species.add(smi)
                available = ensure_species_known(smi)
                if not available:
                    unavailable_species.add(smi)

    if unavailable_species:
        viable_templates: list[BondReactionTemplate] = []
        for template in new_tpls:
            required = {
                template.smiles_a,
                template.smiles_b,
                template.smiles_c,
            }
            invalid = sorted(required & unavailable_species)
            if not invalid:
                viable_templates.append(template)
                continue
            diagnostic = {
                "smiles_a": template.smiles_a,
                "smiles_b": template.smiles_b,
                "smiles_c": template.smiles_c,
                "reason": "permanently_invalid_species",
                "species": invalid,
            }
            if diagnostic not in reg["invalid_templates"]:
                reg["invalid_templates"].append(diagnostic)
            _log.error(
                "Runtime template %s + %s <-> %s is unavailable because "
                "species definitions are permanently invalid: %s",
                template.smiles_a,
                template.smiles_b,
                template.smiles_c,
                invalid,
            )
        new_tpls = viable_templates

    if find_diffusion:
        _ensure_species_diffusion(
            G, reg, cs, diffusion_species,
            diffusion_max_hops=diffusion_max_hops,
            diffusion_n_shells_pair=diffusion_n_shells_pair,
            diffusion_prune_by_ads_pair=diffusion_prune_by_ads_pair,
            verbose=verbose,
        )

    if not new_tpls:
        # Requested diffusion discovery is complete even without new chemistry.
        reg["expanded_species"].add(cs)
        return []

    # With every referenced species available, enumerate the new bond-reaction
    # iso-classes.
    cumulative_sites: list[AdsorbateSite] = []
    for sites in reg["adsorbate_sites"].values():
        cumulative_sites.extend(sites)
    if verbose:
        print(
            f"  [KMC] bond expansion: enumerating surface permutations for "
            f"{len(new_tpls)} template(s) across {len(cumulative_sites)} "
            "adsorbate iso-class(es)"
        )

    # The bond-site enumerator replaces the graph store. Save the existing
    # sites, enumerate the new sites, merge both sets, and restore the store.
    existing_brs: list[BondReactionSite] = get_bond_reaction_sites(G)
    existing_bond_cliques = _preserve_reverse_index(
        G,
        BOND_CLIQUE_TO_MEMBERS,
    )
    existing_bond_surfaces = _preserve_reverse_index(
        G,
        BOND_SURFACE_NODE_TO_MEMBERS,
    )

    def _enumerate_and_prune_bond_sites() -> list[BondReactionSite]:
        try:
            candidate_sites = find_bond_sites(
                G, cumulative_sites, new_tpls,
                max_hops            = bond_max_hops,
                deduplicate_iso     = deduplicate_iso,
                n_shells_pair       = bond_pair_n_shells,
                # Match the config pipeline: never run the ego-size triple
                # prune before calculator stability pruning, or a stable
                # representative can be discarded in favour of an unstable
                # smaller-ego one.
                prune_by_triple     = False,
                gas_species         = reg["species"],
                gas_lift_height     = float(gas_lift_height),
                verbose             = verbose,
            )

            # Stage-1 calculator-based stability prune of the freshly
            # enumerated iso-classes; followed by a re-application of the
            # iso-class triple prune so the surviving set stays
            # one-per-triple.
            if (
                bond_prune_with_calculator
                and calculator is not None
                and candidate_sites
            ):
                candidate_sites = prune_unstable_bond_sites(
                    G,
                    list(candidate_sites),
                    reg["species"],
                    calculator,
                    frozen_indices=frozen_indices,
                    fmax=bond_prune_fmax,
                    max_steps=bond_prune_max_steps,
                    nl_mult=nl_mult,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    verbose=verbose,
                )
            if bond_prune_by_triple and candidate_sites:
                prefix = " (post-stability)" if (
                    bond_prune_with_calculator and calculator is not None
                ) else ""
                candidate_sites = _prune_one_per_adsorption_triple(
                    candidate_sites,
                    verbose=verbose,
                    prefix=prefix,
                )
            return list(candidate_sites)
        except Exception:
            # Enumeration replaces graph-level stores before returning.  Keep
            # the previous usable network intact across retries and failures.
            set_bond_reaction_sites(G, existing_brs)
            if existing_bond_cliques is None:
                G.graph.pop(BOND_CLIQUE_TO_MEMBERS, None)
            else:
                G.graph[BOND_CLIQUE_TO_MEMBERS] = existing_bond_cliques
            if existing_bond_surfaces is None:
                G.graph.pop(BOND_SURFACE_NODE_TO_MEMBERS, None)
            else:
                G.graph[BOND_SURFACE_NODE_TO_MEMBERS] = existing_bond_surfaces
            raise

    new_brs = _retry_expansion_operation(
        reg,
        cs,
        "find_bond_sites",
        _enumerate_and_prune_bond_sites,
    )

    # Merge the new sites and renumber them so each iso-class remains unique
    # across all expansions.
    combined = existing_brs + list(new_brs)
    for i, brs in enumerate(combined):
        brs.iso_class = i
    set_bond_reaction_sites(G, combined)

    # The enumerators build indexes for only the new subset. Preserve the
    # existing indexes, and append the members that survived pruning.
    if (
        existing_bond_cliques is not None
        and existing_bond_surfaces is not None
    ):
        _append_bond_reverse_indexes(
            existing_bond_cliques,
            existing_bond_surfaces,
            new_brs,
        )
        G.graph[BOND_CLIQUE_TO_MEMBERS] = existing_bond_cliques
        G.graph[BOND_SURFACE_NODE_TO_MEMBERS] = existing_bond_surfaces
    else:
        _rebuild_bond_reverse_indexes(G, combined)

    if verbose:
        print(
            f"  → species {cs!r}: +{len(new_tpls)} template(s), "
            f"+{len(new_brs)} bond iso-class(es)  "
            f"(total: {len(combined)} iso-class(es))"
        )

    for template in new_tpls:
        reg["templates"].add(
            (template.smiles_a, template.smiles_b, template.smiles_c)
        )

    # Finally, mark the species as fully expanded so later calls can return
    # without repeating the work.
    reg["expanded_species"].add(cs)

    return list(new_brs)


# ---------------------------------------------------------------------------
# Convenience hook for the KMC inner loop
# ---------------------------------------------------------------------------

def expand_bond_sites_after_event(
    G: nx.Graph,
    reaction,
    *,
    calculator,
    **expand_kwargs,
) -> list[BondReactionSite]:
    """KMC-loop hook: expand the registry if *reaction* produced a new species.

    Inspects *reaction* and, when it is a
    :class:`~autokmc.reactions.bond.BondReaction` (``kind == "bond"``), triggers
    :func:`expand_bond_sites_for_new_species` for any species that has not
    yet been fully expanded, and checks for missing diffusion when enabled:

    * **Coupling** ``A + B → C``: expands for ``smiles_c``.  This is the
      primary path that introduces a genuinely new product species.
    * **Dissociation** ``C → A + B``: expands for both ``smiles_a`` and
      ``smiles_b``.  A and B may have been pre-built as leaf nodes (Reactant
      and adsorbate sites already in the registry) but their *own* bond
      templates (A + Z → W, etc.) were never derived.  Expansion here
      derives those templates and enumerates the resulting bond iso-classes
      so they are available for subsequent KMC steps.

    Returns the list of newly enumerated :class:`BondReactionSite`'s
    (empty when the reaction is not of the right kind, or when all
    relevant species are already in ``reg["expanded_species"]``).

    Wire it into the KMC step like::

        affected = execute_reaction(G, chosen)
        expand_bond_sites_after_event(
            G, chosen, calculator=calc, frozen_indices=frozen_indices,
            find_diffusion=True,   # also discover NEB hops for the new species
        )
    """
    if getattr(reaction, "kind", None) != "bond":
        return []
    direction = getattr(reaction, "direction", None)

    if direction == "couple":
        # A + B → C: the product C may be a completely new species.
        smi_c = reaction.site.template.smiles_c
        return expand_bond_sites_for_new_species(
            G, smi_c, calculator=calculator, **expand_kwargs,
        )

    if direction == "dissoc":
        # C → A + B: the fragments A and B were pre-built as leaf nodes when
        # C's templates were derived, but their OWN bond templates (A + Z → W)
        # may never have been enumerated.  Expand each one that hasn't been
        # fully expanded yet; the idempotency guard on expanded_species makes
        # double-calling safe (homo-dissociation where A == B is handled
        # correctly: the second call returns [] immediately).
        smi_a = reaction.site.template.smiles_a
        smi_b = reaction.site.template.smiles_b
        new_a = expand_bond_sites_for_new_species(
            G, smi_a, calculator=calculator, **expand_kwargs,
        )
        new_b = expand_bond_sites_for_new_species(
            G, smi_b, calculator=calculator, **expand_kwargs,
        )
        return new_a + new_b

    raise ValueError(f"unknown bond direction {direction!r}")


__all__ = [
    "SpeciesExpansionError",
    "initialise_bond_registry",
    "bond_species_known",
    "expand_bond_sites_for_new_species",
    "expand_bond_sites_after_event",
]
