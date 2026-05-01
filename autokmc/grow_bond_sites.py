"""
autokmc.grow_bond_sites
=======================
On-the-fly growth of the bond-reaction network during the KMC loop.

When a coupling event ``A + B → C`` fires for the first time and produces
a species *C* that the package has not seen before, three things have to
happen before the next KMC step:

1. **Build a Reactant for C** — the gas-phase reference structure +
   energy used downstream.
2. **Find adsorbate sites for C** — so the geometric enumerator
   :func:`autokmc.find_bond_sites.find_bond_sites` has placements to
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

The actual list of enumerated :class:`~autokmc.find_bond_sites.BondReactionSite`'s
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
* Species whose Reactant build or site enumeration fails are recorded
  with ``None`` / ``[]`` in the registry so we do not retry on every
  KMC step.

Public API
----------
* :func:`initialise_bond_registry`        — bootstrap from the user's
  initial reactants / sites / templates.
* :func:`bond_species_known`              — has the package already
  built this SMILES?
* :func:`expand_bond_sites_for_new_species` — main on-the-fly entry
  point; safe to call from inside the KMC inner loop.
* :func:`expand_bond_sites_after_event`   — convenience hook that takes
  a fired :class:`~autokmc.kmc_bond.BondReaction` and triggers expansion
  for the C-side product when needed.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import networkx as nx

from .find_adsorbate_sites import AdsorbateSite, find_adsorbate_sites
from .find_bond_sites import (
    BondReactionTemplate,
    BondReactionSite,
    derive_dissociation_templates,
    derive_coupling_templates,
    find_bond_sites,
    prune_unstable_bond_sites,
    _prune_one_per_adsorption_triple,
    _canon_smiles,
)
from .find_diffusion_sites import DiffusionSite, find_diffusion_sites
from .reactants import Reactant, build_reactant
from .constants import (
    BOND_MAX_HOPS,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    BOND_PRUNE_WITH_CALCULATOR,
    DIFFUSION_MAX_HOPS,
    MAX_PAIR_SHELLS,
    NL_MULT_DEFAULT,
    N_SHELLS_DEFAULT,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
)
from .logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry helpers — state lives on G.graph["bond_registry"]
# ---------------------------------------------------------------------------

def _registry(G: nx.Graph) -> dict:
    """Return the lazily-initialised on-graph registry dict."""
    reg = G.graph.get("bond_registry")
    if reg is None:
        reg = {
            "species":          {},     # smi (canon) → Reactant | None
            "adsorbate_sites":  {},     # smi (canon) → list[AdsorbateSite]
            "templates":        set(),  # set[(smi_a, smi_b, smi_c)]
            # Species for which bond templates have been fully derived and
            # enumerated (not just pre-built as leaf / product nodes).
            # Used by expand_bond_sites_for_new_species as the idempotency
            # guard so that leaf species (pre-built but not template-derived)
            # can still be expanded when they first appear on the surface.
            "expanded_species": set(),  # set[canonical_smiles]
        }
        G.graph["bond_registry"] = reg
    # Back-fill for registries created before this field was added.
    reg.setdefault("expanded_species", set())
    return reg


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
    :func:`autokmc.find_bond_sites.find_bond_sites` so subsequent calls to
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
        :func:`~autokmc.find_bond_sites.derive_bond_templates`, not merely
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

    # Reactants
    if isinstance(reactants, Reactant):
        items: list[tuple[str, Reactant]] = [(reactants.smiles, reactants)]
    elif isinstance(reactants, Mapping):
        items = [(str(k), v) for k, v in reactants.items()]
    else:
        items = [(r.smiles, r) for r in reactants]
    for smi, r in items:
        reg["species"][_canon_smiles(smi)] = r

    # Adsorbate sites
    if adsorbate_sites is not None:
        if isinstance(adsorbate_sites, Mapping):
            for smi, sites in adsorbate_sites.items():
                reg["adsorbate_sites"][_canon_smiles(smi)] = list(sites)
        else:
            for s in adsorbate_sites:
                reg["adsorbate_sites"].setdefault(
                    _canon_smiles(s.reactant), []
                ).append(s)

    # Templates
    if templates is not None:
        for t in templates:
            reg["templates"].add((t.smiles_a, t.smiles_b, t.smiles_c))

    # Mark explicitly-expanded species so the on-the-fly expander won't
    # needlessly re-derive their templates during the KMC loop.
    if expanded_smiles is not None:
        for smi in expanded_smiles:
            cs = _canon_smiles(smi)
            if cs:
                reg["expanded_species"].add(cs)

    # Bond sites — primary store stays on G.graph for compatibility
    if bond_sites is not None:
        G.graph["bond_reaction_sites"] = list(bond_sites)

    return reg


def bond_species_known(G: nx.Graph, smiles: str) -> bool:
    """Return True iff *smiles* is already in the bond-reaction registry."""
    reg = G.graph.get("bond_registry")
    if reg is None:
        return False
    return _canon_smiles(smiles) in reg["species"]


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
    prune_fmax: float,
    prune_max_steps: int,
    add_hydrogens: bool,
    verbose: bool,
) -> bool:
    """Build a Reactant + find adsorbate sites for *smi* if not already known.

    Returns ``True`` on success, ``False`` if the build / enumeration
    failed (the registry still records the failure to prevent retries).
    """
    if smi in reg["species"]:
        return reg["species"][smi] is not None

    try:
        # Newly-discovered species introduced during a KMC run should
        # default to zero partial pressure (they are produced on-surface
        # and are not assumed to be present in the gas phase unless the
        # user explicitly adds them to the config).  Pass
        # partial_pressure_bar=0.0 to enforce this behaviour for
        # on-the-fly builds while leaving the global API default intact.
        r = build_reactant(
            smi,
            calculator            = calculator,
            add_hydrogens         = add_hydrogens,
            nl_mult               = nl_mult,
            partial_pressure_bar = 0.0,
        )
    except Exception as exc:
        _log.warning(
            "expand_bond_sites: build_reactant(%r) failed: %s "
            "— species marked unbuildable",
            smi, exc,
        )
        reg["species"][smi]         = None
        reg["adsorbate_sites"][smi] = []
        return False
    reg["species"][smi] = r

    try:
        sites = find_adsorbate_sites(
            G, r,
            prune_stable_only = True,
            calculator        = calculator,
            frozen_indices    = frozen_indices,
            prune_fmax        = prune_fmax,
            prune_max_steps   = prune_max_steps,
            verbose           = verbose,
        )
    except Exception as exc:
        _log.warning(
            "expand_bond_sites: find_adsorbate_sites(%r) failed: %s",
            smi, exc,
        )
        sites = []
    reg["adsorbate_sites"][smi] = list(sites)

    if verbose:
        print(
            f"  + species {smi!r}: built Reactant + "
            f"{len(sites)} adsorbate iso-class(es)"
        )
    return True


# ---------------------------------------------------------------------------
# Public API — main expansion entry point
# ---------------------------------------------------------------------------

def expand_bond_sites_for_new_species(
    G: nx.Graph,
    new_smiles: str,
    *,
    calculator,
    frozen_indices: list[int] | None = None,
    bond_max_hops: int = BOND_MAX_HOPS,
    surface_apsp_cutoff: int = MAX_PAIR_SHELLS,
    nl_mult: float = NL_MULT_DEFAULT,
    prune_fmax: float = PRUNE_FMAX,
    prune_max_steps: int = PRUNE_MAX_STEPS,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    add_hydrogens: bool = True,
    include_homo_coupling: bool = True,
    find_diffusion: bool = False,
    diffusion_max_hops: int = DIFFUSION_MAX_HOPS,
    diffusion_n_shells_pair: int = N_SHELLS_DEFAULT,
    diffusion_prune_by_ads_pair: bool | None = None,
    bond_pair_n_shells: int = BOND_PAIR_N_SHELLS,
    bond_prune_by_triple: bool = BOND_PRUNE_BY_TRIPLE,
    bond_prune_with_calculator: bool = BOND_PRUNE_WITH_CALCULATOR,
    verbose: bool = False,
) -> list[BondReactionSite]:
    """Add a newly-formed species to the bond-reaction registry and expand.

    Idempotent: if *new_smiles* is already in the registry the function
    returns ``[]`` without rebuilding anything.

    Steps
    -----
    1. Canonicalise *new_smiles*; bail out if already known.
    2. Build a :class:`~autokmc.reactants.Reactant` and find
       :class:`~autokmc.find_adsorbate_sites.AdsorbateSite`'s for it.
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
    5. Run :func:`autokmc.find_bond_sites.find_bond_sites` for the new
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
        ASE calculator handed to :func:`autokmc.reactants.build_reactant`
        and :func:`autokmc.find_adsorbate_sites.find_adsorbate_sites`.
    frozen_indices, nl_mult, prune_fmax, prune_max_steps, add_hydrogens
        Forwarded to the per-species reactant + site enumeration.
    bond_max_hops, surface_apsp_cutoff
        Forwarded to :func:`find_bond_sites`.
    bond_types, include_ring_bonds
        Forwarded to :func:`derive_dissociation_templates`.
    include_homo_coupling : bool
        Include the homo-coupling template ``new_smiles + new_smiles → W``.
        Default ``True``.
    verbose : bool

    Returns
    -------
    list[BondReactionSite]
        The newly enumerated bond-reaction iso-classes (also appended to
        ``G.graph["bond_reaction_sites"]``).
    """
    reg = _registry(G)
    cs = _canon_smiles(new_smiles)
    if not cs:
        return []
    if cs in reg["expanded_species"]:
        if verbose:
            print(f"  ⏭  species {cs!r} already expanded — no expansion needed")
        return []

    # 1–2. Build Reactant + sites for the newly-introduced species.
    built_ok = _ensure_species_known(
        G, cs, reg,
        calculator      = calculator,
        frozen_indices  = frozen_indices,
        nl_mult         = nl_mult,
        prune_fmax      = prune_fmax,
        prune_max_steps = prune_max_steps,
        add_hydrogens   = add_hydrogens,
        verbose         = verbose,
    )
    if not built_ok:
        # Still mark as expanded so we do not retry on every KMC step.
        reg["expanded_species"].add(cs)
        return []

    # 3. Derive new templates centred on cs.
    new_tpls: list[BondReactionTemplate] = []

    # 3a. Dissociation: cs → X + Y
    for t in derive_dissociation_templates(
        cs,
        bond_types         = bond_types,
        include_ring_bonds = include_ring_bonds,
        add_hydrogens      = add_hydrogens,
    ):
        key = (t.smiles_a, t.smiles_b, t.smiles_c)
        if key not in reg["templates"]:
            reg["templates"].add(key)
            new_tpls.append(t)

    # 3b. Coupling: cs + Z → W for every known Z (including cs itself).
    known_smiles = list(reg["species"].keys())  # cs is now in here
    for z in known_smiles:
        if z == cs:
            if not include_homo_coupling:
                continue
            pair_inputs = [cs]
            kwargs = dict(include_homo=True, include_hetero=False)
        else:
            pair_inputs = [cs, z]
            kwargs = dict(include_homo=False, include_hetero=True)
        for t in derive_coupling_templates(pair_inputs, **kwargs):
            # Only keep templates that involve cs (skip Z+Z spurious entries).
            if cs not in (t.smiles_a, t.smiles_b):
                continue
            key = (t.smiles_a, t.smiles_b, t.smiles_c)
            if key not in reg["templates"]:
                reg["templates"].add(key)
                new_tpls.append(t)

    if not new_tpls:
        if verbose:
            print(
                f"  → species {cs!r}: no new templates generated"
            )
        # Mark as expanded so we don't retry on future KMC steps.
        reg["expanded_species"].add(cs)
        return []

    # 4. Ensure every species referenced by new_tpls is known.
    newly_built: list[str] = []
    if cs in reg["adsorbate_sites"]:
        newly_built.append(cs)
    for t in new_tpls:
        for smi in (t.smiles_a, t.smiles_b, t.smiles_c):
            if smi not in reg["species"]:
                _ensure_species_known(
                    G, smi, reg,
                    calculator      = calculator,
                    frozen_indices  = frozen_indices,
                    nl_mult         = nl_mult,
                    prune_fmax      = prune_fmax,
                    prune_max_steps = prune_max_steps,
                    add_hydrogens   = add_hydrogens,
                    verbose         = verbose,
                )
                if reg["adsorbate_sites"].get(smi):
                    newly_built.append(smi)

    # 4b. Discover diffusion site-pairs for every newly-introduced species
    # so the KMC loop can hop them as soon as they appear on the surface.
    # Only the new species' sites are passed to ``find_diffusion_sites`` —
    # results are merged into ``G.graph["diffusion_sites"]`` rather than
    # overwriting it.
    if find_diffusion and newly_built:
        new_ads_sites: list[AdsorbateSite] = []
        seen_ids: set[int] = set()
        for smi in newly_built:
            for s in reg["adsorbate_sites"].get(smi, []):
                if id(s) not in seen_ids:
                    seen_ids.add(id(s))
                    new_ads_sites.append(s)
        if new_ads_sites:
            diff_kwargs: dict = dict(
                max_hops            = diffusion_max_hops,
                n_shells_pair       = diffusion_n_shells_pair,
                surface_apsp_cutoff = max(int(diffusion_max_hops),
                                          int(surface_apsp_cutoff)),
                verbose             = verbose,
            )
            if diffusion_prune_by_ads_pair is not None:
                diff_kwargs["prune_by_adsorption_pair"] = diffusion_prune_by_ads_pair

            existing_diff: dict = dict(G.graph.get("diffusion_sites", {}) or {})
            try:
                new_diff = find_diffusion_sites(G, new_ads_sites, **diff_kwargs)
            except Exception as exc:
                _log.warning(
                    "expand_bond_sites: find_diffusion_sites for new "
                    "species %r failed: %s", newly_built, exc,
                )
                new_diff = {}

            # ``find_diffusion_sites`` overwrites ``G.graph["diffusion_sites"]``
            # with whatever it just enumerated.  Merge with existing entries
            # so previously-discovered diffusion channels are preserved.
            merged: dict[str, list[DiffusionSite]] = {
                k: list(v) for k, v in existing_diff.items()
            }
            for smi, sites in new_diff.items():
                merged.setdefault(smi, [])
                # Avoid duplicate DiffusionSite identity on re-entry.
                seen_ds = {id(x) for x in merged[smi]}
                for ds in sites:
                    if id(ds) not in seen_ds:
                        merged[smi].append(ds)
                        seen_ds.add(id(ds))
            G.graph["diffusion_sites"] = merged
            if verbose:
                added = sum(len(v) for v in new_diff.values())
                print(
                    f"  → diffusion: +{added} site-pair(s) across "
                    f"species {newly_built}"
                )

    # 5. Enumerate bond-reaction iso-classes for the new templates.
    cumulative_sites: list[AdsorbateSite] = []
    for sites in reg["adsorbate_sites"].values():
        cumulative_sites.extend(sites)

    # ``find_bond_sites`` overwrites G.graph["bond_reaction_sites"] with
    # whatever it just enumerated.  Save → enumerate → splice → restore.
    existing_brs: list[BondReactionSite] = list(
        G.graph.get("bond_reaction_sites", [])
    )

    new_brs = find_bond_sites(
        G, cumulative_sites, new_tpls,
        max_hops            = bond_max_hops,
        surface_apsp_cutoff = surface_apsp_cutoff,
        deduplicate_iso     = True,
        n_shells_pair       = bond_pair_n_shells,
        prune_by_triple     = bond_prune_by_triple,
        verbose             = verbose,
    )

    # Stage-1 calculator-based stability prune of the freshly-enumerated
    # iso-classes; followed by a re-application of the iso-class triple
    # prune so the surviving set stays one-per-triple.
    if bond_prune_with_calculator and calculator is not None and new_brs:
        new_brs = prune_unstable_bond_sites(
            G, list(new_brs), reg["species"], calculator,
            frozen_indices = frozen_indices,
            fmax           = prune_fmax,
            max_steps      = prune_max_steps,
            nl_mult        = nl_mult,
            verbose        = verbose,
        )
        if bond_prune_by_triple and new_brs:
            new_brs = _prune_one_per_adsorption_triple(
                new_brs, verbose=verbose, prefix=" (post-stability)",
            )

    # Splice + globally renumber so iso_class is unique across all expansions.
    combined = existing_brs + list(new_brs)
    for i, brs in enumerate(combined):
        brs.iso_class = i
    G.graph["bond_reaction_sites"] = combined

    # ``prune_unstable_bond_sites`` (when invoked above) rebuilt
    # ``bond_clique_to_members`` from the *new* survivors only, wiping
    # entries belonging to ``existing_brs``.  Rebuild the full reverse
    # index from the combined list so the KMC loop sees every event.
    if bond_prune_with_calculator and calculator is not None:
        G.graph["bond_clique_to_members"] = {}
        G.graph["bond_surface_node_to_members"] = {}
        clq_idx = G.graph["bond_clique_to_members"]
        surf_idx = G.graph["bond_surface_node_to_members"]
        for brs in combined:
            for m_idx, (cliques_a, cliques_b, cliques_c) in enumerate(
                brs._member_cliques
            ):
                for clq in (*cliques_a, *cliques_b, *cliques_c):
                    clq_idx.setdefault(clq, []).append((brs, m_idx))
                    for surf_id in clq:
                        surf_idx.setdefault(int(surf_id), []).append(
                            (brs, m_idx)
                        )

    if verbose:
        print(
            f"  → species {cs!r}: +{len(new_tpls)} template(s), "
            f"+{len(new_brs)} bond iso-class(es)  "
            f"(total: {len(combined)} iso-class(es))"
        )

    # Mark this species as fully expanded so future calls are no-ops.
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
    :class:`~autokmc.kmc_bond.BondReaction` (``kind == "bond"``), triggers
    :func:`expand_bond_sites_for_new_species` for any species that has not
    yet been fully expanded:

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

    return []


__all__ = [
    "initialise_bond_registry",
    "bond_species_known",
    "expand_bond_sites_for_new_species",
    "expand_bond_sites_after_event",
]

