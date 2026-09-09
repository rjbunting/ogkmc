"""
autokmc.sites.bond
=======================
Enumerate **bond-changing surface reactions**

    A + B  ⇌  C

between materialised :class:`~autokmc.sites.adsorbate.AdsorbateSite`'s.
Each *bond reaction site* is an unordered triple of adsorbate placements

    (placement_A, placement_B, placement_C)

satisfying two locality constraints (default ``max_hops = 0``):

1. The bonded surface cliques of A and B must be **connected** through the
   surface graph (default: share at least one surface atom).
   They must not claim the same exact adsorption clique, because A and B are
   simultaneously occupied on the reactant side.
2. The bonded surface clique of C must be **connected** to the union
   (A ∪ B) through the surface graph (default: share at least one surface
   atom with A∪B).

This module follows the same separation of concerns as
:mod:`autokmc.sites.adsorbate` / :mod:`autokmc.sites.stability.adsorption`
and :mod:`autokmc.sites.diffusion` / :mod:`autokmc.sites.stability.diffusion`:

* **find_bond_sites** (this module) — defines the iso-class dataclasses
  (:class:`BondReactionTemplate`, :class:`BondReactionLateral`,
  :class:`BondReactionSite`) and the static enumerator
  :func:`find_bond_sites`.
* :mod:`autokmc.sites.stability.bond` — on-the-fly lateral-environment
  classifier and the energy / NEB stability hooks that populate
  :class:`BondReactionLateral`'s energy fields lazily during the KMC loop.

Templates
---------
A :class:`BondReactionTemplate` ties three SMILES (``smiles_a``,
``smiles_b``, ``smiles_c``) together as one reversible bond-change.
Templates are derived from :mod:`autokmc.species.bond_chemistry`:

* :func:`derive_dissociation_templates`  — for each input SMILES *X*,
  produces ``(B, C, X)`` for every single-bond cleavage ``X → B + C``.
* :func:`derive_coupling_templates`      — for each unordered pair of
  input SMILES ``(X, Y)`` (including ``X == Y``), produces ``(X, Y, Z)``
  for every species ``Z`` formed by joining ``X + Y`` with one new bond.
* :func:`derive_bond_templates`          — convenience wrapper that calls
  both of the above and concatenates / deduplicates the results.

The enumerator only emits bond-reaction sites for templates whose three
SMILES *all* have at least one materialised :class:`AdsorbateSite` on the
graph, so the caller is free to derive the broadest possible template set
and let the geometry filter it.

Storage on the graph
--------------------
``G.graph["bond_reaction_sites"]``
    Flat list of :class:`BondReactionSite` returned by
    :func:`find_bond_sites`.

``G.graph["bond_clique_to_members"]``
    ``dict[frozenset, list[(BondReactionSite, member_index)]]`` — for the
    KMC incremental-update path.  Every clique of every endpoint (A, B, C)
    is registered so toggling any one of them reaches every bond reaction
    that touches it.

Public API
----------
* :class:`BondReactionTemplate`        — one (smi_a, smi_b, smi_c) pattern.
* :class:`BondReactionLateral`         — one lateral-interaction class
  (energies / TS / atoms filled in lazily during the KMC loop).
* :class:`BondReactionSite`            — one iso-class of triple placements.
* :func:`derive_dissociation_templates` / :func:`derive_coupling_templates`
  / :func:`derive_bond_templates`       — template generators.
* :func:`find_bond_sites`              — universal triple enumerator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations, product
from math import factorial
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.io.calculators import acquire_calculator
from autokmc.core.pbc import full_pbc_for_cell
from autokmc.core.atom_metadata import apply_atom_metadata, atom_metadata_key
from autokmc.sites.adsorbate import (
    AdsorbateSite,
)
from autokmc.sites.identity import SiteId, member_identifier, site_identifier
from autokmc.sites.diffusion import _member_clique_union, _reactant_orbit_label
from autokmc.sites.stability.adsorption import _surface_bfs_shells
from autokmc.species.smiles import canonical_atom_inventory_smiles
from autokmc.core.constants import (
    BOND_MAX_HOPS,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    NL_MULT_DEFAULT,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_OPTIMIZER

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# SMILES canonicalisation helper
# ---------------------------------------------------------------------------

def _canon_smiles(smi: str) -> str:
    """Return atom-inventory-safe canonical SMILES for bond chemistry."""
    return canonical_atom_inventory_smiles(smi)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BondReactionTemplate:
    """One reversible bond-change pattern  ``A + B ⇌ C``.

    Attributes
    ----------
    smiles_a, smiles_b, smiles_c : str
        Canonical SMILES of the two fragments and the combined species.
    bond_type : str
        RDKit bond order changed by the reaction (``"SINGLE"``, ``"DOUBLE"``,
        …).  ``"SINGLE"`` for templates from
        :func:`autokmc.species.bond_chemistry.combine_fragments`.
    element_a, element_b : str
        Element symbols of the bonding atoms (one in A, one in B / C).
    source : str
        How the template was generated: ``"dissociation"`` (from
        :func:`derive_dissociation_templates`) or ``"coupling"`` (from
        :func:`derive_coupling_templates`).
    """
    smiles_a  : str
    smiles_b  : str
    smiles_c  : str
    bond_type : str = "SINGLE"
    element_a : str = ""
    element_b : str = ""
    source    : str = ""

    def __repr__(self) -> str:
        return (
            f"BondReactionTemplate({self.smiles_a!r}+{self.smiles_b!r}"
            f"⇌{self.smiles_c!r}, {self.bond_type}, {self.source})"
        )

    @property
    def is_symmetric(self) -> bool:
        """True when ``smiles_a == smiles_b`` (A and B are interchangeable)."""
        return self.smiles_a == self.smiles_b


@dataclass
class BondReactionLateral:
    """One lateral-interaction environment of a :class:`BondReactionSite`.

    Mirrors :class:`autokmc.sites.diffusion.DiffusionLateral` but for a
    three-state reversible reaction.  Energies / atoms / TS are filled in
    lazily during the KMC loop by :mod:`autokmc.sites.stability.bond`.

    Attributes
    ----------
    lateral_class : int
        0-based index within the parent :class:`BondReactionSite`'s
        ``lateral_classes`` list.
    ego_graph : nx.Graph | None
        Lateral ego-graph used for iso-class matching (built later by the
        on-the-fly lateral classifier in :mod:`autokmc.sites.stability.bond`).
    n_shells : int
        BFS depth used to build :attr:`ego_graph`.
    members : list[int]
        Indices into the parent :class:`BondReactionSite`'s ``members``
        whose current local environment matches this lateral class.
    energy_ab : float | None
        Potential energy (eV) of the relaxed *A occupied + B occupied,
        C empty* state.
    energy_c : float | None
        Potential energy (eV) of the *C occupied, A and B empty* state.  For
        gas products this is the empty-surface + gas-phase thermodynamic
        reference used by KMC rates, not the molecular precursor energy.
    energy_c_precursor : float | None
        Potential energy (eV) of the relaxed intact molecule above the
        surface used as the physical NEB endpoint for gas-product channels.
    energy_c_gas_reference : float | None
        Potential energy (eV) of the relaxed empty surface/lateral
        environment for a gas-product channel.  Together with the standalone
        gas-molecule energy this reproduces :attr:`energy_c`.
    energy_ts : float | None
        Potential energy (eV) of the highest NEB image between the two
        states above (the climbing-image saddle when ``climb=True``).
    atoms_ab, atoms_c, atoms_ts : Atoms | None
        Relaxed ASE atoms snapshots persisted by
        :class:`autokmc.io.persistence.ReactionWriter`.
    atoms_c_gas_reference, atoms_gas_molecule : Atoms | None
        Separate empty-surface and optimized gas-molecule structures used for
        the thermodynamic C-state reference of a gas-product channel.  The
        empty-surface structure contains no molecule in the vacuum region.
    atoms_ab_initial, atoms_c_initial : Atoms | None
        Pre-optimization endpoint structures supplied to the relaxations.
    atoms_neb_path_initial : list[Atoms] | None
        Interpolated NEB band before any NEB optimization.
    atoms_neb_path : list[Atoms] | None
        Full NEB band — optional, only kept when ``persist_neb_path=True``.
    neb_path_energies : list[float] | None
        Per-image energies along :attr:`atoms_neb_path`.
    stable : bool | None
        ``True`` when both endpoint relaxations and the NEB converged
        without changing surface / adsorbate connectivity; ``False`` on
        a demonstrated stability failure; ``None`` before evaluation or
        after an unresolved numerical failure.
    invalid_reason : str | None
        Human-readable explanation of why this lateral class is invalid.
    last_failure_reason : str | None
        Most recent numerical failure.  Unlike :attr:`invalid_reason`, this
        does not set :attr:`stable` to ``False``; instead, the non-empty value
        suppresses automatic reevaluation of this lateral class.
    """
    lateral_class    : int
    ego_graph        : Any              = None
    n_shells         : int              = 0
    members          : list[int]        = field(default_factory=list)
    energy_ab        : float | None     = None
    energy_c         : float | None     = None
    energy_c_precursor: float | None    = None
    energy_c_gas_reference: float | None = None
    energy_ts        : float | None     = None
    atoms_ab         : Any              = None
    atoms_c          : Any              = None
    atoms_ts         : Any              = None
    atoms_ab_initial : Any              = None
    atoms_c_initial  : Any              = None
    atoms_c_gas_reference: Any          = None
    atoms_gas_molecule: Any             = None
    atoms_neb_path_initial: Any         = None
    atoms_neb_path   : Any              = None
    neb_path_energies: list[float] | None = None
    neb_n_images   : int | None       = None
    neb_n_frames   : int | None       = None
    neb_max_endpoint_displacement: float | None = None
    neb_target_image_spacing: float | None = None
    neb_estimated_image_spacing: float | None = None
    neb_image_count_limited_by: str | None = None
    neb_intermediate_refinement: dict[str, Any] | None = None
    atoms_neb_refinement_initial: Any = None
    atoms_neb_refinement_final: Any = None
    stable           : bool | None      = None
    invalid_reason   : str | None       = None
    last_failure_reason: str | None     = None
    gas_precursor_relaxed: bool | None  = None
    # The free-energy module populates these vibrational fields.
    g_correction_ab  : float | None = None
    g_correction_c   : float | None = None
    g_correction_ts  : float | None = None
    g_ab             : float | None = None
    g_c              : float | None = None
    g_ts             : float | None = None
    zpe_ab           : float | None = None
    zpe_c            : float | None = None
    zpe_ts           : float | None = None
    entropy_ab       : float | None = None
    entropy_c        : float | None = None
    entropy_ts       : float | None = None
    frequencies_ab_ev : list = field(default_factory=list)
    frequencies_c_ev  : list = field(default_factory=list)
    frequencies_ts_ev : list = field(default_factory=list)
    imaginary_ab_ev   : list = field(default_factory=list)
    imaginary_c_ev    : list = field(default_factory=list)
    imaginary_ts_ev   : list = field(default_factory=list)
    vib_indices_ab    : list = field(default_factory=list)
    vib_indices_c     : list = field(default_factory=list)
    vib_indices_ts    : list = field(default_factory=list)
    gas_product       : bool = False
    gas_pressure_bar  : float = 0.0
    #: Atom-correspondence diagnostics chosen for the bond NEB.  For surface
    #: products, ``atom_mapping`` is C graph-node order aligned to the AB
    #: reacting block; for gas products it is the gas atom order.
    atom_matching_method : str | None = None
    atom_mapping         : list = field(default_factory=list)
    matching_diagnostics : dict = field(default_factory=dict)
    # Appended after every pre-existing init field for positional-checkpoint
    # compatibility.
    neb_intermediate_refinement_history: list[dict[str, Any]] = field(
        default_factory=list
    )
    direct_event_status: str | None = None
    direct_event_reason: str | None = None
    direct_event_certificate: dict[str, Any] | None = None
    direct_event_network_signature: str | None = None
    #: Gas C combines a remaining-surface Hessian and isolated-gas modes.
    #: These separate components identify the state to which each mode list
    #: belongs; vib_indices_c refers only to the remaining surface.
    thermochemistry_c_components: dict[str, Any] = field(default_factory=dict)
    #: Accepted endpoint-like/below-endpoint NEB energy, retained in reaction
    #: output and calculation caches without replacing the raw energies.
    ts_energy_diagnostic: dict[str, Any] | None = None
    if TYPE_CHECKING:
        _fingerprint : tuple = field(init=False, repr=False, compare=False)
        _rate_cache : dict = field(init=False, repr=False, compare=False)


@dataclass
class BondReactionSite:
    """One iso-class of bond-reaction triples ``(A, B, C)``.

    Attributes
    ----------
    template : BondReactionTemplate
        The (smi_a, smi_b, smi_c) pattern this iso-class implements.
    iso_class : int
        0-based global index across all bond-reaction iso-classes.
    members : list[tuple[AdsorbateSite, int, AdsorbateSite, int, AdsorbateSite, int]]
        ``(site_a, m_a, site_b, m_b, site_c, m_c)`` for every concrete
        triple folded into this iso-class.  When
        ``template.is_symmetric``, the pair ``(site_a, m_a)`` is always
        canonically ordered ≤ ``(site_b, m_b)`` to avoid double-counting.
    member_node_ids : list[tuple[list[int], list[int], list[int]]]
        ``(A_node_ids, B_node_ids, C_node_ids)`` per member — convenience
        handles into the live graph.
    lateral_classes : list[BondReactionLateral]
        Lazily populated by
        :func:`autokmc.sites.stability.bond.check_bond_site_lateral`.
    """
    template        : BondReactionTemplate
    iso_class       : int
    members         : list[tuple] = field(default_factory=list)
    member_node_ids : list[tuple] = field(default_factory=list)
    lateral_classes : list[BondReactionLateral] = field(default_factory=list)
    #: Triple ego-graph for the *representative* (member 0) — surface-only
    #: BFS around the union of A, B, C bonded cliques with the three
    #: placements stamped on as labelled occupied leaves
    #: (``endpoint_role ∈ {"a","b","c"}`` or ``"ab"`` when symmetric).
    #: Used by :func:`_prune_one_per_adsorption_triple` to pick the
    #: smallest / most-direct iso-class per adsorption triple.
    ego_graph             : Any = None
    n_shells_pair_settled : int = 0
    gas_product           : bool = False
    gas_reactant          : Any = None
    gas_lift_height       : float = 6.0

    # Cached per-member tuples ``(cliques_a, cliques_b, cliques_c)`` —
    # populated by :func:`find_bond_sites`.  Used by :mod:`autokmc.reactions.bond`
    # for the clique-collision applicability guard.
    _member_cliques : list[tuple[tuple[frozenset, ...],
                                 tuple[frozenset, ...],
                                 tuple[frozenset, ...]]] = field(default_factory=list)
    #: Stable KMC identity, assigned lazily once member nodes are available.
    # Keep this after every pre-existing init field so older positional
    # constructors continue to bind ``_member_cliques`` correctly.
    site_id                : str = field(default="", compare=False)
    # Lazily attached so older checkpoints and manual instances retain the
    # established ``hasattr``-based initialisation path.
    if TYPE_CHECKING:
        _lateral_fp_index : dict[tuple, list[BondReactionLateral]] = field(
            init=False, repr=False, compare=False,
        )
        _member_lc : dict[int, BondReactionLateral] = field(
            init=False, repr=False, compare=False,
        )
        applicable_reactions : list[Any] = field(
            init=False, repr=False, compare=False,
        )


def rebuild_bond_reverse_indexes(
    G: nx.Graph,
    bond_sites: Iterable[BondReactionSite] | None = None,
) -> None:
    """Rebuild bond reverse indexes from the complete active site list."""
    active_sites = (
        list(bond_sites)
        if bond_sites is not None
        else list(G.graph.get("bond_reaction_sites", []) or [])
    )
    bond_clique_to_members: dict = {}
    bond_surface_to_members: dict = {}
    for brs in active_sites:
        for m_idx, (cliques_a, cliques_b, cliques_c) in enumerate(
            brs._member_cliques
        ):
            for clq in (*cliques_a, *cliques_b, *cliques_c):
                bond_clique_to_members.setdefault(clq, []).append((brs, m_idx))
                for surf_id in clq:
                    bond_surface_to_members.setdefault(
                        int(surf_id), [],
                    ).append((brs, m_idx))

    G.graph["bond_clique_to_members"] = bond_clique_to_members
    G.graph["bond_surface_node_to_members"] = bond_surface_to_members


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def _template_inventory_maps(atom_inventory_smiles):
    """Keep simulated atom inventories separate from registered feed labels."""
    inventories = {
        _canon_smiles(label): _canon_smiles(inventory)
        for label, inventory in (atom_inventory_smiles or {}).items()
    }
    aliases: dict[str, str] = {}
    for label, inventory in inventories.items():
        aliases.setdefault(inventory, label)
    return inventories, aliases


def derive_dissociation_templates(
    smiles: str | Iterable[str],
    *,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    add_hydrogens: bool = True,
    atom_inventory_smiles: Mapping[str, str] | None = None,
) -> list[BondReactionTemplate]:
    """Generate ``(B, C) → A`` templates from each input SMILES *A*.

    For every single bond cleavage ``A → B + C`` discovered by
    :func:`autokmc.species.bond_chemistry.get_all_fragments` (with
    ``strip_dummies=True``) one :class:`BondReactionTemplate` is emitted
    with ``smiles_c = canonical(A)`` and ``(smiles_a, smiles_b)`` set to
    the two fragments.

    Parameters
    ----------
    smiles, bond_types, include_ring_bonds, add_hydrogens
        Forwarded to :func:`autokmc.species.bond_chemistry.get_all_fragments`
        (the latter receives ``strip_dummies=True``).
    atom_inventory_smiles
        Registered species label to SMILES with its simulated H atoms explicit.
        Overrides ``add_hydrogens`` for those species and reuses known labels
        for generated fragments, preserving feed identity and gas pressure.
    """
    from autokmc.species.bond_chemistry import get_all_fragments

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    out: list[BondReactionTemplate] = []
    seen: set[tuple[str, str, str]] = set()

    inventories, aliases = _template_inventory_maps(atom_inventory_smiles)
    for raw_a in smiles_list:
        big = _canon_smiles(raw_a)
        try:
            pairs = get_all_fragments(
                inventories.get(big, raw_a),
                add_hydrogens      = False if big in inventories else add_hydrogens,
                bond_types         = bond_types,
                include_ring_bonds = include_ring_bonds,
                strip_dummies      = True,
            )
        except Exception as exc:
            _log.warning(
                "derive_dissociation_templates: get_all_fragments(%r) failed: %s",
                raw_a, exc,
            )
            continue

        for p in pairs:
            smi_a = _canon_smiles(p.smiles_a)
            smi_b = _canon_smiles(p.smiles_b)
            smi_a = aliases.get(smi_a, smi_a)
            smi_b = aliases.get(smi_b, smi_b)
            # Canonicalise unordered (smi_a, smi_b)
            ordered = tuple(sorted((smi_a, smi_b)))
            key = (ordered[0], ordered[1], big)
            if key in seen:
                continue
            seen.add(key)
            out.append(BondReactionTemplate(
                smiles_a  = ordered[0],
                smiles_b  = ordered[1],
                smiles_c  = big,
                bond_type = p.bond_type,
                element_a = p.element_a,
                element_b = p.element_b,
                source    = "dissociation",
            ))

    _log.debug(
        "derive_dissociation_templates: %d template(s) from %d SMILES",
        len(out), len(smiles_list),
    )
    return out


def derive_coupling_templates(
    smiles: str | Iterable[str],
    *,
    include_homo: bool = True,
    include_hetero: bool = True,
    add_hydrogens: bool = False,
    atom_inventory_smiles: Mapping[str, str] | None = None,
) -> list[BondReactionTemplate]:
    """Generate ``A + B → C`` templates from every (unordered) pair.

    For every unordered pair ``(A, B)`` of input SMILES (including the
    homo-pair ``(A, A)`` when ``include_homo``), one
    :class:`BondReactionTemplate` is emitted per
    :class:`~autokmc.species.bond_chemistry.CombinedSpecies` returned by
    :func:`autokmc.species.bond_chemistry.combine_fragments`.

    Parameters
    ----------
    smiles : str or iterable[str]
        SMILES list (or single SMILES); duplicates are removed.
    include_homo : bool
        Include homo-coupling pairs ``(A, A)``.  Default ``True``.
    include_hetero : bool
        Include hetero-coupling pairs ``(A, B)`` with ``A != B``.
        Default ``True``.
    add_hydrogens : bool
        Materialize implicit H atoms before joining fragments. Default ``False``.
    atom_inventory_smiles
        Registered species label to SMILES with its simulated H atoms explicit.
        Overrides ``add_hydrogens`` for those species; templates keep feed labels.
    """
    from autokmc.species.bond_chemistry import combine_fragments

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    resolved = {
        label: canonical_atom_inventory_smiles(label, add_hydrogens=add_hydrogens)
        for label in smiles_list
    }
    resolved.update(atom_inventory_smiles or {})
    inventories, aliases = _template_inventory_maps(resolved)

    # Canonicalise + deduplicate inputs.
    canon: list[str] = []
    seen_in: set[str] = set()
    for s in smiles_list:
        cs = _canon_smiles(s)
        if cs and cs not in seen_in:
            seen_in.add(cs)
            canon.append(cs)

    out: list[BondReactionTemplate] = []
    seen: set[tuple[str, str, str]] = set()

    for i in range(len(canon)):
        for j in range(i, len(canon)):
            if i == j and not include_homo:
                continue
            if i != j and not include_hetero:
                continue
            smi_a, smi_b = canon[i], canon[j]
            try:
                products = combine_fragments(
                    inventories.get(smi_a, smi_a),
                    inventories.get(smi_b, smi_b),
                )
            except Exception as exc:
                _log.warning(
                    "derive_coupling_templates: combine_fragments(%r, %r) failed: %s",
                    smi_a, smi_b, exc,
                )
                continue
            for sp in products:
                big = _canon_smiles(sp.smiles)
                big = aliases.get(big, big)
                ordered = tuple(sorted((smi_a, smi_b)))
                key = (ordered[0], ordered[1], big)
                if key in seen:
                    continue
                seen.add(key)
                out.append(BondReactionTemplate(
                    smiles_a  = ordered[0],
                    smiles_b  = ordered[1],
                    smiles_c  = big,
                    bond_type = sp.bond_type,
                    element_a = sp.element_a,
                    element_b = sp.element_b,
                    source    = "coupling",
                ))

    _log.debug(
        "derive_coupling_templates: %d template(s) from %d SMILES",
        len(out), len(canon),
    )
    return out


def derive_bond_templates(
    smiles: str | Iterable[str],
    *,
    include_dissociation: bool = True,
    include_coupling: bool = True,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    add_hydrogens: bool = True,
    include_homo_coupling: bool = True,
    include_hetero_coupling: bool = True,
    atom_inventory_smiles: Mapping[str, str] | None = None,
) -> list[BondReactionTemplate]:
    """Convenience: union of dissociation + coupling templates.

    The combined list is deduplicated by ``(smiles_a, smiles_b, smiles_c)``.
    ``atom_inventory_smiles`` preserves per-species H policies without changing
    the labels used by existing adsorption sites and gas reservoirs.
    """
    # Both families consume the same inventory, including when the caller
    # supplies a generator instead of a reusable sequence.
    smiles = [smiles] if isinstance(smiles, str) else list(smiles)
    inventories = {
        label: canonical_atom_inventory_smiles(label, add_hydrogens=add_hydrogens)
        for label in smiles
    }
    inventories.update(atom_inventory_smiles or {})
    out: list[BondReactionTemplate] = []
    if include_dissociation:
        out.extend(derive_dissociation_templates(
            smiles,
            bond_types         = bond_types,
            include_ring_bonds = include_ring_bonds,
            add_hydrogens      = add_hydrogens,
            atom_inventory_smiles = inventories,
        ))
    if include_coupling:
        out.extend(derive_coupling_templates(
            smiles,
            include_homo   = include_homo_coupling,
            include_hetero = include_hetero_coupling,
            atom_inventory_smiles = inventories,
        ))

    # Deduplicate by (smi_a, smi_b, smi_c) — keep first occurrence so
    # dissociation entries (which carry an authoritative bond_type) win.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[BondReactionTemplate] = []
    for t in out:
        key = (t.smiles_a, t.smiles_b, t.smiles_c)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


# ---------------------------------------------------------------------------
# Internal helpers — graph enumeration
# ---------------------------------------------------------------------------

def _flatten_sites(
    sites: Iterable[AdsorbateSite],
) -> list[tuple[AdsorbateSite, int, frozenset[int]]]:
    """Return ``[(site, m_idx, clique_union), ...]`` for every member."""
    out: list[tuple[AdsorbateSite, int, frozenset[int]]] = []
    for s in sites:
        for m_idx in range(len(s.member_node_ids)):
            clq_union = _member_clique_union(s, m_idx)
            if not clq_union:
                continue
            out.append((s, m_idx, clq_union))
    # Stable canonical key — every AdsorbateSite has a unique iso_class
    # within its SMILES; we sort by (smiles, iso_class, m_idx) for
    # cross-SMILES determinism.
    out.sort(key=lambda t: (str(t[0].reactant), int(t[0].iso_class), int(t[1])))
    return out


def _placement_key(site: AdsorbateSite, m_idx: int) -> tuple[str, int, int, str]:
    """Canonical sortable key for one (site, m_idx) placement."""
    return (
        _canon_smiles(site.reactant),
        int(site.iso_class),
        int(m_idx),
        site_identifier(site),
    )


def _placement_cliques(
    site: AdsorbateSite, m_idx: int,
) -> tuple[frozenset, ...]:
    """Bonded surface cliques of one placement (per-atom tuple)."""
    member_cliques = getattr(site, "_member_cliques", None)
    if member_cliques is None or m_idx >= len(member_cliques):
        return tuple()
    return tuple(member_cliques[m_idx])


def _placements_share_exact_clique(
    cliques_a: Iterable[frozenset],
    cliques_b: Iterable[frozenset],
) -> bool:
    """Return True when two simultaneously-occupied placements collide."""
    return bool(set(cliques_a) & set(cliques_b))


def _surface_node_index_for_placements(
    flat: list[tuple[AdsorbateSite, int, frozenset[int]]],
) -> dict[int, list[int]]:
    """Map surface atom id to placement indexes touching that atom."""
    index: dict[int, list[int]] = {}
    for i, (_, _, clq_union) in enumerate(flat):
        for surf_id in clq_union:
            index.setdefault(int(surf_id), []).append(i)
    return index


def _nearby_placement_indices(
    G: nx.Graph,
    surface_index: dict[int, list[int]],
    seed_clique: frozenset[int],
    max_hops: int,
) -> list[int]:
    """Return placement indexes whose clique intersects the local surface shell."""
    if not seed_clique:
        return []
    shell = _surface_bfs_shells(G, seed_clique, max(0, int(max_hops)))
    seen: set[int] = set()
    for surf_id in shell:
        for idx in surface_index.get(int(surf_id), ()):
            seen.add(int(idx))
    return sorted(seen)


@dataclass(slots=True)
class _PlacementRecord:
    """Precomputed hot-loop data for one materialised placement.

    ``find_bond_sites`` used to recover all of these values independently
    inside the nested A/B/C loops.  In particular, its sortable key called
    RDKit canonicalisation and stable-ID construction for every concrete
    triple.  A record is built once per placement and can safely be reused by
    every template referencing the species.
    """

    site: AdsorbateSite
    member_index: int
    species: str
    key: tuple[str, int, int, str]
    clique_union: frozenset[int]
    cliques: tuple[frozenset, ...]
    clique_set: frozenset[frozenset]
    node_ids: tuple[int, ...]
    n_shells: int
    surface_shell: frozenset[int]
    sort_key: tuple[str, int, int]


def _prepare_placement_records(
    G: nx.Graph,
    by_smiles: dict[str, list[AdsorbateSite]],
    n_shells_pair: int,
) -> dict[str, list[_PlacementRecord]]:
    """Build complete placement records, sharing repeated surface shells."""
    records_by_smiles: dict[str, list[_PlacementRecord]] = {}
    shell_cache: dict[tuple[frozenset[int], int], frozenset[int]] = {}
    minimum_depth = int(n_shells_pair)

    for species, sites in by_smiles.items():
        records: list[_PlacementRecord] = []
        for site in sites:
            canonical_reactant = _canon_smiles(site.reactant)
            stable_site_id = site_identifier(site)
            depth = max(
                int(getattr(site, "n_shells_settled", 0) or 0),
                minimum_depth,
            )
            for member_index, raw_node_ids in enumerate(site.member_node_ids):
                clique_union = _member_clique_union(site, member_index)
                if not clique_union:
                    continue
                clique_union = frozenset(int(n) for n in clique_union)
                cliques = _placement_cliques(site, member_index)
                shell_key = (clique_union, depth)
                surface_shell = shell_cache.get(shell_key)
                if surface_shell is None:
                    surface_shell = frozenset(
                        int(n)
                        for n in _surface_bfs_shells(
                            G, clique_union, max(0, depth),
                        )
                    )
                    shell_cache[shell_key] = surface_shell
                records.append(_PlacementRecord(
                    site=site,
                    member_index=int(member_index),
                    species=species,
                    key=(
                        canonical_reactant,
                        int(site.iso_class),
                        int(member_index),
                        stable_site_id,
                    ),
                    clique_union=clique_union,
                    cliques=cliques,
                    clique_set=frozenset(cliques),
                    node_ids=tuple(int(n) for n in raw_node_ids),
                    n_shells=depth,
                    surface_shell=surface_shell,
                    # Preserve the historical deterministic ordering.
                    sort_key=(
                        str(site.reactant),
                        int(site.iso_class),
                        int(member_index),
                    ),
                ))
        records.sort(key=lambda record: record.sort_key)
        records_by_smiles[species] = records

    return records_by_smiles


def _surface_node_index_for_records(
    records: list[_PlacementRecord],
) -> dict[int, tuple[int, ...]]:
    """Map each surface atom to the placement records touching it."""
    mutable: dict[int, list[int]] = {}
    for record_index, record in enumerate(records):
        for surface_id in record.clique_union:
            mutable.setdefault(int(surface_id), []).append(record_index)
    return {
        surface_id: tuple(record_indexes)
        for surface_id, record_indexes in mutable.items()
    }


class _NearbyPlacementLookup:
    """Memoise local joins for repeated clique unions.

    The target species is part of the cache key, so equal seed cliques cannot
    accidentally reuse indexes belonging to a different placement table.
    """

    def __init__(
        self,
        G: nx.Graph,
        indexes: dict[str, dict[int, tuple[int, ...]]],
    ) -> None:
        self._G = G
        self._indexes = indexes
        self._join_cache: dict[
            tuple[str, frozenset[int], int], tuple[int, ...]
        ] = {}
        self._shell_cache: dict[
            tuple[frozenset[int], int], frozenset[int]
        ] = {}

    def get(
        self,
        target_species: str,
        seed_clique: frozenset[int],
        max_hops: int,
    ) -> tuple[int, ...]:
        hops = max(0, int(max_hops))
        seed = frozenset(int(n) for n in seed_clique)
        cache_key = (target_species, seed, hops)
        cached = self._join_cache.get(cache_key)
        if cached is not None:
            return cached
        if not seed:
            self._join_cache[cache_key] = tuple()
            return tuple()

        if hops == 0:
            shell = seed
        else:
            shell_key = (seed, hops)
            shell = self._shell_cache.get(shell_key)
            if shell is None:
                shell = frozenset(
                    int(n)
                    for n in _surface_bfs_shells(self._G, seed, hops)
                )
                self._shell_cache[shell_key] = shell

        target_index = self._indexes.get(target_species, {})
        nearby: set[int] = set()
        for surface_id in shell:
            nearby.update(target_index.get(int(surface_id), ()))
        result = tuple(sorted(nearby))
        self._join_cache[cache_key] = result
        return result


# ---------------------------------------------------------------------------
# Triple ego-graph + iso-class triple pruning
# ---------------------------------------------------------------------------

def _build_triple_ego_graph(
    G: nx.Graph,
    a_node_ids: list[int],
    b_node_ids: list[int],
    c_node_ids: list[int],
    a_clique_union: frozenset[int],
    b_clique_union: frozenset[int],
    c_clique_union: frozenset[int],
    n_shells_a: int,
    n_shells_b: int,
    n_shells_c: int,
    *,
    is_symmetric: bool,
) -> nx.Graph:
    """Surface-only ego-graph spanning the A, B and C placements of a triple.

    Generalisation of
    :func:`autokmc.sites.diffusion._build_pair_ego_graph` to three
    endpoint placements.  Each placement is stamped with an
    ``endpoint_role`` label so the iso-match keeps the role distinction
    (``"a"`` / ``"b"`` / ``"c"``, collapsed to ``"ab"`` / ``"c"`` when
    *is_symmetric* — A and B are interchangeable on a symmetric template).

    G is **not mutated**.
    """
    role_a = "ab" if is_symmetric else "a"
    role_b = "ab" if is_symmetric else "b"

    endpoint_lists = (
        (a_node_ids, role_a),
        (b_node_ids, role_b),
        (c_node_ids, "c"),
    )
    all_endpoint_nids: list[int] = [
        nid for ids, _ in endpoint_lists for nid in ids if nid in G
    ]
    endpoint_ids: frozenset = frozenset(all_endpoint_nids)

    # Surface-only BFS from each clique union.
    visited_a = _surface_bfs_shells(G, a_clique_union, n_shells_a)
    visited_b = _surface_bfs_shells(G, b_clique_union, n_shells_b)
    visited_c = _surface_bfs_shells(G, c_clique_union, n_shells_c)
    visited: set = (set(visited_a) | set(visited_b) | set(visited_c)) - endpoint_ids

    # Other occupied adsorbates belong to the later lateral classification.
    result = G.subgraph(visited).copy()

    # Stamp each endpoint placement with its role label.
    for ids, role in endpoint_lists:
        for nid in ids:
            if nid not in G:
                continue
            d = G.nodes[nid]
            if nid not in result:
                result.add_node(
                    nid,
                    element        = d.get("element"),
                    atom_arrays    = d.get("atom_arrays", {}),
                    type           = d.get("type", "adsorbate"),
                    iso_class      = int(d.get("iso_class", -1)),
                    reactant       = str(d.get("reactant",  "")),
                    reactant_index = int(d.get("reactant_index", -1)),
                    reactant_orbit = _reactant_orbit_label(d),
                    occupied       = True,
                    endpoint_role  = role,
                )
            else:
                result.nodes[nid]["occupied"]      = True
                result.nodes[nid]["endpoint_role"] = role
            for sib in G.neighbors(nid):
                sib = int(sib)
                if sib in endpoint_ids and sib in result:
                    result.add_edge(nid, sib, **G.edges[nid, sib])
            clq = d.get("clique")
            if clq is not None:
                for surf_id in clq:
                    if surf_id in result and not result.has_edge(nid, surf_id):
                        result.add_edge(nid, surf_id, anchor_bond=True)

    return result


@dataclass(frozen=True, slots=True)
class _TripleEgoBlueprint:
    """Minimal labelled topology needed by triple isomorphism.

    The blueprint deliberately omits coordinates and all other node/edge
    metadata ignored by :func:`_triple_node_match`.  It is cheap to construct,
    hash and canonically encode, and can be materialised as a NetworkX graph
    only when an exact ``GraphMatcher`` fallback is necessary.
    """

    nodes: tuple[tuple[int, tuple[str, str, int, str, int, str, str]], ...]
    edges: tuple[tuple[int, int], ...]


def _triple_match_label(
    data: dict,
    *,
    endpoint_role: str | None = None,
    endpoint_default: bool = False,
) -> tuple[str, str, int, str, int, str, str]:
    """Return the exact node attributes observed by ``_triple_node_match``."""
    node_type = data.get("type", "adsorbate" if endpoint_default else None)
    type_label = "" if node_type is None else str(node_type)
    element = data.get("element")
    element_label = "" if element is None else str(element)
    if node_type != "adsorbate":
        return (type_label, element_label, -1, "", -1, "", atom_metadata_key(data))
    role = (
        endpoint_role
        if endpoint_role is not None
        else (data.get("endpoint_role") or "")
    )
    return (
        type_label,
        element_label,
        int(data.get("iso_class", -1)),
        str(data.get("reactant", "")),
        _reactant_orbit_label(data),
        str(role or ""),
        atom_metadata_key(data),
    )


def _build_triple_ego_blueprint(
    G: nx.Graph,
    record_a: _PlacementRecord,
    record_b: _PlacementRecord,
    record_c: _PlacementRecord | None,
    *,
    is_symmetric: bool,
) -> _TripleEgoBlueprint:
    """Build an occupancy-independent triple topology without copying a graph."""
    role_a = "ab" if is_symmetric else "a"
    role_b = "ab" if is_symmetric else "b"
    endpoint_records = (
        (record_a, role_a),
        (record_b, role_b),
    ) + (((record_c, "c"),) if record_c is not None else ())
    endpoint_roles: dict[int, str] = {
        int(node_id): role
        for record, role in endpoint_records
        for node_id in record.node_ids
        if node_id in G
    }
    endpoint_ids = frozenset(endpoint_roles)

    visited: set[int] = (
        set(record_a.surface_shell)
        | set(record_b.surface_shell)
        | (set(record_c.surface_shell) if record_c is not None else set())
    ) - endpoint_ids
    base_nodes = visited
    labels: dict[int, tuple[str, str, int, str, int, str, str]] = {}
    for node_id in base_nodes:
        labels[int(node_id)] = _triple_match_label(G.nodes[node_id])
    for node_id, role in endpoint_roles.items():
        labels[int(node_id)] = _triple_match_label(
            G.nodes[node_id],
            endpoint_role=role,
            endpoint_default=True,
        )

    edges: set[tuple[int, int]] = set()
    # These are the edges copied by ``G.subgraph(base_nodes).copy()``.
    for node_id in base_nodes:
        for neighbour in G.neighbors(node_id):
            if neighbour not in base_nodes:
                continue
            edge = tuple(sorted((int(node_id), int(neighbour))))
            edges.add(edge)

    # Endpoint nodes are added after the base subgraph. Copy actual molecular
    # bonds; siblings records placement membership, not bond connectivity.
    current_nodes = set(base_nodes)
    for record, _ in endpoint_records:
        for node_id in record.node_ids:
            if node_id not in G:
                continue
            current_nodes.add(int(node_id))
            data = G.nodes[node_id]
            for sibling in G.neighbors(node_id):
                sibling = int(sibling)
                if sibling in endpoint_ids and sibling in current_nodes:
                    edges.add(tuple(sorted((int(node_id), sibling))))
            clique = data.get("clique")
            if clique is not None:
                for surface_id in clique:
                    surface_id = int(surface_id)
                    if surface_id in current_nodes:
                        edges.add(tuple(sorted((int(node_id), surface_id))))

    return _TripleEgoBlueprint(
        nodes=tuple(sorted(labels.items())),
        edges=tuple(sorted(edges)),
    )


def _materialise_triple_blueprint(blueprint: _TripleEgoBlueprint) -> nx.Graph:
    """Materialise a minimal graph for the authoritative matcher fallback."""
    graph = nx.Graph()
    for node_id, label in blueprint.nodes:
        (
            node_type,
            element,
            iso_class,
            reactant,
            reactant_orbit,
            endpoint_role,
            metadata_key,
        ) = label
        attributes: dict[str, Any] = {
            "type": node_type,
            "element": element,
            "atom_metadata_key": metadata_key,
        }
        if node_type == "adsorbate":
            attributes.update(
                iso_class=iso_class,
                reactant=reactant,
                reactant_orbit=reactant_orbit,
                endpoint_role=endpoint_role,
            )
        graph.add_node(node_id, **attributes)
    graph.add_edges_from(blueprint.edges)
    return graph


def _triple_wl_analysis(
    blueprint: _TripleEgoBlueprint,
) -> tuple[tuple, dict[int, int], tuple[tuple[int, ...], ...]]:
    """Return a strong coloured 1-WL fingerprint and its stable partition.

    The fingerprint is only a necessary isomorphism condition.  Ambiguous
    buckets continue to the exact canonical certificate or ``GraphMatcher``.
    """
    labels = dict(blueprint.nodes)
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in labels}
    for left, right in blueprint.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    unique_labels = sorted(set(labels.values()))
    label_colors = {
        label: color for color, label in enumerate(unique_labels)
    }
    colors = {
        node_id: label_colors[label]
        for node_id, label in labels.items()
    }
    refinement_trace: list[tuple[int, ...]] = [
        tuple(sorted(Counter(colors.values()).values()))
    ]

    for _ in range(max(1, len(labels))):
        signatures = {
            node_id: (
                colors[node_id],
                tuple(sorted(colors[neighbour] for neighbour in neighbours)),
            )
            for node_id, neighbours in adjacency.items()
        }
        unique_signatures = sorted(set(signatures.values()))
        signature_colors = {
            signature: color
            for color, signature in enumerate(unique_signatures)
        }
        refined = {
            node_id: signature_colors[signature]
            for node_id, signature in signatures.items()
        }
        refinement_trace.append(
            tuple(sorted(Counter(refined.values()).values()))
        )
        if refined == colors:
            colors = refined
            break
        colors = refined

    classes_mutable: dict[int, list[int]] = {}
    for node_id, color in colors.items():
        classes_mutable.setdefault(color, []).append(node_id)
    classes = tuple(
        tuple(sorted(classes_mutable[color]))
        for color in sorted(classes_mutable)
    )

    edge_color_counts = Counter(
        tuple(sorted((colors[left], colors[right])))
        for left, right in blueprint.edges
    )
    fingerprint = (
        len(labels),
        len(blueprint.edges),
        tuple(sorted(labels.values())),
        tuple(refinement_trace),
        tuple(sorted(Counter(colors.values()).items())),
        tuple(sorted(edge_color_counts.items())),
    )
    return fingerprint, colors, classes


def _exact_triple_certificate(
    blueprint: _TripleEgoBlueprint,
    classes: tuple[tuple[int, ...], ...],
    *,
    max_permutations: int = 64,
) -> tuple | None:
    """Return an exact canonical labelled-graph form when inexpensive.

    Stable WL color classes constrain every possible isomorphism.  Enumerating
    every permutation *within* those classes and taking the minimum labelled
    adjacency encoding therefore gives a collision-free canonical form.
    Highly symmetric cases that would exceed ``max_permutations`` return
    ``None`` and are checked by ``GraphMatcher`` instead.
    """
    permutation_count = 1
    for color_class in classes:
        permutation_count *= factorial(len(color_class))
        if permutation_count > max(1, int(max_permutations)):
            return None

    labels = dict(blueprint.nodes)
    edge_set = set(blueprint.edges)
    best: tuple | None = None
    class_permutations = [
        tuple(permutations(color_class))
        for color_class in classes
    ]
    for ordered_classes in product(*class_permutations):
        ordering = tuple(
            node_id
            for ordered_class in ordered_classes
            for node_id in ordered_class
        )
        ordered_labels = tuple(labels[node_id] for node_id in ordering)
        adjacency_bits = tuple(
            int(tuple(sorted((ordering[i], ordering[j]))) in edge_set)
            for i in range(len(ordering))
            for j in range(i, len(ordering))
        )
        encoding = (ordered_labels, adjacency_bits)
        if best is None or encoding < best:
            best = encoding
    return best


def _triple_node_match(d1: dict, d2: dict) -> bool:
    """Node-match predicate for triple iso-class deduplication.

    * ``type == "surface"``   — must share ``element``.
    * ``type == "adsorbate"`` — must share ``element``, ``iso_class``,
      ``reactant``, molecular ``reactant_orbit`` *and* ``endpoint_role`` so
      that the A/B/C roles are preserved, symmetry-equivalent atoms may be
      interchanged, and symmetry-inequivalent atoms remain distinct.
    """
    if d1.get("type") != d2.get("type"):
        return False
    if d1.get("element") != d2.get("element"):
        return False
    if atom_metadata_key(d1) != atom_metadata_key(d2):
        return False
    if d1.get("type") == "adsorbate":
        if d1.get("iso_class") != d2.get("iso_class"):
            return False
        if d1.get("reactant") != d2.get("reactant"):
            return False
        if _reactant_orbit_label(d1) != _reactant_orbit_label(d2):
            return False
        if (d1.get("endpoint_role") or "") != (d2.get("endpoint_role") or ""):
            return False
    return True


def _triple_fingerprint(g: nx.Graph) -> tuple:
    """Cheap fingerprint to bucket triple ego-graphs before the full GraphMatcher.

    Mirrors :func:`_triple_node_match`: ``endpoint_role`` is only included
    for adsorbate nodes.
    """
    sigs = tuple(sorted(
        (
            d.get("type",    "X"),
            d.get("element", "X"),
            atom_metadata_key(d),
            int(d.get("iso_class",      -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",       "")) if d.get("type") == "adsorbate" else "",
            _reactant_orbit_label(d) if d.get("type") == "adsorbate" else -1,
            (d.get("endpoint_role") or "") if d.get("type") == "adsorbate" else "",
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(sorted(g.degree(n) for n in g.nodes())),
        sigs,
    )


def _prune_one_per_adsorption_triple(
    bond_sites: list[BondReactionSite],
    *,
    verbose: bool = False,
    prefix: str = "",
) -> list[BondReactionSite]:
    """Keep one smallest-ego :class:`BondReactionSite` per adsorption triple.

    For every species-aware adsorption triple that appears in *bond_sites*,
    retain only the iso-class whose ``ego_graph`` has the fewest
    ``nodes + edges`` (most direct / closest reaction).
    Mirrors :func:`autokmc.sites.diffusion._prune_one_per_adsorption_pair`.
    """
    groups: dict[tuple, list[BondReactionSite]] = {}
    for brs in bond_sites:
        if not brs.members:
            continue
        sa, _, sb, _, sc, _ = brs.members[0]
        a_key = (_canon_smiles(sa.reactant), int(sa.iso_class))
        b_key = (_canon_smiles(sb.reactant), int(sb.iso_class))
        if getattr(brs, "gas_product", False) or sc is None:
            c_key = ("gas", brs.template.smiles_c, -1)
        else:
            c_key = (_canon_smiles(sc.reactant), int(sc.iso_class))
        ab_key = (
            tuple(sorted((a_key, b_key)))
            if brs.template.is_symmetric
            else (a_key, b_key)
        )
        template_key = (
            brs.template.smiles_a,
            brs.template.smiles_b,
            brs.template.smiles_c,
            brs.template.bond_type,
        )
        key = (template_key, ab_key, c_key)
        groups.setdefault(key, []).append(brs)

    def _ego_size(brs: BondReactionSite) -> int:
        g = brs.ego_graph
        if g is None:
            return 10**12
        return g.number_of_nodes() + g.number_of_edges()

    kept_ids: set[SiteId] = set()
    for key, candidates in groups.items():
        best = min(candidates, key=_ego_size)
        kept_ids.add(site_identifier(best))
        if verbose and len(candidates) > 1:
            discarded = [c for c in candidates if c is not best]
            print(
                f"  prune_one_per_adsorption_triple{prefix}: "
                f"triple {key}: kept iso {best.iso_class} "
                f"(ego size {_ego_size(best)}), discarded "
                f"{[c.iso_class for c in discarded]}"
            )

    return [brs for brs in bond_sites if site_identifier(brs) in kept_ids]


# ---------------------------------------------------------------------------
# Public API — find_bond_sites
# ---------------------------------------------------------------------------

def find_bond_sites(
    G: nx.Graph,
    adsorbate_sites: Iterable[AdsorbateSite],
    templates: Iterable[BondReactionTemplate],
    *,
    max_hops: int = BOND_MAX_HOPS,
    deduplicate_iso: bool = True,
    n_shells_pair: int = BOND_PAIR_N_SHELLS,
    prune_by_triple: bool = BOND_PRUNE_BY_TRIPLE,
    gas_species: dict[str, Any] | None = None,
    allow_gas_products: bool = True,
    gas_lift_height: float = 6.0,
    verbose: bool = False,
) -> list[BondReactionSite]:
    """Enumerate bond-reaction triples that satisfy the locality constraints.

    For each :class:`BondReactionTemplate` ``(A, B, C)``:

    1. Group all materialised :class:`AdsorbateSite`'s by canonical SMILES.
    2. Skip the template if any of A / B / C has no materialised site.
    3. For every pair of placements ``(p_A, p_B)`` whose bonded cliques are
       within ``max_hops`` surface-graph hops of each other …
    4. … and for every C placement ``p_C`` whose bonded clique is within
       ``max_hops`` of ``clique(p_A) ∪ clique(p_B)``, emit a triple.
    5. Triples are bucketed into :class:`BondReactionSite` iso-classes by
       the unordered tuple of adsorption iso-classes
       ``({iso_a, iso_b}, iso_c)`` (when ``deduplicate_iso=True``) or kept
       one-per-triple otherwise.

    Symmetric templates (``smiles_a == smiles_b``) only generate each
    ``(p_A, p_B)`` pair once via ``key(p_A) < key(p_B)``.

    Parameters
    ----------
    G : nx.Graph
        Surface + adsorbate graph (must already contain the materialised
        adsorbate-site nodes).
    adsorbate_sites : iterable of AdsorbateSite
        Sites to consider.  Typically the union of the
        :func:`~autokmc.sites.adsorbate.find_adsorbate_sites` output
        for every fragment / product SMILES referenced by *templates*.
    templates : iterable of BondReactionTemplate
        Templates to enumerate.
    max_hops : int
        Maximum surface-graph hop distance (default
        :data:`autokmc.core.constants.BOND_MAX_HOPS`).
    deduplicate_iso : bool
        Group triples sharing the same ``({iso_a, iso_b}, iso_c)`` into
        one :class:`BondReactionSite`.
    verbose : bool
        Print one summary line per template.

    Returns
    -------
    list[BondReactionSite]
        Flat list across every template, also stored on
        ``G.graph["bond_reaction_sites"]``.  The reverse index
        ``G.graph["bond_clique_to_members"]`` is populated for the KMC
        incremental-update path.
    """
    sites_list = list(adsorbate_sites)
    if not sites_list:
        # An empty collection is a valid outcome after adsorbate discovery and
        # stability pruning. Preserve an empty reaction network instead of
        # misreporting that the required discovery stage was skipped.
        if isinstance(G.graph.get("adsorbate_sites"), dict):
            G.graph["bond_reaction_sites"] = []
            rebuild_bond_reverse_indexes(G, [])
            return []
        raise ValueError(
            "find_bond_sites: no AdsorbateSites were supplied. "
            "Run `find_adsorbate_sites(G, reactant)` for every species "
            "referenced by your bond-reaction templates first — bond "
            "reactions need materialised adsorbate placements for A, B "
            "and C to enumerate triples on the graph."
        )
    if not any(d.get("type") == "adsorbate" for _, d in G.nodes(data=True)):
        raise ValueError(
            "find_bond_sites: graph contains no nodes of type 'adsorbate'. "
            "Adsorbate-site nodes must be added to the graph (via "
            "`find_adsorbate_sites`) before bond-reaction triples can be "
            "enumerated."
        )

    # Group materialised AdsorbateSites by canonical SMILES.
    by_smiles: dict[str, list[AdsorbateSite]] = {}
    for s in sites_list:
        by_smiles.setdefault(_canon_smiles(s.reactant), []).append(s)

    out: list[BondReactionSite] = []

    gas_species = dict(gas_species or {})

    # All placement identity, clique, node and shell metadata is immutable
    # during enumeration.  Materialise it once instead of rebuilding it in
    # each A/B/C iteration.
    records_by_smiles = _prepare_placement_records(
        G, by_smiles, int(n_shells_pair),
    )
    surface_index_by_smiles = {
        smiles: _surface_node_index_for_records(records)
        for smiles, records in records_by_smiles.items()
    }
    nearby_lookup = _NearbyPlacementLookup(G, surface_index_by_smiles)

    def _full_ego_graph(
        record_a: _PlacementRecord,
        record_b: _PlacementRecord,
        record_c: _PlacementRecord | None,
        *,
        is_symmetric: bool,
    ) -> nx.Graph:
        return _build_triple_ego_graph(
            G,
            list(record_a.node_ids),
            list(record_b.node_ids),
            [] if record_c is None else list(record_c.node_ids),
            record_a.clique_union,
            record_b.clique_union,
            frozenset() if record_c is None else record_c.clique_union,
            record_a.n_shells,
            record_b.n_shells,
            int(n_shells_pair) if record_c is None else record_c.n_shells,
            is_symmetric=is_symmetric,
        )

    for tpl in templates:
        sites_a = by_smiles.get(tpl.smiles_a, [])
        sites_b = by_smiles.get(tpl.smiles_b, [])
        sites_c = by_smiles.get(tpl.smiles_c, [])
        gas_reactant = gas_species.get(tpl.smiles_c)
        gas_product = (
            bool(allow_gas_products)
            and not sites_c
            and gas_reactant is not None
        )
        if not (sites_a and sites_b and (sites_c or gas_product)):
            if verbose:
                missing = [
                    name for name, lst in (
                        (tpl.smiles_a, sites_a),
                        (tpl.smiles_b, sites_b),
                        (tpl.smiles_c, sites_c),
                    ) if not lst
                ]
                print(
                    f"  SKIPPED template {tpl.smiles_a!r}+{tpl.smiles_b!r}"
                    f"⇌{tpl.smiles_c!r}: no sites for {missing}"
                )
            continue

        records_a = records_by_smiles.get(tpl.smiles_a, [])
        records_b = records_by_smiles.get(tpl.smiles_b, [])
        records_c = (
            [] if gas_product else records_by_smiles.get(tpl.smiles_c, [])
        )

        n_considered = 0
        n_kept       = 0
        template_out_start = len(out)

        # Exact certificates handle the common case without constructing a
        # NetworkX graph.  Highly symmetric WL buckets retain GraphMatcher as
        # the authoritative, collision-safe fallback.
        exact_index: dict[tuple, BondReactionSite] = {}
        wl_index: dict[tuple, list[BondReactionSite]] = {}
        legacy_fp_index: dict[tuple, list[BondReactionSite]] = {}

        def _new_bond_site(
            ego: nx.Graph | None,
            settled_depth: int,
        ) -> BondReactionSite:
            bond_site = BondReactionSite(
                template=tpl,
                iso_class=-1,
                ego_graph=ego,
                n_shells_pair_settled=settled_depth,
                gas_product=gas_product,
                gas_reactant=gas_reactant if gas_product else None,
                gas_lift_height=(
                    float(gas_lift_height) if gas_product else 6.0
                ),
            )
            out.append(bond_site)
            return bond_site

        def _classify_triple(
            record_a: _PlacementRecord,
            record_b: _PlacementRecord,
            record_c: _PlacementRecord | None,
        ) -> BondReactionSite:
            settled_depth = max(
                record_a.n_shells,
                record_b.n_shells,
                (
                    int(n_shells_pair)
                    if record_c is None
                    else record_c.n_shells
                ),
            )

            if not deduplicate_iso:
                try:
                    ego = _full_ego_graph(
                        record_a,
                        record_b,
                        record_c,
                        is_symmetric=tpl.is_symmetric,
                    )
                except Exception as exc:  # pragma: no cover
                    _log.debug(
                        "find_bond_sites: triple ego build failed: %s", exc,
                    )
                    ego = None
                return _new_bond_site(ego, settled_depth)

            try:
                blueprint = _build_triple_ego_blueprint(
                    G,
                    record_a,
                    record_b,
                    record_c,
                    is_symmetric=tpl.is_symmetric,
                )
            except Exception as exc:  # pragma: no cover
                _log.debug(
                    "find_bond_sites: triple blueprint build failed: %s", exc,
                )
                blueprint = None

            if blueprint is not None:
                wl_fingerprint, _, color_classes = _triple_wl_analysis(
                    blueprint,
                )
                exact_certificate = _exact_triple_certificate(
                    blueprint, color_classes,
                )

                if exact_certificate is not None:
                    matched = exact_index.get(exact_certificate)
                    if matched is not None:
                        matched.n_shells_pair_settled = max(
                            matched.n_shells_pair_settled, settled_depth,
                        )
                        return matched
                else:
                    # A WL collision is never taken as proof.  Build a small
                    # query graph and ask the exact matcher.
                    query_graph = _materialise_triple_blueprint(blueprint)
                    for candidate in wl_index.get(wl_fingerprint, ()):
                        if candidate.ego_graph is None:
                            continue
                        matcher = isomorphism.GraphMatcher(
                            query_graph,
                            candidate.ego_graph,
                            node_match=_triple_node_match,
                        )
                        if matcher.is_isomorphic():
                            candidate.n_shells_pair_settled = max(
                                candidate.n_shells_pair_settled,
                                settled_depth,
                            )
                            return candidate

                try:
                    ego = _full_ego_graph(
                        record_a,
                        record_b,
                        record_c,
                        is_symmetric=tpl.is_symmetric,
                    )
                except Exception as exc:  # pragma: no cover
                    _log.debug(
                        "find_bond_sites: triple ego build failed: %s", exc,
                    )
                    ego = None
                created = _new_bond_site(ego, settled_depth)
                if ego is not None:
                    if exact_certificate is not None:
                        exact_index[exact_certificate] = created
                    else:
                        wl_index.setdefault(wl_fingerprint, []).append(created)
                return created

            # Defensive compatibility path: if the lightweight blueprint ever
            # rejects unusual graph metadata, retain the historical graph
            # construction and exact-matcher behavior.
            try:
                ego = _full_ego_graph(
                    record_a,
                    record_b,
                    record_c,
                    is_symmetric=tpl.is_symmetric,
                )
            except Exception as exc:  # pragma: no cover
                _log.debug(
                    "find_bond_sites: triple ego build failed: %s", exc,
                )
                ego = None
            if ego is not None:
                legacy_fingerprint = _triple_fingerprint(ego)
                for candidate in legacy_fp_index.get(
                    legacy_fingerprint, (),
                ):
                    if candidate.ego_graph is None:
                        continue
                    matcher = isomorphism.GraphMatcher(
                        ego,
                        candidate.ego_graph,
                        node_match=_triple_node_match,
                    )
                    if matcher.is_isomorphic():
                        candidate.n_shells_pair_settled = max(
                            candidate.n_shells_pair_settled, settled_depth,
                        )
                        return candidate
            created = _new_bond_site(ego, settled_depth)
            if ego is not None:
                legacy_fp_index.setdefault(
                    _triple_fingerprint(ego), [],
                ).append(created)
            return created

        for record_a in records_a:
            for index_b in nearby_lookup.get(
                tpl.smiles_b,
                record_a.clique_union,
                int(max_hops),
            ):
                record_b = records_b[index_b]
                # Avoid degenerate self-pair.
                if record_a.key == record_b.key:
                    continue
                # Symmetric template: pick canonical ordering only.
                if tpl.is_symmetric and record_a.key >= record_b.key:
                    continue
                if not record_a.clique_set.isdisjoint(
                    record_b.clique_set,
                ):
                    continue

                n_considered += 1
                ab_union = (
                    record_a.clique_union | record_b.clique_union
                )

                if gas_product:
                    n_kept += 1
                    brs = _classify_triple(record_a, record_b, None)
                    brs.members.append((
                        record_a.site,
                        record_a.member_index,
                        record_b.site,
                        record_b.member_index,
                        None,
                        -1,
                    ))
                    brs.member_node_ids.append((
                        list(record_a.node_ids),
                        list(record_b.node_ids),
                        [],
                    ))
                    brs._member_cliques.append((
                        record_a.cliques,
                        record_b.cliques,
                        tuple(),
                    ))
                    continue

                for index_c in nearby_lookup.get(
                    tpl.smiles_c, ab_union, int(max_hops),
                ):
                    record_c = records_c[index_c]
                    if (
                        record_c.key == record_a.key
                        or record_c.key == record_b.key
                    ):
                        continue
                    n_kept += 1

                    brs = _classify_triple(
                        record_a, record_b, record_c,
                    )
                    brs.members.append((
                        record_a.site,
                        record_a.member_index,
                        record_b.site,
                        record_b.member_index,
                        record_c.site,
                        record_c.member_index,
                    ))
                    brs.member_node_ids.append((
                        list(record_a.node_ids),
                        list(record_b.node_ids),
                        list(record_c.node_ids),
                    ))
                    brs._member_cliques.append((
                        record_a.cliques,
                        record_b.cliques,
                        record_c.cliques,
                    ))

        if verbose:
            n_iso = len(out) - template_out_start
            print(
                f"  template {tpl.smiles_a!r}+{tpl.smiles_b!r}"
                f"⇌{tpl.smiles_c!r} ({tpl.source}): "
                f"{n_iso} iso-class(es), {n_kept} triple(s) kept "
                f"from {n_considered} pair(s) considered"
            )

    # If requested, keep one bond-reaction site for each adsorption triple.
    if prune_by_triple and out:
        before = len(out)
        out = _prune_one_per_adsorption_triple(
            out, verbose=verbose, prefix="",
        )
        if verbose and len(out) != before:
            print(
                f"  prune_by_triple: {before} → {len(out)} iso-class(es)"
            )

    # Finally, renumber the iso-classes and build the reverse index.
    for new_idx, brs in enumerate(out):
        brs.iso_class = new_idx

    G.graph["bond_reaction_sites"] = out
    rebuild_bond_reverse_indexes(G, out)

    _log.debug(
        "find_bond_sites: %d iso-class(es)", len(out),
    )
    if verbose:
        n_members = sum(len(brs.members) for brs in out)
        print(
            f"find_bond_sites: {len(out)} iso-class(es), "
            f"{n_members} triple member(s) total"
        )
    return out


__all__ = [
    "BondReactionTemplate",
    "BondReactionLateral",
    "BondReactionSite",
    "derive_dissociation_templates",
    "derive_coupling_templates",
    "derive_bond_templates",
    "find_bond_sites",
    "prune_unstable_bond_sites",
    "rebuild_bond_reverse_indexes",
]


# ---------------------------------------------------------------------------
# Calculator-based stability pruning of bond reactions  (Stage 1)
# ---------------------------------------------------------------------------

def _build_ab_pruning_atoms(
    G: nx.Graph,
    site_a: AdsorbateSite,
    m_a: int,
    site_b: AdsorbateSite,
    m_b: int,
    react_sym_a: list[str],
    react_sym_b: list[str],
    *,
    frozen_indices: list[int] | None = None,
):
    """Build an ASE Atoms object containing the slab + adsorbate A + adsorbate B.

    The C placement is **not** present — this builds the pre-reaction
    endpoint of an ``A + B → C`` bond change.

    Adsorbate atom positions are read from the live graph
    (``G.nodes[nid]["position"]``) so Kabsch-propagated representative-only
    refinements (see
    :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`)
    are honoured.

    Returns
    -------
    atoms : Atoms
    n_slab : int
    n_a, n_b : int
        Number of A and B adsorbate atoms appended (in that order, after the
        slab).  A occupies indices ``n_slab .. n_slab+n_a``; B occupies
        ``n_slab+n_a .. n_slab+n_a+n_b``.
    node_to_ase : dict[int, int]
        G surface/bulk node-id → ASE atom index in *atoms* for slab nodes.
    """
    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms

    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True)
         if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )
    slab_sym = [G.nodes[n]["element"]  for n in slab_nodes]
    slab_pos = [G.nodes[n]["position"] for n in slab_nodes]
    slab_tag = [
        1 if G.nodes[n].get("type") == "surface" else 0
        for n in slab_nodes
    ]

    a_nids = list(site_a.member_node_ids[m_a])
    b_nids = list(site_b.member_node_ids[m_b])
    a_pos  = [np.asarray(G.nodes[n]["position"], dtype=float) for n in a_nids]
    b_pos  = [np.asarray(G.nodes[n]["position"], dtype=float) for n in b_nids]
    n_a, n_b = len(react_sym_a), len(react_sym_b)
    if len(a_pos) != n_a or len(b_pos) != n_b:
        raise ValueError(
            f"_build_ab_pruning_atoms: adsorbate atom count mismatch "
            f"(A: {len(a_pos)} != {n_a}, B: {len(b_pos)} != {n_b})"
        )

    symbols   = slab_sym + list(react_sym_a) + list(react_sym_b)
    positions = (
        slab_pos
        + [p.tolist() for p in a_pos]
        + [p.tolist() for p in b_pos]
    )
    surface_array = np.asarray(
        slab_tag + [2] * (n_a + n_b), dtype=np.int8,
    )

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = full_pbc_for_cell(cell)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    apply_atom_metadata(atoms, [G.nodes[node] for node in slab_nodes + a_nids + b_nids])
    atoms.arrays["surface"] = surface_array
    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))

    node_to_ase = {int(nid): i for i, nid in enumerate(slab_nodes)}
    return atoms, len(slab_nodes), n_a, n_b, node_to_ase


def _intended_ab_edges(
    site_a: AdsorbateSite, react_a, n_slab: int, node_to_ase: dict[int, int],
    site_b: AdsorbateSite, react_b, ab_offset: int,
    m_a: int, m_b: int,
) -> set[frozenset]:
    """Edge set the relaxed graph must contain for the (A + B) endpoint.

    Includes intramolecular bonds for A and B (from each
    ``Reactant.graph``) and anchor bonds for every bonded reactant atom
    of A and B mapping to its placement-specific clique.  No A–B bonds
    are required (A and B are independent adsorbates pre-reaction).
    """
    edges: set[frozenset] = set()

    # A intramolecular
    if react_a.graph is not None:
        for u, v in react_a.graph.edges():
            edges.add(frozenset((n_slab + int(u), n_slab + int(v))))
    # B intramolecular
    if react_b.graph is not None:
        for u, v in react_b.graph.edges():
            edges.add(frozenset((ab_offset + int(u), ab_offset + int(v))))

    # Members retain one entry per atom, including unbound atoms. The reverse
    # index's _member_cliques omits None entries and cannot supply atom indices.
    if m_a < len(site_a.members):
        for i, clq in enumerate(site_a.members[m_a]):
            if clq is None:
                continue
            ads_idx = n_slab + int(i)
            for surf_nid in clq:
                ase_surf = node_to_ase.get(int(surf_nid))
                if ase_surf is None:
                    continue
                edges.add(frozenset((ads_idx, ase_surf)))

    # B anchor bonds.
    if m_b < len(site_b.members):
        for i, clq in enumerate(site_b.members[m_b]):
            if clq is None:
                continue
            ads_idx = ab_offset + int(i)
            for surf_nid in clq:
                ase_surf = node_to_ase.get(int(surf_nid))
                if ase_surf is None:
                    continue
                edges.add(frozenset((ads_idx, ase_surf)))

    return edges


def _ab_adsorbate_edges_from_graph(
    G_relaxed: nx.Graph, n_slab: int,
) -> set[frozenset]:
    """Edges of *G_relaxed* that touch at least one adsorbate atom."""
    edges: set[frozenset] = set()
    for u, v in G_relaxed.edges():
        if int(u) < n_slab and int(v) < n_slab:
            continue
        edges.add(frozenset((int(u), int(v))))
    return edges


def prune_unstable_bond_sites(
    G: nx.Graph,
    bond_sites: list[BondReactionSite],
    species_by_smiles,
    calculator,
    *,
    frozen_indices: list[int] | None = None,
    fmax: float = PRUNE_FMAX,
    max_steps: int = PRUNE_MAX_STEPS,
    nl_mult: float = NL_MULT_DEFAULT,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    debug_output_dir: str | None = None,
) -> list[BondReactionSite]:
    """Drop bond-reaction iso-classes whose A+B endpoint is bond-changing-unstable.

    For every :class:`BondReactionSite`, build the pre-reaction state (slab
    with A and B adsorbates placed at the representative member's sites,
    C absent), run a calculator relaxation, and compare the relaxed
    adsorbate-touching connectivity against the *intended* edge set
    (intramolecular bonds + anchor bonds for A and B).  If the bonding
    changed — e.g. A and B spontaneously form C, A dissociates, or any
    anchor bond is gained / lost — the BRS is **not viable** and is
    pruned.  Mirrors
    :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`
    for the bond-reaction case.

    Stage 2 (iso-class triple pruning by ego size) should be applied
    **after** this function — viability comes first.

    Parameters
    ----------
    G : nx.Graph
        Live surface + adsorbate graph.
    bond_sites : list[BondReactionSite]
        Iso-classes to check.
    species_by_smiles : Mapping[str, Reactant]
        Map from canonical SMILES → :class:`~autokmc.species.reactant.Reactant`.
        Used to look up chemical symbols and intramolecular bond
        topology.  Pass ``G.graph["bond_registry"]["species"]`` after
        :func:`autokmc.kmc.expansion.initialise_bond_registry` has run.
    calculator
        ASE-compatible calculator or CalculatorPool.  A calculator is
        acquired for each relaxation; calculator instances are never
        deep-copied.
    frozen_indices, fmax, max_steps, nl_mult, verbose
        See :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`.
    debug_output_dir
        Optional directory for debugging endpoint pruning.  When supplied,
        the unrelaxed and relaxed representative A+B endpoint structures are
        written as ``extxyz`` files under one folder per pre-pruning
        ``bond_iso``.

    Returns
    -------
    list[BondReactionSite]
        Surviving viable iso-classes in original order.  ``iso_class`` is
        **renumbered** to be sequential.
        ``G.graph["bond_reaction_sites"]`` is updated in place.
    """
    import numpy as np

    if calculator is None or not bond_sites:
        return list(bond_sites)

    from autokmc.io.atoms import copy_atoms_with_results
    from autokmc.structure import StructureOptimisationError, optimise_structure
    from autokmc.core.graph import build_graph

    debug_dir = None
    if debug_output_dir is not None:
        from pathlib import Path

        debug_dir = Path(debug_output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    def _write_debug_endpoint(bond_iso, filename: str, atoms) -> None:
        if debug_dir is None:
            return
        try:
            from autokmc.io.extxyz import write_extxyz as ase_write

            out_dir = debug_dir / f"bond_iso_{int(bond_iso):03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ase_write(out_dir / filename, atoms, format="extxyz")
        except Exception as exc:
            _log.warning(
                "prune_unstable_bond_sites: failed to write debug endpoint "
                "%s for bond_iso=%s (%s)",
                filename, bond_iso, exc,
            )

    if verbose:
        print(
            f"\nprune_unstable_bond_sites: {len(bond_sites)} iso-class(es)  "
            f"fmax={fmax} eV/Å  max_steps={max_steps}"
        )

    # Cache outcome by an unordered placement-key pair so members shared
    # across BRSs are not re-relaxed.
    cache: dict[frozenset, bool] = {}

    def _placement_id(s, m):
        return member_identifier(s, m)

    def _check_pair(sa, ma, sb, mb, bond_iso) -> bool:
        key = frozenset({_placement_id(sa, ma), _placement_id(sb, mb)})
        if key in cache:
            return cache[key]

        smi_a = _canon_smiles(sa.reactant)
        smi_b = _canon_smiles(sb.reactant)
        react_a = species_by_smiles.get(smi_a)
        react_b = species_by_smiles.get(smi_b)
        if react_a is None or react_b is None:
            _log.warning(
                "prune_unstable_bond_sites: missing Reactant for %r or %r — "
                "keeping conservatively.", smi_a, smi_b,
            )
            cache[key] = True
            return True

        sym_a = list(react_a.atoms.get_chemical_symbols())
        sym_b = list(react_b.atoms.get_chemical_symbols())

        try:
            atoms_init, n_slab, n_a, n_b, node_to_ase = _build_ab_pruning_atoms(
                G, sa, ma, sb, mb, sym_a, sym_b,
                frozen_indices=frozen_indices,
            )
        except Exception as exc:
            _log.warning(
                "prune_unstable_bond_sites: build failed (%s) — keeping.",
                exc,
            )
            cache[key] = True
            return True

        _write_debug_endpoint(bond_iso, "endpoint_ab_before_opt.extxyz", atoms_init)

        intended = _intended_ab_edges(
            sa, react_a, n_slab, node_to_ase,
            sb, react_b, n_slab + n_a, ma, mb,
        )

        try:
            with acquire_calculator(
                calculator, purpose="bond-site pruning"
            ) as calc:
                atoms_opt = optimise_structure(
                    atoms_init,
                    calculator = calc,
                    fmax       = fmax,
                    steps      = max_steps,
                    optimizer  = optimizer,
                    optimizer_kwargs = optimizer_kwargs,
                    verbose    = False,
                )
                forces = atoms_opt.get_forces()
                if frozen_indices:
                    free_mask = np.ones(len(atoms_opt), dtype=bool)
                    free_mask[list(frozen_indices)] = False
                    max_force = float(np.linalg.norm(forces[free_mask], axis=1).max())
                else:
                    max_force = float(np.linalg.norm(forces, axis=1).max())
                energy = float(atoms_opt.get_potential_energy())
                atoms_opt = copy_atoms_with_results(
                    atoms_opt,
                    energy=energy,
                    forces=forces,
                )
        except Exception as exc:
            if not (
                isinstance(exc, StructureOptimisationError)
                and exc.converged is False
            ):
                raise
            _log.debug(
                "prune_unstable_bond_sites: relaxation raised %s", exc,
            )
            cache[key] = False
            return False

        _write_debug_endpoint(bond_iso, "endpoint_ab_after_opt.extxyz", atoms_opt)

        if max_force > fmax:
            cache[key] = False
            return False

        atoms_for_graph = atoms_opt.copy()
        atoms_for_graph.arrays["surface"] = atoms_init.arrays["surface"]
        try:
            G_relaxed = build_graph(atoms_for_graph, nl_mult=nl_mult)
        except Exception as exc:
            _log.warning(
                "prune_unstable_bond_sites: build_graph(relaxed) failed (%s) "
                "— pruned.", exc,
            )
            cache[key] = False
            return False

        relaxed = _ab_adsorbate_edges_from_graph(G_relaxed, n_slab)
        viable = (relaxed == intended)
        cache[key] = viable
        return viable

    survivors: list[BondReactionSite] = []
    for brs in bond_sites:
        if not brs.members:
            continue
        sa, ma, sb, mb, _sc, _mc = brs.members[0]
        viable = _check_pair(sa, ma, sb, mb, brs.iso_class)
        if viable:
            survivors.append(brs)
            if verbose:
                print(
                    f"  STABLE bond_iso={brs.iso_class}: A+B endpoint stable"
                )
        else:
            if verbose:
                print(
                    f"  PRUNED bond_iso={brs.iso_class}: A+B endpoint changes "
                    f"bonding — pruned"
                )

    for new_idx, brs in enumerate(survivors):
        brs.iso_class = new_idx

    G.graph["bond_reaction_sites"] = survivors
    rebuild_bond_reverse_indexes(G, survivors)

    if verbose:
        print(
            f"  prune_unstable_bond_sites: "
            f"{len(survivors)}/{len(bond_sites)} iso-class(es) survived"
        )

    _log.debug(
        "prune_unstable_bond_sites: %d/%d iso-classes survived",
        len(survivors), len(bond_sites),
    )
    return survivors
