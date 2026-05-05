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

from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.sites.adsorbate import (
    AdsorbateSite,
    _get_surface_apsp,
    _shortest_path_between_cliques,
)
from autokmc.sites.diffusion import _member_clique_union
from autokmc.sites.stability.adsorption import _surface_bfs_shells
from autokmc.core.constants import (
    BOND_MAX_HOPS,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    MAX_PAIR_SHELLS,
    NL_MULT_DEFAULT,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
)
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# SMILES canonicalisation helper
# ---------------------------------------------------------------------------

def _canon_smiles(smi: str) -> str:
    """Return RDKit-canonical SMILES; falls back to the input string."""
    if smi is None:
        return ""
    try:
        from rdkit import Chem
    except ImportError:
        return str(smi)
    try:
        return Chem.CanonSmiles(str(smi))
    except Exception:
        return str(smi)


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
        Potential energy (eV) of the relaxed *C occupied, A and B empty*
        state.
    energy_ts : float | None
        Potential energy (eV) of the highest NEB image between the two
        states above (the climbing-image saddle when ``climb=True``).
    atoms_ab, atoms_c, atoms_ts : Atoms | None
        Relaxed ASE atoms snapshots persisted by
        :class:`autokmc.io.persistence.ReactionWriter`.
    atoms_neb_path : list[Atoms] | None
        Full NEB band — optional, only kept when ``persist_neb_path=True``.
    neb_path_energies : list[float] | None
        Per-image energies along :attr:`atoms_neb_path`.
    stable : bool | None
        ``True`` when both endpoint relaxations and the NEB converged
        without changing surface / adsorbate connectivity; ``False`` on
        any stability failure; ``None`` until the check has run.
    invalid_reason : str | None
        Human-readable explanation of why this lateral class is invalid.
    """
    lateral_class    : int
    ego_graph        : Any              = None
    n_shells         : int              = 0
    members          : list[int]        = field(default_factory=list)
    energy_ab        : float | None     = None
    energy_c         : float | None     = None
    energy_ts        : float | None     = None
    atoms_ab         : Any              = None
    atoms_c          : Any              = None
    atoms_ts         : Any              = None
    atoms_neb_path   : Any              = None
    neb_path_energies: list[float] | None = None
    stable           : bool | None      = None
    invalid_reason   : str | None       = None
    # ── Free-energy / vibrational fields (autokmc.thermo.free_energy) ────────────
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
    frequencies_ab_cm : list = field(default_factory=list)
    frequencies_c_cm  : list = field(default_factory=list)
    frequencies_ts_cm : list = field(default_factory=list)
    imaginary_ab_cm   : list = field(default_factory=list)
    imaginary_c_cm    : list = field(default_factory=list)
    imaginary_ts_cm   : list = field(default_factory=list)
    vib_indices_ab    : list = field(default_factory=list)
    vib_indices_c     : list = field(default_factory=list)
    vib_indices_ts    : list = field(default_factory=list)


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

    # Cached per-member tuples ``(cliques_a, cliques_b, cliques_c)`` —
    # populated by :func:`find_bond_sites`.  Used by :mod:`autokmc.reactions.bond`
    # for the clique-collision applicability guard.
    _member_cliques : list[tuple[tuple[frozenset, ...],
                                 tuple[frozenset, ...],
                                 tuple[frozenset, ...]]] = field(default_factory=list)


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

def derive_dissociation_templates(
    smiles: str | Iterable[str],
    *,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    add_hydrogens: bool = True,
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
    """
    from autokmc.species.bond_chemistry import get_all_fragments

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    out: list[BondReactionTemplate] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_a in smiles_list:
        big = _canon_smiles(raw_a)
        try:
            pairs = get_all_fragments(
                raw_a,
                add_hydrogens      = add_hydrogens,
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
    """
    from autokmc.species.bond_chemistry import combine_fragments

    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

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
                products = combine_fragments(smi_a, smi_b)
            except Exception as exc:
                _log.warning(
                    "derive_coupling_templates: combine_fragments(%r, %r) failed: %s",
                    smi_a, smi_b, exc,
                )
                continue
            for sp in products:
                big = _canon_smiles(sp.smiles)
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
) -> list[BondReactionTemplate]:
    """Convenience: union of dissociation + coupling templates.

    The combined list is deduplicated by ``(smiles_a, smiles_b, smiles_c)``.
    """
    out: list[BondReactionTemplate] = []
    if include_dissociation:
        out.extend(derive_dissociation_templates(
            smiles,
            bond_types         = bond_types,
            include_ring_bonds = include_ring_bonds,
            add_hydrogens      = add_hydrogens,
        ))
    if include_coupling:
        out.extend(derive_coupling_templates(
            smiles,
            include_homo   = include_homo_coupling,
            include_hetero = include_hetero_coupling,
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


def _placement_key(site: AdsorbateSite, m_idx: int) -> tuple[str, int, int, int]:
    """Canonical sortable key for one (site, m_idx) placement."""
    return (_canon_smiles(site.reactant), int(site.iso_class), int(m_idx), id(site))


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

    # Other occupied adsorbate leaves adjacent to the BFS set.
    ads_leaves: set = set()
    for n in visited:
        for nb in G.neighbors(n):
            if nb in visited or nb in endpoint_ids:
                continue
            d = G.nodes[nb]
            if d.get("type") != "adsorbate":
                continue
            if d.get("occupied", False):
                ads_leaves.add(nb)

    result = G.subgraph(visited | ads_leaves).copy()

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
                    type           = d.get("type", "adsorbate"),
                    iso_class      = int(d.get("iso_class", -1)),
                    reactant       = str(d.get("reactant",  "")),
                    reactant_index = int(d.get("reactant_index", -1)),
                    occupied       = True,
                    endpoint_role  = role,
                )
            else:
                result.nodes[nid]["occupied"]      = True
                result.nodes[nid]["endpoint_role"] = role
            for sib in d.get("siblings", ()):
                sib = int(sib)
                if sib in result and not result.has_edge(nid, sib):
                    result.add_edge(nid, sib, intra_adsorbate=True)
            clq = d.get("clique")
            if clq is not None:
                for surf_id in clq:
                    if surf_id in result and not result.has_edge(nid, surf_id):
                        result.add_edge(nid, surf_id, anchor_bond=True)

    return result


def _triple_node_match(d1: dict, d2: dict) -> bool:
    """Node-match predicate for triple iso-class deduplication.

    * ``type == "surface"``   — must share ``element``.
    * ``type == "adsorbate"`` — must share ``element``, ``iso_class``,
      ``reactant``, ``reactant_index`` *and* ``endpoint_role`` so that the
      A/B/C roles are preserved across the mapping, and symmetry-inequivalent
      atoms of the same element within a multi-atom adsorbate are never
      interchanged.
    """
    if d1.get("type") != d2.get("type"):
        return False
    if d1.get("element") != d2.get("element"):
        return False
    if d1.get("type") == "adsorbate":
        if d1.get("iso_class") != d2.get("iso_class"):
            return False
        if d1.get("reactant") != d2.get("reactant"):
            return False
        if d1.get("reactant_index") != d2.get("reactant_index"):
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
            int(d.get("iso_class",      -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",       "")) if d.get("type") == "adsorbate" else "",
            int(d.get("reactant_index", -1)) if d.get("type") == "adsorbate" else -1,
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

    kept_ids: set[int] = set()
    for key, candidates in groups.items():
        best = min(candidates, key=_ego_size)
        kept_ids.add(id(best))
        if verbose and len(candidates) > 1:
            discarded = [c for c in candidates if c is not best]
            print(
                f"  prune_one_per_adsorption_triple{prefix}: "
                f"triple {key}: kept iso {best.iso_class} "
                f"(ego size {_ego_size(best)}), discarded "
                f"{[c.iso_class for c in discarded]}"
            )

    return [brs for brs in bond_sites if id(brs) in kept_ids]


# ---------------------------------------------------------------------------
# Public API — find_bond_sites
# ---------------------------------------------------------------------------

def find_bond_sites(
    G: nx.Graph,
    adsorbate_sites: Iterable[AdsorbateSite],
    templates: Iterable[BondReactionTemplate],
    *,
    max_hops: int = BOND_MAX_HOPS,
    surface_apsp_cutoff: int = MAX_PAIR_SHELLS,
    deduplicate_iso: bool = True,
    n_shells_pair: int = BOND_PAIR_N_SHELLS,
    prune_by_triple: bool = BOND_PRUNE_BY_TRIPLE,
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
    surface_apsp_cutoff : int
        Cutoff handed to
        :func:`autokmc.sites.adsorbate._get_surface_apsp` for the
        cached APSP table.  Must be ≥ ``max_hops``.
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
        raise ValueError(
            "find_bond_sites: no AdsorbateSite's were supplied. "
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

    apsp = _get_surface_apsp(
        G, cutoff=max(int(max_hops), int(surface_apsp_cutoff)),
    )

    out: list[BondReactionSite] = []

    for tpl in templates:
        sites_a = by_smiles.get(tpl.smiles_a, [])
        sites_b = by_smiles.get(tpl.smiles_b, [])
        sites_c = by_smiles.get(tpl.smiles_c, [])
        if not (sites_a and sites_b and sites_c):
            if verbose:
                missing = [
                    name for name, lst in (
                        (tpl.smiles_a, sites_a),
                        (tpl.smiles_b, sites_b),
                        (tpl.smiles_c, sites_c),
                    ) if not lst
                ]
                print(
                    f"  ⏭  template {tpl.smiles_a!r}+{tpl.smiles_b!r}"
                    f"⇌{tpl.smiles_c!r}: no sites for {missing}"
                )
            continue

        flat_a = _flatten_sites(sites_a)
        flat_b = _flatten_sites(sites_b)
        flat_c = _flatten_sites(sites_c)
        surface_index_b = _surface_node_index_for_placements(flat_b)
        surface_index_c = _surface_node_index_for_placements(flat_c)

        n_considered = 0
        n_kept       = 0

        # Fingerprint → list[BondReactionSite] index for isomorphism dedup.
        # Using graph isomorphism instead of a simple iso_class-index tuple key
        # because the latter incorrectly merges geometrically distinct triples
        # that share the same individual iso-class numbers (e.g. hops in
        # different crystallographic directions between the same site types).
        # See the analogous fix note in find_diffusion_sites (~lines 583-591).
        fp_index: dict[tuple, list[BondReactionSite]] = {}

        for sa, ma, clq_a in flat_a:
            ka = _placement_key(sa, ma)
            cliques_a = _placement_cliques(sa, ma)
            for j_b in _nearby_placement_indices(
                G, surface_index_b, clq_a, int(max_hops),
            ):
                sb, mb, clq_b = flat_b[j_b]
                kb = _placement_key(sb, mb)
                # Avoid degenerate self-pair.
                if ka == kb:
                    continue
                # Symmetric template: pick canonical ordering only.
                if tpl.is_symmetric and ka >= kb:
                    continue
                cliques_b = _placement_cliques(sb, mb)
                if _placements_share_exact_clique(cliques_a, cliques_b):
                    continue

                n_considered += 1

                if _shortest_path_between_cliques(clq_a, clq_b, apsp) > int(max_hops):
                    continue

                ab_union = clq_a | clq_b

                for j_c in _nearby_placement_indices(
                    G, surface_index_c, ab_union, int(max_hops),
                ):
                    sc, mc, clq_c = flat_c[j_c]
                    kc = _placement_key(sc, mc)
                    if kc == ka or kc == kb:
                        continue
                    if _shortest_path_between_cliques(
                        clq_c, ab_union, apsp,
                    ) > int(max_hops):
                        continue

                    n_kept += 1

                    a_nids = list(sa.member_node_ids[ma])
                    b_nids = list(sb.member_node_ids[mb])
                    c_nids = list(sc.member_node_ids[mc])
                    ns_a = max(int(getattr(sa, "n_shells_settled", 0) or 0),
                               int(n_shells_pair))
                    ns_b = max(int(getattr(sb, "n_shells_settled", 0) or 0),
                               int(n_shells_pair))
                    ns_c = max(int(getattr(sc, "n_shells_settled", 0) or 0),
                               int(n_shells_pair))

                    # ── Bucket into BondReactionSite via graph isomorphism ──
                    try:
                        ego = _build_triple_ego_graph(
                            G, a_nids, b_nids, c_nids,
                            clq_a, clq_b, clq_c,
                            ns_a, ns_b, ns_c,
                            is_symmetric=tpl.is_symmetric,
                        )
                    except Exception as exc:  # pragma: no cover
                        _log.debug(
                            "find_bond_sites: triple ego build failed: %s",
                            exc,
                        )
                        ego = None

                    merged = False
                    if deduplicate_iso and ego is not None:
                        fkey = _triple_fingerprint(ego)
                        for brs in fp_index.get(fkey, ()):
                            if brs.ego_graph is None:
                                continue
                            gm = isomorphism.GraphMatcher(
                                ego, brs.ego_graph,
                                node_match=_triple_node_match,
                            )
                            if gm.is_isomorphic():
                                brs.n_shells_pair_settled = max(
                                    brs.n_shells_pair_settled,
                                    max(ns_a, ns_b, ns_c),
                                )
                                merged = True
                                break

                    if not merged:
                        brs = BondReactionSite(
                            template              = tpl,
                            iso_class             = -1,   # renumbered below
                            ego_graph             = ego,
                            n_shells_pair_settled = max(ns_a, ns_b, ns_c),
                        )
                        out.append(brs)
                        if ego is not None:
                            fkey = _triple_fingerprint(ego)
                            fp_index.setdefault(fkey, []).append(brs)

                    brs.members.append((sa, ma, sb, mb, sc, mc))
                    brs.member_node_ids.append((
                        list(a_nids),
                        list(b_nids),
                        list(c_nids),
                    ))
                    brs._member_cliques.append((
                        cliques_a,
                        cliques_b,
                        _placement_cliques(sc, mc),
                    ))

        if verbose:
            n_iso = sum(len(v) for v in fp_index.values()) if deduplicate_iso else n_kept
            print(
                f"  ✓ template {tpl.smiles_a!r}+{tpl.smiles_b!r}"
                f"⇌{tpl.smiles_c!r} ({tpl.source}): "
                f"{n_iso} iso-class(es), {n_kept} triple(s) kept "
                f"from {n_considered} pair(s) considered"
            )

    # ── Optional: keep one BondReactionSite per adsorption triple ──────────
    if prune_by_triple and out:
        before = len(out)
        out = _prune_one_per_adsorption_triple(
            out, verbose=verbose, prefix="",
        )
        if verbose and len(out) != before:
            print(
                f"  prune_by_triple: {before} → {len(out)} iso-class(es)"
            )

    # ── Renumber iso_class globally and build reverse index ────────────────
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
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
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

    # A anchor bonds — read per-member cliques (placement-specific).
    a_member_cliques = getattr(site_a, "_member_cliques", None)
    if a_member_cliques is not None and m_a < len(a_member_cliques):
        for i, clq in enumerate(a_member_cliques[m_a]):
            if clq is None:
                continue
            ads_idx = n_slab + int(i)
            for surf_nid in clq:
                ase_surf = node_to_ase.get(int(surf_nid))
                if ase_surf is None:
                    continue
                edges.add(frozenset((ads_idx, ase_surf)))

    # B anchor bonds.
    b_member_cliques = getattr(site_b, "_member_cliques", None)
    if b_member_cliques is not None and m_b < len(b_member_cliques):
        for i, clq in enumerate(b_member_cliques[m_b]):
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
    verbose: bool = False,
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
        ASE-compatible calculator.  A :func:`copy.deepcopy` is made for
        each relaxation so the caller's instance is never mutated.
    frozen_indices, fmax, max_steps, nl_mult, verbose
        See :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`.

    Returns
    -------
    list[BondReactionSite]
        Surviving viable iso-classes in original order.  ``iso_class`` is
        **renumbered** to be sequential.
        ``G.graph["bond_reaction_sites"]`` is updated in place.
    """
    import copy
    import numpy as np

    if calculator is None or not bond_sites:
        return list(bond_sites)

    from autokmc.structure import optimise_structure
    from autokmc.core.graph import build_graph

    if verbose:
        print(
            f"\nprune_unstable_bond_sites: {len(bond_sites)} iso-class(es)  "
            f"fmax={fmax} eV/Å  max_steps={max_steps}"
        )

    # Cache outcome by an unordered placement-key pair so members shared
    # across BRSs are not re-relaxed.
    cache: dict[frozenset, bool] = {}

    def _placement_id(s, m):
        return (id(s), int(m))

    def _check_pair(sa, ma, sb, mb) -> bool:
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

        intended = _intended_ab_edges(
            sa, react_a, n_slab, node_to_ase,
            sb, react_b, n_slab + n_a, ma, mb,
        )

        try:
            calc_copy = copy.deepcopy(calculator)
            atoms_opt = optimise_structure(
                atoms_init,
                calculator = calc_copy,
                fmax       = fmax,
                steps      = max_steps,
                verbose    = False,
            )
        except Exception as exc:
            _log.debug(
                "prune_unstable_bond_sites: relaxation raised %s", exc,
            )
            cache[key] = False
            return False

        forces = atoms_opt.get_forces()
        if frozen_indices:
            free_mask = np.ones(len(atoms_opt), dtype=bool)
            free_mask[list(frozen_indices)] = False
            max_force = float(np.linalg.norm(forces[free_mask], axis=1).max())
        else:
            max_force = float(np.linalg.norm(forces, axis=1).max())

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
        viable = _check_pair(sa, ma, sb, mb)
        if viable:
            survivors.append(brs)
            if verbose:
                print(
                    f"  ✓ bond_iso={brs.iso_class}: A+B endpoint stable"
                )
        else:
            if verbose:
                print(
                    f"  ✗ bond_iso={brs.iso_class}: A+B endpoint changes "
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
