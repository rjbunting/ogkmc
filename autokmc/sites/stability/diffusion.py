"""
autokmc.sites.stability.diffusion
=============================
On-the-fly lateral-interaction classifier and CI-NEB stability / barrier
calculator for materialised diffusion (hop) sites.

This is the diffusion analogue of :mod:`autokmc.sites.stability.adsorption`.  Each
:class:`~autokmc.sites.diffusion.DiffusionSite` enumerates an
isomorphism class of unordered hop *pairs* (A ↔ B) for one migrating SMILES.
Two things have to be evaluated lazily, only when the KMC loop actually
needs them:

1. The **lateral-interaction class** of a specific pair member — i.e. the
   pattern of occupied adsorbate neighbours surrounding the union of A's
   and B's bonded surface cliques.  See :func:`check_diffusion_site_lateral`.

2. The **NEB barrier** plus the two endpoint relaxations for that lateral
   class — one ML relaxation each for the A-occupied and B-occupied
   endpoints, then an ordinary NEB and, unless either directional barrier is
   below the KMC floor, a CI-NEB refinement.  See
   :func:`check_diffusion_stability`.

Lateral ego-graph conventions (extends :mod:`autokmc.sites.stability.adsorption`)
---------------------------------------------------------------------------
* **BFS seed** — the *union* of A's and B's bonded surface cliques.
* **BFS frontier** — only ``type == "surface"`` nodes are traversed; both
  endpoint placements are excluded from the visited surface set.
* **Occupied adsorbate leaves** — every occupied ``type == "adsorbate"``
  node adjacent to the BFS surface set, *excluding* the two endpoints.
* **Endpoint inclusion** — both endpoints' atoms are added as labelled
  leaves with ``occupied=True`` and distinct ``endpoint_role="a"`` / ``"b"``
  labels.  A lateral match preserves the ordering of the cached endpoint
  energies.  Each member still supports both firing directions.

NEB conventions
---------------
The migrating molecule has a single physical instance — the same atoms
move from A's geometry to B's geometry.  Per-image atom layout::

    [ slab | lat_neighbours | migrating_molecule ]

Atom ordering inside the migrating-molecule block follows the molecule's
SMILES atom order (``reactant_index`` on the graph node), so corresponding
atoms in image 0 (A's relaxed positions) and image -1 (B's relaxed
positions) are guaranteed to line up — a hard requirement for ASE's NEB
interpolators.

Calculator handling
-------------------
The supplied *calculator* may be a concrete ASE calculator or a
CalculatorPool.  Endpoint relaxations acquire one calculator at a time; NEB
bands acquire one calculator and use ASE's shared-calculator NEB mode.
Calculators are never deep-copied.  All NEB-specific defaults live in
:mod:`autokmc.core.constants` (``NEB_*``).

Public API
----------
* :class:`DiffusionStabilityError`     — base error.
* :class:`EndpointStabilityError`      — endpoint relaxation failed.
* :class:`NEBNotConvergedError`        — NEB band did not converge.
* :class:`TransitionStateInvalidError` — TS energies are non-finite or the
  migrating molecule lost its internal connectivity.
* (Re-exported) :class:`SurfaceConnectivityError`,
  :class:`AdsorbateDissociationError`,
  :class:`OptimisationFailedError`     — from
  :mod:`autokmc.sites.stability.adsorption`.
* :func:`check_diffusion_site_lateral` — classify the lateral environment
  of one specific hop-pair member.
* :func:`check_diffusion_stability`    — relax both endpoints and the NEB,
  store and return ``(E_a, E_b, E_ts)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.constraints import FixAtoms

from autokmc.io.calculators import acquire_calculator
from autokmc.io.atoms import copy_atoms_with_results
from autokmc.io.calculation_cache import (
    CalculationFingerprintMemo,
    apply_cached_states,
    calculation_cache_key,
    calculator_identity,
    input_coordinate_frame_fingerprint,
    load_calculation_record,
    make_calculation_record,
    state_payload,
    write_calculation_record,
)
from autokmc.io.reaction_graph import normalise_reaction_graph
from autokmc.species.smiles import smiles_to_dirname
from autokmc.core.pbc import full_pbc_for_cell
from autokmc.sites.diffusion import (
    DiffusionSite,
    DiffusionLateral,
    _member_clique_union,
    _reactant_orbit_label,
)
from autokmc.sites.stability.adsorption import (
    _surface_bfs_shells,
    _lateral_node_order,
    _discard_lateral_calculation,
    _promote_full_occupied_lateral,
    _check_connectivity_stable,
    _check_intended_coordination_stable,
    _bond_set,
    SurfaceConnectivityError,
    AdsorbateDissociationError,
    OptimisationFailedError,
)
from autokmc.sites.stability.neb import (
    make_neb_band,
    neb_optimizer_logfile,
    project_neb_path,
    resolve_neb_image_count,
    run_neb,
)
from autokmc.sites.stability.intermediate_pruning import (
    CompositeDirectEventDetected,
    DIRECT_EVENT_ELEMENTARY,
    classify_diffusion_intermediate,
    intermediate_pruning_network_signature,
    retain_refinement_and_maybe_suppress,
)
from autokmc.core.constants import (
    LATERAL_SHELLS_DEFAULT,
    NL_MULT_DEFAULT,
    NEB_BAND_EVAL,
    NEB_IMAGE_SPACING,
    NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    NEB_INTERMEDIATE_MAX_REFINEMENTS,
    NEB_INTERMEDIATE_MINIMUM_PROMINENCE,
    NEB_INTERMEDIATE_REFINEMENT_POLICY,
    NEB_INTERMEDIATE_STAGNATION_STEPS,
    NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER,
    NEB_MAX_IMAGES,
    NEB_MIN_IMAGES,
    NEB_N_IMAGES,
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
    NEB_METHOD,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Diffusion-specific lateral predicates
# ---------------------------------------------------------------------------
#
# Preserve the distinction between endpoint A, endpoint B, and a third-party
# occupied adsorbate.  The cached energy_a / energy_b belong to the member's
# ordered endpoints; allowing an isomorphism to exchange A and B would reuse
# an uphill barrier for a downhill hop without exchanging those energies.


def _diffusion_lateral_node_match(d1: dict, d2: dict) -> bool:
    """Lateral-iso predicate for the diffusion ego-graph.

    Same as :func:`autokmc.sites.stability.adsorption._lateral_node_match` but
    additionally requires molecular ``reactant_orbit`` to agree on adsorbate
    nodes.  Endpoint roles preserve A/B ordering and keep endpoints distinct
    from third-party neighbours of the same SMILES; symmetry-equivalent atoms
    may be interchanged, and symmetry-inequivalent atoms of the same element
    remain distinct.
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
        if _reactant_orbit_label(d1) != _reactant_orbit_label(d2):
            return False
        # Treat missing ``endpoint_role`` as None on both sides.
        if d1.get("endpoint_role") != d2.get("endpoint_role"):
            return False
    return True


def _diffusion_lateral_fingerprint(g: nx.Graph) -> tuple:
    """Cheap pre-filter mirroring :func:`_diffusion_lateral_node_match`."""
    node_sigs = tuple(
        sorted(
            (
                d.get("type", "X"),
                d.get("element", "X"),
                int(d.get("iso_class", -1)) if d.get("type") == "adsorbate" else -1,
                str(d.get("reactant", "")) if d.get("type") == "adsorbate" else "",
                _reactant_orbit_label(d) if d.get("type") == "adsorbate" else -1,
                str(d.get("endpoint_role", "")) if d.get("type") == "adsorbate" else "",
                g.degree(n),
            )
            for n, d in g.nodes(data=True)
        )
    )
    return (
        g.graph.get("environment_scope", "local"),
        g.number_of_nodes(), g.number_of_edges(), node_sigs,
    )


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class DiffusionStabilityError(Exception):
    """Base class for diffusion (NEB) stability failures."""


class EndpointStabilityError(DiffusionStabilityError):
    """One of the two endpoint relaxations failed.

    Wraps the underlying :class:`SurfaceConnectivityError`,
    :class:`AdsorbateDissociationError`, or :class:`OptimisationFailedError`
    in ``__cause__``.
    """


class NEBNotConvergedError(DiffusionStabilityError):
    """CI-NEB band did not converge within ``max_steps``."""


class TransitionStateInvalidError(DiffusionStabilityError):
    """The selected diffusion-path image is chemically invalid.

    Either its energy is non-finite or the migrating molecule fragmented.
    An endpoint-like maximum is allowed because a valid low-barrier or
    barrierless diffusion path can fall back into either endpoint basin.
    """


# Compatibility aliases for callers that imported the former channel-local
# helpers.  The implementation now lives in ``stability.neb``.
_make_neb_band = make_neb_band
_neb_optimizer_logfile = neb_optimizer_logfile


# ---------------------------------------------------------------------------
# Lateral ego-graph (diffusion variant)
# ---------------------------------------------------------------------------


def _build_diffusion_lateral_ego_graph(
    G: nx.Graph,
    seed_clique_union: frozenset,
    n_shells: int,
    *,
    endpoint_a_ids: frozenset,
    endpoint_b_ids: frozenset,
    ignore_occupied_neighbours: bool = False,
    include_all_occupied: bool = False,
) -> nx.Graph:
    """Build the lateral ego-graph for a diffusion pair.

    Mirrors :func:`autokmc.sites.stability.adsorption._build_lateral_ego_graph`
    but seeds the surface BFS from the *union* of both endpoints' bonded
    cliques and treats both endpoints as labelled occupied leaves with a
    distinct ``endpoint_role="a"`` or ``"b"`` tag.

    Parameters
    ----------
    G : nx.Graph
    seed_clique_union : frozenset[int]
        Surface-node ids forming the BFS seed (``clq_a ∪ clq_b``).
    n_shells : int
        BFS depth through ``type == "surface"`` nodes.
    endpoint_a_ids, endpoint_b_ids : frozenset[int]
        Adsorbate-node ids of the two endpoints.  Excluded from the
        surface BFS visited set and from the occupied-neighbour leaf
        collection, then re-added as labelled occupied leaves with
        intramolecular and anchor edges.
    ignore_occupied_neighbours : bool
        When ``True``, third-party occupied adsorbate neighbours are **not**
        collected as leaves (endpoints are still added with their role tag).
        Members with equivalent ordered endpoints share a "bare" lateral
        class, effectively disabling lateral interactions for diffusion.
        Default ``False``.
    """
    endpoint_ids: frozenset = frozenset(endpoint_a_ids) | frozenset(endpoint_b_ids)

    # Static surface BFS (cached on G.graph["_surface_shells_cache"]).
    visited_full = (
        frozenset(n for n, d in G.nodes(data=True) if d.get("type") == "surface")
        if include_all_occupied else _surface_bfs_shells(G, seed_clique_union, n_shells)
    )
    visited: set = set(visited_full) - endpoint_ids

    # Collect *other* occupied adsorbate leaves adjacent to the BFS set;
    # endpoints are added explicitly afterwards with roles that preserve
    # their A/B ordering and distinguish them from occupied spectators.
    ads_leaves: set = set()
    if include_all_occupied:
        ads_leaves.update(
            n for n, d in G.nodes(data=True)
            if d.get("type") == "adsorbate"
            and d.get("occupied", False)
            and n not in endpoint_ids
        )
    if not ignore_occupied_neighbours:
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
    result.graph["environment_scope"] = "all_occupied" if include_all_occupied else "local"

    for nid in endpoint_ids:
        if nid not in G:
            continue
        d = G.nodes[nid]
        role = "a" if nid in endpoint_a_ids else "b"
        if nid not in result:
            result.add_node(
                nid,
                element=d.get("element"),
                type=d.get("type", "adsorbate"),
                iso_class=int(d.get("iso_class", -1)),
                reactant=str(d.get("reactant", "")),
                reactant_index=int(d.get("reactant_index", -1)),
                reactant_orbit=_reactant_orbit_label(d),
                occupied=True,
                endpoint_role=role,
            )
        else:
            result.nodes[nid]["occupied"] = True
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


# ---------------------------------------------------------------------------
# Public — lateral classifier
# ---------------------------------------------------------------------------


def check_diffusion_site_lateral(
    G: nx.Graph,
    diffusion_site: DiffusionSite,
    member_index: int,
    *,
    n_shells: int | None = None,
    ignore_lateral: bool = False,
    include_all_occupied: bool = False,
    _assign_member: bool = True,
) -> DiffusionLateral:
    """Classify the lateral-interaction environment of one hop-pair member.

    Builds the diffusion lateral ego-graph and either appends *member_index*
    to a matching :class:`DiffusionLateral` already on *diffusion_site* or
    creates a new one.  Returns the matching (or new) lateral class.

    Parameters
    ----------
    G : nx.Graph
    diffusion_site : DiffusionSite
        Parent iso-class whose ``lateral_classes`` list is mutated.
    member_index : int
        Index into ``diffusion_site.member_node_ids``.
    n_shells : int | None
        BFS depth.  ``None`` (default) → :data:`LATERAL_SHELLS_DEFAULT`.
    ignore_lateral : bool
        When ``True``, third-party occupied adsorbate neighbours are excluded
        from the ego-graph.  Members share a bare class when their ordered
        A/B environments are equivalent.  Effectively disables lateral
        interactions for diffusion.  Default ``False``.
    include_all_occupied : bool
        Classify the complete occupied surface, overriding local-shell and
        ignored-neighbour approximations for free-energy calculations.

    Raises
    ------
    IndexError
        *member_index* out of range.
    ValueError
        Either endpoint has no bonded surface atoms (empty clique union).
    """
    if member_index < 0 or member_index >= len(diffusion_site.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — DiffusionSite "
            f"iso_class={diffusion_site.iso_class} has "
            f"{len(diffusion_site.member_node_ids)} member(s)."
        )

    depth: int = LATERAL_SHELLS_DEFAULT if n_shells is None else int(n_shells)

    site_a, m_a, site_b, m_b = diffusion_site.members[member_index]
    a_node_ids, b_node_ids = diffusion_site.member_node_ids[member_index]

    clq_a = _member_clique_union(site_a, m_a)
    clq_b = _member_clique_union(site_b, m_b)
    if not clq_a or not clq_b:
        raise ValueError(
            f"Member {member_index} of DiffusionSite "
            f"iso_class={diffusion_site.iso_class} has an empty bonded "
            f"surface clique on one of its endpoints — cannot build "
            f"a lateral ego-graph."
        )

    seed = clq_a | clq_b
    endpoint_a_ids = frozenset(int(n) for n in a_node_ids if n in G)
    endpoint_b_ids = frozenset(int(n) for n in b_node_ids if n in G)

    ego = _build_diffusion_lateral_ego_graph(
        G,
        frozenset(seed),
        depth,
        endpoint_a_ids=endpoint_a_ids,
        endpoint_b_ids=endpoint_b_ids,
        ignore_occupied_neighbours=ignore_lateral,
        include_all_occupied=include_all_occupied,
    )

    fkey = _diffusion_lateral_fingerprint(ego)

    fp_index: dict | None = getattr(diffusion_site, "_lateral_fp_index", None)
    if fp_index is None:
        fp_index = {}
        diffusion_site._lateral_fp_index = fp_index

    # If this member was previously assigned to a different lateral class,
    # remove it from that class's members list before re-assigning so
    # ``lc.members`` always reflects the current classification.
    def _drop_from_other_classes(new_lc=None) -> None:
        if not _assign_member:
            return
        for other in diffusion_site.lateral_classes:
            if other is new_lc:
                continue
            if member_index in other.members:
                other.members.remove(member_index)

    for lc in fp_index.get(fkey, ()):
        if lc.n_shells != depth or lc.ego_graph is None:
            continue
        if lc.ego_graph.graph.get("environment_scope", "local") != ego.graph["environment_scope"]:
            continue
        gm = isomorphism.GraphMatcher(
            ego,
            lc.ego_graph,
            node_match=_diffusion_lateral_node_match,
        )
        if gm.is_isomorphic():
            if _assign_member:
                _drop_from_other_classes(new_lc=lc)
                if member_index not in lc.members:
                    lc.members.append(member_index)
                lc._seed_only = False
            _log.debug(
                "check_diffusion_site_lateral: diff_iso=%d member=%d → existing lateral_class=%d",
                diffusion_site.iso_class,
                member_index,
                lc.lateral_class,
            )
            return lc

    _drop_from_other_classes(new_lc=None)
    new_lc = DiffusionLateral(
        lateral_class=len(diffusion_site.lateral_classes),
        ego_graph=ego,
        n_shells=depth,
        members=[member_index] if _assign_member else [],
    )
    new_lc._seed_only = not _assign_member
    new_lc._fingerprint = fkey
    diffusion_site.lateral_classes.append(new_lc)
    fp_index.setdefault(fkey, []).append(new_lc)

    _log.debug(
        "check_diffusion_site_lateral: diff_iso=%d member=%d → new lateral_class=%d  (total=%d)",
        diffusion_site.iso_class,
        member_index,
        new_lc.lateral_class,
        len(diffusion_site.lateral_classes),
    )
    return new_lc


def get_diffusion_bare_lateral(
    G: nx.Graph,
    diffusion_site: DiffusionSite,
    member_index: int,
    *,
    n_shells: int | None = None,
) -> DiffusionLateral:
    """Find or create the bare class without reassigning the live member.

    The normal lateral classifier mutates class membership.  Bare-first NEB
    seeding needs the same canonical graph/class lookup while leaving the
    member attached to its real, currently occupied environment.
    """
    lateral_class = check_diffusion_site_lateral(
        G,
        diffusion_site,
        member_index,
        n_shells=n_shells,
        ignore_lateral=True,
        _assign_member=False,
    )
    if not lateral_class.members:
        lateral_class._seed_only = True
    return lateral_class


# ---------------------------------------------------------------------------
# Atoms-builder helpers
# ---------------------------------------------------------------------------


def _ordered_endpoint_nodes(G: nx.Graph, endpoint_ids) -> list[int]:
    """Order endpoint adsorbate nodes by ``reactant_index`` (SMILES atom index).

    Both endpoints of a hop share the same SMILES, so this gives an
    atom-to-atom correspondence across A and B (required by ASE NEB).
    """
    present = [int(n) for n in endpoint_ids if n in G]
    return sorted(
        present,
        key=lambda n: int(G.nodes[n].get("reactant_index", n)),
    )


def _build_diffusion_atoms(
    G: nx.Graph,
    lateral_class: DiffusionLateral,
    endpoint_a_ids: list[int],
    endpoint_b_ids: list[int],
    *,
    endpoint_position: str,
    frozen_indices: list[int] | None = None,
    base_atoms: Atoms | None = None,
) -> tuple[Atoms, int, int, list[int], list[int]]:
    """Build a single-endpoint Atoms object for a diffusion calculation.

    Atom layout::

        [ slab | lat_neighbours | migrating_molecule ]

    The migrating-molecule block uses A's nodes as the canonical atom
    identity (ordered by ``reactant_index``); only the *positions* depend
    on ``endpoint_position``:

    * ``"a"`` → coordinates from A's nodes.
    * ``"b"`` → coordinates from B's nodes (matched 1-to-1 by
      ``reactant_index``).

    Lateral-neighbour atoms are read from ``lateral_class.ego_graph`` and
    expanded to full placements via
    :func:`autokmc.sites.stability.adsorption._expand_to_full_placement`, with
    both endpoint id-sets excluded so the migrating molecule is never
    double-counted.

    Returns
    -------
    atoms : Atoms
    n_slab : int
    n_lat : int
    migrating_atom_indices : list[int]
        ASE indices of the migrating molecule (length = SMILES atom count).
    migrating_node_ids : list[int]
        Graph node ids backing the migrating-molecule block, in the same
        order as ``migrating_atom_indices``.  These are A's nodes when
        ``endpoint_position == "a"`` and B's nodes when ``endpoint_position
        == "b"`` — i.e. the node ids whose ``G.nodes[nid]["clique"]``
        entries describe the *intended* surface coordination at the
        endpoint we just built (used by the post-relaxation
        :func:`_check_intended_coordination_stable` check).
    """
    if endpoint_position not in ("a", "b"):
        raise ValueError(f"endpoint_position must be 'a' or 'b', got {endpoint_position!r}")

    a_ordered = _ordered_endpoint_nodes(G, endpoint_a_ids)
    b_ordered = _ordered_endpoint_nodes(G, endpoint_b_ids)
    if len(a_ordered) != len(b_ordered):
        raise ValueError(
            "Diffusion endpoints have differing atom counts — cannot pair "
            "atoms across A and B for an NEB band."
        )

    endpoint_id_set: frozenset = frozenset(a_ordered) | frozenset(b_ordered)

    # First, add the slab atoms.
    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True) if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )

    # Next, add the lateral neighbors without either endpoint.
    lat_nodes = _lateral_node_order(G, lateral_class, endpoint_id_set)

    # Then add the migrating molecule at the selected endpoint.
    chosen_nodes = a_ordered if endpoint_position == "a" else b_ordered
    symbols_mig = [G.nodes[n]["element"] for n in a_ordered]  # always A
    positions_mig = [np.asarray(G.nodes[n]["position"], dtype=float) for n in chosen_nodes]

    # Finally, assemble the complete structure.
    slab_lat_nodes = slab_nodes + lat_nodes
    n_slab = len(slab_nodes)
    n_lat = len(lat_nodes)
    n_mig = len(symbols_mig)

    cell = np.array(G.graph["cell"], dtype=float)
    pbc = full_pbc_for_cell(cell)

    if base_atoms is not None:
        # Reuse the relaxed slab+lat positions from a prior endpoint
        # relaxation so the NEB endpoints share the same surface basin.
        # Only the migrating-molecule block is overwritten with the
        # endpoint-specific coordinates.
        if len(base_atoms) != n_slab + n_lat + n_mig:
            raise ValueError(
                "base_atoms has wrong length for the slab+lat+mig layout: "
                f"got {len(base_atoms)}, expected {n_slab + n_lat + n_mig}."
            )
        atoms = base_atoms.copy()
        positions = atoms.get_positions()
        positions[n_slab + n_lat : n_slab + n_lat + n_mig] = np.asarray(
            positions_mig,
            dtype=float,
        )
        atoms.set_positions(positions)
        atoms.set_pbc(pbc)
    else:
        symbols = [G.nodes[n]["element"] for n in slab_lat_nodes] + symbols_mig
        positions = [
            np.asarray(G.nodes[n]["position"], dtype=float) for n in slab_lat_nodes
        ] + positions_mig
        atoms = Atoms(
            symbols=symbols,
            positions=np.asarray(positions, dtype=float),
            cell=cell,
            pbc=pbc,
        )

    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))

    migrating_atom_indices = list(range(n_slab + n_lat, n_slab + n_lat + n_mig))
    # Return the node-id ordering that matches each migrating atom's actual
    # identity at this endpoint: A's nodes when at A, B's nodes when at B.
    # The element symbols are taken from A throughout (both endpoints share
    # the SMILES) but the per-atom *clique* attribute comes from G under
    # whichever node id physically describes the placement at the relevant
    # endpoint.  ``_check_intended_coordination_stable`` looks up
    # ``G.nodes[nid]["clique"]`` so it must receive the endpoint-correct nids.
    return atoms, n_slab, n_lat, migrating_atom_indices, list(chosen_nodes)


# ---------------------------------------------------------------------------
# Endpoint relaxation helper
# ---------------------------------------------------------------------------


def _relax_endpoint(
    atoms_init: Atoms,
    *,
    calculator,
    fmax: float,
    max_steps: int,
    optimizer: str,
    optimizer_kwargs: dict[str, Any] | None = None,
    frozen_indices: list[int] | None,
    nl_mult: float,
    n_slab: int,
    n_lat: int,
    n_mig: int,
    G: nx.Graph,
    self_node_ids: frozenset,
    self_node_order: list[int] | None,
    state_label: str,
    verbose: bool,
    lateral_node_order: list[int] | None = None,
) -> tuple[Atoms, float]:
    """Relax one endpoint and run the standard stability checks.

    Wraps every underlying :class:`SiteStabilityError` subclass into an
    :class:`EndpointStabilityError` with ``__cause__`` preserved.
    """
    from autokmc.structure import (  # local: avoid cycle
        StructureOptimisationError,
        optimise_structure,
    )

    atoms_opt: Atoms | None = None
    try:
        with acquire_calculator(calculator, purpose=f"diffusion {state_label} relaxation") as calc:
            atoms_opt = optimise_structure(
                atoms_init,
                calculator=calc,
                fmax=fmax,
                steps=max_steps,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                verbose=verbose,
            )

            forces = atoms_opt.get_forces()
            if frozen_indices:
                free_mask = np.ones(len(atoms_opt), dtype=bool)
                free_mask[list(frozen_indices)] = False
                max_force = float(np.linalg.norm(forces[free_mask], axis=1).max())
            else:
                max_force = float(np.linalg.norm(forces, axis=1).max())

            if max_force > fmax:
                raise OptimisationFailedError(
                    f"[{state_label}] optimizer {optimizer!r} did not converge: "
                    f"max|F|={max_force:.4f} eV/Å after {max_steps} steps "
                    f"(fmax={fmax} eV/Å)."
                )

            energy = float(atoms_opt.get_potential_energy())
            atoms_opt.set_pbc(atoms_init.get_pbc())

            n_ads = n_lat + n_mig
            ads_indices = set(range(n_slab, n_slab + n_ads))
            _check_connectivity_stable(
                atoms_init,
                atoms_opt,
                n_slab,
                n_ads,
                state_label,
                nl_mult,
                relevant_indices=ads_indices,
                n_lat=n_lat,
            )
            _check_intended_coordination_stable(
                atoms_opt,
                G,
                self_node_ids,
                n_slab,
                n_lat,
                nl_mult,
                self_node_order=self_node_order,
            )
            if lateral_node_order:
                _check_intended_coordination_stable(
                    atoms_opt, G, lateral_node_order, n_slab, 0, nl_mult,
                    self_node_order=lateral_node_order,
                )

            if verbose:
                print(f"  [{state_label}] E={energy:.4f} eV  max|F|={max_force:.4f} eV/Å  stable")
            atoms_opt = copy_atoms_with_results(
                atoms_opt,
                energy=energy,
                forces=forces,
            )
        return atoms_opt, energy

    except StructureOptimisationError as exc:
        wrapped = EndpointStabilityError(f"Endpoint '{state_label}' relaxation failed: {exc}")
        wrapped.atoms = exc.atoms
        wrapped.state_label = state_label
        raise wrapped from exc
    except (SurfaceConnectivityError, AdsorbateDissociationError, OptimisationFailedError) as exc:
        wrapped = EndpointStabilityError(f"Endpoint '{state_label}' relaxation failed: {exc}")
        if atoms_opt is not None:
            wrapped.atoms = copy_atoms_with_results(atoms_opt)
        wrapped.state_label = state_label
        raise wrapped from exc


# ---------------------------------------------------------------------------
# NEB band construction + TS validity
# ---------------------------------------------------------------------------


def _check_ts_validity(
    atoms_ts: Atoms,
    atoms_a: Atoms,
    atoms_b: Atoms,
    *,
    n_slab: int,
    n_lat: int,
    n_mig: int,
    nl_mult: float,
    e_a: float,
    e_b: float,
    e_ts: float,
    energy_tol: float = 1e-3,
) -> None:
    """Validate the chemistry of the selected diffusion-path image.

    Detects failure modes that otherwise propagate silently into the KMC
    rate:

    1. **Energy ordering** — ``E_ts < max(E_a, E_b) − energy_tol`` (eV) means
       the band is monotonic / reversed and there is no interior saddle. This
       is accepted with a warning because it is valid for low-barrier or
       barrierless diffusion and the KMC applies the ``EA_MIN`` floor.
    2. **Migrating-molecule fragmentation** — the bond topology *within*
       the migrating block changed at the TS relative to **both** endpoints.
       (We tolerate matching the topology of either A or B — at the saddle
       the molecule may have already passed through bond rearrangement on
       one side.)

    Raises
    ------
    TransitionStateInvalidError
        If any of the above checks fires.
    """
    # 1. Energy ordering.
    if not (np.isfinite(e_ts) and np.isfinite(e_a) and np.isfinite(e_b)):
        raise TransitionStateInvalidError(
            f"TS / endpoint energies are not finite (E_a={e_a}, E_b={e_b}, E_ts={e_ts})."
        )
    e_max_endpoint = max(float(e_a), float(e_b))
    if float(e_ts) < e_max_endpoint - float(energy_tol):
        # No genuine saddle — the path is monotonic or the TS image sits
        # below both endpoints.  This is physically valid for a barrierless
        # hop; the KMC will apply the EA_MIN (0.1 eV) floor automatically.
        # Just warn and continue — do NOT raise.
        _log.warning(
            "NEB has no genuine saddle: E_ts=%.4f eV is below "
            "max(E_a, E_b)=%.4f eV (tol=%.3f). "
            "The KMC barrier will be floored at EA_MIN.",
            e_ts,
            e_max_endpoint,
            energy_tol,
        )

    # 2. Migrating-molecule connectivity.  Compute intra-mig bonds for
    # A, B and TS using the relevant_indices filter so only bonds involving
    # the migrating atoms are compared.  TS must match A *or* B.
    if n_mig >= 2:
        mig_indices = set(range(n_slab + n_lat, n_slab + n_lat + n_mig))
        bonds_a = _bond_set(atoms_a, nl_mult=nl_mult, relevant_indices=mig_indices)
        bonds_b = _bond_set(atoms_b, nl_mult=nl_mult, relevant_indices=mig_indices)
        bonds_ts = _bond_set(atoms_ts, nl_mult=nl_mult, relevant_indices=mig_indices)

        # Restrict the comparison to *intra*-migrating bonds (both ends in
        # the mig block) so that surface↔mig bond rearrangement at the saddle
        # is not flagged as fragmentation.
        def _intra(bonds: set) -> set:
            return {b for b in bonds if all(int(i) in mig_indices for i in b)}

        bonds_a_in = _intra(bonds_a)
        bonds_b_in = _intra(bonds_b)
        bonds_ts_in = _intra(bonds_ts)
        if bonds_ts_in != bonds_a_in and bonds_ts_in != bonds_b_in:
            raise TransitionStateInvalidError(
                "Migrating molecule fragmented (or its intramolecular bond "
                "topology changed at the TS): "
                f"bonds_ts={sorted(map(tuple, bonds_ts_in))} differ from "
                f"both bonds_A={sorted(map(tuple, bonds_a_in))} and "
                f"bonds_B={sorted(map(tuple, bonds_b_in))}."
            )


# ---------------------------------------------------------------------------
# Public — stability + NEB
# ---------------------------------------------------------------------------


def _apply_diffusion_thermochemistry(
    lateral_class: DiffusionLateral,
    diffusion_site: DiffusionSite,
    *,
    atoms_a: Atoms,
    atoms_b: Atoms,
    atoms_ts: Atoms,
    energy_a: float,
    energy_b: float,
    energy_ts: float,
    n_slab: int,
    n_lateral: int,
    n_migrating: int,
    calculator,
    free_energy_options,
    temperature_k: float | None,
    vib_cache_root: str | None,
) -> None:
    """Populate thermochemistry from cached or freshly calculated states."""
    if (
        free_energy_options is None
        or not getattr(free_energy_options, "enabled", False)
        or temperature_k is None
    ):
        return

    from pathlib import Path as _Path

    from autokmc.thermo.free_energy import (
        VibrationalStabilityError,
        compute_harmonic_thermo,
    )

    cache_dir_root = _Path(vib_cache_root) if vib_cache_root is not None else None
    per_lat_dir = (
        cache_dir_root
        / f"diff_{smiles_to_dirname(diffusion_site.reactant)}"
        / (f"diff_iso{diffusion_site.iso_class}_lat{lateral_class.lateral_class}")
        if cache_dir_root is not None
        else None
    )

    def _harm(atoms, label, energy_ev, drop_imag):
        is_ts = label == "ts"
        vib_indices = list(range(n_slab, len(atoms)))
        suffix = label.removeprefix("state_")
        setattr(lateral_class, f"vib_indices_{suffix}", vib_indices)
        try:
            return compute_harmonic_thermo(
                atoms,
                vib_indices,
                energy_ev=float(energy_ev),
                temperature_k=float(temperature_k),
                calculator=calculator,
                options=free_energy_options,
                cache_dir=(str(per_lat_dir) if per_lat_dir is not None else None),
                label=label,
                drop_imaginary=drop_imag,
                stationary_point=(
                    "diffusion_transition_state" if is_ts else "minimum"
                ),
            )
        except VibrationalStabilityError as exc:
            setattr(lateral_class, f"imaginary_{suffix}_ev", list(exc.imaginary_ev))
            lateral_class.stable = False
            lateral_class.invalid_reason = str(exc)
            error = TransitionStateInvalidError if is_ts else EndpointStabilityError
            raise error(str(exc)) from exc

    a_thermo = _harm(atoms_a, "state_a", energy_a, True)
    b_thermo = _harm(atoms_b, "state_b", energy_b, True)
    if getattr(free_energy_options, "include_ts_vibrations", True):
        ts_thermo = _harm(atoms_ts, "ts", energy_ts, True)
    else:
        ts_thermo = None
        lateral_class.vib_indices_ts = []

    if a_thermo is not None:
        lateral_class.g_correction_a = a_thermo["g_corr_ev"]
        lateral_class.g_a = a_thermo["g_total_ev"]
        lateral_class.zpe_a = a_thermo["zpe_ev"]
        lateral_class.entropy_a = a_thermo["entropy_ev_per_k"]
        lateral_class.frequencies_a_ev = a_thermo["frequencies_ev"]
        lateral_class.imaginary_a_ev = a_thermo["imaginary_ev"]
    if b_thermo is not None:
        lateral_class.g_correction_b = b_thermo["g_corr_ev"]
        lateral_class.g_b = b_thermo["g_total_ev"]
        lateral_class.zpe_b = b_thermo["zpe_ev"]
        lateral_class.entropy_b = b_thermo["entropy_ev_per_k"]
        lateral_class.frequencies_b_ev = b_thermo["frequencies_ev"]
        lateral_class.imaginary_b_ev = b_thermo["imaginary_ev"]
    if ts_thermo is not None:
        lateral_class.g_correction_ts = ts_thermo["g_corr_ev"]
        lateral_class.g_ts = ts_thermo["g_total_ev"]
        lateral_class.zpe_ts = ts_thermo["zpe_ev"]
        lateral_class.entropy_ts = ts_thermo["entropy_ev_per_k"]
        lateral_class.frequencies_ts_ev = ts_thermo["frequencies_ev"]
        lateral_class.imaginary_ts_ev = ts_thermo["imaginary_ev"]
    elif a_thermo is not None and b_thermo is not None:
        avg_corr = 0.5 * (a_thermo["g_corr_ev"] + b_thermo["g_corr_ev"])
        lateral_class.g_correction_ts = float(avg_corr)
        lateral_class.g_ts = float(energy_ts) + float(avg_corr)


def _write_diffusion_calculation_cache(
    calculation_cache_root: str,
    cache_key: str,
    cache_graph: nx.Graph,
    cache_parameters: dict,
    cache_inputs: dict,
    fingerprint_memo: CalculationFingerprintMemo,
    diffusion_site: DiffusionSite,
    lateral_class: DiffusionLateral,
    *,
    atoms_a: Atoms,
    atoms_b: Atoms,
    atoms_ts: Atoms,
    energy_a: float,
    energy_b: float,
    energy_ts: float,
) -> None:
    props_a = {
        name: getattr(lateral_class, name, None)
        for name in (
            "g_correction_a",
            "g_a",
            "zpe_a",
            "entropy_a",
            "frequencies_a_ev",
            "imaginary_a_ev",
            "vib_indices_a",
        )
    }
    props_b = {
        name: getattr(lateral_class, name, None)
        for name in (
            "g_correction_b",
            "g_b",
            "zpe_b",
            "entropy_b",
            "frequencies_b_ev",
            "imaginary_b_ev",
            "vib_indices_b",
        )
    }
    props_ts = {
        name: getattr(lateral_class, name, None)
        for name in (
            "g_correction_ts",
            "g_ts",
            "zpe_ts",
            "entropy_ts",
            "frequencies_ts_ev",
            "imaginary_ts_ev",
            "vib_indices_ts",
        )
    }
    public_neb_path = getattr(lateral_class, "atoms_neb_path", None)
    private_neb_path = getattr(lateral_class, "_warm_start_neb_path", None)
    cache_neb_path = public_neb_path or private_neb_path
    neb = None
    if cache_neb_path:
        cache_neb_energies = (
            getattr(lateral_class, "neb_path_energies", None)
            if public_neb_path
            else getattr(lateral_class, "_warm_start_neb_energies", None)
        )
        neb = {
            "energies_ev": list(cache_neb_energies or []),
            "path_atoms": list(cache_neb_path),
        }
    states = {
        "state_a": state_payload(
            atoms_a,
            energy_ev=energy_a,
            properties=props_a,
        ),
        "state_b": state_payload(
            atoms_b,
            energy_ev=energy_b,
            properties=props_b,
        ),
        "transition": state_payload(
            atoms_ts,
            energy_ev=energy_ts,
            properties=props_ts,
        ),
    }
    refinement_initial = getattr(
        lateral_class,
        "atoms_neb_refinement_initial",
        None,
    )
    refinement_final = getattr(
        lateral_class,
        "atoms_neb_refinement_final",
        None,
    )
    refinement = getattr(lateral_class, "neb_intermediate_refinement", None)
    if (
        isinstance(refinement_initial, Atoms)
        and isinstance(refinement_final, Atoms)
        and isinstance(refinement, dict)
        and refinement.get("refinement_initial_energy_ev") is not None
        and refinement.get("refinement_final_energy_ev") is not None
    ):
        states.update(
            {
                "neb_refinement_initial": state_payload(
                    refinement_initial,
                    energy_ev=float(refinement["refinement_initial_energy_ev"]),
                ),
                "neb_refinement_final": state_payload(
                    refinement_final,
                    energy_ev=float(refinement["refinement_final_energy_ev"]),
                ),
            }
        )
    record = make_calculation_record(
        kind="diffusion",
        cache_key=cache_key,
        operation={
            "label": f"diffusion:{diffusion_site.reactant}",
            "reaction": f"{diffusion_site.reactant}* site_a -> site_b",
            "reactant_smiles": diffusion_site.reactant,
            "iso_class": int(diffusion_site.iso_class),
            "lateral_class": int(lateral_class.lateral_class),
            "temperature_k": cache_parameters["temperature_k"],
        },
        parameters=cache_parameters,
        inputs={
            **cache_inputs,
            "reactant_smiles": diffusion_site.reactant,
            "iso_class": int(diffusion_site.iso_class),
            "lateral_class": int(lateral_class.lateral_class),
        },
        states=states,
        reaction_graph=cache_graph,
        neb=neb,
        lateral_attributes={
            "neb_initialization": getattr(
                lateral_class,
                "neb_initialization",
                None,
            ),
            "neb_seed_fingerprint": getattr(
                lateral_class,
                "neb_seed_fingerprint",
                None,
            ),
            "neb_n_images": getattr(lateral_class, "neb_n_images", None),
            "neb_n_frames": getattr(lateral_class, "neb_n_frames", None),
            "neb_max_endpoint_displacement": getattr(
                lateral_class,
                "neb_max_endpoint_displacement",
                None,
            ),
            "neb_target_image_spacing": getattr(
                lateral_class,
                "neb_target_image_spacing",
                None,
            ),
            "neb_estimated_image_spacing": getattr(
                lateral_class,
                "neb_estimated_image_spacing",
                None,
            ),
            "neb_image_count_limited_by": getattr(
                lateral_class,
                "neb_image_count_limited_by",
                None,
            ),
            "neb_climb_performed": getattr(
                lateral_class,
                "neb_climb_performed",
                None,
            ),
            "neb_intermediate_refinement": getattr(
                lateral_class,
                "neb_intermediate_refinement",
                None,
            ),
            "neb_intermediate_refinement_history": getattr(
                lateral_class,
                "neb_intermediate_refinement_history",
                [],
            ),
            "direct_event_status": getattr(
                lateral_class,
                "direct_event_status",
                None,
            ),
            "direct_event_reason": getattr(
                lateral_class,
                "direct_event_reason",
                None,
            ),
            "direct_event_certificate": getattr(
                lateral_class,
                "direct_event_certificate",
                None,
            ),
            "direct_event_network_signature": getattr(
                lateral_class,
                "direct_event_network_signature",
                None,
            ),
        },
    )
    write_calculation_record(
        calculation_cache_root,
        "diffusion",
        cache_key,
        record,
        fingerprint_memo=fingerprint_memo,
    )


def check_diffusion_stability(
    G: nx.Graph,
    diffusion_site: DiffusionSite,
    member_index: int,
    lateral_class: DiffusionLateral,
    calculator,
    *,
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
    neb_geometry_guard_multiplier: float = (NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER),
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS,
    neb_intermediate_max_refinements: int = NEB_INTERMEDIATE_MAX_REFINEMENTS,
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    ),
    verbose: bool = False,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    neb_seed_path: Sequence[Atoms] | None = None,
    neb_seed_member_index: int | None = None,
    capture_neb_path: bool = False,
) -> tuple[float, float, float]:
    """Relax both endpoints and the NEB band; store and return energies.

    Pipeline
    --------
    1. Build endpoint-A Atoms (slab + lateral neighbours + migrating
       molecule at A's positions); relax with the configured optimizer via
       :func:`~autokmc.structure.optimise_structure`; run the standard
       connectivity / coordination stability checks.
    2. Same for endpoint-B (migrating molecule at B's positions, atom
       ordering identical to A).
    3. Build an ``n_images``-image NEB band between the two relaxed endpoints
       (IDPP or linear interpolation, configurable spring constant).  All
       images share one acquired calculator via ASE's
       SingleCalculatorNEB-style path.
    4. Run the selected NEB optimizer on the ordinary NEB to ``fmax``. If
       *climb* is enabled, retain the same band and spring constant, enable its
       climbing image, and converge a second optimization regardless of raw
       barrier height.
    5. Identify the TS as the highest-energy interior image; validate its
       finite energy and migrating-molecule connectivity; store all
       energies, atoms, and (optionally) the full band on *lateral_class*.

    On success ``lateral_class.stable`` is set to ``True`` and the energies
    / relaxed atoms are persisted on the lateral class.

    Parameters
    ----------
    G : nx.Graph
    diffusion_site : DiffusionSite
    member_index : int
    lateral_class : DiffusionLateral
        The lateral class returned by :func:`check_diffusion_site_lateral`
        for this member.
    calculator
        ASE calculator or CalculatorPool.  Endpoint relaxations and NEB
        calculations acquire calculators from the pool instead of copying
        them.
    frozen_indices : list[int] | None
        Slab indices to freeze with :class:`~ase.constraints.FixAtoms`.
        Same convention as :func:`autokmc.sites.stability.adsorption.check_site_stability`.
    fmax, max_steps : float, int
        NEB / endpoint convergence threshold (eV/Å) and max optimiser steps.
    n_images : int
        Number of *intermediate* NEB images (default :data:`NEB_N_IMAGES`).
    climb : bool
        Refine the converged ordinary NEB with a climbing image.
    spring_k : float
        NEB spring constant (eV/Å²).
    interpolation : str
        ``"linear"`` (default) or ``"idpp"``.
    nl_mult : float
        Cutoff multiplier for the connectivity stability checks.
    persist_neb_path : bool
        Store the full relaxed band on
        ``lateral_class.atoms_neb_path``.  Default ``False``.
    neb_seed_path : Sequence[Atoms] | None
        Optimized bare band used as a warm-start candidate.  It is projected
        into the current lateral atom layout after endpoint relaxation.
    capture_neb_path : bool
        Retain the optimized band privately for future warm starts without
        changing the public ``persist_neb_path`` output contract.
    verbose : bool

    Returns
    -------
    tuple[float, float, float]
        ``(E_a, E_b, E_ts)`` in eV.

    Raises
    ------
    IndexError
        *member_index* out of range.
    EndpointStabilityError
        An endpoint relaxation broke connectivity or did not converge.
    NEBNotConvergedError
        NEB band did not reach *fmax* in *max_steps*.
    TransitionStateInvalidError
        TS energy is non-finite or the migrating molecule fragmented.
    """
    if member_index < 0 or member_index >= len(diffusion_site.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — DiffusionSite "
            f"iso_class={diffusion_site.iso_class} has "
            f"{len(diffusion_site.member_node_ids)} member(s)."
        )

    site_a, m_a, site_b, m_b = diffusion_site.members[member_index]
    a_node_ids, b_node_ids = diffusion_site.member_node_ids[member_index]

    clq_a = _member_clique_union(site_a, m_a)
    clq_b = _member_clique_union(site_b, m_b)
    if not clq_a or not clq_b:
        raise ValueError(
            f"Member {member_index} of DiffusionSite "
            f"iso_class={diffusion_site.iso_class} has an empty bonded "
            f"surface clique on one of its endpoints — cannot run NEB."
        )

    self_a = frozenset(int(n) for n in a_node_ids if n in G)
    self_b = frozenset(int(n) for n in b_node_ids if n in G)
    if free_energy_options is not None and getattr(free_energy_options, "enabled", False):
        _promote_full_occupied_lateral(
            diffusion_site, lateral_class,
            _build_diffusion_lateral_ego_graph(
                G, clq_a | clq_b, lateral_class.n_shells,
                endpoint_a_ids=self_a, endpoint_b_ids=self_b,
                include_all_occupied=True,
            ),
            member_index, _diffusion_lateral_fingerprint,
        )
    cache_kind = "diffusion"
    cache_key: str | None = None
    cache_graph: nx.Graph | None = None
    cache_fingerprint_memo = CalculationFingerprintMemo()
    electronic_cache_state: tuple[float, float, float, Atoms, Atoms, Atoms] | None = None
    try:
        seed_images = None if neb_seed_path is None else list(neb_seed_path)
    except TypeError:
        seed_images = None
    seed_fingerprint = (
        None
        if not seed_images
        else input_coordinate_frame_fingerprint({"neb_seed_path": seed_images})
    )
    try:
        seed_member_index = int(neb_seed_member_index)
    except (TypeError, ValueError, OverflowError):
        seed_member_index = None
    same_member_seed = seed_images is not None and seed_member_index == int(member_index)
    if not same_member_seed:
        seed_images = None
        seed_fingerprint = None
    seed_projection_scope = "slab_and_reacting"
    thermochemistry_requested = bool(
        free_energy_options is not None
        and getattr(free_energy_options, "enabled", False)
        and free_energy_temperature_k is not None
    )
    cache_parameters = {
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "optimizer": str(optimizer).strip().lower(),
        "optimizer_kwargs": dict(optimizer_kwargs or {}),
        "neb_optimizer": str(neb_optimizer).strip().lower(),
        "neb_optimizer_kwargs": dict(neb_optimizer_kwargs or {}),
        "neb_climb_optimizer": str(neb_climb_optimizer or neb_optimizer).strip().lower(),
        "neb_climb_optimizer_kwargs": dict(
            (neb_optimizer_kwargs or {})
            if neb_climb_optimizer_kwargs is None
            else neb_climb_optimizer_kwargs
        ),
        "neb_method": str(neb_method).strip().lower(),
        "neb_geometry_guard_multiplier": float(neb_geometry_guard_multiplier),
        "neb_intermediate_stagnation_steps": int(
            neb_intermediate_stagnation_steps
        ),
        "neb_intermediate_max_refinements": int(
            neb_intermediate_max_refinements
        ),
        "neb_intermediate_energy_tolerance": float(
            neb_intermediate_energy_tolerance
        ),
        "neb_intermediate_minimum_prominence": float(
            neb_intermediate_minimum_prominence
        ),
        "n_images": int(n_images),
        "image_spacing": (None if image_spacing is None else float(image_spacing)),
        "min_images": int(min_images),
        "max_images": int(max_images),
        "climb": bool(climb),
        "spring_k": float(spring_k),
        "interpolation": str(interpolation),
        "nl_mult": float(nl_mult),
        "n_shells": int(lateral_class.n_shells),
        "persist_neb_path": bool(persist_neb_path),
        "capture_neb_path": bool(capture_neb_path),
        "neb_seed_policy": "auto_bare_transfer_resampled_v2",
        "neb_geometry_guard": (
            None
            if image_spacing is None
            else "max_gap_configurable_restore_lowest_fmax_halve_controls_v3"
        ),
        "neb_intermediate_refinement_policy": {
            "name": NEB_INTERMEDIATE_REFINEMENT_POLICY,
            "stagnation_steps": int(neb_intermediate_stagnation_steps),
            "max_refinements": int(neb_intermediate_max_refinements),
            "energy_tolerance_ev": float(neb_intermediate_energy_tolerance),
            "minimum_prominence_ev": float(
                neb_intermediate_minimum_prominence
            ),
        },
        "free_energy_enabled": bool(
            free_energy_options is not None and getattr(free_energy_options, "enabled", False)
        ),
        "temperature_k": (
            None if free_energy_temperature_k is None else float(free_energy_temperature_k)
        ),
    }
    if free_energy_options is not None:
        from autokmc.thermo.free_energy import vibrational_validation_parameters

        cache_parameters["free_energy"] = {
            **vibrational_validation_parameters(free_energy_options),
            "vibration_displacement": float(free_energy_options.vibration_displacement),
            "vibration_nfree": int(free_energy_options.vibration_nfree),
            "include_ts_vibrations": bool(free_energy_options.include_ts_vibrations),
            "min_frequency_ev": float(free_energy_options.min_frequency_ev),
            "symmetry_tolerance": float(free_energy_options.symmetry_tolerance),
            "default_spin": float(free_energy_options.default_spin),
            "default_geometry": str(free_energy_options.default_geometry),
        }

    # First, relax endpoint A.
    atoms_a_init, n_slab, n_lat, mig_idx_a, mig_node_order_a = _build_diffusion_atoms(
        G,
        lateral_class,
        list(a_node_ids),
        list(b_node_ids),
        endpoint_position="a",
        frozen_indices=frozen_indices,
    )
    lateral_class.atoms_a_initial = atoms_a_init.copy()
    lateral_class.atoms_a_initial.calc = None
    n_mig = len(mig_idx_a)

    if calculation_cache_root is not None:
        try:
            cache_parameters["calculator"] = calculator_identity(calculator)
            cache_graph = normalise_reaction_graph(lateral_class.ego_graph)
            cache_graph.graph["n_shells"] = int(lateral_class.n_shells)
            atoms_b_seed, _, _, _, _ = _build_diffusion_atoms(
                G,
                lateral_class,
                list(a_node_ids),
                list(b_node_ids),
                endpoint_position="b",
                frozen_indices=frozen_indices,
            )
            lateral_class.atoms_b_initial = atoms_b_seed.copy()
            lateral_class.atoms_b_initial.calc = None
            cache_identity = {
                "kind": cache_kind,
                "reactant_smiles": diffusion_site.reactant,
                "iso_class": int(diffusion_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
            }
            cache_inputs = {
                "state_a_initial": atoms_a_init,
                "state_b_seed_initial": atoms_b_seed,
                "neb_seed": {
                    "mode": (
                        "bare_transfer"
                        if seed_fingerprint is not None
                        else "configured_interpolation"
                    ),
                    "projection_scope": (
                        seed_projection_scope if seed_fingerprint is not None else None
                    ),
                    "path_sha256": seed_fingerprint,
                },
            }
            cache_key = calculation_cache_key(
                kind=cache_kind,
                identity=cache_identity,
                parameters=cache_parameters,
                inputs=cache_inputs,
            )
            cached = None
            if calculation_cache_lookup_enabled:
                cached = load_calculation_record(
                    calculation_cache_root,
                    cache_kind,
                    cache_key,
                    reaction_graph=cache_graph,
                    operation=cache_identity,
                    parameters=cache_parameters,
                    inputs=cache_inputs,
                    allow_electronic_match=True,
                    fingerprint_memo=cache_fingerprint_memo,
                )
            if cached is not None and apply_cached_states(
                lateral_class,
                cached,
                {
                    "state_a": ("energy_a", "atoms_a"),
                    "state_b": ("energy_b", "atoms_b"),
                    "transition": ("energy_ts", "atoms_ts"),
                },
                include_properties=cached.get("_cache_match") != "electronic",
            ):
                if thermochemistry_requested and n_lat:
                    spectator_nodes = _lateral_node_order(G, lateral_class, self_a | self_b)
                    try:
                        for endpoint in (lateral_class.atoms_a, lateral_class.atoms_b):
                            _check_intended_coordination_stable(
                                endpoint, G, spectator_nodes, n_slab, 0, nl_mult,
                                self_node_order=spectator_nodes,
                            )
                    except AdsorbateDissociationError:
                        _discard_lateral_calculation(lateral_class)
                        raise
                cached_refinement = getattr(
                    lateral_class,
                    "neb_intermediate_refinement",
                    None,
                )
                cached_initial = getattr(
                    lateral_class,
                    "atoms_neb_refinement_initial",
                    None,
                )
                cached_final = getattr(
                    lateral_class,
                    "atoms_neb_refinement_final",
                    None,
                )
                current_network_signature = intermediate_pruning_network_signature(
                    G,
                    "diffusion",
                    diffusion_site,
                )
                previous_network_signature = getattr(
                    lateral_class,
                    "direct_event_network_signature",
                    None,
                )
                lateral_class.direct_event_network_signature = (
                    current_network_signature
                )
                if (
                    previous_network_signature != current_network_signature
                    and isinstance(cached_refinement, dict)
                    and isinstance(cached_initial, Atoms)
                    and isinstance(cached_final, Atoms)
                ):
                    cached_certificate = classify_diffusion_intermediate(
                        G,
                        diffusion_site,
                        member_index,
                        cached_initial,
                        cached_final,
                        cached_refinement,
                        n_slab=n_slab,
                        n_lateral=n_lat,
                        n_reacting=n_mig,
                        nl_mult=nl_mult,
                    )
                    if cached_certificate is not None:
                        retain_refinement_and_maybe_suppress(
                            lateral_class,
                            cached_initial,
                            cached_final,
                            cached_refinement,
                            cached_certificate,
                        )
                if getattr(lateral_class, "direct_event_status", None) is None:
                    lateral_class.direct_event_status = DIRECT_EVENT_ELEMENTARY
                if capture_neb_path:
                    cached_path = list(getattr(lateral_class, "atoms_neb_path", None) or [])
                    cached_interior = len(cached_path) - 2
                    if cached_interior < 1:
                        lateral_class.stable = None
                        raise ValueError(
                            "cached bare diffusion result has no compatible optimized NEB path"
                        )
                    lateral_class._warm_start_neb_path = [image.copy() for image in cached_path]
                    for image in lateral_class._warm_start_neb_path:
                        image.calc = None
                    lateral_class._warm_start_neb_energies = list(
                        getattr(lateral_class, "neb_path_energies", None) or []
                    )
                    lateral_class._warm_start_member_index = int(member_index)
                    lateral_class.neb_n_images = cached_interior
                    lateral_class.neb_n_frames = len(cached_path)
                    if not persist_neb_path:
                        lateral_class.atoms_neb_path = None
                        lateral_class.neb_path_energies = None
                electronic_only = cached.get("_cache_match") == "electronic"
                if electronic_only and thermochemistry_requested:
                    # ``apply_cached_states`` marks the electronic states
                    # stable.  Clear that marker until the requested
                    # thermochemistry has completed so every failure between
                    # cache hydration and vibration completion is retryable.
                    lateral_class.stable = None
                if verbose:
                    print(
                        f"  [cache] diffusion iso={diffusion_site.iso_class} "
                        f"lat={lateral_class.lateral_class}: loaded "
                        "endpoint/NEB calculation"
                        + (
                            "; recomputing thermochemistry"
                            if electronic_only and thermochemistry_requested
                            else ""
                        )
                    )
                energies = (
                    float(lateral_class.energy_a),
                    float(lateral_class.energy_b),
                    float(lateral_class.energy_ts),
                )
                if not electronic_only or not thermochemistry_requested:
                    return energies
                electronic_cache_state = (
                    *energies,
                    lateral_class.atoms_a,
                    lateral_class.atoms_b,
                    lateral_class.atoms_ts,
                )
        except CompositeDirectEventDetected:
            raise
        except Exception as exc:
            _log.debug(
                "check_diffusion_stability: calculation cache lookup failed "
                "(diff_iso=%d lat=%d): %s",
                diffusion_site.iso_class,
                lateral_class.lateral_class,
                exc,
            )

    if electronic_cache_state is not None:
        (
            E_a,
            E_b,
            E_ts,
            atoms_a_opt,
            atoms_b_opt,
            atoms_ts,
        ) = electronic_cache_state
        _apply_diffusion_thermochemistry(
            lateral_class,
            diffusion_site,
            atoms_a=atoms_a_opt,
            atoms_b=atoms_b_opt,
            atoms_ts=atoms_ts,
            energy_a=E_a,
            energy_b=E_b,
            energy_ts=E_ts,
            n_slab=n_slab,
            n_lateral=n_lat,
            n_migrating=n_mig,
            calculator=calculator,
            free_energy_options=free_energy_options,
            temperature_k=free_energy_temperature_k,
            vib_cache_root=vib_cache_root,
        )
        lateral_class.direct_event_status = DIRECT_EVENT_ELEMENTARY
        lateral_class.direct_event_reason = None
        lateral_class.direct_event_certificate = None
        lateral_class.stable = True
        assert calculation_cache_root is not None
        assert cache_key is not None
        assert cache_graph is not None
        _write_diffusion_calculation_cache(
            calculation_cache_root,
            cache_key,
            cache_graph,
            cache_parameters,
            cache_inputs,
            cache_fingerprint_memo,
            diffusion_site,
            lateral_class,
            atoms_a=atoms_a_opt,
            atoms_b=atoms_b_opt,
            atoms_ts=atoms_ts,
            energy_a=E_a,
            energy_b=E_b,
            energy_ts=E_ts,
        )
        return E_a, E_b, E_ts

    if verbose:
        print(
            f"  [endpoint A] atoms={len(atoms_a_init)}  (slab={n_slab}, lat={n_lat}, mig={n_mig})"
        )

    try:
        atoms_a_opt, E_a = _relax_endpoint(
            atoms_a_init,
            calculator=calculator,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            frozen_indices=frozen_indices,
            nl_mult=nl_mult,
            n_slab=n_slab,
            n_lat=n_lat,
            n_mig=n_mig,
            G=G,
            self_node_ids=self_a,
            self_node_order=mig_node_order_a,
            lateral_node_order=(
                _lateral_node_order(G, lateral_class, self_a | self_b)
                if thermochemistry_requested else None
            ),
            state_label="endpoint_a",
            verbose=verbose,
        )
    except EndpointStabilityError as exc:
        failed_atoms = getattr(exc, "atoms", None)
        if failed_atoms is not None:
            lateral_class.atoms_a = failed_atoms
        raise
    # Store endpoint A immediately so a later failure does not discard it.
    lateral_class.energy_a = E_a
    lateral_class.atoms_a = atoms_a_opt

    # Next, relax endpoint B. Start from the relaxed slab and lateral neighbors
    # of endpoint A, and replace only the migrating molecule. This keeps both
    # endpoints in the same surface basin for the NEB interpolation.
    atoms_b_init, _, _, mig_idx_b, mig_node_order_b = _build_diffusion_atoms(
        G,
        lateral_class,
        list(a_node_ids),
        list(b_node_ids),
        endpoint_position="b",
        frozen_indices=frozen_indices,
        base_atoms=atoms_a_opt,
    )
    lateral_class.atoms_b_initial = atoms_b_init.copy()
    lateral_class.atoms_b_initial.calc = None

    if verbose:
        print(
            f"  [endpoint B] atoms={len(atoms_b_init)}  (slab={n_slab}, lat={n_lat}, mig={n_mig})"
        )

    try:
        atoms_b_opt, E_b = _relax_endpoint(
            atoms_b_init,
            calculator=calculator,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            frozen_indices=frozen_indices,
            nl_mult=nl_mult,
            n_slab=n_slab,
            n_lat=n_lat,
            n_mig=n_mig,
            G=G,
            self_node_ids=self_b,
            self_node_order=mig_node_order_b,
            lateral_node_order=(
                _lateral_node_order(G, lateral_class, self_a | self_b)
                if thermochemistry_requested else None
            ),
            state_label="endpoint_b",
            verbose=verbose,
        )
    except EndpointStabilityError as exc:
        failed_atoms = getattr(exc, "atoms", None)
        if failed_atoms is not None:
            lateral_class.atoms_b = failed_atoms
        raise
    # Store endpoint B immediately so a later failure does not discard it.
    lateral_class.energy_b = E_b
    lateral_class.atoms_b = atoms_b_opt

    # With both endpoints relaxed, build and optimize the NEB band.
    image_selection = resolve_neb_image_count(
        atoms_a_opt,
        atoms_b_opt,
        fixed_n_images=int(n_images),
        image_spacing=image_spacing,
        min_images=int(min_images),
        max_images=int(max_images),
    )
    resolved_n_images = image_selection.n_images
    lateral_class.neb_n_images = resolved_n_images
    lateral_class.neb_n_frames = image_selection.n_frames
    lateral_class.neb_max_endpoint_displacement = image_selection.max_endpoint_displacement
    lateral_class.neb_target_image_spacing = image_selection.target_spacing
    lateral_class.neb_estimated_image_spacing = image_selection.estimated_linear_spacing
    lateral_class.neb_image_count_limited_by = image_selection.limited_by
    projected_seed_path = None
    if seed_images:
        projected_seed_path = project_neb_path(
            seed_images,
            atoms_a_opt,
            atoms_b_opt,
            n_slab=n_slab,
            n_lateral=n_lat,
            n_images=resolved_n_images,
            interpolation=interpolation,
            frozen_indices=frozen_indices,
            neb_method=neb_method,
            spring_k=spring_k,
        )
    lateral_class.neb_seed_fingerprint = seed_fingerprint
    lateral_class.neb_initialization = (
        "bare_transfer"
        if projected_seed_path is not None
        else ("configured_interpolation_fallback" if seed_images else "configured_interpolation")
    )
    if verbose:
        print(
            f"  [NEB] images={resolved_n_images} interior / "
            f"{image_selection.n_frames} frames  "
            f"spacing≈{image_selection.estimated_linear_spacing:.3f} Å  "
            f"climb={bool(climb)}  "
            f"fmax={float(fmax):.4f} eV/Å  max_steps={int(max_steps)}"
        )
        if projected_seed_path is not None:
            print(f"  [NEB] initialization=bare optimized path ({seed_projection_scope})")
        elif seed_images:
            print(f"  [NEB] bare path incompatible; using {interpolation} interpolation")

    def retain_intermediate_refinement(
        refinement_initial: Atoms,
        refinement_final: Atoms,
        metadata: dict[str, Any],
    ) -> None:
        lateral_class.direct_event_network_signature = (
            intermediate_pruning_network_signature(
                G,
                "diffusion",
                diffusion_site,
            )
        )
        certificate = classify_diffusion_intermediate(
            G,
            diffusion_site,
            member_index,
            refinement_initial,
            refinement_final,
            metadata,
            n_slab=n_slab,
            n_lateral=n_lat,
            n_reacting=n_mig,
            nl_mult=nl_mult,
        )
        retain_refinement_and_maybe_suppress(
            lateral_class,
            refinement_initial,
            refinement_final,
            metadata,
            certificate,
        )

    neb_result = run_neb(
        atoms_a_opt,
        atoms_b_opt,
        calculator=calculator,
        purpose="diffusion NEB",
        n_images=resolved_n_images,
        interpolation=str(interpolation),
        spring_k=float(spring_k),
        climb=bool(climb),
        frozen_indices=frozen_indices,
        fmax=float(fmax),
        max_steps=int(max_steps),
        optimizer=neb_optimizer,
        optimizer_kwargs=neb_optimizer_kwargs,
        climb_optimizer=neb_climb_optimizer,
        climb_optimizer_kwargs=neb_climb_optimizer_kwargs,
        neb_method=neb_method,
        band_eval=neb_band_eval,
        image_spacing=image_selection.target_spacing,
        geometry_guard_multiplier=neb_geometry_guard_multiplier,
        intermediate_stagnation_steps=neb_intermediate_stagnation_steps,
        intermediate_max_refinements=neb_intermediate_max_refinements,
        intermediate_energy_tolerance=neb_intermediate_energy_tolerance,
        intermediate_minimum_prominence=neb_intermediate_minimum_prominence,
        intermediate_optimizer=optimizer,
        intermediate_optimizer_kwargs=optimizer_kwargs,
        intermediate_min_images=min_images,
        intermediate_max_images=max_images,
        intermediate_refinement_callback=retain_intermediate_refinement,
        verbose=verbose,
        not_converged_error=NEBNotConvergedError,
        persist_path=persist_neb_path,
        # Keep the completed band until transition-state validation and
        # thermochemistry finish. A successful non-persistent run clears it
        # below.
        capture_path=True,
        initial_path=projected_seed_path,
        initial_path_callback=(
            lambda images: setattr(
                lateral_class,
                "atoms_neb_path_initial",
                images,
            )
        ),
        failure_path_callback=(
            lambda images: setattr(
                lateral_class,
                "atoms_neb_path",
                images,
            )
        ),
        logfile_factory=_neb_optimizer_logfile,
    )
    E_ts = neb_result.energy_ts
    atoms_ts = neb_result.atoms_ts
    k_ts = neb_result.transition_index

    # Store the transition state and path before validation so they remain
    # available when a channel-specific check rejects the result.
    lateral_class.energy_ts = E_ts
    lateral_class.neb_climb_performed = neb_result.climb_performed
    lateral_class.neb_n_images = neb_result.n_interior
    lateral_class.neb_n_frames = neb_result.n_interior + 2
    if neb_result.intermediate_refinement_performed:
        lateral_class.neb_max_endpoint_displacement = (
            neb_result.refinement_max_endpoint_displacement
        )
        lateral_class.neb_target_image_spacing = (
            neb_result.refinement_target_image_spacing
        )
        lateral_class.neb_estimated_image_spacing = (
            neb_result.refinement_estimated_image_spacing
        )
        lateral_class.neb_image_count_limited_by = (
            neb_result.refinement_image_count_limited_by
        )
    lateral_class.neb_intermediate_refinement = (
        {
            "performed": True,
            "policy": NEB_INTERMEDIATE_REFINEMENT_POLICY,
            "refinement_count": neb_result.intermediate_refinement_count,
            "max_refinements": neb_result.intermediate_max_refinements,
            "trigger": neb_result.intermediate_trigger,
            "source_stage": neb_result.intermediate_source_stage,
            "checkpoint_fmax_ev_per_ang": neb_result.intermediate_checkpoint_fmax,
            "checkpoint_optimizer_steps": (
                neb_result.intermediate_checkpoint_optimizer_steps
            ),
            "stagnation_steps": neb_result.intermediate_stagnation_steps,
            "optimizer_steps_at_detection": neb_result.intermediate_stalled_steps,
            "peak_image_index": neb_result.intermediate_peak_index,
            "left_state_image_index": neb_result.intermediate_left_index,
            "right_state_image_index": neb_result.intermediate_right_index,
            "stalled_profile_energies_ev": neb_result.intermediate_profile_energies,
            "refinement_initial_energy_ev": neb_result.refinement_initial_energy,
            "refinement_final_energy_ev": neb_result.refinement_final_energy,
            "replacement_interior_images": neb_result.n_interior,
            "replacement_max_endpoint_displacement_ang": (
                neb_result.refinement_max_endpoint_displacement
            ),
            "replacement_target_image_spacing_ang": (
                neb_result.refinement_target_image_spacing
            ),
            "replacement_estimated_image_spacing_ang": (
                neb_result.refinement_estimated_image_spacing
            ),
            "replacement_image_count_limited_by": (
                neb_result.refinement_image_count_limited_by
            ),
            "other_segments_refined": False,
        }
        if neb_result.intermediate_refinement_performed
        else None
    )
    lateral_class.atoms_neb_refinement_initial = (
        neb_result.refinement_initial_atoms
    )
    lateral_class.atoms_neb_refinement_final = neb_result.refinement_final_atoms
    lateral_class.atoms_ts = atoms_ts
    # Keep the final path public until every post-NEB check succeeds. If a
    # check fails, the path remains available for invalid-candidate output.
    lateral_class.neb_path_energies = neb_result.path_energies
    lateral_class.atoms_neb_path = neb_result.path_images
    if capture_neb_path and neb_result.path_images:
        lateral_class._warm_start_neb_path = [
            copy_atoms_with_results(image) for image in neb_result.path_images
        ]
        lateral_class._warm_start_neb_energies = list(neb_result.path_energies or [])
        lateral_class._warm_start_member_index = int(member_index)

    validation_initial = (
        neb_result.refinement_initial_atoms
        if neb_result.refinement_initial_atoms is not None
        else atoms_a_opt
    )
    validation_final = (
        neb_result.refinement_final_atoms
        if neb_result.refinement_final_atoms is not None
        else atoms_b_opt
    )
    validation_initial_energy = (
        neb_result.refinement_initial_energy
        if neb_result.refinement_initial_energy is not None
        else E_a
    )
    validation_final_energy = (
        neb_result.refinement_final_energy
        if neb_result.refinement_final_energy is not None
        else E_b
    )
    _check_ts_validity(
        atoms_ts,
        validation_initial,
        validation_final,
        n_slab=n_slab,
        n_lat=n_lat,
        n_mig=n_mig,
        nl_mult=nl_mult,
        e_a=validation_initial_energy,
        e_b=validation_final_energy,
        e_ts=E_ts,
    )

    _apply_diffusion_thermochemistry(
        lateral_class,
        diffusion_site,
        atoms_a=atoms_a_opt,
        atoms_b=atoms_b_opt,
        atoms_ts=atoms_ts,
        energy_a=E_a,
        energy_b=E_b,
        energy_ts=E_ts,
        n_slab=n_slab,
        n_lateral=n_lat,
        n_migrating=n_mig,
        calculator=calculator,
        free_energy_options=free_energy_options,
        temperature_k=free_energy_temperature_k,
        vib_cache_root=vib_cache_root,
    )
    if not persist_neb_path:
        lateral_class.atoms_neb_path_initial = None
        lateral_class.atoms_neb_path = None
        lateral_class.neb_path_energies = None
    # Finally, mark the class stable after the requested thermochemistry has
    # succeeded.
    lateral_class.direct_event_status = DIRECT_EVENT_ELEMENTARY
    lateral_class.direct_event_reason = None
    lateral_class.direct_event_certificate = None
    lateral_class.stable = True
    if verbose:
        print(
            f"  [NEB] converged=True steps={neb_result.optimizer_steps}  "
            f"E_ts={E_ts:.4f} eV  "
            f"image={k_ts}/{neb_result.n_interior}  stable"
        )

    _log.debug(
        "check_diffusion_stability: diff_iso=%d member=%d lat=%d  "
        "E_a=%.4f  E_b=%.4f  E_ts=%.4f eV  Ea_fwd=%.4f  Ea_rev=%.4f",
        diffusion_site.iso_class,
        member_index,
        lateral_class.lateral_class,
        E_a,
        E_b,
        E_ts,
        E_ts - E_a,
        E_ts - E_b,
    )
    if calculation_cache_root is not None and cache_key is not None and cache_graph is not None:
        _write_diffusion_calculation_cache(
            calculation_cache_root,
            cache_key,
            cache_graph,
            cache_parameters,
            cache_inputs,
            cache_fingerprint_memo,
            diffusion_site,
            lateral_class,
            atoms_a=atoms_a_opt,
            atoms_b=atoms_b_opt,
            atoms_ts=atoms_ts,
            energy_a=E_a,
            energy_b=E_b,
            energy_ts=E_ts,
        )
    return E_a, E_b, E_ts
