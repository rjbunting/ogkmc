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
   endpoints, then a CI-NEB band between them.  See
   :func:`check_diffusion_stability`.

Lateral ego-graph conventions (extends :mod:`autokmc.sites.stability.adsorption`)
---------------------------------------------------------------------------
* **BFS seed** — the *union* of A's and B's bonded surface cliques.
* **BFS frontier** — only ``type == "surface"`` nodes are traversed; both
  endpoint placements are excluded from the visited surface set.
* **Occupied adsorbate leaves** — every occupied ``type == "adsorbate"``
  node adjacent to the BFS surface set, *excluding* the two endpoints.
* **Endpoint inclusion** — both endpoints' atoms are added as labelled
  leaves with ``occupied=True`` and ``endpoint_role="endpoint"``.  The
  symmetric ``endpoint`` tag (rather than ``a`` / ``b``) ensures the
  iso-match is symmetric under A ↔ B, as a hop is intrinsically reversible.

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
* :class:`TransitionStateInvalidError` — TS lost connectivity / collapsed
  onto an endpoint.
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

import os
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from autokmc.io.calculators import acquire_calculator
from autokmc.io.calculation_cache import (
    apply_cached_states,
    calculation_cache_key,
    calculator_identity,
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
)
from autokmc.sites.stability.adsorption import (
    _surface_bfs_shells,
    _expand_to_full_placement,
    _check_connectivity_stable,
    _check_intended_coordination_stable,
    _bond_set,
    SurfaceConnectivityError,
    AdsorbateDissociationError,
    OptimisationFailedError,
)
from autokmc.core.constants import (
    LATERAL_SHELLS_DEFAULT,
    NL_MULT_DEFAULT,
    NEB_N_IMAGES,
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
)
from autokmc.utils.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Diffusion-specific lateral predicates
# ---------------------------------------------------------------------------
#
# The adsorption helpers ``_lateral_node_match`` / ``_lateral_fingerprint``
# do not look at ``endpoint_role``.  For diffusion we *must* preserve the
# distinction between an endpoint adsorbate (the migrating species at A or
# B) and a third-party occupied adsorbate that happens to share the same
# SMILES / iso_class — otherwise two physically different lateral
# environments (endpoint at site P with neighbour at Q vs. endpoint at Q
# with neighbour at P) collapse into the same lateral class and we cache
# the wrong NEB barrier against them.
#
# The tag is ``"endpoint"`` for both A and B, so the iso match remains
# symmetric under A↔B (a hop is intrinsically reversible).

def _diffusion_lateral_node_match(d1: dict, d2: dict) -> bool:
    """Lateral-iso predicate for the diffusion ego-graph.

    Same as :func:`autokmc.sites.stability.adsorption._lateral_node_match` but
    additionally requires ``endpoint_role`` and ``reactant_index`` to agree on
    adsorbate nodes so endpoints never map onto third-party neighbours of the
    same SMILES, and so symmetry-inequivalent atoms of the same element within a
    multi-atom adsorbate are not interchanged.
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
        # Treat missing ``endpoint_role`` as None on both sides.
        if d1.get("endpoint_role") != d2.get("endpoint_role"):
            return False
    return True


def _diffusion_lateral_fingerprint(g: nx.Graph) -> tuple:
    """Cheap pre-filter mirroring :func:`_diffusion_lateral_node_match`."""
    node_sigs = tuple(sorted(
        (
            d.get("type",      "X"),
            d.get("element",   "X"),
            int(d.get("iso_class",      -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("reactant",       "")) if d.get("type") == "adsorbate" else "",
            int(d.get("reactant_index", -1)) if d.get("type") == "adsorbate" else -1,
            str(d.get("endpoint_role",  "")) if d.get("type") == "adsorbate" else "",
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (g.number_of_nodes(), g.number_of_edges(), node_sigs)


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
    """TS image is not a valid saddle.

    Either the migrating molecule fragmented, the surface bond topology
    changed at the TS, or the TS image collapsed onto one of the endpoints
    (its bonded surface clique exactly matches A's or B's).
    """


# ---------------------------------------------------------------------------
# ASE NEB import shim (modern: ``ase.mep``; older: ``ase.neb``).
# ---------------------------------------------------------------------------

try:                                                              # pragma: no cover
    from ase.mep import NEB                                       # type: ignore
except ImportError:                                               # pragma: no cover
    from ase.neb import NEB                                       # type: ignore

try:                                                              # pragma: no cover
    from ase.mep import idpp_interpolate as _idpp_interpolate    # type: ignore
except ImportError:                                               # pragma: no cover
    try:
        from ase.neb import idpp_interpolate as _idpp_interpolate  # type: ignore
    except ImportError:                                           # pragma: no cover
        _idpp_interpolate = None


def _neb_optimizer_logfile(verbose: bool) -> str:
    return "-" if verbose else os.devnull


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
) -> nx.Graph:
    """Build the lateral ego-graph for a diffusion pair.

    Mirrors :func:`autokmc.sites.stability.adsorption._build_lateral_ego_graph`
    but seeds the surface BFS from the *union* of both endpoints' bonded
    cliques and treats both endpoints as labelled occupied leaves with a
    symmetric ``endpoint_role="endpoint"`` tag.

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
        This collapses all members to a single "bare" lateral class,
        effectively disabling lateral interactions for diffusion.
        Default ``False``.
    """
    endpoint_ids: frozenset = frozenset(endpoint_a_ids) | frozenset(endpoint_b_ids)

    # Static surface BFS (cached on G.graph["_surface_shells_cache"]).
    visited_full = _surface_bfs_shells(G, seed_clique_union, n_shells)
    visited: set = set(visited_full) - endpoint_ids

    # Collect *other* occupied adsorbate leaves adjacent to the BFS set;
    # endpoints are added explicitly afterwards so we can stamp them with
    # the symmetric endpoint_role label (preventing them from being
    # mistaken for third-party occupied adsorbates of the same SMILES).
    ads_leaves: set = set()
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

    for nid in endpoint_ids:
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
                endpoint_role  = "endpoint",
            )
        else:
            result.nodes[nid]["occupied"]      = True
            result.nodes[nid]["endpoint_role"] = "endpoint"
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
        from the ego-graph so every member always maps to the single bare
        lat0.  Effectively disables lateral interactions for diffusion.
        Default ``False``.

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
    a_node_ids, b_node_ids   = diffusion_site.member_node_ids[member_index]

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
        G, frozenset(seed), depth,
        endpoint_a_ids              = endpoint_a_ids,
        endpoint_b_ids              = endpoint_b_ids,
        ignore_occupied_neighbours  = ignore_lateral,
    )

    fkey = _diffusion_lateral_fingerprint(ego)

    fp_index: dict | None = getattr(diffusion_site, "_lateral_fp_index", None)
    if fp_index is None:
        fp_index = {}
        diffusion_site._lateral_fp_index = fp_index   # type: ignore[attr-defined]

    # If this member was previously assigned to a different lateral class,
    # remove it from that class's members list before re-assigning so
    # ``lc.members`` always reflects the current classification.
    def _drop_from_other_classes(new_lc=None) -> None:
        for other in diffusion_site.lateral_classes:
            if other is new_lc:
                continue
            if member_index in other.members:
                other.members.remove(member_index)

    for lc in fp_index.get(fkey, ()):
        if lc.n_shells != depth or lc.ego_graph is None:
            continue
        gm = isomorphism.GraphMatcher(
            ego, lc.ego_graph, node_match=_diffusion_lateral_node_match,
        )
        if gm.is_isomorphic():
            _drop_from_other_classes(new_lc=lc)
            if member_index not in lc.members:
                lc.members.append(member_index)
            _log.debug(
                "check_diffusion_site_lateral: diff_iso=%d member=%d "
                "→ existing lateral_class=%d",
                diffusion_site.iso_class, member_index, lc.lateral_class,
            )
            return lc

    _drop_from_other_classes(new_lc=None)
    new_lc = DiffusionLateral(
        lateral_class = len(diffusion_site.lateral_classes),
        ego_graph     = ego,
        n_shells      = depth,
        members       = [member_index],
    )
    new_lc._fingerprint = fkey  # type: ignore[attr-defined]
    diffusion_site.lateral_classes.append(new_lc)
    fp_index.setdefault(fkey, []).append(new_lc)

    _log.debug(
        "check_diffusion_site_lateral: diff_iso=%d member=%d "
        "→ new lateral_class=%d  (total=%d)",
        diffusion_site.iso_class, member_index,
        new_lc.lateral_class, len(diffusion_site.lateral_classes),
    )
    return new_lc


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
        raise ValueError(
            f"endpoint_position must be 'a' or 'b', got {endpoint_position!r}"
        )

    a_ordered = _ordered_endpoint_nodes(G, endpoint_a_ids)
    b_ordered = _ordered_endpoint_nodes(G, endpoint_b_ids)
    if len(a_ordered) != len(b_ordered):
        raise ValueError(
            "Diffusion endpoints have differing atom counts — cannot pair "
            "atoms across A and B for an NEB band."
        )

    endpoint_id_set: frozenset = frozenset(a_ordered) | frozenset(b_ordered)

    # ── 1. Slab atoms ───────────────────────────────────────────────────
    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True)
         if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )

    # ── 2. Lateral neighbours (excluding both endpoints) ────────────────
    lat_seed: set[int] = set()
    if lateral_class.ego_graph is not None:
        for n, d in lateral_class.ego_graph.nodes(data=True):
            if d.get("type") == "adsorbate" and n not in endpoint_id_set:
                lat_seed.add(int(n))
    lat_nodes: list[int] = sorted(_expand_to_full_placement(G, lat_seed))

    # ── 3. Migrating molecule ───────────────────────────────────────────
    chosen_nodes  = a_ordered if endpoint_position == "a" else b_ordered
    symbols_mig   = [G.nodes[n]["element"] for n in a_ordered]   # always A
    positions_mig = [
        np.asarray(G.nodes[n]["position"], dtype=float)
        for n in chosen_nodes
    ]

    # ── Assemble ────────────────────────────────────────────────────────
    slab_lat_nodes = slab_nodes + lat_nodes
    n_slab = len(slab_nodes)
    n_lat  = len(lat_nodes)
    n_mig  = len(symbols_mig)

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = full_pbc_for_cell(cell)

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
            positions_mig, dtype=float,
        )
        atoms.set_positions(positions)
        atoms.set_pbc(pbc)
    else:
        symbols   = [G.nodes[n]["element"] for n in slab_lat_nodes] + symbols_mig
        positions = [
            np.asarray(G.nodes[n]["position"], dtype=float) for n in slab_lat_nodes
        ] + positions_mig
        atoms = Atoms(
            symbols   = symbols,
            positions = np.asarray(positions, dtype=float),
            cell      = cell,
            pbc       = pbc,
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
) -> tuple[Atoms, float]:
    """Relax one endpoint and run the standard stability checks.

    Wraps every underlying :class:`SiteStabilityError` subclass into an
    :class:`EndpointStabilityError` with ``__cause__`` preserved.
    """
    from autokmc.structure import optimise_structure  # local: avoid cycle

    try:
        with acquire_calculator(
            calculator, purpose=f"diffusion {state_label} relaxation"
        ) as calc:
            atoms_opt = optimise_structure(
                atoms_init,
                calculator = calc,
                fmax       = fmax,
                steps      = max_steps,
                verbose    = verbose,
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
                    f"[{state_label}] LBFGS did not converge: "
                    f"max|F|={max_force:.4f} eV/Å after {max_steps} steps "
                    f"(fmax={fmax} eV/Å)."
                )

            energy = float(atoms_opt.get_potential_energy())
            atoms_opt.set_pbc(atoms_init.get_pbc())

            n_ads = n_lat + n_mig
            ads_indices = set(range(n_slab, n_slab + n_ads))
            _check_connectivity_stable(
                atoms_init, atoms_opt, n_slab, n_ads, state_label, nl_mult,
                relevant_indices=ads_indices,
                n_lat=n_lat,
            )
            _check_intended_coordination_stable(
                atoms_opt, G, self_node_ids,
                n_slab, n_lat, nl_mult,
                self_node_order=self_node_order,
            )

            if verbose:
                print(
                    f"  [{state_label}] E={energy:.4f} eV  "
                    f"max|F|={max_force:.4f} eV/Å  ✓ stable"
                )
            atoms_opt.calc = None
        return atoms_opt, energy

    except (SurfaceConnectivityError, AdsorbateDissociationError,
            OptimisationFailedError) as exc:
        raise EndpointStabilityError(
            f"Endpoint '{state_label}' relaxation failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# NEB band construction + TS validity
# ---------------------------------------------------------------------------

def _make_neb_band(
    atoms_a: Atoms,
    atoms_b: Atoms,
    *,
    n_images: int,
    interpolation: str,
    spring_k: float,
    climb: bool,
    calculator,
    frozen_indices: list[int] | None,
) -> tuple["NEB", list[Atoms]]:
    """Build an ASE NEB band of ``n_images + 2`` images between A and B.

    The same calculator instance is attached to every image and ASE's
    ``allow_shared_calculator`` path is enabled.  This is the current ASE
    equivalent of deprecated ``SingleCalculatorNEB`` and avoids requiring
    calculators to be deep-copyable.

    ``interpolation == "idpp"`` uses the IDPP interpolator (falling back
    to linear if the optional ASE module is unavailable); otherwise a
    straight-line cartesian interpolation is used.
    """
    images: list[Atoms] = [atoms_a.copy()]
    for _ in range(int(n_images)):
        images.append(atoms_a.copy())
    images.append(atoms_b.copy())

    # Atoms.copy() preserves constraints, but be defensive in case the
    # caller passed un-constrained endpoints and frozen_indices separately.
    if frozen_indices:
        for im in images:
            im.set_constraint(FixAtoms(indices=list(frozen_indices)))

    for im in images:
        im.calc = calculator

    neb = NEB(
        images,
        k=float(spring_k),
        climb=bool(climb),
        method="improvedtangent",
        allow_shared_calculator=True,
    )

    if interpolation == "idpp" and _idpp_interpolate is not None:
        try:
            _idpp_interpolate(neb, mic=True)
        except Exception as exc:
            # IDPP can fail for very short bands or pathological geometries;
            # fall back gracefully so a single bad pair doesn't kill the run.
            _log.warning(
                "IDPP interpolation failed (%s: %s); falling back to linear.",
                type(exc).__name__, exc,
            )
            neb.interpolate("linear", mic=True)
    else:
        neb.interpolate("linear", mic=True)

    return neb, images


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
    ts_index: int,
    n_interior: int,
    energy_tol: float = 1e-3,
) -> None:
    """Validate that the highest-energy NEB image is a real saddle.

    Detects three failure modes that otherwise propagate silently into the
    KMC rate:

    1. **Energy ordering** — ``E_ts < max(E_a, E_b) − energy_tol`` (eV) means
       the band is monotonic / reversed and there is no saddle.
    2. **Endpoint collapse** — the saddle is one of the boundary interior
       images (1 or n_interior) *and* its energy is within ``energy_tol``
       of the adjacent endpoint, i.e. the band trivially recovers an
       endpoint energy with no genuine barrier.
    3. **Migrating-molecule fragmentation** — the bond topology *within*
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
            f"TS / endpoint energies are not finite "
            f"(E_a={e_a}, E_b={e_b}, E_ts={e_ts})."
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
            e_ts, e_max_endpoint, energy_tol,
        )

    # 2. Endpoint collapse — TS sits at the band edge and matches its
    # adjacent endpoint within energy_tol.
    if n_interior >= 1:
        if ts_index == 1 and abs(float(e_ts) - float(e_a)) < float(energy_tol):
            raise TransitionStateInvalidError(
                f"TS image (k={ts_index}) collapsed onto endpoint A: "
                f"E_ts={e_ts:.4f} eV ≈ E_a={e_a:.4f} eV "
                f"(tol={energy_tol})."
            )
        if ts_index == n_interior and abs(float(e_ts) - float(e_b)) < float(energy_tol):
            raise TransitionStateInvalidError(
                f"TS image (k={ts_index}) collapsed onto endpoint B: "
                f"E_ts={e_ts:.4f} eV ≈ E_b={e_b:.4f} eV "
                f"(tol={energy_tol})."
            )

    # 3. Migrating-molecule connectivity.  Compute intra-mig bonds for
    # A, B and TS using the relevant_indices filter so only bonds involving
    # the migrating atoms are compared.  TS must match A *or* B.
    if n_mig >= 2:
        mig_indices = set(range(n_slab + n_lat, n_slab + n_lat + n_mig))
        bonds_a  = _bond_set(atoms_a,  nl_mult=nl_mult, relevant_indices=mig_indices)
        bonds_b  = _bond_set(atoms_b,  nl_mult=nl_mult, relevant_indices=mig_indices)
        bonds_ts = _bond_set(atoms_ts, nl_mult=nl_mult, relevant_indices=mig_indices)
        # Restrict the comparison to *intra*-migrating bonds (both ends in
        # the mig block) so that surface↔mig bond rearrangement at the saddle
        # is not flagged as fragmentation.
        def _intra(bonds: set) -> set:
            return {
                b for b in bonds
                if all(int(i) in mig_indices for i in b)
            }
        bonds_a_in  = _intra(bonds_a)
        bonds_b_in  = _intra(bonds_b)
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
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    verbose: bool = False,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
    calculation_cache_root: str | None = None,
) -> tuple[float, float, float]:
    """Relax both endpoints and the NEB band; store and return energies.

    Pipeline
    --------
    1. Build endpoint-A Atoms (slab + lateral neighbours + migrating
       molecule at A's positions); relax with LBFGS via
       :func:`~autokmc.structure.optimise_structure`; run the standard
       connectivity / coordination stability checks.
    2. Same for endpoint-B (migrating molecule at B's positions, atom
       ordering identical to A).
    3. Build an ``n_images``-image CI-NEB band between the two relaxed
       endpoints (IDPP or linear interpolation, configurable spring
       constant).  All images share one acquired calculator via ASE's
       SingleCalculatorNEB-style path.
    4. Run :class:`~ase.optimize.BFGS` on the NEB to ``fmax``.
    5. Identify the TS as the highest-energy interior image; validate
       (no fragmentation, no collapse onto an endpoint); store all
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
        Use climbing-image NEB.
    spring_k : float
        NEB spring constant (eV/Å²).
    interpolation : str
        ``"linear"`` (default) or ``"idpp"``.
    nl_mult : float
        Cutoff multiplier for the connectivity stability checks.
    persist_neb_path : bool
        Store the full relaxed band on
        ``lateral_class.atoms_neb_path``.  Default ``False``.
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
        TS image fragmented or collapsed onto an endpoint.
    """
    if member_index < 0 or member_index >= len(diffusion_site.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — DiffusionSite "
            f"iso_class={diffusion_site.iso_class} has "
            f"{len(diffusion_site.member_node_ids)} member(s)."
        )

    site_a, m_a, site_b, m_b = diffusion_site.members[member_index]
    a_node_ids, b_node_ids   = diffusion_site.member_node_ids[member_index]

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
    cache_kind = "diffusion"
    cache_key: str | None = None
    cache_graph: nx.Graph | None = None
    cache_parameters = {
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "n_images": int(n_images),
        "climb": bool(climb),
        "spring_k": float(spring_k),
        "interpolation": str(interpolation),
        "nl_mult": float(nl_mult),
        "n_shells": int(lateral_class.n_shells),
        "persist_neb_path": bool(persist_neb_path),
        "free_energy_enabled": bool(
            free_energy_options is not None
            and getattr(free_energy_options, "enabled", False)
        ),
        "temperature_k": (
            None if free_energy_temperature_k is None
            else float(free_energy_temperature_k)
        ),
        "calculator": calculator_identity(calculator),
    }
    if free_energy_options is not None:
        cache_parameters["free_energy"] = {
            "vibration_displacement": float(free_energy_options.vibration_displacement),
            "vibration_nfree": int(free_energy_options.vibration_nfree),
            "include_ts_vibrations": bool(free_energy_options.include_ts_vibrations),
            "min_frequency_ev": float(free_energy_options.min_frequency_ev),
            "symmetry_tolerance": float(free_energy_options.symmetry_tolerance),
            "default_spin": float(free_energy_options.default_spin),
            "default_geometry": str(free_energy_options.default_geometry),
        }

    # ── 1. Endpoint A relaxation ────────────────────────────────────────
    atoms_a_init, n_slab, n_lat, mig_idx_a, mig_node_order_a = _build_diffusion_atoms(
        G, lateral_class, list(a_node_ids), list(b_node_ids),
        endpoint_position = "a",
        frozen_indices    = frozen_indices,
    )
    n_mig = len(mig_idx_a)

    if calculation_cache_root is not None:
        try:
            cache_graph = normalise_reaction_graph(lateral_class.ego_graph)
            cache_graph.graph["n_shells"] = int(lateral_class.n_shells)
            atoms_b_seed, _, _, _, _ = _build_diffusion_atoms(
                G, lateral_class, list(a_node_ids), list(b_node_ids),
                endpoint_position="b",
                frozen_indices=frozen_indices,
            )
            cache_identity = {
                "kind": cache_kind,
                "reactant_smiles": diffusion_site.reactant,
                "iso_class": int(diffusion_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
            }
            cache_key = calculation_cache_key(
                kind=cache_kind,
                identity=cache_identity,
                parameters=cache_parameters,
                inputs={
                    "state_a_initial": atoms_a_init,
                    "state_b_seed_initial": atoms_b_seed,
                },
            )
            cached = load_calculation_record(
                calculation_cache_root,
                cache_kind,
                cache_key,
                reaction_graph=cache_graph,
                operation=cache_identity,
                parameters=cache_parameters,
            )
            if cached is not None and apply_cached_states(
                lateral_class,
                cached,
                {
                    "state_a": ("energy_a", "atoms_a"),
                    "state_b": ("energy_b", "atoms_b"),
                    "transition": ("energy_ts", "atoms_ts"),
                },
            ):
                if verbose:
                    print(
                        f"  [cache] diffusion iso={diffusion_site.iso_class} "
                        f"lat={lateral_class.lateral_class}: loaded "
                        "endpoint/NEB calculation"
                    )
                return (
                    float(lateral_class.energy_a),
                    float(lateral_class.energy_b),
                    float(lateral_class.energy_ts),
                )
        except Exception as exc:
            _log.debug(
                "check_diffusion_stability: calculation cache lookup failed "
                "(diff_iso=%d lat=%d): %s",
                diffusion_site.iso_class,
                lateral_class.lateral_class,
                exc,
            )

    if verbose:
        print(
            f"  [endpoint A] atoms={len(atoms_a_init)}  "
            f"(slab={n_slab}, lat={n_lat}, mig={n_mig})"
        )

    atoms_a_opt, E_a = _relax_endpoint(
        atoms_a_init,
        calculator      = calculator,
        fmax            = fmax,
        max_steps       = max_steps,
        frozen_indices  = frozen_indices,
        nl_mult         = nl_mult,
        n_slab          = n_slab,
        n_lat           = n_lat,
        n_mig           = n_mig,
        G               = G,
        self_node_ids   = self_a,
        self_node_order = mig_node_order_a,
        state_label     = "endpoint_a",
        verbose         = verbose,
    )
    # Store A immediately so it survives any later exception.
    lateral_class.energy_a = E_a
    lateral_class.atoms_a  = atoms_a_opt

    # ── 2. Endpoint B relaxation ────────────────────────────────────────
    # Reuse A's relaxed slab + lateral-neighbour positions as B's starting
    # geometry: only the migrating-molecule block is overwritten with B's
    # coordinates.  Both endpoints then sit in the same surface basin, which
    # makes the linear / IDPP NEB interpolation between them well-posed.
    atoms_b_init, _, _, mig_idx_b, mig_node_order_b = _build_diffusion_atoms(
        G, lateral_class, list(a_node_ids), list(b_node_ids),
        endpoint_position = "b",
        frozen_indices    = frozen_indices,
        base_atoms        = atoms_a_opt,
    )

    if verbose:
        print(
            f"  [endpoint B] atoms={len(atoms_b_init)}  "
            f"(slab={n_slab}, lat={n_lat}, mig={n_mig})"
        )

    atoms_b_opt, E_b = _relax_endpoint(
        atoms_b_init,
        calculator      = calculator,
        fmax            = fmax,
        max_steps       = max_steps,
        frozen_indices  = frozen_indices,
        nl_mult         = nl_mult,
        n_slab          = n_slab,
        n_lat           = n_lat,
        n_mig           = n_mig,
        G               = G,
        self_node_ids   = self_b,
        self_node_order = mig_node_order_b,
        state_label     = "endpoint_b",
        verbose         = verbose,
    )
    # Store B immediately so it survives any later exception.
    lateral_class.energy_b = E_b
    lateral_class.atoms_b  = atoms_b_opt

    # ── 3-4. NEB band ───────────────────────────────────────────────────
    if verbose:
        print(
            f"  [NEB] images={int(n_images)}  climb={bool(climb)}  "
            f"fmax={float(fmax):.4f} eV/Å  max_steps={int(max_steps)}"
        )

    with acquire_calculator(calculator, purpose="diffusion NEB") as neb_calc:
        neb, images = _make_neb_band(
            atoms_a_opt, atoms_b_opt,
            n_images       = int(n_images),
            interpolation  = str(interpolation),
            spring_k       = float(spring_k),
            climb          = bool(climb),
            calculator     = neb_calc,
            frozen_indices = frozen_indices,
        )
        try:
            opt = BFGS(neb, logfile=_neb_optimizer_logfile(verbose))
            opt.run(fmax=float(fmax), steps=int(max_steps))

            if not opt.converged():
                raise NEBNotConvergedError(
                    f"CI-NEB did not converge: fmax={fmax} eV/Å not reached in "
                    f"{max_steps} steps."
                )

            # ── 5. Identify TS = highest-energy interior image; validate ────────
            energies = np.array([float(im.get_potential_energy()) for im in images])
            interior = energies[1:-1]
            if len(interior) == 0:
                raise NEBNotConvergedError(
                    "NEB band has no interior images (n_images=0); "
                    "cannot identify a TS."
                )
            k_ts = 1 + int(np.argmax(interior))
            E_ts = float(energies[k_ts])
            atoms_ts = images[k_ts].copy()
            atoms_ts.calc = None

            # Store TS and NEB path immediately — they will be available even if
            # _check_ts_validity raises so callers can read partial results from
            # lateral_class after catching the exception.
            lateral_class.energy_ts = E_ts
            lateral_class.atoms_ts  = atoms_ts
            if persist_neb_path:
                lateral_class.neb_path_energies = [
                    float(im.get_potential_energy()) for im in images
                ]
                lateral_class.atoms_neb_path = []
                for im in images:
                    snap = im.copy()
                    snap.calc = None
                    lateral_class.atoms_neb_path.append(snap)
        finally:
            for im in images:
                im.calc = None

    _check_ts_validity(
        atoms_ts, atoms_a_opt, atoms_b_opt,
        n_slab     = n_slab,
        n_lat      = n_lat,
        n_mig      = n_mig,
        nl_mult    = nl_mult,
        e_a        = E_a,
        e_b        = E_b,
        e_ts       = E_ts,
        ts_index   = k_ts,
        n_interior = len(interior),
    )

    # ── 6. Mark stable ────────────────────────────────────────────────────
    lateral_class.stable     = True
    if verbose:
        print(
            f"  [NEB] converged=True steps={opt.nsteps}  "
            f"E_ts={E_ts:.4f} eV  image={k_ts}/{len(interior)}  ✓ stable"
        )

    # ── 7. Optional harmonic thermochemistry on A / B / TS ──────────────
    # The migrating molecule occupies the tail of the per-image atom array
    # (slab | lateral_neighbours | migrating_block).  Vibrate only those
    # indices so frozen slab + lateral neighbours contribute zero.
    if (free_energy_options is not None
            and getattr(free_energy_options, "enabled", False)
            and free_energy_temperature_k is not None):
        from autokmc.thermo.free_energy import compute_harmonic_thermo
        from pathlib import Path as _Path

        vib_indices = list(range(n_slab + n_lat, n_slab + n_lat + n_mig))
        cache_dir_root = (
            _Path(vib_cache_root) if vib_cache_root is not None else None
        )
        per_lat_dir = (
            cache_dir_root / f"diff_{smiles_to_dirname(diffusion_site.reactant)}" /
            f"diff_iso{diffusion_site.iso_class}_lat{lateral_class.lateral_class}"
            if cache_dir_root is not None else None
        )

        def _harm(atoms, label, energy_ev, drop_imag):
            return compute_harmonic_thermo(
                atoms, vib_indices,
                energy_ev      = float(energy_ev),
                temperature_k  = float(free_energy_temperature_k),
                calculator     = calculator,
                options        = free_energy_options,
                cache_dir      = (str(per_lat_dir) if per_lat_dir is not None else None),
                label          = label,
                drop_imaginary = drop_imag,
            )

        a_thermo  = _harm(atoms_a_opt, "state_a", E_a,  True)
        b_thermo  = _harm(atoms_b_opt, "state_b", E_b,  True)
        if getattr(free_energy_options, "include_ts_vibrations", True):
            ts_thermo = _harm(atoms_ts, "ts", E_ts, True)
        else:
            ts_thermo = None

        if a_thermo is not None:
            lateral_class.g_correction_a   = a_thermo["g_corr_ev"]
            lateral_class.g_a              = a_thermo["g_total_ev"]
            lateral_class.zpe_a            = a_thermo["zpe_ev"]
            lateral_class.entropy_a        = a_thermo["entropy_ev_per_k"]
            lateral_class.frequencies_a_ev = a_thermo["frequencies_ev"]
            lateral_class.imaginary_a_ev   = a_thermo["imaginary_ev"]
        if b_thermo is not None:
            lateral_class.g_correction_b   = b_thermo["g_corr_ev"]
            lateral_class.g_b              = b_thermo["g_total_ev"]
            lateral_class.zpe_b            = b_thermo["zpe_ev"]
            lateral_class.entropy_b        = b_thermo["entropy_ev_per_k"]
            lateral_class.frequencies_b_ev = b_thermo["frequencies_ev"]
            lateral_class.imaginary_b_ev   = b_thermo["imaginary_ev"]
        if ts_thermo is not None:
            lateral_class.g_correction_ts   = ts_thermo["g_corr_ev"]
            lateral_class.g_ts              = ts_thermo["g_total_ev"]
            lateral_class.zpe_ts            = ts_thermo["zpe_ev"]
            lateral_class.entropy_ts        = ts_thermo["entropy_ev_per_k"]
            lateral_class.frequencies_ts_ev = ts_thermo["frequencies_ev"]
            lateral_class.imaginary_ts_ev   = ts_thermo["imaginary_ev"]
        elif a_thermo is not None and b_thermo is not None:
            # Endpoint-ZPE-only fallback: use the average correction of A/B
            # as the TS correction so detailed-balance ratios are preserved
            # without paying the cost of a dedicated TS vib analysis.
            avg_corr = 0.5 * (a_thermo["g_corr_ev"] + b_thermo["g_corr_ev"])
            lateral_class.g_correction_ts = float(avg_corr)
            lateral_class.g_ts            = float(E_ts) + float(avg_corr)

    _log.debug(
        "check_diffusion_stability: diff_iso=%d member=%d lat=%d  "
        "E_a=%.4f  E_b=%.4f  E_ts=%.4f eV  Ea_fwd=%.4f  Ea_rev=%.4f",
        diffusion_site.iso_class, member_index, lateral_class.lateral_class,
        E_a, E_b, E_ts, E_ts - E_a, E_ts - E_b,
    )
    if calculation_cache_root is not None and cache_key is not None:
        props_a = {
            name: getattr(lateral_class, name, None)
            for name in (
                "g_correction_a",
                "g_a",
                "zpe_a",
                "entropy_a",
                "frequencies_a_ev",
                "imaginary_a_ev",
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
            )
        }
        neb = None
        if getattr(lateral_class, "atoms_neb_path", None):
            neb = {
                "energies_ev": list(getattr(lateral_class, "neb_path_energies", []) or []),
                "path_atoms": list(getattr(lateral_class, "atoms_neb_path", []) or []),
            }
        record = make_calculation_record(
            kind=cache_kind,
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
                "reactant_smiles": diffusion_site.reactant,
                "iso_class": int(diffusion_site.iso_class),
                "lateral_class": int(lateral_class.lateral_class),
            },
            states={
                "state_a": state_payload(
                    atoms_a_opt, energy_ev=E_a, properties=props_a,
                ),
                "state_b": state_payload(
                    atoms_b_opt, energy_ev=E_b, properties=props_b,
                ),
                "transition": state_payload(
                    atoms_ts, energy_ev=E_ts, properties=props_ts,
                ),
            },
            reaction_graph=cache_graph,
            neb=neb,
        )
        write_calculation_record(
            calculation_cache_root, cache_kind, cache_key, record,
        )
    return E_a, E_b, E_ts
