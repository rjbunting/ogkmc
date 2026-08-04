"""
autokmc.sites.stability.bond
========================
On-the-fly lateral-interaction classifier and CI-NEB stability / barrier
calculator for materialised :class:`~autokmc.sites.bond.BondReactionSite`'s.

This is the bond-reaction analogue of :mod:`autokmc.sites.stability.diffusion`.
Each :class:`~autokmc.sites.bond.BondReactionSite` enumerates an
isomorphism class of unordered triples (placement_A, placement_B,
placement_C) for one reversible reaction ``A + B ⇌ C``.  Three things have
to be evaluated lazily, only when the KMC loop actually needs them:

1. The **lateral-interaction class** of a specific triple member — i.e.
   the pattern of occupied adsorbate neighbours surrounding the union of
   A's, B's and C's bonded surface cliques.  See
   :func:`check_bond_site_lateral`.

2. The **two endpoint relaxations** for that lateral class — one ML
   relaxation each for the *A occupied + B occupied + C empty* state and
   the *C occupied + A and B empty* state — so we can quote
   ``energy_ab`` and ``energy_c`` on the
   :class:`~autokmc.sites.bond.BondReactionLateral`.

3. The **CI-NEB transition-state energy** ``energy_ts`` between the two
   endpoints, with the standard connectivity / saddle-validity guards.

See :func:`check_bond_site_stability`.

Lateral ego-graph conventions (extends :mod:`autokmc.sites.stability.adsorption`)
---------------------------------------------------------------------------
* **BFS seed** — the *union* of A's, B's and C's bonded surface cliques.
* **BFS frontier** — only ``type == "surface"`` nodes are traversed; the
  three placements themselves are excluded from the visited surface set.
* **Occupied adsorbate leaves** — every occupied ``type == "adsorbate"``
  node adjacent to the BFS surface set, *excluding* the three placements.
* **Triple inclusion** — atoms of the three placements are added as
  labelled leaves with ``occupied=True`` and ``endpoint_role`` set to
  ``"a"``, ``"b"``, or ``"c"``.  When the parent template is symmetric
  (``smiles_a == smiles_b``), the ``"a"`` and ``"b"`` labels are
  collapsed to ``"ab"`` so the iso-match treats A ↔ B as interchangeable.

NEB conventions
---------------
The total atom count is conserved across ``A + B ⇌ C``.  Per-image atom
layout::

    [ slab | lat_neighbours | reacting_block ]

For the **AB endpoint** the reacting block holds A's atoms (ordered by
``reactant_index``) followed by B's atoms (ordered by ``reactant_index``)
at A's and B's graph positions respectively.  For the **C endpoint** the
reacting block holds C's atoms in an order chosen by the configured
same-element matching strategy.  The default ``auto`` mode tries the legacy
greedy order, a global Hungarian assignment using minimum-image distances,
reactant-index order when chemically valid, and bounded same-element swap
trials, then keeps the lowest-displacement pre-NEB path.  This pairing is
what enables ASE's NEB interpolators to draw a smooth A+B → C path.

Public API
----------
* :class:`BondStabilityError`           — base error.
* :class:`BondEndpointStabilityError`   — endpoint relaxation failed.
* :class:`BondNEBNotConvergedError`     — NEB band did not converge.
* :class:`BondTransitionStateInvalidError` — TS lost connectivity / collapsed
  onto an endpoint.
* (Re-exported) :class:`SurfaceConnectivityError`,
  :class:`AdsorbateDissociationError`,
  :class:`OptimisationFailedError`     — from
  :mod:`autokmc.sites.stability.adsorption`.
* :func:`check_bond_site_lateral`       — classify the lateral environment
  of one specific triple member.
* :func:`check_bond_site_stability`     — relax both endpoints and the NEB,
  store and return ``(E_ab, E_c, E_ts)``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.constraints import FixAtoms

from autokmc.io.calculators import acquire_calculator
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
from autokmc.core.pbc import (
    full_pbc_for_cell,
    minimum_image_vectors,
    unwrap_positions_about_reference,
)
from autokmc.sites.stability.adsorption import (
    SurfaceConnectivityError,
    AdsorbateDissociationError,
    OptimisationFailedError,
    _surface_bfs_shells,
    _expand_to_full_placement,
    _check_connectivity_stable,
    _check_intended_coordination_stable,
    _bond_set,
)
from autokmc.sites.stability.neb import (
    make_neb_band,
    neb_optimizer_logfile,
    project_neb_path,
    run_neb,
)
from autokmc.sites.bond import BondReactionSite, BondReactionLateral
from autokmc.sites.diffusion import _member_clique_union
from autokmc.core.constants import (
    LATERAL_SHELLS_DEFAULT,
    NL_MULT_DEFAULT,
    NEB_N_IMAGES,
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_CLIMB,
    NEB_SPRING_K,
    BOND_NEB_INTERPOLATION,
    BOND_ATOM_MATCHING,
    BOND_MATCHING_TRIALS,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# Compatibility aliases for callers that imported the former channel-local
# helpers.  The implementation now lives in ``stability.neb``.
_make_neb_band = make_neb_band
_neb_optimizer_logfile = neb_optimizer_logfile


# ---------------------------------------------------------------------------
# Errors  (mirror autokmc.sites.stability.diffusion)
# ---------------------------------------------------------------------------

class BondStabilityError(Exception):
    """Base class for all bond-reaction stability / NEB failures.

    Caught by :func:`autokmc.reactions.bond.get_applicable_bond_reactions` and
    converted into ``lc.stable = False`` so the offending lateral class is
    permanently excluded from future KMC steps.
    """


class BondEndpointStabilityError(BondStabilityError):
    """One of the two endpoint relaxations did not satisfy the stability check.

    Raised when the *A+B occupied / C empty* or *C occupied / A+B empty*
    relaxation diverged, lost surface connectivity, or had its molecular
    framework dissociate into unintended fragments.
    """


class BondNEBNotConvergedError(BondStabilityError):
    """The NEB band did not reach the requested ``fmax`` within ``max_steps``."""


class BondTransitionStateInvalidError(BondStabilityError):
    """The converged transition state is not a meaningful saddle.

    Raised when the highest NEB image either collapses back onto one of the
    relaxed endpoints (no barrier) or loses surface / adsorbate
    connectivity (numerical instability rather than a real saddle).
    """


# ---------------------------------------------------------------------------
# Bond-specific lateral predicates  (extends adsorption ones with endpoint_role)
# ---------------------------------------------------------------------------

def _bond_lateral_node_match(d1: dict, d2: dict) -> bool:
    """Lateral-iso predicate for the bond-reaction ego-graph.

    Like :func:`autokmc.sites.stability.diffusion._diffusion_lateral_node_match`
    but the ``endpoint_role`` distinguishes ``"a"`` / ``"b"`` / ``"c"``
    (or ``"ab"`` / ``"c"`` for symmetric templates).
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
        if d1.get("endpoint_role") != d2.get("endpoint_role"):
            return False
    return True


def _bond_lateral_fingerprint(g: nx.Graph) -> tuple:
    """Cheap pre-filter mirroring :func:`_bond_lateral_node_match`."""
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
# Lateral ego-graph (bond variant)
# ---------------------------------------------------------------------------

def _build_bond_lateral_ego_graph(
    G: nx.Graph,
    seed_clique_union: frozenset,
    n_shells: int,
    *,
    a_ids: frozenset,
    b_ids: frozenset,
    c_ids: frozenset,
    is_symmetric: bool,
    ignore_occupied_neighbours: bool = False,
) -> nx.Graph:
    """Build the lateral ego-graph for one bond-reaction triple member.

    Mirrors :func:`autokmc.sites.stability.diffusion._build_diffusion_lateral_ego_graph`
    but stamps three placements with ``endpoint_role`` labels (collapsed
    to ``"ab"`` when *is_symmetric*).
    """
    role_a = "ab" if is_symmetric else "a"
    role_b = "ab" if is_symmetric else "b"

    endpoint_lists = (
        (a_ids, role_a),
        (b_ids, role_b),
        (c_ids, "c"),
    )
    endpoint_ids: frozenset = frozenset(a_ids) | frozenset(b_ids) | frozenset(c_ids)

    visited_full = _surface_bfs_shells(G, seed_clique_union, n_shells)
    visited: set = set(visited_full) - endpoint_ids

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


# ---------------------------------------------------------------------------
# Public API — lateral classifier
# ---------------------------------------------------------------------------

def check_bond_site_lateral(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
    *,
    n_shells: int = LATERAL_SHELLS_DEFAULT,
    ignore_lateral: bool = False,
    _assign_member: bool = True,
) -> BondReactionLateral:
    """Classify the lateral-interaction environment of one triple member.

    Builds the bond-reaction lateral ego-graph (BFS seeded from
    ``clq_a ∪ clq_b ∪ clq_c`` with the three placements stamped on as
    labelled occupied leaves carrying ``endpoint_role``) and either
    appends *member_index* to a matching :class:`BondReactionLateral`
    already on *brs* or creates a new one.  Returns the matching (or new)
    lateral class.

    Raises
    ------
    IndexError
        *member_index* out of range.
    ValueError
        Any of the three placements has an empty bonded surface clique.
    """
    if member_index < 0 or member_index >= len(brs.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — BondReactionSite "
            f"iso_class={brs.iso_class} has "
            f"{len(brs.member_node_ids)} member(s)."
        )

    depth: int = LATERAL_SHELLS_DEFAULT if n_shells is None else int(n_shells)

    gas_product = bool(getattr(brs, "gas_product", False))
    site_a, m_a, site_b, m_b, site_c, m_c = brs.members[member_index]
    # Read node IDs live from the AdsorbateSite objects rather than the cached
    # copies in brs.member_node_ids.  _materialise_adsorbate_nodes rebuilds
    # AdsorbateSite.member_node_ids in-place with fresh graph-node ids whenever
    # it is called; the list(...) copies kept in brs.member_node_ids are never
    # updated and become stale after any re-materialisation.
    a_node_ids = list(site_a.member_node_ids[m_a])
    b_node_ids = list(site_b.member_node_ids[m_b])
    c_node_ids = (
        []
        if gas_product or site_c is None
        else list(site_c.member_node_ids[m_c])
    )

    clq_a = _member_clique_union(site_a, m_a)
    clq_b = _member_clique_union(site_b, m_b)
    clq_c = (
        frozenset()
        if gas_product or site_c is None
        else _member_clique_union(site_c, m_c)
    )
    if not clq_a or not clq_b or (not gas_product and not clq_c):
        raise ValueError(
            f"Member {member_index} of BondReactionSite "
            f"iso_class={brs.iso_class} has an empty bonded surface "
            f"clique on one of its three placements — cannot build "
            f"a lateral ego-graph."
        )

    seed   = clq_a | clq_b | clq_c
    a_ids  = frozenset(int(n) for n in a_node_ids if n in G)
    b_ids  = frozenset(int(n) for n in b_node_ids if n in G)
    c_ids  = frozenset(int(n) for n in c_node_ids if n in G)

    ego = _build_bond_lateral_ego_graph(
        G, frozenset(seed), depth,
        a_ids                       = a_ids,
        b_ids                       = b_ids,
        c_ids                       = c_ids,
        is_symmetric                = bool(brs.template.is_symmetric),
        ignore_occupied_neighbours  = ignore_lateral,
    )

    fkey = _bond_lateral_fingerprint(ego)

    fp_index: dict | None = getattr(brs, "_lateral_fp_index", None)
    if fp_index is None:
        fp_index = {}
        brs._lateral_fp_index = fp_index

    def _stamp_gas_product(target_lc: BondReactionLateral) -> BondReactionLateral:
        if gas_product:
            target_lc.gas_product = True
            gas_reactant = getattr(brs, "gas_reactant", None)
            target_lc.gas_pressure_bar = float(
                getattr(gas_reactant, "partial_pressure_bar", 0.0) or 0.0
            )
        return target_lc

    def _drop_from_other_classes(new_lc=None) -> None:
        if not _assign_member:
            return
        for other in brs.lateral_classes:
            if other is new_lc:
                continue
            if member_index in other.members:
                other.members.remove(member_index)

    for lc in fp_index.get(fkey, ()):
        if lc.n_shells != depth or lc.ego_graph is None:
            continue
        gm = isomorphism.GraphMatcher(
            ego, lc.ego_graph, node_match=_bond_lateral_node_match,
        )
        if gm.is_isomorphic():
            if _assign_member:
                _drop_from_other_classes(new_lc=lc)
                if member_index not in lc.members:
                    lc.members.append(member_index)
                lc._seed_only = False
            _log.debug(
                "check_bond_site_lateral: bond_iso=%d member=%d "
                "→ existing lateral_class=%d",
                brs.iso_class, member_index, lc.lateral_class,
            )
            return _stamp_gas_product(lc)

    _drop_from_other_classes(new_lc=None)
    new_lc = BondReactionLateral(
        lateral_class = len(brs.lateral_classes),
        ego_graph     = ego,
        n_shells      = depth,
        members       = [member_index] if _assign_member else [],
    )
    new_lc._seed_only = not _assign_member
    _stamp_gas_product(new_lc)
    new_lc._fingerprint = fkey
    brs.lateral_classes.append(new_lc)
    fp_index.setdefault(fkey, []).append(new_lc)

    _log.debug(
        "check_bond_site_lateral: bond_iso=%d member=%d "
        "→ new lateral_class=%d  (total=%d)",
        brs.iso_class, member_index,
        new_lc.lateral_class, len(brs.lateral_classes),
    )
    return new_lc


def get_bond_bare_lateral(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
    *,
    n_shells: int = LATERAL_SHELLS_DEFAULT,
) -> BondReactionLateral:
    """Find or create the bare class without reassigning the live member."""
    lateral_class = check_bond_site_lateral(
        G,
        brs,
        member_index,
        n_shells=n_shells,
        ignore_lateral=True,
        _assign_member=False,
    )
    if not lateral_class.members:
        lateral_class._seed_only = True
    return lateral_class


# ---------------------------------------------------------------------------
# Atoms-builder helpers (bond variant)
# ---------------------------------------------------------------------------

def _ordered_endpoint_nodes(G: nx.Graph, endpoint_ids) -> list[int]:
    """Order placement nodes by ``reactant_index`` (SMILES atom index)."""
    present = [int(n) for n in endpoint_ids if n in G]
    return sorted(
        present,
        key=lambda n: int(G.nodes[n].get("reactant_index", n)),
    )


@dataclass
class _MappingCandidate:
    """One atom-index correspondence candidate for a bond NEB endpoint."""

    method: str
    c_node_order: list[int]
    distances: list[float]
    total_distance: float
    total_distance_sq: float
    rms_distance: float
    max_distance: float
    score: float

    def to_dict(self, G: nx.Graph) -> dict:
        return {
            "method": self.method,
            "c_node_order": list(self.c_node_order),
            "c_reactant_indices": [
                int(G.nodes[n].get("reactant_index", -1))
                for n in self.c_node_order
            ],
            "distances_ang": list(self.distances),
            "total_distance_ang": float(self.total_distance),
            "total_distance_sq_ang2": float(self.total_distance_sq),
            "rms_distance_ang": float(self.rms_distance),
            "max_distance_ang": float(self.max_distance),
            "score": float(self.score),
        }


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small wrapper around scipy with a brute-force fallback for tiny groups."""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover - scipy is a declared dependency.
        n_rows, n_cols = cost.shape
        if n_rows != n_cols or n_rows > 8:
            raise
        best_perm = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(n_cols)):
            total = float(sum(cost[i, j] for i, j in enumerate(perm)))
            if total < best_cost:
                best_cost = total
                best_perm = perm
        if best_perm is None:
            raise ValueError("no finite assignment available")
        return np.arange(n_rows, dtype=int), np.asarray(best_perm, dtype=int)
    return linear_sum_assignment(cost)


def _validate_c_order_symbols(
    G: nx.Graph,
    ab_symbols: list[str],
    c_order: list[int],
    *,
    method: str,
) -> None:
    if len(c_order) != len(ab_symbols):
        raise ValueError(
            f"Bond NEB {method} pairing: |C|={len(c_order)} differs from "
            f"|A|+|B|={len(ab_symbols)}."
        )
    c_symbols = [str(G.nodes[n]["element"]) for n in c_order]
    if c_symbols != list(ab_symbols):
        raise ValueError(
            f"Bond NEB {method} pairing produced element pattern "
            f"{c_symbols!r}, expected {list(ab_symbols)!r}."
        )


def _mapping_candidate(
    G: nx.Graph,
    ab_symbols: list[str],
    ab_positions: list[np.ndarray],
    c_order: list[int],
    *,
    method: str,
    cell,
    pbc,
) -> _MappingCandidate:
    """Score a C-node order against the relaxed AB reacting-block positions."""
    c_order = [int(n) for n in c_order]
    _validate_c_order_symbols(G, ab_symbols, c_order, method=method)

    ab_pos = np.asarray(ab_positions, dtype=float)
    c_pos = np.asarray(
        [np.asarray(G.nodes[n]["position"], dtype=float) for n in c_order],
        dtype=float,
    )
    disp = minimum_image_vectors(c_pos - ab_pos, cell, pbc)
    distances_arr = np.linalg.norm(disp, axis=1)
    total_sq = float(np.sum(distances_arr ** 2))
    rms = float(np.sqrt(total_sq / max(1, len(distances_arr))))
    max_d = float(distances_arr.max()) if len(distances_arr) else 0.0
    total = float(distances_arr.sum())
    # Keep the global-distance objective dominant, but add a modest max-jump
    # term so auto mode can prefer smoother paths over one very long crossing.
    score = float(total_sq + 0.5 * max_d * max_d)
    return _MappingCandidate(
        method=method,
        c_node_order=c_order,
        distances=[float(x) for x in distances_arr],
        total_distance=total,
        total_distance_sq=total_sq,
        rms_distance=rms,
        max_distance=max_d,
        score=score,
    )


def _greedy_pair_c_to_ab(
    G: nx.Graph,
    ab_symbols: list[str],
    ab_positions: list[np.ndarray],
    c_nodes: list[int],
    *,
    cell=None,
    pbc=None,
) -> list[int]:
    """Reorder *c_nodes* so that c[k]'s element matches ab[k]'s and c[k]'s
    physical position is closest (per element class) to ab[k]'s.

    Greedy nearest-neighbour assignment per element.  Returns a
    permutation of *c_nodes* of the same length.

    Raises
    ------
    ValueError
        Element multisets of the AB and C reacting blocks differ — the
        bond-conservation invariant is broken (templates that violate
        this should never reach the NEB stage).
    """
    if len(c_nodes) != len(ab_symbols):
        raise ValueError(
            f"Bond NEB pairing: |C|={len(c_nodes)} differs from "
            f"|A|+|B|={len(ab_symbols)}."
        )

    cell = np.asarray(
        G.graph.get("cell", np.eye(3)) if cell is None else cell,
        dtype=float,
    )
    pbc = (
        full_pbc_for_cell(cell)
        if pbc is None
        else np.asarray(pbc, dtype=bool)
    )

    c_remaining_by_elem: dict[str, list[int]] = {}
    for nid in c_nodes:
        elem = G.nodes[nid]["element"]
        c_remaining_by_elem.setdefault(elem, []).append(int(nid))

    ordered: list[int] = []
    for k, (sym, pos) in enumerate(zip(ab_symbols, ab_positions)):
        candidates = c_remaining_by_elem.get(sym)
        if not candidates:
            raise ValueError(
                f"Bond NEB pairing: AB atom {k} has element {sym!r} but "
                f"no matching unassigned C atom remains.  Element "
                f"multisets must match across endpoints."
            )
        # Pick the closest remaining C atom of this element.
        best_idx = 0
        best_d2  = float("inf")
        for i, cnid in enumerate(candidates):
            cpos = np.asarray(G.nodes[cnid]["position"], dtype=float)
            displacement = minimum_image_vectors(
                cpos - np.asarray(pos, dtype=float),
                cell,
                pbc,
            )
            d2 = float(np.sum(displacement ** 2))
            if d2 < best_d2:
                best_d2  = d2
                best_idx = i
        ordered.append(candidates.pop(best_idx))
    return ordered


def _hungarian_pair_c_to_ab(
    G: nx.Graph,
    ab_symbols: list[str],
    ab_positions: list[np.ndarray],
    c_nodes: list[int],
    *,
    cell=None,
    pbc=None,
) -> list[int]:
    """Return the same-element assignment minimizing total MIC distance."""
    if len(c_nodes) != len(ab_symbols):
        raise ValueError(
            f"Bond NEB Hungarian pairing: |C|={len(c_nodes)} differs from "
            f"|A|+|B|={len(ab_symbols)}."
        )

    cell = np.asarray(G.graph.get("cell", np.eye(3)) if cell is None else cell, dtype=float)
    pbc = full_pbc_for_cell(cell) if pbc is None else np.asarray(pbc, dtype=bool)

    c_by_elem: dict[str, list[int]] = {}
    for nid in c_nodes:
        c_by_elem.setdefault(str(G.nodes[nid]["element"]), []).append(int(nid))

    ordered: list[int | None] = [None] * len(ab_symbols)
    ab_pos = np.asarray(ab_positions, dtype=float)
    for sym in sorted(set(ab_symbols)):
        ab_idx = [i for i, s in enumerate(ab_symbols) if s == sym]
        candidates = c_by_elem.get(sym, [])
        if len(candidates) != len(ab_idx):
            raise ValueError(
                f"Bond NEB Hungarian pairing: element {sym!r} count mismatch "
                f"between AB ({len(ab_idx)}) and C ({len(candidates)})."
            )
        c_pos = np.asarray(
            [np.asarray(G.nodes[n]["position"], dtype=float) for n in candidates],
            dtype=float,
        )
        disp = minimum_image_vectors(
            c_pos[None, :, :] - ab_pos[ab_idx, None, :],
            cell,
            pbc,
        )
        cost = np.sum(disp ** 2, axis=2)
        rows, cols = _linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            ordered[ab_idx[int(row)]] = candidates[int(col)]

    if any(n is None for n in ordered):
        raise ValueError("Bond NEB Hungarian pairing left atoms unassigned.")
    return [int(n) for n in ordered if n is not None]


def _candidate_orders_from_swaps(
    base_order: list[int],
    ab_symbols: list[str],
    *,
    limit: int,
) -> list[tuple[str, list[int]]]:
    """Generate bounded same-element swap trials around a base assignment."""
    out: list[tuple[str, list[int]]] = []
    if limit <= 0:
        return out
    for sym in sorted(set(ab_symbols)):
        idx = [i for i, s in enumerate(ab_symbols) if s == sym]
        for i, j in itertools.combinations(idx, 2):
            swapped = list(base_order)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            out.append((f"same_element_swap:{sym}:{i}-{j}", swapped))
            if len(out) >= limit:
                return out
    return out


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return row-vector rotation R minimizing ``||P @ R - Q||``."""
    if len(P) == 0:
        return np.eye(3)
    H = np.asarray(P, dtype=float).T @ np.asarray(Q, dtype=float)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _assign_indices_by_element(
    source_symbols: list[str],
    source_positions: np.ndarray,
    target_symbols: list[str],
    target_positions: np.ndarray,
) -> list[int]:
    """Order source indices so each target slot gets the nearest same element."""
    if sorted(source_symbols) != sorted(target_symbols):
        raise ValueError(
            "assignment requires matching element multisets: "
            f"source={source_symbols!r}, target={target_symbols!r}"
        )
    ordered: list[int | None] = [None] * len(target_symbols)
    source_symbols = [str(s) for s in source_symbols]
    target_symbols = [str(s) for s in target_symbols]
    for sym in sorted(set(target_symbols)):
        src_idx = [i for i, s in enumerate(source_symbols) if s == sym]
        tgt_idx = [i for i, s in enumerate(target_symbols) if s == sym]
        cost = np.sum(
            (
                source_positions[src_idx][None, :, :]
                - target_positions[tgt_idx][:, None, :]
            ) ** 2,
            axis=2,
        )
        rows, cols = _linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            ordered[tgt_idx[int(row)]] = src_idx[int(col)]
    if any(i is None for i in ordered):
        raise ValueError("element assignment left atoms unassigned")
    return [int(i) for i in ordered if i is not None]


def _align_gas_product_to_target(
    gas_symbols: list[str],
    gas_positions: np.ndarray,
    target_symbols: list[str],
    target_positions_centered: np.ndarray,
    *,
    max_iter: int = 3,
) -> tuple[np.ndarray, list[int], dict]:
    """Return gas positions assigned/oriented to best match target positions."""
    gas_positions = np.asarray(gas_positions, dtype=float)
    gas_centered = gas_positions - gas_positions.mean(axis=0)
    target_centered = np.asarray(target_positions_centered, dtype=float)

    order = _assign_indices_by_element(
        gas_symbols, gas_centered, target_symbols, target_centered,
    )
    rotated_all = gas_centered.copy()
    for _ in range(max(1, int(max_iter))):
        P = gas_centered[order]
        R = _kabsch_rotation(P, target_centered)
        rotated_all = gas_centered @ R
        new_order = _assign_indices_by_element(
            gas_symbols, rotated_all, target_symbols, target_centered,
        )
        if new_order == order:
            break
        order = new_order

    ordered = rotated_all[order]
    d = np.linalg.norm(ordered - target_centered, axis=1)
    diag = {
        "gas_atom_order": list(order),
        "gas_symbols_ordered": [gas_symbols[i] for i in order],
        "alignment_rms_ang": float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0,
        "alignment_max_ang": float(d.max()) if len(d) else 0.0,
    }
    return ordered, order, diag


def _select_c_to_ab_mapping(
    G: nx.Graph,
    ab_symbols: list[str],
    ab_positions: list[np.ndarray],
    c_nodes: list[int],
    *,
    atom_matching: str,
    matching_trials: int,
) -> tuple[list[int], dict]:
    """Select the smoothest C->AB atom correspondence before endpoint C NEB.

    ``auto`` gathers a small set of chemically legal same-element mappings:
    reactant-index order (when valid), the legacy greedy result, the global
    Hungarian result, and bounded same-element swap trials around the
    Hungarian mapping.  The candidate with the lowest pre-NEB displacement
    score is kept.
    """
    method = str(atom_matching or BOND_ATOM_MATCHING).strip().lower()
    if method == "nearest":
        method = "greedy"
    if method not in {"auto", "hungarian", "greedy", "reactant_index", "symmetry_trials"}:
        raise ValueError(
            "bond atom_matching must be one of 'auto', 'hungarian', "
            "'greedy', 'reactant_index', or 'symmetry_trials', got "
            f"{atom_matching!r}."
        )

    c_present = [int(n) for n in c_nodes if n in G]
    if len(c_present) != len(ab_symbols):
        raise ValueError(
            f"Bond NEB pairing: |C|={len(c_present)} differs from "
            f"|A|+|B|={len(ab_symbols)}."
        )
    if sorted(str(G.nodes[n]["element"]) for n in c_present) != sorted(ab_symbols):
        raise ValueError(
            "Bond NEB pairing: element multisets must match across endpoints."
        )

    cell = np.asarray(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc = full_pbc_for_cell(cell)
    raw_orders: list[tuple[str, list[int]]] = []

    if method in {"auto", "reactant_index"}:
        try:
            _validate_c_order_symbols(G, ab_symbols, c_present, method="reactant_index")
            raw_orders.append(("reactant_index", list(c_present)))
        except ValueError:
            if method == "reactant_index":
                raise

    if method in {"auto", "greedy", "symmetry_trials"}:
        raw_orders.append((
            "greedy",
            _greedy_pair_c_to_ab(
                G,
                ab_symbols,
                ab_positions,
                c_present,
                cell=cell,
                pbc=pbc,
            ),
        ))

    if method in {"auto", "hungarian", "symmetry_trials"}:
        hungarian = _hungarian_pair_c_to_ab(
            G, ab_symbols, ab_positions, c_present, cell=cell, pbc=pbc,
        )
        raw_orders.append(("hungarian", hungarian))
        trial_budget = max(0, int(matching_trials) - len(raw_orders))
        if method in {"auto", "symmetry_trials"} and trial_budget > 0:
            raw_orders.extend(
                _candidate_orders_from_swaps(
                    hungarian, ab_symbols, limit=trial_budget,
                )
            )

    # Deduplicate orders while preserving the method labels that produced them.
    seen: set[tuple[int, ...]] = set()
    candidates: list[_MappingCandidate] = []
    for name, order in raw_orders:
        key = tuple(int(n) for n in order)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidates.append(_mapping_candidate(
                G,
                ab_symbols,
                ab_positions,
                list(order),
                method=name,
                cell=cell,
                pbc=pbc,
            ))
        except ValueError:
            if method not in {"auto", "symmetry_trials"}:
                raise
            continue

    if not candidates:
        raise ValueError("Bond NEB pairing: no valid atom-mapping candidates.")

    if method == "hungarian":
        selected = next(c for c in candidates if c.method == "hungarian")
    elif method == "greedy":
        selected = next(c for c in candidates if c.method == "greedy")
    elif method == "reactant_index":
        selected = next(c for c in candidates if c.method == "reactant_index")
    else:
        selected = min(
            candidates,
            key=lambda c: (c.score, c.total_distance_sq, c.max_distance),
        )

    diagnostics = {
        "requested_method": method,
        "selected_method": selected.method,
        "n_candidates": len(candidates),
        "selected": selected.to_dict(G),
        "candidates": [c.to_dict(G) for c in sorted(
            candidates,
            key=lambda c: (c.score, c.total_distance_sq, c.max_distance),
        )],
    }
    return list(selected.c_node_order), diagnostics



def _build_bond_atoms(
    G: nx.Graph,
    lc: BondReactionLateral,
    a_node_ids: list[int],
    b_node_ids: list[int],
    c_node_ids: list[int],
    *,
    endpoint: str,                 # "ab" or "c"
    frozen_indices: list[int] | None = None,
    base_atoms: Atoms | None = None,
    c_node_order: list[int] | None = None,
) -> tuple[Atoms, int, int, list[int], list[int], list[int]]:
    """Build a single-endpoint Atoms object for a bond-reaction calculation.

    Atom layout::

        [ slab | lat_neighbours | reacting_block ]

    where ``reacting_block`` has length ``n_a + n_b == n_c``.

    For ``endpoint == "ab"`` the reacting block is laid out as
    ``[A_atoms_by_reactant_index | B_atoms_by_reactant_index]`` at the
    A and B graph positions.

    For ``endpoint == "c"`` the reacting block uses *c_node_order* (a
    permutation of C's nodes obtained from :func:`_greedy_pair_c_to_ab`)
    so atom k matches atom k of the AB endpoint by element + nearest
    initial position.  The element symbols still come from the AB
    side (which by construction equals C's element multiset in the
    matched order) — this guarantees identical chemical_symbols across
    endpoints, a hard requirement for ASE NEB.

    Returns
    -------
    atoms, n_slab, n_lat, react_atom_indices, react_node_ids, react_node_ids_ab
        ``react_node_ids`` are the graph nodes physically present at this
        endpoint (A∪B's nodes in the AB layout, or C's permuted nodes in
        the C layout) — used by :func:`_check_intended_coordination_stable`
        to look up per-atom intended cliques.
        ``react_node_ids_ab`` is always the AB-side ordering (A then B)
        and is provided as a convenience handle for the caller.
    """
    if endpoint not in ("ab", "c"):
        raise ValueError(f"endpoint must be 'ab' or 'c', got {endpoint!r}")

    a_ordered = _ordered_endpoint_nodes(G, a_node_ids)
    b_ordered = _ordered_endpoint_nodes(G, b_node_ids)
    c_present = [int(n) for n in c_node_ids if n in G]

    if c_present and len(a_ordered) + len(b_ordered) != len(c_present):
        raise ValueError(
            f"Bond NEB layout: |A|+|B|={len(a_ordered)+len(b_ordered)} "
            f"differs from |C|={len(c_present)} — bond change must "
            f"conserve total atom count."
        )

    react_node_ids_ab = a_ordered + b_ordered                # canonical AB order
    symbols_react     = [G.nodes[n]["element"] for n in react_node_ids_ab]

    if endpoint == "ab":
        react_node_ids = list(react_node_ids_ab)
    else:
        if c_node_order is None:
            raise ValueError(
                "endpoint='c' requires a precomputed c_node_order from "
                "_greedy_pair_c_to_ab so the C reacting block lines up "
                "atom-for-atom with the AB reacting block."
            )
        react_node_ids = list(c_node_order)
        # Sanity check: matched element pattern must equal AB's.
        c_syms = [G.nodes[n]["element"] for n in react_node_ids]
        if c_syms != symbols_react:
            raise ValueError(
                "Bond NEB layout: greedy C pairing produced a different "
                "element pattern than the AB block — pairing is broken."
            )

    positions_react = [
        np.asarray(G.nodes[n]["position"], dtype=float) for n in react_node_ids
    ]

    endpoint_id_set: frozenset = (
        frozenset(a_ordered) | frozenset(b_ordered) | frozenset(c_present)
    )

    # ── 1. Slab atoms ───────────────────────────────────────────────────
    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True)
         if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )

    # ── 2. Lateral neighbours (excluding all three placements) ──────────
    lat_seed: set[int] = set()
    if lc.ego_graph is not None:
        for n, d in lc.ego_graph.nodes(data=True):
            if d.get("type") == "adsorbate" and n not in endpoint_id_set:
                lat_seed.add(int(n))
    lat_nodes: list[int] = sorted(_expand_to_full_placement(G, lat_seed))

    # ── Assemble ────────────────────────────────────────────────────────
    slab_lat_nodes = slab_nodes + lat_nodes
    n_slab  = len(slab_nodes)
    n_lat   = len(lat_nodes)
    n_react = len(symbols_react)

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = full_pbc_for_cell(cell)

    if base_atoms is not None:
        if len(base_atoms) != n_slab + n_lat + n_react:
            raise ValueError(
                "base_atoms has wrong length for the slab+lat+react layout: "
                f"got {len(base_atoms)}, expected {n_slab + n_lat + n_react}."
            )
        atoms = base_atoms.copy()
        positions = atoms.get_positions()
        positions[n_slab + n_lat : n_slab + n_lat + n_react] = np.asarray(
            positions_react, dtype=float,
        )
        atoms.set_positions(positions)
        # Replace symbols in the reacting block (chemical identity may
        # differ from base_atoms's reacting block when crossing endpoints
        # — though in this implementation the element pattern is the
        # same by construction, set_chemical_symbols is cheap and safe).
        all_syms = list(atoms.get_chemical_symbols())
        all_syms[n_slab + n_lat : n_slab + n_lat + n_react] = symbols_react
        atoms.set_chemical_symbols(all_syms)
        atoms.set_pbc(pbc)
    else:
        symbols   = [G.nodes[n]["element"] for n in slab_lat_nodes] + symbols_react
        positions = [
            np.asarray(G.nodes[n]["position"], dtype=float) for n in slab_lat_nodes
        ] + positions_react
        atoms = Atoms(
            symbols   = symbols,
            positions = np.asarray(positions, dtype=float),
            cell      = cell,
            pbc       = pbc,
        )

    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))

    react_atom_indices = list(range(n_slab + n_lat, n_slab + n_lat + n_react))
    return (atoms, n_slab, n_lat, react_atom_indices,
            react_node_ids, react_node_ids_ab)


def _gas_product_neb_endpoint(
    *,
    atoms_empty: Atoms,
    atoms_ab: Atoms,
    n_slab: int,
    n_lat: int,
    n_react: int,
    react_nodes_ab: list[int],
    gas_reactant,
    G: nx.Graph,
    lift_height: float,
) -> tuple[Atoms, dict]:
    """Return a same-size NEB endpoint with aligned C(gas) lifted above A+B."""
    if gas_reactant is None or getattr(gas_reactant, "atoms", None) is None:
        raise ValueError("gas-product bond reaction requires a gas Reactant for C")

    gas_atoms = gas_reactant.atoms
    gas_symbols = list(gas_atoms.get_chemical_symbols())
    target_symbols = [G.nodes[n]["element"] for n in react_nodes_ab]
    if sorted(gas_symbols) != sorted(target_symbols):
        raise ValueError(
            "gas-product bond reaction is not atom-conserving: "
            f"C(gas) symbols={gas_symbols} do not match A+B symbols={target_symbols}"
        )

    ab_positions = np.asarray(atoms_ab.get_positions(), dtype=float)
    react_slice = slice(n_slab + n_lat, n_slab + n_lat + n_react)
    target_positions = unwrap_positions_about_reference(
        ab_positions[react_slice],
        atoms_ab.cell.array,
        atoms_ab.pbc,
    )
    centroid = target_positions.mean(axis=0)
    target_centered = target_positions - centroid

    gas_pos = np.asarray(gas_atoms.get_positions(), dtype=float)
    gas_aligned_centered, gas_order, align_diag = _align_gas_product_to_target(
        gas_symbols,
        gas_pos,
        target_symbols,
        target_centered,
    )

    requested_lift = float(lift_height)
    min_lift = min(requested_lift, max(4.0, 0.75 * requested_lift))
    lift_candidates = sorted(
        {requested_lift, 0.875 * requested_lift, min_lift},
        reverse=True,
    )
    lift_scores: list[dict] = []
    best_payload: tuple[float, np.ndarray, dict] | None = None
    for h in lift_candidates:
        lifted_center = centroid + np.array([0.0, 0.0, float(h)])
        lifted_positions = np.asarray(
            [p + lifted_center for p in gas_aligned_centered],
            dtype=float,
        )
        d = np.linalg.norm(lifted_positions - target_positions, axis=1)
        score = float(np.sum(d ** 2) + 0.5 * float(d.max()) ** 2)
        payload = {
            "lift_height_ang": float(h),
            "rms_distance_ang": float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0,
            "max_distance_ang": float(d.max()) if len(d) else 0.0,
            "score": score,
        }
        lift_scores.append(payload)
        if best_payload is None or score < best_payload[2]["score"]:
            best_payload = (float(h), lifted_positions, payload)

    assert best_payload is not None
    selected_lift, lifted_positions, selected_lift_diag = best_payload

    atoms_c = atoms_ab.copy()
    pos = atoms_c.get_positions()
    pos[: n_slab + n_lat] = atoms_empty.get_positions()
    pos[react_slice] = lifted_positions
    atoms_c.set_positions(pos)
    symbols = list(atoms_c.get_chemical_symbols())
    symbols[react_slice] = target_symbols
    atoms_c.set_chemical_symbols(symbols)

    diagnostics = {
        "requested_method": "gas_product_kabsch",
        "selected_method": "gas_product_kabsch",
        "gas_atom_order": list(gas_order),
        "gas_reactant_indices": list(gas_order),
        "target_symbols": list(target_symbols),
        "requested_lift_height_ang": requested_lift,
        "selected_lift_height_ang": selected_lift,
        "alignment": align_diag,
        "selected": selected_lift_diag,
        "lift_candidates": lift_scores,
    }
    return atoms_c, diagnostics


# ---------------------------------------------------------------------------
# Endpoint relaxation helper (bond variant — supports two self-groups)
# ---------------------------------------------------------------------------

def _relax_bond_endpoint(
    atoms_init: Atoms,
    *,
    calculator,
    fmax: float,
    max_steps: int,
    optimizer: str,
    frozen_indices: list[int] | None,
    nl_mult: float,
    n_slab: int,
    n_lat: int,
    n_react: int,
    G: nx.Graph,
    self_groups: list[tuple[frozenset, list[int], int]],
    state_label: str,
    verbose: bool,
) -> tuple[Atoms, float]:
    """Relax one bond-reaction endpoint and run the standard stability checks.

    *self_groups* is a list of ``(self_node_ids, self_node_order,
    lat_offset)`` tuples — one per occupied placement at this endpoint
    (one entry for the C endpoint; two entries for the AB endpoint).
    ``lat_offset`` is the per-group ``n_lat`` value passed to
    :func:`_check_intended_coordination_stable` so that group's atoms
    sit at indices ``n_slab + lat_offset + j``.
    """
    from autokmc.structure import StructureOptimisationError, optimise_structure

    atoms_opt: Atoms | None = None
    try:
        with acquire_calculator(
            calculator, purpose=f"bond {state_label} relaxation"
        ) as calc:
            atoms_opt = optimise_structure(
                atoms_init,
                calculator = calc,
                fmax       = fmax,
                steps      = max_steps,
                optimizer  = optimizer,
                verbose    = verbose,
            )

            forces = atoms_opt.get_forces()
            if frozen_indices:
                free_mask: np.ndarray = np.ones(len(atoms_opt), dtype=bool)
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

            n_ads = n_lat + n_react
            ads_indices = set(range(n_slab, n_slab + n_ads))
            _check_connectivity_stable(
                atoms_init, atoms_opt, n_slab, n_ads, state_label, nl_mult,
                relevant_indices=ads_indices,
                n_lat=n_lat,
            )
            for self_ids, self_order, lat_offset in self_groups:
                _check_intended_coordination_stable(
                    atoms_opt, G, self_ids,
                    n_slab, lat_offset, nl_mult,
                    self_node_order=self_order,
                )

            if verbose:
                print(
                    f"  [{state_label}] E={energy:.4f} eV  "
                    f"max|F|={max_force:.4f} eV/Å  ✓ stable"
                )
            atoms_opt.calc = None
        return atoms_opt, energy

    except StructureOptimisationError as exc:
        wrapped = BondEndpointStabilityError(
            f"Endpoint '{state_label}' relaxation failed: {exc}"
        )
        wrapped.atoms = exc.atoms
        wrapped.state_label = state_label
        raise wrapped from exc
    except (SurfaceConnectivityError, AdsorbateDissociationError,
            OptimisationFailedError) as exc:
        wrapped = BondEndpointStabilityError(
            f"Endpoint '{state_label}' relaxation failed: {exc}"
        )
        if atoms_opt is not None:
            wrapped.atoms = atoms_opt.copy()
            wrapped.atoms.calc = None
        wrapped.state_label = state_label
        raise wrapped from exc


# ---------------------------------------------------------------------------
# NEB band construction + TS validity
# ---------------------------------------------------------------------------

def _check_bond_ts_validity(
    atoms_ts: Atoms,
    atoms_ab: Atoms,
    atoms_c: Atoms,
    *,
    n_slab: int,
    n_lat: int,
    n_react: int,
    nl_mult: float,
    e_ab: float,
    e_c: float,
    e_ts: float,
    ts_index: int,
    n_interior: int,
    energy_tol: float = 1e-3,
) -> None:
    """Validate that the highest-energy NEB image is a real saddle.

    Mirrors :func:`autokmc.sites.stability.diffusion._check_ts_validity` with
    one key relaxation: the AB and C endpoints **legitimately** differ by
    exactly one intra-reacting-block bond, so the TS bond topology is
    accepted whenever it matches *either* endpoint (i.e. the saddle has
    not split off into a third species).
    """
    if not (np.isfinite(e_ts) and np.isfinite(e_ab) and np.isfinite(e_c)):
        raise BondTransitionStateInvalidError(
            f"TS / endpoint energies are not finite "
            f"(E_ab={e_ab}, E_c={e_c}, E_ts={e_ts})."
        )
    e_max_endpoint = max(float(e_ab), float(e_c))
    if float(e_ts) < e_max_endpoint - float(energy_tol):
        _log.warning(
            "Bond NEB has no genuine saddle: E_ts=%.4f eV is below "
            "max(E_ab, E_c)=%.4f eV (tol=%.3f). "
            "The KMC barrier will be floored at EA_MIN.",
            e_ts, e_max_endpoint, energy_tol,
        )

    if n_interior >= 1:
        # Check energy proximity regardless of image index — a TS image at
        # position k=2 can still collapse to an endpoint energy if the NEB
        # is nearly flat near that end.  Restricting to ts_index == 1 or
        # ts_index == n_interior misses these interior-image collapses.
        # (Mirrors the BUG-9 fix applied to check_diffusion_sites._check_ts_validity.)
        if abs(float(e_ts) - float(e_ab)) < float(energy_tol):
            raise BondTransitionStateInvalidError(
                f"TS image (k={ts_index}) has energy indistinguishable from "
                f"endpoint AB: E_ts={e_ts:.4f} eV ≈ E_ab={e_ab:.4f} eV "
                f"(tol={energy_tol})."
            )
        if abs(float(e_ts) - float(e_c)) < float(energy_tol):
            raise BondTransitionStateInvalidError(
                f"TS image (k={ts_index}) has energy indistinguishable from "
                f"endpoint C: E_ts={e_ts:.4f} eV ≈ E_c={e_c:.4f} eV "
                f"(tol={energy_tol})."
            )

    # Intra-reacting-block bond topology — TS must match AB *or* C
    # (the bond change happens on exactly one side of the saddle).
    if n_react >= 2:
        react_indices = set(range(n_slab + n_lat, n_slab + n_lat + n_react))
        bonds_ab = _bond_set(atoms_ab, nl_mult=nl_mult, relevant_indices=react_indices)
        bonds_c  = _bond_set(atoms_c,  nl_mult=nl_mult, relevant_indices=react_indices)
        bonds_ts = _bond_set(atoms_ts, nl_mult=nl_mult, relevant_indices=react_indices)

        def _intra(bonds: set) -> set:
            return {
                b for b in bonds
                if all(int(i) in react_indices for i in b)
            }

        bonds_ab_in = _intra(bonds_ab)
        bonds_c_in  = _intra(bonds_c)
        bonds_ts_in = _intra(bonds_ts)
        if bonds_ts_in != bonds_ab_in and bonds_ts_in != bonds_c_in:
            raise BondTransitionStateInvalidError(
                "Reacting block fragmented (or its intramolecular bond "
                "topology is neither AB's nor C's at the TS): "
                f"bonds_ts={sorted(map(tuple, bonds_ts_in))} differ from "
                f"both bonds_AB={sorted(map(tuple, bonds_ab_in))} and "
                f"bonds_C={sorted(map(tuple, bonds_c_in))}."
            )


# ---------------------------------------------------------------------------
# Public API — stability / NEB
# ---------------------------------------------------------------------------

def _gas_product_cache_inputs(
    gas_reactant,
    *,
    include_thermochemistry: bool,
) -> dict:
    """Return every gas-species value consumed by the bond calculation.

    Partial pressure is deliberately absent: it scales the live reverse rate
    but does not alter the standard-state endpoint/NEB calculation.
    """
    inputs = {
        "gas_reactant_smiles": getattr(gas_reactant, "smiles", None),
        "gas_reactant_atoms": getattr(gas_reactant, "atoms", None),
        "gas_energy_ev": getattr(gas_reactant, "energy", None),
    }
    if include_thermochemistry:
        inputs.update(
            {
                "gas_gibbs_energy_ev": getattr(
                    gas_reactant, "gibbs_energy", None,
                ),
                "gas_zpe_ev": getattr(gas_reactant, "zpe", None),
                "gas_entropy_ev_per_k": getattr(
                    gas_reactant, "entropy", None,
                ),
                "gas_frequencies_ev": list(
                    getattr(gas_reactant, "frequencies_ev", []) or []
                ),
                "gas_imaginary_ev": list(
                    getattr(gas_reactant, "imaginary_ev", []) or []
                ),
            }
        )
    return inputs


def _stamp_gas_product_runtime_state(
    lc: BondReactionLateral,
    brs: BondReactionSite,
) -> None:
    """Restamp pressure-sensitive state after calculation or cache hydration."""
    gas_product = bool(getattr(brs, "gas_product", False))
    lc.gas_product = gas_product
    if gas_product:
        gas_reactant = getattr(brs, "gas_reactant", None)
        lc.gas_pressure_bar = float(
            getattr(gas_reactant, "partial_pressure_bar", 0.0) or 0.0
        )


def _apply_bond_thermochemistry(
    lc: BondReactionLateral,
    brs: BondReactionSite,
    *,
    atoms_ab: Atoms,
    atoms_c: Atoms,
    atoms_ts: Atoms,
    energy_ab: float,
    energy_c: float,
    energy_ts: float,
    n_slab: int,
    n_lateral: int,
    n_reacting: int,
    gas_product: bool,
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

    from autokmc.thermo.free_energy import compute_harmonic_thermo

    tpl = brs.template
    process = smiles_to_dirname(
        f"{tpl.smiles_a}+{tpl.smiles_b}~{tpl.smiles_c}"
    )
    cache_root = _Path(vib_cache_root) if vib_cache_root is not None else None
    per_lat_dir = (
        cache_root
        / f"bond_{process}"
        / f"bond_iso{brs.iso_class}_lat{lc.lateral_class}"
        if cache_root is not None
        else None
    )
    vib_indices = list(
        range(
            n_slab + n_lateral,
            n_slab + n_lateral + n_reacting,
        )
    )

    def _harm(atoms, label, energy_ev):
        return compute_harmonic_thermo(
            atoms,
            vib_indices,
            energy_ev=float(energy_ev),
            temperature_k=float(temperature_k),
            calculator=calculator,
            options=free_energy_options,
            cache_dir=str(per_lat_dir) if per_lat_dir is not None else None,
            label=label,
            drop_imaginary=True,
        )

    ab_thermo = _harm(atoms_ab, "state_ab", energy_ab)
    if gas_product:
        gas_reactant = brs.gas_reactant
        gas_g = float(getattr(gas_reactant, "gibbs_energy", float("nan")))
        gas_e = float(getattr(gas_reactant, "energy", float("nan")))
        if not np.isfinite(gas_g) or not np.isfinite(gas_e):
            raise ValueError(
                f"gas product {tpl.smiles_c!r} lacks finite free-energy data"
            )
        c_thermo = {
            "g_corr_ev": gas_g - gas_e,
            "g_total_ev": float(energy_c) + gas_g - gas_e,
            "zpe_ev": float(getattr(gas_reactant, "zpe", 0.0)),
            "entropy_ev_per_k": float(
                getattr(gas_reactant, "entropy", 0.0)
            ),
            "frequencies_ev": list(
                getattr(gas_reactant, "frequencies_ev", []) or []
            ),
            "imaginary_ev": list(
                getattr(gas_reactant, "imaginary_ev", []) or []
            ),
        }
    else:
        c_thermo = _harm(atoms_c, "state_c", energy_c)

    if getattr(free_energy_options, "include_ts_vibrations", True):
        ts_thermo = _harm(atoms_ts, "ts", energy_ts)
    else:
        average = 0.5 * (ab_thermo["g_corr_ev"] + c_thermo["g_corr_ev"])
        ts_thermo = {
            "g_corr_ev": float(average),
            "g_total_ev": float(energy_ts) + float(average),
            "zpe_ev": None,
            "entropy_ev_per_k": None,
            "frequencies_ev": [],
            "imaginary_ev": [],
        }

    for suffix, thermo in (
        ("ab", ab_thermo),
        ("c", c_thermo),
        ("ts", ts_thermo),
    ):
        setattr(lc, f"g_correction_{suffix}", thermo["g_corr_ev"])
        setattr(lc, f"g_{suffix}", thermo["g_total_ev"])
        setattr(lc, f"zpe_{suffix}", thermo["zpe_ev"])
        setattr(lc, f"entropy_{suffix}", thermo["entropy_ev_per_k"])
        setattr(
            lc,
            f"frequencies_{suffix}_ev",
            list(thermo["frequencies_ev"]),
        )
        setattr(
            lc,
            f"imaginary_{suffix}_ev",
            list(thermo["imaginary_ev"]),
        )
        setattr(lc, f"vib_indices_{suffix}", list(vib_indices))


def _write_bond_calculation_cache(
    calculation_cache_root: str,
    cache_key: str,
    cache_graph: nx.Graph,
    cache_parameters: dict[str, Any],
    cache_inputs: dict[str, Any],
    fingerprint_memo: CalculationFingerprintMemo,
    brs: BondReactionSite,
    lc: BondReactionLateral,
    *,
    atoms_ab: Atoms,
    atoms_c: Atoms,
    atoms_ts: Atoms,
    energy_ab: float,
    energy_c: float,
    energy_ts: float,
    gas_product: bool,
) -> None:
    tpl = getattr(brs, "template", None)
    props_ab = {
        name: getattr(lc, name, None)
        for name in (
            "g_correction_ab",
            "g_ab",
            "zpe_ab",
            "entropy_ab",
            "frequencies_ab_ev",
            "imaginary_ab_ev",
        )
    }
    props_c = {
        name: getattr(lc, name, None)
        for name in (
            "g_correction_c",
            "g_c",
            "zpe_c",
            "entropy_c",
            "frequencies_c_ev",
            "imaginary_c_ev",
        )
    }
    props_ts = {
        name: getattr(lc, name, None)
        for name in (
            "g_correction_ts",
            "g_ts",
            "zpe_ts",
            "entropy_ts",
            "frequencies_ts_ev",
            "imaginary_ts_ev",
        )
    }
    public_neb_path = getattr(lc, "atoms_neb_path", None)
    private_neb_path = getattr(lc, "_warm_start_neb_path", None)
    cache_neb_path = public_neb_path or private_neb_path
    neb = None
    if cache_neb_path:
        cache_neb_energies = (
            getattr(lc, "neb_path_energies", None)
            if public_neb_path
            else getattr(lc, "_warm_start_neb_energies", None)
        )
        neb = {
            "energies_ev": list(cache_neb_energies or []),
            "path_atoms": list(cache_neb_path),
        }
    record = make_calculation_record(
        kind="bond",
        cache_key=cache_key,
        operation={
            "label": (
                f"bond:{getattr(tpl, 'smiles_a', '')}+"
                f"{getattr(tpl, 'smiles_b', '')}->"
                f"{getattr(tpl, 'smiles_c', '')}"
            ),
            "reaction": (
                f"{getattr(tpl, 'smiles_a', '')}* + "
                f"{getattr(tpl, 'smiles_b', '')}* -> "
                f"{getattr(tpl, 'smiles_c', '')}*"
            ),
            "smiles_a": getattr(tpl, "smiles_a", ""),
            "smiles_b": getattr(tpl, "smiles_b", ""),
            "smiles_c": getattr(tpl, "smiles_c", ""),
            "iso_class": int(brs.iso_class),
            "lateral_class": int(lc.lateral_class),
            "gas_product": bool(gas_product),
        },
        parameters=cache_parameters,
        inputs={
            **cache_inputs,
            "iso_class": int(brs.iso_class),
            "lateral_class": int(lc.lateral_class),
            "gas_product": bool(gas_product),
        },
        states={
            "state_ab": state_payload(
                atoms_ab,
                energy_ev=energy_ab,
                properties=props_ab,
            ),
            "state_c": state_payload(
                atoms_c,
                energy_ev=energy_c,
                properties=props_c,
            ),
            "transition": state_payload(
                atoms_ts,
                energy_ev=energy_ts,
                properties=props_ts,
            ),
        },
        reaction_graph=cache_graph,
        neb=neb,
        lateral_attributes={
            "atom_matching_method": getattr(
                lc,
                "atom_matching_method",
                None,
            ),
            "atom_mapping": list(getattr(lc, "atom_mapping", []) or []),
            "matching_diagnostics": dict(
                getattr(lc, "matching_diagnostics", {}) or {}
            ),
            "gas_product": getattr(lc, "gas_product", None),
            "neb_initialization": getattr(
                lc,
                "neb_initialization",
                None,
            ),
            "neb_seed_fingerprint": getattr(
                lc,
                "neb_seed_fingerprint",
                None,
            ),
        },
    )
    write_calculation_record(
        calculation_cache_root,
        "bond",
        cache_key,
        record,
        fingerprint_memo=fingerprint_memo,
    )


def check_bond_site_stability(
    G: nx.Graph,
    brs: BondReactionSite,
    member_index: int,
    lc: BondReactionLateral,
    calculator,
    *,
    frozen_indices: list[int] | None = None,
    fmax: float = NEB_FMAX,
    max_steps: int = NEB_MAX_STEPS,
    n_images: int = NEB_N_IMAGES,
    climb: bool = NEB_CLIMB,
    spring_k: float = NEB_SPRING_K,
    interpolation: str = BOND_NEB_INTERPOLATION,
    atom_matching: str = BOND_ATOM_MATCHING,
    matching_trials: int = BOND_MATCHING_TRIALS,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    optimizer: str = DEFAULT_OPTIMIZER,
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER,
    verbose: bool = False,
    calculation_cache_root: str | None = None,
    calculation_cache_lookup_enabled: bool = False,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    vib_cache_root: str | None = None,
    neb_seed_path: Sequence[Atoms] | None = None,
    neb_seed_member_index: int | None = None,
    capture_neb_path: bool = False,
) -> tuple[float, float, float]:
    """Relax both endpoints and the NEB band; return ``(E_ab, E_c, E_ts)``.

    Pipeline (mirrors :func:`autokmc.sites.stability.diffusion.check_diffusion_stability`):

    1. Build the AB endpoint (slab + lateral neighbours + A's atoms +
       B's atoms at their graph positions); relax with the configured optimizer; verify
       both A's and B's intended surface coordination survive.
    2. Pair C's atoms to the AB reacting block using the configured
       same-element matching strategy (``auto`` defaults to a Hungarian/MIC
       assignment plus bounded swap trials), then build the C endpoint with
       C's atoms overwriting the reacting-block positions inherited from the
       relaxed AB slab+lat.  Relax; verify C's intended surface coordination
       survives.
    3. Converge an ordinary NEB band of ``n_images`` interior images between
       the two relaxed endpoints with the requested *interpolation* and
       *spring_k*.  If *climb* is enabled, retain the same band and spring
       constant, enable its climbing image, and converge it again.  All images
       share one acquired calculator via ASE's SingleCalculatorNEB-style path.
    4. Identify the TS as the highest-energy interior image; validate
       (no fragmentation into a third species, no collapse onto an
       endpoint); store all energies / atoms / (optional) full band on
       *lc*.

    On success ``lc.stable`` is set to ``True`` and the energies / relaxed
    atoms are persisted on the lateral class.

    Raises
    ------
    IndexError
        *member_index* out of range.
    ValueError
        Any placement has an empty bonded clique, or the AB/C element
        multisets disagree (templates that violate atom-conservation
        should never reach this stage).
    BondEndpointStabilityError
        An endpoint relaxation broke connectivity or did not converge.
    BondNEBNotConvergedError
        NEB band did not reach *fmax* in *max_steps*.
    BondTransitionStateInvalidError
        TS image fragmented or collapsed onto an endpoint.
    """
    if member_index < 0 or member_index >= len(brs.member_node_ids):
        raise IndexError(
            f"member_index={member_index} out of range — BondReactionSite "
            f"iso_class={brs.iso_class} has "
            f"{len(brs.member_node_ids)} member(s)."
        )

    gas_product = bool(getattr(brs, "gas_product", False))
    site_a, m_a, site_b, m_b, site_c, m_c = brs.members[member_index]
    # Read node IDs live from the AdsorbateSite objects rather than the cached
    # copies in brs.member_node_ids.  _materialise_adsorbate_nodes rebuilds
    # AdsorbateSite.member_node_ids in-place with fresh graph-node ids whenever
    # it is called; the list(...) copies kept in brs.member_node_ids are never
    # updated and become stale after any re-materialisation.
    a_node_ids = list(site_a.member_node_ids[m_a])
    b_node_ids = list(site_b.member_node_ids[m_b])
    c_node_ids = (
        []
        if gas_product or site_c is None
        else list(site_c.member_node_ids[m_c])
    )

    clq_a = _member_clique_union(site_a, m_a)
    clq_b = _member_clique_union(site_b, m_b)
    clq_c = (
        frozenset()
        if gas_product or site_c is None
        else _member_clique_union(site_c, m_c)
    )
    if not clq_a or not clq_b or (not gas_product and not clq_c):
        raise ValueError(
            f"Member {member_index} of BondReactionSite "
            f"iso_class={brs.iso_class} has an empty bonded surface "
            f"clique on one of its three placements — cannot run NEB."
        )

    self_a = frozenset(int(n) for n in a_node_ids if n in G)
    self_b = frozenset(int(n) for n in b_node_ids if n in G)
    self_c = frozenset(int(n) for n in c_node_ids if n in G)
    cache_kind = "bond"
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
        else input_coordinate_frame_fingerprint(
            {"neb_seed_path": seed_images}
        )
    )
    try:
        seed_member_index = int(neb_seed_member_index)
    except (TypeError, ValueError, OverflowError):
        seed_member_index = None
    same_member_seed = (
        seed_images is not None
        and seed_member_index == int(member_index)
    )
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
        "neb_optimizer": str(neb_optimizer).strip().lower(),
        "n_images": int(n_images),
        "climb": bool(climb),
        "spring_k": float(spring_k),
        "interpolation": str(interpolation),
        "atom_matching": str(atom_matching),
        "matching_trials": int(matching_trials),
        "nl_mult": float(nl_mult),
        "n_shells": int(lc.n_shells),
        "persist_neb_path": bool(persist_neb_path),
        "capture_neb_path": bool(capture_neb_path),
        "neb_seed_policy": "auto_bare_transfer_v1",
        "gas_product": bool(gas_product),
        "gas_lift_height": float(getattr(brs, "gas_lift_height", 6.0)),
        "free_energy_enabled": bool(
            free_energy_options is not None
            and getattr(free_energy_options, "enabled", False)
        ),
        "temperature_k": (
            None
            if free_energy_temperature_k is None
            else float(free_energy_temperature_k)
        ),
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

    # ── 1. AB endpoint ──────────────────────────────────────────────────
    (atoms_ab_init, n_slab, n_lat, react_idx,
     react_nodes_ab, _) = _build_bond_atoms(
        G, lc, list(a_node_ids), list(b_node_ids), list(c_node_ids),
        endpoint        = "ab",
        frozen_indices  = frozen_indices,
    )
    lc.atoms_ab_initial = atoms_ab_init.copy()
    lc.atoms_ab_initial.calc = None
    n_react = len(react_idx)
    a_ordered = _ordered_endpoint_nodes(G, a_node_ids)
    b_ordered = _ordered_endpoint_nodes(G, b_node_ids)
    n_a = len(a_ordered)
    n_b = len(b_ordered)

    if calculation_cache_root is not None:
        try:
            cache_parameters["calculator"] = calculator_identity(calculator)
            cache_graph = normalise_reaction_graph(lc.ego_graph)
            cache_graph.graph["n_shells"] = int(lc.n_shells)
            tpl = getattr(brs, "template", None)
            cache_identity = {
                "kind": cache_kind,
                "smiles_a": getattr(tpl, "smiles_a", ""),
                "smiles_b": getattr(tpl, "smiles_b", ""),
                "smiles_c": getattr(tpl, "smiles_c", ""),
                "iso_class": int(brs.iso_class),
                "lateral_class": int(lc.lateral_class),
            }
            cache_inputs = {
                "state_ab_initial": atoms_ab_init,
                "a_node_ids": list(a_node_ids),
                "b_node_ids": list(b_node_ids),
                "c_node_ids": list(c_node_ids),
                "gas_product": bool(gas_product),
                "neb_seed": {
                    "mode": (
                        "bare_transfer"
                        if seed_fingerprint is not None
                        else "configured_interpolation"
                    ),
                    "projection_scope": (
                        seed_projection_scope
                        if seed_fingerprint is not None
                        else None
                    ),
                    "path_sha256": seed_fingerprint,
                },
            }
            if gas_product:
                cache_inputs.update(
                    _gas_product_cache_inputs(
                        getattr(brs, "gas_reactant", None),
                        include_thermochemistry=bool(
                            free_energy_options is not None
                            and getattr(free_energy_options, "enabled", False)
                            and free_energy_temperature_k is not None
                        ),
                    )
                )
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
                lc,
                cached,
                {
                    "state_ab": ("energy_ab", "atoms_ab"),
                    "state_c": ("energy_c", "atoms_c"),
                    "transition": ("energy_ts", "atoms_ts"),
                },
                include_properties=cached.get("_cache_match") != "electronic",
            ):
                if capture_neb_path:
                    cached_path = list(
                        getattr(lc, "atoms_neb_path", None) or []
                    )
                    if len(cached_path) != int(n_images) + 2:
                        lc.stable = None
                        raise ValueError(
                            "cached bare bond result has no compatible "
                            "optimized NEB path"
                        )
                    lc._warm_start_neb_path = [
                        image.copy() for image in cached_path
                    ]
                    for image in lc._warm_start_neb_path:
                        image.calc = None
                    lc._warm_start_neb_energies = list(
                        getattr(lc, "neb_path_energies", None) or []
                    )
                    lc._warm_start_member_index = int(member_index)
                    if not persist_neb_path:
                        lc.atoms_neb_path = None
                        lc.neb_path_energies = None
                electronic_only = cached.get("_cache_match") == "electronic"
                if electronic_only and thermochemistry_requested:
                    # ``apply_cached_states`` marks the electronic states
                    # stable.  Clear that marker until the requested
                    # thermochemistry has completed so every failure between
                    # cache hydration and vibration completion is retryable.
                    lc.stable = None
                _stamp_gas_product_runtime_state(lc, brs)
                if verbose:
                    print(
                        f"  [cache] bond_iso={brs.iso_class} "
                        f"lat={lc.lateral_class}: loaded endpoint/NEB "
                        "calculation"
                        + (
                            "; recomputing thermochemistry"
                            if electronic_only and thermochemistry_requested
                            else ""
                        )
                    )
                energies = (
                    float(lc.energy_ab),
                    float(lc.energy_c),
                    float(lc.energy_ts),
                )
                if not electronic_only or not thermochemistry_requested:
                    return energies
                electronic_cache_state = (
                    *energies,
                    lc.atoms_ab,
                    lc.atoms_c,
                    lc.atoms_ts,
                )
        except Exception as exc:
            _log.debug(
                "check_bond_site_stability: calculation cache lookup failed "
                "(bond_iso=%d lat=%d): %s",
                brs.iso_class,
                lc.lateral_class,
                exc,
            )

    if electronic_cache_state is not None:
        (
            E_ab,
            E_c,
            E_ts,
            atoms_ab_opt,
            atoms_c_opt,
            atoms_ts,
        ) = electronic_cache_state
        _apply_bond_thermochemistry(
            lc,
            brs,
            atoms_ab=atoms_ab_opt,
            atoms_c=atoms_c_opt,
            atoms_ts=atoms_ts,
            energy_ab=E_ab,
            energy_c=E_c,
            energy_ts=E_ts,
            n_slab=n_slab,
            n_lateral=n_lat,
            n_reacting=n_react,
            gas_product=gas_product,
            calculator=calculator,
            free_energy_options=free_energy_options,
            temperature_k=free_energy_temperature_k,
            vib_cache_root=vib_cache_root,
        )
        lc.stable = True
        assert calculation_cache_root is not None
        assert cache_key is not None
        assert cache_graph is not None
        _write_bond_calculation_cache(
            calculation_cache_root,
            cache_key,
            cache_graph,
            cache_parameters,
            cache_inputs,
            cache_fingerprint_memo,
            brs,
            lc,
            atoms_ab=atoms_ab_opt,
            atoms_c=atoms_c_opt,
            atoms_ts=atoms_ts,
            energy_ab=E_ab,
            energy_c=E_c,
            energy_ts=E_ts,
            gas_product=gas_product,
        )
        return E_ab, E_c, E_ts

    if verbose:
        print(
            f"  [endpoint AB] atoms={len(atoms_ab_init)}  "
            f"(slab={n_slab}, lat={n_lat}, react={n_react} = "
            f"A={n_a}+B={n_b})"
        )

    # AB has two disjoint occupied groups: A at offset n_lat, B at n_lat+n_a.
    self_groups_ab = [
        (self_a, a_ordered, n_lat),
        (self_b, b_ordered, n_lat + n_a),
    ]
    try:
        atoms_ab_opt, E_ab = _relax_bond_endpoint(
            atoms_ab_init,
            calculator      = calculator,
            fmax            = fmax,
            max_steps       = max_steps,
            optimizer       = optimizer,
            frozen_indices  = frozen_indices,
            nl_mult         = nl_mult,
            n_slab          = n_slab,
            n_lat           = n_lat,
            n_react         = n_react,
            G               = G,
            self_groups     = self_groups_ab,
            state_label     = "endpoint_ab",
            verbose         = verbose,
        )
    except BondEndpointStabilityError as exc:
        failed_atoms = getattr(exc, "atoms", None)
        if failed_atoms is not None:
            lc.atoms_ab = failed_atoms
        raise
    lc.energy_ab = E_ab
    lc.atoms_ab  = atoms_ab_opt

    # ── 2. C endpoint ───────────────────────────────────────────────────
    if gas_product:
        from autokmc.structure import StructureOptimisationError, optimise_structure

        gas_reactant = getattr(brs, "gas_reactant", None)
        gas_energy = getattr(gas_reactant, "energy", float("nan"))
        if not np.isfinite(float(gas_energy)):
            raise ValueError(
                f"Gas product {brs.template.smiles_c!r} has no finite "
                "gas-phase energy; cannot compute bond reaction energetics."
            )

        atoms_empty_init = atoms_ab_opt[: n_slab + n_lat].copy()
        lc.atoms_c_initial = atoms_empty_init.copy()
        lc.atoms_c_initial.calc = None
        if frozen_indices:
            atoms_empty_init.set_constraint(
                FixAtoms(indices=[i for i in frozen_indices if i < len(atoms_empty_init)])
            )
        try:
            with acquire_calculator(
                calculator, purpose="bond gas-product empty-slab relaxation"
            ) as calc:
                atoms_empty_opt = optimise_structure(
                    atoms_empty_init,
                    calculator=calc,
                    fmax=fmax,
                    steps=max_steps,
                    optimizer=optimizer,
                    verbose=verbose,
                )
                E_empty = float(atoms_empty_opt.get_potential_energy())
                atoms_empty_opt.set_pbc(atoms_empty_init.get_pbc())
                atoms_empty_opt.calc = None
        except StructureOptimisationError as exc:
            lc.atoms_c = exc.atoms
            wrapped = BondEndpointStabilityError(
                "Endpoint 'endpoint_c' empty-slab relaxation failed: "
                f"{exc}"
            )
            wrapped.atoms = exc.atoms
            wrapped.state_label = "endpoint_c"
            raise wrapped from exc
        E_c = E_empty + float(gas_energy)
        atoms_c_opt, gas_mapping_diag = _gas_product_neb_endpoint(
            atoms_empty=atoms_empty_opt,
            atoms_ab=atoms_ab_opt,
            n_slab=n_slab,
            n_lat=n_lat,
            n_react=n_react,
            react_nodes_ab=react_nodes_ab,
            gas_reactant=gas_reactant,
            G=G,
            lift_height=float(getattr(brs, "gas_lift_height", 6.0)),
        )
        # The gas-product C endpoint is constructed directly rather than
        # relaxed as one combined structure, so this is the actual
        # pre-NEB endpoint supplied to the band.
        lc.atoms_c_initial = atoms_c_opt.copy()
        lc.atoms_c_initial.calc = None
        lc.atom_matching_method = gas_mapping_diag["selected_method"]
        lc.atom_mapping = list(gas_mapping_diag.get("gas_atom_order", []))
        lc.matching_diagnostics = gas_mapping_diag
        _stamp_gas_product_runtime_state(lc, brs)
        if verbose:
            print(
                f"  [endpoint C(gas)] E_empty={E_empty:.4f} eV  "
                f"E_gas={float(gas_energy):.4f} eV  "
                f"E_c={E_c:.4f} eV  "
                f"NEB final molecule lifted "
                f"{gas_mapping_diag['selected_lift_height_ang']:.2f} Å"
            )
    else:
        # Pair C's atoms to the AB reacting block so atom k aligns across
        # endpoints for the NEB.  Use relaxed AB positions rather than the
        # unrelaxed graph positions: after AB relaxation A and B can move
        # substantially, and the correspondence should minimize the actual
        # endpoint displacement.
        ab_symbols       = [G.nodes[n]["element"] for n in react_nodes_ab]
        _relaxed_ab_pos  = atoms_ab_opt.get_positions()
        ab_positions     = [
            _relaxed_ab_pos[n_slab + n_lat + k] for k in range(n_react)
        ]
        c_present    = [int(n) for n in c_node_ids if n in G]
        c_node_order, mapping_diag = _select_c_to_ab_mapping(
            G,
            ab_symbols,
            ab_positions,
            c_present,
            atom_matching=atom_matching,
            matching_trials=matching_trials,
        )
        lc.atom_matching_method = mapping_diag["selected_method"]
        lc.atom_mapping = list(c_node_order)
        lc.matching_diagnostics = mapping_diag
        if verbose:
            sel = mapping_diag["selected"]
            print(
                f"  [matching] requested={mapping_diag['requested_method']}  "
                f"selected={mapping_diag['selected_method']}  "
                f"candidates={mapping_diag['n_candidates']}  "
                f"rms={sel['rms_distance_ang']:.3f} Å  "
                f"max={sel['max_distance_ang']:.3f} Å"
            )

        (atoms_c_init, _, _, _, react_nodes_c, _) = _build_bond_atoms(
            G, lc, list(a_node_ids), list(b_node_ids), list(c_node_ids),
            endpoint        = "c",
            frozen_indices  = frozen_indices,
            base_atoms      = atoms_ab_opt,
            c_node_order    = c_node_order,
        )
        lc.atoms_c_initial = atoms_c_init.copy()
        lc.atoms_c_initial.calc = None

        if verbose:
            print(
                f"  [endpoint C ] atoms={len(atoms_c_init)}  "
                f"(slab={n_slab}, lat={n_lat}, react={n_react})"
            )

        # C endpoint has a single occupied group (C's atoms in matched order).
        self_groups_c = [(self_c, react_nodes_c, n_lat)]
        try:
            atoms_c_opt, E_c = _relax_bond_endpoint(
                atoms_c_init,
                calculator      = calculator,
                fmax            = fmax,
                max_steps       = max_steps,
                optimizer       = optimizer,
                frozen_indices  = frozen_indices,
                nl_mult         = nl_mult,
                n_slab          = n_slab,
                n_lat           = n_lat,
                n_react         = n_react,
                G               = G,
                self_groups     = self_groups_c,
                state_label     = "endpoint_c",
                verbose         = verbose,
            )
        except BondEndpointStabilityError as exc:
            failed_atoms = getattr(exc, "atoms", None)
            if failed_atoms is not None:
                lc.atoms_c = failed_atoms
            raise
    lc.energy_c = E_c
    lc.atoms_c  = atoms_c_opt

    # ── 3-4. NEB band ───────────────────────────────────────────────────
    projected_seed_path = None
    if seed_images:
        projected_seed_path = project_neb_path(
            seed_images,
            atoms_ab_opt,
            atoms_c_opt,
            n_slab=n_slab,
            n_lateral=n_lat,
        )
    lc.neb_seed_fingerprint = seed_fingerprint
    lc.neb_initialization = (
        "bare_transfer"
        if projected_seed_path is not None
        else (
            "configured_interpolation_fallback"
            if seed_images
            else "configured_interpolation"
        )
    )
    if verbose:
        print(
            f"  [NEB] images={int(n_images)}  climb={bool(climb)}  "
            f"fmax={float(fmax):.4f} eV/Å  max_steps={int(max_steps)}"
        )
        if projected_seed_path is not None:
            print(
                "  [NEB] initialization=bare optimized path "
                f"({seed_projection_scope})"
            )
        elif seed_images:
            print(
                "  [NEB] bare path incompatible; using "
                f"{interpolation} interpolation"
            )

    neb_result = run_neb(
        atoms_ab_opt,
        atoms_c_opt,
        calculator=calculator,
        purpose="bond NEB",
        n_images=int(n_images),
        interpolation=str(interpolation),
        spring_k=float(spring_k),
        climb=bool(climb),
        frozen_indices=frozen_indices,
        fmax=float(fmax),
        max_steps=int(max_steps),
        optimizer=neb_optimizer,
        verbose=verbose,
        not_converged_error=BondNEBNotConvergedError,
        persist_path=persist_neb_path,
        # Always retain a temporary completed band so post-NEB failures can be
        # diagnosed.  Successful non-persistent runs clear it below.
        capture_path=True,
        initial_path=projected_seed_path,
        initial_path_callback=(
            lambda images: setattr(
                lc,
                "atoms_neb_path_initial",
                images,
            )
        ),
        failure_path_callback=(
            lambda images: setattr(
                lc,
                "atoms_neb_path",
                images,
            )
        ),
        band_factory=_make_neb_band,
        logfile_factory=_neb_optimizer_logfile,
    )
    E_ts = neb_result.energy_ts
    atoms_ts = neb_result.atoms_ts
    k_ts = neb_result.transition_index

    lc.energy_ts = E_ts
    lc.atoms_ts = atoms_ts
    # Keep the final path attached until all validation and thermochemistry
    # steps succeed.  Failed candidates are then self-contained diagnostics.
    lc.neb_path_energies = neb_result.path_energies
    lc.atoms_neb_path = neb_result.path_images
    if capture_neb_path and neb_result.path_images:
        lc._warm_start_neb_path = [
            image.copy() for image in neb_result.path_images
        ]
        for image in lc._warm_start_neb_path:
            image.calc = None
        lc._warm_start_neb_energies = list(
            neb_result.path_energies or []
        )
        lc._warm_start_member_index = int(member_index)

    _check_bond_ts_validity(
        atoms_ts, atoms_ab_opt, atoms_c_opt,
        n_slab     = n_slab,
        n_lat      = n_lat,
        n_react    = n_react,
        nl_mult    = nl_mult,
        e_ab       = E_ab,
        e_c        = E_c,
        e_ts       = E_ts,
        ts_index   = k_ts,
        n_interior = neb_result.n_interior,
    )

    _apply_bond_thermochemistry(
        lc,
        brs,
        atoms_ab=atoms_ab_opt,
        atoms_c=atoms_c_opt,
        atoms_ts=atoms_ts,
        energy_ab=E_ab,
        energy_c=E_c,
        energy_ts=E_ts,
        n_slab=n_slab,
        n_lateral=n_lat,
        n_reacting=n_react,
        gas_product=gas_product,
        calculator=calculator,
        free_energy_options=free_energy_options,
        temperature_k=free_energy_temperature_k,
        vib_cache_root=vib_cache_root,
    )

    if not persist_neb_path:
        lc.atoms_neb_path_initial = None
        lc.atoms_neb_path = None
        lc.neb_path_energies = None
    lc.stable = True
    if verbose:
        print(
            f"  [NEB] converged=True steps={neb_result.optimizer_steps}  "
            f"E_ts={E_ts:.4f} eV  "
            f"image={k_ts}/{neb_result.n_interior}  ✓ stable"
        )

    _log.debug(
        "check_bond_site_stability: bond_iso=%d member=%d lat=%d  "
        "E_ab=%.4f  E_c=%.4f  E_ts=%.4f eV  Ea_fwd=%.4f  Ea_rev=%.4f",
        brs.iso_class, member_index, lc.lateral_class,
        E_ab, E_c, E_ts, E_ts - E_ab, E_ts - E_c,
    )
    if (
        calculation_cache_root is not None
        and cache_key is not None
        and cache_graph is not None
    ):
        _write_bond_calculation_cache(
            calculation_cache_root,
            cache_key,
            cache_graph,
            cache_parameters,
            cache_inputs,
            cache_fingerprint_memo,
            brs,
            lc,
            atoms_ab=atoms_ab_opt,
            atoms_c=atoms_c_opt,
            atoms_ts=atoms_ts,
            energy_ab=E_ab,
            energy_c=E_c,
            energy_ts=E_ts,
            gas_product=gas_product,
        )
    return E_ab, E_c, E_ts


__all__ = [
    "BondStabilityError",
    "BondEndpointStabilityError",
    "BondNEBNotConvergedError",
    "BondTransitionStateInvalidError",
    # Re-exported for callers that want a single import surface.
    "SurfaceConnectivityError",
    "AdsorbateDissociationError",
    "OptimisationFailedError",
    "check_bond_site_lateral",
    "check_bond_site_stability",
    "get_bond_bare_lateral",
]
