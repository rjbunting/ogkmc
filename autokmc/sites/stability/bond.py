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
reacting block holds C's atoms in an order obtained by greedy
element-aware nearest-neighbour matching against the AB ordering, so that
each k-th atom in the AB reacting block corresponds physically to the
k-th atom in the C reacting block.  This pairing is what enables ASE's
NEB interpolators to draw a smooth A+B → C path.

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

import os
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from autokmc.io.calculators import acquire_calculator
from autokmc.sites.anchors import _effective_pbc
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
    NEB_INTERPOLATION,
)
from autokmc.utils.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)


# ASE NEB import shim (modern: ``ase.mep``; older: ``ase.neb``).
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
        brs._lateral_fp_index = fp_index   # type: ignore[attr-defined]

    def _stamp_gas_product(target_lc: BondReactionLateral) -> BondReactionLateral:
        if gas_product:
            target_lc.gas_product = True
            gas_reactant = getattr(brs, "gas_reactant", None)
            target_lc.gas_pressure_bar = float(
                getattr(gas_reactant, "partial_pressure_bar", 0.0) or 0.0
            )
        return target_lc

    def _drop_from_other_classes(new_lc=None) -> None:
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
            _drop_from_other_classes(new_lc=lc)
            if member_index not in lc.members:
                lc.members.append(member_index)
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
        members       = [member_index],
    )
    _stamp_gas_product(new_lc)
    new_lc._fingerprint = fkey  # type: ignore[attr-defined]
    brs.lateral_classes.append(new_lc)
    fp_index.setdefault(fkey, []).append(new_lc)

    _log.debug(
        "check_bond_site_lateral: bond_iso=%d member=%d "
        "→ new lateral_class=%d  (total=%d)",
        brs.iso_class, member_index,
        new_lc.lateral_class, len(brs.lateral_classes),
    )
    return new_lc


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


def _greedy_pair_c_to_ab(
    G: nx.Graph,
    ab_symbols: list[str],
    ab_positions: list[np.ndarray],
    c_nodes: list[int],
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
            d2 = float(np.sum((cpos - np.asarray(pos, dtype=float)) ** 2))
            if d2 < best_d2:
                best_d2  = d2
                best_idx = i
        ordered.append(candidates.pop(best_idx))
    return ordered


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
    pbc  = _effective_pbc(G, cell)

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
) -> Atoms:
    """Return a same-size NEB endpoint with C(gas) lifted above A+B."""
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

    gas_pos = np.asarray(gas_atoms.get_positions(), dtype=float)
    gas_centered = gas_pos - gas_pos.mean(axis=0)
    available: dict[str, list[int]] = {}
    for idx, sym in enumerate(gas_symbols):
        available.setdefault(sym, []).append(idx)

    ordered_gas_positions: list[np.ndarray] = []
    for sym in target_symbols:
        bucket = available.get(sym)
        if not bucket:
            raise ValueError(
                f"gas-product endpoint cannot match required symbol {sym!r}"
            )
        ordered_gas_positions.append(gas_centered[bucket.pop(0)])

    ab_positions = np.asarray(atoms_ab.get_positions(), dtype=float)
    react_slice = slice(n_slab + n_lat, n_slab + n_lat + n_react)
    centroid = ab_positions[react_slice].mean(axis=0)
    lifted_center = centroid + np.array([0.0, 0.0, float(lift_height)])
    lifted_positions = np.asarray(
        [p + lifted_center for p in ordered_gas_positions],
        dtype=float,
    )

    atoms_c = atoms_ab.copy()
    pos = atoms_c.get_positions()
    pos[: n_slab + n_lat] = atoms_empty.get_positions()
    pos[react_slice] = lifted_positions
    atoms_c.set_positions(pos)
    symbols = list(atoms_c.get_chemical_symbols())
    symbols[react_slice] = target_symbols
    atoms_c.set_chemical_symbols(symbols)
    return atoms_c


# ---------------------------------------------------------------------------
# Endpoint relaxation helper (bond variant — supports two self-groups)
# ---------------------------------------------------------------------------

def _relax_bond_endpoint(
    atoms_init: Atoms,
    *,
    calculator,
    fmax: float,
    max_steps: int,
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
    from autokmc.structure import optimise_structure

    try:
        with acquire_calculator(
            calculator, purpose=f"bond {state_label} relaxation"
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

    except (SurfaceConnectivityError, AdsorbateDissociationError,
            OptimisationFailedError) as exc:
        raise BondEndpointStabilityError(
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
    """
    images: list[Atoms] = [atoms_a.copy()]
    for _ in range(int(n_images)):
        images.append(atoms_a.copy())
    images.append(atoms_b.copy())

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
            _log.warning(
                "IDPP interpolation failed (%s: %s); falling back to linear.",
                type(exc).__name__, exc,
            )
            neb.interpolate("linear", mic=True)
    else:
        neb.interpolate("linear", mic=True)

    return neb, images


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
    interpolation: str = NEB_INTERPOLATION,
    nl_mult: float = NL_MULT_DEFAULT,
    persist_neb_path: bool = False,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Relax both endpoints and the CI-NEB band; return ``(E_ab, E_c, E_ts)``.

    Pipeline (mirrors :func:`autokmc.sites.stability.diffusion.check_diffusion_stability`):

    1. Build the AB endpoint (slab + lateral neighbours + A's atoms +
       B's atoms at their graph positions); relax with LBFGS; verify
       both A's and B's intended surface coordination survive.
    2. Greedily pair C's atoms to the AB reacting block by element +
       nearest-position, then build the C endpoint with C's atoms
       overwriting the reacting-block positions inherited from the
       relaxed AB slab+lat.  Relax; verify C's intended surface
       coordination survives.
    3. Run a CI-NEB band of ``n_images`` interior images between the
       two relaxed endpoints with the requested *interpolation* and
       *spring_k*.  All images share one acquired calculator via ASE's
       SingleCalculatorNEB-style path.
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

    # ── 1. AB endpoint ──────────────────────────────────────────────────
    (atoms_ab_init, n_slab, n_lat, react_idx,
     react_nodes_ab, _) = _build_bond_atoms(
        G, lc, list(a_node_ids), list(b_node_ids), list(c_node_ids),
        endpoint        = "ab",
        frozen_indices  = frozen_indices,
    )
    n_react = len(react_idx)
    a_ordered = _ordered_endpoint_nodes(G, a_node_ids)
    b_ordered = _ordered_endpoint_nodes(G, b_node_ids)
    n_a = len(a_ordered)
    n_b = len(b_ordered)

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
    atoms_ab_opt, E_ab = _relax_bond_endpoint(
        atoms_ab_init,
        calculator      = calculator,
        fmax            = fmax,
        max_steps       = max_steps,
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
    lc.energy_ab = E_ab
    lc.atoms_ab  = atoms_ab_opt

    # ── 2. C endpoint ───────────────────────────────────────────────────
    if gas_product:
        from autokmc.structure import optimise_structure

        gas_reactant = getattr(brs, "gas_reactant", None)
        gas_energy = getattr(gas_reactant, "energy", float("nan"))
        if not np.isfinite(float(gas_energy)):
            raise ValueError(
                f"Gas product {brs.template.smiles_c!r} has no finite "
                "gas-phase energy; cannot compute bond reaction energetics."
            )

        atoms_empty_init = atoms_ab_opt[: n_slab + n_lat].copy()
        if frozen_indices:
            atoms_empty_init.set_constraint(
                FixAtoms(indices=[i for i in frozen_indices if i < len(atoms_empty_init)])
            )
        with acquire_calculator(
            calculator, purpose="bond gas-product empty-slab relaxation"
        ) as calc:
            atoms_empty_opt = optimise_structure(
                atoms_empty_init,
                calculator=calc,
                fmax=fmax,
                steps=max_steps,
                verbose=verbose,
            )
            E_empty = float(atoms_empty_opt.get_potential_energy())
            atoms_empty_opt.set_pbc(atoms_empty_init.get_pbc())
            atoms_empty_opt.calc = None
        E_c = E_empty + float(gas_energy)
        atoms_c_opt = _gas_product_neb_endpoint(
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
        lc.gas_product = True
        lc.gas_pressure_bar = float(
            getattr(gas_reactant, "partial_pressure_bar", 0.0) or 0.0
        )
        if verbose:
            print(
                f"  [endpoint C(gas)] E_empty={E_empty:.4f} eV  "
                f"E_gas={float(gas_energy):.4f} eV  "
                f"E_c={E_c:.4f} eV  "
                f"NEB final molecule lifted {float(getattr(brs, 'gas_lift_height', 6.0)):.2f} Å"
            )
    else:
        # Greedily pair C's atoms to the AB reacting block (element + nearest
        # position) so atom k aligns across endpoints for the NEB.
        # BUG-B2 FIX: use the *relaxed* AB reacting-block positions from
        # atoms_ab_opt rather than the unrelaxed graph positions stored on G.
        # After AB relaxation A and B can move substantially from their initial
        # placements; matching C against relaxed positions gives a physically
        # meaningful atom correspondence and a smoother NEB initial path.
        ab_symbols       = [G.nodes[n]["element"] for n in react_nodes_ab]
        _relaxed_ab_pos  = atoms_ab_opt.get_positions()
        ab_positions     = [
            _relaxed_ab_pos[n_slab + n_lat + k] for k in range(n_react)
        ]
        c_present    = [int(n) for n in c_node_ids if n in G]
        c_node_order = _greedy_pair_c_to_ab(G, ab_symbols, ab_positions, c_present)

        (atoms_c_init, _, _, _, react_nodes_c, _) = _build_bond_atoms(
            G, lc, list(a_node_ids), list(b_node_ids), list(c_node_ids),
            endpoint        = "c",
            frozen_indices  = frozen_indices,
            base_atoms      = atoms_ab_opt,
            c_node_order    = c_node_order,
        )

        if verbose:
            print(
                f"  [endpoint C ] atoms={len(atoms_c_init)}  "
                f"(slab={n_slab}, lat={n_lat}, react={n_react})"
            )

        # C endpoint has a single occupied group (C's atoms in matched order).
        self_groups_c = [(self_c, react_nodes_c, n_lat)]
        atoms_c_opt, E_c = _relax_bond_endpoint(
            atoms_c_init,
            calculator      = calculator,
            fmax            = fmax,
            max_steps       = max_steps,
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
    lc.energy_c = E_c
    lc.atoms_c  = atoms_c_opt

    # ── 3-4. NEB band ───────────────────────────────────────────────────
    if verbose:
        print(
            f"  [NEB] images={int(n_images)}  climb={bool(climb)}  "
            f"fmax={float(fmax):.4f} eV/Å  max_steps={int(max_steps)}"
        )

    with acquire_calculator(calculator, purpose="bond NEB") as neb_calc:
        neb, images = _make_neb_band(
            atoms_ab_opt, atoms_c_opt,
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
                raise BondNEBNotConvergedError(
                    f"CI-NEB did not converge: fmax={fmax} eV/Å not reached in "
                    f"{max_steps} steps."
                )

            # ── 5. Identify TS = highest-energy interior image; validate ────────
            energies = np.array([float(im.get_potential_energy()) for im in images])
            interior = energies[1:-1]
            if len(interior) == 0:
                raise BondNEBNotConvergedError(
                    "NEB band has no interior images (n_images=0); "
                    "cannot identify a TS."
                )
            k_ts = 1 + int(np.argmax(interior))
            E_ts = float(energies[k_ts])
            atoms_ts = images[k_ts].copy()
            atoms_ts.calc = None

            lc.energy_ts = E_ts
            lc.atoms_ts  = atoms_ts
            if persist_neb_path:
                lc.neb_path_energies = [
                    float(im.get_potential_energy()) for im in images
                ]
                lc.atoms_neb_path = []
                for im in images:
                    snap = im.copy()
                    snap.calc = None
                    lc.atoms_neb_path.append(snap)
        finally:
            for im in images:
                im.calc = None

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
        n_interior = len(interior),
    )

    lc.stable = True
    if verbose:
        print(
            f"  [NEB] converged=True steps={opt.nsteps}  "
            f"E_ts={E_ts:.4f} eV  image={k_ts}/{len(interior)}  ✓ stable"
        )

    _log.debug(
        "check_bond_site_stability: bond_iso=%d member=%d lat=%d  "
        "E_ab=%.4f  E_c=%.4f  E_ts=%.4f eV  Ea_fwd=%.4f  Ea_rev=%.4f",
        brs.iso_class, member_index, lc.lateral_class,
        E_ab, E_c, E_ts, E_ts - E_ab, E_ts - E_c,
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
]
