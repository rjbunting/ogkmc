"""
autokmc.sites.adsorbate
============================
N-atom adsorbate site enumeration on a surface graph, built cleanly on top of
:mod:`autokmc.sites.anchors` and :mod:`autokmc.species.reactant`.

Where :func:`~autokmc.sites.anchors.find_anchor_sites` finds all single-atom
adsorption *anchor* sites for one element, this module finds all geometrically
feasible placements of an arbitrary :class:`~autokmc.species.reactant.Reactant`
(``N >= 1`` atoms).  A placement is stored as one surface clique per reactant
atom rather than a single node.

Single-atom reactants (``N == 1``) take a degenerate fast path: the lone
anchor atom is placed at every raw anchor-node position of its element and
iso-classes are deduplicated by ego-graph isomorphism around the bonded
clique.  The calculator-free rotational refinement is skipped, while the
potential-driven rigid translation and relaxed stability stages run as for
any other reactant.

Strategy
--------
1. **Pick anchors.**  Only atoms in ``reactant.anchor_atoms`` (convex-hull-
   exposed, sterically unhindered) may bond to a surface clique.  Every other
   atom floats; its position is reconstructed by rigid alignment of the
   gas-phase geometry.

2. **Choose shell depth from molecular reach.**  Default
   ``n_shells_anchor = max(1, ceil(reach / d_nn))`` where *reach* is the
   largest intramolecular anchor-pair distance and *d_nn* ≈ 2.5 Å is a
   typical metal nearest-neighbour distance.

3. **Backtracking chain placement.**  For every orbit-canonicalised non-empty
   subset of anchors:

   * The first anchor is placed at every raw anchor-node position of its
     element (i.e. all members across all iso-classes, seeds from
     :func:`~autokmc.sites.anchors.find_anchor_sites`).
   * Subsequent anchors are placed at any *raw* anchor-node position of their
     element whose MIC distance to every already-placed anchor matches the
     intramolecular distance within ``bond_tolerance``.

4. **Surface-connectivity guard.**  The bonded cliques of every placement must
   be mutually reachable through ``type=="surface"`` edges; placements whose
   cliques are too far apart (> ``max_pair_shells`` hops) are dropped.

5. **Reduce by isomorphism.**  Decorate the union-of-cliques substrate ego-
   subgraph with the complete reactant graph and every adsorbate--surface
   coordination edge, then group placements by labeled graph isomorphism.
   This preserves which surface clique belongs to each adsorbate atom while
   still folding symmetry-equivalent molecular orientations together.

6. **Auto-grow** ``n_shells_anchor`` if any placement reaches further than the
   iso-class ego could see (up to ``max_shell_retries`` extra passes).

7. **Materialise** one adsorbate-site node per reactant atom per member on *G*
   (``type="adsorbate"``, ``occupied=False``).

8. **Optional calculator-free rigid-body refinement.**
   :func:`optimise_adsorbate_site_positions` does a calculator-free L-BFGS-B
   refinement of each iso-class representative's 6 rigid-body DOF.

9. **Potential stability pruning.**  With ``prune_stable_only=True``, each
   representative is first optimized against the configured potential as an
   exact rigid body on a fixed slab.  Only after that stage converges is the
   ordinary atom-level relaxation run and its adsorption topology checked.

Storage
-------
Results are stored in ``G.graph["adsorbate_sites"][smiles]``: a flat list of
:class:`AdsorbateSite` objects.

Public API
----------
* :class:`AdsorbateSite`                    — one iso-class of molecule placements.
* :func:`find_adsorbate_sites`              — universal N-atom enumerator.
* :func:`optimise_adsorbate_site_positions` — calculator-free rigid refinement.
* :func:`push_member_positions_to_graph`    — write refined positions back to G.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.core.pbc import (
    full_pbc_for_cell,
    minimum_image_vectors,
    periodic_image_offsets,
    unwrap_positions_about_reference,
    wrap_positions_into_cell,
)
from autokmc.io.calculators import CalculatorConfigError, acquire_calculator
from autokmc.sites.anchors import (
    ANCHOR_K_MAX_BY_ELEMENT,
    find_anchor_sites,
    _build_ego_graph,
    _effective_pbc,
    _get_cell,
    _kabsch,
    _reserve_node_ids,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import (
    DEFAULT_OPTIMIZER,
    REGULAR_OPTIMIZERS,
    normalize_optimizer_kwargs,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuneable defaults — single source of truth in :mod:`autokmc.core.constants`.
# Re-exported here for backwards compatibility and readability of the
# function-default kwargs.  See ``constants.py`` for full docstrings.
# ---------------------------------------------------------------------------

from autokmc.core.constants import (
    BOND_TOLERANCE,
    NN_DISTANCE,
    MAX_PAIR_SHELLS,
    N_SHELLS_DEFAULT,
    CO_FACTOR,
    OPT_FACTOR,
    REPULSION_WEIGHT,
    REPULSION_CUTOFF,
    CONTACT_FACTOR,
    STANDOFF_FACTOR,
    N_ADSORBATE_RESTARTS as N_RESTARTS,
    NEIGHBORLIST_SKIN,
    NL_MULT_DEFAULT,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
    HULL_TOL,
    KABSCH_MAX_MAPPINGS,
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class AdsorbateSiteLateral:
    """One distinct lateral-interaction environment of an :class:`AdsorbateSite`.

    A lateral-interaction environment is the local surface-graph pattern
    around a specific member placement, **including** the pattern of occupied
    neighbouring adsorbate sites.  Two members share a lateral class when
    their local environments are graph-isomorphic (element-matched for surface
    nodes; element + iso_class + reactant matched for occupied adsorbate leaf
    nodes).

    Lateral classes are built on demand by
    :func:`autokmc.sites.stability.adsorption.check_adsorbate_site_lateral` — one
    call per occupied member — and are stored on the parent
    :class:`AdsorbateSite` under ``lateral_classes``.

    Attributes
    ----------
    lateral_class : int
        0-based index within the parent :class:`AdsorbateSite`'s
        ``lateral_classes`` list.
    ego_graph : nx.Graph | None
        The ``n_shells``-shell ego-subgraph of the surface atoms bonded to
        this placement, built by
        :func:`autokmc.sites.stability.adsorption._build_lateral_ego_graph`.
        Surface nodes carry an ``element`` attribute; occupied adsorbate leaf
        nodes additionally carry ``iso_class`` and ``reactant``.
    n_shells : int
        Shell depth used to build :attr:`ego_graph`.
    members : list[int]
        Indices into the parent :class:`AdsorbateSite`'s
        ``member_node_ids`` list — every member whose current local
        environment is isomorphic to this lateral class.
    """
    lateral_class : int
    ego_graph     : Any              = None
    n_shells      : int              = 0
    members       : list[int]        = field(default_factory=list)
    #: Energy (eV) of the slab + lateral neighbours + **this site occupied**,
    #: set by :func:`autokmc.sites.stability.adsorption.check_site_stability`.
    energy_occupied   : float | None = None
    #: Energy (eV) of the slab + lateral neighbours with **this site empty**,
    #: set by :func:`autokmc.sites.stability.adsorption.check_site_stability`.
    energy_unoccupied : float | None = None
    #: ``True`` when both the occupied and unoccupied relaxations passed the
    #: connectivity stability check (no bonds appeared or disappeared).
    #: ``None`` until :func:`autokmc.sites.stability.adsorption.check_site_stability`
    #: has been run successfully.
    stable            : bool | None  = None
    #: Relaxed ASE :class:`~ase.Atoms` snapshot of the **occupied** state used
    #: in :func:`autokmc.sites.stability.adsorption.check_site_stability` (i.e. the
    #: structure that produced ``energy_occupied``).  Persisted by
    #: :class:`autokmc.io.persistence.ReactionWriter` as ``occupied.extxyz``
    #: in the reaction's per-lateral-class folder.
    atoms_occupied    : Any          = None
    #: Relaxed ASE :class:`~ase.Atoms` snapshot of the **unoccupied** state.
    atoms_unoccupied  : Any          = None
    #: Pre-optimization structures supplied to the occupied/unoccupied
    #: relaxations. Persisted beside the relaxed structures for diagnostics.
    atoms_occupied_initial   : Any    = None
    atoms_unoccupied_initial : Any    = None
    # The free-energy module populates these vibrational fields.
    #: Gibbs/Helmholtz correction (eV) for the **occupied** state — added
    #: to ``energy_occupied`` to obtain the surface free energy at *T*.
    #: ``None`` when free-energy mode is disabled or vibrations failed.
    g_correction_occupied   : float | None = None
    g_correction_unoccupied : float | None = None
    #: Absolute G (eV) cached for the rate code: ``energy_* + g_correction_*``.
    g_occupied              : float | None = None
    g_unoccupied            : float | None = None
    #: ZPE / entropy (eV, eV/K).
    zpe_occupied            : float | None = None
    zpe_unoccupied          : float | None = None
    entropy_occupied        : float | None = None
    entropy_unoccupied      : float | None = None
    #: Vibrational mode energies (eV) — real + imaginary buckets.
    frequencies_occupied_ev   : list = field(default_factory=list)
    frequencies_unoccupied_ev : list = field(default_factory=list)
    imaginary_occupied_ev     : list = field(default_factory=list)
    imaginary_unoccupied_ev   : list = field(default_factory=list)
    #: Atom indices in ``atoms_occupied`` that were displaced (for audit).
    vib_indices_occupied      : list = field(default_factory=list)
    vib_indices_unoccupied    : list = field(default_factory=list)
    # Appended after the established init fields for checkpoint/positional
    # constructor compatibility.
    invalid_reason            : str | None = None
    # Runtime indexes/caches used by lateral classification and rate building.
    # They remain lazily attached so old checkpoints retain their exact state,
    # but are explicit to type checkers and readers.
    if TYPE_CHECKING:
        _fingerprint : tuple = field(init=False, repr=False, compare=False)
        _rate_cache : dict = field(init=False, repr=False, compare=False)


@dataclass
class AdsorbateSite:
    """One isomorphism class of N-atom molecule placements on a surface.

    The iso-class deduplication uses a labeled coordination graph containing
    the n-shell substrate ego-subgraph around the union of bonded cliques,
    the complete reactant graph, and every adsorbate--surface coordination
    edge.

    Attributes
    ----------
    reactant : str
        SMILES of the reactant; key into the surrounding ``Reactant``
        registry from which gas-phase energy / structure can be retrieved.
    n_atoms : int
        Number of atoms in the reactant (length of ``atom_cliques`` and
        ``positions``).
    atom_cliques : list[frozenset[int] | None]
        Surface clique each reactant atom binds to, in reactant atom-index
        order.  ``None`` means "not bonded to the surface in this placement"
        (e.g. the dangling end of a physisorbed diatomic).
    positions : np.ndarray, shape (n_atoms, 3)
        Cartesian positions for the iso-class **representative** adsorbate
        atoms (Å).  Member positions live on *G* via ``member_node_ids``;
        they are not duplicated here.
    iso_class : int
        0-based index within the per-reactant iso-class list.
    members : list[list[frozenset | None]]
        All raw ``atom_cliques`` tuples folded into this iso-class (the
        first entry is the representative).
    member_node_ids : list[list[int]]
        Per-member graph node ids for the materialised adsorbate-site nodes
        (``len == n_atoms`` each, in reactant atom-index order).  Use to
        look up an iso-class member directly on the graph::

            [G.nodes[n] for n in member_node_ids[k]]

    ego_graph : nx.Graph | None
        The labeled adsorption coordination graph used for iso-class
        reduction.  Its substrate portion is the ``n_shells_pair`` ego-
        subgraph around the union of bonded cliques; its adsorbate portion is
        the complete reactant graph joined to the assigned surface cliques.
        Live occupied neighbours are deliberately excluded and represented by
        :class:`AdsorbateSiteLateral`.
    """

    reactant        : str
    n_atoms         : int
    atom_cliques    : list[frozenset | None]
    positions       : Any
    iso_class       : int
    members         : list[list[frozenset | None]] = field(default_factory=list)
    member_node_ids : list[list[int]]              = field(default_factory=list)
    ego_graph       : Any                          = None
    #: Anchor-enumeration depth selected after any ``auto_grow_shells`` retries
    #: inside :func:`find_adsorbate_sites`. It is also used when projecting a
    #: relaxed representative back into the graph's Cartesian frame.
    n_shells_settled: int                          = 0
    #: Substrate-ego depth used by the decorated coordination graph that
    #: defined this iso-class. This is distinct from anchor-enumeration depth.
    coordination_n_shells: int                     = 0
    #: Lateral-interaction classes discovered so far for this iso-class
    #: (populated on demand by
    #: :func:`autokmc.sites.stability.adsorption.check_adsorbate_site_lateral`).
    lateral_classes : list[AdsorbateSiteLateral]   = field(default_factory=list)
    #: Stable KMC identity.  Empty values from older checkpoints are upgraded
    #: lazily by :func:`autokmc.sites.identity.site_identifier` after graph
    #: materialisation has supplied the member-node signature.
    site_id         : str                          = field(default="", compare=False)
    # Mutable runtime state populated by graph materialisation, lateral
    # classification, and reaction construction.  ``Any`` avoids importing
    # reaction models back into the site layer.
    # These remain lazily attached at runtime because ``hasattr`` is the
    # compatibility signal used by older checkpoints and manually-built site
    # objects.  Type checkers still see the fields explicitly, while the false
    # runtime branch keeps them out of dataclass serialisation.
    if TYPE_CHECKING:
        _member_cliques : list[tuple[frozenset, ...]] = field(
            init=False, repr=False, compare=False,
        )
        _n_occupied : int = field(init=False, repr=False, compare=False)
        _lateral_fp_index : dict[tuple, list[AdsorbateSiteLateral]] = field(
            init=False, repr=False, compare=False,
        )
        _member_lc : dict[int, AdsorbateSiteLateral] = field(
            init=False, repr=False, compare=False,
        )
        applicable_reactions : list[Any] = field(
            init=False, repr=False, compare=False,
        )

    # ------------------------------------------------------------------
    # Occupancy helpers
    # ------------------------------------------------------------------

    def _member_is_occupied(self, G: nx.Graph, node_ids: list[int]) -> bool:
        for nid in node_ids:
            if nid in G and G.nodes[nid].get("occupied", False):
                return True
        return False

    def occupied_member_indices(self, G: nx.Graph) -> list[int]:
        """Indices into :attr:`members` flagged ``occupied=True`` on *G*."""
        return [k for k, nids in enumerate(self.member_node_ids)
                if self._member_is_occupied(G, nids)]

    def unoccupied_member_indices(self, G: nx.Graph) -> list[int]:
        """Indices into :attr:`members` flagged ``occupied=False`` on *G*."""
        return [k for k, nids in enumerate(self.member_node_ids)
                if not self._member_is_occupied(G, nids)]

    def occupied_member_node_ids(self, G: nx.Graph) -> list[list[int]]:
        """Node-id groups for currently-occupied members."""
        return [self.member_node_ids[k]
                for k in self.occupied_member_indices(G)]

    def unoccupied_member_node_ids(self, G: nx.Graph) -> list[list[int]]:
        """Node-id groups for currently-unoccupied members."""
        return [self.member_node_ids[k]
                for k in self.unoccupied_member_indices(G)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _suggested_n_shells(reach: float, nn_distance: float = NN_DISTANCE) -> int:
    """Heuristic shell depth from molecular geometric reach."""
    return max(1, int(np.ceil(reach / nn_distance)))


def _mic_distance(
    p: np.ndarray,
    q: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
    use_mic: bool,
) -> float:
    """MIC distance between two Cartesian points."""
    dv = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    if use_mic and cell_inv is not None:
        dv = minimum_image_vectors(dv, cell, pbc)
    return float(np.linalg.norm(dv))


@dataclass(frozen=True)
class _AnchorCandidateSpatialIndex:
    tree: Any
    source: np.ndarray
    tiled_positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    cutoff: float


def _build_anchor_spatial_index(
    candidates: list[tuple[frozenset, np.ndarray]],
    cell: np.ndarray,
    pbc: np.ndarray,
    use_mic: bool,
    *,
    cutoff: float,
) -> _AnchorCandidateSpatialIndex | None:
    """Build a KD-tree over anchor candidates, including periodic images."""
    if not candidates:
        return None
    try:
        from scipy.spatial import cKDTree
    except Exception:  # pragma: no cover - SciPy is expected but optional here
        return None

    radius = float(cutoff)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("spatial-index cutoff must be finite and non-negative")

    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    pos = np.asarray([p for _, p in candidates], dtype=float)
    source = np.arange(len(candidates), dtype=int)
    tiled = pos

    if use_mic and pbc_arr.any():
        pos = wrap_positions_into_cell(pos, cell_arr, pbc_arr)
        shifts = periodic_image_offsets(cell_arr, pbc_arr, radius)
        translations = shifts.astype(float) @ cell_arr
        tiled = (pos[None, :, :] + translations[:, None, :]).reshape(-1, 3)
        source = np.tile(source, len(shifts))

    return _AnchorCandidateSpatialIndex(
        tree=cKDTree(tiled),
        source=source,
        tiled_positions=tiled,
        cell=cell_arr,
        pbc=pbc_arr,
        cutoff=radius,
    )


def _source_indices_within_radius(
    spatial_index,
    centers: np.ndarray,
    radius: float,
) -> list[int]:
    """Return source indexes with any tiled point inside *radius* of centers."""
    if spatial_index is None:
        return []
    radius_value = max(0.0, float(radius))
    if radius_value > spatial_index.cutoff + 1.0e-12:
        raise ValueError("query radius exceeds the spatial-index cutoff")
    centers_arr = np.asarray(centers, dtype=float).reshape(-1, 3)
    if spatial_index.pbc.any():
        centers_arr = wrap_positions_into_cell(
            centers_arr,
            spatial_index.cell,
            spatial_index.pbc,
        )
    seen: set[int] = set()
    for center in centers_arr:
        for h in spatial_index.tree.query_ball_point(center, radius_value):
            seen.add(int(spatial_index.source[h]))
    return sorted(seen)


def _candidate_indices_in_annulus(
    spatial_index,
    center: np.ndarray,
    target: float,
    tolerance: float,
) -> list[int]:
    """Return source candidate indexes within a Cartesian annulus."""
    if spatial_index is None:
        return []
    outer = max(0.0, float(target) + float(tolerance))
    if outer > spatial_index.cutoff + 1.0e-12:
        raise ValueError("annulus radius exceeds the spatial-index cutoff")
    center_arr = np.asarray(center, dtype=float)
    if spatial_index.pbc.any():
        center_arr = wrap_positions_into_cell(
            center_arr,
            spatial_index.cell,
            spatial_index.pbc,
        )
    hits = spatial_index.tree.query_ball_point(center_arr, outer)
    if not hits:
        return []

    lower = max(0.0, float(target) - float(tolerance))
    out: set[int] = set()
    for h in hits:
        if lower > 0.0:
            d = float(
                np.linalg.norm(spatial_index.tiled_positions[h] - center_arr)
            )
            if d < lower:
                continue
        out.add(int(spatial_index.source[h]))
    return sorted(out)


def _outward_normal_at(
    G: nx.Graph, p: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Unit outward direction at *p*: +z for slabs, radial-out for NPs."""
    if pbc.any():
        return np.array([0.0, 0.0, 1.0])
    surf_pos = np.array(
        [d["position"] for _, d in G.nodes(data=True)
         if d.get("type") == "surface"],
        dtype=float,
    )
    if not len(surf_pos):
        return np.array([0.0, 0.0, 1.0])
    n = p - surf_pos.mean(axis=0)
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])


def _full_adsorbate_positions(
    reactant,
    bonded_indices: list[int],
    bonded_positions: np.ndarray,
    G: nx.Graph,
    pbc: np.ndarray,
) -> np.ndarray:
    """Reconstruct Cartesian positions for every reactant atom.

    * Two or more bonded anchors → Kabsch rigid alignment.
    * One bonded anchor → translation + outward-normal orientation.
    """
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    n_atoms   = react_pos.shape[0]

    if len(bonded_indices) >= 2:
        cell = np.asarray(G.graph.get("cell", np.eye(3)), dtype=float)
        bonded_positions = unwrap_positions_about_reference(
            bonded_positions,
            cell,
            pbc,
        )
        R, t = _kabsch(react_pos[bonded_indices], bonded_positions)
        return react_pos @ R.T + t

    # Single bonded anchor: translate; rotate unbonded atoms outward.
    i0       = bonded_indices[0]
    p_target = bonded_positions[0]
    rel      = react_pos - react_pos[i0]
    n_out    = _outward_normal_at(G, p_target, pbc)

    if n_atoms > 1:
        other  = np.delete(np.arange(n_atoms), i0)
        c      = rel[other].mean(axis=0)
        norm_c = float(np.linalg.norm(c))

        def _rotate_from_to(v_from: np.ndarray, v_to: np.ndarray) -> None:
            nonlocal rel
            v_from = np.asarray(v_from, dtype=float)
            v_to = np.asarray(v_to, dtype=float)
            n_from = float(np.linalg.norm(v_from))
            n_to = float(np.linalg.norm(v_to))
            if n_from <= 1e-10 or n_to <= 1e-10:
                return
            v = v_from / n_from
            tgt = v_to / n_to
            axis = np.cross(v, tgt)
            s = float(np.linalg.norm(axis))
            cos = float(np.dot(v, tgt))
            if s > 1e-8:
                axis /= s
                K = np.array([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])
                rel = rel @ (np.eye(3) + s * K + (1.0 - cos) * (K @ K)).T
            elif cos < 0.0:
                rel = -rel

        # Planar one-anchor fragments such as CH3 have a near-zero centroid
        # of the non-bonded atoms.  Aligning that numerical noise to the
        # surface normal tilts two H atoms into the slab.  Instead, align the
        # molecular plane normal to the outward normal so the substituents stay
        # away from extra surface contacts.
        if len(other) >= 3 and norm_c <= 1e-3:
            _u, svals, vh = np.linalg.svd(rel[other], full_matrices=False)
            if svals[0] > 1e-12 and svals[-1] / svals[0] <= 1e-3:
                normal = np.asarray(vh[-1], dtype=float)
                if float(np.dot(normal, n_out)) < 0.0:
                    normal = -normal
                _rotate_from_to(normal, n_out)
                symbols = reactant.atoms.get_chemical_symbols()
                if (
                    symbols[int(i0)] != "H"
                    and all(symbols[int(j)] == "H" for j in other)
                ):
                    normal_fraction = 1.0 / 3.0
                    for j in other:
                        vec = np.asarray(rel[int(j)], dtype=float)
                        length = float(np.linalg.norm(vec))
                        if length <= 1e-10:
                            continue
                        normal_part = float(np.dot(vec, n_out)) * n_out
                        lateral = vec - normal_part
                        lat_norm = float(np.linalg.norm(lateral))
                        target_normal = normal_fraction * length
                        target_lateral = float(np.sqrt(max(
                            length * length - target_normal * target_normal,
                            0.0,
                        )))
                        if lat_norm > 1e-10:
                            rel[int(j)] = (
                                lateral / lat_norm * target_lateral
                                + n_out * target_normal
                            )
                        else:
                            rel[int(j)] = n_out * target_normal
            return rel + p_target

        if norm_c > 1e-8:
            v    = c / norm_c
            axis = np.cross(v, n_out)
            s    = float(np.linalg.norm(axis))
            cos  = float(np.dot(v, n_out))
            if s > 1e-8:
                axis /= s
                K = np.array([[0.0, -axis[2], axis[1]],
                              [axis[2], 0.0, -axis[0]],
                              [-axis[1], axis[0], 0.0]])
                rel = rel @ (np.eye(3) + s * K + (1.0 - cos) * (K @ K)).T
            elif cos < 0.0:
                rel = -rel
    return rel + p_target


def _adsorbate_pose_is_outward(
    G: nx.Graph,
    atom_cliques: list,
    positions: np.ndarray,
    pbc: np.ndarray,
    *,
    margin: float = 1e-8,
) -> bool:
    """Return False when a propagated pose points into the surface."""
    pos_arr = np.asarray(positions, dtype=float)
    is_slab = bool(np.asarray(pbc, dtype=bool).any())
    for atom_i, clq in enumerate(atom_cliques):
        if clq is None:
            continue
        rows = [
            np.asarray(G.nodes[int(s)]["position"], dtype=float)
            for s in clq if int(s) in G
        ]
        if not rows:
            continue
        if is_slab:
            z_ref = max(float(p[2]) for p in rows)
            if float(pos_arr[int(atom_i), 2] - z_ref) <= margin:
                return False
            continue
        centroid = np.asarray(rows, dtype=float).mean(axis=0)
        n_hat = _outward_normal_at(G, centroid, pbc)
        if float(np.dot(pos_arr[int(atom_i)] - centroid, n_hat)) <= margin:
            return False
    return True


# ---------------------------------------------------------------------------
# Reactant molecule helpers
# ---------------------------------------------------------------------------

def _orbit_id_of(reactant) -> dict[int, tuple[str, int]]:
    """Map each reactant atom index → ``(element, orbit_idx)``."""
    out: dict[int, tuple[str, int]] = {}
    for elem, orbits in reactant.unique_nodes.items():
        for k, orb in enumerate(orbits):
            for i in orb:
                out[int(i)] = (elem, k)
    return out


def _canonical_subset_key(
    subset: tuple[int, ...],
    orbit_id: dict[int, tuple[str, int]],
) -> tuple:
    """Orbit-multiset signature — equal keys → symmetry-equivalent subsets."""
    return tuple(sorted(orbit_id[i] for i in subset))


def _anchor_subset_allowed(
    subset: tuple[int, ...],
    orbit_id: dict[int, tuple[str, int]],
) -> bool:
    """Return whether an anchor subset is chemically useful to enumerate.

    Equivalent hydrogens represent alternative orientations of the same intact
    molecule, not distinct multi-dentate surface binding atoms.  Keeping at
    most one H from each automorphism orbit preserves the top-H methane mode
    while avoiding H-H/H-H-H variants that force the rest of CH4 into the
    surface.
    """
    h_counts: dict[tuple[str, int], int] = {}
    for atom_i in subset:
        key = orbit_id[int(atom_i)]
        if key[0] != "H":
            continue
        h_counts[key] = h_counts.get(key, 0) + 1
        if h_counts[key] > 1:
            return False
    return True


def _fingerprint(g: nx.Graph) -> tuple:
    """Cheap graph fingerprint — unequal → guaranteed non-isomorphic."""
    elem_deg = tuple(sorted(
        (
            d.get("coordination_role", "substrate"),
            d.get("element", "X"),
            g.degree(n),
        )
        for n, d in g.nodes(data=True)
    ))
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(sorted(g.degree(n) for n in g.nodes())),
        elem_deg,
        tuple(sorted(
            d.get("coordination_kind", "substrate")
            for _u, _v, d in g.edges(data=True)
        )),
    )


def _placement_signature(atom_cliques: list) -> tuple:
    """Hashable canonical key for a raw placement."""
    return tuple(None if c is None else frozenset(c) for c in atom_cliques)


def _build_adsorbate_coordination_graph(
    substrate_ego: nx.Graph,
    atom_cliques: list,
    reactant_graph: nx.Graph,
) -> nx.Graph:
    """Return the complete labeled graph used to classify one placement.

    Node ids are namespaced so reactant indices cannot collide with substrate
    ids.  ``coordination_role`` prevents a same-element adsorbate atom from
    being mapped onto a substrate atom, while ``coordination_kind`` preserves
    substrate, molecular, and adsorption edges as distinct edge types.

    Including the individual adsorption edges is essential: the union of the
    cliques alone cannot distinguish, for example, O2 bridge--bridge from
    hollow--bridge when both placements touch the same set of surface atoms.
    """
    decorated = nx.Graph()

    for node, data in substrate_ego.nodes(data=True):
        attrs = dict(data)
        attrs["coordination_role"] = "substrate"
        decorated.add_node(("substrate", node), **attrs)
    for left, right, data in substrate_ego.edges(data=True):
        attrs = dict(data)
        attrs["coordination_kind"] = "substrate"
        decorated.add_edge(
            ("substrate", left),
            ("substrate", right),
            **attrs,
        )

    for atom_index in range(len(atom_cliques)):
        attrs = dict(reactant_graph.nodes[atom_index])
        attrs["coordination_role"] = "adsorbate"
        decorated.add_node(("adsorbate", atom_index), **attrs)
    for left, right, data in reactant_graph.edges(data=True):
        attrs = dict(data)
        attrs["coordination_kind"] = "molecular"
        decorated.add_edge(
            ("adsorbate", int(left)),
            ("adsorbate", int(right)),
            **attrs,
        )

    for atom_index, clique in enumerate(atom_cliques):
        if clique is None:
            continue
        for surface_node in clique:
            substrate_node = ("substrate", surface_node)
            if substrate_node not in decorated:
                raise ValueError(
                    "Adsorbate coordination clique contains substrate node "
                    f"{surface_node!r} outside its substrate ego graph."
                )
            decorated.add_edge(
                ("adsorbate", atom_index),
                substrate_node,
                coordination_kind="adsorption",
            )

    return decorated


def _coordination_graph_for_placement(
    G: nx.Graph,
    atom_cliques: list,
    reactant_graph: nx.Graph,
    n_shells: int,
) -> nx.Graph:
    """Build the complete coordination graph for one stored placement."""
    seed = frozenset(
        int(surface_node)
        for clique in atom_cliques
        if clique is not None
        for surface_node in clique
    )
    if not seed:
        raise ValueError("an adsorbate placement must contain a bonded surface clique")
    return _build_adsorbate_coordination_graph(
        _build_ego_graph(G, seed, int(n_shells)),
        atom_cliques,
        reactant_graph,
    )


# ---------------------------------------------------------------------------
# Surface graph helpers
# ---------------------------------------------------------------------------

def _surface_subgraph(G: nx.Graph) -> nx.Graph:
    """Induced subgraph on ``type == "surface"`` nodes."""
    return G.subgraph(
        [n for n, d in G.nodes(data=True) if d.get("type") == "surface"]
    )


def _get_surface_apsp(G: nx.Graph, *, cutoff: int = MAX_PAIR_SHELLS) -> dict:
    """Surface-only all-pairs shortest paths ≤ *cutoff* hops.

    Cached in ``G.graph["surface_apsp"]`` as
    ``{"_cutoff": cutoff, "data": {u: {v: dist}}}``.

    A cached table built with ``cached_cutoff >= cutoff`` is reused as-is
    (extra entries beyond ``cutoff`` are ignored by callers, which always
    threshold by ``cutoff`` themselves).  Only a strictly smaller cached
    cutoff triggers a rebuild — and the rebuild grows to
    ``max(cached_cutoff, cutoff)`` so subsequent callers with the smaller
    cutoff also hit the cache.
    """
    cached = G.graph.get("surface_apsp")
    if isinstance(cached, dict):
        cached_cutoff = cached.get("_cutoff")
        if isinstance(cached_cutoff, int) and cached_cutoff >= int(cutoff):
            return cached["data"]
        if isinstance(cached_cutoff, int):
            cutoff = max(int(cached_cutoff), int(cutoff))
    G_surf = _surface_subgraph(G)
    data: dict[int, dict[int, int]] = {}
    for u in G_surf.nodes:
        data[u] = dict(
            nx.single_source_shortest_path_length(G_surf, u, cutoff=cutoff)
        )
    G.graph["surface_apsp"] = {"_cutoff": int(cutoff), "data": data}
    return data


def _shortest_path_between_cliques(
    clique_a,
    clique_b,
    apsp: dict,
) -> int:
    """Minimum surface-graph hops between the closest nodes of two cliques."""
    if not clique_a or not clique_b:
        return 10 ** 6
    if set(clique_a) & set(clique_b):
        return 0
    target = set(clique_b)
    best   = 10 ** 6
    for a in clique_a:
        a_dists = apsp.get(a, {})
        for b in target:
            d = a_dists.get(b)
            if d is not None and d < best:
                best = d
    return best


def _min_ego_depth_for_connectivity(
    bonded_cliques: list,
    apsp: dict,
) -> int:
    """Smallest ego depth needed so every pair of *bonded_cliques* is connected.

    For two cliques with closest-node hop distance L, a connecting path lies
    within the ``floor(L/2)``-shell ego of their union.
    """
    if len(bonded_cliques) < 2:
        return 0
    max_d = 0
    for i in range(len(bonded_cliques)):
        for j in range(i + 1, len(bonded_cliques)):
            L = _shortest_path_between_cliques(
                bonded_cliques[i], bonded_cliques[j], apsp
            )
            if L >= 10 ** 6:
                return 10 ** 6
            d = L // 2
            if d > max_d:
                max_d = d
    return max_d


def _required_n_shells(
    G: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    apsp: dict,
) -> int:
    """Largest depth needed so the iso ego contains every bonded clique pair."""
    needed = 0
    for ms in adsorbate_sites:
        bonded = [c for c in ms.atom_cliques if c is not None]
        if len(bonded) < 2:
            continue
        first = bonded[0]
        for other in bonded[1:]:
            d = _shortest_path_between_cliques(first, other, apsp)
            if d != 10 ** 6 and d > needed:
                needed = d
    return needed


# ---------------------------------------------------------------------------
# Anchor-site management
# ---------------------------------------------------------------------------

def _ensure_anchor_sites(
    G: nx.Graph,
    element: str,
    *,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    repulsion_cutoff: float | None = REPULSION_CUTOFF,
    n_shells: int = N_SHELLS_DEFAULT,
    anchor_k_max: int | None = None,
    hull_tolerance: float = HULL_TOL,
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS,
    verbose: bool = False,
) -> None:
    """Lazily run :func:`~autokmc.sites.anchors.find_anchor_sites` if needed."""
    cap_by_element = G.graph.get(ANCHOR_K_MAX_BY_ELEMENT, {})
    needs_enumeration = element not in G.graph.get("anchor_sites", {})
    if not needs_enumeration and anchor_k_max is not None:
        if element in cap_by_element:
            needs_enumeration = cap_by_element[element] != anchor_k_max
        else:
            # Anchors restored from an older graph have no cap provenance.
            needs_enumeration = True
    if needs_enumeration:
        find_anchor_sites(
            G, element,
            co_factor=co_factor,
            opt_factor=opt_factor,
            repulsion_weight=repulsion_weight,
            repulsion_cutoff=repulsion_cutoff,
            n_shells=n_shells,
            k_max=anchor_k_max,
            hull_tolerance=hull_tolerance,
            kabsch_max_mappings=kabsch_max_mappings,
            verbose=verbose,
        )


def _all_raw_sites_with_positions(
    G: nx.Graph, element: str
) -> list[tuple[frozenset, np.ndarray]]:
    """Return ``[(clique, position)]`` for every anchor node of *element* on *G*.

    Anchor nodes are materialised by
    :func:`~autokmc.sites.anchors.find_anchor_sites` — one per raw clique,
    each carrying a Kabsch-propagated position.
    """
    out: list[tuple[frozenset, np.ndarray]] = []
    for _n, d in G.nodes(data=True):
        if d.get("type") == "anchor" and d.get("element") == element:
            clq = d.get("clique")
            pos = d.get("position")
            if clq is not None and pos is not None:
                out.append((frozenset(clq), np.asarray(pos, dtype=float)))
    return out


def _clique_position_index(
    G: nx.Graph, elements: list[str]
) -> dict[str, dict[frozenset, np.ndarray]]:
    """``element → {clique: position}`` from every anchor node in *G*.

    Cached on ``G.graph["_clique_position_index_cache"]`` keyed by the
    set of (anchor-node-id, element) pairs.  Adding or removing anchor
    nodes (e.g. by re-running :func:`find_anchor_sites`) automatically
    invalidates the cache because the key changes.
    """
    elem_set = frozenset(elements)
    # Build a structural key that changes whenever anchor nodes change.
    anchor_key = tuple(sorted(
        (int(n), d.get("element"))
        for n, d in G.nodes(data=True)
        if d.get("type") == "anchor"
    ))
    cache = G.graph.get("_clique_position_index_cache")
    if (isinstance(cache, dict)
            and cache.get("_anchor_key") == anchor_key
            and elem_set.issubset(cache.get("_elements", frozenset()))):
        # Sub-restrict the cached dict to the requested elements.
        return {el: cache["data"].get(el, {}) for el in elem_set}

    out: dict[str, dict[frozenset, np.ndarray]] = {}
    for el in elem_set:
        idx: dict[frozenset, np.ndarray] = {}
        for _n, d in G.nodes(data=True):
            if d.get("type") == "anchor" and d.get("element") == el:
                clq = d.get("clique")
                pos = d.get("position")
                if clq is not None and pos is not None:
                    idx[frozenset(clq)] = np.asarray(pos, dtype=float)
        out[el] = idx

    # Merge into the cache (preserving any prior elements computed).
    if isinstance(cache, dict) and cache.get("_anchor_key") == anchor_key:
        cache["data"].update(out)
        cache["_elements"] = cache.get("_elements", frozenset()) | elem_set
    else:
        G.graph["_clique_position_index_cache"] = {
            "_anchor_key": anchor_key,
            "_elements":   elem_set,
            "data":        dict(out),
        }
    return out


# ---------------------------------------------------------------------------
# Iso-class reduction
# ---------------------------------------------------------------------------

def _try_merge_or_new(
    adsorbate_sites: list[AdsorbateSite],
    *,
    reactant_smiles: str,
    atom_cliques: list,
    positions: np.ndarray,
    ego_graph: nx.Graph,
    node_match,
    edge_match,
    seen_signatures: set,
) -> None:
    """Merge into an existing iso-class if isomorphic, else append a new one.

    Exact duplicate placements (same surface cliques in the same atom slots)
    are dropped via *seen_signatures*.
    """
    sig = _placement_signature(atom_cliques)
    if sig in seen_signatures:
        return
    seen_signatures.add(sig)

    bonded_pattern = tuple(c is None for c in atom_cliques)
    fkey = _fingerprint(ego_graph)
    pos_arr = np.asarray(positions, dtype=float)

    for ms in adsorbate_sites:
        if tuple(c is None for c in ms.atom_cliques) != bonded_pattern:
            continue
        if ms.ego_graph is None:
            continue
        if _fingerprint(ms.ego_graph) != fkey:
            continue
        if isomorphism.GraphMatcher(
            ego_graph,
            ms.ego_graph,
            node_match=node_match,
            edge_match=edge_match,
        ).is_isomorphic():
            ms.members.append(list(atom_cliques))
            return

    adsorbate_sites.append(AdsorbateSite(
        reactant     = reactant_smiles,
        n_atoms      = len(atom_cliques),
        atom_cliques = list(atom_cliques),
        positions    = pos_arr,
        iso_class    = len(adsorbate_sites),
        members      = [list(atom_cliques)],
        ego_graph    = ego_graph,
    ))


# ---------------------------------------------------------------------------
# Graph materialisation
# ---------------------------------------------------------------------------


def _remove_adsorbate_nodes(G: nx.Graph, smiles: str) -> None:
    """Drop all adsorbate-site nodes carrying ``reactant == smiles``."""
    stale = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "adsorbate" and d.get("reactant") == smiles
    ]
    if stale:
        G.remove_nodes_from(stale)


def rebuild_adsorbate_reverse_indexes(G: nx.Graph) -> None:
    """Rebuild adsorption reverse indexes from the active graph-side site list."""
    clique_to_members: dict = {}
    occupied_by_clique: dict = {}
    surface_node_to_members: dict = {}
    n_occupied = 0

    sites_by_smiles = G.graph.get("adsorbate_sites", {}) or {}
    for sites in sites_by_smiles.values():
        for ms in sites:
            ms._member_cliques = []
            ms._n_occupied = 0
            for m_idx, node_ids in enumerate(ms.member_node_ids):
                cliques: list = []
                member_occupied = False
                for nid in node_ids:
                    if nid not in G:
                        continue
                    clq = G.nodes[nid].get("clique")  # already a frozenset
                    if clq is None:
                        continue
                    cliques.append(clq)
                    clique_to_members.setdefault(clq, []).append((ms, m_idx))
                    occupied_by_clique.setdefault(clq, set())
                    if G.nodes[nid].get("occupied", False):
                        member_occupied = True
                        occupied_by_clique[clq].add(nid)
                    for surf_id in clq:
                        surface_node_to_members.setdefault(
                            int(surf_id), [],
                        ).append((ms, m_idx))
                ms._member_cliques.append(tuple(cliques))
                if member_occupied:
                    ms._n_occupied += 1
                    n_occupied += 1

    G.graph["clique_to_members"] = clique_to_members
    G.graph["occupied_by_clique"] = occupied_by_clique
    G.graph["surface_node_to_members"] = surface_node_to_members
    G.graph["n_occupied"] = n_occupied


def _member_positions(
    reactant,
    atom_cliques: list,
    pos_index: dict[str, dict[frozenset, np.ndarray]],
    elements: list[str],
    G: nx.Graph,
    pbc: np.ndarray,
) -> np.ndarray | None:
    """Reconstruct ``(n_atoms, 3)`` Cartesians for one member.

    Returns ``None`` if any bonded clique has no cached position.
    """
    bonded: list[int] = []
    bonded_pos: list[np.ndarray] = []
    for i, clq in enumerate(atom_cliques):
        if clq is None:
            continue
        p = pos_index.get(elements[i], {}).get(frozenset(clq))
        if p is None:
            return None
        bonded.append(i)
        bonded_pos.append(p)
    if not bonded:
        return None
    return _full_adsorbate_positions(
        reactant, bonded, np.array(bonded_pos, dtype=float), G, pbc
    )


def _wrap_adsorbate_positions_for_storage(
    positions: np.ndarray,
    atom_cliques,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Canonicalize one adsorbate geometry while preserving molecular shape."""
    pos_arr = np.asarray(positions, dtype=float)
    if not pbc.any() or pos_arr.size == 0:
        return pos_arr.copy()
    ref = pos_arr[0]
    for atom_i, clq in enumerate(atom_cliques):
        if clq is not None:
            ref = pos_arr[atom_i]
            break
    return wrap_positions_into_cell(pos_arr, cell, pbc, reference=ref)


def _materialise_adsorbate_nodes(
    G: nx.Graph,
    reactant,
    adsorbate_sites: list[AdsorbateSite],
) -> None:
    """Create one connected adsorbate-site subgraph per member placement on *G*.

    For each :class:`AdsorbateSite` and each member, ``n_atoms`` nodes are
    added with::

        type           = "adsorbate"
        occupied       = False          ← invisible to ego-graph BFS
        reactant       = reactant.smiles
        iso_class      = ms.iso_class
        reactant_index = atom index within the reactant
        reactant_orbit = molecular automorphism-orbit index for that element
        element        = atom element symbol
        clique         = frozenset of bonded surface atom ids (or None)
        is_bonded      = clique is not None
        siblings       = tuple of the other node ids in the same placement

    Intramolecular edges mirror ``reactant.graph`` topology
    (``intra_adsorbate=True``).  Bonded atoms receive ``anchor_bond=True``
    edges to every surface atom in their clique.

    Per-member node ids are pushed onto ``ms.member_node_ids``.
    """
    smiles      = reactant.smiles
    react_sym   = reactant.atoms.get_chemical_symbols()
    if reactant.graph is not None:
        react_radii: list[float] = []
        missing: list[tuple[int, str]] = []
        for n, d in reactant.graph.nodes(data=True):
            r = d.get("covalent_radius")
            if r is None:
                missing.append((int(n), str(d.get("element", "?"))))
                react_radii.append(0.0)
            else:
                react_radii.append(float(r))
        if missing:
            # This should not happen for any standard element produced by
            # ``build_reactant`` (which routes through ``build_graph`` →
            # ``ase.data.covalent_radii``), but exotic / non-standard
            # elements or hand-built graphs may be missing the attribute.
            warnings.warn(
                f"_materialise_adsorbate_nodes: reactant {smiles!r} has "
                f"{len(missing)} node(s) without a 'covalent_radius' "
                f"attribute (atoms: {missing}); using r_cov=0.0 for these "
                "atoms — steric exclusion against the surface will be "
                "weakened.  Check that every reactant atom is a standard "
                "element recognised by ase.data.covalent_radii.",
                RuntimeWarning,
                stacklevel=3,
            )
    else:
        react_radii = [0.0] * len(react_sym)
    intra_edges = (
        list(reactant.graph.edges())
        if reactant.graph is not None else []
    )
    reactant_orbits = _orbit_id_of(reactant)

    _remove_adsorbate_nodes(G, smiles)

    pos_index = _clique_position_index(G, react_sym)
    cell = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc = _effective_pbc(G, cell)

    for ms in adsorbate_sites:
        ms.member_node_ids = []
        ms.positions = _wrap_adsorbate_positions_for_storage(
            np.asarray(ms.positions, dtype=float),
            ms.atom_cliques,
            cell,
            pbc,
        )
        rep_positions = np.asarray(ms.positions, dtype=float)

        for member_index, atom_cliques in enumerate(ms.members):
            # Geometry for this member (falls back to representative if any
            # bonded clique is missing from the position index).
            positions = _member_positions(
                reactant, atom_cliques, pos_index, react_sym, G, pbc
            )
            if positions is None:
                positions = rep_positions
            positions = np.asarray(positions, dtype=float)
            positions = _wrap_adsorbate_positions_for_storage(
                positions, atom_cliques, cell, pbc,
            )

            # Allocate a contiguous block without rescanning every graph node
            # for every member.
            node_ids = list(_reserve_node_ids(G, len(react_sym)))

            # Add one node per reactant atom.
            for i, nid in enumerate(node_ids):
                clq = atom_cliques[i]
                G.add_node(
                    nid,
                    element         = react_sym[i],
                    position        = positions[i].copy(),
                    index           = nid,
                    type            = "adsorbate",
                    occupied        = False,
                    covalent_radius = react_radii[i],
                    reactant        = smiles,
                    iso_class       = int(ms.iso_class),
                    site_iso_class  = int(ms.iso_class),
                    site_member_index = int(member_index),
                    reactant_index  = int(i),
                    reactant_orbit  = int(reactant_orbits[i][1]),
                    clique          = (frozenset(clq) if clq is not None else None),
                    k               = (len(clq) if clq is not None else 0),
                    is_bonded       = (clq is not None),
                    siblings        = tuple(n for n in node_ids if n != nid),
                    optimised       = False,
                )

            # Intramolecular edges (mirror reactant.graph topology).
            for u, v in intra_edges:
                if int(u) >= len(node_ids) or int(v) >= len(node_ids):
                    continue
                a, b = node_ids[int(u)], node_ids[int(v)]
                d = float(np.linalg.norm(
                    positions[int(u)] - positions[int(v)]
                ))
                G.add_edge(a, b, distance=d, offset=(0, 0, 0),
                           intra_adsorbate=True)

            # Surface attachment edges for bonded atoms.
            for i, nid in enumerate(node_ids):
                clq = atom_cliques[i]
                if clq is None:
                    continue
                for surf_id in clq:
                    if surf_id not in G:
                        continue
                    displacement = (
                        np.asarray(G.nodes[surf_id]["position"], dtype=float)
                        - positions[i]
                    )
                    if pbc.any():
                        displacement = minimum_image_vectors(
                            displacement,
                            cell,
                            pbc,
                        )
                    d = float(np.linalg.norm(displacement))
                    G.add_edge(nid, surf_id, distance=d, offset=(0, 0, 0),
                               anchor_bond=True)

            ms.member_node_ids.append(node_ids)


# ---------------------------------------------------------------------------
# Stability-pruning helpers
# ---------------------------------------------------------------------------

def _build_pruning_atoms(
    G: nx.Graph,
    ms: AdsorbateSite,
    react_sym: list[str],
    *,
    frozen_indices: list[int] | None = None,
):
    """Build an ASE Atoms object for a bare (no lateral neighbours) stability check.

    Slab atoms are taken from *G* (bulk + surface, sorted by original ASE
    ``index``).  Adsorbate atoms are placed at the representative positions
    ``ms.positions``.

    The returned :class:`~ase.Atoms` carries an ``arrays["surface"]`` int8
    array that mirrors :mod:`autokmc.core.graph` conventions (0=bulk, 1=surface,
    2=adsorbate) so that :func:`autokmc.core.graph.build_graph` can be called
    directly on the relaxed structure for the connectivity check.

    Returns
    -------
    atoms : Atoms
    n_slab : int
    n_ads : int
    node_to_ase : dict[int, int]
        Mapping from G surface/bulk node-id → ASE atom index in *atoms*.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms

    slab_nodes = sorted(
        (n for n, d in G.nodes(data=True) if d.get("type") in ("bulk", "surface")),
        key=lambda n: G.nodes[n].get("index", n),
    )
    slab_sym = [G.nodes[n]["element"]  for n in slab_nodes]
    slab_pos = [G.nodes[n]["position"] for n in slab_nodes]
    slab_tag = [
        1 if G.nodes[n].get("type") == "surface" else 0
        for n in slab_nodes
    ]

    ads_pos = np.asarray(ms.positions, dtype=float)  # (n_atoms, 3)
    n_ads   = len(react_sym)

    symbols   = slab_sym + list(react_sym)
    positions = slab_pos + [ads_pos[i].tolist() for i in range(n_ads)]
    surface_array = np.asarray(slab_tag + [2] * n_ads, dtype=np.int8)

    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = full_pbc_for_cell(cell)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    atoms.arrays["surface"] = surface_array
    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))

    node_to_ase = {int(nid): i for i, nid in enumerate(slab_nodes)}
    return atoms, len(slab_nodes), n_ads, node_to_ase


class _RigidAdsorbateOptimizable:
    """Expose one adsorbate's rigid modes to an ASE optimizer.

    The first pseudo-position contains translation in Angstrom.  Molecular
    adsorbates have a second pseudo-position containing an axis-angle rotation
    multiplied by a characteristic molecular length.  The corresponding
    pseudo-forces are the net adsorbate force and torque divided by that same
    length, so ASE's ordinary ``fmax`` test has eV/Angstrom units for both
    rigid modes.  Slab coordinates are never exposed to the optimizer.
    """

    def __init__(self, atoms, n_slab: int) -> None:
        self.atoms = atoms
        self.n_slab = int(n_slab)
        self.n_ads = len(atoms) - self.n_slab
        if self.n_ads < 1:
            raise ValueError(
                "rigid adsorbate optimization requires at least one atom"
            )

        adsorbate_positions = np.asarray(
            atoms.get_positions()[self.n_slab :],
            dtype=float,
        )
        adsorbate_positions = unwrap_positions_about_reference(
            adsorbate_positions,
            np.asarray(atoms.cell, dtype=float),
            np.asarray(atoms.pbc, dtype=bool),
        )
        self.center = adsorbate_positions.mean(axis=0)
        self.reference_offsets = adsorbate_positions - self.center
        squared_radii = np.einsum(
            "ij,ij->i",
            self.reference_offsets,
            self.reference_offsets,
        )
        rms_radius = float(np.sqrt(np.mean(squared_radii)))
        # A 1-Angstrom floor prevents tiny or nearly coincident structures
        # from turning a harmless torque into a numerically enormous force.
        self.rotation_scale = max(1.0, rms_radius)
        self.has_rotation = self.n_ads > 1 and rms_radius > 1.0e-12
        self.coordinates = np.zeros((2 if self.has_rotation else 1, 3))

        positions = atoms.get_positions()
        positions[self.n_slab :] = adsorbate_positions
        atoms.set_positions(positions)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __ase_optimizable__(self):
        return self

    def get_positions(self) -> np.ndarray:
        return self.coordinates.copy()

    def set_positions(self, coordinates) -> None:
        candidate = np.asarray(coordinates, dtype=float)
        if candidate.shape != self.coordinates.shape:
            raise ValueError(
                "rigid optimizer coordinates must have shape "
                f"{self.coordinates.shape}, got {candidate.shape}"
            )
        self.coordinates = candidate.copy()
        if self.has_rotation:
            rotvec = self.coordinates[1] / self.rotation_scale
            rotation = _rotation_from_axis_angle(rotvec)
            adsorbate_positions = self.reference_offsets @ rotation.T
        else:
            adsorbate_positions = self.reference_offsets.copy()
        adsorbate_positions += self.center + self.coordinates[0]

        positions = self.atoms.get_positions()
        positions[self.n_slab :] = adsorbate_positions
        self.atoms.set_positions(positions)

    def get_forces(self) -> np.ndarray:
        atomic_forces = np.asarray(self.atoms.get_forces(), dtype=float)
        adsorbate_forces = atomic_forces[self.n_slab :]
        if not np.all(np.isfinite(adsorbate_forces)):
            raise ValueError("calculator returned non-finite adsorbate forces")

        generalized_forces = np.empty_like(self.coordinates)
        generalized_forces[0] = adsorbate_forces.sum(axis=0)
        if self.has_rotation:
            rotvec = self.coordinates[1] / self.rotation_scale
            rotation = _rotation_from_axis_angle(rotvec)
            right_jacobian = _rotation_right_jacobian(rotvec)
            body_forces = adsorbate_forces @ rotation
            body_torque = np.cross(
                self.reference_offsets,
                body_forces,
            ).sum(axis=0)
            generalized_forces[1] = (
                right_jacobian.T @ body_torque
            ) / self.rotation_scale
        return generalized_forces

    def get_potential_energy(self) -> float:
        return float(self.atoms.get_potential_energy())

    def iterimages(self):
        return self.atoms.iterimages()

    def converged(self, forces, fmax: float) -> bool:
        force_norms = np.linalg.norm(np.asarray(forces, dtype=float), axis=1)
        return bool(force_norms.max() < float(fmax))

    def is_neb(self) -> bool:
        return False


def _optimise_rigid_adsorbate_with_potential(
    atoms,
    n_slab: int,
    calculator,
    *,
    fmax: float,
    max_steps: int,
    optimizer: str,
    optimizer_kwargs: dict[str, Any] | None,
):
    """Optimize only rigid adsorbate translation/rotation with a potential."""
    from autokmc.structure.optimization import (
        StructureOptimisationError,
        _optimizer_class,
    )

    result = atoms.copy()
    result.calc = calculator
    rigid = _RigidAdsorbateOptimizable(result, n_slab)
    optimizer_cls = _optimizer_class(optimizer)
    constructor_kwargs = normalize_optimizer_kwargs(
        optimizer,
        optimizer_kwargs,
        allowed=REGULAR_OPTIMIZERS,
        setting="optimizer_kwargs",
    )
    # One global optimizer restart/trajectory cannot safely be shared by the
    # two-coordinate rigid body and the subsequent full atomic relaxation.
    constructor_kwargs.pop("restart", None)
    constructor_kwargs.pop("trajectory", None)
    opt = optimizer_cls(rigid, logfile=os.devnull, **constructor_kwargs)
    try:
        opt.run(fmax=fmax, steps=max_steps)
        generalized_forces = rigid.get_forces()
    except CalculatorConfigError:
        raise
    except Exception as exc:
        completed_steps = int(opt.get_number_of_steps())
        raise StructureOptimisationError(
            "rigid adsorbate optimization failed after "
            f"{completed_steps} steps: {type(exc).__name__}: {exc}",
            result,
            converged=None,
            steps=completed_steps,
        ) from exc

    completed_steps = int(opt.get_number_of_steps())
    max_generalized_force = float(
        np.linalg.norm(generalized_forces, axis=1).max()
    )
    if not rigid.converged(generalized_forces, fmax):
        raise StructureOptimisationError(
            "rigid adsorbate optimization did not converge within "
            f"{max_steps} steps (fmax={fmax} eV/Angstrom; "
            f"max rigid force={max_generalized_force:.6g} eV/Angstrom)",
            result,
            converged=False,
            steps=completed_steps,
        )
    return result, max_generalized_force, completed_steps


def _relaxed_adsorbate_positions_in_graph_frame(
    G: nx.Graph,
    ms: AdsorbateSite,
    atoms_opt,
    n_slab: int,
    n_ads: int,
    node_to_ase: dict[int, int],
    *,
    frame_depth: int,
) -> np.ndarray:
    """Return relaxed adsorbate positions projected back onto G's slab frame.

    The pruning relaxation may move unfrozen slab atoms.  The live graph keeps
    the original slab positions, so raw ``atoms_opt`` adsorbate coordinates are
    in the wrong frame for graph storage and member propagation.  Align the
    relaxed local surface ego back onto the graph ego, then apply that transform
    to the relaxed adsorbate atoms.
    """
    all_pos = np.asarray(atoms_opt.get_positions(), dtype=float)
    ads_pos = np.asarray(all_pos[n_slab : n_slab + n_ads], dtype=float)

    rep_seed: frozenset = frozenset(
        int(n)
        for c in ms.atom_cliques if c is not None
        for n in c
    )
    if not rep_seed:
        return ads_pos

    ego = _build_ego_graph(G, rep_seed, max(0, int(frame_depth)))
    frame_nodes = [
        int(n)
        for n, d in ego.nodes(data=True)
        if d.get("type") in ("bulk", "surface") and int(n) in node_to_ase
    ]
    if not frame_nodes:
        return ads_pos

    graph_pos_raw = np.asarray(
        [G.nodes[n]["position"] for n in frame_nodes],
        dtype=float,
    )
    relaxed_pos_raw = np.asarray(
        [all_pos[node_to_ase[n]] for n in frame_nodes],
        dtype=float,
    )

    cell = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc = _effective_pbc(G, cell)
    if pbc.any():
        graph_pos = unwrap_positions_about_reference(
            graph_pos_raw,
            cell,
            pbc,
        )
        relaxed_pos = graph_pos + minimum_image_vectors(
            relaxed_pos_raw - graph_pos_raw,
            cell,
            pbc,
        )
        ads_pos = unwrap_positions_about_reference(
            ads_pos,
            cell,
            pbc,
            reference=graph_pos[0],
        )
    else:
        graph_pos = graph_pos_raw
        relaxed_pos = relaxed_pos_raw

    R, t = _kabsch(relaxed_pos, graph_pos)
    return ads_pos @ R.T + t


def _intended_adsorbate_edges(
    ms: AdsorbateSite,
    reactant,
    n_slab: int,
    node_to_ase: dict[int, int],
) -> set[frozenset]:
    """Edge set the relaxed graph **must** contain in the adsorbate region.

    Includes:

    * Intramolecular bonds from ``reactant.graph`` (re-mapped to ASE indices
      ``n_slab + reactant_atom_index``).
    * Anchor bonds: for every reactant atom *i* with
      ``ms.atom_cliques[i] is not None``, one edge from ``n_slab + i`` to
      every surface atom in that clique (mapped via *node_to_ase*).

    Used by :func:`prune_unstable_adsorbate_sites` to compare against the
    edges actually produced by :func:`autokmc.core.graph.build_graph` on the
    ML-relaxed structure.
    """
    edges: set[frozenset] = set()

    if reactant.graph is not None:
        for u, v in reactant.graph.edges():
            edges.add(frozenset((n_slab + int(u), n_slab + int(v))))

    for i, clq in enumerate(ms.atom_cliques):
        if clq is None:
            continue
        ads_idx = n_slab + i
        for surf_nid in clq:
            ase_surf = node_to_ase.get(int(surf_nid))
            if ase_surf is None:
                continue
            edges.add(frozenset((ads_idx, ase_surf)))

    return edges


def _adsorbate_edges_from_graph(
    G_relaxed: nx.Graph,
    n_slab: int,
) -> set[frozenset]:
    """Edges of *G_relaxed* that touch at least one adsorbate atom.

    Adsorbate atoms occupy ASE indices ``>= n_slab`` (the layout produced
    by :func:`_build_pruning_atoms`); slab–slab edges are excluded because
    small ML-driven relaxations of un-frozen metal atoms can flip pairs
    across the natural-cutoff threshold without affecting the adsorbate.
    """
    edges: set[frozenset] = set()
    for u, v in G_relaxed.edges():
        if int(u) < n_slab and int(v) < n_slab:
            continue
        edges.add(frozenset((int(u), int(v))))
    return edges


@dataclass(frozen=True)
class _GeometryConnectivityContext:
    """Precomputed invariant data for rigid-pose connectivity checks."""

    slab_positions: np.ndarray
    slab_cutoffs: np.ndarray
    adsorbate_cutoffs: np.ndarray
    node_to_slab_index: dict[int, int]
    intramolecular_edges: frozenset[frozenset]
    cell: np.ndarray
    pbc: np.ndarray

    @property
    def n_slab(self) -> int:
        return int(len(self.slab_positions))


def _prepare_geometry_connectivity_context(
    G: nx.Graph,
    reactant,
    *,
    nl_mult: float,
) -> _GeometryConnectivityContext:
    """Build the invariant half of the ASE-neighbour-list connectivity test."""
    from ase.data import atomic_numbers as _AN, covalent_radii as _RC

    slab_nodes = sorted(
        (
            int(node)
            for node, data in G.nodes(data=True)
            if data.get("type") in ("bulk", "surface")
        ),
        key=lambda node: G.nodes[node].get("index", node),
    )
    slab_positions = np.asarray(
        [G.nodes[node]["position"] for node in slab_nodes],
        dtype=float,
    ).reshape((-1, 3))
    slab_cutoffs = np.asarray(
        [
            float(nl_mult) * float(_RC[_AN[G.nodes[node]["element"]]])
            + NEIGHBORLIST_SKIN
            for node in slab_nodes
        ],
        dtype=float,
    )
    adsorbate_cutoffs = np.asarray(
        [
            float(nl_mult) * float(_RC[_AN[symbol]])
            + NEIGHBORLIST_SKIN
            for symbol in reactant.atoms.get_chemical_symbols()
        ],
        dtype=float,
    )
    n_slab = len(slab_nodes)
    intramolecular_edges = frozenset(
        frozenset((n_slab + int(u), n_slab + int(v)))
        for u, v in (
            reactant.graph.edges()
            if reactant.graph is not None
            else ()
        )
    )
    cell = np.asarray(G.graph.get("cell", np.zeros((3, 3))), dtype=float)
    return _GeometryConnectivityContext(
        slab_positions=slab_positions,
        slab_cutoffs=slab_cutoffs,
        adsorbate_cutoffs=adsorbate_cutoffs,
        node_to_slab_index={
            int(node): index for index, node in enumerate(slab_nodes)
        },
        intramolecular_edges=intramolecular_edges,
        cell=cell,
        # Match ``_build_pruning_atoms`` / ``build_graph`` exactly: a real
        # material cell is treated as periodic along every lattice vector.
        pbc=full_pbc_for_cell(cell),
    )


def _intended_geometry_edges(
    context: _GeometryConnectivityContext,
    atom_cliques: list,
) -> set[frozenset]:
    """Return intended intramolecular plus surface-anchor edges."""
    intended = set(context.intramolecular_edges)
    for adsorbate_index, clique in enumerate(atom_cliques):
        if clique is None:
            continue
        adsorbate_node = context.n_slab + int(adsorbate_index)
        for surface_node in clique:
            slab_index = context.node_to_slab_index.get(int(surface_node))
            if slab_index is not None:
                intended.add(frozenset((adsorbate_node, slab_index)))
    return intended


def _actual_geometry_edges(
    context: _GeometryConnectivityContext,
    positions: np.ndarray,
) -> set[frozenset]:
    """Vectorised equivalent of ASE ``NeighborList`` adsorbate edges."""
    positions = np.asarray(positions, dtype=float).reshape((-1, 3))
    n_adsorbate = len(positions)
    actual: set[frozenset] = set()

    if context.n_slab and n_adsorbate:
        displacements = (
            context.slab_positions[None, :, :] - positions[:, None, :]
        )
        if context.pbc.any():
            displacements = minimum_image_vectors(
                displacements,
                context.cell,
                context.pbc,
            )
        distances = np.linalg.norm(displacements, axis=2)
        thresholds = (
            context.adsorbate_cutoffs[:, None]
            + context.slab_cutoffs[None, :]
        )
        for adsorbate_index, slab_index in np.argwhere(
            distances < thresholds
        ):
            actual.add(
                frozenset(
                    (
                        context.n_slab + int(adsorbate_index),
                        int(slab_index),
                    )
                )
            )

    if n_adsorbate > 1:
        pair_displacements = positions[None, :, :] - positions[:, None, :]
        if context.pbc.any():
            pair_displacements = minimum_image_vectors(
                pair_displacements,
                context.cell,
                context.pbc,
            )
        pair_distances = np.linalg.norm(pair_displacements, axis=2)
        pair_thresholds = (
            context.adsorbate_cutoffs[:, None]
            + context.adsorbate_cutoffs[None, :]
        )
        for first in range(n_adsorbate):
            for second in range(first + 1, n_adsorbate):
                if pair_distances[first, second] < pair_thresholds[first, second]:
                    actual.add(
                        frozenset(
                            (
                                context.n_slab + first,
                                context.n_slab + second,
                            )
                        )
                    )
    return actual


def _geometry_connectivity_mismatch(
    G: nx.Graph,
    ms: AdsorbateSite,
    reactant,
    positions: np.ndarray,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
    _context: _GeometryConnectivityContext | None = None,
) -> tuple[set[frozenset], set[frozenset]] | None:
    """Return ``(missing, extra)`` if a calc-free geometry has wrong bonds.

    The calculation is exactly equivalent to the ASE ``NeighborList`` cutoff
    used by :func:`autokmc.core.graph.build_graph`, including its 0.3 Å
    per-atom skin, but evaluates only adsorbate-touching pairs using vectorised
    minimum-image distances.  Rigid refinement can therefore reuse one
    precomputed slab/cutoff context instead of rebuilding a complete
    ``Atoms`` and neighbour list after every orientation.
    """
    return _geometry_connectivity_mismatch_for_cliques(
        G,
        ms.atom_cliques,
        reactant,
        positions,
        nl_mult=nl_mult,
        _context=_context,
    )


def _geometry_connectivity_mismatch_for_cliques(
    G: nx.Graph,
    atom_cliques: list,
    reactant,
    positions: np.ndarray,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
    _context: _GeometryConnectivityContext | None = None,
) -> tuple[set[frozenset], set[frozenset]] | None:
    """Connectivity mismatch for explicit per-atom surface cliques."""
    context = _context or _prepare_geometry_connectivity_context(
        G,
        reactant,
        nl_mult=float(nl_mult),
    )
    intended_edges = _intended_geometry_edges(context, atom_cliques)
    actual_edges = _actual_geometry_edges(context, positions)
    missing = intended_edges - actual_edges
    extra = actual_edges - intended_edges
    if missing or extra:
        return missing, extra
    return None


def _compact_graph_positions(
    G: nx.Graph,
    nodes: list[int],
    seed_nodes: list[int],
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return graph-node positions in one MIC-consistent local image."""
    positions = np.asarray(
        [G.nodes[int(node)]["position"] for node in nodes],
        dtype=float,
    )
    if not np.asarray(pbc, dtype=bool).any():
        return positions

    seed_positions = np.asarray(
        [G.nodes[int(node)]["position"] for node in seed_nodes],
        dtype=float,
    )
    seed_unwrapped = unwrap_positions_about_reference(
        seed_positions,
        cell,
        pbc,
    )
    reference = seed_unwrapped.mean(axis=0)
    return unwrap_positions_about_reference(
        positions,
        cell,
        pbc,
        reference=reference,
    )


def _mapped_adsorbate_positions(
    representative_positions: np.ndarray,
    mapping: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply a decorated-graph mapping, including its molecular permutation."""
    representative_positions = np.asarray(representative_positions, dtype=float)
    transformed = representative_positions @ rotation.T + translation
    member_positions = np.empty_like(transformed)
    assigned: set[int] = set()

    for representative_index in range(len(representative_positions)):
        mapped_node = mapping.get(("adsorbate", representative_index))
        if (
            not isinstance(mapped_node, tuple)
            or len(mapped_node) != 2
            or mapped_node[0] != "adsorbate"
        ):
            raise ValueError(
                "coordination-graph mapping did not preserve an adsorbate node"
            )
        member_index = int(mapped_node[1])
        if member_index < 0 or member_index >= len(member_positions):
            raise ValueError("coordination-graph mapping produced an invalid atom index")
        if member_index in assigned:
            raise ValueError("coordination-graph mapping is not a molecular permutation")
        member_positions[member_index] = transformed[representative_index]
        assigned.add(member_index)

    if len(assigned) != len(member_positions):
        raise ValueError("coordination-graph mapping omitted an adsorbate atom")
    return member_positions


def _unconstrained_orthogonal_alignment(
    source: np.ndarray,
    destination: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Best orthogonal alignment, permitting a reflection when required."""
    source_centroid = source.mean(axis=0)
    destination_centroid = destination.mean(axis=0)
    covariance = (
        (source - source_centroid).T
        @ (destination - destination_centroid)
    )
    left, _singular_values, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    translation = destination_centroid - rotation @ source_centroid
    return rotation, translation


def _molecular_mapping_preserves_handedness(
    representative_positions: np.ndarray,
    mapped_positions: np.ndarray,
    *,
    rmsd_tol: float = 1.0e-6,
) -> bool:
    """Return whether a mapped molecular pose is proper-rotation equivalent.

    An improper substrate isometry is safe for a linear, planar, or achiral
    molecular mapping when the resulting atom-ordered geometry can itself be
    superimposed on the representative by a proper rotation. It is rejected
    when that test detects a chiral inversion.
    """
    rotation, translation = _kabsch(
        np.asarray(representative_positions, dtype=float),
        np.asarray(mapped_positions, dtype=float),
    )
    residual = (
        np.asarray(representative_positions, dtype=float) @ rotation.T
        + translation
        - np.asarray(mapped_positions, dtype=float)
    )
    rmsd = float(np.sqrt(np.mean(np.einsum("ij,ij->i", residual, residual))))
    return rmsd <= float(rmsd_tol)


def _propagate_adsorbate_member_positions(
    G: nx.Graph,
    representative_cliques: list,
    member_cliques: list,
    representative_positions: np.ndarray,
    reactant,
    *,
    n_shells: int,
    nl_mult: float = NL_MULT_DEFAULT,
    max_mappings: int = KABSCH_MAX_MAPPINGS,
    rmsd_tol: float = 1.0e-4,
    _context: _GeometryConnectivityContext | None = None,
) -> np.ndarray:
    """Map a representative pose to one member using the full site graph.

    Every candidate graph isomorphism supplies both the substrate-node
    correspondence used for the MIC-aware Kabsch fit and the molecular atom
    permutation used to write the transformed coordinates. A candidate is
    accepted only if it remains outward and has exactly the member's intended
    intramolecular and surface-anchor connectivity.

    Raises
    ------
    RuntimeError
        If no valid decorated mapping is found within *max_mappings*. This is
        deliberately fail-closed: leaving stale positions or writing a pose
        validated only against the union of surface cliques can corrupt later
        adsorption, diffusion, and bond-reaction structures.
    """
    mapping_limit = int(max_mappings)
    if mapping_limit < 1:
        raise ValueError("max_mappings must be at least 1")

    representative_graph = _coordination_graph_for_placement(
        G,
        representative_cliques,
        reactant.graph,
        n_shells,
    )
    member_graph = _coordination_graph_for_placement(
        G,
        member_cliques,
        reactant.graph,
        n_shells,
    )
    node_match = isomorphism.categorical_node_match(
        ["coordination_role", "element"],
        ["substrate", "X"],
    )
    edge_match = isomorphism.categorical_edge_match(
        "coordination_kind",
        "substrate",
    )
    matcher = isomorphism.GraphMatcher(
        representative_graph,
        member_graph,
        node_match=node_match,
        edge_match=edge_match,
    )

    representative_surface_nodes = sorted(
        int(node[1])
        for node, data in representative_graph.nodes(data=True)
        if data.get("coordination_role") == "substrate"
    )
    representative_seed = sorted(
        {
            int(surface_node)
            for clique in representative_cliques
            if clique is not None
            for surface_node in clique
        }
    )
    member_seed = sorted(
        {
            int(surface_node)
            for clique in member_cliques
            if clique is not None
            for surface_node in clique
        }
    )
    cell, _cell_inv, pbc, _use_mic = _get_cell(G)
    source = _compact_graph_positions(
        G,
        representative_surface_nodes,
        representative_seed,
        cell,
        pbc,
    )

    best_positions: np.ndarray | None = None
    best_rmsd = np.inf
    mappings_tested = 0
    outward_rejections = 0
    connectivity_rejections = 0

    for mapping_index, mapping in enumerate(matcher.isomorphisms_iter()):
        if mapping_index >= mapping_limit:
            break
        mappings_tested += 1
        mapped_member_nodes = [
            int(mapping[("substrate", node)][1])
            for node in representative_surface_nodes
        ]
        destination = _compact_graph_positions(
            G,
            mapped_member_nodes,
            member_seed,
            cell,
            pbc,
        )
        proper_transform = _kabsch(source, destination)
        orthogonal_transform = _unconstrained_orthogonal_alignment(
            source,
            destination,
        )
        transforms = [proper_transform]
        if np.linalg.det(orthogonal_transform[0]) < 0.0:
            transforms.append(orthogonal_transform)

        for rotation, translation in transforms:
            residual = source @ rotation.T + translation - destination
            rmsd = float(np.sqrt(np.mean(np.einsum("ij,ij->i", residual, residual))))
            candidate = _mapped_adsorbate_positions(
                representative_positions,
                mapping,
                rotation,
                translation,
            )
            if np.linalg.det(rotation) < 0.0:
                reference_positions = np.asarray(
                    reactant.atoms.get_positions(),
                    dtype=float,
                )
                mapped_reference = _mapped_adsorbate_positions(
                    reference_positions,
                    mapping,
                    rotation,
                    np.zeros(3, dtype=float),
                )
                if not _molecular_mapping_preserves_handedness(
                    reference_positions,
                    mapped_reference,
                ):
                    connectivity_rejections += 1
                    continue
            if not np.isfinite(candidate).all():
                connectivity_rejections += 1
                continue
            if not _adsorbate_pose_is_outward(G, member_cliques, candidate, pbc):
                outward_rejections += 1
                continue
            if _geometry_connectivity_mismatch_for_cliques(
                G,
                member_cliques,
                reactant,
                candidate,
                nl_mult=nl_mult,
                _context=_context,
            ) is not None:
                connectivity_rejections += 1
                continue
            if rmsd < best_rmsd:
                best_positions = candidate
                best_rmsd = rmsd
                if rmsd <= float(rmsd_tol):
                    return best_positions

    if best_positions is not None:
        return best_positions

    raise RuntimeError(
        "no valid decorated adsorption mapping found "
        f"(tested={mappings_tested}, limit={mapping_limit}, "
        f"outward_rejections={outward_rejections}, "
        f"connectivity_rejections={connectivity_rejections})"
    )


def _remove_iso_class_nodes(G: nx.Graph, ms: AdsorbateSite) -> None:
    """Remove all materialised G-nodes for *ms* (no-op if absent)."""
    to_remove = [
        nid
        for node_ids in ms.member_node_ids
        for nid in node_ids
        if nid in G
    ]
    if to_remove:
        G.remove_nodes_from(to_remove)


# ---------------------------------------------------------------------------
# Public API — prune_unstable_adsorbate_sites
# ---------------------------------------------------------------------------

def prune_unstable_adsorbate_sites(
    G: nx.Graph,
    adsorbate_sites: list[AdsorbateSite],
    reactant,
    calculator,
    *,
    frozen_indices: list[int] | None = None,
    fmax: float = PRUNE_FMAX,
    max_steps: int = PRUNE_MAX_STEPS,
    nl_mult: float = NL_MULT_DEFAULT,
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS,
    diagnostics_dir: str | None = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> list[AdsorbateSite]:
    """Remove iso-classes whose representative placement is unstable under ML relaxation.

    For each iso-class in *adsorbate_sites*:

    1. Build a bare slab + adsorbate :class:`~ase.Atoms` from the
       representative positions (``ms.positions``) — see
       :func:`_build_pruning_atoms`.  The adsorbate atoms are tagged
       ``surface == 2`` so :func:`autokmc.core.graph.build_graph` can be called
       on it directly.
    2. Run a potential-driven rigid-body optimization with the slab fixed and
       the adsorbate's internal coordinates held exactly constant.  The only
       degrees of freedom are molecular translation and rotation.
    3. Starting from that constrained minimum, run the existing full ML
       relaxation via :func:`~autokmc.structure.optimise_structure`.  The
       configured slab constraints are restored for this stage and the
       adsorbate's internal coordinates may relax.
    4. Build the graph of the relaxed structure (same NL cutoff as the rest
       of the package) and compare its **adsorbate-touching edge set** to
       the *intended* edge set:

         * intramolecular bonds copied from ``reactant.graph``;
         * one anchor bond per (reactant atom, surface clique member) for
           every reactant atom that ``ms.atom_cliques`` says is bonded.

       Any mismatch — missing intended edge OR unintended new edge touching
       the adsorbate (dissociation, hop to a different clique, gained or
       lost anchor bond, intramolecular bond broken) — prunes the iso-class.
       Slab–slab edges are *ignored* because un-frozen metal atoms relax by
       O(0.01–0.1 Å) under a real ML potential and can flip pairs across
       the natural-cutoff threshold without affecting the adsorbate.
    5. Both stages must converge.  The rigid stage applies *fmax* to the net
       translational force and length-scaled torque; the relaxed stage applies
       it to the un-frozen atomic forces.  Non-converged classes are pruned.
    6. For surviving iso-classes the relaxed adsorbate positions are
       written back to ``ms.positions`` and propagated to every member through
       a validated adsorption-coordination graph mapping.

    ``G.graph["adsorbate_sites"][reactant.smiles]`` is updated in place
    with the surviving list.

    Parameters
    ----------
    G : nx.Graph
        Surface graph from :func:`~autokmc.core.graph.build_graph`.
    adsorbate_sites : list[AdsorbateSite]
        Iso-classes to check (typically the return value of
        :func:`find_adsorbate_sites`).
    reactant : Reactant
        Gas-phase molecule that produced *adsorbate_sites*.
    calculator
        Any ASE-compatible ML/empirical potential or CalculatorPool.  A
        calculator is acquired for each relaxation; calculator instances are
        never deep-copied.
    frozen_indices : list[int] | None
        0-based indices into the **slab** portion (bulk + surface nodes
        sorted by their original ASE atom ``index``) to freeze during
        relaxation.  Pass ``atoms.info["frozen_indices"]`` from the slab
        directly.
    fmax : float
        Force convergence threshold (eV/Å) used for both the rigid and fully
        relaxed stages.  Default :data:`~autokmc.core.constants.PRUNE_FMAX`.
    max_steps : int
        Maximum optimizer steps for each of the two stages.  Default
        :data:`~autokmc.core.constants.PRUNE_MAX_STEPS`.
    nl_mult : float
        Neighbour-list cutoff multiplier handed to
        :func:`autokmc.core.graph.build_graph` for the connectivity comparison.
        Default :data:`~autokmc.core.constants.NL_MULT_DEFAULT` — matches the
        cutoff used everywhere else in the package.
    kabsch_max_mappings : int
        Maximum number of graph-isomorphism mappings tested when propagating
        a relaxed representative to equivalent members.  Default
        :data:`~autokmc.core.constants.KABSCH_MAX_MAPPINGS`.
    diagnostics_dir : str | None
        Run diagnostics directory. Rejected MLIP-pruning structures are
        written below ``invalid_adsorption`` when supplied.
    verbose : bool
        Print whether each iso-class is stable or pruned to stdout.

    Returns
    -------
    list[AdsorbateSite]
        Only the stable iso-classes, in their original order.  The
        ``iso_class`` integer labels are **not** renumbered.
    """
    from autokmc.structure import (  # avoid circular at module level
        StructureOptimisationError,
        optimise_structure,
    )
    from autokmc.core.graph import build_graph
    from autokmc.io.atoms import copy_atoms_with_results
    from autokmc.io.persistence import write_invalid_adsorption_diagnostic

    react_sym = list(reactant.atoms.get_chemical_symbols())
    stable: list[AdsorbateSite] = []
    n_pruned = 0
    resolved_diagnostics_dir = (
        diagnostics_dir or G.graph.get("diagnostics_dir")
    )

    def _persist_invalid(
        site: AdsorbateSite,
        atoms_initial,
        atoms_optimized,
        reason: str,
        **details,
    ) -> None:
        if resolved_diagnostics_dir is None:
            return
        try:
            write_invalid_adsorption_diagnostic(
                resolved_diagnostics_dir,
                site,
                reactant_smiles=reactant.smiles,
                atoms_initial=atoms_initial,
                atoms_optimized=atoms_optimized,
                invalid_reason=reason,
                details=details,
            )
        except Exception as exc:
            _log.warning(
                "prune_unstable_adsorbate_sites: failed to persist invalid "
                "adsorption iso_class=%d (%s)",
                site.iso_class,
                exc,
            )

    if verbose:
        print(
            f"\nprune_unstable_adsorbate_sites: {len(adsorbate_sites)} iso-class(es)  "
            f"rigid + relaxed potential stages  fmax={fmax} eV/Å  "
            f"max_steps/stage={max_steps}"
        )

    for ms in adsorbate_sites:
        # First, build the initial tagged structure and its node-to-atom map.
        try:
            atoms_init, n_slab, n_ads, node_to_ase = _build_pruning_atoms(
                G, ms, react_sym, frozen_indices=frozen_indices,
            )
        except Exception as exc:
            _log.warning(
                "prune_unstable_adsorbate_sites: iso_class=%d build failed (%s) "
                "— keeping.", ms.iso_class, exc,
            )
            if verbose:
                print(f"  ? iso={ms.iso_class}: build failed ({exc}) — kept")
            stable.append(ms)
            continue

        intended_edges = _intended_adsorbate_edges(
            ms, reactant, n_slab, node_to_ase
        )

        # First optimize only the molecule's rigid translation and rotation
        # against the potential while leaving every slab atom stationary.
        # Then restore the ordinary atom-level degrees of freedom and slab
        # constraints for the existing relaxed stability check.
        optimization_stage = "rigid"
        rigid_max_force: float | None = None
        rigid_steps: int | None = None
        try:
            with acquire_calculator(
                calculator, purpose="adsorbate-site pruning"
            ) as calc:
                atoms_rigid, rigid_max_force, rigid_steps = (
                    _optimise_rigid_adsorbate_with_potential(
                        atoms_init,
                        n_slab,
                        calc,
                        fmax=fmax,
                        max_steps=max_steps,
                        optimizer=optimizer,
                        optimizer_kwargs=optimizer_kwargs,
                    )
                )
                optimization_stage = "relaxed"
                atoms_opt = optimise_structure(
                    atoms_rigid,
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
                E = float(atoms_opt.get_potential_energy())
                atoms_opt = copy_atoms_with_results(
                    atoms_opt,
                    energy=E,
                    forces=forces,
                )
        except Exception as exc:
            if isinstance(exc, CalculatorConfigError):
                raise
            failed_atoms = (
                exc.atoms
                if isinstance(exc, StructureOptimisationError)
                else None
            )
            did_not_converge = (
                isinstance(exc, StructureOptimisationError)
                and exc.converged is False
            )
            if optimization_stage == "rigid":
                invalid_reason = (
                    "rigid_not_converged"
                    if did_not_converge
                    else "rigid_relaxation_failed"
                )
            else:
                # Preserve the established diagnostics contract for the
                # pre-existing fully relaxed stage.
                invalid_reason = (
                    "not_converged"
                    if did_not_converge
                    else "relaxation_failed"
                )
            _log.debug(
                "prune_unstable_adsorbate_sites: iso_class=%d relaxation raised %s",
                ms.iso_class, exc,
            )
            if verbose:
                print(
                    f"  PRUNED iso={ms.iso_class}: {optimization_stage} "
                    f"optimization failed ({exc})"
                )
            _persist_invalid(
                ms,
                atoms_init,
                failed_atoms,
                invalid_reason,
                error=f"{type(exc).__name__}: {exc}",
                optimization_stage=optimization_stage,
                rigid_max_force_ev_per_ang=rigid_max_force,
                rigid_optimizer_steps=rigid_steps,
                optimizer_steps=(
                    exc.steps
                    if isinstance(exc, StructureOptimisationError)
                    else None
                ),
            )
            n_pruned += 1
            _remove_iso_class_nodes(G, ms)
            continue

        assert rigid_max_force is not None
        assert rigid_steps is not None

        # After relaxation, reject structures that did not reach the requested
        # force threshold.
        if max_force > fmax:
            if verbose:
                print(
                    f"  PRUNED iso={ms.iso_class}: not converged "
                    f"(max|F|={max_force:.4f} eV/Å > {fmax})"
                )
            _persist_invalid(
                ms,
                atoms_init,
                atoms_opt,
                "not_converged",
                max_force_ev_per_ang=float(max_force),
                fmax_ev_per_ang=float(fmax),
                energy_ev=float(E),
                rigid_max_force_ev_per_ang=float(rigid_max_force),
                rigid_optimizer_steps=int(rigid_steps),
            )
            n_pruned += 1
            _remove_iso_class_nodes(G, ms)
            continue

        # A converged structure must also preserve the intended connectivity.
        # Carry the surface tags into the relaxed graph, and compare every edge
        # that touches the adsorbate with the intended edge set. Missing bonds
        # and new bonds to atoms outside the intended clique both cause the
        # iso-class to be pruned.
        atoms_for_graph = atoms_opt.copy()
        atoms_for_graph.arrays["surface"] = atoms_init.arrays["surface"]
        try:
            G_relaxed = build_graph(atoms_for_graph, nl_mult=nl_mult)
        except Exception as exc:
            _log.warning(
                "prune_unstable_adsorbate_sites: iso_class=%d build_graph "
                "failed on relaxed atoms (%s) — pruned.",
                ms.iso_class, exc,
            )
            if verbose:
                print(
                    f"  PRUNED iso={ms.iso_class}: build_graph(relaxed) failed "
                    f"({exc})"
                )
            _persist_invalid(
                ms,
                atoms_init,
                atoms_opt,
                "relaxed_graph_failed",
                error=f"{type(exc).__name__}: {exc}",
                max_force_ev_per_ang=float(max_force),
                energy_ev=float(E),
                rigid_max_force_ev_per_ang=float(rigid_max_force),
                rigid_optimizer_steps=int(rigid_steps),
            )
            n_pruned += 1
            _remove_iso_class_nodes(G, ms)
            continue

        relaxed_edges = _adsorbate_edges_from_graph(G_relaxed, n_slab)

        if relaxed_edges != intended_edges:
            missing = intended_edges - relaxed_edges
            extra   = relaxed_edges - intended_edges
            if verbose:
                def _fmt(e: frozenset) -> str:
                    a, b = sorted(int(x) for x in e)
                    return f"({a},{b})"
                miss_s = ", ".join(_fmt(e) for e in list(missing)[:3])
                extr_s = ", ".join(_fmt(e) for e in list(extra)[:3])
                print(
                    f"  PRUNED iso={ms.iso_class}: adsorbate connectivity changed "
                    f"(missing={len(missing)} [{miss_s}"
                    f"{'…' if len(missing) > 3 else ''}], "
                    f"extra={len(extra)} [{extr_s}"
                    f"{'…' if len(extra) > 3 else ''}]) — pruned"
                )
            _persist_invalid(
                ms,
                atoms_init,
                atoms_opt,
                "connectivity_changed",
                missing_edge_count=len(missing),
                extra_edge_count=len(extra),
                max_force_ev_per_ang=float(max_force),
                energy_ev=float(E),
                rigid_max_force_ev_per_ang=float(rigid_max_force),
                rigid_optimizer_steps=int(rigid_steps),
            )
            n_pruned += 1
            _remove_iso_class_nodes(G, ms)
            continue

        # The relaxed adsorbate becomes the representative for this iso-class.
        # Project its coordinates back into the live graph frame, and preserve
        # the reactant atom order established when the structure was built.
        frame_depth = max(1, int(ms.n_shells_settled))
        new_pos: np.ndarray = _relaxed_adsorbate_positions_in_graph_frame(
            G,
            ms,
            atoms_opt,
            n_slab,
            n_ads,
            node_to_ase,
            frame_depth=frame_depth,
        )
        cell_store = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
        pbc_store = _effective_pbc(G, cell_store)
        new_pos = _wrap_adsorbate_positions_for_storage(
            new_pos, ms.atom_cliques, cell_store, pbc_store,
        )
        # Finally, propagate the representative geometry to the other members
        # with the complete adsorption-coordination mapping. Build and validate
        # every pose before mutating graph coordinates so a failure cannot leave
        # a partially updated iso-class.
        n_propagated = 0
        if ms.member_node_ids:
            propagation_context = _prepare_geometry_connectivity_context(
                G,
                reactant,
                nl_mult=float(nl_mult),
            )
            if _geometry_connectivity_mismatch_for_cliques(
                G,
                ms.atom_cliques,
                reactant,
                new_pos,
                nl_mult=nl_mult,
                _context=propagation_context,
            ) is not None:
                raise RuntimeError(
                    "prune_unstable_adsorbate_sites: projected representative "
                    f"connectivity is invalid for iso-class {ms.iso_class}"
                )
            pending_positions = [new_pos]
            if len(ms.members) > 1:
                propagation_depth = int(ms.coordination_n_shells)
                for m_idx in range(1, len(ms.members)):
                    try:
                        member_pos = _propagate_adsorbate_member_positions(
                            G,
                            ms.atom_cliques,
                            ms.members[m_idx],
                            new_pos,
                            reactant,
                            n_shells=propagation_depth,
                            nl_mult=nl_mult,
                            max_mappings=kabsch_max_mappings,
                            _context=propagation_context,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "prune_unstable_adsorbate_sites: decorated "
                            f"propagation failed for iso-class {ms.iso_class}, "
                            f"member {m_idx}"
                        ) from exc
                    pending_positions.append(member_pos)
            for member_index, member_pos in enumerate(pending_positions):
                push_member_positions_to_graph(
                    G,
                    ms,
                    member_index,
                    member_pos,
                )
            n_propagated = len(pending_positions) - 1
        ms.positions = new_pos

        if verbose:
            suffix = (
                f"propagated {n_propagated}/{max(0, len(ms.members) - 1)} members"
                if ms.member_node_ids
                else "no members yet"
            )
            print(
                f"  STABLE iso={ms.iso_class}: "
                f"rigid max|F|={rigid_max_force:.4f} eV/Å "
                f"({rigid_steps} steps); relaxed E={E:.4f} eV  "
                f"max|F|={max_force:.4f} eV/Å  {suffix}"
            )

        stable.append(ms)

    if verbose:
        print(
            f"  Pruning complete: {len(stable)}/{len(adsorbate_sites)} "
            f"iso-class(es) survived  ({n_pruned} pruned)"
        )

    _log.debug(
        "prune_unstable_adsorbate_sites: %r  %d/%d iso-classes survived",
        reactant.smiles, len(stable), len(adsorbate_sites),
    )

    # Store the surviving sites so the graph and the returned list agree.
    G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = stable
    rebuild_adsorbate_reverse_indexes(G)
    return stable


# ---------------------------------------------------------------------------
# push_member_positions_to_graph
# ---------------------------------------------------------------------------

def push_member_positions_to_graph(
    G: nx.Graph,
    adsorbate_site: AdsorbateSite,
    member_index: int,
    positions: np.ndarray,
) -> None:
    """Write *positions* for one :class:`AdsorbateSite` member into *G*.

    Updates ``G.nodes[nid]["position"]`` for every node in
    ``adsorbate_site.member_node_ids[member_index]`` and refreshes all
    incident edge distances (``intra_adsorbate`` and ``anchor_bond``).

    Parameters
    ----------
    G : nx.Graph
    adsorbate_site : AdsorbateSite
    member_index : int
        Index into ``adsorbate_site.member_node_ids``.
    positions : np.ndarray, shape (n_atoms, 3)
    """
    if member_index >= len(adsorbate_site.member_node_ids):
        return
    node_ids = adsorbate_site.member_node_ids[member_index]
    pos_arr  = np.asarray(positions, dtype=float)
    if pos_arr.shape[0] != len(node_ids):
        raise ValueError(
            f"positions has {pos_arr.shape[0]} rows but member "
            f"{member_index} has {len(node_ids)} nodes."
        )
    cell = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc = _effective_pbc(G, cell)
    if pbc.any():
        pos_arr = wrap_positions_into_cell(
            pos_arr, cell, pbc, reference=pos_arr[0],
        )

    for i, nid in enumerate(node_ids):
        if nid not in G:
            continue
        G.nodes[nid]["position"]  = pos_arr[i].copy()
        G.nodes[nid]["optimised"] = True
    try:
        cell_inv = np.linalg.inv(cell) if np.any(pbc) else None
    except np.linalg.LinAlgError:
        cell_inv = None

    # Refresh all incident edge distances.
    for nid_a in node_ids:
        if nid_a not in G:
            continue
        p_a = np.asarray(G.nodes[nid_a]["position"], dtype=float)
        for nid_b in G.neighbors(nid_a):
            if nid_b not in G:
                continue
            p_b = np.asarray(G.nodes[nid_b]["position"], dtype=float)
            edge = G.edges[nid_a, nid_b]
            offset = np.asarray(edge.get("offset", (0, 0, 0)), dtype=float)
            if int(nid_a) > int(nid_b):
                offset = -offset
            dv = p_b + offset @ cell - p_a
            if not np.any(offset) and cell_inv is not None:
                dv = minimum_image_vectors(dv, cell, pbc)
            edge["distance"] = float(np.linalg.norm(dv))


# ---------------------------------------------------------------------------
# Public API — find_adsorbate_sites
# ---------------------------------------------------------------------------

def find_adsorbate_sites(
    G: nx.Graph,
    reactant,
    *,
    bond_tolerance: float = BOND_TOLERANCE,
    n_shells_anchor: int | None = None,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    anchor_k_max: int | None = None,
    co_factor: float = CO_FACTOR,
    opt_factor: float = OPT_FACTOR,
    repulsion_weight: float = REPULSION_WEIGHT,
    repulsion_cutoff: float | None = REPULSION_CUTOFF,
    contact_factor: float = CONTACT_FACTOR,
    standoff_factor: float = STANDOFF_FACTOR,
    n_restarts: int = N_RESTARTS,
    nn_distance: float = NN_DISTANCE,
    include_partial: bool = True,
    require_anchors: bool = True,
    auto_grow_shells: bool = True,
    max_shell_retries: int = 3,
    require_surface_connected: bool = True,
    max_pair_shells: int = MAX_PAIR_SHELLS,
    hull_tolerance: float = HULL_TOL,
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS,
    nl_mult: float = NL_MULT_DEFAULT,
    # If requested, prune the sites that do not remain stable.
    prune_stable_only: bool = True,
    calculator=None,
    frozen_indices: list[int] | None = None,
    prune_fmax: float = PRUNE_FMAX,
    prune_max_steps: int = PRUNE_MAX_STEPS,
    diagnostics_dir: str | None = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> list[AdsorbateSite]:
    """Universal N-atom adsorbate site enumerator (N ≥ 1).

    Single-atom reactants (``len(reactant.atoms) == 1``) are handled as a
    degenerate fast path: the (single) anchor atom is placed at every raw
    anchor-node position of its element, iso-classes are deduplicated by
    ego-graph isomorphism around the bonded surface clique, and the
    calculator-free rigid-body refinement is skipped (a single atom has no
    rotational DOF and the anchor-site position is already the correct
    initial geometry).  ML stability pruning still runs when *calculator*
    is supplied.

    Parameters
    ----------
    G : nx.Graph
        Surface graph from :func:`autokmc.core.graph.build_graph`.  Anchor sites
        for every anchor element are lazily computed if not already on *G*.
    reactant : :class:`autokmc.species.reactant.Reactant`
        Gas-phase molecule.  ``len(reactant.atoms) >= 2`` is required.
    bond_tolerance : float
        Allowed Å deviation between any surface anchor-pair distance and the
        corresponding intramolecular distance.  Default 0.4 Å.
    n_shells_anchor : int or None
        Shell depth for reducing single-atom sites into iso-classes (used to
        seed the *first* anchor of each subset).  ``None`` picks
        ``max(1, ceil(reach / 2.5 Å))`` automatically.
    n_shells_pair : int
        Minimum ego-graph depth for iso-class discrimination of adsorbate
        placements.  Default 1.
    anchor_k_max : int or None
        Optional hard cap on the coordination size enumerated for each
        surface anchor element.  ``None`` preserves natural enumeration.
    co_factor, opt_factor, repulsion_weight, repulsion_cutoff :
        Forwarded to :func:`~autokmc.sites.anchors.find_anchor_sites` if
        anchor sites have not yet been computed for an element.
        *repulsion_cutoff* is also used by the rigid-body refinement.
    contact_factor : float
        Minimum adsorbate–surface contact scale used by rigid-body
        refinement.  Default
        :data:`~autokmc.core.constants.CONTACT_FACTOR`.
    standoff_factor : float
        Bonded-anchor standoff scale used by rigid-body refinement.  Default
        :data:`~autokmc.core.constants.STANDOFF_FACTOR`.
    n_restarts : int
        Number of rotational restarts used by rigid-body refinement.  Default
        :data:`~autokmc.core.constants.N_ADSORBATE_RESTARTS`.
    nn_distance : float
        Typical surface nearest-neighbour distance (Å) used to choose
        *n_shells_anchor* when it is ``None``.  Default
        :data:`~autokmc.core.constants.NN_DISTANCE`.
    include_partial : bool
        Enumerate all non-empty anchor subsets (including singletons).
        Set to ``False`` to try only the full-anchor subset.  Default True.
    require_anchors : bool
        Raise ``ValueError`` when the reactant has no anchor atoms.
    auto_grow_shells : bool
        Re-enumerate at a larger ``n_shells_anchor`` if any emitted placement
        spans more hops than the current ego depth (up to *max_shell_retries*).
    max_shell_retries : int
        Maximum grow-retry attempts.  Default 3.
    require_surface_connected : bool
        Drop placements whose bonded cliques cannot reach each other through
        surface edges within *max_pair_shells* hops.  Default True.
    max_pair_shells : int
        Hard cap on the per-placement connectivity check.  Default 10.
    hull_tolerance : float
        Signed-distance tolerance (Å) forwarded to anchor-site enumeration
        when filtering buried cliques.  Default
        :data:`~autokmc.core.constants.HULL_TOL`.
    kabsch_max_mappings : int
        Maximum graph-isomorphism mappings tested during each Kabsch
        propagation.  Default
        :data:`~autokmc.core.constants.KABSCH_MAX_MAPPINGS`.
    nl_mult : float
        Neighbour-list cutoff multiplier used by calculator-free and
        post-relaxation connectivity checks.  Default
        :data:`~autokmc.core.constants.NL_MULT_DEFAULT`.
    prune_stable_only : bool
        If ``True`` (default), optimize every iso-class representative against
        the ML potential first as an exact rigid molecule on a fixed slab and
        then with the ordinary atomic degrees of freedom.  Discard any class
        whose bond topology changes or whose optimization does not converge.
        Requires *calculator* to be set; if *calculator* is ``None`` a
        :class:`RuntimeWarning` is issued and pruning is skipped.
    calculator
        ASE-compatible ML/empirical potential or CalculatorPool used for
        stability pruning.  Calculators are acquired rather than deep-copied.
        ``None`` disables pruning even when *prune_stable_only* is ``True``.
    frozen_indices : list[int] | None
        Atom indices into the **slab** portion (0-based, sorted by original
        ASE atom ``index``) to freeze during the pruning relaxation.  Pass
        ``atoms.info["frozen_indices"]`` directly for slab structures.
    prune_fmax : float
        Force convergence threshold (eV/Å) for each of the rigid and relaxed
        pruning stages.
        Default :data:`~autokmc.core.constants.PRUNE_FMAX` (0.05).
    prune_max_steps : int
        Maximum optimizer steps for each pruning stage.
        Default :data:`~autokmc.core.constants.PRUNE_MAX_STEPS` (500).
    diagnostics_dir : str | None
        Run diagnostics directory. Adsorption candidates rejected by MLIP
        pruning are written below ``invalid_adsorption`` when supplied.
    verbose : bool
        Print per-step progress to stdout.

    Returns
    -------
    list[AdsorbateSite]
        One entry per iso-class.  Also stored at
        ``G.graph["adsorbate_sites"][reactant.smiles]``.
    """
    if anchor_k_max is not None and int(anchor_k_max) < 1:
        raise ValueError("anchor_k_max must be at least 1 when supplied")
    n_atoms = len(reactant.atoms)
    if n_atoms < 1:
        raise ValueError(
            f"find_adsorbate_sites requires ≥ 1 atom, got {n_atoms}."
        )

    anchors = sorted(int(i) for i in reactant.anchor_atoms)
    if not anchors:
        _remove_adsorbate_nodes(G, reactant.smiles)
        G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = []
        rebuild_adsorbate_reverse_indexes(G)
        if require_anchors:
            raise ValueError("Reactant has no anchor atoms.")
        return []

    # First, remove adsorbate nodes left by an earlier call for this SMILES.
    _remove_adsorbate_nodes(G, reactant.smiles)

    elements: dict[int, str] = {
        i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)
    }
    react_pos = np.asarray(reactant.atoms.get_positions(), dtype=float)
    # This matrix contains every intramolecular atom-pair distance.
    D = np.linalg.norm(
        react_pos[:, None, :] - react_pos[None, :, :], axis=-1
    )

    # If no shell depth was given, estimate it from the molecular reach.
    if n_shells_anchor is None:
        if len(anchors) >= 2:
            reach = float(max(D[i, j] for i in anchors for j in anchors if i < j))
        else:
            reach = float(D[anchors[0]].max())
        n_shells_eff = _suggested_n_shells(
            reach,
            nn_distance=nn_distance,
        )
    else:
        n_shells_eff = int(n_shells_anchor)

    cell, cell_inv, pbc, use_mic = _get_cell(G)
    orbit_id   = _orbit_id_of(reactant)
    node_match = isomorphism.categorical_node_match(
        ["coordination_role", "element"],
        ["substrate", "X"],
    )
    edge_match = isomorphism.categorical_edge_match(
        "coordination_kind",
        "substrate",
    )
    apsp = (
        _get_surface_apsp(G, cutoff=max_pair_shells)
        if require_surface_connected else None
    )

    if verbose:
        print(
            f"\nfind_adsorbate_sites: smiles={reactant.smiles!r}  "
            f"N={n_atoms}  anchors={anchors}  "
            f"n_shells_anchor={n_shells_eff}  n_shells_pair={n_shells_pair}  "
            f"tol={bond_tolerance} Å"
        )

    # Next, reduce the anchor subsets by their molecular symmetry orbits.
    if include_partial:
        all_subsets: list[tuple[int, ...]] = []
        for r in range(1, len(anchors) + 1):
            all_subsets.extend(combinations(anchors, r))
    else:
        all_subsets = [tuple(anchors)]

    seen_subset_keys: set = set()
    canonical_subsets: list[tuple[int, ...]] = []
    for sub in all_subsets:
        if not _anchor_subset_allowed(sub, orbit_id):
            continue
        key = _canonical_subset_key(sub, orbit_id)
        if key not in seen_subset_keys:
            seen_subset_keys.add(key)
            canonical_subsets.append(sub)

    # The enumeration pass builds candidate placements at the selected depth.
    def _run_pass(depth: int) -> list[AdsorbateSite]:
        anchor_elements = {elements[i] for i in anchors}
        for el in anchor_elements:
            _ensure_anchor_sites(
                G, el,
                co_factor=co_factor,
                opt_factor=opt_factor,
                repulsion_weight=repulsion_weight,
                repulsion_cutoff=repulsion_cutoff,
                n_shells=depth,
                anchor_k_max=anchor_k_max,
                hull_tolerance=hull_tolerance,
                kabsch_max_mappings=kabsch_max_mappings,
                verbose=verbose,
            )

        raw_by_elem: dict[str, list[tuple[frozenset, np.ndarray]]] = {
            el: _all_raw_sites_with_positions(G, el) for el in anchor_elements
        }
        spatial_cutoff = max(
            0.0,
            float(D.max(initial=0.0)) + float(bond_tolerance),
        )
        spatial_by_elem = {
            el: _build_anchor_spatial_index(
                candidates,
                cell,
                pbc,
                use_mic,
                cutoff=spatial_cutoff,
            )
            for el, candidates in raw_by_elem.items()
        }

        adsorbate_sites: list[AdsorbateSite] = []
        seen_signatures: set = set()
        rejected_disconnected = 0

        def _emit(
            bonded: list[int],
            assigned_pos: dict[int, np.ndarray],
            assigned_clique: dict[int, frozenset],
        ) -> None:
            nonlocal rejected_disconnected
            atom_cliques: list = [None] * n_atoms
            for i in bonded:
                atom_cliques[i] = assigned_clique[i]

            # Reject a placement when its bonded cliques are too far apart on
            # the surface graph.
            if apsp is not None:
                bonded_cliques = [c for c in atom_cliques if c is not None]
                needed = _min_ego_depth_for_connectivity(bonded_cliques, apsp)
                if needed >= 10 ** 6 or needed > max_pair_shells:
                    rejected_disconnected += 1
                    return

            bonded_positions = np.array(
                [assigned_pos[i] for i in bonded], dtype=float
            )
            positions = _full_adsorbate_positions(
                reactant, bonded, bonded_positions, G, pbc
            )

            # Build the local ego graph with a breadth-first search from the
            # union of the bonded cliques.
            union: set[int] = set()
            for c in atom_cliques:
                if c is not None:
                    union |= set(c)
            substrate_ego = _build_ego_graph(
                G,
                frozenset(union),
                n_shells_pair,
            )
            ego = _build_adsorbate_coordination_graph(
                substrate_ego,
                atom_cliques,
                reactant.graph,
            )

            _try_merge_or_new(
                adsorbate_sites,
                reactant_smiles = reactant.smiles,
                atom_cliques    = atom_cliques,
                positions       = positions,
                ego_graph       = ego,
                node_match      = node_match,
                edge_match      = edge_match,
                seen_signatures = seen_signatures,
            )

        def _recurse(
            bonded: list[int],
            idx: int,
            assigned_pos: dict[int, np.ndarray],
            assigned_clique: dict[int, frozenset],
        ) -> None:
            """Backtracking placement with a MIC-safe spatial prefilter."""
            if idx == len(bonded):
                _emit(bonded, assigned_pos, assigned_clique)
                return
            next_atom = bonded[idx]
            next_el   = elements[next_atom]
            candidates = raw_by_elem.get(next_el, [])

            candidate_indices: list[int] | None = None
            spatial_index = spatial_by_elem.get(next_el)
            if spatial_index is not None and assigned_pos:
                for prev_idx, p_prev in assigned_pos.items():
                    target = float(D[prev_idx, next_atom])
                    hits = _candidate_indices_in_annulus(
                        spatial_index,
                        p_prev,
                        target,
                        bond_tolerance,
                    )
                    if candidate_indices is None or len(hits) < len(candidate_indices):
                        candidate_indices = hits
                    if not hits:
                        break

            if candidate_indices is None:
                candidate_iter = candidates
            else:
                candidate_iter = (candidates[i] for i in candidate_indices)

            for clique, pos in candidate_iter:
                # No two adsorbate atoms may share the same surface clique.
                if any(clique == c for c in assigned_clique.values()):
                    continue

                # Verify the MIC distance to every prior placed anchor.
                ok = True
                for prev_idx, p_prev in assigned_pos.items():
                    target = float(D[prev_idx, next_atom])
                    d = _mic_distance(p_prev, pos, cell, cell_inv, pbc, use_mic)
                    if abs(d - target) > bond_tolerance:
                        ok = False
                        break
                if not ok:
                    continue

                assigned_pos[next_atom]    = pos
                assigned_clique[next_atom] = clique
                _recurse(bonded, idx + 1, assigned_pos, assigned_clique)
                del assigned_pos[next_atom]
                del assigned_clique[next_atom]

        if verbose:
            print(
                f"  [depth={depth}]  {len(canonical_subsets)} canonical subsets "
                f"(from {len(all_subsets)} raw)"
            )

        for sub in canonical_subsets:
            bonded   = sorted(sub)
            first    = bonded[0]
            first_el = elements[first]
            # Iterate over ALL raw anchor sites for the first atom so that
            # every surface site gets tried, not just the iso-class
            # representative.  Equivalent placements are still folded into
            # the same AdsorbateSite by _try_merge_or_new via isomorphism.
            # (Previously only iso_by_elem[first_el] was iterated, which
            # seeds only one position per iso-class and therefore misses all
            # non-representative members as the first anchor.)
            for clique, pos in raw_by_elem.get(first_el, []):
                assigned_pos    = {first: pos}
                assigned_clique = {first: clique}
                _recurse(bonded, 1, assigned_pos, assigned_clique)

        if verbose and rejected_disconnected:
            print(
                f"  [depth={depth}]  dropped {rejected_disconnected} "
                f"disconnected placement(s)  (max_pair_shells={max_pair_shells})"
            )

        return adsorbate_sites

    # Repeat the enumeration with a larger shell when the initial depth cannot
    # contain every bonded clique.
    adsorbate_sites = _run_pass(n_shells_eff)
    retries = 0
    while auto_grow_shells and apsp is not None and retries < max_shell_retries:
        needed = _required_n_shells(G, adsorbate_sites, apsp)
        if needed <= n_shells_eff:
            break
        if verbose:
            print(
                f"  WARNING bonded cliques span {needed} hops but iso depth was "
                f"{n_shells_eff} — growing to {needed} and re-enumerating"
            )
        n_shells_eff    = needed
        adsorbate_sites = _run_pass(n_shells_eff)
        retries += 1

    if verbose:
        n_full = sum(
            1 for ms in adsorbate_sites
            if all(ms.atom_cliques[i] is not None for i in anchors)
        )
        n_partial     = len(adsorbate_sites) - n_full
        total_members = sum(len(ms.members) for ms in adsorbate_sites)
        print(
            f"  → {len(adsorbate_sites)} iso-classes  "
            f"({n_full} full-anchor + {n_partial} partial)  "
            f"{total_members} placements total"
        )

    # First, record the selected shell depth and materialize the graph nodes.
    # Geometry optimization needs these nodes, so this happens before pruning.
    for ms in adsorbate_sites:
        ms.n_shells_settled = int(n_shells_eff)
        ms.coordination_n_shells = int(n_shells_pair)

    _materialise_adsorbate_nodes(G, reactant, adsorbate_sites)
    G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = adsorbate_sites

    # Next, build the reverse indexes used by the KMC update paths.
    # Each adsorbate node already stores its bonded surface clique as a
    # frozenset. The indexes below reuse that value directly.
    #
    #   G.graph["clique_to_members"]
    #       dict[frozenset, list[(AdsorbateSite, m_idx)]]
    #       Maps every bonded surface clique to every (site, member) pair
    #       that touches it.  Used by ``execute_reaction`` to find affected
    #       members in O(1) per affected clique instead of an O(sites *
    #       members) scan.
    #
    #   G.graph["occupied_by_clique"]
    #       dict[frozenset, set[node_id]]
    #       Per-clique set of currently-occupied adsorbate node ids.
    #       Maintained by ``_set_member_occupied`` so that
    #       ``is_clique_blocked`` is O(1) instead of an O(N_nodes) scan.
    #
    #   G.graph["n_occupied"]
    #       Running total of currently-occupied members across all sites,
    #       so the per-step KMC log line is O(1) instead of O(total members).
    #
    #   G.graph["surface_node_to_members"]
    #       dict[int, list[(AdsorbateSite, m_idx)]]
    #       Maps each individual surface atom id to every (site, member) pair
    #       bonded to any clique containing that atom.  Used by the KMC
    #       incremental update to find ALL members whose lateral ego-graph
    #       may include the toggled member — those within n_shells surface
    #       hops, not just clique-collision ones.
    #
    #   site._member_cliques[m_idx] : tuple[frozenset, ...]
    #       Cached per-member tuple of bonded-surface frozensets.  Avoids
    #       re-scanning the member's adsorbate node ids to rebuild the same
    #       frozensets every KMC step.
    rebuild_adsorbate_reverse_indexes(G)

    # After materialization, optimize the geometry, test its stability, prune
    # invalid sites, and propagate each representative.
    if prune_stable_only:
        if calculator is not None:
            # First, refine the representative with a calculator-free rigid-body
            # optimization. This gives the later calculator relaxation a useful
            # starting geometry instead of the raw clique centroids.
            #
            # Skipped for single-atom reactants: a 1-atom adsorbate has no
            # rotational DOF and the anchor-node position is already the
            # correct initial geometry; the rigid-body objective (and the
            # underlying Kabsch SVD) would also be degenerate on one point.
            if n_atoms >= 2:
                if verbose:
                    print(
                        "\n  ──────────────────────────────────────────────────────\n"
                        "  Stage B-1 : calc-free geometric optimisation\n"
                        "  ──────────────────────────────────────────────────────"
                    )
                adsorbate_sites = optimise_adsorbate_site_positions(
                    G,
                    reactant.smiles,
                    reactant,
                    repulsion_cutoff=repulsion_cutoff,
                    contact_factor=contact_factor,
                    standoff_factor=standoff_factor,
                    n_restarts=n_restarts,
                    n_shells_pair=n_shells_pair,
                    nl_mult=nl_mult,
                    kabsch_max_mappings=kabsch_max_mappings,
                    verbose=verbose,
                )
                G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = adsorbate_sites
            elif verbose:
                print(
                    "\n  Stage B-1 skipped (single-atom reactant — no rigid-body DOF)"
                )

            # Then use the configured calculator for a rigid molecular
            # optimization followed by the ordinary atom-level relaxation.
            # This prunes unstable iso-classes and propagates each relaxed
            # representative to its remaining members.
            if verbose:
                print(
                    "\n  ──────────────────────────────────────────────────────\n"
                    "  Stage B-2/3 : rigid + relaxed ML stability check\n"
                    "  ──────────────────────────────────────────────────────"
                )
            adsorbate_sites = prune_unstable_adsorbate_sites(
                G, adsorbate_sites, reactant, calculator,
                frozen_indices = frozen_indices,
                fmax           = prune_fmax,
                max_steps      = prune_max_steps,
                nl_mult        = nl_mult,
                kabsch_max_mappings=kabsch_max_mappings,
                diagnostics_dir=diagnostics_dir,
                optimizer       = optimizer,
                optimizer_kwargs=optimizer_kwargs,
                verbose        = verbose,
            )
            # prune_unstable_adsorbate_sites already updates G.graph; keep
            # the local variable consistent.
            G.graph.setdefault("adsorbate_sites", {})[reactant.smiles] = adsorbate_sites
        else:
            warnings.warn(
                "find_adsorbate_sites: prune_stable_only=True but calculator=None "
                "— stability pruning skipped.  Pass calculator=<your_calc> to "
                "enable pruning, or set prune_stable_only=False to suppress this "
                "warning.",
                RuntimeWarning,
                stacklevel=2,
            )


    _log.debug(
        "find_adsorbate_sites: %r done — %d iso-classes, %d total placements",
        reactant.smiles,
        len(adsorbate_sites),
        sum(len(ms.members) for ms in adsorbate_sites),
    )
    return adsorbate_sites


# ---------------------------------------------------------------------------
# Rigid-body refinement
# ---------------------------------------------------------------------------

def _skew_matrix(vector: np.ndarray) -> np.ndarray:
    """Return the matrix whose product with ``x`` is ``vector × x``."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def _rotation_from_axis_angle(rotvec: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix from a Rodrigues axis-angle vector."""
    rotvec = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rotvec))
    skew = _skew_matrix(rotvec)
    if theta < 1e-6:
        theta2 = theta * theta
        return (
            np.eye(3)
            + (1.0 - theta2 / 6.0) * skew
            + (0.5 - theta2 / 24.0) * (skew @ skew)
        )
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * skew
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (skew @ skew)
    )


def _rotation_right_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """Right Jacobian of the SO(3) exponential map for an axis-angle vector."""
    rotvec = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rotvec))
    skew = _skew_matrix(rotvec)
    if theta < 1e-6:
        # Stable series through O(theta²).
        return np.eye(3) - 0.5 * skew + (skew @ skew) / 6.0
    theta2 = theta * theta
    return (
        np.eye(3)
        - ((1.0 - np.cos(theta)) / theta2) * skew
        + ((theta - np.sin(theta)) / (theta2 * theta)) * (skew @ skew)
    )


def _surface_atoms_array(
    G: nx.Graph,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return ``(positions[N,3], r_cov[N], node_ids[N])`` for surface nodes.

    Cached on ``G.graph["_surface_atoms_array_cache"]`` keyed by the
    sorted tuple of surface node ids so that repeated calls within one
    optimisation pass don't rebuild the (potentially huge) array.
    Invalidate by clearing the cache or by adding/removing surface nodes
    (the key changes automatically).
    """
    from ase.data import covalent_radii as _RC, atomic_numbers as _AN

    surf_ids = tuple(sorted(
        n for n, d in G.nodes(data=True) if d.get("type") == "surface"
    ))
    cache = G.graph.get("_surface_atoms_array_cache")
    if isinstance(cache, dict) and cache.get("_key") == surf_ids:
        return cache["pos"], cache["r_cov"], list(cache["ids"])

    pos:   list = []
    r_cov: list = []
    for n in surf_ids:
        d = G.nodes[n]
        pos.append(d["position"])
        r_cov.append(float(_RC[_AN[d["element"]]]))
    pos_arr   = np.asarray(pos,   dtype=float).reshape(-1, 3)
    r_cov_arr = np.asarray(r_cov, dtype=float)
    G.graph["_surface_atoms_array_cache"] = {
        "_key": surf_ids,
        "pos":  pos_arr,
        "r_cov": r_cov_arr,
        "ids":  list(surf_ids),
    }
    return pos_arr, r_cov_arr, list(surf_ids)


def optimise_adsorbate_site_positions(
    G: nx.Graph,
    smiles: str,
    reactant,
    *,
    restraint_weight: float = 10.0,
    repulsion_weight: float = 1.0,
    repulsion_cutoff: float | None = REPULSION_CUTOFF,
    contact_factor: float = CONTACT_FACTOR,
    standoff_factor: float = STANDOFF_FACTOR,
    n_restarts: int = N_RESTARTS,
    try_flip: bool = True,
    max_connectivity_attempts: int = 3,
    max_iter: int = 100,
    n_shells_pair: int = N_SHELLS_DEFAULT,
    nl_mult: float = NL_MULT_DEFAULT,
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS,
    verbose: bool = False,
) -> list[AdsorbateSite]:
    """Rigid-body refinement of every :attr:`AdsorbateSite.positions` for
    ``G.graph["adsorbate_sites"][smiles]``.

    Treats each placement as 6 rigid-body DOF (3 translation + 3 axis-angle
    rotation) applied to the gas-phase reactant geometry.  Minimises::

        E(R,t) = restraint_weight × Σ_{bonded i}  ‖p_i − p*_i‖²
               + repulsion_weight × Σ_{adsorbate a, surface s}
                                       max(0, R_min − d_as)²
               + connectivity_weight × (
                   Σ_required max(0, d_as − R_graph)²
                   + Σ_forbidden max(0, R_graph − d_as)²
                 )

    where ``p*_i`` is the clique centroid lifted by
    ``standoff_factor × (r_cov_a + ⟨r_cov_s⟩)`` along the local outward
    normal and ``R_min = contact_factor × (r_cov_a + r_cov_s)``.

    Multi-start: ``n_restarts`` rotational kicks about the local outward
    normal.  When ``try_flip=True`` (default) each kick is also tried with
    a 180° in-plane flip (essential for adsorbates with unbonded atoms that
    must point away from the surface).  Failed connectivity rounds warm-start
    each orientation from its previous optimum.  The rigid-body objective
    supplies an analytical six-coordinate Jacobian to L-BFGS-B.

    After refining the representative, full adsorption-coordination graph
    mappings propagate the new geometry to every other member, including any
    molecular atom permutation implied by the site isomorphism.

    Parameters
    ----------
    G : nx.Graph
    smiles : str
        Key into ``G.graph["adsorbate_sites"]``.
    reactant : :class:`autokmc.species.reactant.Reactant`
    restraint_weight, repulsion_weight, contact_factor, standoff_factor :
        Objective-function weights / scales.
    repulsion_cutoff : float | None
        Surface atoms farther than this from both the current pose and bonded
        target positions are excluded from the repulsion tensor.  ``None``
        keeps the exact all-surface objective.
    n_restarts : int
        Rotational restarts per placement.  Default 6.
    try_flip : bool
        Also perform 180° in-plane flips.  Default True.
    max_connectivity_attempts : int
        Maximum number of connectivity-validity refinement rounds per
        iso-class.  The calculator-free geometry is accepted only once its
        graph has exactly the required intramolecular and surface-anchor
        connectivity.  Default 3.
    max_iter : int
        L-BFGS-B iteration cap per restart.  Default 100.
    n_shells_pair : int
        Lower bound on the ego depth used to propagate the refined
        representative onto every other member via a decorated coordination
        graph mapping and MIC-aware substrate alignment.
        Each :class:`AdsorbateSite` records the classification depth in
        ``coordination_n_shells``; the propagation depth is
        ``max(n_shells_pair, ms.coordination_n_shells)`` so it cannot be
        shallower than the graph that defined the iso-class.
    nl_mult : float
        Neighbour-list cutoff multiplier used to validate the refined
        adsorbate connectivity.  Default
        :data:`~autokmc.core.constants.NL_MULT_DEFAULT`.
    kabsch_max_mappings : int
        Maximum graph-isomorphism mappings tested while propagating the
        refined representative to equivalent members.  Default
        :data:`~autokmc.core.constants.KABSCH_MAX_MAPPINGS`.
    verbose : bool

    Returns
    -------
    list[AdsorbateSite]
        The same list at ``G.graph["adsorbate_sites"][smiles]``;
        :attr:`AdsorbateSite.positions` updated in place.

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    KeyError
        If *smiles* is not in ``G.graph["adsorbate_sites"]``.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "optimise_adsorbate_site_positions requires scipy "
            "(pip install scipy)."
        ) from e

    ads_dict = G.graph.get("adsorbate_sites", {})
    if smiles not in ads_dict:
        raise KeyError(
            f"No adsorbate sites for SMILES {smiles!r}; "
            "call find_adsorbate_sites first."
        )
    adsorbate_sites: list[AdsorbateSite] = ads_dict[smiles]
    if not adsorbate_sites:
        return adsorbate_sites

    from ase.data import covalent_radii as _RC, atomic_numbers as _AN
    q_ref    = np.asarray(reactant.atoms.get_positions(), dtype=float)
    n_atoms  = q_ref.shape[0]
    ads_rcov = np.array([
        float(_RC[_AN[reactant.graph.nodes[i]["element"]]])
        for i in range(n_atoms)
    ])
    q_centroid = q_ref.mean(axis=0)
    q_centred  = q_ref - q_centroid

    surf_pos, surf_r, surf_ids = _surface_atoms_array(G)
    surf_id_to_row = {nid: i for i, nid in enumerate(surf_ids)}
    elements = {i: reactant.graph.nodes[i]["element"] for i in range(n_atoms)}
    site_idx = _clique_position_index(G, list(elements.values()))
    cell, cell_inv, pbc, use_mic = _get_cell(G)
    connectivity_context = _prepare_geometry_connectivity_context(
        G,
        reactant,
        nl_mult=float(nl_mult),
    )
    surface_spatial_index = (
        _build_anchor_spatial_index(
            [(frozenset(), p) for p in surf_pos],
            cell,
            pbc,
            use_mic,
            cutoff=float(repulsion_cutoff),
        )
        if repulsion_cutoff is not None
        else None
    )

    def _refine(ms: AdsorbateSite) -> tuple[np.ndarray, float, float, int, int]:
        bonded_idx = [i for i, c in enumerate(ms.atom_cliques) if c is not None]
        cur_pos      = np.asarray(ms.positions, dtype=float)
        cur_centroid = cur_pos.mean(axis=0)

        targets: list[np.ndarray] = []
        exclude_surf_by_atom: list[set[int]] = [set() for _ in range(n_atoms)]
        for i in bonded_idx:
            el     = elements[i]
            clique = ms.atom_cliques[i]
            base   = site_idx.get(el, {}).get(
                frozenset(clique),
                np.asarray(ms.positions[i], dtype=float),
            )
            if standoff_factor > 0.0:
                rows    = [surf_id_to_row[int(s)] for s in clique
                           if int(s) in surf_id_to_row]
                r_s_avg = float(np.mean(surf_r[rows])) if rows else 0.0
                standoff = standoff_factor * (ads_rcov[i] + r_s_avg)
                # The standoff must be measured from the *surface* clique
                # centroid, NOT from `base` (which is the anchor-node
                # position already lifted ~OPT_FACTOR*(r_a+r_s) above the
                # surface).  Using `base` here double-counts the height,
                # producing targets ~2× too far from the surface.
                if rows:
                    clique_positions = unwrap_positions_about_reference(
                        surf_pos[rows],
                        cell,
                        pbc,
                    )
                    clique_centroid = clique_positions.mean(axis=0)
                else:
                    clique_centroid = base
                n_hat = _outward_normal_at(G, clique_centroid, pbc)
                targets.append(clique_centroid + standoff * n_hat)
            else:
                targets.append(base)
            exclude_surf_by_atom[i].update(int(s) for s in clique)

        target_arr = np.array(targets, dtype=float).reshape(-1, 3)
        if use_mic and target_arr.size:
            current_bonded_positions = cur_pos[np.asarray(bonded_idx, dtype=int)]
            target_arr = current_bonded_positions + minimum_image_vectors(
                target_arr - current_bonded_positions,
                cell,
                pbc,
            )
        free_rows  = np.arange(len(surf_ids), dtype=int)
        if (
            repulsion_cutoff is not None
            and free_rows.size
            and surface_spatial_index is not None
        ):
            centers = cur_pos
            if target_arr.size:
                centers = np.vstack((centers, target_arr))
            nearby = set(_source_indices_within_radius(
                surface_spatial_index,
                centers,
                float(repulsion_cutoff),
            ))
            free_rows = np.asarray(
                [row for row in free_rows if int(row) in nearby],
                dtype=int,
            )
        free_pos   = surf_pos[free_rows]
        free_r     = surf_r[free_rows]
        free_ids   = [int(surf_ids[int(row)]) for row in free_rows]
        repulsion_pair_mask = np.ones((n_atoms, len(free_ids)), dtype=bool)
        for atom_i, excluded in enumerate(exclude_surf_by_atom):
            if not excluded:
                continue
            for row_i, surf_id in enumerate(free_ids):
                if surf_id in excluded:
                    repulsion_pair_mask[atom_i, row_i] = False

        R0, _        = _kabsch(q_ref, cur_pos)
        n_hat        = _outward_normal_at(G, cur_centroid, pbc)

        if try_flip:
            e1 = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(e1, n_hat))) > 0.9:
                e1 = np.array([0.0, 1.0, 0.0])
            flip_ax  = e1 - float(np.dot(e1, n_hat)) * n_hat
            flip_ax /= float(np.linalg.norm(flip_ax))
            R_flip   = _rotation_from_axis_angle(flip_ax * np.pi)
        else:
            R_flip = None

        required_surface_mask = np.zeros(
            (n_atoms, connectivity_context.n_slab),
            dtype=bool,
        )
        for atom_index, clique in enumerate(ms.atom_cliques):
            if clique is None:
                continue
            for surface_node in clique:
                slab_index = connectivity_context.node_to_slab_index.get(
                    int(surface_node)
                )
                if slab_index is not None:
                    required_surface_mask[atom_index, slab_index] = True
        connectivity_thresholds = (
            connectivity_context.adsorbate_cutoffs[:, None]
            + connectivity_context.slab_cutoffs[None, :]
        )
        # A narrow buffer keeps optimized poses away from the strict
        # neighbour-list boundary without materially changing their geometry.
        connectivity_margin = 0.02
        connectivity_weight = max(
            10.0,
            float(restraint_weight),
            float(repulsion_weight),
        )

        def _make_pose(R_base):
            def _pose(x):
                return (
                    q_centred
                    @ (
                        R_base
                        @ _rotation_from_axis_angle(
                            np.asarray(x[3:], dtype=float)
                        )
                    ).T
                    + (cur_centroid + np.asarray(x[:3], dtype=float))
                )
            return _pose

        def _make_energy_and_jac(
            R_base,
            restraint_scale: float,
            repulsion_scale: float,
            connectivity_scale: float,
        ):
            """Return cached scalar objective and analytical Jacobian."""
            cache: dict[str, np.ndarray | float | None] = {
                "x": None,
                "energy": None,
                "gradient": None,
            }

            def _evaluate(x):
                x = np.asarray(x, dtype=float)
                cached_x = cache["x"]
                if (
                    isinstance(cached_x, np.ndarray)
                    and np.array_equal(x, cached_x)
                ):
                    return float(cache["energy"]), np.asarray(cache["gradient"])

                rotvec = x[3:]
                rotation = R_base @ _rotation_from_axis_angle(rotvec)
                p = (
                    q_centred @ rotation.T
                    + (cur_centroid + x[:3])
                )
                right_jacobian = _rotation_right_jacobian(rotvec)
                rotation_jacobian = np.empty((n_atoms, 3, 3), dtype=float)
                for atom_index, reference_position in enumerate(q_centred):
                    rotation_jacobian[atom_index] = (
                        -rotation
                        @ _skew_matrix(reference_position)
                        @ right_jacobian
                    )

                energy = 0.0
                gradient_positions = np.zeros_like(p)
                if bonded_idx:
                    displacement = p[bonded_idx] - target_arr
                    weight = float(restraint_weight) * float(restraint_scale)
                    energy += weight * float(
                        np.einsum("ij,ij->", displacement, displacement)
                    )
                    gradient_positions[bonded_idx] += 2.0 * weight * displacement

                if free_pos.size:
                    # Vectorised pairwise (adsorbate, free-surface) repulsion.
                    dv = free_pos[None, :, :] - p[:, None, :]
                    if use_mic and cell_inv is not None:
                        dv = minimum_image_vectors(dv, cell, pbc)
                    d2 = np.einsum("ijk,ijk->ij", dv, dv)
                    distance = np.sqrt(d2 + 1e-12)
                    minimum_distance = contact_factor * (
                        ads_rcov[:, None] + free_r[None, :]
                    )
                    overlap = np.maximum(minimum_distance - distance, 0.0)
                    overlap = np.where(repulsion_pair_mask, overlap, 0.0)
                    weight = float(repulsion_weight) * float(repulsion_scale)
                    energy += weight * float(
                        np.einsum("ij,ij->", overlap, overlap)
                    )
                    coefficient = 2.0 * weight * overlap / distance
                    gradient_positions += np.einsum(
                        "ij,ijk->ik",
                        coefficient,
                        dv,
                    )

                if connectivity_context.n_slab:
                    dv = (
                        connectivity_context.slab_positions[None, :, :]
                        - p[:, None, :]
                    )
                    if connectivity_context.pbc.any():
                        dv = minimum_image_vectors(
                            dv,
                            connectivity_context.cell,
                            connectivity_context.pbc,
                        )
                    d2 = np.einsum("ijk,ijk->ij", dv, dv)
                    distance = np.sqrt(d2 + 1e-12)
                    required_excess = np.where(
                        required_surface_mask,
                        np.maximum(
                            distance
                            - (connectivity_thresholds - connectivity_margin),
                            0.0,
                        ),
                        0.0,
                    )
                    forbidden_overlap = np.where(
                        ~required_surface_mask,
                        np.maximum(
                            (connectivity_thresholds + connectivity_margin)
                            - distance,
                            0.0,
                        ),
                        0.0,
                    )
                    weight = connectivity_weight * float(connectivity_scale)
                    energy += weight * float(
                        np.einsum(
                            "ij,ij->",
                            required_excess,
                            required_excess,
                        )
                        + np.einsum(
                            "ij,ij->",
                            forbidden_overlap,
                            forbidden_overlap,
                        )
                    )
                    coefficient = (
                        2.0
                        * weight
                        * (forbidden_overlap - required_excess)
                        / distance
                    )
                    gradient_positions += np.einsum(
                        "ij,ijk->ik",
                        coefficient,
                        dv,
                    )

                gradient = np.empty(6, dtype=float)
                gradient[:3] = gradient_positions.sum(axis=0)
                gradient[3:] = np.einsum(
                    "ni,nij->j",
                    gradient_positions,
                    rotation_jacobian,
                )
                cache["x"] = x.copy()
                cache["energy"] = float(energy)
                cache["gradient"] = gradient.copy()
                return float(energy), gradient

            def _energy(x):
                return _evaluate(x)[0]

            def _jacobian(x):
                return _evaluate(x)[1]

            return _energy, _jacobian

        energy0, _jacobian0 = _make_energy_and_jac(R0, 1.0, 1.0, 1.0)
        E0 = energy0(np.zeros(6))
        if _geometry_connectivity_mismatch(
            G,
            ms,
            reactant,
            cur_pos,
            nl_mult=nl_mult,
            _context=connectivity_context,
        ) is None:
            return cur_pos.copy(), E0, E0, -1, 0

        n_starts  = max(1, int(n_restarts))
        bases     = []
        for k in range(n_starts):
            angle  = 2.0 * np.pi * k / n_starts
            R_kick = (_rotation_from_axis_angle(n_hat * angle)
                      if angle != 0.0 else np.eye(3))
            R_b = R_kick @ R0
            bases.append(R_b)
            if R_flip is not None:
                bases.append(R_flip @ R_b)

        best_E, best_pose, best_k = np.inf, None, 0
        best_mismatch: tuple[set[frozenset], set[frozenset]] | None = None
        last_exc: Exception | None = None
        attempts = max(1, int(max_connectivity_attempts))
        start_vectors = [np.zeros(6, dtype=float) for _ in bases]
        for attempt in range(attempts):
            restraint_scale = 1.0 + float(attempt)
            repulsion_scale = 1.0 + 0.25 * float(attempt)
            connectivity_scale = 1.0 + float(attempt)
            connected: list[tuple[float, np.ndarray, int]] = []
            for k, R_base in enumerate(bases):
                pose_k = _make_pose(R_base)
                energy_k, jacobian_k = _make_energy_and_jac(
                    R_base,
                    restraint_scale,
                    repulsion_scale,
                    connectivity_scale,
                )
                try:
                    res = minimize(
                        energy_k,
                        start_vectors[k],
                        method="L-BFGS-B",
                        jac=jacobian_k,
                        options={"maxiter": max_iter, "ftol": 1e-7},
                    )
                    result_vector = np.asarray(res.x, dtype=float)
                    if result_vector.shape != (6,) or not np.isfinite(
                        result_vector
                    ).all():
                        raise ValueError(
                            "optimizer returned a non-finite rigid-body pose"
                        )
                    E_k = float(res.fun)
                    if not np.isfinite(E_k):
                        raise ValueError(
                            "optimizer returned a non-finite objective"
                        )
                    start_vectors[k] = result_vector.copy()
                    p_k = pose_k(result_vector)
                    mismatch = _geometry_connectivity_mismatch(
                        G,
                        ms,
                        reactant,
                        p_k,
                        nl_mult=nl_mult,
                        _context=connectivity_context,
                    )
                except Exception as exc:
                    last_exc = exc
                    continue
                if mismatch is None:
                    connected.append((E_k, p_k, k))
                    continue
                if E_k < best_E:
                    best_E, best_pose, best_k = E_k, p_k, k
                    best_mismatch = mismatch

            if connected:
                E_k, p_k, k = min(connected, key=lambda item: item[0])
                return p_k, E0, E_k, k, attempt + 1

        if best_pose is None:
            raise RuntimeError(
                f"optimise_adsorbate_site_positions: every one of "
                f"{len(bases) * attempts} rigid-body restarts failed for iso-class "
                f"{ms.iso_class} of {smiles!r}.  Last exception: {last_exc!r}"
            )
        missing, extra = best_mismatch or (set(), set())
        raise RuntimeError(
            "optimise_adsorbate_site_positions: calculator-free geometry for "
            f"iso-class {ms.iso_class} of {smiles!r} did not satisfy required "
            f"connectivity after {attempts} attempt(s) "
            f"(missing={len(missing)}, extra={len(extra)})."
        )

    if verbose:
        print(
            f"optimise_adsorbate_site_positions: smiles={smiles!r}  "
            f"placements={len(adsorbate_sites)}  "
            f"restraint={restraint_weight}  repulsion={repulsion_weight}  "
            f"repulsion_cutoff={repulsion_cutoff}  contact={contact_factor}  "
            f"standoff={standoff_factor}  "
            f"restarts={n_restarts}  flip={try_flip}  "
            f"connectivity_attempts={max_connectivity_attempts}"
        )

    kept_sites: list[AdsorbateSite] = []
    n_rejected = 0
    for ms in adsorbate_sites:
        try:
            new_pos, E0, Ef, best_idx, n_conn_attempts = _refine(ms)
        except Exception as exc:
            if verbose:
                print(
                    f"  iso-class {ms.iso_class}: refinement failed ({exc!r}) "
                    "— rejected"
                )
            n_rejected += 1
            _remove_iso_class_nodes(G, ms)
            continue

        position_change = new_pos - np.asarray(ms.positions)
        if use_mic:
            position_change = minimum_image_vectors(position_change, cell, pbc)
        rms = float(np.sqrt(np.mean(np.sum(position_change ** 2, axis=1))))
        new_pos = _wrap_adsorbate_positions_for_storage(
            new_pos, ms.atom_cliques, cell, pbc,
        )

        # Push representative positions and propagate to other members using
        # the same decorated coordination graph that defines the iso-class.
        # Validate the complete class before writing any graph positions.
        n_propagated = 0
        if ms.member_node_ids:
            pending_positions = [new_pos]
            if any(c is not None for c in ms.atom_cliques):
                # Rebuild members at the same substrate-ego depth that defined
                # their decorated iso-class. The explicit kwarg remains a
                # lower bound for manually constructed sites.
                depth = max(
                    int(n_shells_pair),
                    int(ms.coordination_n_shells),
                )
                for m_idx in range(1, len(ms.members)):
                    try:
                        member_pos = _propagate_adsorbate_member_positions(
                            G,
                            ms.atom_cliques,
                            ms.members[m_idx],
                            new_pos,
                            reactant,
                            n_shells=depth,
                            nl_mult=nl_mult,
                            max_mappings=kabsch_max_mappings,
                            _context=connectivity_context,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "optimise_adsorbate_site_positions: decorated "
                            f"propagation failed for iso-class {ms.iso_class}, "
                            f"member {m_idx}"
                        ) from exc
                    pending_positions.append(member_pos)
            for member_index, member_pos in enumerate(pending_positions):
                push_member_positions_to_graph(
                    G,
                    ms,
                    member_index,
                    member_pos,
                )
            n_propagated = len(pending_positions) - 1
        ms.positions = new_pos

        if verbose:
            print(
                f"  iso-class {ms.iso_class:3d}:  "
                f"E {E0:9.4f} → {Ef:9.4f}  ΔRMS {rms:6.3f} Å  "
                f"best_restart={best_idx}  "
                f"connectivity_attempts={n_conn_attempts}  "
                f"propagated {n_propagated}/{max(0, len(ms.members) - 1)} members"
            )

        kept_sites.append(ms)

    if n_rejected:
        adsorbate_sites[:] = kept_sites
        G.graph.setdefault("adsorbate_sites", {})[smiles] = adsorbate_sites
        rebuild_adsorbate_reverse_indexes(G)
        if verbose:
            print(
                f"  calc-free refinement rejected {n_rejected} "
                f"iso-class(es); {len(adsorbate_sites)} remain"
            )

    return adsorbate_sites
