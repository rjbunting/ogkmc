"""
autokmc.opt_site
================
Calculator-driven (ML potential) refinement of every unique
:class:`~autokmc.find_adsorbate_site.AdsorbateSite` placement.

Workflow
--------
For every iso-class in ``cache.adsorbate_sites[smiles]``:

1. **Build combined Atoms.**  Append the gas-phase reactant geometry,
   translated to ``AdsorbateSite.positions``, to a copy of the slab / NP
   ``atoms_template``.
2. **Constrain frozen surface atoms.**  ``FixAtoms`` on
   ``atoms_template.info["frozen_indices"]`` (populated by
   :func:`autokmc.structure.build_surface`).
3. **Relax** with the supplied ASE calculator (BFGS by default).
4. **Connectivity check** (ASE :class:`~ase.neighborlist.NeighborList`
   at ``nl_mult=NL_MULT_DEFAULT`` — identical to
   :func:`autokmc.graph.build_graph`):

   * every intramolecular reactant bond is preserved,
   * no spurious intramolecular bond appears,
   * every anchor remains bonded to ≥1 atom of its original surface
     clique,
   * non-anchor reactant atoms do **not** acquire a surface contact.

   Failure flags ``AdsorbateSite.stable = False``; success continues.
5. **Save relaxed adsorbate geometry** onto ``AdsorbateSite.positions``.
6. **Propagate to every other member** of the iso-class by rigid
   alignment of the **n_shells=1 ego subgraph** of the anchor cliques.
   Algorithm:

   a. Initial guess: Kabsch on the bonded-clique centroids
      (in molecular atom order).
   b. Build the n_shells=1 surface-only ego atom set around each
      member's anchor cliques (= anchors ∪ their direct surface
      neighbours).  At opt-time *G* is bare so this is just surface
      atoms; on a later occupied snapshot the **adsorbate-chain
      closure** in :func:`lateral_neighbour_atoms` automatically
      pulls in every neighbouring adsorbate molecule too.
   c. Apply the initial transform to the representative's ego atom
      positions; for each predicted point, snap to the nearest
      same-element member ego atom (one-step ICP).
   d. Re-Kabsch on the full matched set (anchors + ego neighbours)
      and apply to the relaxed adsorbate Cartesians.

   Single-anchor placements with no extra ego neighbours degenerate to
   a pure MIC translation (rotation is undetermined and any choice is
   equivalent under the iso-class symmetry).  The propagated
   Cartesians land on ``AdsorbateSite.member_positions[k]``, the ego atom
   set of every member lands on ``AdsorbateSite.member_neighbour_atoms[k]``
   (``frozenset[int]`` of node ids in *G*), and the materialised
   ego subgraph (a stand-alone ``nx.Graph`` copy) lands on
   ``AdsorbateSite.member_subgraphs[k]`` — the canonical
   lateral-interaction signature, suitable for hashing,
   isomorphism-matching against future KMC snapshots, or building
   higher-order interaction tables.
7. **Adsorption energy:**
   ``E_ads = E_relaxed − E_clean − E_gas`` stored on
   ``AdsorbateSite.adsorption_energy`` (eV).  ``E_clean`` is computed once
   on the bare ``atoms_template``; ``E_gas`` is taken from
   ``reactant.energy`` if present, otherwise re-evaluated.

Public API
----------
* :func:`optimise_adsorbate_sites_ml` — main entry point; mutates
  ``cache.adsorbate_sites[smiles]`` in place and returns the list.
* :func:`lateral_neighbour_atoms` / :func:`lateral_neighbour_subgraph`
  — graph helpers that compute a placement's lateral-interaction
  environment.  Surface-only n-shell BFS, plus optional **transitive
  adsorbate-chain closure**: when an adsorbate node is encountered,
  the *entire* connected component of adsorbate atoms is added (without
  consuming a shell), so an occupied snapshot's ego graph naturally
  contains every neighbouring adsorbate molecule.

Optimization-populated fields on :class:`~autokmc.find_adsorbate_site.AdsorbateSite`
-----------------------------------------------------------------------------
The optimiser populates these explicit ``AdsorbateSite`` dataclass fields:

* ``stable : bool``                                     — connectivity preserved & site unchanged?
* ``adsorption_energy : float | None``                  — eV (``None`` if unstable)
* ``member_positions  : list[np.ndarray] | None``       — per-member adsorbate
  Cartesians (member 0 == relaxed representative)
* ``member_neighbour_atoms : list[frozenset[int]] | None`` — per-member
  ``n_shells=1`` lateral-interaction node set on *G* (chain-following on)
* ``member_subgraphs : list[nx.Graph] | None``          — per-member
  materialised lateral-interaction subgraph (``G.subgraph(…).copy()``);
  the canonical signature used downstream to detect overlaps with
  other placements / KMC snapshots.
* ``relaxed_graph : nx.Graph | None``                   — fresh graph built
  on the relaxed combined Atoms via :func:`autokmc.graph.build_graph`
  (populated for both ``stable`` and ``site_changed`` outcomes).
* ``relaxed_cliques : list[frozenset[int] | None] | None`` — per-reactant-atom
  *actual* surface neighbour sets after relaxation; differs from
  ``ms.atom_cliques`` iff the placement migrated to a different site.
* ``relaxed_lateral_subgraph : nx.Graph | None``        — the lateral
  subgraph spanned by ``relaxed_cliques``; used as the cross-iso-class
  collapse-target signature.
* ``collapsed_into_iso_class : int | None``             — for unstable
  (``site_changed``) placements, the ``iso_class`` index of the stable
  iso-class whose representative ``member_subgraphs[0]`` is
  element-isomorphic to ``relaxed_lateral_subgraph`` (``None`` if no
  match).  Stable placements always carry ``None``.

Outcome semantics
-----------------
Each iso-class is classified into one of three buckets after the
ASE relaxation, by rebuilding a fresh connectivity graph on the
relaxed combined system and comparing it to the original site:

* **stable**       — intramolecular adsorbate bonds are intact AND every
  anchor's actual surface-neighbour set equals its original clique
  (set comparison).  ``stable=True``; energy + propagation populated.
* **site_changed** — adsorbate intact but at least one anchor's actual
  clique differs (e.g. bridge → hollow).  ``stable=False`` but
  ``relaxed_*`` attrs are populated for the cross-class sanity check.
* **broken**       — any intramolecular bond changed, a non-anchor
  reactant atom acquired a surface contact, or an anchor lost all
  contact with the surface.  Result is discarded entirely.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import networkx as nx

from ase import Atoms
from ase.constraints import FixAtoms
from networkx.algorithms import isomorphism as _nx_iso
from networkx.algorithms.isomorphism import categorical_node_match

from autokmc.cache import get_cache
from autokmc.constants import NL_MULT_DEFAULT, N_SHELLS_DEFAULT
from autokmc.default_sites import _iso_prefilter_key
from autokmc.logging_utils import get_logger, verbose_scope

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _resolve_cell(G: nx.Graph):
    cell = np.array(G.graph["cell"], dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)
    use_mic = bool(pbc.any())
    cell_inv: np.ndarray | None = None
    if use_mic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            use_mic = False
    return cell, cell_inv, pbc, use_mic


def _mic_unwrap(points: np.ndarray, ref: np.ndarray,
                cell, cell_inv, pbc, use_mic) -> np.ndarray:
    """Shift each row of *points* into the same image as *ref* (MIC)."""
    out = np.asarray(points, dtype=float).copy()
    if not use_mic:
        return out
    ref = np.asarray(ref, dtype=float)
    dv = out - ref
    frac = dv @ cell_inv
    for k in range(3):
        if pbc[k]:
            frac[:, k] -= np.round(frac[:, k])
    return ref + frac @ cell


def _clique_centroid(G: nx.Graph, clique, *,
                     use_mic, cell, cell_inv, pbc) -> np.ndarray:
    """MIC-aware centroid of a frozenset of node ids."""
    pts = np.array([G.nodes[int(n)]["position"] for n in clique], dtype=float)
    if pts.shape[0] == 1 or not use_mic:
        return pts.mean(axis=0)
    pts = _mic_unwrap(pts, pts[0], cell, cell_inv, pbc, use_mic)
    return pts.mean(axis=0)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid alignment ``R, t`` mapping rows of *src* onto *dst*."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    return R, t


# ---------------------------------------------------------------------------
# Combined-system construction
# ---------------------------------------------------------------------------

def _build_combined_atoms(
    atoms_template: Atoms, ms, reactant,
) -> tuple[Atoms, list[int]]:
    """Slab/NP + adsorbate at ``ms.positions``.  Returns (combined, ads_indices)."""
    combined = atoms_template.copy()
    n_surf = len(combined)
    react_atoms = reactant.atoms.copy()
    react_atoms.set_positions(np.asarray(ms.positions, dtype=float))
    combined += react_atoms
    ads_indices = list(range(n_surf, n_surf + len(react_atoms)))

    # Carry frozen-atom info forward and apply the constraint.
    frozen = list(atoms_template.info.get("frozen_indices", []))
    if frozen:
        combined.info["frozen_indices"] = frozen
        combined.set_constraint(FixAtoms(indices=frozen))

    # Surface tag for adsorbate atoms (=2) so a downstream build_graph works.
    if "surface" in atoms_template.arrays:
        tag = atoms_template.arrays["surface"].astype(np.int8)
        ads_tag = np.full(len(react_atoms), 2, dtype=np.int8)
        combined.arrays["surface"] = np.concatenate([tag, ads_tag])

    return combined, ads_indices


# ---------------------------------------------------------------------------
# Connectivity verification (post-relaxation graph rebuild)
# ---------------------------------------------------------------------------

def _rebuild_relaxed_graph(combined: Atoms, *, nl_mult: float) -> nx.Graph:
    """Rebuild a fresh connectivity graph on the relaxed combined system.

    Uses :func:`autokmc.graph.build_graph` so the cutoff convention
    (``natural_cutoffs * nl_mult``) is byte-identical to the one used by
    the original surface graph and by the geometric site enumerator.
    """
    # Local import: graph.py imports cache helpers that pull from this
    # module's siblings; keeping this lazy avoids a hypothetical cycle.
    from autokmc.graph import build_graph
    return build_graph(combined.copy(), nl_mult=nl_mult)


def _adsorbate_intramolecular_changed(
    G_relaxed: nx.Graph, ads_indices: list[int], reactant,
) -> tuple[bool, str]:
    """True if any reactant bond is missing or any spurious one appeared."""
    n_ads = len(reactant.atoms)
    react_to_g = {i: ads_indices[i] for i in range(n_ads)}
    expected = {
        tuple(sorted((react_to_g[u], react_to_g[v])))
        for u, v in reactant.graph.edges()
    }
    ads_set = set(ads_indices)
    actual = set()
    for a in ads_indices:
        for b in G_relaxed.neighbors(a):
            if b in ads_set and b > a:
                actual.add((a, b))
    missing = expected - actual
    spurious = actual - expected
    if missing:
        return True, f"intramolecular bond(s) lost: {sorted(missing)}"
    if spurious:
        return True, f"new intramolecular bond(s) formed: {sorted(spurious)}"
    return False, "ok"


def _actual_anchor_cliques(
    G_relaxed: nx.Graph, ads_indices: list[int], reactant,
) -> list[frozenset[int] | None]:
    """For every reactant atom, the set of *surface* atoms it now bonds to.

    Returns one entry per reactant atom (in molecular order); ``None``
    means the atom has no surface neighbour at all (vacuum).
    """
    ads_set = set(ads_indices)
    out: list[frozenset[int] | None] = []
    for ai in range(len(reactant.atoms)):
        node = ads_indices[ai]
        surf_nbrs = {
            int(n) for n in G_relaxed.neighbors(node)
            if int(n) not in ads_set
            and G_relaxed.nodes[int(n)].get("type") == "surface"
        }
        out.append(frozenset(surf_nbrs) if surf_nbrs else None)
    return out


def _verify_relaxed(
    combined: Atoms,
    ads_indices: list[int],
    reactant,
    ms,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
) -> tuple[
    str,                                # status: 'stable' | 'site_changed' | 'broken'
    str,                                # reason (human-readable)
    nx.Graph | None,                    # rebuilt graph
    list[frozenset[int] | None] | None, # actual per-anchor surface cliques
    frozenset[int] | None,              # rebuilt lateral neighbour atom set
    nx.Graph | None,                    # rebuilt lateral subgraph
]:
    """Full post-relaxation verification.

    Outcomes:
    * ``'broken'``      — adsorbate intramolecular topology changed, or a
      non-anchor reactant atom acquired a surface contact, or an anchor
      lost all contact with the surface.  Discard the placement.
    * ``'site_changed'`` — adsorbate intact, but the per-anchor *surface*
      neighbour set differs (set comparison) from ``ms.atom_cliques``.
      The placement migrated to a different site (e.g. bridge → hollow);
      mark unstable but **return the rebuilt graph + new clique pattern
      + lateral subgraph** for downstream sanity checks.
    * ``'stable'``      — every anchor's actual surface clique equals the
      original (as sets), and intramolecular bonds are intact.
    """
    G_rel = _rebuild_relaxed_graph(combined, nl_mult=nl_mult)

    # 1) Intramolecular topology (vacuous for monatomic reactants).
    if len(reactant.atoms) >= 2:
        changed, why = _adsorbate_intramolecular_changed(
            G_rel, ads_indices, reactant,
        )
        if changed:
            return "broken", why, None, None, None, None

    # 2) Per-anchor actual surface cliques.
    actual = _actual_anchor_cliques(G_rel, ads_indices, reactant)

    # 3) Non-anchor atoms must not touch the surface.
    bonded_idx = {ai for ai, c in enumerate(ms.atom_cliques) if c is not None}
    for ai in range(len(reactant.atoms)):
        if ai in bonded_idx:
            continue
        s = actual[ai]
        if s:
            return ("broken",
                    f"non-anchor atom {ai} bonded to surface "
                    f"{sorted(s)}",
                    None, None, None, None)

    # 4) Anchors must remain bonded to the surface.
    for ai in bonded_idx:
        if not actual[ai]:
            return ("broken",
                    f"anchor {ai} lost all contact with the surface",
                    None, None, None, None)

    # 5) Compare expected vs actual cliques (set semantics).
    site_changed = False
    diffs: list[str] = []
    for ai in bonded_idx:
        expected = {int(n) for n in ms.atom_cliques[ai]}
        got = set(actual[ai] or ())
        if expected != got:
            site_changed = True
            diffs.append(f"anchor {ai}: {sorted(expected)} -> {sorted(got)}")

    # Build the rebuilt lateral subgraph spanned by the *actual* cliques
    # (using the same lateral-neighbour helper / chain-following defaults
    # as the stored member subgraphs, so the two are directly comparable).
    actual_anchor_nodes: set[int] = set()
    for s in actual:
        if s is not None:
            actual_anchor_nodes.update(int(n) for n in s)
    lateral_atoms = lateral_neighbour_atoms(
        G_rel, actual_anchor_nodes, n_shells=N_SHELLS_DEFAULT,
        follow_adsorbate_chains=True, include_anchors=True,
    )
    lateral_sub = G_rel.subgraph(lateral_atoms).copy()

    if site_changed:
        return ("site_changed",
                "site migration: " + "; ".join(diffs),
                G_rel, actual, lateral_atoms, lateral_sub)

    return "stable", "ok", G_rel, actual, lateral_atoms, lateral_sub


# ---------------------------------------------------------------------------
# Propagation: rep → all members
# ---------------------------------------------------------------------------

def _bonded_clique_centroids(
    G: nx.Graph, atom_cliques, *, use_mic, cell, cell_inv, pbc,
) -> tuple[list[int], np.ndarray]:
    """Return ``(bonded_atom_indices, centroids[N,3])`` in molecular order."""
    bonded_idx, pts = [], []
    for ai, c in enumerate(atom_cliques):
        if c is None:
            continue
        bonded_idx.append(ai)
        pts.append(_clique_centroid(
            G, c, use_mic=use_mic, cell=cell, cell_inv=cell_inv, pbc=pbc,
        ))
    return bonded_idx, np.array(pts, dtype=float).reshape(-1, 3)


def _ego_atoms(
    G: nx.Graph, anchor_nodes: set[int], *, n_shells: int = 1,
) -> frozenset[int]:
    """Internal alias for :func:`lateral_neighbour_atoms` with the
    package's lateral-interaction defaults (chain-following on, anchors
    included).
    """
    return lateral_neighbour_atoms(
        G, anchor_nodes,
        n_shells=n_shells,
        follow_adsorbate_chains=True,
        include_anchors=True,
    )


# ---------------------------------------------------------------------------
# Public lateral-interaction graph utilities
# ---------------------------------------------------------------------------

def lateral_neighbour_atoms(
    G: nx.Graph, anchor_nodes,
    *,
    n_shells: int = 1,
    follow_adsorbate_chains: bool = True,
    include_anchors: bool = True,
) -> frozenset[int]:
    """Compute the set of node ids that constitute the lateral-interaction
    environment of a placement anchored at *anchor_nodes*.

    Semantics
    ---------
    1. BFS over **surface-typed** nodes only, starting from *anchor_nodes*,
       up to ``n_shells`` hops.  This collects ``anchors ∪ direct surface
       neighbours`` for the default ``n_shells=1`` (anchors are included
       only when ``include_anchors=True``).
    2. If ``follow_adsorbate_chains`` (default), every adsorbate-typed
       atom directly bonded to any surface node in (1) is added, and the
       full **adsorbate connected component** of each such seed (closed
       under adsorbate↔adsorbate edges) is added too.  Surface atoms
       reached via adsorbate atoms are **not** revisited as fresh BFS
       seeds — adsorbate chains do not consume shells.

    On a bare surface graph (no nodes of type ``"adsorbate"``), step 2 is
    a no-op, so the result equals the legacy surface-only n-shell ego.
    On an occupied snapshot (KMC state with multiple placements), the
    result also captures every neighbouring adsorbate molecule in full,
    which is exactly what the lateral-interaction graph needs.

    Parameters
    ----------
    G : nx.Graph
        Atom-connectivity graph (nodes have ``type ∈ {"bulk", "surface",
        "adsorbate"}``).
    anchor_nodes : iterable[int]
        Node ids to start the BFS from (typically the union of the
        ``AdsorbateSite.atom_cliques`` for one member).
    n_shells : int
        Surface-only BFS radius.  Default 1.
    follow_adsorbate_chains : bool
        Enable transitive adsorbate-chain expansion.  Default True.
    include_anchors : bool
        Include the *anchor_nodes* themselves in the result.  Default True.

    Returns
    -------
    frozenset[int]
        Node ids in the lateral-interaction environment.
    """
    anchors = {int(a) for a in anchor_nodes}
    seen: set[int] = set(anchors) if include_anchors else set()

    # ── Step 1: surface-only BFS up to n_shells. ───────────────────────────
    frontier = set(anchors)
    visited_surface = set(anchors)
    for _ in range(int(n_shells)):
        next_frontier: set[int] = set()
        for u in frontier:
            for v in G.neighbors(u):
                v = int(v)
                if v in visited_surface:
                    continue
                if G.nodes[v].get("type") == "surface":
                    visited_surface.add(v)
                    next_frontier.add(v)
                    seen.add(v)
        if not next_frontier:
            break
        frontier = next_frontier

    # ── Step 2: full adsorbate-chain closure off the surface ego. ─────────
    if follow_adsorbate_chains:
        # Adsorbate atoms directly bonded to any surface node we've seen
        # (including the anchors themselves) are seeds.
        ads_seeds: set[int] = set()
        for u in visited_surface:
            for v in G.neighbors(u):
                if G.nodes[v].get("type") == "adsorbate":
                    ads_seeds.add(int(v))
        # Transitive closure within adsorbate-typed nodes.
        ads_seen: set[int] = set()
        stack = list(ads_seeds)
        while stack:
            u = stack.pop()
            if u in ads_seen:
                continue
            ads_seen.add(u)
            for v in G.neighbors(u):
                v = int(v)
                if G.nodes[v].get("type") == "adsorbate" and v not in ads_seen:
                    stack.append(v)
        seen |= ads_seen

    return frozenset(seen)


def lateral_neighbour_subgraph(
    G: nx.Graph, anchor_nodes,
    *,
    n_shells: int = 1,
    follow_adsorbate_chains: bool = True,
    include_anchors: bool = True,
) -> nx.Graph:
    """:func:`lateral_neighbour_atoms` + materialised subgraph.

    Returns ``G.subgraph(node_set).copy()`` so the result is independent
    of any later mutation of *G* — suitable for storing as the canonical
    lateral-interaction signature of a placement (it can be hashed,
    pickled, isomorphism-matched against the equivalent subgraph from a
    later KMC snapshot, etc.).
    """
    nodes = lateral_neighbour_atoms(
        G, anchor_nodes,
        n_shells=n_shells,
        follow_adsorbate_chains=follow_adsorbate_chains,
        include_anchors=include_anchors,
    )
    return G.subgraph(nodes).copy()


def _ego_positions_unwrapped(
    G: nx.Graph, ego_nodes: list[int], ref: np.ndarray,
    *, cell, cell_inv, pbc, use_mic,
) -> np.ndarray:
    """MIC-unwrap each ego atom's position into the same image as *ref*."""
    pts = np.array([G.nodes[n]["position"] for n in ego_nodes], dtype=float)
    return _mic_unwrap(pts, ref, cell, cell_inv, pbc, use_mic)



def _kabsch_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Pure-rotation Kabsch (no translation) for displacement-vector pairs.

    Both *src* and *dst* are ``(N, 3)`` arrays of vectors anchored at the
    origin.  Returns ``R`` minimising ``Σ ‖R·src_i − dst_i‖²``.  Falls
    back to the identity if the system is degenerate (rank < 2).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape[0] == 0:
        return np.eye(3)
    H = src.T @ dst
    U, S, Vt = np.linalg.svd(H)
    if np.sum(S > 1e-9) < 2:
        # Degenerate (e.g. all displacements collinear / single vector):
        # the rotation around that axis is undetermined.
        return np.eye(3)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def _hungarian_match_by_element(
    rep_pred: np.ndarray, rep_elems: list[str],
    mem_pts:  np.ndarray, mem_elems:  list[str],
) -> tuple[list[int], list[int]]:
    """Element-restricted optimal assignment of rep predictions onto member atoms.

    Returns ``(rep_idx, mem_idx)`` pairs (greedy fallback if SciPy is
    unavailable).  Only same-element pairs are considered.
    """
    n_rep, n_mem = len(rep_elems), len(mem_elems)
    if n_rep == 0 or n_mem == 0:
        return [], []

    INF = 1e9
    D = np.full((n_rep, n_mem), INF, dtype=float)
    for i in range(n_rep):
        for j in range(n_mem):
            if rep_elems[i] != mem_elems[j]:
                continue
            d = rep_pred[i] - mem_pts[j]
            D[i, j] = float(np.sqrt(d @ d))

    try:
        from scipy.optimize import linear_sum_assignment
        # Pad to a square matrix so unmatched rows/cols are tolerated.
        n = max(n_rep, n_mem)
        Dpad = np.full((n, n), INF, dtype=float)
        Dpad[:n_rep, :n_mem] = D
        row, col = linear_sum_assignment(Dpad)
        ri, mi = [], []
        for r, c in zip(row, col):
            if r < n_rep and c < n_mem and Dpad[r, c] < INF:
                ri.append(int(r))
                mi.append(int(c))
        return ri, mi
    except Exception:
        # Greedy fallback (same as the legacy implementation).
        used_rep = np.zeros(n_rep, dtype=bool)
        used_mem = np.zeros(n_mem, dtype=bool)
        ri, mi = [], []
        while True:
            masked = D.copy()
            masked[used_rep, :] = INF
            masked[:, used_mem] = INF
            i, j = np.unravel_index(np.argmin(masked), masked.shape)
            if masked[i, j] >= INF:
                break
            ri.append(int(i)); mi.append(int(j))
            used_rep[i] = True; used_mem[j] = True
        return ri, mi


def _propagate_to_members(
    G: nx.Graph, ms, relaxed_ads: np.ndarray, *, n_shells: int = 1,
) -> tuple[list[np.ndarray], list[frozenset[int]], list[nx.Graph]]:
    """Rigid-align ``relaxed_ads`` onto every member using *displacement vectors
    to n_shells=1 surface neighbours* (anchor pinned).

    Algorithm (per non-representative member):

    1. Build the n_shells=1 surface-only ego of each member's anchor union
       (``follow_adsorbate_chains=False`` because at opt time *G* is bare
       and the alignment basis must not pull in adsorbates).
    2. Translation is locked: ``t = anchor_origin_member − anchor_origin_rep``
       where ``anchor_origin`` is the MIC centroid of the **first** bonded
       clique.
    3. Match rep ego atoms to member ego atoms by element, using a
       Hungarian assignment seeded with a centroid-Kabsch initial guess
       (so the cost matrix is a meaningful displacement distance).
    4. Build displacement vectors ``u_i = rep_ego_i − anchor_origin_rep``
       and ``v_i = mem_ego_i − anchor_origin_mem``; solve a **rotation-only
       Kabsch** on the matched ``(u_i, v_i)`` pairs.  Anchors contribute
       zero vectors (they are the pivot), so the rotation is set entirely
       by the n_shells=1 *surface neighbours* of the anchor — this is what
       gives the propagation directionality on the surface.
    5. Apply ``R, t`` to ``relaxed_ads`` (which is in the rep frame, with
       its own anchor origin already coincident with the rep anchor
       centroid in absolute coordinates).

    Single-anchor placements with no surface neighbours fall back to MIC
    translation (rotation undetermined).  The stored
    ``member_neighbour_atoms`` / ``member_subgraphs`` keep using the
    chain-following helper (``follow_adsorbate_chains=True``) so the
    canonical lateral-interaction signature is preserved on later
    occupied snapshots.

    Returns ``(member_positions, member_neighbour_atoms, member_subgraphs)``.
    """
    cell, cell_inv, pbc, use_mic = _resolve_cell(G)

    # Representative anchor + ego (alignment basis: surface only).
    rep_anchor_nodes: set[int] = set()
    for c in ms.atom_cliques:
        if c is not None:
            rep_anchor_nodes.update(int(n) for n in c)
    rep_align_set = lateral_neighbour_atoms(
        G, rep_anchor_nodes, n_shells=n_shells,
        follow_adsorbate_chains=False, include_anchors=True,
    )
    rep_align_nodes = sorted(rep_align_set)
    rep_align_elems = [G.nodes[n]["element"] for n in rep_align_nodes]

    # Representative bonded centroids (used to define anchor_origin_rep).
    rep_bonded, rep_cent = _bonded_clique_centroids(
        G, ms.atom_cliques,
        use_mic=use_mic, cell=cell, cell_inv=cell_inv, pbc=pbc,
    )
    if rep_cent.shape[0] == 0:
        # Should not happen: an AdsorbateSite with no bonded anchors.
        return ([np.asarray(relaxed_ads, dtype=float)] * len(ms.members),
                [frozenset()] * len(ms.members),
                [nx.Graph()] * len(ms.members))
    rep_origin = rep_cent[0]
    rep_align_pts = _ego_positions_unwrapped(
        G, rep_align_nodes, rep_origin,
        cell=cell, cell_inv=cell_inv, pbc=pbc, use_mic=use_mic,
    )
    rep_disp = rep_align_pts - rep_origin   # displacement vectors u_i

    member_positions: list[np.ndarray] = []
    member_neighbour: list[frozenset[int]] = []
    member_subgraphs: list[nx.Graph] = []

    for k, mem_cliques in enumerate(ms.members):
        # Stored lateral-interaction signature (chain-following ON).
        mem_anchor_nodes: set[int] = set()
        for c in mem_cliques:
            if c is not None:
                mem_anchor_nodes.update(int(n) for n in c)
        mem_lateral = _ego_atoms(G, mem_anchor_nodes, n_shells=n_shells)
        member_neighbour.append(mem_lateral)
        member_subgraphs.append(G.subgraph(mem_lateral).copy())

        if k == 0:
            member_positions.append(np.asarray(relaxed_ads, dtype=float))
            continue

        # Member anchor origin (== centroid of the first bonded clique).
        mem_bonded, mem_cent = _bonded_clique_centroids(
            G, mem_cliques,
            use_mic=use_mic, cell=cell, cell_inv=cell_inv, pbc=pbc,
        )
        if mem_bonded != rep_bonded or mem_cent.shape[0] == 0:
            _log.warning(
                "iso-class %d, member %d: anchor pattern mismatch "
                "(rep=%s, mem=%s); falling back to MIC translation",
                ms.iso_class, k, rep_bonded, mem_bonded,
            )
            shift = mem_cent[0] - rep_origin if mem_cent.shape[0] else np.zeros(3)
            member_positions.append(np.asarray(relaxed_ads) + shift)
            continue
        mem_origin = mem_cent[0]
        translation = mem_origin - rep_origin
        if use_mic:
            frac = translation @ cell_inv
            for kk in range(3):
                if pbc[kk]:
                    frac[kk] -= np.round(frac[kk])
            translation = frac @ cell

        # Member ego (alignment basis).
        mem_align_set = lateral_neighbour_atoms(
            G, mem_anchor_nodes, n_shells=n_shells,
            follow_adsorbate_chains=False, include_anchors=True,
        )
        mem_align_nodes = sorted(mem_align_set)
        mem_align_elems = [G.nodes[n]["element"] for n in mem_align_nodes]
        mem_align_pts = _ego_positions_unwrapped(
            G, mem_align_nodes, mem_origin,
            cell=cell, cell_inv=cell_inv, pbc=pbc, use_mic=use_mic,
        )
        mem_disp = mem_align_pts - mem_origin   # displacement vectors v_j

        # Initial guess for matching: identity rotation + the locked
        # translation.  (Centroid Kabsch on a single anchor is degenerate
        # and biases the matching; instead we use displacement *directions*
        # so the cost is meaningful even with R=I.)
        rep_pred = rep_disp.copy()  # in member-anchor frame, R=I guess
        ri, mi = _hungarian_match_by_element(
            rep_pred, rep_align_elems,
            mem_disp, mem_align_elems,
        )

        # Drop self-matches at the anchor (zero-vectors); they don't
        # constrain the rotation.  Keep at least the surface-neighbour
        # displacements.
        if ri:
            src_disp = rep_disp[ri]
            dst_disp = mem_disp[mi]
        else:
            src_disp = np.empty((0, 3))
            dst_disp = np.empty((0, 3))

        if src_disp.shape[0] >= 2:
            R = _kabsch_rotation(src_disp, dst_disp)
        else:
            # Degenerate: single anchor, no surface neighbours matched.
            # MIC translation is the best we can do.
            R = np.eye(3)

        # Apply: pivot the relaxed adsorbate around rep_origin, rotate,
        # then translate to member frame.
        new_ads = (relaxed_ads - rep_origin) @ R.T + rep_origin + translation
        member_positions.append(new_ads)

    return member_positions, member_neighbour, member_subgraphs


# ---------------------------------------------------------------------------
# Cross-iso-class collapse sanity check
# ---------------------------------------------------------------------------

def _find_collapse_target(
    unstable_subgraph: nx.Graph,
    stable_adsorbate_sites: list[Any],
) -> int | None:
    """Return the ``iso_class`` of the first stable AdsorbateSite whose
    representative lateral subgraph is element-isomorphic to
    *unstable_subgraph*; or ``None`` if no match is found.

    Uses the two-tier prefilter pattern from
    :func:`autokmc.default_sites.reduce_sites_by_isomorphism`:
    cheap structural fingerprint via :func:`_iso_prefilter_key` first,
    full ``GraphMatcher`` only on candidates that survive.
    """
    if unstable_subgraph is None or unstable_subgraph.number_of_nodes() == 0:
        return None
    try:
        fkey = _iso_prefilter_key(unstable_subgraph)
    except Exception:
        return None
    nm = categorical_node_match("element", "X")
    for ms in stable_adsorbate_sites:
        ref = getattr(ms, "member_subgraphs", None)
        if not ref:
            continue
        ref0 = ref[0]
        try:
            if _iso_prefilter_key(ref0) != fkey:
                continue
        except Exception:
            continue
        if _nx_iso.GraphMatcher(
            unstable_subgraph, ref0, node_match=nm,
        ).is_isomorphic():
            return int(ms.iso_class)
    return None


# ---------------------------------------------------------------------------
# Single-atom adapter: synthesise AdsorbateSites from IsoClasses
# ---------------------------------------------------------------------------

def seed_single_atom_adsorbate_sites(
    G: nx.Graph,
    reactant,
    *,
    n_shells: int = N_SHELLS_DEFAULT,
    overwrite: bool = False,
    verbose: bool = False,
) -> list[Any]:
    """Build :class:`AdsorbateSite`-shaped wrappers from
    ``cache.unique_sites[element][n_shells]`` for monatomic *reactant*.

    Single-atom adsorbates never enter the multi-atom enumerator
    (:func:`autokmc.find_adsorbate_site.find_adsorbate_sites` requires
    ``n_atoms >= 2``); they are produced by the ``default_sites``
    pipeline as :class:`~autokmc.default_sites.IsoClass` objects under
    ``cache.unique_sites[element]``.  This adapter wraps each IsoClass
    in a :class:`AdsorbateSite` (one anchor = the IsoClass clique; one
    member per IsoClass member) so that
    :func:`optimise_adsorbate_sites_ml` can relax single-atom and
    multi-atom adsorbates through the **same** code path.

    Lazily ensures the ``default_sites`` pipeline has been run for
    *reactant*'s element via
    :func:`autokmc.find_adsorbate_site._ensure_default_sites`.

    Parameters
    ----------
    G : nx.Graph
        Surface graph.
    reactant : :class:`~autokmc.reactants.Reactant`
        Must satisfy ``len(reactant.atoms) == 1``.
    n_shells : int
        Iso-class shell depth to read from
        ``cache.unique_sites[element]``.  Default ``N_SHELLS_DEFAULT``.
    overwrite : bool
        If False (default) and ``cache.adsorbate_sites[reactant.smiles]`` is
        already populated, this is a no-op.

    Returns
    -------
    list[AdsorbateSite]
        Same list stored in ``cache.adsorbate_sites[reactant.smiles]``;
        ordered ``(k, iso_class)`` ascending so iso_class indices are
        unique within the list.
    """
    # Local imports to avoid a circular dependency with find_adsorbate_site.
    from autokmc.find_adsorbate_site import AdsorbateSite, _ensure_default_sites

    if len(reactant.atoms) != 1:
        raise ValueError(
            "seed_single_atom_adsorbate_sites is only for monatomic reactants; "
            f"got {len(reactant.atoms)} atoms."
        )

    cache = get_cache(G)
    smiles = reactant.smiles
    if not overwrite and cache.adsorbate_sites.get(smiles):
        return cache.adsorbate_sites[smiles]

    element = reactant.atoms.get_chemical_symbols()[0]
    _ensure_default_sites(G, element, n_shells, verbose=verbose)

    by_n = cache.unique_sites.get(element, {})
    if n_shells not in by_n:
        raise KeyError(
            f"cache.unique_sites[{element!r}][{n_shells}] not populated "
            "even after _ensure_default_sites; nothing to seed."
        )

    adsorbate_sites: list[Any] = []
    iso_idx = 0
    for k in sorted(by_n[n_shells]):
        for iso in by_n[n_shells][k]:
            pos = iso.position if iso.position is not None else iso.centroid
            ms = AdsorbateSite(
                smiles       = smiles,
                n_atoms      = 1,
                atom_cliques = [iso.representative],
                positions    = np.asarray(pos, dtype=float).reshape(1, 3),
                iso_class    = iso_idx,
                members      = [[m] for m in iso.members],
                ego_graph    = iso.ego_graph,
            )
            # Forward-link to the underlying IsoClass so users can also
            # read the optimisation results from the canonical home.
            ms.iso_class_ref = iso  # type: ignore[attr-defined]
            ms.coordination = k     # type: ignore[attr-defined]
            adsorbate_sites.append(ms)
            iso_idx += 1

    cache.adsorbate_sites[smiles] = adsorbate_sites
    _log.info(
        "seed_single_atom_adsorbate_sites(%r): synthesised %d AdsorbateSites "
        "from cache.unique_sites[%r][%d]",
        smiles, len(adsorbate_sites), element, n_shells,
    )
    return adsorbate_sites


# ---------------------------------------------------------------------------

def optimise_adsorbate_sites_ml(
    G: nx.Graph,
    smiles: str,
    reactant,
    atoms_template: Atoms,
    calculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: Callable | None = None,
    clean_energy: float | None = None,
    gas_energy: float | None = None,
    nl_mult: float = NL_MULT_DEFAULT,
    only_iso_classes: list[int] | None = None,
    verbose: bool = False,
) -> list[Any]:
    """Relax every :class:`AdsorbateSite` in ``cache.adsorbate_sites[smiles]`` with *calculator*.

    For each iso-class, this constructs ``slab + adsorbate`` from
    *atoms_template* and ``AdsorbateSite.positions``, runs an ASE
    optimiser, verifies adsorbate–adsorbate and adsorbate–surface
    connectivity (using the same neighbour-list cutoff convention as
    :func:`autokmc.graph.build_graph`), and on success writes back the
    relaxed adsorbate Cartesians plus the adsorption energy.  The
    relaxed pose is propagated to every other member of the iso-class
    by Kabsch alignment of the bonded-clique centroids (n_shells=1 of
    each anchor is implicitly captured because all anchor atoms enter
    the alignment in molecular order).

    Parameters
    ----------
    G : nx.Graph
        Surface graph carrying ``cache.adsorbate_sites[smiles]``.
    smiles : str
        SMILES key into the adsorbate-sites cache.
    reactant : :class:`~autokmc.reactants.Reactant`
        Gas-phase reactant whose ``atoms`` provides element identities
        (positions are overwritten by ``ms.positions``) and whose
        ``graph`` defines the intramolecular bonds we expect to survive.
    atoms_template : Atoms
        The clean slab / NP that *G* was built from.  Should carry
        ``info["frozen_indices"]`` (populated by
        :func:`autokmc.structure.build_surface`); these atoms are
        constrained via :class:`~ase.constraints.FixAtoms` during the
        relaxation.
    calculator : ase.calculators.calculator.Calculator
        The ML potential (NequIP, MACE, EMT, …) used for both the
        per-placement relaxation **and** the clean-surface / gas-phase
        single points (unless those energies are supplied).
    fmax, max_steps : float, int
        Convergence criterion and step cap forwarded to *optimizer*.
    optimizer : callable, optional
        ASE optimiser class.  Defaults to :class:`ase.optimize.BFGS`.
    clean_energy, gas_energy : float, optional
        Pre-computed reference energies in eV.  ``clean_energy`` is the
        bare ``atoms_template`` energy with this calculator;
        ``gas_energy`` defaults to ``reactant.energy`` if finite,
        otherwise a single point on a fresh copy of ``reactant.atoms``.
    nl_mult : float
        Neighbour-list cutoff multiplier for the connectivity check.
        Defaults to ``NL_MULT_DEFAULT`` (1.0) — same as ``build_graph``.
    only_iso_classes : list[int], optional
        If given, only relax the specified ``AdsorbateSite.iso_class``
        indices; useful for incremental / debugging runs.
    verbose : bool
        Wraps the call in :func:`verbose_scope` (DEBUG logging on).

    Notes
    -----
    **Single-atom adsorbates** (``len(reactant.atoms) == 1``) are
    handled too: if ``cache.adsorbate_sites[smiles]`` is empty, the
    function lazily calls :func:`seed_single_atom_adsorbate_sites`, which
    synthesises one :class:`AdsorbateSite` per IsoClass in
    ``cache.unique_sites[element][N_SHELLS_DEFAULT]``.  The
    relaxation, connectivity check and member propagation then run
    through the same code path as multi-atom adsorbates — connectivity
    checks 1, 2 and 4 (intramolecular bonds, spurious bonds, non-anchor
    surface contact) are vacuously true for monatomic reactants, and
    only the "anchor stays bonded to its clique" check (#3) is active.

    Returns
    -------
    list[AdsorbateSite]
        The same list stored in ``cache.adsorbate_sites[smiles]``.  Each
        entry selected for optimisation is updated in place with result
        fields such as ``stable``, ``adsorption_energy`` and
        ``member_positions``.
    """
    cache = get_cache(G)
    if (smiles not in cache.adsorbate_sites or not cache.adsorbate_sites[smiles]) \
            and len(reactant.atoms) == 1:
        seed_single_atom_adsorbate_sites(G, reactant, verbose=verbose)
    with verbose_scope(_log, verbose):
        return _run(
            G, smiles, reactant, atoms_template, calculator,
            fmax=fmax, max_steps=max_steps, optimizer=optimizer,
            clean_energy=clean_energy, gas_energy=gas_energy,
            nl_mult=nl_mult, only_iso_classes=only_iso_classes,
        )


def _run(
    G, smiles, reactant, atoms_template, calculator,
    *, fmax, max_steps, optimizer,
    clean_energy, gas_energy, nl_mult, only_iso_classes,
) -> list[Any]:
    if optimizer is None:
        from ase.optimize import BFGS
        optimizer = BFGS

    cache = get_cache(G)
    if smiles not in cache.adsorbate_sites:
        raise KeyError(
            f"No adsorbate sites enumerated for SMILES {smiles!r}; "
            "call find_adsorbate_sites first."
        )
    adsorbate_sites = cache.adsorbate_sites[smiles]
    if not adsorbate_sites:
        return adsorbate_sites

    # ── Reference energies ────────────────────────────────────────────────
    if clean_energy is None:
        clean = atoms_template.copy()
        clean.calc = calculator
        clean_energy = float(clean.get_potential_energy())
        _log.info("clean surface energy = %.4f eV", clean_energy)

    if gas_energy is None:
        if (reactant.energy is not None
                and np.isfinite(reactant.energy)):
            gas_energy = float(reactant.energy)
        else:
            gas = reactant.atoms.copy()
            gas.calc = calculator
            gas_energy = float(gas.get_potential_energy())
    _log.info("gas-phase reactant energy = %.4f eV", gas_energy)

    selected = (set(only_iso_classes)
                if only_iso_classes is not None else None)

    n_stable = 0
    n_site_changed = 0
    n_broken = 0
    site_changed_this_run: list[Any] = []
    for ms in adsorbate_sites:
        if selected is not None and ms.iso_class not in selected:
            continue

        original_positions = np.asarray(ms.positions, dtype=float).copy()
        original_member_positions = [
            (None if p is None else np.asarray(p, dtype=float).copy())
            for p in (ms.member_positions or [])
        ]

        # Default-initialise the dynamic attributes so every AdsorbateSite
        # selected for optimisation carries a uniform outcome shape.  Start
        # pessimistically: a selected site is unstable until the relaxed
        # connectivity check proves otherwise.  Its geometry is restored on
        # every non-stable outcome below.
        ms.stable = False
        ms.adsorption_energy = None
        ms.member_neighbour_atoms = None
        ms.member_subgraphs = None
        ms.relaxed_graph = None              # type: ignore[attr-defined]
        ms.relaxed_cliques = None            # type: ignore[attr-defined]
        ms.relaxed_lateral_subgraph = None   # type: ignore[attr-defined]
        ms.collapsed_into_iso_class = None   # type: ignore[attr-defined]

        try:
            combined, ads_indices = _build_combined_atoms(
                atoms_template, ms, reactant,
            )
            combined.calc = calculator
            opt = optimizer(combined, logfile=None)
            opt.run(fmax=fmax, steps=max_steps)
            E_total = float(combined.get_potential_energy())
        except Exception as exc:
            _log.warning("iso-class %d: relaxation failed (%r)",
                         ms.iso_class, exc)
            ms.positions = original_positions.copy()
            ms.member_positions = original_member_positions
            n_broken += 1
            continue

        status, reason, G_rel, actual_cliques, lateral_atoms, lateral_sub = (
            _verify_relaxed(
                combined, ads_indices, reactant, ms, nl_mult=nl_mult,
            )
        )

        if status == "broken":
            _log.info(
                "iso-class %d: discarded (%s)", ms.iso_class, reason,
            )
            ms.positions = original_positions.copy()
            ms.member_positions = original_member_positions
            n_broken += 1
            continue

        if status == "site_changed":
            _log.info(
                "iso-class %d: marked unstable due to site migration (%s)",
                ms.iso_class, reason,
            )
            ms.positions = original_positions.copy()
            ms.member_positions = original_member_positions
            # Stash the rebuilt graph + new clique pattern for the
            # downstream cross-iso-class sanity check.
            ms.relaxed_graph = G_rel
            ms.relaxed_cliques = actual_cliques
            ms.relaxed_lateral_subgraph = lateral_sub
            site_changed_this_run.append(ms)
            n_site_changed += 1
            continue

        # status == "stable"
        relaxed_ads = combined.get_positions()[ads_indices]
        E_ads = E_total - float(clean_energy) - float(gas_energy)

        ms.positions = relaxed_ads
        ms.stable = True
        ms.adsorption_energy = E_ads
        member_positions, member_neighbour, member_subgraphs = (
            _propagate_to_members(G, ms, relaxed_ads)
        )
        ms.member_positions = member_positions
        ms.member_neighbour_atoms = member_neighbour
        ms.member_subgraphs = member_subgraphs
        if ms.member_node_ids:
            from autokmc.find_adsorbate_site import push_member_positions_to_graph
            for member_index in range(len(member_positions)):
                push_member_positions_to_graph(G, ms, member_index)
        # Record the rebuilt info on stable poses too — useful for
        # downstream lateral-interaction tabulation.
        ms.relaxed_graph = G_rel
        ms.relaxed_cliques = actual_cliques
        ms.relaxed_lateral_subgraph = lateral_sub
        n_stable += 1
        _log.info(
            "iso-class %d: stable, E_ads = %+.4f eV "
            "(propagated to %d members; ego sizes = %s)",
            ms.iso_class, E_ads, len(ms.member_positions),
            [len(s) for s in member_neighbour],
        )

    # ── Cross-iso-class sanity check ─────────────────────────────────────
    stable_list = [ms for ms in adsorbate_sites if getattr(ms, "stable", False)]
    for ms in site_changed_this_run:
        if getattr(ms, "relaxed_lateral_subgraph", None) is None:
            continue
        target = _find_collapse_target(
            ms.relaxed_lateral_subgraph, stable_list,
        )
        ms.collapsed_into_iso_class = target  # type: ignore[attr-defined]
        if target is not None:
            _log.info(
                "iso-class %d collapsed into stable iso-class %d "
                "after relaxation (lateral subgraphs are isomorphic)",
                ms.iso_class, target,
            )
        else:
            _log.info(
                "iso-class %d: post-relaxation lateral subgraph does not "
                "match any stable iso-class",
                ms.iso_class,
            )

    _log.info(
        "optimise_adsorbate_sites_ml(%r): %d stable, %d site-changed, %d broken "
        "(of %d total)",
        smiles, n_stable, n_site_changed, n_broken, len(adsorbate_sites),
    )
    return adsorbate_sites


