"""
autokmc.graph
=============
Build an atom-connectivity graph from an :class:`~ase.Atoms` object.

Each node represents one atom and carries:

* ``element``        – chemical symbol (str)
* ``position``       – Cartesian coordinates (np.ndarray, shape (3,))
* ``index``          – atom index in the original :class:`~ase.Atoms` object (int)
* ``type``           – one of ``"bulk"``, ``"surface"``, or ``"adsorbate"`` (str)
* ``covalent_radius``– covalent radius in Å from ASE data (float)

For **surface** nodes, additional attributes are stored after
:func:`_annotate_surface_shells` runs:

* ``surf_wl_k{1..4}``  – WL-refinement hash at depth k on the surface-only
                          subgraph (str).  Two surface atoms with equal hashes
                          at depth k have the same chemical environment out to
                          k bond-hops, ignoring all bulk/adsorbate atoms.
* ``surf_nn_k{1..4}``  – minimum Cartesian distance (Å) from this node to any
                          surface atom exactly k hops away (float).

These are used by :func:`~autokmc.site.find_adsorption_sites` to restrict
probe-grid sampling to one representative per surface equivalence class.

Typical usage
-------------
::

    from autokmc.surface import find_surface_atoms
    from autokmc.graph import build_graph
    from ase.build import fcc111

    slab = fcc111("Cu", size=(4, 4, 4), vacuum=10.0, periodic=True)
    find_surface_atoms(slab, tag_atoms=True)   # writes slab.arrays["surface"]
    G = build_graph(slab)

    surface_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]

Public API
----------
* :func:`build_graph`       – main entry point
* :func:`surface_subgraph`  – induced subgraph of surface nodes
* :func:`ego_graph`         – node + its 1st-shell neighbours
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.neighborlist import NeighborList, natural_cutoffs


# ---------------------------------------------------------------------------
# Surface-shell annotation
# ---------------------------------------------------------------------------

def _annotate_surface_shells(G: nx.Graph, k_max: int = 4) -> None:
    """Annotate every surface node with WL hashes and shell distances.

    Only surface-to-surface edges are traversed, so bulk/adsorbate atoms
    never influence the classification.

    Attributes written
    ------------------
    ``surf_wl_k{k}`` (str)
        WL-refinement label after *k* iterations on the surface-only
        subgraph.  Atoms with equal labels at depth *k* have identical
        chemical environments out to *k* bond-hops on the surface.
    ``surf_nn_k{k}`` (float)
        Minimum Cartesian distance (Å) to surface atoms at exactly *k*
        hops from this node (the *k*-th coordination shell).

    Graph-level summary
    -------------------
    ``G.graph["surf_nn_k{k}"]`` stores the minimum of ``surf_nn_k{k}``
    over all surface atoms, allowing quick shell-distance look-up.
    """
    surf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]
    if len(surf_nodes) < 2:
        return

    surf_set  = set(surf_nodes)
    positions = {n: G.nodes[n]["position"] for n in surf_nodes}

    # Surface-only adjacency — never traverse bulk or adsorbate edges
    surf_adj: dict[int, list[int]] = {
        n: [u for u in G.neighbors(n) if u in surf_set]
        for n in surf_nodes
    }

    # ── Per-node BFS: record min distance to atoms at each shell ─────────
    for n in surf_nodes:
        visited: set[int] = {n}
        frontier: list[int] = [n]
        for hop in range(1, k_max + 1):
            next_frontier: list[int] = []
            for v in frontier:
                for u in surf_adj[v]:
                    if u not in visited:
                        visited.add(u)
                        next_frontier.append(u)
            if next_frontier:
                d_min = float(min(
                    np.linalg.norm(positions[n] - positions[u])
                    for u in next_frontier
                ))
                G.nodes[n][f"surf_nn_k{hop}"] = d_min
            frontier = next_frontier

    # ── WL refinement: k iterations on surface-only adjacency ────────────
    wl: dict[int, str] = {n: G.nodes[n]["element"] for n in surf_nodes}
    for k in range(1, k_max + 1):
        new_wl: dict[int, str] = {}
        for n in surf_nodes:
            nbr = tuple(sorted(wl[u] for u in surf_adj[n]))
            new_wl[n] = str((wl[n], nbr))
        wl = new_wl
        for n in surf_nodes:
            G.nodes[n][f"surf_wl_k{k}"] = wl[n]

    # ── Graph-level summary ───────────────────────────────────────────────
    for k in range(1, k_max + 1):
        vals = [G.nodes[n][f"surf_nn_k{k}"]
                for n in surf_nodes if f"surf_nn_k{k}" in G.nodes[n]]
        if vals:
            G.graph[f"surf_nn_k{k}"] = float(min(vals))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_graph(
    atoms: Atoms,
    *,
    nl_mult: float = 1.0,
) -> nx.Graph:
    """Build an atom-connectivity graph from *atoms*.

    The surface classification is read from ``atoms.arrays["surface"]``,
    which must have been written beforehand by
    :func:`~autokmc.surface.tag_surface_atoms` or by calling
    :func:`~autokmc.surface.find_surface_atoms` with ``tag_atoms=True``.

    Parameters
    ----------
    atoms : Atoms
        The structure to graph.  Must have ``atoms.arrays["surface"]``
        populated (int8 array: 0 = bulk, 1 = surface).
    nl_mult : float
        Multiplier for ASE :func:`~ase.neighborlist.natural_cutoffs` used
        to determine which atom pairs are bonded.  Default 1.1.

    Returns
    -------
    G : nx.Graph
        Connectivity graph.

        **Node attributes** (every node):

        =========  ================================================
        element          Chemical symbol (str)
        position         Cartesian coordinates – np.ndarray, shape (3,)
        index            Atom index in *atoms* (int)
        type             ``"bulk"``, ``"surface"``, or ``"adsorbate"`` (str)
        covalent_radius  Covalent radius in Å from ASE data (float)
        =========  ================================================

        **Graph attributes**:

        ====  =============================================
        cell  Unit-cell matrix – np.ndarray, shape (3, 3)
        pbc   Periodic boundary conditions – list[bool]
        ====  =============================================

    Raises
    ------
    KeyError
        If ``atoms.arrays["surface"]`` does not exist.  Run
        ``find_surface_atoms(atoms, tag_atoms=True)`` first.
    """
    if "surface" not in atoms.arrays:
        raise KeyError(
            'atoms.arrays["surface"] not found.  '
            "Run find_surface_atoms(atoms, tag_atoms=True) before building the graph."
        )

    surface_mask = atoms.arrays["surface"]   # int8: 0=bulk, 1=surface, 2=adsorbate

    _TYPE_MAP = {0: "bulk", 1: "surface", 2: "adsorbate"}

    # ------------------------------------------------------------------
    # Build neighbour list
    # ------------------------------------------------------------------
    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    G = nx.Graph()

    G.graph["cell"] = np.array(atoms.get_cell())
    G.graph["pbc"]  = atoms.get_pbc().tolist()

    positions      = atoms.get_positions()
    symbols        = atoms.get_chemical_symbols()
    atomic_numbers = atoms.get_atomic_numbers()

    for i in range(len(atoms)):
        G.add_node(
            i,
            element         = symbols[i],
            position        = positions[i].copy(),
            index           = i,
            type            = _TYPE_MAP.get(int(surface_mask[i]), "bulk"),
            covalent_radius = float(ASE_COVALENT_RADII[atomic_numbers[i]]),
        )

    for i in range(len(atoms)):
        neighbours, _ = nl.get_neighbors(i)
        for j in map(int, neighbours):
            if j > i:
                G.add_edge(i, j)

    # Annotate surface nodes with WL hashes and shell distances.
    # Used by find_adsorption_sites to restrict probe-grid sampling.
    _annotate_surface_shells(G)

    return G
