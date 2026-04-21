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

The ``type`` attribute is read from ``atoms.arrays["surface"]`` (int8):

* ``0`` → ``"bulk"``
* ``1`` → ``"surface"``
* ``2`` → ``"adsorbate"``

Edges connect atoms whose covalent-radius neighbour-lists overlap (ASE
:class:`~ase.neighborlist.NeighborList` with ``mult=1.1``).

The graph also carries cell-level metadata as :attr:`~networkx.Graph.graph`
attributes: ``"cell"``, ``"pbc"``.

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

    return G
