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
:class:`~ase.neighborlist.NeighborList` with ``mult=1.0``).

The graph also carries cell-level metadata as :attr:`~networkx.Graph.graph`
attributes: ``"cell"``, ``"pbc"``.  Note that ``"pbc"`` is **derived** from
the neighbour-list — an axis is reported periodic only if at least one bond
crosses the cell image along it.  This means a nanoparticle sitting in a
periodic cubic cell with sufficient vacuum reports ``pbc=[False, False, False]``
even when ``atoms.get_pbc()`` is all True.

Typical usage
-------------
::

    from autokmc.surface import find_surface_atoms
    from autokmc.graph import build_graph
    from autokmc.default_sites import compute_sites_for_element
    from ase.build import fcc111

    slab = fcc111("Cu", size=(4, 4, 4), vacuum=10.0, periodic=True)
    find_surface_atoms(slab, tag_atoms=True)   # writes slab.arrays["surface"]
    G = build_graph(slab)
    compute_sites_for_element(G, "O", verbose=True)
    # G.graph["sites"]["O"] now holds the unique site iso-classes

Public API
----------
* :func:`build_graph` – main entry point

Backward-compatible re-exports (from :mod:`autokmc.default_sites`)
-------------------------------------------------------------------
* :class:`~autokmc.default_sites.SiteClass`
* :func:`~autokmc.default_sites.find_unique_surface_sites`
* :func:`~autokmc.default_sites.k_max_for_radius`
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.neighborlist import NeighborList, natural_cutoffs

# Re-export for backward compatibility (site.py imports these from graph)
from autokmc.default_sites import (  # noqa: F401
    k_max_for_radius,
)


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
        pbc   Effective periodic boundary conditions – list[bool],
              length 3.  An axis is True iff at least one bond crosses
              the cell image along that axis (derived from the neighbour
              list, *not* read from ``atoms.get_pbc()``).
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
    # Use the atoms object's own PBC flag for the neighbour search (this is
    # what controls whether ASE looks across cell images at all).  We then
    # *derive* the effective periodicity from whether any bond actually
    # crosses an image (offset != 0) per axis — this means a nanoparticle
    # placed in a periodic cubic cell with sufficient vacuum will correctly
    # report pbc=[False, False, False] even though atoms.pbc=[True]*3.
    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    G = nx.Graph()

    G.graph["cell"] = np.array(atoms.get_cell())

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

    # Track per-axis whether *any* bond crosses an image — this defines the
    # graph's effective periodicity.
    pbc_effective = np.zeros(3, dtype=bool)

    for i in range(len(atoms)):
        neighbours, offsets = nl.get_neighbors(i)
        for j, off in zip(map(int, neighbours), offsets):
            if j > i:
                G.add_edge(i, j)
            # Detect cross-image bond regardless of i,j ordering so we don't
            # miss anything in the bothways=True list.
            off = np.asarray(off, dtype=int)
            if np.any(off != 0):
                pbc_effective |= (off != 0)

    G.graph["pbc"] = pbc_effective.tolist()

    return G
