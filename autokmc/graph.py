"""
autokmc.graph
=============
Build an atom-connectivity graph from an :class:`~ase.Atoms` object.

Each node represents one atom and carries:

* ``element``        – chemical symbol (str)
* ``position``       – Cartesian coordinates (np.ndarray, shape (3,))
* ``index``          – atom index in the original :class:`~ase.Atoms` object (int)
* ``type``           – one of ``"bulk"``, ``"surface"``, ``"adsorbate"``,
  or ``"anchor"`` (str).  The first three are written here from
  ``atoms.arrays["surface"]``; ``"anchor"`` nodes are added later by
  :func:`autokmc.find_anchors.find_anchor_sites` (one node per
  raw site clique).
* ``covalent_radius``– covalent radius in Å from ASE data (float)

Each edge carries:

* ``distance`` – minimum-image bond length in Å (float)
* ``offset``   – integer cell-image offset ``(i, j, k)`` of the partner
  atom; ``(0, 0, 0)`` for in-cell bonds (tuple[int, int, int])

The ``type`` attribute is read from ``atoms.arrays["surface"]`` (int8):

* ``0`` → ``"bulk"``
* ``1`` → ``"surface"``
* ``2`` → ``"adsorbate"``

Edges connect atoms whose covalent-radius neighbour-lists overlap (ASE
:class:`~ase.neighborlist.NeighborList` with
``mult=autokmc.constants.NL_MULT_DEFAULT``).

Graph-level metadata (``G.graph[...]``):

* ``"cell"`` – :class:`numpy.ndarray`, shape ``(3, 3)`` (rows = lattice vectors).
* ``"pbc"``  – :class:`numpy.ndarray` of three :class:`bool`.  **Derived** from
  the neighbour-list — an axis is True iff at least one bond crosses
  the cell image along it (so a nanoparticle in a periodic cubic cell
  with sufficient vacuum reports ``pbc=array([False, False, False])``).
* ``"hull_equations"`` – ``(n_facets, 4)`` convex-hull equations array,
  present only for nanoparticle structures.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import networkx as nx

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.neighborlist import NeighborList, natural_cutoffs

from autokmc.constants import NL_MULT_DEFAULT

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_graph(
    atoms: Atoms,
    *,
    nl_mult: float = NL_MULT_DEFAULT,
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
        populated (int8 array: 0 = bulk, 1 = surface, 2 = adsorbate).
    nl_mult : float
        Multiplier for ASE :func:`~ase.neighborlist.natural_cutoffs` used
        to determine which atom pairs are bonded.  Defaults to
        :data:`autokmc.constants.NL_MULT_DEFAULT` (currently ``1.0``).

    Returns
    -------
    G : nx.Graph
        Connectivity graph (see module docstring for full attribute list).

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

    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    G = nx.Graph()

    cell_arr = np.array(atoms.get_cell(), dtype=float)
    G.graph["cell"] = cell_arr

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

    # Walk the bothways=True neighbour list and add each bond once (i<j),
    # recording the integer cell offset of the partner atom and the MIC
    # bond distance so downstream code can avoid re-deriving them.
    for i in range(len(atoms)):
        neighbours, offsets = nl.get_neighbors(i)
        for jj, off in zip(map(int, neighbours), offsets):
            off = np.asarray(off, dtype=int)
            if np.any(off != 0):
                pbc_effective |= (off != 0)
            if jj <= i:
                continue
            # Cartesian displacement of the partner image relative to atom i.
            dv = positions[jj] + off @ cell_arr - positions[i]
            d = float(np.linalg.norm(dv))
            G.add_edge(i, jj,
                       distance=d,
                       offset=(int(off[0]), int(off[1]), int(off[2])))

    G.graph["pbc"] = pbc_effective

    # Warn if user-declared PBC disagrees with what the bonding-derived
    # effective PBC says.  Common causes: a slab's vacuum gap is too small
    # so atoms bond across z (False→True), or a nanoparticle is centred in
    # a too-small periodic cell (False→True), or a nominally periodic axis
    # has no inter-image bonds because the cell vector is huge (True→False
    # — usually fine and intentional).  Only the first case is a real bug.
    user_pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not np.array_equal(user_pbc, pbc_effective):
        # We only warn for the dangerous direction (user said no-PBC but
        # bonding says yes).  The other direction is the documented NP
        # convention and is silent.
        unexpected = (~user_pbc) & pbc_effective
        if unexpected.any():
            warnings.warn(
                f"build_graph: cross-image bonds detected along axes "
                f"{np.where(unexpected)[0].tolist()} where atoms.pbc was "
                f"{user_pbc.tolist()}.  Effective pbc is "
                f"{pbc_effective.tolist()}.  This is usually a vacuum-gap "
                f"or cell-size bug; downstream code uses G.graph['pbc'].",
                RuntimeWarning,
                stacklevel=2,
            )

    # Re-use any hull computed by find_surface_atoms (nanoparticle path) so
    # downstream code need not rebuild it.  Stored as the (n_facets, 4)
    # equations array on the graph directly.
    hull_eq = atoms.info.get("_hull_equations")
    if hull_eq is not None:
        G.graph["hull_equations"] = np.asarray(hull_eq, dtype=float)

    _log.debug(
        "build_graph: %d nodes, %d edges, pbc=%s, nl_mult=%g",
        G.number_of_nodes(), G.number_of_edges(), pbc_effective.tolist(), nl_mult,
    )

    return G
