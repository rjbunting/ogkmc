"""Build atom-connectivity graphs from :class:`~ase.Atoms` objects.

Each node represents one atom and carries:

* ``element``        – chemical symbol (str)
* ``position``       – Cartesian coordinates (np.ndarray, shape (3,))
* ``index``          – atom index in the original :class:`~ase.Atoms` object (int)
* ``type``           – one of ``"bulk"``, ``"surface"``, ``"adsorbate"``,
  or ``"anchor"`` (str).  The first three are written here from
  ``atoms.arrays["surface"]``; ``"anchor"`` nodes are added later by
  :func:`autokmc.sites.anchors.find_anchor_sites` (one node per
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
``mult=autokmc.core.constants.NL_MULT_DEFAULT``).

Graph-level metadata (``G.graph[...]``):

* ``"cell"`` – :class:`numpy.ndarray`, shape ``(3, 3)`` (rows = lattice vectors).
* ``"pbc"``  – :class:`numpy.ndarray` of three :class:`bool`.  Material
  structures with a real cell are stored as fully periodic
  (``[True, True, True]``); adsorbate-only gas reactants stay non-periodic.
* ``"connectivity_pbc"`` – :class:`numpy.ndarray` of three :class:`bool`.
  Derived from the neighbour-list — an axis is True iff at least one bond
  crosses the cell image along it.
* ``"hull_equations"`` – ``(n_facets, 4)`` convex-hull equations array,
  present only for nanoparticle structures.
"""

from __future__ import annotations

import warnings

import numpy as np
import networkx as nx

from ase import Atoms
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.neighborlist import NeighborList, natural_cutoffs

from autokmc.core.constants import NEIGHBORLIST_SKIN, NL_MULT_DEFAULT
from autokmc.core.pbc import graph_pbc_for_atoms
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)

_TYPE_MAP = {0: "bulk", 1: "surface", 2: "adsorbate"}
_VALID_SURFACE_CODES = frozenset(_TYPE_MAP)


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
    :func:`autokmc.structure.tag_surface_atoms` or by calling
    :func:`autokmc.structure.find_surface_atoms` with ``tag_atoms=True``.

    Parameters
    ----------
    atoms : Atoms
        The structure to graph.  Must have ``atoms.arrays["surface"]``
        populated (int8 array: 0 = bulk, 1 = surface, 2 = adsorbate).
    nl_mult : float
        Multiplier for ASE :func:`~ase.neighborlist.natural_cutoffs` used
        to determine which atom pairs are bonded.  Defaults to
        :data:`autokmc.core.constants.NL_MULT_DEFAULT` (currently ``1.0``).

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
    if len(surface_mask) != len(atoms):
        raise ValueError(
            'atoms.arrays["surface"] length does not match atoms: '
            f"{len(surface_mask)} != {len(atoms)}"
        )
    surface_codes = {int(v) for v in np.asarray(surface_mask).ravel()}
    invalid_codes = sorted(surface_codes - _VALID_SURFACE_CODES)
    if invalid_codes:
        raise ValueError(
            'atoms.arrays["surface"] contains invalid code(s) '
            f"{invalid_codes}; expected only 0=bulk, 1=surface, 2=adsorbate"
        )

    declared_pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    # Neighbor-list construction needs the material cell to be fully periodic,
    # but callers must not see their Atoms object mutated as a side effect.
    atoms = atoms.copy()
    atoms.arrays["surface"] = np.asarray(surface_mask).copy()
    graph_pbc = graph_pbc_for_atoms(atoms)
    atoms.set_pbc(graph_pbc)

    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(
        cutoffs,
        skin=NEIGHBORLIST_SKIN,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)

    G = nx.Graph()

    cell_arr = np.array(atoms.get_cell(), dtype=float)
    G.graph["cell"] = cell_arr
    G.graph["pbc"] = graph_pbc.copy()

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

    # Track per-axis whether *any* bond crosses an image. This is diagnostic
    # connectivity metadata; material graph structures keep full PBC when
    # they have a real cell, while adsorbate-only reactants stay non-periodic.
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

    G.graph["connectivity_pbc"] = pbc_effective

    # Warn if the caller's original PBC declaration hid cross-image bonds.
    # Material structures with a real cell are stored with full PBC, but this
    # still catches too-small vacuum gaps or cells in inputs that arrived with
    # a partially/non-periodic PBC setting.
    if not np.array_equal(declared_pbc, pbc_effective):
        unexpected = (~declared_pbc) & pbc_effective
        if unexpected.any():
            warnings.warn(
                f"build_graph: cross-image bonds detected along axes "
                f"{np.where(unexpected)[0].tolist()} where input atoms.pbc was "
                f"{declared_pbc.tolist()}.  Connectivity pbc is "
                f"{pbc_effective.tolist()}.  This is usually a vacuum-gap "
                f"or cell-size bug.",
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
        G.number_of_nodes(), G.number_of_edges(), G.graph["pbc"].tolist(), nl_mult,
    )

    return G
