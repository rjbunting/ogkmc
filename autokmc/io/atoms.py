"""ASE Atoms conversion helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
from ase import Atoms

from autokmc.core.pbc import full_pbc_for_cell


def atoms_from_graph(G: nx.Graph) -> Atoms:
	"""Build an :class:`~ase.Atoms` snapshot from the live graph state.

	Includes slab nodes (``type`` in ``{"bulk", "surface"}``) plus occupied
	adsorbate nodes. Anchor bookkeeping nodes are skipped.
	"""
	slab_nodes = sorted(
		(n for n, d in G.nodes(data=True)
		 if d.get("type") in ("bulk", "surface")),
		key=lambda n: G.nodes[n].get("index", n),
	)
	ads_nodes = sorted(
		n for n, d in G.nodes(data=True)
		if d.get("type") == "adsorbate" and d.get("occupied", False)
	)

	all_ids = slab_nodes + ads_nodes
	symbols = [G.nodes[n]["element"] for n in all_ids]
	positions = [G.nodes[n]["position"] for n in all_ids]

	cell = np.array(G.graph["cell"], dtype=float)
	if slab_nodes:
		pbc = full_pbc_for_cell(cell)
	else:
		pbc = np.asarray(G.graph.get("pbc", full_pbc_for_cell(cell)), dtype=bool)

	return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)


__all__ = ["atoms_from_graph"]
