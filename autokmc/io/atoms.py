"""ASE Atoms conversion helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

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

	atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

	# Extended-XYZ atom arrays preserve the graph identity needed to interpret
	# every KMC frame after the run, rather than only its Cartesian geometry.
	def _text(name: str, default: str = "_") -> np.ndarray:
		return np.asarray(
			[str(G.nodes[node].get(name, default)) for node in all_ids],
			dtype=str,
		)

	atoms.new_array("graph_node_id", np.asarray([str(node) for node in all_ids], dtype=str))
	atoms.new_array("node_type", _text("type"))
	atoms.new_array("reactant_smiles", _text("reactant"))
	atoms.new_array("reactant_index", _text("reactant_index"))
	atoms.new_array("site_iso_class", _text("site_iso_class"))
	atoms.new_array("site_member_index", _text("site_member_index"))
	atoms.new_array(
		"occupied",
		np.asarray([bool(G.nodes[node].get("occupied", False)) for node in all_ids]),
	)
	atoms.info["autokmc_graph_schema"] = str(G.graph.get("schema", "unknown"))
	if G.graph.get("run_id") is not None:
		atoms.info["run_id"] = str(G.graph["run_id"])

	frozen = {int(index) for index in G.graph.get("frozen_indices", [])}
	atoms.new_array(
		"frozen",
		np.asarray(
			[
				int(G.nodes[node].get("index", -1)) in frozen
				if node in slab_nodes else False
				for node in all_ids
			],
			dtype=bool,
		),
	)
	fixed_output_indices = [
		output_index
		for output_index, node in enumerate(slab_nodes)
		if int(G.nodes[node].get("index", -1)) in frozen
	]
	if fixed_output_indices:
		atoms.set_constraint(FixAtoms(indices=fixed_output_indices))
	return atoms


__all__ = ["atoms_from_graph"]
