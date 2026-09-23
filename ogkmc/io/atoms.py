"""ASE Atoms conversion and result-snapshot helpers."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from ogkmc.core.pbc import full_pbc_for_cell
from ogkmc.core.atom_metadata import apply_atom_metadata


def copy_atoms_with_results(
	atoms: Atoms,
	*,
	energy: float | None = None,
	forces: Any | None = None,
) -> Atoms:
	"""Return a live-calculator-free copy retaining energy and forces.

	ASE does not copy calculators in :meth:`ase.Atoms.copy`, so cached
	calculator results normally disappear when optimized structures are saved
	for later persistence.  Replace the live model with an ASE
	``SinglePointCalculator`` containing the two results used by OGKMC.
	Extended XYZ then writes its standard ``energy`` and ``forces`` fields, and
	the values remain queryable without rerunning the model.

	Explicit *energy* and *forces* take precedence.  Otherwise, valid cached
	values are copied from the attached calculator without requesting a new
	calculation.  A calculator whose state no longer matches *atoms* is ignored.
	"""
	snapshot = atoms.copy()
	snapshot.calc = None

	cached_results = {}
	calculator = atoms.calc
	if calculator is not None:
		try:
			has_result_for = getattr(calculator, "has_result_for", None)
			if callable(has_result_for) and has_result_for(atoms):
				cached_results = {
					"energy": calculator.get_potential_energy(atoms),
					"forces": calculator.get_forces(atoms),
				}
			else:
				check_state = getattr(calculator, "check_state", None)
				state_changes = check_state(atoms) if callable(check_state) else ()
				if not state_changes:
					results = getattr(calculator, "results", None)
					if isinstance(results, dict):
						cached_results = results
		except Exception:
			# Persistence must never invoke or depend on a live calculator.  If
			# its cache cannot be inspected safely, retain geometry only.
			cached_results = {}

	resolved_energy = energy
	if resolved_energy is None:
		resolved_energy = cached_results.get(
			"energy",
			cached_results.get(
				"free_energy",
				snapshot.info.get("energy"),
			),
		)
	result_payload: dict[str, Any] = {}
	if resolved_energy is not None:
		try:
			resolved_energy = float(resolved_energy)
		except (TypeError, ValueError):
			if energy is not None:
				raise ValueError("optimized-structure energy must be numeric") from None
		else:
			if not np.isfinite(resolved_energy):
				if energy is not None:
					raise ValueError("optimized-structure energy must be finite")
			else:
				result_payload["energy"] = resolved_energy

	resolved_forces = forces
	if resolved_forces is None:
		resolved_forces = cached_results.get(
			"forces",
			snapshot.arrays.get("forces"),
		)
	if resolved_forces is not None:
		try:
			force_array = np.asarray(resolved_forces, dtype=float)
		except (TypeError, ValueError):
			if forces is not None:
				raise ValueError("optimized-structure forces must be numeric") from None
		else:
			valid_shape = force_array.shape == (len(snapshot), 3)
			valid_values = bool(np.all(np.isfinite(force_array)))
			if not valid_shape and forces is not None:
				raise ValueError(
					"optimized-structure forces must have shape "
					f"({len(snapshot)}, 3), got {force_array.shape}"
				)
			if not valid_values and forces is not None:
				raise ValueError("optimized-structure forces must be finite")
			if valid_shape and valid_values:
				result_payload["forces"] = force_array.copy()

	# Avoid duplicate reserved extxyz fields if the source came from an
	# explicitly annotated Atoms object rather than an ASE reader.
	snapshot.info.pop("energy", None)
	if "forces" in snapshot.arrays:
		snapshot.set_array("forces", None)
	if result_payload:
		snapshot.calc = SinglePointCalculator(snapshot, **result_payload)

	return snapshot


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
	apply_atom_metadata(atoms, [G.nodes[node] for node in all_ids])

	# Extended-XYZ atom arrays preserve the graph identity needed to interpret
	# every KMC frame after the run, rather than only its Cartesian geometry.
	def _text(name: str, default: str = "_") -> np.ndarray:
		return np.asarray(
			[str(G.nodes[node].get(name, default)) for node in all_ids],
			dtype=str,
		)

	def _site_iso_class(node) -> str:
		data = G.nodes[node]
		return str(data.get("site_iso_class", data.get("iso_class", "_")))

	atoms.new_array("graph_node_id", np.asarray([str(node) for node in all_ids], dtype=str))
	atoms.new_array("node_type", _text("type"))
	atoms.new_array("reactant_smiles", _text("reactant"))
	atoms.new_array("reactant_index", _text("reactant_index"))
	atoms.new_array(
		"site_iso_class",
		np.asarray([_site_iso_class(node) for node in all_ids], dtype=str),
	)
	atoms.new_array("site_member_index", _text("site_member_index"))
	atoms.new_array(
		"occupied",
		np.asarray([bool(G.nodes[node].get("occupied", False)) for node in all_ids]),
	)
	atoms.info["ogkmc_graph_schema"] = str(G.graph.get("schema", "unknown"))
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
