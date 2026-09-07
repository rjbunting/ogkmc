"""Preserve per-atom calculator inputs across graph and ASE layouts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

# Geometry, topology bookkeeping and computed results have separate owners.
_EXCLUDED_ARRAYS = frozenset({
    "numbers", "positions", "surface", "forces", "energies", "stresses",
    "graph_node_id", "node_type", "reactant_smiles", "reactant_index",
    "site_iso_class", "site_member_index", "occupied", "frozen",
})


def atom_metadata(atoms: Atoms, index: int) -> dict[str, Any]:
    """Copy one atom's explicit input arrays, including isotope masses."""
    return {
        name: np.array(values[index], copy=True)
        for name, values in atoms.arrays.items()
        if name not in _EXCLUDED_ARRAYS
    }


def node_mass(data: Mapping[str, Any]) -> float:
    arrays = data.get("atom_arrays", {})
    return float(arrays.get(
        "masses", atomic_masses[atomic_numbers.get(str(data.get("element", "X")), 0)],
    ))


def atom_metadata_key(data: Mapping[str, Any]) -> str:
    """Hashable physical identity, treating absent ASE defaults consistently."""
    if "atom_metadata_key" in data:
        return str(data["atom_metadata_key"])
    arrays = dict(data.get("atom_arrays", {}))
    arrays.setdefault("masses", node_mass(data))
    arrays.setdefault("initial_charges", 0.0)
    arrays.setdefault("initial_magmoms", 0.0)
    arrays.setdefault("tags", 0)
    return json.dumps(
        {name: np.asarray(value).tolist() for name, value in arrays.items()},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def physical_node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("element") == right.get("element")
        and atom_metadata_key(left) == atom_metadata_key(right)
    )


def apply_atom_metadata(atoms: Atoms, nodes: Sequence[Mapping[str, Any]]) -> None:
    """Restore input arrays in the caller's exact node-to-atom order.

    Missing masses use ASE elemental defaults; other missing entries use zero
    (or empty strings), matching ASE's concatenation of heterogeneous Atoms.
    Scalar magnetic moments are along z when combined with vector moments.
    """
    if len(nodes) != len(atoms):
        raise ValueError("Atom metadata requires one source node per output atom")
    payloads = [node.get("atom_arrays", {}) for node in nodes]
    names = set().union(*(payload.keys() for payload in payloads)) if payloads else set()
    for name in set(atoms.arrays) - _EXCLUDED_ARRAYS - names:
        atoms.set_array(name, None)
    for name in sorted(names - _EXCLUDED_ARRAYS):
        examples = [np.asarray(payload[name]) for payload in payloads if name in payload]
        dtype = np.result_type(*(value.dtype for value in examples))
        if name in {"masses", "initial_charges", "initial_magmoms"}:
            dtype = np.result_type(dtype, float)
        shapes = {value.shape for value in examples}
        if name == "initial_magmoms" and shapes <= {(), (3,)}:
            shape = (3,) if (3,) in shapes else ()
        elif len(shapes) == 1:
            shape = next(iter(shapes))
        else:
            raise ValueError(f"Incompatible per-atom shapes for {name}: {shapes}")
        values = []
        for node, payload in zip(nodes, payloads):
            if name in payload:
                value = np.asarray(payload[name], dtype=dtype)
            elif name == "masses":
                value = np.asarray(node_mass(node), dtype=dtype)
            else:
                value = np.zeros(shape, dtype=dtype)
            if name == "initial_magmoms" and shape == (3,) and value.shape == ():
                value = np.asarray([0, 0, value.item()], dtype=dtype)
            values.append(value)
        # ASE set_array cannot change an existing scalar/vector trailing shape.
        atoms.set_array(name, None)
        atoms.set_array(name, np.asarray(values, dtype=dtype))
