"""Tests for core graph contracts and graph-state helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from autokmc.core.graph import build_graph
from autokmc.core.graph_state import (
    get_adsorbate_sites,
    get_bond_reaction_sites,
    get_bond_registry,
    get_clique_to_members,
    get_frozen_indices,
    get_n_occupied,
    get_occupied_by_clique,
    get_run_id,
    invalidate_position_caches,
    invalidate_surface_apsp,
    invalidate_surface_shell_cache,
    set_bond_reaction_sites,
    set_frozen_indices,
    set_n_occupied,
    set_run_id,
    validate_runtime_indexes,
)


def test_build_graph_rejects_invalid_surface_codes(tiny_atoms):
    tiny_atoms.set_array("surface", np.array([0, 1, 2, 3], dtype=np.int8))

    with pytest.raises(ValueError, match="invalid code"):
        build_graph(tiny_atoms)


def test_build_graph_rejects_wrong_surface_array_length(tiny_atoms):
    tiny_atoms.arrays["surface"] = tiny_atoms.arrays["numbers"][:2]

    with pytest.raises(ValueError, match="length does not match"):
        build_graph(tiny_atoms)


def test_build_graph_does_not_mutate_input_pbc(tiny_atoms):
    tiny_atoms.set_pbc([True, True, False])
    tiny_atoms.set_array("surface", np.array([0, 1, 2, 2], dtype=np.int8))

    graph = build_graph(tiny_atoms)

    assert tuple(tiny_atoms.pbc) == (True, True, False)
    assert tuple(graph.graph["pbc"]) == (True, True, True)


def test_graph_state_create_and_invalidate_helpers():
    G = nx.Graph()

    assert get_adsorbate_sites(G, "CO") == []
    assert get_n_occupied(G) == 0

    occupied = get_occupied_by_clique(G, create=True)
    occupied[(1, 2)] = 100
    assert get_occupied_by_clique(G) == {(1, 2): 100}

    clique_members = get_clique_to_members(G, create=True)
    clique_members[(1, 2)] = [(1, 2), (2, 3)]
    assert get_clique_to_members(G) == {(1, 2): [(1, 2), (2, 3)]}

    bond_registry = get_bond_registry(G, create=True)
    bond_registry["CO+O"] = object()
    assert "CO+O" in get_bond_registry(G)

    marker = object()
    set_bond_reaction_sites(G, [marker])
    assert get_bond_reaction_sites(G) == [marker]

    set_n_occupied(G, -5)
    assert get_n_occupied(G) == 0
    set_n_occupied(G, 2)
    assert get_n_occupied(G) == 2

    G.graph["surface_apsp"] = {}
    G.graph["_surface_shells_cache"] = {}
    G.graph["_clique_position_index_cache"] = {}
    G.graph["_surface_atoms_array_cache"] = {}

    invalidate_surface_apsp(G)
    invalidate_surface_shell_cache(G)
    invalidate_position_caches(G)

    assert "surface_apsp" not in G.graph
    assert "_surface_shells_cache" not in G.graph
    assert "_clique_position_index_cache" not in G.graph
    assert "_surface_atoms_array_cache" not in G.graph


def test_graph_runtime_identity_and_occupancy_invariants():
    graph = nx.Graph()
    clique = frozenset({1})
    graph.add_node(
        10,
        type="adsorbate",
        occupied=True,
        clique=clique,
        reactant="[O]",
        site_iso_class=0,
        site_member_index=0,
    )
    graph.graph["occupied_by_clique"] = {clique: {10}}
    graph.graph["n_occupied"] = 1

    set_run_id(graph, "run-123")
    set_frozen_indices(graph, [4, 2])

    assert get_run_id(graph) == "run-123"
    assert get_frozen_indices(graph) == [4, 2]
    validate_runtime_indexes(graph)

    graph.graph["n_occupied"] = 2
    with pytest.raises(ValueError, match="n_occupied"):
        validate_runtime_indexes(graph)

    graph.graph["n_occupied"] = 1
    graph.nodes[10]["occupied"] = False
    with pytest.raises(ValueError, match="unoccupied node"):
        validate_runtime_indexes(graph)
