"""Physical atom inputs survive topology, calculation layouts, and cache keys."""

import json
import pickle
from types import SimpleNamespace

from ase import Atoms
from ase.vibrations import VibrationsData
import networkx as nx
import numpy as np
import pytest

from autokmc.core.atom_metadata import apply_atom_metadata, atom_metadata, atom_metadata_key
from autokmc.core.graph import build_graph
from autokmc.io.atoms import atoms_from_graph
from autokmc.io.reaction_graph import (
    reaction_graph_from_payload, reaction_graph_payload, reaction_graphs_isomorphic,
)
from autokmc.sites.adsorbate import (
    AdsorbateSite, _build_pruning_atoms, _materialise_adsorbate_nodes,
)
from autokmc.sites.bond import _build_ab_pruning_atoms
from autokmc.sites.stability.adsorption import _build_stability_atoms, check_adsorbate_site_lateral
from autokmc.sites.stability.bond import (
    _align_gas_product_to_target, _build_bond_atoms, _gas_product_neb_endpoint,
    _select_c_to_ab_mapping,
)
from autokmc.sites.stability.diffusion import _build_diffusion_atoms
from autokmc.species.reactant import build_reactant


def _system():
    atoms = Atoms(
        "FeFeHHHHC", positions=[[i * 1.5, 0, 0] for i in range(7)],
        cell=np.eye(3) * 25, pbc=True,
    )
    atoms.set_masses([57, 58, 1.007825, 2.014102, 2.014102, 1.007825, 13])
    atoms.set_initial_magnetic_moments([2.2, -2.2, 0.1, 0.2, 0.3, 0.4, 0.5])
    atoms.set_initial_charges([0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.0])
    atoms.set_tags([11, 12, 21, 22, 23, 24, 31])
    atoms.new_array("custom_input", np.arange(14).reshape(7, 2))
    atoms.new_array("surface", np.array([0, 1, 2, 2, 2, 2, 2]))
    graph = build_graph(atoms)
    for node in range(2, 7):
        graph.nodes[node].update(
            occupied=node in (2, 3, 6), reactant="[H]" if node < 6 else "[C]",
            reactant_index=0, iso_class=0, clique=frozenset({1}), siblings=(),
        )
    graph.nodes[4].update(reactant_index=0, siblings=(5,))
    graph.nodes[5].update(reactant_index=1, siblings=(4,))
    lateral = SimpleNamespace(ego_graph=graph.subgraph([0, 1, 6]).copy())
    return atoms, graph, lateral


def _assert_metadata(actual, source, order):
    for name in ("masses", "initial_magmoms", "initial_charges", "tags", "custom_input"):
        np.testing.assert_array_equal(actual.arrays[name], source.arrays[name][order])


def _site(graph, nodes):
    return AdsorbateSite(
        reactant="[H]", n_atoms=len(nodes), iso_class=0,
        atom_cliques=[frozenset({1})] * len(nodes),
        positions=np.array([graph.nodes[n]["position"] for n in nodes]),
        members=[[frozenset({1})] * len(nodes)], member_node_ids=[list(nodes)],
    )


def test_graph_snapshot_and_checkpoint_preserve_atomic_inputs():
    source, graph, _ = _system()
    # Simulate graph persistence and a different graph insertion order.
    restored = pickle.loads(pickle.dumps(graph))
    reordered = nx.Graph(**restored.graph)
    reordered.add_nodes_from(reversed(list(restored.nodes(data=True))))
    reordered.add_edges_from(restored.edges(data=True))
    _assert_metadata(atoms_from_graph(reordered), source, [0, 1, 2, 3, 6])
    source.arrays["masses"][0] = 99
    assert graph.nodes[0]["atom_arrays"]["masses"] == 57  # no alias into caller arrays


def test_every_surface_calculation_layout_preserves_atom_inputs():
    source, graph, lateral = _system()
    _assert_metadata(_build_stability_atoms(graph, lateral, frozenset({3}), include_self=True)[0],
                     source, [0, 1, 6, 3])
    _assert_metadata(_build_stability_atoms(graph, lateral, frozenset({3}), include_self=False)[0],
                     source, [0, 1, 6])
    site_a, site_b = _site(graph, [2]), _site(graph, [3])
    _assert_metadata(_build_pruning_atoms(graph, site_a, ["H"])[0], source, [0, 1, 2])
    _assert_metadata(_build_ab_pruning_atoms(graph, site_a, 0, site_b, 0, ["H"], ["H"])[0],
                     source, [0, 1, 2, 3])
    diffusion_a = _build_diffusion_atoms(graph, lateral, [2], [5], endpoint_position="a")[0]
    _assert_metadata(diffusion_a, source, [0, 1, 6, 2])
    for base in (None, diffusion_a):
        diffusion_b = _build_diffusion_atoms(
            graph, lateral, [2], [5], endpoint_position="b", base_atoms=base,
        )[0]
        _assert_metadata(diffusion_b, source, [0, 1, 6, 2])
        np.testing.assert_allclose(diffusion_b.positions[-1], source.positions[5])
    bond_ab = _build_bond_atoms(graph, lateral, [2], [3], [4, 5], endpoint="ab")[0]
    _assert_metadata(bond_ab, source, [0, 1, 6, 2, 3])
    for base in (None, bond_ab):
        bond_c = _build_bond_atoms(
            graph, lateral, [2], [3], [4, 5], endpoint="c", c_node_order=[5, 4], base_atoms=base,
        )[0]
        _assert_metadata(bond_c, source, [0, 1, 6, 5, 4])
        np.testing.assert_array_equal(bond_c.get_masses(), bond_ab.get_masses())


def test_isotope_smiles_masses_reach_materialized_adsorption_and_vibrations():
    heavy = build_reactant("[2H][2H]", add_hydrogens=False, relax=False)
    assert heavy.atoms.get_masses() == pytest.approx([2.014101778, 2.014101778])
    slab = Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3) * 20)
    slab.new_array("surface", np.array([1]))
    graph = build_graph(slab)
    site = AdsorbateSite(
        reactant=heavy.smiles, n_atoms=2, iso_class=0,
        atom_cliques=[frozenset({0}), None],
        positions=np.array([[0, 0, 2], [0, 0, 2.75]]),
        members=[[frozenset({0}), None]],
    )
    _materialise_adsorbate_nodes(graph, heavy, [site])
    atoms = _build_pruning_atoms(graph, site, ["H", "H"])[0]
    assert atoms.get_masses()[1:] == pytest.approx(heavy.atoms.get_masses())
    light = atoms.copy()
    light.set_masses([55.845, 1.008, 1.008])
    hessian = np.eye(6)
    heavy_energies = VibrationsData.from_2d(atoms, hessian, indices=[1, 2]).get_energies()
    light_energies = VibrationsData.from_2d(light, hessian, indices=[1, 2]).get_energies()
    assert heavy_energies / light_energies == pytest.approx(np.sqrt(1.008 / 2.014101778))
    mixed = build_reactant("[H][2H]", add_hydrogens=False, relax=False)
    assert sorted(map(len, mixed.unique_nodes["H"])) == [1, 1]


@pytest.mark.parametrize("method", ["auto", "greedy", "hungarian", "symmetry_trials"])
def test_bond_correspondence_cannot_swap_isotopes_to_shorten_path(method):
    _, graph, _ = _system()
    graph.remove_edges_from(list(graph.edges()))
    # The wrong isotope is spatially closest at each target slot.
    graph.nodes[4]["position"] = graph.nodes[2]["position"].copy()
    graph.nodes[5]["position"] = graph.nodes[3]["position"].copy()
    order, _ = _select_c_to_ab_mapping(
        graph, ["H", "H"], [graph.nodes[n]["position"] for n in [2, 3]], [4, 5],
        atom_matching=method, matching_trials=20, ab_node_order=[2, 3],
    )
    assert order == [5, 4]


def test_gas_product_correspondence_conserves_isotopes():
    _, order, _ = _align_gas_product_to_target(
        ["H", "H"], np.array([[-1, 0, 0], [1, 0, 0]]),
        ["H", "H"], np.array([[-1, 0, 0], [1, 0, 0]]),
        gas_masses=[2.014102, 1.007825], target_masses=[1.007825, 2.014102],
    )
    assert order == [1, 0]
    with pytest.raises(ValueError, match="isotope"):
        _align_gas_product_to_target(
            ["H"], np.zeros((1, 3)), ["H"], np.zeros((1, 3)),
            gas_masses=[2.014102], target_masses=[1.007825],
        )


@pytest.mark.parametrize("field,value", [
    ("masses", 57.0), ("initial_magmoms", 2.2), ("initial_charges", 0.2), ("tags", 3),
])
def test_lateral_and_persistent_cache_identity_includes_atom_inputs(field, value):
    graph = nx.Graph(cell=np.eye(3) * 20)
    for node in (0, 1):
        graph.add_node(node, type="surface", element="Fe")
        graph.add_node(node + 2, type="adsorbate", element="H", reactant="[H]",
                       iso_class=0, occupied=False, clique=frozenset({node}), siblings=(),
                       is_bonded=True)
        graph.add_edge(node, node + 2)
    graph.nodes[1]["atom_arrays"] = {field: value}
    site = _site(graph, [])
    site.members = [[frozenset({0})], [frozenset({1})]]
    site.member_node_ids = [[2], [3]]
    first = check_adsorbate_site_lateral(graph, site, 0, n_shells=0)
    second = check_adsorbate_site_lateral(graph, site, 1, n_shells=0)
    assert first is not second
    payload = json.loads(json.dumps(reaction_graph_payload(second.ego_graph)))
    restored = reaction_graph_from_payload(payload)
    assert reaction_graphs_isomorphic(second.ego_graph, restored)
    assert not reaction_graphs_isomorphic(first.ego_graph, restored)


def test_defaults_and_mixed_vector_magnetic_moments():
    default = {"element": "H"}
    explicit = {"element": "H", "atom_arrays": {
        "masses": 1.008, "initial_charges": 0.0, "initial_magmoms": 0.0, "tags": 0,
    }}
    assert atom_metadata_key(default) == atom_metadata_key(explicit)
    scalar, vector = Atoms("Fe"), Atoms("H")
    scalar.set_initial_magnetic_moments([2.2])
    vector.set_initial_magnetic_moments([[1.0, 2.0, 3.0]])
    output = Atoms("FeH")
    output.set_initial_magnetic_moments([0, 0])
    apply_atom_metadata(output, [
        {"element": "Fe", "atom_arrays": atom_metadata(scalar, 0)},
        {"element": "H", "atom_arrays": atom_metadata(vector, 0)},
    ])
    np.testing.assert_allclose(output.get_initial_magnetic_moments(), [[0, 0, 2.2], [1, 2, 3]])


def test_warm_start_cannot_retain_inputs_absent_from_current_graph():
    atoms = Atoms("FeH")
    atoms.set_initial_magnetic_moments([2.2, 1.0])
    atoms.new_array("obsolete_input", np.array([1, 2]))
    apply_atom_metadata(atoms, [{"element": "Fe"}, {"element": "H", "atom_arrays": {"masses": 2}}])
    assert "obsolete_input" not in atoms.arrays
    np.testing.assert_array_equal(atoms.get_initial_magnetic_moments(), [0, 0])
    np.testing.assert_allclose(atoms.get_masses(), [55.845, 2])


def test_gas_forming_bond_endpoint_retains_ordered_gas_inputs():
    source, graph, lateral = _system()
    graph.remove_edges_from(list(graph.edges()))
    atoms_ab, n_slab, n_lat, _, _, react_nodes = _build_bond_atoms(
        graph, lateral, [2], [3], [4, 5], endpoint="ab",
    )
    empty = atoms_ab[:n_slab + n_lat]
    gas_atoms = source[[4, 5]]
    gas = SimpleNamespace(atoms=gas_atoms, graph=nx.empty_graph(2))
    atoms_c, _ = _gas_product_neb_endpoint(
        G=graph, gas_reactant=gas, atoms_ab=atoms_ab, atoms_empty=empty,
        n_slab=n_slab, n_lat=n_lat, n_react=2, react_nodes_ab=react_nodes, lift_height=8.0,
    )
    _assert_metadata(atoms_c, source, [0, 1, 6, 5, 4])
