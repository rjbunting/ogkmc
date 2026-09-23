"""Molecular connectivity must distinguish processes and cached environments."""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from ogkmc.sites import bond as bond_module
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import BondReactionSite, BondReactionTemplate
from ogkmc.sites.diffusion import DiffusionSite, find_diffusion_sites
from ogkmc.sites.stability.bond import check_bond_site_lateral
from ogkmc.sites.stability.adsorption import (
    _build_stability_atoms, check_adsorbate_site_lateral,
)
from ogkmc.reactions.adsorption import get_applicable_reaction_for_member
from ogkmc.reactions.rates import KB_EV
from ogkmc.sites.stability.diffusion import (
    _diffusion_lateral_node_match,
    check_diffusion_site_lateral,
)
from ogkmc.sites.stability.intermediate_pruning import (
    _expected_graph,
    _observed_graph,
    _same_endpoint,
)


def _ring_hops():
    # The complete substrate graph makes these individual placements
    # equivalent. In a pair, the shared vertices can nevertheless be adjacent
    # or opposite around the molecule, which is a molecular graph invariant.
    graph = nx.complete_graph(6)
    for node in graph:
        graph.nodes[node].update(
            type="surface", element="Pd", position=np.array([node, 0.0, 0.0]),
        )
    orders = [(0, 1, 2, 3), (0, 1, 4, 5), (0, 4, 1, 5)]
    blocks = []
    for member, order in enumerate(orders):
        nodes = list(range(10 + 4 * member, 14 + 4 * member))
        blocks.append(nodes)
        for atom, (node, substrate) in enumerate(zip(nodes, order)):
            graph.add_node(
                node, type="adsorbate", element="C", reactant="C1CCC1",
                iso_class=0, reactant_index=atom, reactant_orbit=0,
                clique=frozenset([substrate]),
                siblings=tuple(sibling for sibling in nodes if sibling != node),
                occupied=False, position=np.array([substrate, 0.0, 1.5]),
            )
            graph.add_edge(node, substrate, anchor_bond=True)
        for atom in range(4):
            graph.add_edge(nodes[atom], nodes[(atom + 1) % 4], intra_adsorbate=True)
    cliques = [[frozenset([node]) for node in order] for order in orders]
    site = AdsorbateSite(
        reactant="C1CCC1", n_atoms=4, atom_cliques=cliques[0],
        positions=np.zeros((4, 3)), iso_class=0, members=cliques,
        member_node_ids=blocks,
    )
    site._member_cliques = [tuple(member) for member in cliques]
    hops = find_diffusion_sites(
        graph, [site], max_hops=0, prune_by_adsorption_pair=False,
    )["C1CCC1"]
    return graph, site, hops


def test_diffusion_pair_graph_preserves_actual_molecular_edges():
    graph, site, hops = _ring_hops()
    nodes = site.member_node_ids[0]
    assert graph.subgraph(nodes).number_of_edges() == 4
    assert hops[0].ego_graph.subgraph(nodes).number_of_edges() == 4


def test_nonisomorphic_hops_do_not_reuse_one_lateral_class():
    graph, site, hops = _ring_hops()
    pair_sites = {
        (a, b): hop for hop in hops for _, a, _, b in hop.members
    }
    assert pair_sites[(0, 1)] is not pair_sites[(0, 2)]
    # Also validate runtime classification independently of static grouping.
    hop = DiffusionSite(
        reactant=site.reactant, iso_class=0,
        members=[(site, 0, site, 1), (site, 0, site, 2)],
        member_node_ids=[(site.member_node_ids[0], site.member_node_ids[b]) for b in (1, 2)],
    )

    def true_endpoint_graph(member):
        _, a, _, b = member
        nodes_a = site.member_node_ids[a]
        nodes_b = site.member_node_ids[b]
        result = graph.subgraph(list(range(6)) + nodes_a + nodes_b).copy()
        for node in nodes_a:
            result.nodes[node]["endpoint_role"] = "a"
        for node in nodes_b:
            result.nodes[node]["endpoint_role"] = "b"
        return result

    true_graphs = [true_endpoint_graph(member) for member in hop.members]
    assert not nx.is_isomorphic(
        *true_graphs, node_match=_diffusion_lateral_node_match,
    )
    classes = [
        check_diffusion_site_lateral(graph, hop, member, n_shells=1)
        for member in range(2)
    ]
    assert classes[0] is not classes[1]


def test_registered_co2_intermediate_matches_its_own_geometry():
    atoms = Atoms(
        "PdOCO",
        positions=[[0, 0, 0], [-1.2, 0, 2], [0, 0, 2], [1.2, 0, 2]],
        cell=[20, 20, 20],
    )
    graph = nx.Graph()
    graph.add_node(0, type="surface", element="Pd", index=0)
    for node, element in zip([1, 2, 3], ["O", "C", "O"]):
        graph.add_node(
            node, type="adsorbate", element=element, clique=frozenset([0]),
            siblings=tuple(sibling for sibling in [1, 2, 3] if sibling != node),
            reactant_index=node - 1,
        )
        graph.add_edge(node, 0)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    observed = _observed_graph(
        graph, atoms, n_slab=1, n_lateral=0, n_reacting=3, nl_mult=1.2,
    )
    expected = _expected_graph(graph, [1, 2, 3])
    assert dict(observed.nodes(data=True)) == dict(expected.nodes(data=True))
    assert set(observed.edges()) == {(0, 1), (1, 2)}
    assert _same_endpoint(observed, expected)


def test_bond_enumeration_blueprint_and_lateral_graph_preserve_ring_bonds():
    graph, site, _ = _ring_hops()
    records = bond_module._prepare_placement_records(graph, {site.reactant: [site]}, 1)[site.reactant]
    for member_b in (1, 2):
        a, b = records[0], records[member_b]
        explicit = bond_module._build_triple_ego_graph(
            graph, list(a.node_ids), list(b.node_ids), [],
            a.clique_union, b.clique_union, frozenset(), 1, 1, 1,
            is_symmetric=True,
        )
        blueprint = bond_module._build_triple_ego_blueprint(graph, a, b, None, is_symmetric=True)
        optimized = bond_module._materialise_triple_blueprint(blueprint)
        assert set(optimized) == set(explicit)
        assert {frozenset(e) for e in optimized.edges} == {frozenset(e) for e in explicit.edges}
        assert optimized.subgraph(a.node_ids).number_of_edges() == 4
        assert optimized.subgraph(b.node_ids).number_of_edges() == 4
    reaction_site = BondReactionSite(
        BondReactionTemplate(site.reactant, site.reactant, "C1CCC1.C1CCC1"), 0,
        members=[(site, 0, site, b, None, -1) for b in (1, 2)],
        member_node_ids=[(site.member_node_ids[0], site.member_node_ids[b], []) for b in (1, 2)],
        gas_product=True,
    )
    classes = [check_bond_site_lateral(graph, reaction_site, member, n_shells=1) for member in (0, 1)]
    assert classes[0] is not classes[1]
    for lateral in classes:
        assert lateral.ego_graph.subgraph(site.member_node_ids[0]).number_of_edges() == 4


def _spectator_environment():
    graph = nx.Graph(cell=np.diag([20.0, 20.0, 20.0]))
    for node, position in enumerate([(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)]):
        graph.add_node(node, type="surface", element="Pd", index=node, position=position)
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    # A top-to-bridge H hop and an H+H bond pair use surface nodes 0 and 1.
    sites = []
    for node, clique in [(10, frozenset({0, 1})), (11, frozenset({0}))]:
        graph.add_node(
            node, type="adsorbate", element="H", reactant="[H]", iso_class=node - 10,
            clique=clique, is_bonded=True, occupied=node == 10, siblings=(),
            reactant_index=0, position=(1.0, 0.0, 1.5),
        )
        graph.add_edges_from((surface, node) for surface in clique)
        site = AdsorbateSite(
            "[H]", 1, [clique], np.array([[1.0, 0.0, 1.5]]), node - 10,
            members=[[clique]], member_node_ids=[[node]],
        )
        site._member_cliques = [(clique,)]
        sites.append(site)

    def molecule(base, a, b):
        nodes = [base, base + 1, base + 2]
        for i, element in enumerate(["O", "C", "O"]):
            surface = a if i == 0 else b if i == 2 else None
            graph.add_node(
                nodes[i], type="adsorbate", element=element, reactant="[O]C[O]", iso_class=2,
                clique=frozenset({surface}) if surface is not None else None,
                is_bonded=surface is not None, occupied=False,
                siblings=tuple(n for n in nodes if n != nodes[i]), reactant_index=i,
                position=(float(a), float(b), 2.0 + i),
            )
            if surface is not None:
                graph.add_edge(surface, nodes[i])
        graph.add_edges_from([(base, base + 1), (base + 1, base + 2)])
        return nodes

    one = molecule(20, 0, 1)
    two = molecule(30, 0, 2) + molecule(40, 1, 3)
    return graph, sites, one, two


@pytest.mark.parametrize("channel", ["adsorption", "diffusion", "bond"])
def test_local_classification_includes_whole_spectators_and_their_attachments(channel):
    graph, (a, b), one, two = _spectator_environment()
    if channel == "adsorption":
        site, classify = a, check_adsorbate_site_lateral
    elif channel == "diffusion":
        site = DiffusionSite(
            "[H]", 0, members=[(a, 0, b, 0)], member_node_ids=[([10], [11])],
        )
        classify = check_diffusion_site_lateral
    else:
        site = BondReactionSite(
            BondReactionTemplate("[H]", "[H]", "[H][H]"), 0,
            members=[(a, 0, b, 0, None, -1)], member_node_ids=[([10], [11], [])],
            gas_product=True,
        )
        classify = check_bond_site_lateral
    for node in one:
        graph.nodes[node]["occupied"] = True
    original = classify(graph, site, 0, n_shells=0)
    for node in one:
        graph.nodes[node]["occupied"] = False
    for node in two:
        graph.nodes[node]["occupied"] = True
    changed = classify(graph, site, 0, n_shells=0)
    assert changed is not original
    assert set(one).issubset(original.ego_graph)
    assert set(two).issubset(changed.ego_graph)
    assert not set(one) & set(changed.ego_graph)
    assert changed.ego_graph.has_edge(2, 32)
    assert changed.ego_graph.has_edge(3, 42)
    # Adding attachment nodes must not recursively collect more molecules.
    graph.add_node(99, type="adsorbate", element="N", occupied=True, siblings=(), clique=frozenset({2}))
    graph.add_edge(2, 99)
    assert classify(graph, site, 0, n_shells=0) is changed
    for node in two:
        graph.nodes[node]["occupied"] = False
    for node in one:
        graph.nodes[node]["occupied"] = True
    assert classify(graph, site, 0, n_shells=0) is original


def test_adsorption_rates_use_energies_of_complete_current_molecules(monkeypatch):
    graph, (site, _), one, two = _spectator_environment()
    evaluated = []

    def controlled_endpoints(graph, site, member, lateral, calculator, **kwargs):
        _, _, n_lat, _ = _build_stability_atoms(
            graph, lateral, frozenset(site.member_node_ids[member]), include_self=True,
        )
        evaluated.append(n_lat)
        lateral.energy_occupied = -1.0 + 0.1 * n_lat
        lateral.energy_unoccupied = 0.0
        lateral.stable = True

    monkeypatch.setattr("ogkmc.reactions.adsorption.check_site_stability", controlled_endpoints)
    for node in one:
        graph.nodes[node]["occupied"] = True
    original = get_applicable_reaction_for_member(
        graph, site, 0, None, {"[H]": 0.0}, temperature=500.0, lateral_shells=0,
    )
    for node in one:
        graph.nodes[node]["occupied"] = False
    for node in two:
        graph.nodes[node]["occupied"] = True
    changed = get_applicable_reaction_for_member(
        graph, site, 0, None, {"[H]": 0.0}, temperature=500.0, lateral_shells=0,
    )
    assert evaluated == [3, 6]
    assert original.delta_e == pytest.approx(0.7)
    assert changed.delta_e == pytest.approx(0.4)
    assert changed.rate / original.rate == pytest.approx(np.exp(0.3 / (KB_EV * 500.0)))
