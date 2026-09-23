"""Whole anchor subsets, rather than individual atom orbits, define symmetry."""

from itertools import combinations

import networkx as nx
import numpy as np

from ogkmc.sites.adsorbate import (
    _inequivalent_anchor_subsets, _orbit_id_of, find_adsorbate_sites,
)
from ogkmc.species.reactant import build_reactant


def test_benzene_adjacent_meta_and_opposite_pairs_remain_distinct():
    reactant = build_reactant("c1ccccc1", relax=False)
    pairs = _inequivalent_anchor_subsets(
        reactant.graph, list(combinations(range(6), 2)), _orbit_id_of(reactant),
    )
    assert pairs == [(0, 1), (0, 2), (0, 3)]
    assert len(_inequivalent_anchor_subsets(
        reactant.graph, [(i,) for i in range(6)], _orbit_id_of(reactant),
    )) == 1


def test_real_enumeration_retains_opposite_carbon_binding_candidate():
    reactant = build_reactant("c1ccccc1", relax=False)
    graph = nx.Graph(cell=np.eye(3) * 20, pbc=[True, True, False])
    for index, x in enumerate([0, 2.8]):
        graph.add_node(index, type="surface", element="Pd",
                       position=np.array([x, 0., 0.]), index=index, covalent_radius=1.39)
    graph.add_edge(0, 1, distance=2.8, offset=(0, 0, 0))
    sites = find_adsorbate_sites(
        graph, reactant, anchor_k_max=1, prune_stable_only=False,
        bond_tolerance=.15, n_shells_anchor=1, n_shells_pair=1, repulsion_weight=0.0,
    )
    subsets = {
        tuple(i for i, clique in enumerate(site.atom_cliques) if clique is not None)
        for site in sites
    }
    assert (0, 3) in subsets
    assert (0,) in subsets


def test_metadata_breaks_apparent_element_only_subset_symmetry():
    graph = nx.cycle_graph(6)
    nx.set_node_attributes(graph, "C", "element")
    graph.nodes[0]["atom_arrays"] = {"masses": 13.0}
    # Even a coarse orbit bucket cannot authorize an incorrect subset merge.
    orbits = {node: ("C", 0) for node in graph}
    singletons = _inequivalent_anchor_subsets(graph, [(n,) for n in graph], orbits)
    assert singletons == [(0,), (1,), (2,), (3,)]
