"""Regression coverage for the bond-triple enumeration hot path."""

from __future__ import annotations

import networkx as nx
import numpy as np

import autokmc.sites.bond as bond_module
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionTemplate, find_bond_sites
from autokmc.sites.identity import site_identifier


_LEGACY_PATH_IDENTIFIERS = [
    "bond:d51ce7ee5f10e387c451c334",
    "bond:d4fbba4c35b4ba89962769f5",
    "bond:f089c4188b3dad9d06790420",
    "bond:b62b84af137dcd80df8c4219",
    "bond:a3d0fabc13d11debfbc4e7b5",
    "bond:ec47cf233474997f13e85875",
    "bond:5509e3ce521b967e8469ea23",
    "bond:fceb420878ddc04323a6a65b",
    "bond:881568f7b8a51a5ef0dbb78a",
    "bond:d92583f62380c9ee211d1cbb",
]


def _path_triple_fixture():
    graph = nx.path_graph([1, 2, 3, 4])
    for surface_id in (1, 2, 3, 4):
        graph.nodes[surface_id].update(
            type="surface",
            element="Pt",
            position=np.array([float(surface_id), 0.0, 0.0]),
        )

    def _species(smiles: str, element: str, node_offset: int):
        site = AdsorbateSite(
            reactant=smiles,
            n_atoms=1,
            atom_cliques=[frozenset({1})],
            positions=np.zeros((1, 3)),
            iso_class=0,
        )
        site.members = []
        site.member_node_ids = []
        site._member_cliques = []
        for member_index, surface_id in enumerate((1, 2, 3, 4)):
            node_id = node_offset + member_index
            clique = frozenset({surface_id})
            graph.add_node(
                node_id,
                type="adsorbate",
                element=element,
                reactant=smiles,
                iso_class=0,
                reactant_index=0,
                clique=clique,
                occupied=False,
                siblings=(),
                position=np.array([float(surface_id), 0.0, 1.0]),
            )
            graph.add_edge(node_id, surface_id, anchor_bond=True)
            site.members.append([clique])
            site.member_node_ids.append([node_id])
            site._member_cliques.append((clique,))
        return site

    sites = [
        _species("[C]", "C", 10),
        _species("[O]", "O", 20),
        _species("[N]", "N", 30),
    ]
    template = BondReactionTemplate("[C]", "[O]", "[N]")
    return graph, sites, template


def _enumerate_path_fixture(graph, sites, template):
    return find_bond_sites(
        graph,
        sites,
        [template],
        max_hops=1,
        n_shells_pair=1,
        prune_by_triple=False,
    )


def test_optimised_bond_enumeration_preserves_legacy_members_and_ids(
    monkeypatch,
):
    graph, sites, template = _path_triple_fixture()
    # Placement keys must already be present in the records; entering the old
    # per-triple key helper would indicate a hot-loop regression.
    monkeypatch.setattr(
        bond_module,
        "_placement_key",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("placement keys must be precomputed"),
        ),
    )

    found = _enumerate_path_fixture(graph, sites, template)

    assert len(found) == 10
    assert [len(site.members) for site in found] == [2] * 10
    assert sum(len(site.members) for site in found) == 20
    assert [site_identifier(site) for site in found] == _LEGACY_PATH_IDENTIFIERS


def test_equivalent_triples_materialise_one_full_graph_per_iso_class(
    monkeypatch,
):
    graph, sites, template = _path_triple_fixture()
    original = bond_module._build_triple_ego_graph
    full_graph_builds = 0

    def _counted_build(*args, **kwargs):
        nonlocal full_graph_builds
        full_graph_builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        bond_module, "_build_triple_ego_graph", _counted_build,
    )

    found = _enumerate_path_fixture(graph, sites, template)

    assert sum(len(site.members) for site in found) == 20
    assert full_graph_builds == len(found) == 10


def test_forced_wl_collision_still_uses_exact_graph_matcher(monkeypatch):
    graph, sites, template = _path_triple_fixture()
    original_analysis = bond_module._triple_wl_analysis
    original_matcher = bond_module.isomorphism.GraphMatcher
    matcher_calls = 0

    def _colliding_analysis(blueprint):
        _, colors, classes = original_analysis(blueprint)
        return ("forced-wl-collision",), colors, classes

    def _counted_matcher(*args, **kwargs):
        nonlocal matcher_calls
        matcher_calls += 1
        return original_matcher(*args, **kwargs)

    monkeypatch.setattr(
        bond_module, "_triple_wl_analysis", _colliding_analysis,
    )
    monkeypatch.setattr(
        bond_module,
        "_exact_triple_certificate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bond_module.isomorphism, "GraphMatcher", _counted_matcher,
    )

    found = _enumerate_path_fixture(graph, sites, template)

    # The forced prefilter collision must not merge the ten genuinely
    # non-isomorphic labelled triples.
    assert len(found) == 10
    assert [len(site.members) for site in found] == [2] * 10
    assert matcher_calls > 0


def test_nearby_join_cache_is_target_species_scoped(monkeypatch):
    graph = nx.path_graph([1, 2, 3])
    for node_id in graph:
        graph.nodes[node_id]["type"] = "surface"
    calls = 0
    original = bond_module._surface_bfs_shells

    def _counted_shell(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        bond_module, "_surface_bfs_shells", _counted_shell,
    )
    lookup = bond_module._NearbyPlacementLookup(
        graph,
        {
            "species-a": {1: (0,), 2: (1,)},
            "species-b": {1: (7,), 2: (8,)},
        },
    )

    assert lookup.get("species-a", frozenset({1}), 1) == (0, 1)
    assert lookup.get("species-a", frozenset({1}), 1) == (0, 1)
    assert lookup.get("species-b", frozenset({1}), 1) == (7, 8)
    # Both species reuse one surface shell, but their join results remain
    # isolated by the target-species component of the key.
    assert calls == 1
