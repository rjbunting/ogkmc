"""Focused regression tests for site graph bookkeeping."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.sites.adsorbate import (
    AdsorbateSite,
    find_adsorbate_sites,
    rebuild_adsorbate_reverse_indexes,
    push_member_positions_to_graph,
)
from autokmc.sites.bond import (
    BondReactionSite,
    BondReactionTemplate,
    _prune_one_per_adsorption_triple,
    rebuild_bond_reverse_indexes,
)
from autokmc.sites.diffusion import rebuild_diffusion_reverse_indexes
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral


def test_sites_package_exports_public_api():
    import autokmc.sites as sites

    assert sites.AnchorSite.__name__ == "AnchorSite"
    assert sites.AdsorbateSite is AdsorbateSite
    assert callable(sites.find_anchor_sites)
    assert callable(sites.find_adsorbate_sites)
    assert callable(sites.find_diffusion_sites)
    assert callable(sites.find_bond_sites)


def test_stability_package_exports_public_api():
    import autokmc.sites.stability as stability

    assert issubclass(stability.SiteStabilityError, Exception)
    assert issubclass(stability.NEBNotConvergedError, Exception)
    assert issubclass(stability.BondNEBNotConvergedError, Exception)
    assert callable(stability.check_adsorbate_site_lateral)
    assert callable(stability.check_diffusion_site_lateral)
    assert callable(stability.check_bond_site_lateral)


def _site(smiles: str, iso: int, node_id: int, clique: frozenset[int]):
    return SimpleNamespace(
        reactant=smiles,
        iso_class=iso,
        member_node_ids=[[node_id]],
        _member_cliques=[(clique,)],
    )


def test_rebuild_adsorbate_reverse_indexes_drops_stale_members():
    G = nx.Graph()
    active_clique = frozenset({1, 2})
    stale_clique = frozenset({9})
    active = _site("[O]", 0, 10, active_clique)
    stale = _site("[O]", 1, 90, stale_clique)
    G.add_node(10, type="adsorbate", clique=active_clique, occupied=True)
    G.graph["adsorbate_sites"] = {"[O]": [active]}
    G.graph["clique_to_members"] = {stale_clique: [(stale, 0)]}
    G.graph["surface_node_to_members"] = {9: [(stale, 0)]}

    rebuild_adsorbate_reverse_indexes(G)

    assert G.graph["clique_to_members"] == {active_clique: [(active, 0)]}
    assert G.graph["surface_node_to_members"][1] == [(active, 0)]
    assert stale_clique not in G.graph["clique_to_members"]
    assert G.graph["occupied_by_clique"][active_clique] == {10}
    assert G.graph["n_occupied"] == 1


def test_find_adsorbate_sites_no_anchor_path_clears_stale_state():
    G = nx.Graph()
    stale_clique = frozenset({9})
    stale = _site("[He]", 0, 90, stale_clique)
    G.add_node(90, type="adsorbate", reactant="[He]", clique=stale_clique)
    G.graph["adsorbate_sites"] = {"[He]": [stale]}
    G.graph["clique_to_members"] = {stale_clique: [(stale, 0)]}
    reactant = SimpleNamespace(
        atoms=[object()],
        anchor_atoms=[],
        smiles="[He]",
    )

    assert find_adsorbate_sites(G, reactant, require_anchors=False) == []

    assert 90 not in G
    assert G.graph["adsorbate_sites"]["[He]"] == []
    assert G.graph["clique_to_members"] == {}


def test_rebuild_diffusion_reverse_indexes_replaces_old_entries():
    G = nx.Graph()
    old_clique = frozenset({9})
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    site_a = _site("[O]", 0, 10, clique_a)
    site_b = _site("[O]", 1, 20, clique_b)
    ds = SimpleNamespace(
        members=[(site_a, 0, site_b, 0)],
        member_node_ids=[([10], [20])],
    )
    G.graph["diffusion_clique_to_members"] = {old_clique: [(object(), 0)]}

    rebuild_diffusion_reverse_indexes(G, {"[O]": [ds]})

    assert G.graph["diffusion_clique_to_members"][clique_a] == [(ds, 0)]
    assert G.graph["diffusion_clique_to_members"][clique_b] == [(ds, 0)]
    assert old_clique not in G.graph["diffusion_clique_to_members"]
    assert G.graph["diffusion_surface_node_to_members"][1] == [(ds, 0)]


def test_rebuild_bond_reverse_indexes_replaces_old_entries():
    G = nx.Graph()
    old_clique = frozenset({9})
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    clique_c = frozenset({3})
    brs = SimpleNamespace(_member_cliques=[((clique_a,), (clique_b,), (clique_c,))])
    G.graph["bond_clique_to_members"] = {old_clique: [(object(), 0)]}

    rebuild_bond_reverse_indexes(G, [brs])

    assert G.graph["bond_clique_to_members"][clique_a] == [(brs, 0)]
    assert G.graph["bond_clique_to_members"][clique_b] == [(brs, 0)]
    assert G.graph["bond_clique_to_members"][clique_c] == [(brs, 0)]
    assert old_clique not in G.graph["bond_clique_to_members"]
    assert G.graph["bond_surface_node_to_members"][3] == [(brs, 0)]


def test_bond_prune_key_keeps_unrelated_species_with_same_iso_numbers():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    clique_c = frozenset({3})
    first_a = _site("[C]", 0, 10, clique_a)
    first_b = _site("[O]", 0, 20, clique_b)
    first_c = _site("[C]=O", 0, 30, clique_c)
    second_a = _site("[N]", 0, 40, clique_a)
    second_b = _site("[H]", 0, 50, clique_b)
    second_c = _site("[NH]", 0, 60, clique_c)
    ego_small = nx.path_graph(2)
    ego_large = nx.path_graph(4)
    brs_first = BondReactionSite(
        template=BondReactionTemplate("[C]", "[O]", "[C]=O"),
        iso_class=0,
        members=[(first_a, 0, first_b, 0, first_c, 0)],
        member_node_ids=[([10], [20], [30])],
        ego_graph=ego_large,
    )
    brs_second = BondReactionSite(
        template=BondReactionTemplate("[N]", "[H]", "[NH]"),
        iso_class=1,
        members=[(second_a, 0, second_b, 0, second_c, 0)],
        member_node_ids=[([40], [50], [60])],
        ego_graph=ego_small,
    )

    pruned = _prune_one_per_adsorption_triple([brs_first, brs_second])

    assert pruned == [brs_first, brs_second]


def test_adsorption_lateral_reassignment_removes_old_membership():
    G = nx.Graph()
    G.add_node(1, type="surface", element="Pt")
    G.add_node(2, type="surface", element="Pt")
    G.add_edge(1, 2)
    G.add_node(
        10,
        type="adsorbate",
        element="O",
        iso_class=0,
        reactant="[O]",
        clique=frozenset({1}),
        is_bonded=True,
        occupied=False,
        siblings=(),
    )
    G.add_edge(10, 1, anchor_bond=True)
    G.add_node(
        20,
        type="adsorbate",
        element="H",
        iso_class=1,
        reactant="[H]",
        clique=frozenset({2}),
        is_bonded=True,
        occupied=False,
        siblings=(),
    )
    G.add_edge(20, 2, anchor_bond=True)
    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=np.zeros((1, 3)),
        iso_class=0,
        members=[[frozenset({1})]],
        member_node_ids=[[10]],
    )

    bare = check_adsorbate_site_lateral(G, site, 0, n_shells=1)
    G.nodes[20]["occupied"] = True
    occupied = check_adsorbate_site_lateral(G, site, 0, n_shells=1)

    assert bare is not occupied
    assert 0 not in bare.members
    assert occupied.members == [0]


def test_push_member_positions_refreshes_periodic_edge_distance():
    G = nx.Graph()
    G.graph["cell"] = np.diag([10.0, 10.0, 10.0])
    G.graph["pbc"] = np.array([True, False, False])
    G.add_node(
        1,
        type="surface",
        position=np.array([9.8, 0.0, 0.0]),
    )
    G.add_node(
        10,
        type="adsorbate",
        position=np.array([0.0, 0.0, 0.0]),
        optimised=False,
    )
    G.add_edge(1, 10, offset=(1, 0, 0), distance=9.8, anchor_bond=True)
    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=np.zeros((1, 3)),
        iso_class=0,
        members=[[frozenset({1})]],
        member_node_ids=[[10]],
    )

    push_member_positions_to_graph(G, site, 0, np.array([[0.2, 0.0, 0.0]]))

    assert G.edges[1, 10]["distance"] == pytest.approx(0.4)
