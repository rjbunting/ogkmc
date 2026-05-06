"""Focused regression tests for site graph bookkeeping."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from autokmc.sites.anchors import _optimise_position
from autokmc.sites.adsorbate import (
    AdsorbateSite,
    find_adsorbate_sites,
    _geometry_connectivity_mismatch,
    optimise_adsorbate_site_positions,
    prune_unstable_adsorbate_sites,
    rebuild_adsorbate_reverse_indexes,
    push_member_positions_to_graph,
)
from autokmc.sites.bond import (
    BondReactionSite,
    BondReactionTemplate,
    find_bond_sites,
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


def test_methane_site_enumeration_keeps_single_h_top_mode():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(
        1,
        type="surface",
        element="Cu",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.32,
    )
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
    )
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C", covalent_radius=0.76)
    for idx in range(1, 5):
        reactant_graph.add_node(idx, element="H", covalent_radius=0.31)
        reactant_graph.add_edge(0, idx)
    reactant = SimpleNamespace(
        smiles="C",
        atoms=atoms,
        graph=reactant_graph,
        anchor_atoms=[1, 2, 3, 4],
        unique_nodes={"C": [[0]], "H": [[1, 2, 3, 4]]},
    )

    sites = find_adsorbate_sites(
        G,
        reactant,
        prune_stable_only=False,
        n_shells_anchor=1,
        n_shells_pair=1,
        include_partial=True,
    )

    assert sites
    assert {
        tuple(i for i, clique in enumerate(site.atom_cliques) if clique is not None)
        for site in sites
    } == {(1,)}


def test_methyl_site_enumeration_keeps_carbon_top_mode():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(
        1,
        type="surface",
        element="Cu",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.32,
    )
    atoms = Atoms(
        "CH3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
    )
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C", covalent_radius=0.76)
    for idx in range(1, 4):
        reactant_graph.add_node(idx, element="H", covalent_radius=0.31)
        reactant_graph.add_edge(0, idx)
    reactant = SimpleNamespace(
        smiles="[CH3]",
        atoms=atoms,
        graph=reactant_graph,
        anchor_atoms=[0],
        unique_nodes={"C": [[0]], "H": [[1, 2, 3]]},
    )

    sites = find_adsorbate_sites(
        G,
        reactant,
        prune_stable_only=False,
        n_shells_anchor=1,
        n_shells_pair=1,
        include_partial=True,
    )

    assert sites
    assert {
        tuple(i for i, clique in enumerate(site.atom_cliques) if clique is not None)
        for site in sites
    } == {(0,)}
    assert all(site.atom_cliques[0] == frozenset({1}) for site in sites)


def test_methyl_carbon_top_geometry_survives_calc_free_refinement():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(
        1,
        type="surface",
        element="Cu",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.32,
    )
    atoms = Atoms(
        "CH3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
    )
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C", covalent_radius=0.76)
    for idx in range(1, 4):
        reactant_graph.add_node(idx, element="H", covalent_radius=0.31)
        reactant_graph.add_edge(0, idx)
    reactant = SimpleNamespace(
        smiles="[CH3]",
        atoms=atoms,
        graph=reactant_graph,
        anchor_atoms=[0],
        unique_nodes={"C": [[0]], "H": [[1, 2, 3]]},
    )

    sites = find_adsorbate_sites(
        G,
        reactant,
        prune_stable_only=False,
        n_shells_anchor=1,
        n_shells_pair=1,
        include_partial=True,
    )
    G.graph["adsorbate_sites"] = {"[CH3]": sites}

    refined = optimise_adsorbate_site_positions(
        G,
        "[CH3]",
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=1,
        max_iter=1,
    )

    assert len(refined) == 1
    assert _geometry_connectivity_mismatch(
        G, refined[0], reactant, refined[0].positions
    ) is None
    assert np.all(refined[0].positions[1:, 2] > refined[0].positions[0, 2])


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


def test_find_bond_sites_allows_gas_product_without_c_surface_site():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 10.0
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(1, type="surface", element="Pt")
    G.add_node(2, type="surface", element="Pt")
    G.add_edge(1, 2)
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    G.add_node(
        10,
        type="adsorbate",
        element="C",
        reactant="[C]",
        iso_class=0,
        reactant_index=0,
        clique=clique_a,
        occupied=False,
        siblings=(),
    )
    G.add_node(
        20,
        type="adsorbate",
        element="O",
        reactant="[O]",
        iso_class=0,
        reactant_index=0,
        clique=clique_b,
        occupied=False,
        siblings=(),
    )
    G.add_edge(10, 1, anchor_bond=True)
    G.add_edge(20, 2, anchor_bond=True)
    site_a = _site("[C]", 0, 10, clique_a)
    site_b = _site("[O]", 0, 20, clique_b)
    gas_c = SimpleNamespace(
        smiles="[C]=O",
        atoms=Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]),
        energy=-1.0,
        partial_pressure_bar=2.0,
    )

    bond_sites = find_bond_sites(
        G,
        [site_a, site_b],
        [BondReactionTemplate("[C]", "[O]", "[C]=O")],
        max_hops=1,
        prune_by_triple=False,
        gas_species={"[C]=O": gas_c},
    )

    assert len(bond_sites) == 1
    brs = bond_sites[0]
    assert brs.gas_product is True
    assert brs.gas_reactant is gas_c
    assert brs.members[0][4:] == (None, -1)
    assert brs.member_node_ids[0] == ([10], [20], [])
    assert brs._member_cliques[0] == ((clique_a,), (clique_b,), tuple())


def test_geometry_refinement_retries_until_required_connectivity(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([False, False, False])
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
    )

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=np.array([[10.0, 0.0, 0.0]]),
        iso_class=0,
        members=[[frozenset({1})]],
        member_node_ids=[],
    )
    G.graph["adsorbate_sites"] = {"[O]": [site]}
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O")
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
    )

    calls = {"n": 0}

    def fake_minimize(_fn, _x0, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(fun=10.0, x=np.zeros(6))
        return SimpleNamespace(
            fun=1.0,
            x=np.array([-10.0, 0.0, 1.8, 0.0, 0.0, 0.0]),
        )

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    optimise_adsorbate_site_positions(
        G,
        "[O]",
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=3,
        max_iter=1,
    )

    assert calls["n"] == 3
    assert site.positions[0, 2] == pytest.approx(1.8)


def test_geometry_refinement_rejects_iso_class_after_failed_connectivity(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([False, False, False])
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
    )
    G.add_node(
        10,
        type="adsorbate",
        element="O",
        reactant="[O]",
        reactant_index=0,
        clique=frozenset({1}),
        occupied=False,
    )
    G.add_edge(10, 1, anchor_bond=True)

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=np.array([[10.0, 0.0, 0.0]]),
        iso_class=0,
        members=[[frozenset({1})]],
        member_node_ids=[[10]],
    )
    G.graph["adsorbate_sites"] = {"[O]": [site]}
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O")
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
    )

    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda _fn, _x0, **_kwargs: SimpleNamespace(fun=10.0, x=np.zeros(6)),
    )

    out = optimise_adsorbate_site_positions(
        G,
        "[O]",
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=2,
        max_iter=1,
    )

    assert out == []
    assert G.graph["adsorbate_sites"]["[O]"] == []
    assert 10 not in G
    assert G.graph["clique_to_members"] == {}


def test_geometry_refinement_repels_unbonded_atoms_from_bonded_surface(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([False, False, False])
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
    )

    site = AdsorbateSite(
        reactant="[CH]",
        n_atoms=2,
        atom_cliques=[frozenset({1}), None],
        positions=np.array([[1.8, 0.0, 0.0], [0.0, 0.0, 0.2]]),
        iso_class=0,
        members=[[frozenset({1}), None]],
        member_node_ids=[],
    )
    G.graph["adsorbate_sites"] = {"[CH]": [site]}
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C")
    reactant_graph.add_node(1, element="H")
    reactant_graph.add_edge(0, 1)
    reactant = SimpleNamespace(
        smiles="[CH]",
        atoms=Atoms("CH", positions=site.positions.copy()),
        graph=reactant_graph,
    )
    seen = {}

    def fake_minimize(fn, _x0, **_kwargs):
        seen["energy"] = float(fn(np.zeros(6)))
        return SimpleNamespace(fun=seen["energy"], x=np.zeros(6))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    optimise_adsorbate_site_positions(
        G,
        "[CH]",
        reactant,
        restraint_weight=0.0,
        repulsion_weight=1.0,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=1,
        max_iter=1,
    )

    assert seen["energy"] > 1.0


def test_adsorbate_pruning_reads_energy_before_detaching_calculator(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 10.0
    G.graph["pbc"] = np.array([True, True, True])
    G.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
    )

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[0.0, 0.0, 1.8]]),
        iso_class=0,
        members=[[frozenset({0})]],
        member_node_ids=[],
    )
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=nx.Graph(),
    )
    reactant.graph.add_node(0, element="O")

    def fake_optimise_structure(atoms, **_kwargs):
        opt = atoms.copy()
        opt.calc = SinglePointCalculator(
            opt,
            energy=-12.3,
            forces=np.zeros((len(opt), 3)),
        )
        return opt

    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        fake_optimise_structure,
    )

    stable = prune_unstable_adsorbate_sites(
        G,
        [site],
        reactant,
        calculator=object(),
        fmax=0.05,
        max_steps=1,
    )

    assert stable == [site]


def test_slab_anchor_optimisation_keeps_top_site_above_surface(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.36,
    )
    captured = {}

    def fake_minimize(_fn, x0, **kwargs):
        captured["x0"] = np.asarray(x0, dtype=float)
        captured["bounds"] = kwargs["bounds"]
        return SimpleNamespace(x=np.asarray(x0, dtype=float))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    pos = _optimise_position(
        G,
        frozenset({1}),
        r_cov_ads=0.76,
        opt_factor=0.85,
        repulsion_weight=0.0,
    )

    assert captured["bounds"][2][0] > 0.0
    assert pos[2] == pytest.approx(captured["bounds"][2][0])


def test_nanoparticle_anchor_optimisation_uses_connectivity_pbc(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, True])
    G.graph["connectivity_pbc"] = np.array([False, False, False])
    G.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.36,
    )
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([2.0, 0.0, 0.0]),
        index=1,
        covalent_radius=1.36,
    )

    captured = {}

    def fake_minimize(fn, x0, **kwargs):
        captured["method"] = kwargs["method"]
        captured["constraints"] = kwargs.get("constraints")
        captured["bounds"] = kwargs.get("bounds")
        captured["x0"] = np.asarray(x0, dtype=float)
        return SimpleNamespace(x=np.asarray(x0, dtype=float), fun=float(fn(x0)))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    pos = _optimise_position(
        G,
        frozenset({1}),
        r_cov_ads=0.76,
        repulsion_weight=0.0,
    )

    assert captured["method"] == "SLSQP"
    assert captured["constraints"]["type"] == "ineq"
    assert captured["bounds"] is None
    assert pos[0] > 2.0
    assert pos[2] == pytest.approx(0.0)


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
