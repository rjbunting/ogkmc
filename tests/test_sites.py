"""Focused regression tests for site graph bookkeeping."""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read

from autokmc.species.smiles import smiles_to_dirname
from autokmc.sites.anchors import (
    _build_ego_graph,
    _enumerate_cliques,
    _get_cell,
    _optimise_position,
    _outward_height_for_clique,
    _reserve_node_ids,
)
from autokmc.sites.adsorbate import (
    AdsorbateSite,
    AdsorbateSiteLateral,
    _adsorbate_pose_is_outward,
    _build_adsorbate_coordination_graph,
    _full_adsorbate_positions,
    _mapped_adsorbate_positions,
    _molecular_mapping_preserves_handedness,
    _propagate_adsorbate_member_positions,
    _relaxed_adsorbate_positions_in_graph_frame,
    _try_merge_or_new,
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
    _triple_node_match,
    rebuild_bond_reverse_indexes,
)
from autokmc.sites.diffusion import (
    find_diffusion_sites,
    _pair_node_match,
    rebuild_diffusion_reverse_indexes,
)
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral
from autokmc.sites.stability.adsorption import check_site_stability
from autokmc.sites.stability.bond import _select_c_to_ab_mapping
from autokmc.sites.stability.bond import _bond_lateral_node_match
from autokmc.sites.stability.diffusion import (
    check_diffusion_site_lateral,
    _diffusion_lateral_node_match,
)
from autokmc.species.reactant import build_reactant


def test_sites_package_exports_public_api():
    import autokmc.sites as sites

    assert sites.AnchorSite.__name__ == "AnchorSite"
    assert sites.AdsorbateSite is AdsorbateSite
    assert callable(sites.find_anchor_sites)
    assert callable(sites.find_adsorbate_sites)
    assert callable(sites.find_diffusion_sites)
    assert callable(sites.find_bond_sites)


def test_node_id_allocator_uses_persistent_monotonic_cursor():
    graph = nx.Graph()
    graph.add_nodes_from(range(100))

    first = list(_reserve_node_ids(graph, 3))
    graph.add_nodes_from(first)
    graph.remove_node(first[-1])
    second = list(_reserve_node_ids(graph, 2))

    assert first == [100, 101, 102]
    assert second == [103, 104]
    assert graph.graph["_autokmc_next_node_id"] == 105


def test_configured_clique_cap_skips_unbounded_maximal_clique_scan(monkeypatch):
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 10.0
    graph.graph["pbc"] = np.array([False, False, False])
    for node in range(4):
        graph.add_node(
            node,
            type="surface",
            element="Cu",
            position=np.array([float(node), 0.0, 0.0]),
        )
    co_bond = nx.complete_graph(4)

    import autokmc.sites.anchors as anchor_module

    monkeypatch.setattr(
        anchor_module,
        "_build_co_bond_graph",
        lambda *_args, **_kwargs: co_bond,
    )
    monkeypatch.setattr(
        nx,
        "find_cliques",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bounded enumeration must not scan maximal cliques")
        ),
    )

    sites = _enumerate_cliques(
        graph,
        "O",
        0.66,
        k_max=2,
    )

    assert set(sites) == {1, 2}
    assert len(sites[1]) == 4
    assert len(sites[2]) == 6


def test_adsorbate_enumeration_applies_anchor_cap_to_dense_surface(monkeypatch):
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 20.0
    graph.graph["pbc"] = np.array([True, True, False])
    for node, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)):
        graph.add_node(
            node,
            type="surface",
            element="Cu",
            index=node,
            covalent_radius=1.32,
            position=np.array(
                [5.0 + np.cos(angle), 5.0 + np.sin(angle), 0.0]
            ),
        )
    graph.add_edges_from(nx.complete_graph(5).edges)
    co_bond = nx.complete_graph(5)

    import autokmc.sites.anchors as anchor_module

    monkeypatch.setattr(
        anchor_module,
        "_build_co_bond_graph",
        lambda *_args, **_kwargs: co_bond,
    )
    monkeypatch.setattr(
        nx,
        "find_cliques",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("configured workflows must remain bounded")
        ),
    )
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O", covalent_radius=0.66)
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O"),
        graph=reactant_graph,
        anchor_atoms=[0],
        unique_nodes={"O": [[0]]},
    )

    sites = find_adsorbate_sites(
        graph,
        reactant,
        anchor_k_max=2,
        n_shells_anchor=1,
        n_shells_pair=1,
        prune_stable_only=False,
    )

    raw = graph.graph["raw_cliques"]["O"]
    assert sites
    assert set(raw) == {1, 2}
    assert len(raw[1]) == 5
    assert len(raw[2]) == 10
    assert graph.graph["_anchor_k_max_by_element"]["O"] == 2


def test_adsorbate_iso_reduction_preserves_per_atom_coordination_graph():
    substrate_ego = nx.complete_graph(3)
    nx.set_node_attributes(substrate_ego, "Pd", "element")
    nx.set_node_attributes(substrate_ego, "surface", "type")

    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O", type="adsorbate")
    reactant_graph.add_node(1, element="O", type="adsorbate")
    reactant_graph.add_edge(0, 1)

    top_top = [frozenset({0}), frozenset({1})]
    bridge_bridge = [frozenset({0, 1}), frozenset({0, 2})]
    hollow_bridge = [frozenset({0, 1, 2}), frozenset({0, 1})]
    equivalent_top_top = [frozenset({1}), frozenset({2})]
    equivalent_hollow_bridge = [
        frozenset({0, 1}),
        frozenset({0, 1, 2}),
    ]

    sites = []
    seen_signatures = set()
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["coordination_role", "element"],
        ["substrate", "X"],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "coordination_kind",
        "substrate",
    )
    for atom_cliques in (
        top_top,
        bridge_bridge,
        hollow_bridge,
        equivalent_top_top,
        equivalent_hollow_bridge,
    ):
        coordination_graph = _build_adsorbate_coordination_graph(
            substrate_ego,
            atom_cliques,
            reactant_graph,
        )
        _try_merge_or_new(
            sites,
            reactant_smiles="O=O",
            atom_cliques=atom_cliques,
            positions=np.zeros((2, 3)),
            ego_graph=coordination_graph,
            node_match=node_match,
            edge_match=edge_match,
            seen_signatures=seen_signatures,
        )

    assert len(sites) == 3
    assert {
        tuple(sorted(len(clique) for clique in site.atom_cliques))
        for site in sites
    } == {(1, 1), (2, 2), (2, 3)}
    assert sorted(len(site.members) for site in sites) == [1, 2, 2]


def test_decorated_mapping_applies_the_molecular_atom_permutation():
    representative = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    mapping = {
        ("adsorbate", 0): ("adsorbate", 2),
        ("adsorbate", 1): ("adsorbate", 0),
        ("adsorbate", 2): ("adsorbate", 1),
    }
    rotation = np.diag([-1.0, -1.0, 1.0])
    translation = np.array([10.0, 20.0, 30.0])

    mapped = _mapped_adsorbate_positions(
        representative,
        mapping,
        rotation,
        translation,
    )
    transformed = representative @ rotation.T + translation

    np.testing.assert_allclose(mapped[2], transformed[0])
    np.testing.assert_allclose(mapped[0], transformed[1])
    np.testing.assert_allclose(mapped[1], transformed[2])


def test_decorated_mapping_preserves_heteronuclear_atom_identity():
    substrate_ego = nx.Graph()
    substrate_ego.add_node(0, element="Pd", type="surface")
    substrate_ego.add_node(1, element="Pd", type="surface")
    substrate_ego.add_edge(0, 1)
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C", type="adsorbate")
    reactant_graph.add_node(1, element="O", type="adsorbate")
    reactant_graph.add_edge(0, 1)

    representative = _build_adsorbate_coordination_graph(
        substrate_ego,
        [frozenset({0}), frozenset({1})],
        reactant_graph,
    )
    member = _build_adsorbate_coordination_graph(
        substrate_ego,
        [frozenset({1}), frozenset({0})],
        reactant_graph,
    )
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["coordination_role", "element"],
        ["substrate", "X"],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "coordination_kind",
        "substrate",
    )
    mappings = list(
        nx.algorithms.isomorphism.GraphMatcher(
            representative,
            member,
            node_match=node_match,
            edge_match=edge_match,
        ).isomorphisms_iter()
    )

    assert mappings
    assert all(
        mapping[("adsorbate", 0)] == ("adsorbate", 0)
        and mapping[("adsorbate", 1)] == ("adsorbate", 1)
        for mapping in mappings
    )


def test_reflected_mapping_accepts_achiral_geometry_but_rejects_chiral_inversion():
    linear = np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    reflected_linear = linear * np.array([-1.0, 1.0, 1.0])
    assert _molecular_mapping_preserves_handedness(linear, reflected_linear)

    tetrahedron = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.4],
        ]
    )
    reflected_tetrahedron = tetrahedron * np.array([-1.0, 1.0, 1.0])
    assert not _molecular_mapping_preserves_handedness(
        tetrahedron,
        reflected_tetrahedron,
    )


def test_handedness_default_tolerates_optimisation_scale_asymmetry():
    """A small distortion of achiral benzene is below the 0.01 A resolution."""
    reactant = build_reactant(
        "C1=CC=CC=C1",
        add_hydrogens=True,
        relax=False,
    )
    positions = np.asarray(reactant.atoms.positions, dtype=float).copy()
    _left, _singular_values, right_transpose = np.linalg.svd(
        positions - positions.mean(axis=0),
        full_matrices=False,
    )
    positions[-1] += 0.005 * right_transpose[-1]
    reflected = positions * np.array([-1.0, 1.0, 1.0])

    assert not _molecular_mapping_preserves_handedness(
        positions,
        reflected,
        rmsd_tol=1.0e-3,
    )
    assert _molecular_mapping_preserves_handedness(positions, reflected)


@pytest.mark.parametrize(
    "smiles",
    [
        "[C@H](F)(Cl)Br",
        "C[C@H](O)C(=O)O",
        "N[C@@H](C)C(=O)O",
        "N[C@@H](CO)C(=O)O",
    ],
    ids=["bromochlorofluoromethane", "lactic-acid", "alanine", "serine"],
)
def test_handedness_default_rejects_resolved_chiral_molecules(smiles):
    reactant = build_reactant(smiles, add_hydrogens=True, relax=False)
    positions = np.asarray(reactant.atoms.positions, dtype=float)
    reflection = np.diag([-1.0, 1.0, 1.0])
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "element",
        "X",
    )
    automorphisms = nx.algorithms.isomorphism.GraphMatcher(
        reactant.graph,
        reactant.graph,
        node_match=node_match,
    ).isomorphisms_iter()

    n_tested = 0
    for atom_mapping in automorphisms:
        decorated_mapping = {
            ("adsorbate", int(source)): ("adsorbate", int(destination))
            for source, destination in atom_mapping.items()
        }
        reflected = _mapped_adsorbate_positions(
            positions,
            decorated_mapping,
            reflection,
            np.zeros(3),
        )
        assert not _molecular_mapping_preserves_handedness(positions, reflected)
        # These ordinary tetrahedral stereocentres remain well outside even a
        # deliberately much looser tolerance than the 0.01 A package default.
        assert not _molecular_mapping_preserves_handedness(
            positions,
            reflected,
            rmsd_tol=0.1,
        )
        n_tested += 1
    assert n_tested > 0


def test_handedness_default_defines_near_planar_resolution():
    def weakly_pyramidal(height):
        return np.array([
            [0.1, 0.2, height],
            [1.0, 0.0, 0.0],
            [-0.3, 1.4, 0.0],
            [-1.0, -0.7, 0.0],
        ])

    below_resolution = weakly_pyramidal(0.01)
    resolved = weakly_pyramidal(0.02)
    reflection = np.array([-1.0, 1.0, 1.0])

    assert _molecular_mapping_preserves_handedness(
        below_resolution,
        below_resolution * reflection,
    )
    assert not _molecular_mapping_preserves_handedness(
        below_resolution,
        below_resolution * reflection,
        rmsd_tol=1.0e-3,
    )
    assert not _molecular_mapping_preserves_handedness(
        resolved,
        resolved * reflection,
    )


def test_decorated_propagation_fails_closed_for_nonisomorphic_members():
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 20.0
    graph.graph["pbc"] = np.array([False, False, False])
    graph.add_node(
        0,
        type="surface",
        element="Pd",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
        covalent_radius=1.39,
    )
    graph.add_node(
        1,
        type="surface",
        element="Au",
        position=np.array([4.0, 0.0, 0.0]),
        index=1,
        covalent_radius=1.36,
    )
    reactant = build_reactant("[O]", add_hydrogens=False, relax=False)

    with pytest.raises(RuntimeError, match="tested=0"):
        _propagate_adsorbate_member_positions(
            graph,
            [frozenset({0})],
            [frozenset({1})],
            np.array([[0.0, 0.0, 1.8]]),
            reactant,
            n_shells=0,
        )


def test_reflected_o3_propagation_accepts_unequal_terminal_bonds():
    """A relaxed planar O3 must survive a reflected, end-swapping site map."""
    from ase.build import fcc111
    from autokmc.core.graph import build_graph
    from autokmc.sites.adsorbate import _geometry_connectivity_mismatch_for_cliques
    from autokmc.species.reactant import Reactant

    slab = fcc111("Pd", size=(5, 5, 4), a=3.89, vacuum=8.0)
    slab.arrays["surface"] = np.where(slab.get_tags() == 1, 1, 0)
    graph = build_graph(slab)
    center, upper, lower, far_upper = 87, 92, 83, 96
    representative_cliques = [
        frozenset({center, upper}), None, frozenset({center, lower}),
    ]
    member_cliques = [
        frozenset({center, upper}), None, frozenset({upper, far_upper}),
    ]
    # From the calculator-free representative of an O2+O UMA relaxation.
    # The two O-O bonds differ by ~0.00036 A at the requested force tolerance.
    relative_positions = np.array([
        [0.68766126, 1.11338971, 1.84499996],
        [1.30413267, 0.00021096, 2.09205529],
        [0.68766138, -1.11339021, 1.84499967],
    ])
    atoms = Atoms("O3", positions=relative_positions)
    atoms.arrays["surface"] = np.full(3, 2)
    reactant = Reactant("[O]O[O]", atoms, build_graph(atoms))
    representative_positions = relative_positions + slab.positions[center]
    assert _geometry_connectivity_mismatch_for_cliques(
        graph, representative_cliques, reactant, representative_positions,
    ) is None

    member_positions = _propagate_adsorbate_member_positions(
        graph, representative_cliques, member_cliques,
        representative_positions, reactant, n_shells=1,
    )

    assert _adsorbate_pose_is_outward(
        graph, member_cliques, member_positions, graph.graph["pbc"],
    )
    assert _geometry_connectivity_mismatch_for_cliques(
        graph, member_cliques, reactant, member_positions,
    ) is None
    # The mapping exchanges the terminal O atoms while preserving the molecule.
    mapped_atoms = Atoms("O3", positions=member_positions)
    np.testing.assert_allclose(
        mapped_atoms.get_all_distances(),
        atoms.get_all_distances()[::-1, ::-1],
        atol=1.0e-10,
    )


def test_adsorbate_runtime_geometry_parameters_reach_each_algorithm_stage(
    monkeypatch,
):
    import autokmc.sites.adsorbate as adsorbate_module

    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 20.0
    graph.graph["pbc"] = np.array([False, False, False])
    graph.add_node(
        0,
        type="surface",
        element="Cu",
        index=0,
        covalent_radius=1.32,
        position=np.zeros(3),
    )

    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C", covalent_radius=0.76)
    reactant_graph.add_node(1, element="H", covalent_radius=0.31)
    reactant_graph.add_edge(0, 1)
    reactant = SimpleNamespace(
        smiles="[CH]",
        atoms=Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.2]]),
        graph=reactant_graph,
        anchor_atoms=[0],
        unique_nodes={"C": [[0]], "H": [[1]]},
    )

    captured = {}
    real_find_anchor_sites = adsorbate_module.find_anchor_sites

    def capture_anchor(*args, **kwargs):
        captured["anchor"] = dict(kwargs)
        return real_find_anchor_sites(*args, **kwargs)

    def capture_refinement(graph_arg, smiles, _reactant, **kwargs):
        captured["refinement"] = dict(kwargs)
        return graph_arg.graph["adsorbate_sites"][smiles]

    def capture_pruning(_graph, sites, _reactant, _calculator, **kwargs):
        captured["pruning"] = dict(kwargs)
        return sites

    monkeypatch.setattr(
        adsorbate_module,
        "find_anchor_sites",
        capture_anchor,
    )
    monkeypatch.setattr(
        adsorbate_module,
        "optimise_adsorbate_site_positions",
        capture_refinement,
    )
    monkeypatch.setattr(
        adsorbate_module,
        "prune_unstable_adsorbate_sites",
        capture_pruning,
    )

    sites = find_adsorbate_sites(
        graph,
        reactant,
        n_shells_anchor=None,
        n_shells_pair=2,
        co_factor=0.91,
        opt_factor=0.86,
        repulsion_weight=0.23,
        repulsion_cutoff=8.5,
        contact_factor=1.07,
        standoff_factor=0.04,
        n_restarts=3,
        nn_distance=1.1,
        max_pair_shells=7,
        hull_tolerance=-0.15,
        kabsch_max_mappings=123,
        nl_mult=0.88,
        calculator=object(),
    )

    assert sites
    assert captured["anchor"] == {
        "co_factor": 0.91,
        "opt_factor": 0.86,
        "repulsion_weight": 0.23,
        "repulsion_cutoff": 8.5,
        "n_shells": 4,
        "k_max": None,
        "hull_tolerance": -0.15,
        "kabsch_max_mappings": 123,
        "verbose": False,
    }
    assert captured["refinement"] == {
        "repulsion_cutoff": 8.5,
        "contact_factor": 1.07,
        "standoff_factor": 0.04,
        "n_restarts": 3,
        "n_shells_pair": 2,
        "nl_mult": 0.88,
        "kabsch_max_mappings": 123,
        "verbose": False,
    }
    assert captured["pruning"]["nl_mult"] == 0.88
    assert captured["pruning"]["kabsch_max_mappings"] == 123


def test_stability_package_exports_public_api():
    import autokmc.sites.stability as stability

    assert issubclass(stability.SiteStabilityError, Exception)
    assert issubclass(stability.NEBNotConvergedError, Exception)
    assert issubclass(stability.BondNEBNotConvergedError, Exception)
    assert callable(stability.check_adsorbate_site_lateral)
    assert callable(stability.check_diffusion_site_lateral)
    assert callable(stability.check_bond_site_lateral)


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_bond_atom_matching_uses_mic_assignment(method):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 10.0
    # C nodes are deliberately out of order.  Node 10 is closest to the first
    # AB atom only under minimum-image distance across the x boundary.
    G.add_node(10, type="adsorbate", element="H", position=np.array([0.2, 0.0, 0.0]), reactant_index=2)
    G.add_node(11, type="adsorbate", element="H", position=np.array([2.1, 0.0, 0.0]), reactant_index=0)
    G.add_node(12, type="adsorbate", element="H", position=np.array([4.1, 0.0, 0.0]), reactant_index=1)
    ab_symbols = ["H", "H", "H"]
    ab_positions = [
        np.array([9.8, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
    ]

    order, diag = _select_c_to_ab_mapping(
        G,
        ab_symbols,
        ab_positions,
        [10, 11, 12],
        atom_matching=method,
        matching_trials=1,
    )

    assert order == [10, 11, 12]
    assert diag["selected_method"] == method
    assert diag["selected"]["max_distance_ang"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "method",
    ["auto", "hungarian", "greedy", "reactant_index"],
)
def test_bond_atom_matching_preserves_reactant_connectivity(method):
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 20.0

    # AB is OH + O.  The first O is bonded to H and the second O is free.
    ab_nodes = [1, 2, 3]
    ab_symbols = ["O", "H", "O"]
    ab_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]
    for node, symbol, position in zip(ab_nodes, ab_symbols, ab_positions):
        graph.add_node(
            node,
            type="adsorbate",
            element=symbol,
            position=position,
            reactant_index=node - 1,
        )
    graph.add_edge(1, 2, intra_adsorbate=True)

    # C is HOO.  Pure nearest-distance matching maps C node 10 onto AB's
    # hydroxyl O even though H is bonded to C node 12.  That swaps the oxygen
    # identities and turns the unchanged O-H bond into an apparent bond
    # breaking/forming event.
    graph.add_node(
        10,
        type="adsorbate",
        element="O",
        position=np.array([0.2, 0.0, 0.0]),
        reactant_index=0,
    )
    graph.add_node(
        11,
        type="adsorbate",
        element="H",
        position=np.array([3.0, 1.0, 0.0]),
        reactant_index=1,
    )
    graph.add_node(
        12,
        type="adsorbate",
        element="O",
        position=np.array([3.0, 0.0, 0.0]),
        reactant_index=2,
    )
    graph.add_edge(10, 12, intra_adsorbate=True)
    graph.add_edge(11, 12, intra_adsorbate=True)

    order, diagnostics = _select_c_to_ab_mapping(
        graph,
        ab_symbols,
        ab_positions,
        [10, 11, 12],
        atom_matching=method,
        matching_trials=8,
        ab_node_order=ab_nodes,
    )

    assert order == [12, 11, 10]
    assert diagnostics["selected"]["reactant_bonds_preserved"] == 1
    assert diagnostics["selected"]["reactant_bonds_total"] == 1
    assert diagnostics["selected"]["connectivity_preserved"] is True


def test_multi_anchor_adsorbate_kabsch_unwraps_periodic_targets():
    graph = nx.Graph()
    graph.graph["cell"] = np.diag([10.0, 10.0, 10.0])
    reactant = SimpleNamespace(
        atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    )

    positions = _full_adsorbate_positions(
        reactant,
        [0, 1],
        np.array([[9.8, 2.0, 3.0], [0.2, 2.0, 3.0]]),
        graph,
        np.array([True, False, False]),
    )

    np.testing.assert_allclose(
        positions,
        [[9.8, 2.0, 3.0], [10.2, 2.0, 3.0]],
        atol=1.0e-12,
    )


def test_geometry_refinement_uses_periodic_clique_centroid(monkeypatch):
    graph = nx.Graph()
    graph.graph["cell"] = np.diag([10.0, 10.0, 20.0])
    graph.graph["pbc"] = np.array([True, True, False])
    graph.graph["connectivity_pbc"] = np.array([True, True, False])
    for node, x_position in [(1, 9.8), (2, 0.2)]:
        graph.add_node(
            node,
            type="surface",
            element="Cu",
            position=np.array([x_position, 3.0, 2.0]),
            index=node - 1,
            covalent_radius=1.32,
        )

    site = AdsorbateSite(
        reactant="[H]",
        n_atoms=1,
        atom_cliques=[frozenset({1, 2})],
        positions=np.array([[10.0, 3.0, 3.8]]),
        iso_class=0,
        members=[[frozenset({1, 2})]],
        member_node_ids=[],
    )
    graph.graph["adsorbate_sites"] = {"[H]": [site]}
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="H", covalent_radius=0.31)
    reactant = SimpleNamespace(
        smiles="[H]",
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
    )

    mismatch_calls = {"count": 0}

    def fake_mismatch(*_args, **_kwargs):
        mismatch_calls["count"] += 1
        if mismatch_calls["count"] == 1:
            return {frozenset((0, 1))}, set()
        return None

    observed = {}

    def fake_minimize(fn, x0, **_kwargs):
        observed["energy_at_initial_pose"] = float(fn(np.asarray(x0, dtype=float)))
        return SimpleNamespace(
            fun=observed["energy_at_initial_pose"],
            x=np.asarray(x0, dtype=float),
        )

    monkeypatch.setattr(
        "autokmc.sites.adsorbate._geometry_connectivity_mismatch",
        fake_mismatch,
    )
    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    refined = optimise_adsorbate_site_positions(
        graph,
        "[H]",
        reactant,
        standoff_factor=1.8 / (1.32 + 0.31),
        repulsion_weight=0.0,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=1,
        max_iter=1,
    )

    assert refined == [site]
    assert observed["energy_at_initial_pose"] < 1.0e-12


def test_relaxed_graph_frame_alignment_unwraps_periodic_surface_ego():
    graph = nx.Graph()
    graph.graph["cell"] = np.diag([10.0, 10.0, 10.0])
    graph.graph["pbc"] = np.array([True, False, False])
    graph.graph["connectivity_pbc"] = np.array([True, False, False])
    graph_positions = np.array(
        [
            [9.8, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [9.8, 1.0, 0.0],
        ]
    )
    for node, position in enumerate(graph_positions, start=1):
        graph.add_node(node, type="surface", position=position)
    graph.add_edges_from([(1, 2), (1, 3)])

    graph_unwrapped = np.array(
        [
            [9.8, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [9.8, 1.0, 0.0],
        ]
    )
    angle = np.deg2rad(8.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    origin = graph_unwrapped[0]
    translation = np.array([0.05, -0.03, 0.02])
    relaxed_unwrapped = (graph_unwrapped - origin) @ rotation.T + origin + translation
    adsorbate_graph_position = np.array([[10.0, 0.3, 1.5]])
    relaxed_adsorbate = (
        (adsorbate_graph_position - origin) @ rotation.T + origin + translation
    )
    atoms_opt = Atoms(
        "Cu3H",
        positions=np.vstack((relaxed_unwrapped, relaxed_adsorbate)),
        cell=graph.graph["cell"],
        pbc=graph.graph["pbc"],
    )
    site = AdsorbateSite(
        reactant="[H]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=adsorbate_graph_position.copy(),
        iso_class=0,
    )

    projected = _relaxed_adsorbate_positions_in_graph_frame(
        graph,
        site,
        atoms_opt,
        n_slab=3,
        n_ads=1,
        node_to_ase={1: 0, 2: 1, 3: 2},
        frame_depth=1,
    )

    np.testing.assert_allclose(projected, adsorbate_graph_position, atol=1.0e-10)


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


def test_diffusion_enumeration_uses_local_surface_shell_without_apsp():
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 10.0
    graph.graph["pbc"] = np.array([True, True, False])
    for surface_id, x_position in ((1, 0.0), (2, 2.5)):
        graph.add_node(
            surface_id,
            type="surface",
            element="Pt",
            position=np.array([x_position, 0.0, 0.0]),
        )
    graph.add_edge(1, 2)
    sites = []
    for iso_class, node_id, surface_id in ((0, 10, 1), (1, 20, 2)):
        clique = frozenset({surface_id})
        graph.add_node(
            node_id,
            type="adsorbate",
            element="O",
            reactant="[O]",
            iso_class=iso_class,
            reactant_index=0,
            clique=clique,
            occupied=False,
            siblings=(),
            position=np.array([2.5 * iso_class, 0.0, 1.8]),
        )
        graph.add_edge(node_id, surface_id, anchor_bond=True)
        sites.append(_site("[O]", iso_class, node_id, clique))

    found = find_diffusion_sites(
        graph,
        sites,
        max_hops=1,
        prune_by_adsorption_pair=False,
    )

    assert sum(len(group) for group in found.values()) == 1
    assert "surface_apsp" not in graph.graph


def test_diffusion_keeps_distinct_placements_with_the_same_clique_union():
    graph = nx.complete_graph([1, 2, 3])
    for surface_id in (1, 2, 3):
        graph.nodes[surface_id].update(
            type="surface",
            element="Pd",
            position=np.array([float(surface_id), 0.0, 0.0]),
        )

    placements = (
        (0, (frozenset({1, 2}), frozenset({1, 3})), (10, 11)),
        (1, (frozenset({1, 2, 3}), frozenset({1, 2})), (20, 21)),
    )
    sites = []
    for iso_class, cliques, node_ids in placements:
        site = AdsorbateSite(
            reactant="O=O",
            n_atoms=2,
            atom_cliques=list(cliques),
            positions=np.zeros((2, 3)),
            iso_class=iso_class,
            members=[list(cliques)],
            member_node_ids=[list(node_ids)],
        )
        site._member_cliques = [tuple(cliques)]
        for atom_index, (node_id, clique) in enumerate(zip(node_ids, cliques)):
            graph.add_node(
                node_id,
                type="adsorbate",
                element="O",
                reactant="O=O",
                iso_class=iso_class,
                reactant_index=atom_index,
                reactant_orbit=0,
                clique=clique,
                occupied=False,
                siblings=(node_ids[1 - atom_index],),
                position=np.array([float(atom_index), 0.0, 1.5]),
            )
            for surface_id in clique:
                graph.add_edge(node_id, surface_id, anchor_bond=True)
        graph.add_edge(*node_ids, intra_adsorbate=True)
        sites.append(site)

    found = find_diffusion_sites(
        graph,
        sites,
        max_hops=0,
        prune_by_adsorption_pair=False,
    )["O=O"]

    assert len(found) == 1
    assert len(found[0].members) == 1
    site_a, _, site_b, _ = found[0].members[0]
    assert {site_a.iso_class, site_b.iso_class} == {0, 1}


def test_reaction_graph_matchers_use_molecular_orbits_not_atom_indices():
    atom_zero = {
        "type": "adsorbate",
        "element": "O",
        "iso_class": 4,
        "reactant": "O=O",
        "reactant_index": 0,
        "reactant_orbit": 0,
        "endpoint_role": "endpoint",
    }
    equivalent_atom_one = {
        **atom_zero,
        "reactant_index": 1,
    }
    inequivalent_atom = {
        **equivalent_atom_one,
        "reactant_orbit": 1,
    }

    matchers = (
        _pair_node_match,
        _triple_node_match,
        _diffusion_lateral_node_match,
        _bond_lateral_node_match,
    )
    for matcher in matchers:
        assert matcher(atom_zero, equivalent_atom_one)
        assert not matcher(atom_zero, inequivalent_atom)


def test_base_adsorption_ego_graph_ignores_live_adsorbate_occupancy():
    graph = nx.Graph()
    graph.add_node(1, type="surface", element="Pt")
    graph.add_node(2, type="surface", element="Pt")
    graph.add_edge(1, 2)
    graph.add_node(
        10,
        type="adsorbate",
        element="H",
        reactant="[H]",
        iso_class=0,
        reactant_index=0,
        clique=frozenset({2}),
        occupied=False,
        siblings=(),
    )
    graph.add_edge(2, 10, anchor_bond=True)

    bare = _build_ego_graph(graph, frozenset({1}), 2)
    graph.nodes[10]["occupied"] = True
    occupied = _build_ego_graph(graph, frozenset({1}), 2)

    assert set(bare) == {1, 2}
    assert nx.utils.graphs_equal(bare, occupied)


def test_diffusion_base_isomorphs_ignore_occupancy_but_lateral_classes_keep_it():
    graph = nx.cycle_graph(range(1, 7))
    for surface_id in graph:
        graph.nodes[surface_id].update(
            type="surface",
            element="Pt",
            position=np.array([float(surface_id), 0.0, 0.0]),
        )

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({1})],
        positions=np.zeros((1, 3)),
        iso_class=0,
    )
    site.members = []
    site.member_node_ids = []
    site._member_cliques = []
    for member_index, surface_id in enumerate(range(1, 7)):
        node_id = 10 + member_index
        clique = frozenset({surface_id})
        graph.add_node(
            node_id,
            type="adsorbate",
            element="O",
            reactant="[O]",
            iso_class=0,
            reactant_index=0,
            clique=clique,
            occupied=False,
            siblings=(),
            position=np.array([float(surface_id), 0.0, 1.8]),
        )
        graph.add_edge(node_id, surface_id, anchor_bond=True)
        site.members.append([clique])
        site.member_node_ids.append([node_id])
        site._member_cliques.append((clique,))

    graph.add_node(
        100,
        type="adsorbate",
        element="H",
        reactant="[H]",
        iso_class=1,
        reactant_index=0,
        clique=frozenset({1}),
        occupied=False,
        siblings=(),
        position=np.array([1.0, 0.0, 1.8]),
    )
    graph.add_edge(100, 1, anchor_bond=True)

    bare = find_diffusion_sites(
        graph,
        [site],
        max_hops=1,
        n_shells_pair=1,
        prune_by_adsorption_pair=True,
    )["[O]"]
    graph.nodes[100]["occupied"] = True
    occupied = find_diffusion_sites(
        graph,
        [site],
        max_hops=1,
        n_shells_pair=1,
        prune_by_adsorption_pair=True,
    )["[O]"]

    assert len(bare) == len(occupied) == 1
    assert [len(ds.members) for ds in bare] == [6]
    assert [len(ds.members) for ds in occupied] == [6]

    diffusion_site = occupied[0]
    def _member_surface_union(member):
        site_a, member_a, site_b, member_b = member
        return (
            frozenset().union(*site_a._member_cliques[member_a])
            | frozenset().union(*site_b._member_cliques[member_b])
        )

    near_index = next(
        index
        for index, member in enumerate(diffusion_site.members)
        if 1 in _member_surface_union(member)
    )
    far_index = next(
        index
        for index, member in enumerate(diffusion_site.members)
        if 1 not in _member_surface_union(member)
    )
    near_lateral = check_diffusion_site_lateral(
        graph, diffusion_site, near_index, n_shells=0,
    )
    far_lateral = check_diffusion_site_lateral(
        graph, diffusion_site, far_index, n_shells=0,
    )

    assert near_lateral is not far_lateral
    assert sorted(
        len(lateral.ego_graph) for lateral in diffusion_site.lateral_classes
    ) == [4, 5]


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
    assert "surface_apsp" not in G.graph
    assert brs.members[0][4:] == (None, -1)
    assert brs.member_node_ids[0] == ([10], [20], [])
    assert brs._member_cliques[0] == ((clique_a,), (clique_b,), tuple())


def test_find_bond_sites_returns_empty_after_all_adsorbate_sites_are_pruned():
    graph = nx.Graph()
    graph.graph["adsorbate_sites"] = {"[O]": []}

    result = find_bond_sites(
        graph,
        [],
        [BondReactionTemplate("[O]", "[O]", "O=O")],
    )

    assert result == []
    assert graph.graph["bond_reaction_sites"] == []
    assert graph.graph["bond_clique_to_members"] == {}
    assert graph.graph["bond_surface_node_to_members"] == {}


def test_find_bond_sites_still_rejects_empty_input_before_site_discovery():
    with pytest.raises(ValueError, match="no AdsorbateSites were supplied"):
        find_bond_sites(
            nx.Graph(),
            [],
            [BondReactionTemplate("[O]", "[O]", "O=O")],
        )


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

    stages = []

    def fake_rigid_optimisation(atoms, *_args, **_kwargs):
        stages.append("rigid")
        rigid = atoms.copy()
        rigid.positions[-1, 2] += 0.01
        return rigid, 0.0, 3

    def fake_optimise_structure(atoms, **_kwargs):
        stages.append("relaxed")
        assert atoms.positions[-1, 2] == pytest.approx(1.81)
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
    monkeypatch.setattr(
        "autokmc.sites.adsorbate._optimise_rigid_adsorbate_with_potential",
        fake_rigid_optimisation,
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
    assert stages == ["rigid", "relaxed"]


def test_adsorbate_pruning_persists_mlip_rejected_structures(
    tmp_path,
    monkeypatch,
):
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 10.0
    graph.graph["pbc"] = np.array([True, True, True])
    graph.add_node(
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
        optimized = atoms.copy()
        optimized.calc = SinglePointCalculator(
            optimized,
            energy=-12.3,
            forces=np.ones((len(optimized), 3)),
        )
        return optimized

    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        fake_optimise_structure,
    )
    monkeypatch.setattr(
        "autokmc.sites.adsorbate._optimise_rigid_adsorbate_with_potential",
        lambda atoms, *_args, **_kwargs: (atoms.copy(), 0.0, 0),
    )

    stable = prune_unstable_adsorbate_sites(
        graph,
        [site],
        reactant,
        calculator=object(),
        fmax=0.05,
        max_steps=1,
        diagnostics_dir=tmp_path / "diagnostics",
    )

    assert stable == []
    folder = (
        tmp_path
        / "diagnostics"
        / "invalid_adsorption"
        / smiles_to_dirname("[O]")
        / "ads_iso0"
    )
    assert (folder / "initial.extxyz").is_file()
    assert (folder / "optimized.extxyz").is_file()
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["invalid_reason"] == "not_converged"
    assert payload["structures"] == {
        "initial": "initial.extxyz",
        "optimized": "optimized.extxyz",
    }
    assert payload["details"]["max_force_ev_per_ang"] > 0.05


def test_adsorbate_pruning_persists_last_geometry_when_optimizer_raises(
    tmp_path,
    monkeypatch,
):
    from autokmc.structure import StructureOptimisationError

    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 10.0
    graph.graph["pbc"] = np.array([True, True, True])
    graph.add_node(
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

    def fail_optimise_structure(atoms, **_kwargs):
        failed = atoms.copy()
        failed.positions[-1, 2] = 3.25
        raise StructureOptimisationError(
            "forced failure",
            failed,
            converged=None,
            steps=7,
        )

    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        fail_optimise_structure,
    )
    monkeypatch.setattr(
        "autokmc.sites.adsorbate._optimise_rigid_adsorbate_with_potential",
        lambda atoms, *_args, **_kwargs: (atoms.copy(), 0.0, 0),
    )

    with pytest.raises(StructureOptimisationError, match="forced failure"):
        prune_unstable_adsorbate_sites(
            graph,
            [site],
            reactant,
            calculator=object(),
            diagnostics_dir=tmp_path / "diagnostics",
        )

    folder = (
        tmp_path
        / "diagnostics"
        / "invalid_adsorption"
        / smiles_to_dirname("[O]")
        / "ads_iso0"
    )
    optimized = ase_read(folder / "optimized.extxyz")
    assert optimized.positions[-1, 2] == pytest.approx(3.25)
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["invalid_reason"] == "relaxation_failed"
    assert payload["details"]["optimizer_steps"] == 7


def test_adsorbate_pruning_reports_rigid_stage_nonconvergence(
    tmp_path,
    monkeypatch,
):
    from autokmc.structure import StructureOptimisationError

    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 10.0
    graph.graph["pbc"] = np.array([True, True, True])
    graph.add_node(
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

    def fail_rigid(atoms, *_args, **_kwargs):
        failed = atoms.copy()
        failed.positions[-1, 2] = 2.25
        raise StructureOptimisationError(
            "forced rigid nonconvergence",
            failed,
            converged=False,
            steps=9,
        )

    monkeypatch.setattr(
        "autokmc.sites.adsorbate._optimise_rigid_adsorbate_with_potential",
        fail_rigid,
    )
    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        lambda *_args, **_kwargs: pytest.fail(
            "full relaxation must not run after rigid nonconvergence"
        ),
    )

    stable = prune_unstable_adsorbate_sites(
        graph,
        [site],
        reactant,
        calculator=object(),
        diagnostics_dir=tmp_path / "diagnostics",
    )

    assert stable == []
    folder = (
        tmp_path
        / "diagnostics"
        / "invalid_adsorption"
        / smiles_to_dirname("[O]")
        / "ads_iso0"
    )
    optimized = ase_read(folder / "optimized.extxyz")
    assert optimized.positions[-1, 2] == pytest.approx(2.25)
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["invalid_reason"] == "rigid_not_converged"
    assert payload["details"]["optimization_stage"] == "rigid"
    assert payload["details"]["optimizer_steps"] == 9


def test_adsorbate_pruning_projects_relaxed_adsorbate_back_to_graph_frame(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, False])
    for nid, x in ((0, 0.0), (1, 3.0)):
        G.add_node(
            nid,
            type="surface",
            element="Pt",
            position=np.array([x, 0.0, 0.0]),
            index=nid,
        )
    G.add_edge(0, 1)
    for nid, x, surf in ((10, 0.0, 0), (20, 3.0, 1)):
        G.add_node(
            nid,
            type="adsorbate",
            element="O",
            position=np.array([x, 0.0, 1.8]),
            reactant="[O]",
            reactant_index=0,
            clique=frozenset({surf}),
            is_bonded=True,
            occupied=False,
        )
        G.add_edge(surf, nid, anchor_bond=True)

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[0.0, 0.0, 1.8]]),
        iso_class=0,
        members=[[frozenset({0})], [frozenset({1})]],
        member_node_ids=[[10], [20]],
        n_shells_settled=1,
    )
    G.graph["adsorbate_sites"] = {"[O]": [site]}
    rebuild_adsorbate_reverse_indexes(G)

    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=nx.Graph(),
    )
    reactant.graph.add_node(0, element="O")

    def fake_optimise_structure(atoms, **_kwargs):
        opt = atoms.copy()
        pos = opt.get_positions()
        pos[:, 2] -= 0.5
        opt.set_positions(pos)
        opt.calc = SinglePointCalculator(
            opt,
            energy=-1.0,
            forces=np.zeros((len(opt), 3)),
        )
        return opt

    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        fake_optimise_structure,
    )
    monkeypatch.setattr(
        "autokmc.sites.adsorbate._optimise_rigid_adsorbate_with_potential",
        lambda atoms, *_args, **_kwargs: (atoms.copy(), 0.0, 0),
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
    assert G.nodes[10]["position"][2] == pytest.approx(1.8)
    assert G.nodes[20]["position"][2] == pytest.approx(1.8)


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


def test_nanoparticle_propagation_rejects_inward_positions():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, True])
    G.graph["connectivity_pbc"] = np.array([False, False, False])
    G.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        covalent_radius=1.36,
    )
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([2.0, 0.0, 0.0]),
        covalent_radius=1.36,
    )

    cell, cell_inv, pbc, use_mic = _get_cell(G)

    assert not use_mic
    assert _outward_height_for_clique(
        G, frozenset({1}), np.array([3.0, 0.0, 0.0]),
        cell, cell_inv, pbc, use_mic,
    ) > 0.0
    assert _outward_height_for_clique(
        G, frozenset({1}), np.array([1.5, 0.0, 0.0]),
        cell, cell_inv, pbc, use_mic,
    ) < 0.0
    assert _adsorbate_pose_is_outward(
        G, [frozenset({1})], np.array([[3.0, 0.0, 0.0]]), pbc,
    )
    assert not _adsorbate_pose_is_outward(
        G, [frozenset({1})], np.array([[1.5, 0.0, 0.0]]), pbc,
    )


def test_slab_propagation_rejects_positions_below_surface():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, True])
    G.graph["connectivity_pbc"] = np.array([True, True, False])
    G.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 5.0]),
        covalent_radius=1.36,
    )
    G.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([2.0, 0.0, 5.0]),
        covalent_radius=1.36,
    )

    cell, cell_inv, pbc, use_mic = _get_cell(G)

    assert use_mic
    assert _outward_height_for_clique(
        G, frozenset({0, 1}), np.array([1.0, 0.0, 6.0]),
        cell, cell_inv, pbc, use_mic,
    ) > 0.0
    assert _outward_height_for_clique(
        G, frozenset({0, 1}), np.array([1.0, 0.0, 4.5]),
        cell, cell_inv, pbc, use_mic,
    ) < 0.0
    assert _adsorbate_pose_is_outward(
        G, [frozenset({0, 1})], np.array([[1.0, 0.0, 6.0]]), pbc,
    )
    assert not _adsorbate_pose_is_outward(
        G, [frozenset({0, 1})], np.array([[1.0, 0.0, 4.5]]), pbc,
    )


def test_adsorption_lateral_structures_use_full_calculator_pbc(monkeypatch):
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 20.0
    G.graph["pbc"] = np.array([True, True, True])
    G.graph["connectivity_pbc"] = np.array([True, True, False])
    G.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 5.0]),
        index=0,
        covalent_radius=1.36,
    )
    G.add_node(
        10,
        type="adsorbate",
        element="O",
        position=np.array([0.0, 0.0, 6.0]),
        clique=frozenset({0}),
        reactant="[O]",
        reactant_index=0,
        occupied=False,
        siblings=(),
    )
    G.add_edge(0, 10, anchor_bond=True)

    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[0.0, 0.0, 6.0]]),
        iso_class=0,
        members=[[frozenset({0})]],
        member_node_ids=[[10]],
    )
    lateral = AdsorbateSiteLateral(
        lateral_class=0,
        ego_graph=G.subgraph([0, 10]).copy(),
    )

    def fake_optimise_structure(atoms, **_kwargs):
        opt = atoms.copy()
        opt.set_pbc([True, True, True])
        opt.calc = SinglePointCalculator(
            opt,
            energy=-1.0,
            forces=np.zeros((len(opt), 3)),
        )
        return opt

    monkeypatch.setattr(
        "autokmc.structure.optimise_structure",
        fake_optimise_structure,
    )

    check_site_stability(
        G,
        site,
        0,
        lateral,
        calculator=object(),
        fmax=0.05,
        max_steps=1,
    )

    assert tuple(lateral.atoms_occupied.pbc) == (True, True, True)
    assert tuple(lateral.atoms_unoccupied.pbc) == (True, True, True)


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
