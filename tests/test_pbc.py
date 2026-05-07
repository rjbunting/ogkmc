import networkx as nx
import numpy as np
from ase.build import fcc111

from autokmc.core.pbc import minimum_image_vectors
from autokmc.core.graph import build_graph
from autokmc.io.atoms import atoms_from_graph
from autokmc.sites.adsorbate import _mic_distance
from autokmc.sites.anchors import _build_co_bond_graph
from autokmc.sites.adsorbate import find_adsorbate_sites
from autokmc.sites.anchors import find_anchor_sites
from autokmc.species.reactant import build_reactant
from autokmc.structure import find_surface_atoms, find_surface_atoms_raycasting


def _skew_cell():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )


def test_minimum_image_vectors_handles_skew_cells():
    vector = np.array([0.5, 0.3, 0.0])

    mic = minimum_image_vectors(vector, _skew_cell(), [True, True, False])

    assert np.allclose(mic, [-0.3, -0.3, 0.0])
    assert np.linalg.norm(mic) < np.linalg.norm(vector)


def test_adsorbate_mic_distance_uses_true_triclinic_mic():
    cell = _skew_cell()

    d = _mic_distance(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.3, 0.0]),
        cell,
        np.linalg.inv(cell),
        np.array([True, True, False]),
        True,
    )

    assert np.isclose(d, np.sqrt(0.18))


def test_anchor_co_bond_graph_keeps_skew_boundary_pairs():
    G = nx.Graph()
    G.graph["cell"] = _skew_cell()
    G.graph["pbc"] = np.array([True, True, False])
    G.add_node(
        0,
        type="surface",
        element="X",
        position=np.array([0.0, 0.0, 0.0]),
        covalent_radius=0.0,
    )
    G.add_node(
        1,
        type="surface",
        element="X",
        position=np.array([0.5, 0.3, 0.0]),
        covalent_radius=0.0,
    )

    cbg = _build_co_bond_graph(G, r_cov_ads=0.25, co_factor=1.0)

    assert cbg.has_edge(0, 1)


def test_materialised_site_positions_are_wrapped_for_skew_slab():
    atoms = fcc111("Cu", size=(3, 3, 3), vacuum=8.0, orthogonal=False)
    atoms.wrap()
    find_surface_atoms(atoms, tag_atoms=True)
    G = build_graph(atoms)

    anchors = find_anchor_sites(G, "O", k_max=3)
    reactant = build_reactant("[O]", add_hydrogens=False)
    sites = find_adsorbate_sites(G, reactant, prune_stable_only=False)

    cell_inv = np.linalg.inv(np.asarray(G.graph["cell"], dtype=float))
    pbc_axes = np.where(np.asarray(G.graph["pbc"], dtype=bool))[0]

    positions = []
    for anchor in anchors:
        positions.extend(G.nodes[n]["position"] for n in anchor.node_ids)
    for site in sites:
        positions.extend(np.asarray(site.positions, dtype=float))
        for node_ids in site.member_node_ids:
            positions.extend(G.nodes[n]["position"] for n in node_ids)

    frac = np.asarray(positions, dtype=float) @ cell_inv
    periodic_frac = frac[:, pbc_axes]

    assert np.all(periodic_frac >= -1e-10)
    assert np.all(periodic_frac < 1.0 + 1e-10)


def test_structures_with_real_cells_are_full_pbc():
    atoms = fcc111("Cu", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    atoms.set_pbc([True, True, False])
    find_surface_atoms(atoms, tag_atoms=True)
    G = build_graph(atoms)
    snapshot = atoms_from_graph(G)

    assert tuple(atoms.pbc) == (True, True, True)
    assert tuple(G.graph["pbc"]) == (True, True, True)
    assert tuple(snapshot.pbc) == (True, True, True)
    assert tuple(G.graph["connectivity_pbc"]) == (True, True, False)


def test_adsorbate_only_reactants_stay_nonperiodic_with_vacuum_cell():
    reactant = build_reactant("[O]", add_hydrogens=False)

    assert np.linalg.matrix_rank(np.asarray(reactant.atoms.get_cell())) == 3
    assert tuple(reactant.atoms.pbc) == (False, False, False)
    assert tuple(reactant.graph.graph["pbc"]) == (False, False, False)
    assert tuple(reactant.graph.graph["connectivity_pbc"]) == (False, False, False)


def test_raycasting_uses_connectivity_axes_for_tilted_z_axis():
    from ase import Atoms

    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [4.0, 0.0, 10.0]],
        pbc=[True, True, False],
    )

    top_mask, top_indices = find_surface_atoms_raycasting(
        atoms,
        which="top",
        n_disc_sample=5,
        coverage_threshold=0.5,
    )
    bottom_mask, bottom_indices = find_surface_atoms_raycasting(
        atoms,
        which="bottom",
        n_disc_sample=5,
        coverage_threshold=0.5,
    )

    assert tuple(atoms.pbc) == (True, True, True)
    assert top_mask.tolist() == [False, True]
    assert top_indices.tolist() == [1]
    assert bottom_mask.tolist() == [True, False]
    assert bottom_indices.tolist() == [0]
