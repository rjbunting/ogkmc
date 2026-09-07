"""Regression coverage for atom inventory, exposed faces, and handedness."""

from collections import Counter

from ase import Atoms
from ase.build import fcc111
from ase.calculators.calculator import Calculator, all_changes
import networkx as nx
import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation

from autokmc.core.graph import build_graph
from autokmc.sites.adsorbate import (
    find_adsorbate_sites,
    _geometry_connectivity_mismatch,
    _full_adsorbate_positions,
    _molecular_mapping_preserves_handedness,
    optimise_adsorbate_site_positions,
    prune_unstable_adsorbate_sites,
)
from autokmc.species.bond_chemistry import combine_fragments, get_all_fragments
from autokmc.species.reactant import build_reactant
from autokmc.structure.builders import _apply_composition
from autokmc.structure.surface import find_surface_atoms


def test_alloy_substitution_preserves_complete_element_symbols():
    alloy = _apply_composition(Atoms("W4"), {"W": 0.5, "Cu": 0.5}, verbose=False)
    assert Counter(alloy.get_chemical_symbols()) == {"W": 2, "Cu": 2}
    assert "C" not in alloy.get_chemical_symbols()


def test_explicit_hydrogens_do_not_enable_implicit_hydrogen_addition():
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    for smiles in ("[H]C", "[H]O", "[H]N"):
        reactant = build_reactant(smiles, add_hydrogens=False)
        recorded = Chem.MolFromSmiles(reactant.atom_inventory_smiles, parser)
        actual_inventory = Counter(reactant.atoms.get_chemical_symbols())
        recorded_inventory = Counter(atom.GetSymbol() for atom in recorded.GetAtoms())
        assert actual_inventory == recorded_inventory
        assert sum(actual_inventory.values()) == 2


def test_object_fragmentation_and_coupling_preserve_isotopes():
    deuterium = build_reactant("[2H]", add_hydrogens=False)
    hydrogen = build_reactant("[H]", add_hydrogens=False)
    object_product = combine_fragments(deuterium, hydrogen)[0].smiles
    string_product = combine_fragments("[2H]", "[H]")[0].smiles
    assert object_product == string_product
    assert "2H" in string_product
    hd = build_reactant("[2H][H]", add_hydrogens=False)
    fragments = get_all_fragments(hd.atoms, strip_dummies=True)[0]
    assert {fragments.smiles_a, fragments.smiles_b} == {"[H]", "[2H]"}


def test_exposed_slab_faces_place_anchors_outward():
    oxygen = build_reactant("[O]", add_hydrogens=False)
    for side in ("top", "bottom", "both"):
        slab = fcc111("Cu", size=(3, 3, 4), a=3.6, vacuum=8)
        find_surface_atoms(slab, which=side, tag_atoms=True)
        graph = build_graph(slab)
        sites = find_adsorbate_sites(graph, oxygen, anchor_k_max=1, prune_stable_only=False)
        assert sites
        observed_faces = set()
        for site in sites:
            surface_atom = next(iter(site.atom_cliques[0]))
            surface_z = graph.nodes[surface_atom]["position"][2]
            anchor_z = site.positions[0, 2]
            mismatch = _geometry_connectivity_mismatch(graph, site, oxygen, site.positions)
            if abs(surface_z - slab.positions[:, 2].min()) < 1e-8:
                observed_faces.add("bottom")
                assert anchor_z < slab.positions[:, 2].min()
                assert mismatch is None
            else:
                observed_faces.add("top")
                assert anchor_z > slab.positions[:, 2].max()
                assert mismatch is None
        assert observed_faces == ({"top", "bottom"} if side == "both" else {side})


def test_antiparallel_anchor_rotation_preserves_chirality_through_stability():
    reactant = build_reactant("F[C@](Cl)(Br)I", add_hydrogens=False)
    positions = reactant.atoms.positions.copy()
    center_direction = (positions[1:] - positions[0]).mean(axis=0)
    rotation, _ = Rotation.align_vectors(
        [[0, 0, -1]],
        [center_direction / np.linalg.norm(center_direction)],
    )
    positions = rotation.apply(positions)
    reactant.atoms.set_positions(positions)
    assert 0 in reactant.anchor_atoms
    graph = nx.Graph()
    graph.graph["cell"] = np.diag([20, 20, 30])
    placed = _full_adsorbate_positions(
        reactant,
        [0],
        np.array([[10, 10, 20]]),
        graph,
        np.array([True, True, False]),
    )
    before = np.linalg.det(positions[[0, 2, 3]] - positions[1])
    after = np.linalg.det(placed[[0, 2, 3]] - placed[1])
    assert before * after > 0
    assert _molecular_mapping_preserves_handedness(positions, placed)

    # Isolate the eligible F-only subset to keep this full-path regression small.
    # Its initial placement is valid according to the actual topology validator.
    reactant.anchor_atoms = [0]
    slab = fcc111("Cu", size=(3, 3, 4), a=3.6, vacuum=8)
    find_surface_atoms(slab, tag_atoms=True)
    graph = build_graph(slab)
    sites = find_adsorbate_sites(graph, reactant, anchor_k_max=1, prune_stable_only=False)
    assert _geometry_connectivity_mismatch(graph, sites[0], reactant, sites[0].positions) is None
    sites = optimise_adsorbate_site_positions(graph, reactant.smiles, reactant)
    assert len(sites) == 1
    assert _molecular_mapping_preserves_handedness(positions, sites[0].positions)

    class StationaryCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}

    # A stationary test calculator isolates the post-relaxation validation gate;
    # this is not a claim of a physically stable F-bound catalyst geometry.
    sites = prune_unstable_adsorbate_sites(
        graph,
        sites,
        reactant,
        StationaryCalculator(),
        frozen_indices=list(range(len(slab))),
    )
    assert len(sites) == 1
    assert _molecular_mapping_preserves_handedness(positions, sites[0].positions)


def test_fragmentation_rejects_unrepresentable_custom_mass():
    import pytest

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], masses=[1.5, 1.008])
    with pytest.raises(ValueError, match="Cannot represent H mass"):
        get_all_fragments(atoms, strip_dummies=True)


def test_explicit_bracket_hydrogens_and_requested_implicit_hydrogens():
    for smiles, add_hydrogens, formula in (
        ("[H]C", False, "CH"),
        ("[H]C", True, "CH4"),
        ("[CH3]", False, "CH3"),
    ):
        assert (
            build_reactant(smiles, add_hydrogens=add_hydrogens).atoms.get_chemical_formula()
            == formula
        )


def test_bottom_gas_product_endpoint_lifts_away_from_slab():
    from types import SimpleNamespace
    from autokmc.sites.stability.bond import _gas_product_neb_endpoint

    atoms_ab = Atoms(
        "CuHH",
        positions=[[0, 0, 8], [0, 0, 6], [0.74, 0, 6]],
        cell=[10, 10, 30],
        pbc=[True, True, False],
    )
    graph = nx.Graph(surface_side="bottom", pbc=[True, True, False])
    graph.add_node(1, element="H")
    graph.add_node(2, element="H")
    gas = SimpleNamespace(atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    endpoint, diagnostic = _gas_product_neb_endpoint(
        atoms_empty=atoms_ab[:1],
        atoms_ab=atoms_ab,
        n_slab=1,
        n_lat=0,
        n_react=2,
        react_nodes_ab=[1, 2],
        gas_reactant=gas,
        G=graph,
        lift_height=6.0,
    )
    np.testing.assert_allclose(
        endpoint.positions[1:, 2], 6.0 - diagnostic["selected_lift_height_ang"]
    )
