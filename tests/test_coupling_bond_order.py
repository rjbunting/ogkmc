"""Recombination preserves molecular identity and the connecting bond order."""

import networkx as nx
import pytest
from rdkit import Chem

from ogkmc.kmc import expansion
from ogkmc.sites.bond import derive_bond_templates, derive_dissociation_templates
from ogkmc.species.bond_chemistry import combine_fragments, get_all_fragments
from ogkmc.species.reactant import build_reactant
from ogkmc.species.smiles import canonical_atom_inventory_smiles


@pytest.mark.parametrize("strip_dummies", [False, True])
@pytest.mark.parametrize(
    "smiles,bond_type",
    [
        ("O=O", "DOUBLE"), ("N#N", "TRIPLE"), ("[C]=O", "DOUBLE"),
        ("C=C", "DOUBLE"), ("C#C", "TRIPLE"), ("CO", "SINGLE"),
    ],
)
def test_cleavage_recombination_restores_bond_order(smiles, bond_type, strip_dummies):
    pairs = get_all_fragments(smiles, strip_dummies=strip_dummies)
    pair = next(pair for pair in pairs if pair.element_a != "H" and pair.element_b != "H")
    products = combine_fragments(pair.smiles_a, pair.smiles_b)

    expected = canonical_atom_inventory_smiles(smiles, add_hydrogens=True)
    matches = [
        product for product in products
        if canonical_atom_inventory_smiles(product.smiles) == expected
    ]
    assert len(matches) == 1
    assert matches[0].bond_type == bond_type
    if smiles in {"O=O", "N#N", "[C]=O"}:
        assert len(products) == 1


@pytest.mark.parametrize(
    "fragment_a,fragment_b,expected,bond_type",
    [
        ("[O]", "[OH]", "[H]O[O]", "SINGLE"),
        ("[H]", "[H]", "[H][H]", "SINGLE"),
        ("[2H]", "[H]", "[H][2H]", "SINGLE"),
        ("*=O", "[O]", "O=O", "DOUBLE"),
        ("[O]", "*=O", "O=O", "DOUBLE"),
        ("*=O", "[H]", "[H][O]", "SINGLE"),
        ("[O]*", "*[O]", "[O][O]", "SINGLE"),
        ("*=O", "*[O]", "[O][O]", "SINGLE"),
    ],
)
def test_coupling_respects_each_attachment_capacity(fragment_a, fragment_b, expected, bond_type):
    products = combine_fragments(fragment_a, fragment_b)

    assert len(products) == 1
    assert canonical_atom_inventory_smiles(products[0].smiles) == expected
    assert products[0].bond_type == bond_type
    molecule = Chem.MolFromSmiles(products[0].smiles)
    assert all(atom.GetFormalCharge() == 0 for atom in molecule.GetAtoms())


@pytest.mark.parametrize(
    "molecule,atom,bond_type",
    [("O=O", "[O]", "DOUBLE"), ("N#N", "[N]", "TRIPLE")],
)
def test_dissociation_and_coupling_share_one_reversible_template(molecule, atom, bond_type):
    templates = derive_bond_templates([molecule, atom], add_hydrogens=False)
    recombinations = [
        template for template in templates
        if template.smiles_a == template.smiles_b == atom
    ]

    assert len(recombinations) == 1
    assert recombinations[0].smiles_c == molecule
    assert recombinations[0].bond_type == bond_type
    assert recombinations[0].source == "dissociation"


def test_runtime_oxygen_recombination_reuses_registered_dissociation(monkeypatch):
    oxygen = build_reactant("O=O", add_hydrogens=False, relax=False, partial_pressure_bar=0.2)
    atom = build_reactant("[O]", add_hydrogens=False, relax=False)
    graph = nx.Graph()
    registry = expansion.initialise_bond_registry(
        graph,
        reactants=[oxygen, atom],
        adsorbate_sites={"O=O": [], "[O]": []},
        templates=derive_dissociation_templates("O=O", add_hydrogens=False),
        expanded_smiles=["O=O"],
    )
    # Exercise real template derivation and registry deduplication while
    # isolating geometry work for the independent O + O2 candidate.
    built = []
    observed = []

    def ensure_species(_graph, smiles, _registry, **_kwargs):
        built.append(smiles)
        return True

    def enumerate_sites(_graph, _sites, templates, **_kwargs):
        observed.extend(templates)
        return []

    monkeypatch.setattr(expansion, "_ensure_species_known", ensure_species)
    monkeypatch.setattr(expansion, "find_bond_sites", enumerate_sites)
    expansion.expand_bond_sites_for_new_species(
        graph, "[O]", calculator=None, find_diffusion=False, verbose=False,
    )

    assert "[O][O]" not in built
    assert observed
    assert all(template.smiles_c != "[O][O]" for template in observed)
    assert ("[O]", "[O]", "O=O") in registry["templates"]
    assert not any(template.smiles_a == template.smiles_b == "[O]" for template in observed)
    assert registry["species"]["O=O"] is oxygen
    assert oxygen.partial_pressure_bar == 0.2
