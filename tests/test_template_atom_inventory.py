"""Bond templates conserve configured feed atoms throughout network growth."""

from collections import Counter
import pickle
from types import SimpleNamespace

import networkx as nx
import pytest

from autokmc.io.config import BondCfg, ReactantCfg, RunConfig
from autokmc.kmc import expansion
from autokmc.sites.bond import derive_bond_templates, derive_coupling_templates
from autokmc.species.reactant import build_reactant
from autokmc.species.smiles import (
    canonical_atom_inventory_smiles,
    reactant_atom_inventory_smiles,
)
from autokmc.workflow.network import (
    SpeciesNetworkBuilder,
    derive_configured_bond_templates,
)


def _feed(add_hydrogens=True):
    configs = [
        ReactantCfg(smiles="C=O", add_hydrogens=add_hydrogens),
        ReactantCfg(smiles="[H]", add_hydrogens=False),
    ]
    reactants = [
        build_reactant(config.smiles, add_hydrogens=config.add_hydrogens)
        for config in configs
    ]
    reactants[0].partial_pressure_bar = 0.37
    reactants[0].gibbs_energy = -1.5
    return configs, reactants


def _assert_inventory(templates, species):
    assert templates
    for template in templates:
        reactants = [species[template.smiles_a], species[template.smiles_b]]
        product = species[template.smiles_c]
        inventory = sum(
            (Counter(reactant.atoms.numbers) for reactant in reactants), Counter(),
        )
        assert Counter(product.atoms.numbers) == inventory


@pytest.mark.parametrize("add_hydrogens,formula", [(True, "CH3O"), (False, "CHO")])
def test_configured_coupling_preserves_feed_hydrogen_policy(add_hydrogens, formula):
    configs, reactants = _feed(add_hydrogens)
    templates = derive_configured_bond_templates(
        configs, iter(reactants),
        BondCfg(include_dissociation=False, include_homo_coupling=False),
    )
    assert len(templates) == 2
    species = {reactant.smiles: reactant for reactant in reactants}
    for template in templates:
        assert {template.smiles_a, template.smiles_b} == {"C=O", "[H]"}
        product = build_reactant(template.smiles_c, add_hydrogens=False)
        assert product.atoms.get_chemical_formula() == formula
        species[template.smiles_c] = product
    _assert_inventory(templates, species)
    assert species["C=O"] is reactants[0]
    assert species["C=O"].partial_pressure_bar == 0.37
    assert species["C=O"].gibbs_energy == -1.5


def test_template_public_api_applies_h_policy_to_both_families():
    templates = derive_bond_templates(
        iter(["C=O", "[H]"]), add_hydrogens=True, include_homo_coupling=False,
    )
    assert {template.source for template in templates} == {"coupling", "dissociation"}
    species = {
        label: build_reactant(label, add_hydrogens=label in {"C=O", "[H]"})
        for template in templates
        for label in (template.smiles_a, template.smiles_b, template.smiles_c)
    }
    _assert_inventory(templates, species)
    assert all(
        species[template.smiles_c].atoms.get_chemical_formula() == "CH3O"
        for template in templates if template.source == "coupling"
    )


def test_coupling_reuses_known_feed_product_label():
    methanol = build_reactant("CO", add_hydrogens=True, partial_pressure_bar=0.25)
    templates = derive_coupling_templates(
        ["[CH3]", "[OH]"], include_homo=False,
        atom_inventory_smiles={"CO": methanol.atom_inventory_smiles},
    )
    assert len(templates) == 1
    assert templates[0].smiles_c == "CO"
    assert methanol.partial_pressure_bar == 0.25


def test_initial_network_materializes_correct_products_and_preserves_feed(tmp_path, monkeypatch):
    from autokmc.sites import adsorbate, bond

    configs, reactants = _feed()
    feed = reactants[0]
    graph = nx.Graph()
    observed = []

    def enumerate_sites(_graph, _sites, templates, **kwargs):
        species = kwargs["gas_species"]
        _assert_inventory(templates, species)
        observed.extend(templates)
        return []

    monkeypatch.setattr(adsorbate, "find_adsorbate_sites", lambda *args, **kwargs: [])
    monkeypatch.setattr(bond, "find_bond_sites", enumerate_sites)
    builder = SpeciesNetworkBuilder(
        cfg=RunConfig(
            reactants=configs,
            bond=BondCfg(
                enabled=True, include_dissociation=False, include_homo_coupling=False,
            ),
        ),
        identity=SimpleNamespace(output_dir=tmp_path),
        graph=graph,
        calculator_resource=None,
        frozen_indices=None,
        thermo_runtime=SimpleNamespace(vibration_cache_root=None, options=None),
    )
    builder._discover_bonds(reactants, [])
    assert len(observed) == 2
    species = graph.graph["bond_registry"]["species"]
    assert len(species) == 4
    assert species["C=O"] is feed
    assert feed.partial_pressure_bar == 0.37
    assert canonical_atom_inventory_smiles("C=O", add_hydrogens=True) not in species


@pytest.mark.parametrize("legacy", [False, True])
def test_runtime_expansion_couples_new_species_with_full_feed_inventory(monkeypatch, legacy):
    _, reactants = _feed()
    if legacy:
        for reactant in reactants:
            del reactant.atom_inventory_smiles
    graph = nx.Graph()
    registry = expansion.initialise_bond_registry(
        graph, reactants=reactants, adsorbate_sites={"C=O": [], "[H]": []},
        expanded_smiles=["C=O"],
    )
    # Exercise the same persisted objects that a resumed run expands.
    graph = pickle.loads(pickle.dumps(graph))
    registry = graph.graph["bond_registry"]
    observed = []

    def enumerate_sites(_graph, _sites, templates, **kwargs):
        _assert_inventory(templates, kwargs["gas_species"])
        observed.extend(templates)
        return []

    monkeypatch.setattr(expansion, "find_adsorbate_sites", lambda *args, **kwargs: [])
    monkeypatch.setattr(expansion, "find_bond_sites", enumerate_sites)
    expansion.expand_bond_sites_for_new_species(
        graph, "[H]", calculator=None, add_hydrogens=False,
        include_dissociation=False, include_homo_coupling=False,
    )
    assert len(observed) == 2
    assert len(registry["species"]) == 4
    assert registry["species"]["C=O"].partial_pressure_bar == 0.37
    assert registry["species"]["C=O"].gibbs_energy == -1.5
    for template in observed:
        assert registry["species"][template.smiles_c].atoms.get_chemical_formula() == "CH3O"


def test_legacy_inventory_recovery_rejects_mismatched_atoms():
    reactant = build_reactant("C=O")
    del reactant.atom_inventory_smiles
    reactant.atoms.pop(-1)
    with pytest.raises(ValueError, match="atom inventory inconsistent"):
        reactant_atom_inventory_smiles(reactant)
