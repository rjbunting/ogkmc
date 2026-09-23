"""Chemical identity must agree across parsing, templates, and persistence."""

from collections import Counter

import networkx as nx
import pytest
from rdkit import Chem

from ogkmc.io.config import BondCfg, ReactantCfg, RunConfig
from ogkmc.io.config_validation import validate_config
from ogkmc.kmc.expansion import initialise_bond_registry, bond_species_known
from ogkmc.kmc.restart import reactants_for_checkpoint
from ogkmc.sites.bond import derive_bond_templates, derive_coupling_templates
from ogkmc.species.bond_chemistry import (
    combine_fragments, get_all_fragments, _strip_dummy_atoms_from_smiles,
)
from ogkmc.species.reactant import build_reactant, ReactantDefinitionError
from ogkmc.species.smiles import (
    canonical_smiles, canonical_atom_inventory_smiles, molecule_from_smiles,
    molecule_from_reactant, reactant_atom_inventory_smiles, SmilesError,
    smiles_to_dirname,
)


def _config(*reactants, bond_enabled=True):
    cfg = RunConfig(reactants=list(reactants), bond=BondCfg(enabled=bond_enabled))
    cfg.calculator.import_path = "ase.calculators.emt.EMT"
    return cfg


@pytest.mark.parametrize(
    "first,second",
    [
        ("[C]=O", "O=[C]"), ("[C]=O", "[C-]#[O+]"),
        ("[C]=O", "[O+]#[C-]"), ("CO", "OC"),
        ("[OH]", "[H][O]"), ("C1=CC=CC=C1", "c1ccccc1"),
        ("[C]=O", "[C:12]=[O:42]"),
        ("[13C]=[18O]", "[18O+]#[13C-]"),
    ],
)
def test_equivalent_labels_have_one_identity(first, second):
    assert canonical_smiles(first) == canonical_smiles(second)
    for add_hydrogens in (False, True):
        assert canonical_atom_inventory_smiles(first, add_hydrogens=add_hydrogens) == (
            canonical_atom_inventory_smiles(second, add_hydrogens=add_hydrogens)
        )


@pytest.mark.parametrize(
    "first,second",
    [
        ("[H]O", "O"), ("[H]", "[2H]"), ("[C]=O", "[13C]=O"),
        ("O=O", "[O][O]"), ("[O]", "[O-]"),
        ("C[C@H](O)F", "C[C@@H](O)F"), ("F/C=C/F", "F/C=C\\F"),
        ("CC=O", "C=CO"), ("CCO", "COC"),
    ],
)
def test_distinct_species_are_not_collapsed(first, second):
    assert canonical_smiles(first) != canonical_smiles(second)
    add_hydrogens = first != "[H]O"
    assert canonical_atom_inventory_smiles(first, add_hydrogens=add_hydrogens) != (
        canonical_atom_inventory_smiles(second, add_hydrogens=add_hydrogens)
    )


@pytest.mark.parametrize("add_hydrogens", [False, True])
@pytest.mark.parametrize("smiles", ["C=O", "[H]O", "[OH]", "[2H]O", "[C-]#[O+]"])
def test_actual_atom_inventory_round_trips(smiles, add_hydrogens):
    reactant = build_reactant(smiles, add_hydrogens=add_hydrogens, relax=False)
    inventory = reactant_atom_inventory_smiles(reactant)
    molecule = molecule_from_smiles(inventory, add_hydrogens=False)
    assert all(atom.GetNumImplicitHs() == 0 for atom in molecule.GetAtoms())
    assert Counter(atom.GetAtomicNum() for atom in molecule.GetAtoms()) == Counter(
        reactant.atoms.numbers
    )
    assert canonical_atom_inventory_smiles(inventory) == inventory
    assert canonical_smiles(inventory) == inventory
    assert Chem.MolToSmiles(molecule_from_reactant(reactant)) == inventory


@pytest.mark.parametrize("smiles", ["", "*", "[H].[O]", "O water", "O |$x$|", "not-smiles"])
def test_molecular_boundaries_reject_invalid_or_ambiguous_input(smiles):
    with pytest.raises((SmilesError, ReactantDefinitionError)):
        build_reactant(smiles, relax=False)
    with pytest.raises(SmilesError):
        combine_fragments(smiles, "[H]")
    with pytest.raises(ValueError):
        validate_config(_config(ReactantCfg(smiles=smiles)))


@pytest.mark.parametrize("smiles", ["[O-]", "[NH4+]", "C[N+](=O)[O-]"])
def test_unsupported_charged_bond_species_fail_before_enumeration(smiles):
    with pytest.raises(SmilesError, match="charge-free"):
        get_all_fragments(smiles)
    with pytest.raises(SmilesError, match="charge-free"):
        combine_fragments(smiles, "[H]")
    with pytest.raises(SmilesError, match="charge-free"):
        derive_bond_templates([smiles, "[H]"])
    with pytest.raises(ValueError, match="charge-free"):
        validate_config(_config(ReactantCfg(smiles=smiles)))
    validate_config(_config(ReactantCfg(smiles=smiles), bond_enabled=False))


@pytest.mark.parametrize("first,second", [
    (ReactantCfg(smiles="[C]=O"), ReactantCfg(smiles="[C-]#[O+]")),
    (ReactantCfg(smiles="O", add_hydrogens=False), ReactantCfg(smiles="[O]")),
    (ReactantCfg(smiles="[H]O", add_hydrogens=False), ReactantCfg(smiles="[OH]")),
    (ReactantCfg(smiles="C=O", add_hydrogens=True),
     ReactantCfg(smiles="[H]C([H])=O", add_hydrogens=False)),
])
def test_duplicate_feeds_are_rejected_by_actual_inventory(first, second):
    with pytest.raises(ValueError, match="duplicates"):
        validate_config(_config(first, second))


def test_co_and_formaldehyde_feeds_remain_distinct():
    validate_config(_config(
        ReactantCfg(smiles="C=O", add_hydrogens=True),
        ReactantCfg(smiles="[C]=O", add_hydrogens=False),
    ))


@pytest.mark.parametrize("strip_dummies", [False, True])
def test_charged_co_spelling_recombines_to_the_same_neutral_identity(strip_dummies):
    pair = get_all_fragments("[C-]#[O+]", strip_dummies=strip_dummies)[0]
    product, = combine_fragments(pair.smiles_a, pair.smiles_b)
    assert canonical_smiles(product.smiles) == "[C]=O"
    assert pair.bond_type == product.bond_type == "DOUBLE"
    assert str(Chem.MolFromSmiles(product.smiles).GetBondWithIdx(0).GetBondType()) == "DOUBLE"


def test_generated_co_reuses_charged_feed_and_its_pressure():
    feed = build_reactant("[C-]#[O+]", relax=False, partial_pressure_bar=0.3)
    graph = nx.Graph()
    registry = initialise_bond_registry(graph, reactants=[feed])
    assert bond_species_known(graph, "[C]=O")
    assert bond_species_known(graph, "[O+]#[C-]")
    template, = derive_coupling_templates(
        ["[C]", "[O]"], include_homo=False,
        atom_inventory_smiles={feed.smiles: feed.atom_inventory_smiles},
    )
    assert registry["species"][template.smiles_c] is feed
    assert feed.partial_pressure_bar == 0.3


def test_object_coupling_preserves_internal_multiple_bonds():
    co = build_reactant("[C]=O", relax=False)
    oh = build_reactant("[OH]", relax=False)
    assert {(p.smiles, p.bond_type) for p in combine_fragments(co, oh)} == {
        (p.smiles, p.bond_type) for p in combine_fragments("[C]=O", "[OH]")
    }


def test_dummy_cleanup_does_not_erase_explicit_hydrogens():
    assert _strip_dummy_atoms_from_smiles("[H][O]*") == "[H][O]"
    assert _strip_dummy_atoms_from_smiles("[2H][O]*") == "[2H][O]"


def test_join_does_not_reduce_valid_sulfur_bonds_elsewhere():
    products = combine_fragments("[C]S(=O)C", "[H]")
    carbon_additions = [product for product in products if product.atom_idx_a == 0]
    assert carbon_additions
    for product in carbon_additions:
        molecule = Chem.MolFromSmiles(product.smiles)
        assert any(
            {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {"S", "O"}
            and bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in molecule.GetBonds()
        )


def test_checkpoint_does_not_collapse_an_explicit_hydrogen_into_implicit_valence():
    hydroxyl = build_reactant("[H]O", add_hydrogens=False, relax=False)
    oxygen = build_reactant("O", add_hydrogens=False, relax=False)
    graph = nx.Graph()
    initialise_bond_registry(graph, reactants=[hydroxyl, oxygen])
    assert reactants_for_checkpoint([hydroxyl, oxygen], graph) == [hydroxyl, oxygen]


def test_stored_atom_inventory_is_checked_against_real_isotope_masses():
    hydrogen = build_reactant("[H]", relax=False)
    hydrogen.atom_inventory_smiles = "[1H]"
    with pytest.raises(ValueError, match="inconsistent"):
        reactant_atom_inventory_smiles(hydrogen)


def test_stored_inventory_cannot_disagree_with_the_declared_bond_order():
    oxygen = build_reactant("O=O", relax=False)
    oxygen.atom_inventory_smiles = "[O][O]"
    with pytest.raises(SmilesError, match="stored chemical identity"):
        reactant_atom_inventory_smiles(oxygen)


def test_object_coupling_rejects_an_untracked_permutation_of_elements():
    co = build_reactant("[C]=O", relax=False)
    co.atoms.numbers[:] = co.atoms.numbers[::-1]
    with pytest.raises(SmilesError, match="indexed atoms"):
        combine_fragments(co, "[H]")


def test_ase_charge_metadata_is_not_silently_discarded():
    from ase import Atoms

    oxygen = Atoms("O", charges=[-1])
    with pytest.raises(SmilesError, match="charged Atoms"):
        combine_fragments(oxygen, "[H]")
    with pytest.raises(SmilesError, match="charged Atoms"):
        get_all_fragments(oxygen)


def test_template_elements_follow_the_canonical_fragment_order():
    template, = derive_coupling_templates(["[H]", "O=O"], include_homo=False)
    assert (template.smiles_a, template.element_a) == ("O=O", "O")
    assert (template.smiles_b, template.element_b) == ("[H]", "H")


@pytest.mark.parametrize("first,second", [
    ("F/C=C/F", "F/C=C\\F"), ("C1CCCCC1", "c1ccccc1"),
    ("C(C)", "C[C]"), ("C" * 80 + "O", "C" * 80 + "N"),
])
def test_distinct_labels_cannot_share_an_artifact_folder(first, second):
    first_dir, second_dir = smiles_to_dirname(first), smiles_to_dirname(second)
    assert first_dir.casefold() != second_dir.casefold()
    assert len(first_dir) <= 64
    assert len(second_dir) <= 64
    assert not any(character in first_dir for character in '/\\:*?"<>|')


def test_registry_rejects_conflicting_aliases_without_overwriting_feed():
    first = build_reactant("[C]=O", relax=False, partial_pressure_bar=0.3)
    second = build_reactant("[C-]#[O+]", relax=False, partial_pressure_bar=0.7)
    graph = nx.Graph()
    registry = initialise_bond_registry(graph, reactants=[first])
    with pytest.raises(ValueError, match="already registered"):
        initialise_bond_registry(graph, reactants=[second])
    assert registry["species"] == {"[C]=O": first}
    assert first.partial_pressure_bar == 0.3


def test_known_species_lookup_uses_full_feed_inventory():
    feed = build_reactant("C=O", add_hydrogens=True, relax=False)
    graph = nx.Graph()
    initialise_bond_registry(graph, reactants=[feed])
    assert bond_species_known(graph, "[H]C([H])=O")
    assert not bond_species_known(graph, "[C]=O")


def test_cache_identity_normalizes_aliases_without_erasing_hydrogens():
    from ogkmc.io.calculation_cache import calculation_cache_key, _operation_key

    def key(smiles):
        return calculation_cache_key(
            kind="adsorption", identity={"reactant_smiles": smiles},
            parameters={}, inputs={"reactant_smiles": smiles},
        )

    assert key("[C]=O") == key("[C-]#[O+]")
    assert key("[H]O") != key("O")
    assert _operation_key({"reactant_smiles": "[C]=O"}) == _operation_key(
        {"reactant_smiles": "[C-]#[O+]"}
    )
    assert _operation_key({"reactant_smiles": "[H]O"}) != _operation_key(
        {"reactant_smiles": "O"}
    )


def test_stereoisomers_keep_separate_reaction_documents(tmp_path, make_reaction):
    import json
    from ogkmc.io.persistence import ReactionWriter

    writer = ReactionWriter(tmp_path)
    first = make_reaction(smiles="F/C=C/F")
    second = make_reaction(smiles="F/C=C\\F")
    first_path = writer.ensure_reaction(first)
    second_path = writer.ensure_reaction(second)
    writer.close()
    assert first_path != second_path
    assert json.loads((first_path / "reaction.json").read_text())["reactant_smiles"] == (
        "F/C=C/F"
    )
    assert json.loads((second_path / "reaction.json").read_text())["reactant_smiles"] == (
        "F/C=C\\F"
    )


def test_append_preserves_legacy_folder_and_committed_event_counts(tmp_path, make_reaction):
    import json
    from ogkmc.io.persistence import ReactionWriter

    reaction = make_reaction(smiles="[O]")
    writer = ReactionWriter(tmp_path, run_id="legacy-smiles")
    folder = writer.ensure_reaction(reaction)
    writer.record(reaction=reaction, step=1, time_s=0.1, tau_s=0.1)
    writer.close()
    legacy_parent = folder.parent.with_name("(O)")
    folder.parent.rename(legacy_parent)
    legacy_folder = legacy_parent / folder.name
    resumed = ReactionWriter(tmp_path, append=True, run_id="legacy-smiles")
    assert resumed.ensure_reaction(reaction) == legacy_folder
    resumed.record(reaction=reaction, step=2, time_s=0.2, tau_s=0.1)
    resumed.close()
    assert json.loads((legacy_folder / "reaction.json").read_text())["stats"]["count"] == 2
    assert not folder.exists()
