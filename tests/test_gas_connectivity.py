"""A force-converged gas reference must still represent its requested molecule."""

from types import SimpleNamespace
import pickle

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import networkx as nx
import numpy as np
import pytest

from autokmc.kmc.expansion import SpeciesExpansionError, expand_bond_sites_for_new_species
from autokmc.species import ReactantConnectivityError, ReactantGasUnstableError, build_reactant


class TargetMinimum(Calculator):
    """Controlled quadratic minimum; optimization and graph checks remain real."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, target=None):
        super().__init__()
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.initial_positions = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.initial_positions is None:
            self.initial_positions = atoms.positions.copy()
        if self.target is None:
            center = atoms.positions.mean(axis=0)
            self.target = center + 1.05 * (atoms.positions - center) + [0.1, -0.2, 0.3]
        displacement = atoms.positions - self.target
        self.results = {
            "energy": float(0.5 * np.sum(displacement**2)),
            "forces": -displacement,
        }


@pytest.mark.parametrize("smiles,target,missing,extra", [
    ("O=O", [[6, 6, 6], [10, 6, 6]], "[(0, 1)]", "[]"),
    ("[C][C][C]", [[6, 6, 6], [7.4, 6, 6], [6.7, 7.2, 6]], "[]", "[(0, 2)]"),
    ("[C][O][N]", [[6, 6, 6], [4.7, 6, 6], [7.3, 6, 6]], "[(1, 2)]", "[(0, 2)]"),
], ids=["dissociation", "extra-bond", "rearrangement"])
def test_real_relaxation_rejects_changed_chemistry_before_thermochemistry(
    monkeypatch, smiles, target, missing, extra,
):
    calculator = TargetMinimum(target)
    monkeypatch.setattr(
        "autokmc.thermo.free_energy.compute_gas_thermo",
        lambda *_args, **_kwargs: pytest.fail("invalid gas reached thermochemistry"),
    )
    with pytest.raises(ReactantConnectivityError) as caught:
        build_reactant(
            smiles, add_hydrogens=False, calculator=calculator, fmax=1e-7,
            free_energy_options=SimpleNamespace(enabled=True), free_energy_temperature_k=300,
        )
    assert np.max(np.linalg.norm(calculator.results["forces"], axis=1)) < 1e-7
    assert f"missing bonds {missing}" in str(caught.value)
    assert f"extra bonds {extra}" in str(caught.value)
    assert smiles in str(caught.value)
    assert isinstance(caught.value, ReactantGasUnstableError)


@pytest.mark.parametrize("smiles", ["O=O", "[C-]#[O+]", "O", "[O]", "[2H][H]"])
def test_real_relaxation_accepts_correct_structure_and_preserves_energy(smiles):
    calculator = TargetMinimum()
    reactant = build_reactant(smiles, calculator=calculator, fmax=1e-7)
    np.testing.assert_allclose(reactant.atoms.positions, calculator.target, atol=1e-7)
    assert not np.allclose(reactant.atoms.positions, calculator.initial_positions)
    assert reactant.energy == pytest.approx(0, abs=1e-12)
    assert reactant.atoms.get_potential_energy() == pytest.approx(reactant.energy)
    assert not reactant.graph.graph["pbc"].any()


@pytest.mark.parametrize("smiles,add_hydrogens,n_atoms,n_bonds", [
    ("C=O", True, 4, 3), ("C=O", False, 2, 1),
    ("O", True, 3, 2), ("O", False, 1, 0),
    ("[OH]", False, 2, 1), ("[H]O[O]", False, 3, 2),
    ("[2H][2H]", False, 2, 1),
    ("c1ccccc1", True, 12, 12), ("c1ccccc1", False, 6, 6),
])
def test_validation_uses_the_materialized_hydrogen_inventory(smiles, add_hydrogens, n_atoms, n_bonds):
    reactant = build_reactant(smiles, add_hydrogens=add_hydrogens, relax=False)
    assert reactant.graph.number_of_nodes() == n_atoms
    assert reactant.graph.number_of_edges() == n_bonds


@pytest.mark.parametrize("with_calculator", [False, True])
def test_incorrect_generated_geometry_is_rejected_without_ase_relaxation(monkeypatch, with_calculator):
    wrong = Atoms("O2", positions=[[6, 6, 6], [10, 6, 6]], cell=[20, 20, 20])
    monkeypatch.setattr("autokmc.species.reactant._smiles_to_atoms", lambda *_a, **_k: wrong.copy())
    monkeypatch.setattr(
        "autokmc.species.reactant._optimise",
        lambda *_a, **_k: pytest.fail("ASE relaxation should be disabled"),
    )
    with pytest.raises(ReactantConnectivityError, match="missing bonds") as caught:
        build_reactant("O=O", relax=False, calculator=TargetMinimum() if with_calculator else None)
    assert not isinstance(caught.value, ReactantGasUnstableError)


@pytest.mark.parametrize("wrong,smiles,match", [
    (Atoms("O"), "O=O", "atom indices"),
    (Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.2]]), "O=O", "changed atom"),
    (Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]]), "[2H][2H]", "changed atom"),
])
def test_atom_count_element_and_isotope_identity_are_validated(monkeypatch, wrong, smiles, match):
    monkeypatch.setattr("autokmc.species.reactant._smiles_to_atoms", lambda *_a, **_k: wrong.copy())
    with pytest.raises(ReactantConnectivityError, match=match):
        build_reactant(smiles, relax=False)


def test_configured_cutoff_is_used_without_silently_replacing_the_produced_graph():
    with pytest.raises(ReactantConnectivityError, match="nl_mult=0.1.*missing bonds"):
        build_reactant("O=O", nl_mult=0.1, relax=False)


def test_runtime_expansion_excludes_gas_unstable_species_across_restart(monkeypatch):
    graph = nx.Graph()
    monkeypatch.setattr(
        "autokmc.kmc.expansion.find_adsorbate_sites",
        lambda *_a, **_k: pytest.fail("invalid gas reached site enumeration"),
    )
    assert expand_bond_sites_for_new_species(
        graph, "O=O", calculator=TargetMinimum([[6, 6, 6], [10, 6, 6]]),
        include_dissociation=False, include_coupling=False,
    ) == []
    registry = graph.graph["bond_registry"]
    assert registry["species"]["O=O"] is None
    assert registry["adsorbate_sites"]["O=O"] == []
    assert "O=O" in registry["expanded_species"]
    record = registry["expansion_failures"]["O=O"]["build_reactant"]
    assert record["status"] == "unstable_gas"
    assert record["attempts"] == 1
    restored = pickle.loads(pickle.dumps(graph))
    monkeypatch.setattr(
        "autokmc.kmc.expansion.build_reactant",
        lambda *_a, **_k: pytest.fail("gas-unstable species was rebuilt"),
    )
    assert expand_bond_sites_for_new_species(restored, "O=O", calculator=None) == []


def test_unrelaxed_geometry_failure_is_not_permanently_classified(monkeypatch):
    graph = nx.Graph()
    def fail(*_args, **_kwargs):
        raise ReactantConnectivityError("invalid generated geometry")
    monkeypatch.setattr("autokmc.kmc.expansion.build_reactant", fail)
    with pytest.raises(SpeciesExpansionError):
        expand_bond_sites_for_new_species(graph, "O=O", calculator=None)
    registry = graph.graph["bond_registry"]
    assert "O=O" not in registry["species"]
    assert registry["expansion_failures"]["O=O"]["build_reactant"]["attempts"] == 3


def test_runtime_leaf_rejection_filters_only_affected_templates(monkeypatch):
    from autokmc.kmc import expansion
    from autokmc.sites.bond import BondReactionTemplate
    graph = nx.Graph()
    valid = BondReactionTemplate("[H]", "[H]", "[H][H]")
    invalid = BondReactionTemplate("[H]O[O]", "O=O", "[H]OOO[O]")
    builds = []
    enumerated = []
    def build(smiles, **kwargs):
        builds.append(smiles)
        if smiles == invalid.smiles_c:
            raise ReactantGasUnstableError("missing bonds [(2, 3)]")
        return build_reactant(smiles, add_hydrogens=False, relax=False)
    monkeypatch.setattr(expansion, "build_reactant", build)
    monkeypatch.setattr(expansion, "find_adsorbate_sites", lambda *_a, **_k: [])
    monkeypatch.setattr(expansion, "derive_dissociation_templates", lambda *_a, **_k: [invalid, valid])
    def enumerate_sites(graph, sites, templates, **kwargs):
        enumerated.extend(templates)
        return []
    monkeypatch.setattr(expansion, "find_bond_sites", enumerate_sites)
    assert expand_bond_sites_for_new_species(
        graph, "[H]O[O]", calculator=None, include_coupling=False,
    ) == []
    assert enumerated == [valid]
    assert builds.count(invalid.smiles_c) == 1
    registry = graph.graph["bond_registry"]
    assert registry["species"][invalid.smiles_c] is None
    assert registry["invalid_templates"][0]["reason"] == "unstable_gas_species"
    assert invalid.smiles_c not in {key[2] for key in registry["templates"]}
    assert "[H]O[O]" in registry["expanded_species"]
