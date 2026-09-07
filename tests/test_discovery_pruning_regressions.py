"""Exercise discovery with real enumeration, pruning, and controlled calculators."""

import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.calculator import Calculator, all_changes

from autokmc.core.graph import build_graph
from autokmc.io.calculators import CalculatorConfigError
from autokmc.kmc.expansion import (
    SpeciesExpansionError, expand_bond_sites_for_new_species, initialise_bond_registry,
)
from autokmc.sites.adsorbate import (
    AdsorbateSite, find_adsorbate_sites, rebuild_adsorbate_reverse_indexes,
)
from autokmc.sites.bond import (
    BondReactionSite, BondReactionTemplate, prune_unstable_bond_sites,
)
from autokmc.species.reactant import build_reactant
from autokmc.structure import StructureOptimisationError


class ControlledCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, failures=0, error=OSError):
        super().__init__()
        self.failures = failures
        self.error = error
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("calculator worker disconnected")
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def _hydrogen_surface():
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=7)
    slab.set_array("surface", np.asarray(slab.get_tags() == 1, dtype=np.int8))
    graph = build_graph(slab)
    reactant = build_reactant("[H]", add_hydrogens=False)
    reactant.energy = 0.0
    return graph, reactant


def _expand_hydrogen(graph, calculator):
    return expand_bond_sites_for_new_species(
        graph, "[H]", calculator=calculator,
        include_coupling=False, include_dissociation=False,
    )


def test_real_adsorption_pruning_retries_backend_failure_and_retains_sites():
    graph, reactant = _hydrogen_surface()
    initialise_bond_registry(graph, reactants=[reactant])
    calculator = ControlledCalculator(failures=2)
    _expand_hydrogen(graph, calculator)
    registry = graph.graph["bond_registry"]
    sites = registry["adsorbate_sites"]["[H]"]
    assert sites
    assert registry["expanded_species"] == {"[H]"}
    record = registry["expansion_failures"]["[H]"]["find_adsorbate_sites"]
    assert record["status"] == "recovered"
    assert record["attempts"] == 3
    assert graph.graph["adsorbate_sites"]["[H]"] == sites
    materialized = {n for site in sites for member in site.member_node_ids for n in member}
    assert materialized == {
        n for n, data in graph.nodes(data=True) if data.get("type") == "adsorbate"
    }
    calls = calculator.calls
    _expand_hydrogen(graph, calculator)
    assert calculator.calls == calls


def test_exhausted_pruning_remains_retryable_after_graph_roundtrip():
    graph, reactant = _hydrogen_surface()
    initialise_bond_registry(graph, reactants=[reactant])
    calculator = ControlledCalculator(failures=100)
    with pytest.raises(SpeciesExpansionError, match="find_adsorbate_sites"):
        _expand_hydrogen(graph, calculator)
    assert calculator.calls == 3
    registry = graph.graph["bond_registry"]
    assert "[H]" not in registry["adsorbate_sites"]
    assert "[H]" not in registry["expanded_species"]
    assert registry["expansion_failures"]["[H]"]["find_adsorbate_sites"]["status"] == "retry_exhausted"
    restored = pickle.loads(pickle.dumps(graph))
    _expand_hydrogen(restored, ControlledCalculator())
    assert restored.graph["bond_registry"]["adsorbate_sites"]["[H]"]
    assert restored.graph["bond_registry"]["expanded_species"] == {"[H]"}


def test_pruning_configuration_failure_is_fatal_without_retry():
    graph, reactant = _hydrogen_surface()
    initialise_bond_registry(graph, reactants=[reactant])
    calculator = ControlledCalculator(failures=100, error=CalculatorConfigError)
    with pytest.raises(CalculatorConfigError, match="worker disconnected"):
        _expand_hydrogen(graph, calculator)
    assert calculator.calls == 1
    assert not graph.graph["bond_registry"]["expanded_species"]


@pytest.mark.parametrize("initial_cap", [1, 2])
def test_removing_anchor_cap_restores_full_enumeration(initial_cap):
    graph, reactant = _hydrogen_surface()
    restricted = find_adsorbate_sites(
        graph, reactant, anchor_k_max=initial_cap, prune_stable_only=False,
    )
    uncapped = find_adsorbate_sites(
        graph, reactant, anchor_k_max=None, prune_stable_only=False,
    )
    fresh_graph, _ = _hydrogen_surface()
    fresh = find_adsorbate_sites(fresh_graph, reactant, prune_stable_only=False)
    signature = lambda sites: {
        tuple(member) for site in sites for member in site.members
    }
    assert signature(restricted) < signature(uncapped)
    assert signature(uncapped) == signature(fresh)
    assert len(uncapped) == 4
    assert sum(len(site.members) for site in uncapped) == 54


def _oxygen_bound_co_pair(swap=False):
    atoms = Atoms(
        "Pt2COH", positions=[[0, 0, 0], [3, 0, 0], [0, 0, 3.0], [0, 0, 1.8], [3, 0, 1.4]],
        cell=[20, 20, 20], pbc=True,
    )
    atoms.set_array("surface", np.array([1, 1, 2, 2, 2], dtype=np.int8))
    graph = build_graph(atoms)
    for nid, smiles, index, clique in (
        (2, "[C]=O", 0, None), (3, "[C]=O", 1, frozenset({0})),
        (4, "[H]", 0, frozenset({1})),
    ):
        graph.nodes[nid].update(
            reactant=smiles, reactant_index=index, iso_class=0, occupied=False, clique=clique,
        )
    a = AdsorbateSite(
        "[C]=O", 2, [None, frozenset({0})], atoms.positions[2:4], 0,
        members=[[None, frozenset({0})]], member_node_ids=[[2, 3]],
    )
    b = AdsorbateSite(
        "[H]", 1, [frozenset({1})], atoms.positions[4:], 0,
        members=[[frozenset({1})]], member_node_ids=[[4]],
    )
    graph.graph["adsorbate_sites"] = {a.reactant: [a], b.reactant: [b]}
    rebuild_adsorbate_reverse_indexes(graph)
    species = {
        a.reactant: SimpleNamespace(atoms=Atoms("CO"), graph=nx.Graph([(0, 1)])),
        b.reactant: SimpleNamespace(atoms=Atoms("H"), graph=nx.empty_graph(1)),
    }
    if swap:
        a, b = b, a
    reaction_site = BondReactionSite(
        BondReactionTemplate(a.reactant, b.reactant, "[H][C]=O"), 0,
        members=[(a, 0, b, 0, None, -1)],
        member_node_ids=[(a.member_node_ids[0], b.member_node_ids[0], [])],
        _member_cliques=[(a._member_cliques[0], b._member_cliques[0], ())],
        gas_product=True,
    )
    return graph, species, reaction_site


@pytest.mark.parametrize("swap", [False, True])
def test_bond_pruning_preserves_atom_indices_past_unbound_atoms(swap):
    graph, species, site = _oxygen_bound_co_pair(swap)
    assert prune_unstable_bond_sites(
        graph, [site], species, ControlledCalculator(), frozen_indices=[0, 1],
    ) == [site]


def test_bond_pruning_propagates_calculator_failure_without_removing_channels():
    graph, species, site = _oxygen_bound_co_pair()
    graph.graph["bond_reaction_sites"] = [site]
    with pytest.raises(StructureOptimisationError, match="worker disconnected"):
        prune_unstable_bond_sites(
            graph, [site], species, ControlledCalculator(failures=100),
        )
    assert graph.graph["bond_reaction_sites"] == [site]
