"""Direct stability APIs cannot omit occupied spectator molecules in FE mode."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.sites.adsorbate import AdsorbateSiteLateral
from autokmc.sites.bond import BondReactionLateral
from autokmc.sites.diffusion import DiffusionLateral
from autokmc.sites.stability import adsorption, bond, diffusion
from autokmc.thermo.free_energy import FreeEnergyOptions


class _BuiltState(Exception):
    pass


def _graph():
    graph = nx.Graph(cell=np.diag([30.0, 30.0, 30.0]))
    for node in range(3):
        graph.add_node(node, type="surface", element="Cu", index=node,
                       position=[5.0 * node, 0.0, 0.0])
    graph.add_edges_from([(0, 1), (1, 2)])
    for node, symbol, slab, occupied, siblings in (
        (10, "H", 0, True, []),
        (11, "H", 1, False, []),
        (20, "H", 0, False, [21]),
        (21, "H", 1, False, [20]),
        (30, "C", 2, True, [31]),
        (31, "O", None, True, [30]),
        (40, "N", 2, False, []),
    ):
        graph.add_node(
            node, type="adsorbate", element=symbol, occupied=occupied,
            is_bonded=slab is not None,
            clique=frozenset() if slab is None else frozenset([slab]),
            siblings=siblings, reactant_index=node, iso_class=0,
            reactant="CO" if node in {30, 31} else symbol,
            position=[10.0 if node in {30, 31} else 0.0, 0.0, 1.2 + 0.01 * node],
        )
        if slab is not None:
            graph.add_edge(slab, node)
    graph.add_edges_from([(20, 21), (30, 31)])
    return graph


def _site(nodes, clique):
    return SimpleNamespace(
        iso_class=0, member_node_ids=[nodes],
        _member_cliques=[(frozenset([clique]),)],
        lateral_classes=[], reactant="[H]",
    )


@pytest.mark.parametrize("family", ["adsorption", "diffusion", "bond"])
@pytest.mark.parametrize("already_full", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_direct_stability_promotes_current_whole_surface(
    family, already_full, registered, monkeypatch,
):
    graph = _graph()
    site_a, site_b, site_c = _site([10], 0), _site([11], 1), _site([20, 21], 0)
    local_graph = graph.subgraph([0, 10]).copy()
    if already_full:
        local_graph.graph["environment_scope"] = "all_occupied"
    if family == "adsorption":
        module = adsorption
        parent = site_a
        lateral = AdsorbateSiteLateral(0, ego_graph=local_graph, members=[0, 1])
        check = module.check_site_stability
        classify = module.check_adsorbate_site_lateral
        builder_name = "_build_stability_atoms"
        expected_reacting = 1
    elif family == "diffusion":
        module = diffusion
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11])],
            members=[(site_a, 0, site_b, 0)], lateral_classes=[],
        )
        lateral = DiffusionLateral(0, ego_graph=local_graph, members=[0, 1])
        check = module.check_diffusion_stability
        classify = module.check_diffusion_site_lateral
        builder_name = "_build_diffusion_atoms"
        expected_reacting = 1
    else:
        module = bond
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11], [20, 21])],
            members=[(site_a, 0, site_b, 0, site_c, 0)], lateral_classes=[],
            template=SimpleNamespace(is_symmetric=True),
        )
        lateral = BondReactionLateral(0, ego_graph=local_graph, members=[0, 1])
        check = module.check_bond_site_stability
        classify = module.check_bond_site_lateral
        builder_name = "_build_bond_atoms"
        expected_reacting = 2
    lateral.stable = True
    lateral.g_stale = 123.0
    parent.lateral_classes = [lateral] if registered else []
    parent._member_lc = {0: lateral, 1: lateral}
    build = getattr(module, builder_name)
    captured = []

    def inspect(*args, **kwargs):
        result = build(*args, **kwargs)
        captured.append(result)
        raise _BuiltState

    monkeypatch.setattr(module, builder_name, inspect)
    with pytest.raises(_BuiltState):
        check(
            graph, parent, 0, lateral, calculator=None,
            free_energy_options=FreeEnergyOptions(enabled=True),
            free_energy_temperature_k=500.0,
        )
    atoms, n_slab, n_lat = captured[0][:3]
    assert (n_slab, n_lat) == (3, 2)
    assert len(atoms) == n_slab + n_lat + expected_reacting
    assert atoms.get_chemical_symbols()[n_slab:n_slab + n_lat] == ["C", "O"]
    assert {30, 31} <= set(lateral.ego_graph)
    assert 40 not in lateral.ego_graph
    assert lateral.ego_graph.graph["environment_scope"] == "all_occupied"
    assert lateral.stable is None
    assert not hasattr(lateral, "g_stale")
    assert lateral.members == [0]
    assert parent._member_lc == {}
    assert any(lateral is item for group in parent._lateral_fp_index.values() for item in group)
    assert parent.lateral_classes == [lateral]
    assert classify(graph, parent, 0, include_all_occupied=True) is lateral


def test_intended_spectator_topology_detects_already_missing_remote_bond():
    from ase import Atoms

    graph = _graph()
    atoms = Atoms("Cu3COH", positions=[
        [0, 0, 0], [5, 0, 0], [10, 0, 0],
        [10, 0, 8], [10, 0, 9.1], [0, 0, 1.2],
    ], cell=graph.graph["cell"], pbc=True)
    # The spectator already misses its intended bond before relaxation, so
    # comparing identical before/after structures alone cannot detect it.
    adsorption._check_connectivity_stable(
        atoms, atoms, 3, 3, "occupied", 0.9, relevant_indices={3, 4, 5}, n_lat=2,
    )
    with pytest.raises(adsorption.AdsorbateDissociationError, match="intended surface bond"):
        adsorption._check_intended_coordination_stable(
            atoms, graph, [30, 31], 3, 0, 0.9, self_node_order=[30, 31],
        )


@pytest.mark.parametrize("valid_spectator", [False, True])
def test_direct_adsorption_computes_stationary_full_surface_modes(valid_spectator):
    from ase.calculators.calculator import Calculator, all_changes

    graph = _graph()
    graph.nodes[10]["position"] = [0.0, 0.0, 1.5]
    graph.nodes[30]["position"] = [10.0, 0.0, 1.6 if valid_spectator else 8.0]
    graph.nodes[31]["position"] = [10.0, 0.0, 2.75 if valid_spectator else 9.15]
    references = {}
    for node in [0, 1, 2, 10, 30, 31]:
        data = graph.nodes[node]
        references.setdefault(data["element"], []).append(data["position"])

    class HarmonicWells(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            displacements = []
            for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
                centers = np.asarray(references[symbol])
                nearest = np.argmin(np.linalg.norm(centers - position, axis=1))
                displacements.append(position - centers[nearest])
            delta = np.asarray(displacements)
            self.results = {"energy": 0.5 * float(np.sum(delta**2)), "forces": -delta}

    site = _site([10], 0)
    lateral = AdsorbateSiteLateral(0, ego_graph=graph.subgraph([0, 10]).copy())
    arguments = dict(
        calculator=HarmonicWells(), frozen_indices=[0, 1, 2],
        free_energy_options=FreeEnergyOptions(enabled=True),
        free_energy_temperature_k=500.0,
    )
    if not valid_spectator:
        with pytest.raises(adsorption.AdsorbateDissociationError, match="intended surface bond"):
            adsorption.check_site_stability(graph, site, 0, lateral, **arguments)
        return
    adsorption.check_site_stability(graph, site, 0, lateral, **arguments)
    assert lateral.stable is True
    assert len(lateral.atoms_occupied) == 6
    assert len(lateral.atoms_unoccupied) == 5
    assert lateral.vib_indices_occupied == [3, 4, 5]
    assert lateral.vib_indices_unoccupied == [3, 4]
    assert len(lateral.frequencies_occupied_ev) == 9
    assert len(lateral.frequencies_unoccupied_ev) == 6


@pytest.mark.parametrize("family", ["adsorption", "diffusion", "bond", "gas_bond"])
@pytest.mark.parametrize("match", ["electronic", "exact"])
def test_cached_spectator_is_validated_before_free_energy_reuse(family, match, monkeypatch, tmp_path):
    graph = _graph()
    graph.nodes[30]["position"] = [10.0, 0.0, 1.6]
    graph.nodes[31]["position"] = [10.0, 0.0, 2.75]
    site_a, site_b, site_c = _site([10], 0), _site([11], 1), _site([20, 21], 0)
    if family == "adsorption":
        module, parent = adsorption, site_a
        lateral = AdsorbateSiteLateral(0)
        check = module.check_site_stability
        names = ["occupied", "unoccupied"]
        initial_name = "occupied_initial"
        thermo_name = "_apply_adsorption_thermochemistry"
    elif family == "diffusion":
        module = diffusion
        parent = SimpleNamespace(
            iso_class=0, reactant="[H]", member_node_ids=[([10], [11])],
            members=[(site_a, 0, site_b, 0)], lateral_classes=[],
        )
        lateral = DiffusionLateral(0)
        check = module.check_diffusion_stability
        names = ["state_a", "state_b", "transition"]
        initial_name = "state_a_initial"
        thermo_name = "_apply_diffusion_thermochemistry"
    else:
        module = bond
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11], [20, 21])],
            members=[(site_a, 0, site_b, 0, site_c, 0)], lateral_classes=[],
            template=SimpleNamespace(is_symmetric=True),
        )
        lateral = BondReactionLateral(0)
        check = module.check_bond_site_stability
        names = ["state_ab", "state_c", "transition"]
        initial_name = "state_ab_initial"
        thermo_name = "_apply_bond_thermochemistry"
        if family == "gas_bond":
            from autokmc.species.reactant import build_reactant
            parent.gas_product = True
            parent.gas_reactant = build_reactant("[H][H]", add_hydrogens=False)
            parent.gas_reactant.energy = 0.0
            parent.gas_reactant.gibbs_energy = 0.0
            parent.template.smiles_c = "[H][H]"
            parent.member_node_ids = [([10], [11], [])]
            parent.members = [(site_a, 0, site_b, 0, None, -1)]
    parent.lateral_classes = [lateral]
    calls = []

    def load(*args, **kwargs):
        atoms = kwargs["inputs"][initial_name]
        endpoints = [atoms.copy() for _ in names]
        if family == "adsorption":
            endpoints[1] = endpoints[1][:-1]
        # Old endpoint passed a local reacting-molecule check even though its
        # unchanged remote molecule was already detached from its intended site.
        if family != "gas_bond":
            endpoints[1].positions[3:5, 2] += 8.0
        record = {
            "_cache_match": match,
            "states": {
                name: {"atoms": endpoint, "energy_ev": 0.0, "properties": {"g_stale": 9.0}}
                for name, endpoint in zip(names, endpoints)
            },
        }
        if family == "gas_bond":
            remaining_surface = atoms[:-2]
            remaining_surface.positions[3:5, 2] += 8.0
            record["states"].update({
                "state_c_gas_reference": {"atoms": remaining_surface, "energy_ev": 0.0},
                "gas_molecule": {"atoms": parent.gas_reactant.atoms, "energy_ev": 0.0},
            })
        return record

    hydrate = module.apply_cached_states
    validate = module._check_intended_coordination_stable

    def apply(*args, **kwargs):
        calls.append("hydrate")
        return hydrate(*args, **kwargs)

    def guard(*args, **kwargs):
        calls.append("validate")
        return validate(*args, **kwargs)

    def fresh(*args, **kwargs):
        calls.append("fresh")
        assert lateral.stable is None
        assert getattr(lateral, "g_stale", None) is None
        raise _BuiltState

    def unwanted_thermo(*args, **kwargs):
        raise AssertionError("invalid spectator endpoint reached thermochemistry")

    monkeypatch.setattr(module, "load_calculation_record", load)
    monkeypatch.setattr(module, "apply_cached_states", apply)
    monkeypatch.setattr(module, "_check_intended_coordination_stable", guard)
    monkeypatch.setattr(module, thermo_name, unwanted_thermo)
    if family == "adsorption":
        import autokmc.structure
        monkeypatch.setattr(autokmc.structure, "optimise_structure", fresh)
    else:
        monkeypatch.setattr(module, "_relax_endpoint" if family == "diffusion" else "_relax_bond_endpoint", fresh)
    with pytest.raises(_BuiltState):
        check(
            graph, parent, 0, lateral, calculator=None,
            free_energy_options=FreeEnergyOptions(enabled=True),
            free_energy_temperature_k=500.0,
            calculation_cache_root=str(tmp_path), calculation_cache_lookup_enabled=True,
        )
    assert calls == ["hydrate", "validate", "validate", "fresh"]
