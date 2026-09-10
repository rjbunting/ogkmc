"""Free energies use exactly the molecules in the selected local environment."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.sites.stability import adsorption, bond, diffusion
from autokmc.reactions import adsorption as adsorption_reactions
from autokmc.reactions import bond as bond_reactions
from autokmc.reactions import diffusion as diffusion_reactions
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
@pytest.mark.parametrize("lateral_interactions", [False, True])
@pytest.mark.parametrize("lateral_shells", [0, 1, 2])
@pytest.mark.parametrize("thermo_enabled", [False, True])
def test_reaction_lateral_range_controls_actual_atoms(
    family, lateral_interactions, lateral_shells, thermo_enabled, monkeypatch,
):
    graph = _graph()
    site_a, site_b, site_c = _site([10], 0), _site([11], 1), _site([20, 21], 0)
    # CO is two surface hops from adsorption, one from the diffusion/bond
    # endpoint union. A second occupied molecule is outside every tested range.
    for node in (3, 4):
        graph.add_node(node, type="surface", element="Cu", index=node,
                       position=[5.0 * node, 0.0, 0.0])
        graph.add_edge(node - 1, node)
    graph.remove_edge(2, 40)
    graph.add_edge(4, 40)
    graph.nodes[40].update(occupied=True, clique=frozenset({4}), position=[20, 0, 1.6])
    args = ()
    calculator = None
    if family == "adsorption":
        module, parent = adsorption, site_a
        getter = adsorption_reactions.get_applicable_reaction_for_member
        builder_name, n_reacting = "_build_stability_atoms", 1
        args = ({"[H]": 0.0},)
    elif family == "diffusion":
        module = diffusion
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11])],
            members=[(site_a, 0, site_b, 0)], lateral_classes=[],
        )
        getter = diffusion_reactions.get_applicable_diffusion_for_member
        builder_name, n_reacting = "_build_diffusion_atoms", 1
    else:
        module = bond
        graph.nodes[11]["occupied"] = True
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11], [20, 21])],
            members=[(site_a, 0, site_b, 0, site_c, 0)], lateral_classes=[],
            template=SimpleNamespace(is_symmetric=True),
            _member_cliques=[(site_a._member_cliques[0], site_b._member_cliques[0],
                              site_c._member_cliques[0])],
        )
        getter = bond_reactions.get_applicable_bond_reaction_for_member
        builder_name, n_reacting = "_build_bond_atoms", 2
        calculator = object()
        # Inspect the physical reaction, bypassing the optional bare NEB seed.
        monkeypatch.setattr(bond_reactions, "get_bond_bare_lateral", lambda *a, **k: None)
    build = getattr(module, builder_name)
    captured = []

    def inspect(*args, **kwargs):
        captured.append(build(*args, **kwargs))
        raise _BuiltState

    monkeypatch.setattr(module, builder_name, inspect)
    with pytest.raises(_BuiltState):
        getter(
            graph, parent, 0, calculator, *args, temperature=500.0,
            lateral_interactions=lateral_interactions, lateral_shells=lateral_shells,
            free_energy_options=FreeEnergyOptions(enabled=thermo_enabled),
        )
    atoms, n_slab, n_lat = captured[0][:3]
    includes_co = lateral_interactions and lateral_shells >= (2 if family == "adsorption" else 1)
    assert n_slab == 5
    assert n_lat == (2 if includes_co else 0)
    assert len(atoms) == n_slab + n_lat + n_reacting
    assert ("C" in atoms.get_chemical_symbols()) is includes_co
    assert "N" not in atoms.get_chemical_symbols()


@pytest.mark.parametrize("family", ["adsorption", "diffusion", "bond"])
@pytest.mark.parametrize("n_shells", [0, 2])
def test_direct_stability_keeps_selected_local_environment(family, n_shells, monkeypatch):
    graph = _graph()
    site_a, site_b, site_c = _site([10], 0), _site([11], 1), _site([20, 21], 0)
    if family == "adsorption":
        module, parent = adsorption, site_a
        check = module.check_site_stability
        classify = module.check_adsorbate_site_lateral
        builder_name, expected_reacting = "_build_stability_atoms", 1
    elif family == "diffusion":
        module = diffusion
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11])],
            members=[(site_a, 0, site_b, 0)], lateral_classes=[],
        )
        check = module.check_diffusion_stability
        classify = module.check_diffusion_site_lateral
        builder_name, expected_reacting = "_build_diffusion_atoms", 1
    else:
        module = bond
        parent = SimpleNamespace(
            iso_class=0, member_node_ids=[([10], [11], [20, 21])],
            members=[(site_a, 0, site_b, 0, site_c, 0)], lateral_classes=[],
            template=SimpleNamespace(is_symmetric=True),
        )
        check = module.check_bond_site_stability
        classify = module.check_bond_site_lateral
        builder_name, expected_reacting = "_build_bond_atoms", 2
    lateral = classify(graph, parent, 0, n_shells=n_shells)
    original_graph = lateral.ego_graph
    build = getattr(module, builder_name)
    captured = []

    def inspect(*args, **kwargs):
        captured.append(build(*args, **kwargs))
        raise _BuiltState

    monkeypatch.setattr(module, builder_name, inspect)
    with pytest.raises(_BuiltState):
        check(
            graph, parent, 0, lateral, calculator=None,
            free_energy_options=FreeEnergyOptions(enabled=True),
            free_energy_temperature_k=500.0,
        )
    atoms, n_slab, n_lat = captured[0][:3]
    assert (n_slab, n_lat) == (3, 2 if n_shells == 2 else 0)
    assert len(atoms) == n_slab + n_lat + expected_reacting
    assert lateral.ego_graph is original_graph
    assert lateral.ego_graph.graph["environment_scope"] == "local"
    assert classify(graph, parent, 0, n_shells=n_shells) is lateral


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
@pytest.mark.parametrize("lateral_shells", [0, 2])
def test_direct_adsorption_computes_stationary_surface_modes(
    valid_spectator, lateral_shells,
):
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
    lateral = adsorption.check_adsorbate_site_lateral(graph, site, 0, n_shells=lateral_shells)
    arguments = dict(
        calculator=HarmonicWells(), frozen_indices=[0, 1, 2],
        free_energy_options=FreeEnergyOptions(enabled=True),
        free_energy_temperature_k=500.0,
    )
    if not valid_spectator and lateral_shells == 2:
        with pytest.raises(adsorption.AdsorbateDissociationError, match="intended surface bond"):
            adsorption.check_site_stability(graph, site, 0, lateral, **arguments)
        return
    adsorption.check_site_stability(graph, site, 0, lateral, **arguments)
    assert lateral.stable is True
    assert len(lateral.atoms_occupied) == (6 if lateral_shells == 2 else 4)
    assert len(lateral.atoms_unoccupied) == (5 if lateral_shells == 2 else 3)
    assert lateral.vib_indices_occupied == ([3, 4, 5] if lateral_shells == 2 else [3])
    assert lateral.vib_indices_unoccupied == ([3, 4] if lateral_shells == 2 else [])
    assert len(lateral.frequencies_occupied_ev) == (9 if lateral_shells == 2 else 3)
    assert len(lateral.frequencies_unoccupied_ev) == (6 if lateral_shells == 2 else 0)
    assert lateral.g_occupied != pytest.approx(lateral.energy_occupied)


@pytest.mark.parametrize("family", ["adsorption", "diffusion", "bond", "gas_bond"])
@pytest.mark.parametrize("match", ["electronic", "exact"])
def test_cached_spectator_is_validated_before_free_energy_reuse(family, match, monkeypatch, tmp_path):
    graph = _graph()
    graph.nodes[30]["position"] = [10.0, 0.0, 1.6]
    graph.nodes[31]["position"] = [10.0, 0.0, 2.75]
    site_a, site_b, site_c = _site([10], 0), _site([11], 1), _site([20, 21], 0)
    if family == "adsorption":
        module, parent = adsorption, site_a
        classify = module.check_adsorbate_site_lateral
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
        classify = module.check_diffusion_site_lateral
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
        classify = module.check_bond_site_lateral
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
    lateral = classify(graph, parent, 0, n_shells=2)
    calls = []

    def load(*args, **kwargs):
        atoms = kwargs["inputs"][initial_name]
        endpoints = [atoms.copy() for _ in names]
        if family == "adsorption":
            endpoints[1] = endpoints[1][:-1]
        # Old endpoint passed a local reacting-molecule check even though its
        # unchanged selected neighbor was already detached from its intended site.
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
