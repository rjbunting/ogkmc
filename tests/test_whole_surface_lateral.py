"""Whole-surface vibrations require complete state identity and rate refresh."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.kmc import recompute as recompute_module
from autokmc.kmc.index import _ReactionIndex
from autokmc.reactions.adsorption import get_applicable_reaction_for_member
from autokmc.reactions.rates import KB_EV
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionSite, BondReactionTemplate
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral
from autokmc.sites.stability.bond import check_bond_site_lateral
from autokmc.sites.stability.diffusion import check_diffusion_site_lateral
from autokmc.thermo.free_energy import FreeEnergyOptions


def _system():
    graph = nx.path_graph(8)
    for n in range(8):
        graph.nodes[n].update(type="surface", element="Pd")
    for n in (0, 1):
        graph.add_node(
            10 + n, type="adsorbate", element="H", reactant="[H]",
            iso_class=0, occupied=n == 0, is_bonded=True,
            clique=frozenset({n}), siblings=(), reactant_index=0,
            reactant_orbit=0,
        )
        graph.add_edge(n, 10 + n)
    hydrogen = AdsorbateSite(
        reactant="[H]", n_atoms=1, atom_cliques=[frozenset({0})],
        positions=np.zeros((1, 3)), iso_class=0,
        members=[[frozenset({0})], [frozenset({1})]],
        member_node_ids=[[10], [11]],
    )
    hydrogen._member_cliques = [(frozenset({0}),), (frozenset({1}),)]
    for atom in (12, 13):
        graph.add_node(
            atom, type="adsorbate", element="H", reactant="[H][H]",
            iso_class=1, occupied=False, is_bonded=True,
            clique=frozenset({2}), siblings=(25 - atom,),
            reactant_index=atom - 12, reactant_orbit=0,
        )
        graph.add_edge(2, atom)
    graph.add_edge(12, 13)
    product = AdsorbateSite(
        reactant="[H][H]", n_atoms=2, atom_cliques=[frozenset({2})] * 2,
        positions=np.zeros((2, 3)), iso_class=1,
        members=[[frozenset({2})] * 2], member_node_ids=[[12, 13]],
    )
    product._member_cliques = [(frozenset({2}),)]
    diffusion = DiffusionSite(
        reactant="[H]", iso_class=0,
        members=[(hydrogen, 0, hydrogen, 1)],
        member_node_ids=[([10], [11])],
    )
    bond = BondReactionSite(
        template=BondReactionTemplate("[H]", "[H]", "[H][H]"), iso_class=0,
        members=[(hydrogen, 0, hydrogen, 1, product, 0)],
        member_node_ids=[([10], [11], [12, 13])],
    )
    bond._member_cliques = [
        ((frozenset({0}),), (frozenset({1}),), (frozenset({2}),)),
    ]
    # A distant molecule includes a hydrogen with no direct surface edge.
    for atom, element in ((90, "N"), (91, "H")):
        graph.add_node(
            atom, type="adsorbate", element=element, reactant="[NH]",
            iso_class=2, occupied=False, is_bonded=atom == 90,
            clique=frozenset({7}) if atom == 90 else None,
            siblings=(181 - atom,), reactant_index=atom - 90,
            reactant_orbit=atom - 90,
        )
    graph.add_edge(7, 90)
    graph.add_edge(90, 91)
    return graph, hydrogen, diffusion, bond


@pytest.mark.parametrize("kind", ["adsorption", "diffusion", "bond"])
@pytest.mark.parametrize("ignore_lateral", [False, True])
def test_full_scope_classifies_distant_occupied_molecules_and_all_their_atoms(
    kind, ignore_lateral,
):
    graph, adsorption, diffusion, bond = _system()
    site, classify = {
        "adsorption": (adsorption, check_adsorbate_site_lateral),
        "diffusion": (diffusion, check_diffusion_site_lateral),
        "bond": (bond, check_bond_site_lateral),
    }[kind]
    original = classify(
        graph, site, 0, n_shells=0, ignore_lateral=ignore_lateral,
        include_all_occupied=True,
    )
    assert original.ego_graph.graph["environment_scope"] == "all_occupied"
    assert set(range(8)).issubset(original.ego_graph)
    assert 90 not in original.ego_graph
    assert 91 not in original.ego_graph
    graph.nodes[90]["occupied"] = graph.nodes[91]["occupied"] = True
    changed = classify(
        graph, site, 0, n_shells=0, ignore_lateral=ignore_lateral,
        include_all_occupied=True,
    )
    assert changed is not original
    assert {90, 91}.issubset(changed.ego_graph)
    assert changed.ego_graph.has_edge(90, 91)
    # Hypothetical unoccupied product placements remain excluded, except
    # when they are an endpoint of this bond reaction.
    assert (12 in changed.ego_graph) is (kind == "bond")
    assert changed.ego_graph.nodes[10]["endpoint_role"] in {"site", "a", "ab"}
    graph.nodes[90]["occupied"] = graph.nodes[91]["occupied"] = False
    restored = classify(
        graph, site, 0, n_shells=0, ignore_lateral=ignore_lateral,
        include_all_occupied=True,
    )
    assert restored is original


@pytest.mark.parametrize("kind", ["adsorption", "diffusion", "bond"])
def test_full_scope_cannot_reuse_local_thermochemistry_on_an_empty_surface(kind):
    graph, adsorption, diffusion, bond = _system()
    graph.nodes[10]["occupied"] = False
    site, classify = {
        "adsorption": (adsorption, check_adsorbate_site_lateral),
        "diffusion": (diffusion, check_diffusion_site_lateral),
        "bond": (bond, check_bond_site_lateral),
    }[kind]
    local = classify(graph, site, 0, n_shells=20)
    local.stable = True
    global_class = classify(graph, site, 0, n_shells=20, include_all_occupied=True)
    assert set(local.ego_graph) == set(global_class.ego_graph)
    assert global_class is not local
    assert global_class.stable is None
    # Exact matching must retain the scope guard even when a stale index
    # incorrectly points a full-scope fingerprint at an old local class.
    site._lateral_fp_index = {global_class._fingerprint: [local]}
    reclassified = classify(graph, site, 0, n_shells=20, include_all_occupied=True)
    assert reclassified is not local
    assert reclassified.stable is None


def _settings():
    return {
        "temperature": 500.0,
        "transmission_coefficient": 1.0,
        "frozen_indices": None,
        "fmax": 0.05,
        "max_steps": 20,
        "verbose": False,
        "max_n_shells": 0,
        "lateral_shells": 0,
        "lateral_interactions": False,
        "free_energy_options": FreeEnergyOptions(enabled=True),
    }


@pytest.mark.parametrize("affected_cliques", [set(), {frozenset({7})}])
def test_distant_occupancy_recomputes_global_free_energy_and_live_rate(
    monkeypatch, affected_cliques,
):
    graph, site, _, _ = _system()
    evaluated = []

    def exact_endpoints(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append((member, 90 in lateral.ego_graph))
        lateral.energy_occupied = -1.0
        lateral.energy_unoccupied = 0.0
        # A distant occupied molecule changes the combined vibrational free
        # energy difference by 0.2 eV in this controlled Hamiltonian.
        lateral.g_occupied = -1.0 + (0.2 if 90 in lateral.ego_graph else 0.0)
        lateral.g_unoccupied = 0.0
        lateral.stable = True

    monkeypatch.setattr(
        "autokmc.reactions.adsorption.check_site_stability", exact_endpoints,
    )
    options = FreeEnergyOptions(enabled=True)
    initial = get_applicable_reaction_for_member(
        graph, site, 0, None, {"[H]": 0.0}, temperature=500.0,
        gas_g={"[H]": 0.0}, free_energy_options=options,
        lateral_shells=0, lateral_interactions=False,
    )
    index = _ReactionIndex([site])
    index.install(initial, site, 0)
    graph.nodes[90]["occupied"] = graph.nodes[91]["occupied"] = True
    updated, _ = recompute_module.recompute_affected_sites(
        graph, [site], affected_cliques, None, {"[H]": 0.0},
        gas_g={"[H]": 0.0}, rxn_index=index, **_settings(),
    )
    current = next(reaction for reaction in updated if reaction.member_index == 0)
    assert current.lateral_class is not initial.lateral_class
    assert initial.delta_e == pytest.approx(1.0)
    assert current.delta_e == pytest.approx(0.8)
    assert current.rate / initial.rate == pytest.approx(np.exp(0.2 / (KB_EV * 500.0)))
    assert current in index.reactions
    assert (0, True) in evaluated


def test_whole_surface_refreshes_all_active_channels_without_local_indexes(monkeypatch):
    graph, adsorption, diffusion, bond = _system()
    index = _ReactionIndex([adsorption], [diffusion], [bond])
    calls = []

    def evaluate(channel):
        def run(graph, site, member, *args, **kwargs):
            calls.append((channel, member))
            return SimpleNamespace(site=site, member_index=member, rate=1.0)
        return run

    for channel, getter in (
        ("adsorption", "get_applicable_reaction_for_member"),
        ("diffusion", "get_applicable_diffusion_for_member"),
        ("bond", "get_applicable_bond_reaction_for_member"),
    ):
        monkeypatch.setattr(recompute_module, getter, evaluate(channel))

    def local_lookup(*args, **kwargs):
        raise AssertionError("global free energies cannot use a local dependency lookup")

    for helper in ("_lateral_shell_members", "_diffusion_lateral_shell_members", "_bond_lateral_shell_members"):
        monkeypatch.setattr(recompute_module, helper, local_lookup)
    # An inactive duplicate is omitted by the active index, even in global mode.
    inactive = SimpleNamespace(
        member_node_ids=[[99]], lateral_classes=[], applicable_reactions=[],
    )
    updated, changed_diffusions = recompute_module.recompute_affected_sites(
        graph, [adsorption, inactive], {frozenset({7})}, None, {"[H]": 0.0},
        diffusion_sites=[diffusion], bond_sites=[bond], rxn_index=index,
        **_settings(),
    )
    assert calls == [("adsorption", 0), ("adsorption", 1), ("diffusion", 0), ("bond", 0)]
    assert len(updated) == 4
    assert changed_diffusions == [diffusion]
    assert all(reaction is not None for reaction in index.reactions)
