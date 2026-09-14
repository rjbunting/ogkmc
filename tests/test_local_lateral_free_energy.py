"""Local free energies follow lateral shells, molecule identity, and rate dependencies."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.kmc import recompute as recompute_module
from autokmc.kmc.index import _ReactionIndex
from autokmc.reactions.adsorption import get_applicable_reaction_for_member
from autokmc.reactions.bond import get_applicable_bond_reaction_for_member
from autokmc.reactions.diffusion import get_applicable_diffusion_for_member
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
    for atom, remote, sibling in ((80, 90, 81), (81, 91, 80)):
        graph.add_node(atom, **{**graph.nodes[remote], "siblings": (sibling,),
                               "clique": frozenset({3}) if atom == 80 else None})
    graph.add_edge(3, 80)
    graph.add_edge(80, 81)
    return graph, hydrogen, diffusion, bond


@pytest.mark.parametrize("kind", ["adsorption", "diffusion", "bond"])
@pytest.mark.parametrize("ignore_lateral", [False, True])
def test_local_scope_classifies_only_complete_selected_molecules(kind, ignore_lateral):
    graph, adsorption, diffusion, bond = _system()
    site, classify = {
        "adsorption": (adsorption, check_adsorbate_site_lateral),
        "diffusion": (diffusion, check_diffusion_site_lateral),
        "bond": (bond, check_bond_site_lateral),
    }[kind]
    kwargs = dict(n_shells=3, ignore_lateral=ignore_lateral)
    original = classify(graph, site, 0, **kwargs)
    assert original.ego_graph.graph["environment_scope"] == "local"
    graph.nodes[90]["occupied"] = graph.nodes[91]["occupied"] = True
    assert classify(graph, site, 0, **kwargs) is original
    graph.nodes[80]["occupied"] = graph.nodes[81]["occupied"] = True
    changed = classify(graph, site, 0, **kwargs)
    assert (changed is original) is ignore_lateral
    assert (80 in changed.ego_graph) is not ignore_lateral
    assert (81 in changed.ego_graph) is not ignore_lateral
    assert changed.ego_graph.has_edge(80, 81) is not ignore_lateral
    assert 90 not in changed.ego_graph and 91 not in changed.ego_graph
    assert (12 in changed.ego_graph) is (kind == "bond")
    assert changed.ego_graph.nodes[10]["endpoint_role"] in {"site", "a", "ab"}
    graph.nodes[80]["occupied"] = graph.nodes[81]["occupied"] = False
    assert classify(graph, site, 0, **kwargs) is original


@pytest.mark.parametrize("kind", ["adsorption", "diffusion", "bond"])
def test_local_scope_does_not_reuse_legacy_global_class_with_stale_index(kind):
    graph, adsorption, diffusion, bond = _system()
    site, classify = {
        "adsorption": (adsorption, check_adsorbate_site_lateral),
        "diffusion": (diffusion, check_diffusion_site_lateral),
        "bond": (bond, check_bond_site_lateral),
    }[kind]
    legacy = classify(graph, site, 0, n_shells=20)
    legacy.stable = True
    # Simulate a saved global class even when its atoms happen to match the
    # current local environment. The stale local fingerprint must not admit it.
    legacy.ego_graph.graph["environment_scope"] = "all_occupied"
    local = classify(graph, site, 0, n_shells=20)
    assert local is not legacy
    assert local.stable is None
    assert local.ego_graph.graph["environment_scope"] == "local"


def _settings():
    return {
        "temperature": 500.0,
        "transmission_coefficient": 1.0,
        "frozen_indices": None,
        "fmax": 0.05,
        "max_steps": 20,
        "verbose": False,
        "max_n_shells": 3,
        "lateral_shells": 3,
        "lateral_interactions": True,
        "free_energy_options": FreeEnergyOptions(enabled=True),
    }


@pytest.mark.parametrize("kind", ["adsorption", "diffusion", "bond"])
def test_disabled_lateral_interactions_reuse_bare_free_energy_after_occupancy_change(
    monkeypatch, kind,
):
    graph, adsorption, diffusion, bond = _system()
    site, getter, stability, energies = {
        "adsorption": (
            adsorption, get_applicable_reaction_for_member, "check_site_stability",
            {"energy_occupied": -1.0, "energy_unoccupied": 0.0,
             "g_occupied": -0.8, "g_unoccupied": 0.0},
        ),
        "diffusion": (
            diffusion, get_applicable_diffusion_for_member, "check_diffusion_stability",
            {"energy_a": -1.0, "energy_b": -0.9, "energy_ts": -0.5,
             "g_a": -0.8, "g_b": -0.7, "g_ts": -0.3},
        ),
        "bond": (
            bond, get_applicable_bond_reaction_for_member, "check_bond_site_stability",
            {"energy_ab": -1.0, "energy_c": -0.9, "energy_ts": -0.5,
             "g_ab": -0.8, "g_c": -0.7, "g_ts": -0.3},
        ),
    }[kind]
    if kind == "bond":
        graph.nodes[11]["occupied"] = True
    evaluated = []

    def evaluate(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append(lateral)
        assert 90 not in lateral.ego_graph
        assert 91 not in lateral.ego_graph
        assert kwargs["free_energy_options"].enabled
        for key, value in energies.items():
            setattr(lateral, key, value)
        lateral.stable = True

    monkeypatch.setattr(f"autokmc.reactions.{kind}.{stability}", evaluate)
    args = ({"[H]": 0.0},) if kind == "adsorption" else ()
    kwargs = dict(
        temperature=500.0, lateral_shells=20, lateral_interactions=False,
        free_energy_options=FreeEnergyOptions(enabled=True),
    )
    if kind == "adsorption":
        kwargs["gas_g"] = {"[H]": 0.0}
    calculator = object() if kind == "bond" else None
    initial = getter(graph, site, 0, calculator, *args, **kwargs)
    assert initial is not None
    graph.nodes[90]["occupied"] = graph.nodes[91]["occupied"] = True
    current = getter(graph, site, 0, calculator, *args, **kwargs)
    assert current is not None
    assert current.lateral_class is initial.lateral_class
    assert current.rate == initial.rate
    assert len(site.lateral_classes) == 1
    assert evaluated == [initial.lateral_class]


@pytest.mark.parametrize("changed_node", [80, 90])
@pytest.mark.parametrize("lateral_interactions", [False, True])
def test_only_local_occupancy_changes_free_energy_and_live_rate(
    monkeypatch, changed_node, lateral_interactions,
):
    graph, site, _, _ = _system()
    evaluated = []

    def exact_endpoints(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append((member, 80 in lateral.ego_graph))
        assert 90 not in lateral.ego_graph and 91 not in lateral.ego_graph
        lateral.energy_occupied = -1.0
        lateral.energy_unoccupied = 0.0
        # Only a selected neighbor changes the local vibrational correction.
        lateral.g_occupied = -1.0 + (0.2 if 80 in lateral.ego_graph else 0.0)
        lateral.g_unoccupied = 0.0
        lateral.stable = True

    monkeypatch.setattr("autokmc.reactions.adsorption.check_site_stability", exact_endpoints)
    initial = get_applicable_reaction_for_member(
        graph, site, 0, None, {"[H]": 0.0}, temperature=500.0,
        gas_g={"[H]": 0.0}, free_energy_options=FreeEnergyOptions(enabled=True),
        lateral_shells=3, lateral_interactions=lateral_interactions,
    )
    index = _ReactionIndex([site])
    index.install(initial, site, 0)
    graph.nodes[changed_node]["occupied"] = graph.nodes[changed_node + 1]["occupied"] = True
    updated, _ = recompute_module.recompute_affected_sites(
        graph, [site], {graph.nodes[changed_node]["clique"]}, None, {"[H]": 0.0},
        gas_g={"[H]": 0.0}, rxn_index=index,
        **{**_settings(), "lateral_interactions": lateral_interactions,
           "max_n_shells": 3 if lateral_interactions else 0},
    )
    if not lateral_interactions or changed_node == 90:
        assert updated == []
        assert initial in index.reactions
        assert evaluated == [(0, False)]
        assert len(site.lateral_classes) == 1
        return
    current = next(reaction for reaction in updated if reaction.member_index == 0)
    assert current.lateral_class is not initial.lateral_class
    assert initial.delta_e == pytest.approx(1.0)
    assert current.delta_e == pytest.approx(0.8)
    assert current.rate / initial.rate == pytest.approx(np.exp(0.2 / (KB_EV * 500.0)))
    assert current in index.reactions
    assert (0, True) in evaluated


@pytest.mark.parametrize("affected", [3, 7, None])
def test_free_energy_refreshes_only_local_active_channels(monkeypatch, affected):
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
    inactive = SimpleNamespace(member_node_ids=[[99]], lateral_classes=[], applicable_reactions=[])
    updated, changed_diffusions = recompute_module.recompute_affected_sites(
        graph, [adsorption, inactive], set() if affected is None else {frozenset({affected})},
        None, {"[H]": 0.0}, diffusion_sites=[diffusion], bond_sites=[bond], rxn_index=index,
        **_settings(),
    )
    assert calls == ([("adsorption", 0), ("adsorption", 1), ("diffusion", 0), ("bond", 0)]
                     if affected == 3 else [])
    assert len(updated) == len(calls)
    assert changed_diffusions == ([diffusion] if affected == 3 else [])
