"""Reaction-role classification must preserve the energies used by real rates.

Only expensive potential calculations are replaced by exact lattice energies;
classification, applicability, caching, and forward/reverse rates are real.
"""

import networkx as nx
import numpy as np
import pytest

from autokmc.io.reaction_graph import (
    normalise_reaction_graph,
    reaction_graph_from_payload,
    reaction_graph_payload,
    reaction_graphs_isomorphic,
)
from autokmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from autokmc.io.config import CheckpointCfg, FreeEnergyCfg, OutputCfg, RunConfig
from autokmc.reactions.adsorption import get_applicable_reaction_for_member
from autokmc.reactions.diffusion import get_applicable_diffusion_for_member
from autokmc.reactions.rates import EA_MIN, H_EV_S, KB_EV
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral
from autokmc.sites.stability.diffusion import check_diffusion_site_lateral
from autokmc.workflow.runtime import resolve_run_identity


TEMPERATURE = 500.0
KT = KB_EV * TEMPERATURE
PREFACTOR = KT / H_EV_S


def _cycle_with_adsorption_sites(size, element, occupied):
    graph = nx.cycle_graph(size)
    smiles = f"[{element}]"
    for node in range(size):
        graph.nodes[node].update(type="surface", element="Pd")
        graph.add_node(
            10 + node,
            type="adsorbate",
            element=element,
            reactant=smiles,
            iso_class=0,
            occupied=node in occupied,
            is_bonded=True,
            clique=frozenset([node]),
            siblings=(),
            reactant_index=0,
            reactant_orbit=0,
        )
        graph.add_edge(node, 10 + node)
    site = AdsorbateSite(
        reactant=smiles,
        n_atoms=1,
        atom_cliques=[frozenset([0])],
        positions=np.zeros((1, 3)),
        iso_class=0,
        members=[[frozenset([node])] for node in range(size)],
        member_node_ids=[[10 + node] for node in range(size)],
    )
    site._member_cliques = [(frozenset([node]),) for node in range(size)]
    return graph, site


def _diffusion_system(nitrogen_sites=(6, 7)):
    graph, site = _cycle_with_adsorption_sites(8, "C", {0, 4})
    for node in nitrogen_sites:
        graph.add_node(
            20 + node,
            type="adsorbate",
            element="N",
            reactant="[N]",
            iso_class=0,
            occupied=True,
            is_bonded=True,
            clique=frozenset([node]),
            siblings=(),
            reactant_index=0,
            reactant_orbit=0,
        )
        graph.add_edge(node, 20 + node)
    diffusion = DiffusionSite(
        reactant="[C]",
        iso_class=0,
        members=[(site, 0, site, 1), (site, 4, site, 5)],
        member_node_ids=[([10], [11]), ([14], [15])],
    )
    return graph, diffusion


@pytest.mark.parametrize("use_free_energy", [False, True])
@pytest.mark.parametrize("nitrogen_sites", [(6, 7), (3, 7)])
def test_diffusion_rates_preserve_endpoint_orientation(
    monkeypatch, use_free_energy, nitrogen_sites,
):
    graph, diffusion = _diffusion_system(nitrogen_sites)
    evaluated = []
    attraction = 0.3 if use_free_energy else 0.2

    def endpoint_energy(surface_node, strength):
        return -strength * sum(
            graph.has_edge(surface_node, nitrogen) for nitrogen in nitrogen_sites
        )

    def exact_neb(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append(member)
        _, a, _, b = site.members[member]
        lateral.energy_a = endpoint_energy(a, 0.2)
        lateral.energy_b = endpoint_energy(b, 0.2)
        lateral.energy_ts = 0.5
        if use_free_energy:
            lateral.g_a = endpoint_energy(a, attraction)
            lateral.g_b = endpoint_energy(b, attraction)
            lateral.g_ts = 0.5
        lateral.stable = True
        return lateral.energy_a, lateral.energy_b, lateral.energy_ts

    monkeypatch.setattr(
        "autokmc.reactions.diffusion.check_diffusion_stability", exact_neb,
    )
    forward = [
        get_applicable_diffusion_for_member(
            graph, diffusion, member, None,
            temperature=TEMPERATURE, lateral_shells=1,
        )
        for member in (0, 1)
    ]
    equivalent = nitrogen_sites == (3, 7)
    assert evaluated == ([0] if equivalent else [0, 1])
    assert (forward[0].lateral_class is forward[1].lateral_class) is equivalent

    # Each class preserves member A/B ordering, and each member still supports
    # a reverse event with the same TS and the correct reverse barrier.
    for member, reaction in enumerate(forward):
        _, a, _, b = diffusion.members[member]
        expected_delta = endpoint_energy(b, attraction) - endpoint_energy(a, attraction)
        expected_barrier = 0.5 - endpoint_energy(a, attraction)
        assert reaction.delta_e == pytest.approx(expected_delta)
        assert reaction.barrier == pytest.approx(expected_barrier)
        assert reaction.rate == pytest.approx(
            PREFACTOR * np.exp(-expected_barrier / KT),
        )
        graph.nodes[10 + a]["occupied"] = False
        graph.nodes[10 + b]["occupied"] = True
        reverse = get_applicable_diffusion_for_member(
            graph, diffusion, member, None,
            temperature=TEMPERATURE, lateral_shells=1,
        )
        assert reverse.lateral_class is reaction.lateral_class
        assert reverse.direction == "b_to_a"
        assert reverse.delta_e == pytest.approx(-expected_delta)
        assert reverse.barrier == pytest.approx(0.5 - endpoint_energy(b, attraction))
        assert reaction.rate / reverse.rate == pytest.approx(np.exp(-expected_delta / KT))
    assert evaluated == ([0] if equivalent else [0, 1])


@pytest.mark.parametrize("use_free_energy", [False, True])
def test_desorption_rates_distinguish_target_from_spectators(
    monkeypatch, use_free_energy,
):
    graph, site = _cycle_with_adsorption_sites(6, "H", {0, 1, 2})
    evaluated = []
    attraction = 0.3 if use_free_energy else 0.2

    def energy(state, strength):
        pairs = sum(left in state and right in state for left, right in graph.edges())
        return -len(state) - strength * pairs

    def exact_endpoints(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append(member)
        state = {
            surface for surface in range(6)
            if graph.nodes[10 + surface]["occupied"]
        }
        lateral.energy_occupied = energy(state | {member}, 0.2)
        lateral.energy_unoccupied = energy(state - {member}, 0.2)
        if use_free_energy:
            lateral.g_occupied = energy(state | {member}, attraction)
            lateral.g_unoccupied = energy(state - {member}, attraction)
        lateral.stable = True
        return lateral.energy_occupied, lateral.energy_unoccupied

    monkeypatch.setattr(
        "autokmc.reactions.adsorption.check_site_stability", exact_endpoints,
    )
    kwargs = {"gas_g": {"[H]": 0.0}} if use_free_energy else {}
    reactions = [
        get_applicable_reaction_for_member(
            graph, site, member, None, {"[H]": 0.0},
            temperature=TEMPERATURE, lateral_shells=3, **kwargs,
        )
        for member in (0, 1, 2)
    ]
    # End molecules 0 and 2 are symmetry equivalent, but middle molecule 1
    # has two H neighbours and requires a separate removal calculation.
    assert evaluated == [0, 1]
    assert reactions[0].lateral_class is reactions[2].lateral_class
    assert reactions[0].lateral_class is not reactions[1].lateral_class
    for member, reaction in enumerate(reactions):
        expected = energy({0, 1, 2} - {member}, attraction) - energy({0, 1, 2}, attraction)
        assert reaction.kind == "desorption"
        assert reaction.delta_e == pytest.approx(expected)
        assert reaction.barrier == pytest.approx(expected + EA_MIN)
        assert reaction.rate == pytest.approx(PREFACTOR * np.exp(-(expected + EA_MIN) / KT))

    # Removing the target leaves exactly the same lateral class for adsorption
    # back into that site: target occupancy is not part of its role identity.
    graph.nodes[11]["occupied"] = False
    reverse = get_applicable_reaction_for_member(
        graph, site, 1, None, {"[H]": 0.0},
        temperature=TEMPERATURE, lateral_shells=3, **kwargs,
    )
    assert reverse.kind == "adsorption"
    assert reverse.lateral_class is reactions[1].lateral_class
    assert reverse.delta_e == pytest.approx(-reactions[1].delta_e)
    assert reverse.rate / reactions[1].rate == pytest.approx(
        np.exp(reactions[1].delta_e / KT),
    )
    assert evaluated == [0, 1]


@pytest.mark.parametrize("kind", ["adsorption", "diffusion"])
@pytest.mark.parametrize("keep_fingerprint_index", [False, True])
def test_legacy_checkpoint_classes_without_reaction_roles_are_not_reused(
    tmp_path, monkeypatch, kind, keep_fingerprint_index,
):
    if kind == "adsorption":
        graph, site = _cycle_with_adsorption_sites(6, "H", {0, 1, 2})
        classify = check_adsorbate_site_lateral
    else:
        graph, site = _diffusion_system()
        classify = check_diffusion_site_lateral
    legacy = classify(graph, site, 0, n_shells=3)
    legacy.stable = True
    for _, data in legacy.ego_graph.nodes(data=True):
        if data.get("endpoint_role"):
            if kind == "adsorption":
                data.pop("endpoint_role")
            else:
                data["endpoint_role"] = "endpoint"
    # Retaining even a stale index that points to this candidate must not
    # bypass the exact role-aware graph comparison after checkpoint reload.
    if not keep_fingerprint_index:
        del site._lateral_fp_index
    state = make_checkpoint_state(
        step=0,
        time_s=0.0,
        graph=graph,
        adsorbate_sites=[site] if kind == "adsorption" else [],
        diffusion_sites=[site] if kind == "diffusion" else [],
    )
    checkpoint = save_checkpoint(tmp_path / "legacy.pkl", state)
    identity = resolve_run_identity(RunConfig(
        output=OutputCfg(dir=str(tmp_path / "resumed")),
        checkpoint=CheckpointCfg(resume_from=str(checkpoint)),
        free_energy=FreeEnergyCfg(enabled=False),
    ))
    restored = identity.resume_state
    graph = restored.graph
    site = (
        restored.adsorbate_sites[0] if kind == "adsorption"
        else restored.diffusion_sites[0]
    )
    old = site.lateral_classes[0]
    current = classify(graph, site, 0, n_shells=3)
    assert current is not old
    assert current.stable is None
    assert current.members == [0]
    assert old.members == []

    evaluated = []

    def checked_energies(graph, site, member, lateral, calculator, **kwargs):
        evaluated.append(lateral)
        lateral.energy_occupied = -1.0
        lateral.energy_unoccupied = 0.0
        lateral.energy_a = 0.0
        lateral.energy_b = 0.2
        lateral.energy_ts = 0.5
        lateral.stable = True

    if kind == "adsorption":
        monkeypatch.setattr(
            "autokmc.reactions.adsorption.check_site_stability", checked_energies,
        )
        reaction = get_applicable_reaction_for_member(
            graph, site, 0, None, {"[H]": 0.0},
            temperature=TEMPERATURE, lateral_shells=3,
        )
        assert reaction.delta_e == pytest.approx(1.0)
    else:
        monkeypatch.setattr(
            "autokmc.reactions.diffusion.check_diffusion_stability", checked_energies,
        )
        reaction = get_applicable_diffusion_for_member(
            graph, site, 0, None, temperature=TEMPERATURE, lateral_shells=3,
        )
        assert reaction.delta_e == pytest.approx(0.2)
    assert evaluated == [current]
    assert reaction.lateral_class is current


@pytest.mark.parametrize("kind", ["adsorption", "diffusion"])
def test_portable_reaction_graph_retains_target_and_endpoint_roles(kind):
    if kind == "adsorption":
        graph, site = _cycle_with_adsorption_sites(6, "H", {0, 1, 2})
        classify = check_adsorbate_site_lateral
    else:
        graph, site = _diffusion_system()
        classify = check_diffusion_site_lateral
    classes = [classify(graph, site, member, n_shells=3) for member in (0, 1)]
    portable = [
        reaction_graph_from_payload(reaction_graph_payload(lateral.ego_graph))
        for lateral in classes
    ]
    assert not reaction_graphs_isomorphic(*portable)
    assert reaction_graphs_isomorphic(portable[0], classes[0].ego_graph)
    if kind == "diffusion":
        legacy = normalise_reaction_graph(classes[0].ego_graph)
        for _, data in legacy.nodes(data=True):
            if data.get("endpoint_role"):
                data["endpoint_role"] = "endpoint"
        assert not reaction_graphs_isomorphic(portable[0], legacy)
