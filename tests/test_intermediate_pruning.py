"""Regression tests for exact NEB-intermediate pruning."""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from ogkmc.io.persistence import ReactionWriter
from ogkmc.kmc.index import _ReactionIndex
from ogkmc.reactions.diffusion import (
    DiffusionReaction,
    _reclassify_persisted_diffusion_intermediate,
)
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import (
    BondReactionSite,
    BondReactionTemplate,
)
from ogkmc.sites.diffusion import DiffusionLateral, DiffusionSite
from ogkmc.sites.stability.intermediate_pruning import (
    CompositeDirectEventDetected,
    classify_bond_intermediate,
    classify_diffusion_intermediate,
    retain_refinement_and_maybe_suppress,
)


def _surface_graph(n_surface: int) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["cell"] = np.diag([24.0, 20.0, 20.0])
    for index in range(n_surface):
        graph.add_node(
            index,
            type="surface",
            index=index,
            element="Cu",
            position=np.array([4.0 * index, 0.0, 0.0]),
        )
    return graph


def _h_site(
    graph: nx.Graph,
    *,
    surface_index: int,
    node_id: int,
    iso_class: int,
) -> AdsorbateSite:
    clique = frozenset({surface_index})
    position = np.array([4.0 * surface_index, 0.0, 1.5])
    graph.add_node(
        node_id,
        type="adsorbate",
        element="H",
        reactant_index=0,
        clique=clique,
        siblings=frozenset(),
        position=position,
    )
    graph.add_edge(surface_index, node_id)
    site = AdsorbateSite(
        reactant="[H]",
        n_atoms=1,
        atom_cliques=[clique],
        positions=np.array([position]),
        iso_class=iso_class,
        members=[[clique]],
        member_node_ids=[[node_id]],
    )
    site._member_cliques = [(clique,)]
    return site


def _state(graph: nx.Graph, *surface_indices: int) -> Atoms:
    slab = [
        graph.nodes[index]["position"]
        for index in range(len([n for n, d in graph.nodes(data=True) if d.get("type") == "surface"]))
    ]
    positions = [*slab]
    symbols = ["Cu"] * len(slab)
    for surface_index in surface_indices:
        positions.append(np.array([4.0 * surface_index, 0.0, 1.5]))
        symbols.append("H")
    return Atoms(
        symbols,
        positions=np.asarray(positions),
        cell=graph.graph["cell"],
        pbc=False,
    )


def _metadata(*, left: int = 0, right: int = 2) -> dict:
    return {
        "trigger": "converged_final_check",
        "source_stage": "ordinary",
        "left_state_image_index": left,
        "right_state_image_index": right,
        "stalled_profile_energies_ev": [0.0, 0.5, 0.1, 0.7, 0.2],
    }


def test_diffusion_intermediate_requires_unique_registered_component():
    graph = _surface_graph(3)
    site_a = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    site_i = _h_site(graph, surface_index=1, node_id=11, iso_class=1)
    site_b = _h_site(graph, surface_index=2, node_id=12, iso_class=2)
    direct = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[(site_a, 0, site_b, 0)],
        member_node_ids=[([10], [12])],
    )
    component = DiffusionSite(
        reactant="[H]",
        iso_class=1,
        members=[(site_a, 0, site_i, 0)],
        member_node_ids=[([10], [11])],
    )
    component_2 = DiffusionSite(
        reactant="[H]",
        iso_class=2,
        members=[(site_i, 0, site_b, 0)],
        member_node_ids=[([11], [12])],
    )
    graph.graph["adsorbate_sites"] = {"[H]": [site_a, site_i, site_b]}
    graph.graph["diffusion_sites"] = {"[H]": [direct, component]}

    assert classify_diffusion_intermediate(
        graph,
        direct,
        0,
        _state(graph, 0),
        _state(graph, 1),
        _metadata(),
        n_slab=3,
        n_lateral=0,
        n_reacting=1,
        nl_mult=1.25,
    ) is None

    graph.graph["diffusion_sites"]["[H]"].append(component_2)

    certificate = classify_diffusion_intermediate(
        graph,
        direct,
        0,
        _state(graph, 0),
        _state(graph, 1),
        _metadata(),
        n_slab=3,
        n_lateral=0,
        n_reacting=1,
        nl_mult=1.25,
    )

    assert certificate is not None
    assert certificate["classification"] == "composite"
    assert certificate["reason"] == "registered_intermediate_diffusion_placement"
    assert len(certificate["component_member_ids"]) == 2


def test_diffusion_intermediate_ambiguity_does_not_prune():
    graph = _surface_graph(3)
    site_a = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    site_i = _h_site(graph, surface_index=1, node_id=11, iso_class=1)
    duplicate_i = _h_site(graph, surface_index=1, node_id=21, iso_class=2)
    site_b = _h_site(graph, surface_index=2, node_id=12, iso_class=3)
    direct = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[(site_a, 0, site_b, 0)],
        member_node_ids=[([10], [12])],
    )
    component = DiffusionSite(
        reactant="[H]",
        iso_class=1,
        members=[(site_a, 0, site_i, 0)],
        member_node_ids=[([10], [11])],
    )
    component_2 = DiffusionSite(
        reactant="[H]",
        iso_class=2,
        members=[(site_i, 0, site_b, 0)],
        member_node_ids=[([11], [12])],
    )
    graph.graph["adsorbate_sites"] = {
        "[H]": [site_a, site_i, duplicate_i, site_b]
    }
    graph.graph["diffusion_sites"] = {
        "[H]": [direct, component, component_2]
    }

    assert classify_diffusion_intermediate(
        graph,
        direct,
        0,
        _state(graph, 0),
        _state(graph, 1),
        _metadata(),
        n_slab=3,
        n_lateral=0,
        n_reacting=1,
        nl_mult=1.25,
    ) is None


def test_persisted_refinement_is_rechecked_after_network_expansion():
    graph = _surface_graph(3)
    site_a = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    site_i = _h_site(graph, surface_index=1, node_id=11, iso_class=1)
    site_b = _h_site(graph, surface_index=2, node_id=12, iso_class=2)
    direct = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[(site_a, 0, site_b, 0)],
        member_node_ids=[([10], [12])],
    )
    graph.graph["adsorbate_sites"] = {"[H]": [site_a, site_i, site_b]}
    # The original calculation predates discovery of the component member.
    graph.graph["diffusion_sites"] = {"[H]": [direct]}
    lateral = DiffusionLateral(
        lateral_class=0,
        stable=True,
        direct_event_status="elementary",
        neb_intermediate_refinement=_metadata(),
        atoms_neb_refinement_initial=_state(graph, 0),
        atoms_neb_refinement_final=_state(graph, 1),
    )
    _reclassify_persisted_diffusion_intermediate(
        graph,
        direct,
        0,
        lateral,
        nl_mult=1.25,
    )
    assert lateral.direct_event_status == "elementary"

    component = DiffusionSite(
        reactant="[H]",
        iso_class=1,
        members=[(site_a, 0, site_i, 0)],
        member_node_ids=[([10], [11])],
    )
    component_2 = DiffusionSite(
        reactant="[H]",
        iso_class=2,
        members=[(site_i, 0, site_b, 0)],
        member_node_ids=[([11], [12])],
    )
    graph.graph["diffusion_sites"]["[H]"].extend([component, component_2])
    _reclassify_persisted_diffusion_intermediate(
        graph,
        direct,
        0,
        lateral,
        nl_mult=1.25,
    )

    assert lateral.direct_event_status == "composite"
    assert lateral.stable is None


def test_bond_intermediate_matches_alternative_ab_pair_with_same_product():
    graph = _surface_graph(4)
    a0 = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    a1 = _h_site(graph, surface_index=1, node_id=11, iso_class=1)
    a2 = _h_site(graph, surface_index=2, node_id=12, iso_class=2)
    product = AdsorbateSite(
        reactant="[H][H]",
        n_atoms=2,
        atom_cliques=[frozenset({3}), frozenset({3})],
        positions=np.array([[12.0, 0.0, 1.5], [12.0, 0.0, 2.2]]),
        iso_class=3,
        members=[[frozenset({3}), frozenset({3})]],
        member_node_ids=[[20, 21]],
    )
    for node_id, reactant_index, z in ((20, 0, 1.5), (21, 1, 2.2)):
        graph.add_node(
            node_id,
            type="adsorbate",
            element="H",
            reactant_index=reactant_index,
            clique=frozenset({3}),
            siblings=frozenset({21 if node_id == 20 else 20}),
            position=np.array([12.0, 0.0, z]),
        )
        graph.add_edge(3, node_id)
    graph.add_edge(20, 21)
    product._member_cliques = [(frozenset({3}), frozenset({3}))]

    template = BondReactionTemplate("[H]", "[H]", "[H][H]")
    direct = BondReactionSite(
        template=template,
        iso_class=0,
        members=[(a0, 0, a1, 0, product, 0)],
        member_node_ids=[([10], [11], [20, 21])],
    )
    alternative = BondReactionSite(
        template=template,
        iso_class=1,
        members=[(a0, 0, a2, 0, product, 0)],
        member_node_ids=[([10], [12], [20, 21])],
    )
    graph.graph["bond_reaction_sites"] = [direct, alternative]
    assert classify_bond_intermediate(
        graph,
        direct,
        0,
        _state(graph, 0, 1),
        _state(graph, 0, 2),
        _metadata(),
        n_slab=4,
        n_lateral=0,
        n_reacting=2,
        nl_mult=1.25,
    ) is None

    diffusion_component = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[(a1, 0, a2, 0)],
        member_node_ids=[([11], [12])],
    )
    graph.graph["diffusion_sites"] = {"[H]": [diffusion_component]}

    certificate = classify_bond_intermediate(
        graph,
        direct,
        0,
        _state(graph, 0, 1),
        _state(graph, 0, 2),
        _metadata(),
        n_slab=4,
        n_lateral=0,
        n_reacting=2,
        nl_mult=1.25,
    )

    assert certificate is not None
    assert certificate["classification"] == "composite"
    assert certificate["changed_endpoint"] == "ab"


def test_suppression_latches_and_rate_index_zeros_composite(tmp_path):
    graph = _surface_graph(3)
    site_a = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    site_b = _h_site(graph, surface_index=2, node_id=12, iso_class=1)
    direct = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[(site_a, 0, site_b, 0)],
        member_node_ids=[([10], [12])],
    )
    lateral = DiffusionLateral(
        lateral_class=0,
        stable=True,
        energy_a=0.0,
        energy_b=0.1,
        energy_ts=0.5,
    )
    certificate = {
        "classification": "composite",
        "reason": "registered_intermediate_diffusion_placement",
    }
    with pytest.raises(CompositeDirectEventDetected):
        retain_refinement_and_maybe_suppress(
            lateral,
            _state(graph, 0),
            _state(graph, 1),
            _metadata(),
            certificate,
        )
    assert lateral.stable is None
    assert lateral.direct_event_status == "composite"
    assert len(lateral.neb_intermediate_refinement_history) == 1

    reaction = DiffusionReaction(
        kind="diffusion",
        direction="a_to_b",
        site=direct,
        member_index=0,
        lateral_class=lateral,
        delta_e=0.1,
        barrier=0.5,
        rate=1.0e6,
    )
    index = _ReactionIndex([], diffusion_sites=[direct])
    index.install(reaction, direct, 0)
    assert index.total_rate() == 0.0
    assert index.reactions[index.leaf_id(direct, 0)] is None

    folder = ReactionWriter(tmp_path).write_invalid_diffusion(direct, lateral)
    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["diagnostic_status"] == "composite_direct_event"
    assert payload["direct_event_certificate"] == certificate
    assert payload["automatic_retry"] is False


def test_index_removes_sibling_members_sharing_suppressed_lateral():
    graph = _surface_graph(4)
    site_a = _h_site(graph, surface_index=0, node_id=10, iso_class=0)
    site_b = _h_site(graph, surface_index=1, node_id=11, iso_class=1)
    site_c = _h_site(graph, surface_index=2, node_id=12, iso_class=2)
    site_d = _h_site(graph, surface_index=3, node_id=13, iso_class=3)
    direct = DiffusionSite(
        reactant="[H]",
        iso_class=0,
        members=[
            (site_a, 0, site_b, 0),
            (site_c, 0, site_d, 0),
        ],
        member_node_ids=[([10], [11]), ([12], [13])],
    )
    lateral = DiffusionLateral(
        lateral_class=0,
        members=[0, 1],
        stable=True,
        direct_event_status="elementary",
    )
    direct.lateral_classes = [lateral]
    sibling = DiffusionReaction(
        kind="diffusion",
        direction="a_to_b",
        site=direct,
        member_index=1,
        lateral_class=lateral,
        delta_e=0.0,
        barrier=0.2,
        rate=100.0,
    )
    index = _ReactionIndex([], diffusion_sites=[direct])
    index.install(sibling, direct, 1)
    assert index.total_rate() == pytest.approx(100.0)

    lateral.direct_event_status = "composite"
    lateral.stable = None
    index.install(None, direct, 0)

    assert index.total_rate() == 0.0
    assert index.reactions == [None, None]


def test_nonmatching_refinement_is_retained_without_suppression():
    lateral = DiffusionLateral(lateral_class=0)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    retain_refinement_and_maybe_suppress(
        lateral,
        atoms,
        atoms.copy(),
        _metadata(),
        None,
    )

    assert lateral.direct_event_status is None
    assert lateral.neb_intermediate_refinement == _metadata()
    assert lateral.neb_intermediate_refinement_history == [_metadata()]
