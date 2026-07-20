"""Focused tests for KMC state helpers."""

from __future__ import annotations

from types import SimpleNamespace
import random

import networkx as nx
import numpy as np
import pytest

from autokmc.kmc.engine import (
    _capture_rng_state,
    _final_occupancy_by_species,
    _restore_rng_state,
)
from autokmc.kmc.execute import execute_reaction
from autokmc.kmc.expansion import (
    _rebuild_bond_reverse_indexes,
    expand_bond_sites_after_event,
    expand_bond_sites_for_new_species,
    initialise_bond_registry,
)
from autokmc.kmc.sampling import _RateSegmentTree
from autokmc.reactions.bond import _bond_energetics_cached, is_bond_applicable
from autokmc.sites.bond import BondReactionLateral


def _site(smiles: str, iso: int, node_id: int, clique: frozenset[int]):
    return SimpleNamespace(
        reactant=smiles,
        iso_class=iso,
        member_node_ids=[[node_id]],
        _member_cliques=[(clique,)],
        _n_occupied=0,
    )


def _graph_with_adsorbates(*nodes: tuple[int, frozenset[int], bool]) -> nx.Graph:
    G = nx.Graph()
    G.graph["occupied_by_clique"] = {}
    G.graph["n_occupied"] = 0
    for node_id, clique, occupied in nodes:
        G.add_node(node_id, clique=clique, occupied=occupied, type="adsorbate")
        if occupied:
            G.graph["n_occupied"] += 1
            G.graph["occupied_by_clique"].setdefault(clique, set()).add(node_id)
    return G


def test_execute_reaction_rejects_unknown_diffusion_direction():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    site_a = _site("[O]", 0, 10, clique_a)
    site_b = _site("[O]", 1, 20, clique_b)
    G = _graph_with_adsorbates((10, clique_a, True), (20, clique_b, False))
    reaction = SimpleNamespace(
        kind="diffusion",
        direction="sideways",
        site=SimpleNamespace(members=[(site_a, 0, site_b, 0)]),
        member_index=0,
    )

    with pytest.raises(ValueError, match="unknown diffusion direction"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.nodes[20]["occupied"] is False


def test_execute_reaction_rejects_unknown_bond_direction():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    clique_c = frozenset({3})
    site_a = _site("[C]", 0, 10, clique_a)
    site_b = _site("[O]", 0, 20, clique_b)
    site_c = _site("[C]=O", 0, 30, clique_c)
    G = _graph_with_adsorbates(
        (10, clique_a, True),
        (20, clique_b, True),
        (30, clique_c, False),
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="merge-ish",
        site=SimpleNamespace(members=[(site_a, 0, site_b, 0, site_c, 0)]),
        member_index=0,
    )

    with pytest.raises(ValueError, match="unknown bond direction"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.nodes[20]["occupied"] is True
    assert G.nodes[30]["occupied"] is False


def test_final_occupancy_keys_include_species():
    sites = [
        SimpleNamespace(reactant="[C-]#[O+]", iso_class=0, _n_occupied=1),
        SimpleNamespace(reactant="[O]", iso_class=0, _n_occupied=2),
    ]

    assert _final_occupancy_by_species(sites) == {
        "[C-]#[O+]:iso0": 1,
        "[O]:iso0": 2,
    }


def test_final_occupancy_accumulates_duplicate_species_iso_keys():
    sites = [
        SimpleNamespace(reactant="[O]", iso_class=0, _n_occupied=1),
        SimpleNamespace(reactant="[O]", iso_class=0, _n_occupied=2),
    ]

    assert _final_occupancy_by_species(sites) == {"[O]:iso0": 3}


@pytest.mark.parametrize("kind", ["numpy", "python"])
def test_checkpoint_rng_state_continues_exact_random_stream(kind):
    rng = np.random.default_rng(91) if kind == "numpy" else random.Random(91)
    draw = rng.random
    _ = [draw() for _ in range(4)]
    state = _capture_rng_state(rng)
    expected = [draw() for _ in range(8)]

    restored = _restore_rng_state(
        np.random.default_rng(0) if kind == "numpy" else random.Random(0),
        state,
    )

    assert [restored.random() for _ in range(8)] == expected


def test_rebuild_bond_reverse_indexes_keeps_existing_and_new_sites():
    G = nx.Graph()
    old_clique = frozenset({1, 2})
    new_clique = frozenset({3})
    old_brs = SimpleNamespace(
        _member_cliques=[((old_clique,), (frozenset({4}),), (frozenset({5}),))]
    )
    new_brs = SimpleNamespace(
        _member_cliques=[((new_clique,), (frozenset({6}),), (frozenset({7}),))]
    )

    _rebuild_bond_reverse_indexes(G, [old_brs, new_brs])

    assert G.graph["bond_clique_to_members"][old_clique] == [(old_brs, 0)]
    assert G.graph["bond_clique_to_members"][new_clique] == [(new_brs, 0)]
    assert G.graph["bond_surface_node_to_members"][1] == [(old_brs, 0)]
    assert G.graph["bond_surface_node_to_members"][3] == [(new_brs, 0)]


def test_execute_reaction_rejects_stale_diffusion_state():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    site_a = _site("[O]", 0, 10, clique_a)
    site_b = _site("[O]", 1, 20, clique_b)
    G = _graph_with_adsorbates((10, clique_a, True), (20, clique_b, True))
    reaction = SimpleNamespace(
        kind="diffusion",
        direction="a_to_b",
        site=SimpleNamespace(members=[(site_a, 0, site_b, 0)]),
        member_index=0,
    )

    with pytest.raises(ValueError, match="target member is already occupied"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.nodes[20]["occupied"] is True
    assert G.graph["n_occupied"] == 2


def test_execute_reaction_rejects_stale_bond_state():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    clique_c = frozenset({3})
    site_a = _site("[C]", 0, 10, clique_a)
    site_b = _site("[O]", 0, 20, clique_b)
    site_c = _site("[C]=O", 0, 30, clique_c)
    G = _graph_with_adsorbates(
        (10, clique_a, True),
        (20, clique_b, True),
        (30, clique_c, True),
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="couple",
        site=SimpleNamespace(members=[(site_a, 0, site_b, 0, site_c, 0)]),
        member_index=0,
    )

    with pytest.raises(ValueError, match="C member is already occupied"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.nodes[20]["occupied"] is True
    assert G.nodes[30]["occupied"] is True
    assert G.graph["n_occupied"] == 3


def test_execute_reaction_rejects_stale_adsorption_state():
    clique = frozenset({1})
    site = _site("[O]", 0, 10, clique)
    G = _graph_with_adsorbates((10, clique, True))
    reaction = SimpleNamespace(
        kind="adsorption",
        site=site,
        member_index=0,
    )

    with pytest.raises(ValueError, match="member is already occupied"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.graph["n_occupied"] == 1


def test_execute_reaction_rejects_clique_blocked_adsorption():
    clique = frozenset({1})
    target = _site("[O]", 0, 10, clique)
    G = _graph_with_adsorbates((10, clique, False), (20, clique, True))
    reaction = SimpleNamespace(
        kind="adsorption",
        site=target,
        member_index=0,
    )

    with pytest.raises(ValueError, match="clique-blocked"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is False
    assert G.nodes[20]["occupied"] is True
    assert G.graph["n_occupied"] == 1


def test_execute_reaction_rejects_empty_desorption_state():
    clique = frozenset({1})
    site = _site("[O]", 0, 10, clique)
    G = _graph_with_adsorbates((10, clique, False))
    reaction = SimpleNamespace(
        kind="desorption",
        site=site,
        member_index=0,
    )

    with pytest.raises(ValueError, match="member is not occupied"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is False
    assert G.graph["n_occupied"] == 0


def test_execute_reaction_rejects_bond_dissociation_with_ab_clique_collision():
    clique = frozenset({1})
    product_clique = frozenset({2})
    site_a = _site("[C]", 0, 10, clique)
    site_b = _site("[O]", 0, 20, clique)
    site_c = _site("[C]=O", 0, 30, product_clique)
    G = _graph_with_adsorbates(
        (10, clique, False),
        (20, clique, False),
        (30, product_clique, True),
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="dissoc",
        site=SimpleNamespace(members=[(site_a, 0, site_b, 0, site_c, 0)]),
        member_index=0,
    )

    with pytest.raises(ValueError, match="share a surface clique"):
        execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is False
    assert G.nodes[20]["occupied"] is False
    assert G.nodes[30]["occupied"] is True
    assert G.graph["n_occupied"] == 1


def test_gas_product_bond_coupling_releases_ab_to_gas():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    site_a = _site("[C]", 0, 10, clique_a)
    site_b = _site("[O]", 0, 20, clique_b)
    G = _graph_with_adsorbates(
        (10, clique_a, True),
        (20, clique_b, True),
    )
    brs = SimpleNamespace(
        gas_product=True,
        members=[(site_a, 0, site_b, 0, None, -1)],
        _member_cliques=[((clique_a,), (clique_b,), tuple())],
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="couple",
        site=brs,
        member_index=0,
    )

    touched = execute_reaction(G, reaction)

    assert touched == {clique_a, clique_b}
    assert G.nodes[10]["occupied"] is False
    assert G.nodes[20]["occupied"] is False
    assert G.graph["n_occupied"] == 0


def test_gas_product_bond_reverse_consumes_gas_to_make_ab():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    site_a = _site("[C]", 0, 10, clique_a)
    site_b = _site("[O]", 0, 20, clique_b)
    G = _graph_with_adsorbates(
        (10, clique_a, False),
        (20, clique_b, False),
    )
    brs = SimpleNamespace(
        gas_product=True,
        members=[(site_a, 0, site_b, 0, None, -1)],
        _member_cliques=[((clique_a,), (clique_b,), tuple())],
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="dissoc",
        site=brs,
        member_index=0,
    )

    assert is_bond_applicable(G, brs, 0) == (True, "dissoc")
    execute_reaction(G, reaction)

    assert G.nodes[10]["occupied"] is True
    assert G.nodes[20]["occupied"] is True
    assert G.graph["n_occupied"] == 2


def test_gas_product_reverse_bond_rate_scales_with_pressure():
    lc_low = BondReactionLateral(
        lateral_class=0,
        energy_ab=0.0,
        energy_c=0.2,
        energy_ts=0.5,
        gas_product=True,
        gas_pressure_bar=1.0,
    )
    lc_high = BondReactionLateral(
        lateral_class=0,
        energy_ab=0.0,
        energy_c=0.2,
        energy_ts=0.5,
        gas_product=True,
        gas_pressure_bar=3.0,
    )
    lc_zero = BondReactionLateral(
        lateral_class=0,
        energy_ab=0.0,
        energy_c=0.2,
        energy_ts=0.5,
        gas_product=True,
        gas_pressure_bar=0.0,
    )

    couple_low = _bond_energetics_cached(lc_low, "couple", temperature=500.0)
    couple_high = _bond_energetics_cached(lc_high, "couple", temperature=500.0)
    reverse_low = _bond_energetics_cached(lc_low, "dissoc", temperature=500.0)
    reverse_high = _bond_energetics_cached(lc_high, "dissoc", temperature=500.0)
    reverse_zero = _bond_energetics_cached(lc_zero, "dissoc", temperature=500.0)

    assert couple_high[2] == pytest.approx(couple_low[2])
    assert reverse_high[2] == pytest.approx(3.0 * reverse_low[2])
    assert reverse_zero[2] == 0.0


def test_rate_segment_tree_clamps_boundary_samples_to_real_leaves():
    tree = _RateSegmentTree(3)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)

    assert tree.sample(1.0) == 2
    assert tree.sample(2.0) == 2
    assert tree.sample(-1.0) == 0


def test_kmc_package_exports_public_api():
    import autokmc.kmc as kmc

    assert kmc.execute_reaction is execute_reaction
    assert callable(kmc.run_kmc_steps)
    assert callable(kmc.sample_tau)
    assert callable(kmc.initialise_bond_registry)


def test_expand_bond_sites_after_event_rejects_unknown_direction():
    reaction = SimpleNamespace(
        kind="bond",
        direction="sideways",
        site=SimpleNamespace(template=SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="unknown bond direction"):
        expand_bond_sites_after_event(nx.Graph(), reaction, calculator=None)


def test_bond_expansion_defers_triple_prune_until_after_stability(monkeypatch):
    from autokmc.kmc import expansion

    G = nx.Graph()
    initial_site = SimpleNamespace(member_node_ids=[[1]])
    initialise_bond_registry(
        G,
        reactants={
            "C": SimpleNamespace(smiles="C"),
            "A": SimpleNamespace(smiles="A"),
            "B": SimpleNamespace(smiles="B"),
        },
        adsorbate_sites={
            "C": [initial_site],
            "A": [initial_site],
            "B": [initial_site],
        },
        templates=[],
        bond_sites=[],
    )

    template = SimpleNamespace(
        smiles_a="A",
        smiles_b="B",
        smiles_c="C",
        source="dissociation",
    )
    kept = SimpleNamespace(iso_class=-1)
    dropped = SimpleNamespace(iso_class=-1)
    calls = []

    monkeypatch.setattr(
        expansion,
        "derive_dissociation_templates",
        lambda *args, **kwargs: [template],
    )
    monkeypatch.setattr(
        expansion,
        "derive_coupling_templates",
        lambda *args, **kwargs: [],
    )

    def fake_find_bond_sites(*args, **kwargs):
        calls.append(("find", kwargs["prune_by_triple"]))
        return [kept, dropped]

    def fake_prune_unstable(*args, **kwargs):
        calls.append(("stability", [site.iso_class for site in args[1]]))
        return [kept]

    def fake_prune_triple(sites, **kwargs):
        calls.append(("triple", [site.iso_class for site in sites]))
        return list(sites)

    monkeypatch.setattr(expansion, "find_bond_sites", fake_find_bond_sites)
    monkeypatch.setattr(expansion, "prune_unstable_bond_sites", fake_prune_unstable)
    monkeypatch.setattr(expansion, "_prune_one_per_adsorption_triple", fake_prune_triple)
    monkeypatch.setattr(
        expansion,
        "_rebuild_bond_reverse_indexes",
        lambda graph, sites: calls.append(("rebuild", len(sites))),
    )

    out = expand_bond_sites_for_new_species(
        G,
        "C",
        calculator=object(),
        bond_prune_by_triple=True,
        bond_prune_with_calculator=True,
    )

    assert out == [kept]
    assert calls == [
        ("find", False),
        ("stability", [-1, -1]),
        ("triple", [-1]),
        ("rebuild", 1),
    ]
