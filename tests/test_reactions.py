"""Focused regression tests for reaction applicability and rates."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from autokmc.reactions import AdsorptionReaction, BondReaction, DiffusionReaction
from autokmc.reactions.adsorption import fast_reaction_for_member
from autokmc.reactions.adsorption import _energetics_cached
from autokmc.reactions.bond import _bond_energetics_cached, is_bond_applicable
from autokmc.reactions.diffusion import (
    _diffusion_energetics_cached,
    is_diffusion_applicable,
)
from autokmc.sites.bond import BondReactionLateral


def _site(smiles: str, iso: int, node_id: int, clique: frozenset[int]):
    return SimpleNamespace(
        reactant=smiles,
        iso_class=iso,
        member_node_ids=[[node_id]],
        _member_cliques=[(clique,)],
    )


def test_diffusion_target_guard_scans_graph_without_occupancy_index():
    clique_a = frozenset({1})
    clique_b = frozenset({2})
    source = _site("[O]", 0, 10, clique_a)
    target = _site("[O]", 1, 20, clique_b)
    blocker = _site("[H]", 0, 30, clique_b)
    ds = SimpleNamespace(
        member_node_ids=[([10], [20])],
        members=[(source, 0, target, 0)],
    )
    G = nx.Graph()
    G.add_node(10, type="adsorbate", clique=clique_a, occupied=True)
    G.add_node(20, type="adsorbate", clique=clique_b, occupied=False)
    G.add_node(30, type="adsorbate", clique=clique_b, occupied=True)
    G.graph.pop("occupied_by_clique", None)

    assert is_diffusion_applicable(G, ds, 0) == (False, None)
    assert blocker.reactant == "[H]"


def test_fast_adsorption_preserves_free_energy_and_pressure_rate_inputs():
    clique = frozenset({1})
    site = _site("[O]", 0, 10, clique)
    site._member_is_occupied = lambda graph, node_ids: any(
        nid in graph and graph.nodes[nid].get("occupied", False)
        for nid in node_ids
    )
    lc = SimpleNamespace(
        stable=True,
        energy_occupied=5.0,
        energy_unoccupied=4.0,
        g_occupied=3.0,
        g_unoccupied=2.0,
    )
    site._member_lc = {0: lc}
    G = nx.Graph()
    G.add_node(10, type="adsorbate", clique=clique, occupied=False)
    G.graph["occupied_by_clique"] = {clique: set()}

    rxn_1bar = fast_reaction_for_member(
        G,
        site,
        0,
        {"[O]": 10.0},
        temperature=500.0,
        gas_g={"[O]": 0.0},
        partial_pressures={"[O]": 1.0},
    )
    rxn_2bar = fast_reaction_for_member(
        G,
        site,
        0,
        {"[O]": 10.0},
        temperature=500.0,
        gas_g={"[O]": 0.0},
        partial_pressures={"[O]": 2.0},
    )

    assert rxn_1bar is not None
    assert rxn_2bar is not None
    assert rxn_1bar.delta_e == pytest.approx(1.0)
    assert rxn_2bar.rate == pytest.approx(2.0 * rxn_1bar.rate)


def test_diffusion_energetics_rejects_unknown_direction():
    lc = SimpleNamespace(energy_a=0.0, energy_b=1.0, energy_ts=2.0)

    with pytest.raises(ValueError, match="unknown diffusion direction"):
        _diffusion_energetics_cached(lc, "sideways", temperature=500.0)


def test_bond_energetics_rejects_unknown_direction():
    lc = SimpleNamespace(energy_ab=0.0, energy_c=1.0, energy_ts=2.0)

    with pytest.raises(ValueError, match="unknown bond direction"):
        _bond_energetics_cached(lc, "merge-ish", temperature=500.0)


def test_nonfinite_adsorption_energetics_are_rejected():
    lc = SimpleNamespace(energy_occupied=float("nan"), energy_unoccupied=0.0)

    with pytest.raises(ValueError, match="must be finite"):
        _energetics_cached(lc, 0.0, False, temperature=500.0)


def test_bond_rate_uses_free_energies_when_available():
    electronic = BondReactionLateral(
        lateral_class=0, energy_ab=0.0, energy_c=1.0, energy_ts=2.0,
    )
    free = BondReactionLateral(
        lateral_class=0,
        energy_ab=0.0,
        energy_c=1.0,
        energy_ts=2.0,
        g_ab=0.0,
        g_c=0.2,
        g_ts=0.3,
    )

    electronic_result = _bond_energetics_cached(
        electronic, "couple", temperature=500.0,
    )
    free_result = _bond_energetics_cached(free, "couple", temperature=500.0)

    assert electronic_result[0] == pytest.approx(1.0)
    assert free_result[0] == pytest.approx(0.2)
    assert free_result[1] == pytest.approx(0.3)
    assert free_result[2] > electronic_result[2]


def test_bond_applicability_rejects_ab_exact_clique_collision():
    clique = frozenset({1})
    product_clique = frozenset({2})
    site_a = _site("[C]", 0, 10, clique)
    site_b = _site("[O]", 0, 20, clique)
    site_c = _site("[C]=O", 0, 30, product_clique)
    brs = SimpleNamespace(
        member_node_ids=[([10], [20], [30])],
        members=[(site_a, 0, site_b, 0, site_c, 0)],
        _member_cliques=[((clique,), (clique,), (product_clique,))],
    )
    G = nx.Graph()
    G.add_node(10, type="adsorbate", clique=clique, occupied=False)
    G.add_node(20, type="adsorbate", clique=clique, occupied=False)
    G.add_node(30, type="adsorbate", clique=product_clique, occupied=True)
    G.graph["occupied_by_clique"] = {product_clique: {30}, clique: set()}

    assert is_bond_applicable(G, brs, 0) == (False, None)


def test_reactions_package_exports_public_models():
    assert AdsorptionReaction.__name__ == "AdsorptionReaction"
    assert DiffusionReaction.__name__ == "DiffusionReaction"
    assert BondReaction.__name__ == "BondReaction"
