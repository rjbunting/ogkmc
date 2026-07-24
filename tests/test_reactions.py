"""Focused regression tests for reaction applicability and rates."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest
from ase import Atoms

from autokmc.reactions import AdsorptionReaction, BondReaction, DiffusionReaction
import autokmc.reactions.bond as bond_module
import autokmc.reactions.diffusion as diffusion_module
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


def _detached_seed_band() -> list[Atoms]:
    return [
        Atoms("H", positions=[[float(index), 0.0, 0.0]])
        for index in range(3)
    ]


def test_diffusion_runs_missing_bare_path_before_lateral_neb(monkeypatch):
    order = []
    bare = SimpleNamespace(
        stable=None,
        members=[],
        lateral_class=0,
        atoms_neb_path=None,
    )
    lateral = SimpleNamespace(
        stable=None,
        members=[0],
        lateral_class=1,
        energy_a=None,
        energy_b=None,
        energy_ts=None,
    )
    site = SimpleNamespace(
        iso_class=4,
        _member_lc={},
        applicable_reactions=[],
    )

    monkeypatch.setattr(
        diffusion_module,
        "is_diffusion_applicable",
        lambda *_args: (True, "a_to_b"),
    )

    def get_bare(*_args, **_kwargs):
        order.append("bare_lookup")
        return bare

    def classify_lateral(*_args, **_kwargs):
        order.append("lateral_classify")
        return lateral

    def check_stability(_graph, _site, member_index, lc, _calculator, **kwargs):
        if lc is bare:
            order.append("bare_neb")
            assert kwargs["capture_neb_path"] is True
            lc._warm_start_neb_path = _detached_seed_band()
            lc._warm_start_member_index = member_index
            lc.stable = True
            return 0.0, 0.2, 0.8

        order.append("lateral_neb")
        assert kwargs["neb_seed_member_index"] == member_index
        assert len(kwargs["neb_seed_path"]) == 3
        lc.energy_a = 0.0
        lc.energy_b = 0.2
        lc.energy_ts = 0.8
        lc.stable = True
        return 0.0, 0.2, 0.8

    monkeypatch.setattr(diffusion_module, "get_diffusion_bare_lateral", get_bare)
    monkeypatch.setattr(
        diffusion_module,
        "check_diffusion_site_lateral",
        classify_lateral,
    )
    monkeypatch.setattr(
        diffusion_module,
        "check_diffusion_stability",
        check_stability,
    )

    reaction = diffusion_module.get_applicable_diffusion_for_member(
        nx.Graph(),
        site,
        0,
        object(),
        temperature=500.0,
        n_images=1,
    )

    assert reaction is not None
    assert reaction.lateral_class is lateral
    assert site._member_lc[0] is lateral
    assert bare.members == []
    assert order == [
        "bare_lookup",
        "lateral_classify",
        "bare_neb",
        "lateral_neb",
    ]


def test_diffusion_bare_failure_falls_back_to_configured_interpolation(
    monkeypatch,
):
    bare = SimpleNamespace(
        stable=None,
        members=[],
        lateral_class=0,
        atoms_neb_path=None,
    )
    lateral = SimpleNamespace(
        stable=None,
        members=[0],
        lateral_class=1,
        energy_a=None,
        energy_b=None,
        energy_ts=None,
    )
    site = SimpleNamespace(
        iso_class=5,
        _member_lc={},
        applicable_reactions=[],
    )

    monkeypatch.setattr(
        diffusion_module,
        "is_diffusion_applicable",
        lambda *_args: (True, "a_to_b"),
    )
    monkeypatch.setattr(
        diffusion_module,
        "get_diffusion_bare_lateral",
        lambda *_args, **_kwargs: bare,
    )
    monkeypatch.setattr(
        diffusion_module,
        "check_diffusion_site_lateral",
        lambda *_args, **_kwargs: lateral,
    )

    def check_stability(_graph, _site, _member_index, lc, _calculator, **kwargs):
        if lc is bare:
            raise diffusion_module.DiffusionStabilityError("bare failed")
        assert kwargs["neb_seed_path"] is None
        lc.energy_a = 0.0
        lc.energy_b = 0.2
        lc.energy_ts = 0.8
        lc.stable = True
        return 0.0, 0.2, 0.8

    monkeypatch.setattr(
        diffusion_module,
        "check_diffusion_stability",
        check_stability,
    )

    reaction = diffusion_module.get_applicable_diffusion_for_member(
        nx.Graph(),
        site,
        0,
        object(),
        temperature=500.0,
        n_images=1,
    )

    assert reaction is not None
    assert bare.stable is False
    assert "bare failed" in bare.invalid_reason


@pytest.mark.parametrize(
    (
        "reaction_module",
        "applicability_name",
        "bare_getter_name",
        "classifier_name",
        "stability_name",
        "entrypoint_name",
        "direction",
    ),
    [
        (
            diffusion_module,
            "is_diffusion_applicable",
            "get_diffusion_bare_lateral",
            "check_diffusion_site_lateral",
            "check_diffusion_stability",
            "get_applicable_diffusion_for_member",
            "a_to_b",
        ),
        (
            bond_module,
            "is_bond_applicable",
            "get_bond_bare_lateral",
            "check_bond_site_lateral",
            "check_bond_site_stability",
            "get_applicable_bond_reaction_for_member",
            "couple",
        ),
    ],
)
def test_bare_stability_value_errors_are_not_swallowed_as_classifier_errors(
    monkeypatch,
    reaction_module,
    applicability_name,
    bare_getter_name,
    classifier_name,
    stability_name,
    entrypoint_name,
    direction,
):
    bare = SimpleNamespace(
        stable=None,
        members=[],
        lateral_class=0,
        atoms_neb_path=None,
    )
    lateral = SimpleNamespace(stable=None, members=[0], lateral_class=1)
    site = SimpleNamespace(
        iso_class=8,
        _member_lc={},
        applicable_reactions=[],
    )

    monkeypatch.setattr(
        reaction_module,
        applicability_name,
        lambda *_args: (True, direction),
    )
    monkeypatch.setattr(
        reaction_module,
        bare_getter_name,
        lambda *_args, **_kwargs: bare,
    )
    monkeypatch.setattr(
        reaction_module,
        classifier_name,
        lambda *_args, **_kwargs: lateral,
    )

    def fail_stability(*_args, **_kwargs):
        raise ValueError("invalid NEB configuration")

    monkeypatch.setattr(reaction_module, stability_name, fail_stability)

    with pytest.raises(ValueError, match="invalid NEB configuration"):
        getattr(reaction_module, entrypoint_name)(
            nx.Graph(),
            site,
            0,
            object(),
            temperature=500.0,
            n_images=1,
        )


@pytest.mark.parametrize(
    "seed_helper",
    [
        diffusion_module._diffusion_seed_path,
        bond_module._bond_seed_path,
    ],
)
def test_bare_seed_requires_explicit_same_member_provenance(seed_helper):
    lateral_class = SimpleNamespace(
        atoms_neb_path=_detached_seed_band(),
    )

    assert seed_helper(
        lateral_class,
        n_images=1,
        current_member_index=0,
    ) == (None, None)

    lateral_class._warm_start_member_index = 1
    assert seed_helper(
        lateral_class,
        n_images=1,
        current_member_index=0,
    ) == (None, None)

    lateral_class._warm_start_member_index = 0
    path, source_member = seed_helper(
        lateral_class,
        n_images=1,
        current_member_index=0,
    )
    assert path is not None
    assert len(path) == 3
    assert source_member == 0


def test_bond_runs_missing_bare_path_before_lateral_neb(monkeypatch):
    order = []
    bare = SimpleNamespace(
        stable=None,
        members=[],
        lateral_class=0,
        atoms_neb_path=None,
    )
    lateral = SimpleNamespace(
        stable=None,
        members=[0],
        lateral_class=1,
        energy_ab=None,
        energy_c=None,
        energy_ts=None,
    )
    site = SimpleNamespace(
        iso_class=6,
        _member_lc={},
        applicable_reactions=[],
    )

    monkeypatch.setattr(
        bond_module,
        "is_bond_applicable",
        lambda *_args: (True, "couple"),
    )

    def get_bare(*_args, **_kwargs):
        order.append("bare_lookup")
        return bare

    def classify_lateral(*_args, **_kwargs):
        order.append("lateral_classify")
        return lateral

    def check_stability(_graph, _site, member_index, lc, _calculator, **kwargs):
        if lc is bare:
            order.append("bare_neb")
            assert kwargs["capture_neb_path"] is True
            lc._warm_start_neb_path = _detached_seed_band()
            lc._warm_start_member_index = member_index
            lc.stable = True
            return 0.0, 0.2, 0.8

        order.append("lateral_neb")
        assert kwargs["neb_seed_member_index"] == member_index
        assert len(kwargs["neb_seed_path"]) == 3
        lc.energy_ab = 0.0
        lc.energy_c = 0.2
        lc.energy_ts = 0.8
        lc.stable = True
        return 0.0, 0.2, 0.8

    monkeypatch.setattr(bond_module, "get_bond_bare_lateral", get_bare)
    monkeypatch.setattr(
        bond_module,
        "check_bond_site_lateral",
        classify_lateral,
    )
    monkeypatch.setattr(
        bond_module,
        "check_bond_site_stability",
        check_stability,
    )

    reaction = bond_module.get_applicable_bond_reaction_for_member(
        nx.Graph(),
        site,
        0,
        object(),
        temperature=500.0,
        n_images=1,
    )

    assert reaction is not None
    assert reaction.lateral_class is lateral
    assert site._member_lc[0] is lateral
    assert bare.members == []
    assert order == [
        "bare_lookup",
        "lateral_classify",
        "bare_neb",
        "lateral_neb",
    ]
