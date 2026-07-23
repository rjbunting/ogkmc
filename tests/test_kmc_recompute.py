"""Focused behavior tests for incremental KMC recomputation."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from autokmc.io.calculators import CalculatorPool
from autokmc.kmc import recompute as recompute_module


def _settings() -> dict:
    return {
        "temperature": 500.0,
        "transmission_coefficient": 1.0,
        "frozen_indices": None,
        "fmax": 0.05,
        "max_steps": 20,
        "verbose": False,
    }


def test_recompute_empty_affected_set_is_a_noop():
    assert recompute_module.recompute_affected_sites(
        nx.Graph(),
        [],
        set(),
        None,
        {},
        **_settings(),
    ) == ([], [])


def test_member_jobs_reuse_calculator_pool_executor():
    pool = CalculatorPool([object(), object()])
    jobs = [
        lambda: ("adsorption", [], 1),
        lambda: ("adsorption", [], 2),
    ]
    try:
        first = recompute_module._run_member_jobs(
            jobs,
            pool,
            allow_parallel=True,
        )
        executor = pool.executor
        second = recompute_module._run_member_jobs(
            jobs,
            pool,
            allow_parallel=True,
        )
        assert pool.executor is executor
    finally:
        pool.shutdown()

    assert first == second == [
        ("adsorption", [], 1),
        ("adsorption", [], 2),
    ]


def test_recompute_trusts_present_reverse_index_with_no_members(monkeypatch):
    monkeypatch.setattr(
        recompute_module,
        "_lateral_shell_members",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        recompute_module,
        "_fallback_adsorption_members",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("valid empty index must not trigger a full scan")
        ),
    )

    updated, invalid = recompute_module.recompute_affected_sites(
        nx.Graph(),
        [SimpleNamespace()],
        {frozenset({1})},
        None,
        {},
        **_settings(),
    )

    assert updated == []
    assert invalid == []


def test_recompute_scans_members_only_when_reverse_index_is_absent(monkeypatch):
    site = SimpleNamespace(
        member_node_ids=[[10]],
        lateral_classes=[],
        applicable_reactions=[],
    )
    reaction = SimpleNamespace(site=site, member_index=0, rate=1.0)
    monkeypatch.setattr(
        recompute_module,
        "_lateral_shell_members",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        recompute_module,
        "_fallback_adsorption_members",
        lambda *_a, **_k: [(site, 0)],
    )
    monkeypatch.setattr(
        recompute_module,
        "get_applicable_reaction_for_member",
        lambda *_a, **_k: reaction,
    )

    updated, invalid = recompute_module.recompute_affected_sites(
        nx.Graph(),
        [site],
        {frozenset({1})},
        None,
        {},
        **_settings(),
    )

    assert updated == [reaction]
    assert invalid == []


def test_recompute_refreshes_each_channel_and_strips_reserved_kwargs(monkeypatch):
    adsorption_site = SimpleNamespace(
        member_node_ids=[[10]],
        lateral_classes=[],
        applicable_reactions=[],
    )
    diffusion_site = SimpleNamespace(
        member_node_ids=[([10], [11])],
        lateral_classes=[],
        applicable_reactions=[],
    )
    bond_site = SimpleNamespace(
        member_node_ids=[([10], [11], [12])],
        lateral_classes=[],
        applicable_reactions=[],
    )
    adsorption_reaction = SimpleNamespace(
        site=adsorption_site,
        member_index=0,
        rate=1.0,
    )
    diffusion_reaction = SimpleNamespace(
        site=diffusion_site,
        member_index=0,
        rate=2.0,
    )
    bond_reaction = SimpleNamespace(
        site=bond_site,
        member_index=0,
        rate=3.0,
    )
    calls: dict[str, dict] = {}

    monkeypatch.setattr(
        recompute_module,
        "_lateral_shell_members",
        lambda *_a, **_k: [(adsorption_site, 0)],
    )
    monkeypatch.setattr(
        recompute_module,
        "_diffusion_lateral_shell_members",
        lambda *_a, **_k: [(diffusion_site, 0)],
    )
    monkeypatch.setattr(
        recompute_module,
        "_bond_lateral_shell_members",
        lambda *_a, **_k: [(bond_site, 0)],
    )
    def adsorption_kernel(*_args, **kwargs):
        calls["adsorption"] = kwargs
        return adsorption_reaction

    monkeypatch.setattr(
        recompute_module,
        "get_applicable_reaction_for_member",
        adsorption_kernel,
    )

    def diffusion_kernel(*_args, **kwargs):
        calls["diffusion"] = kwargs
        return diffusion_reaction

    def bond_kernel(*_args, **kwargs):
        calls["bond"] = kwargs
        return bond_reaction

    monkeypatch.setattr(
        recompute_module,
        "get_applicable_diffusion_for_member",
        diffusion_kernel,
    )
    monkeypatch.setattr(
        recompute_module,
        "get_applicable_bond_reaction_for_member",
        bond_kernel,
    )

    class Index:
        _adsorbate_ids: set[str] = set()
        _diffusion_ids: set[str] = set()
        _bond_ids: set[str] = set()

        def __init__(self):
            self.installed = []

        def install(self, reaction, site, member_index):
            self.installed.append((site, member_index, reaction))

    index = Index()
    updated, invalid = recompute_module.recompute_affected_sites(
        nx.Graph(),
        [adsorption_site],
        {frozenset({10})},
        None,
        {},
        rxn_index=index,
        diffusion_sites=[diffusion_site],
        diffusion_kwargs={
            "n_images": 3,
            "calculation_cache_root": "must-not-leak",
            "free_energy_options": object(),
            "vib_cache_root": "must-not-leak",
        },
        bond_sites=[bond_site],
        bond_kwargs={
            "n_images": 4,
            "calculation_cache_root": "must-not-leak",
            "free_energy_options": object(),
            "vib_cache_root": "must-not-leak",
        },
        lateral_shells=3,
        calculation_cache_root="cache",
        **_settings(),
    )

    assert updated == [adsorption_reaction, diffusion_reaction, bond_reaction]
    assert invalid == [diffusion_site]
    assert [site for site, _, _ in index.installed] == [
        adsorption_site,
        diffusion_site,
        bond_site,
    ]
    assert calls["diffusion"]["n_images"] == 3
    assert calls["bond"]["n_images"] == 4
    assert calls["adsorption"]["lateral_shells"] == 3
    assert calls["diffusion"]["lateral_shells"] == 3
    assert calls["bond"]["lateral_shells"] == 3
    assert calls["diffusion"]["calculation_cache_root"] == "cache"
    assert calls["bond"]["calculation_cache_root"] == "cache"


def test_recompute_replaces_only_the_affected_member_leaf(monkeypatch):
    site = SimpleNamespace(
        member_node_ids=[[10], [11]],
        lateral_classes=[],
    )
    old_zero = SimpleNamespace(site=site, member_index=0, rate=1.0)
    old_one = SimpleNamespace(site=site, member_index=1, rate=2.0)
    new_one = SimpleNamespace(site=site, member_index=1, rate=4.0)
    site.applicable_reactions = [old_zero, old_one]

    monkeypatch.setattr(
        recompute_module,
        "_lateral_shell_members",
        lambda *_a, **_k: [(site, 1)],
    )
    monkeypatch.setattr(
        recompute_module,
        "_diffusion_lateral_shell_members",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        recompute_module,
        "_bond_lateral_shell_members",
        lambda *_a, **_k: [],
    )
    evaluated = []

    def evaluate(_graph, evaluated_site, member_index, *_args, **_kwargs):
        evaluated.append((evaluated_site, member_index))
        return new_one

    monkeypatch.setattr(
        recompute_module,
        "get_applicable_reaction_for_member",
        evaluate,
    )

    class Index:
        _adsorbate_ids: set[str] = set()
        _diffusion_ids: set[str] = set()
        _bond_ids: set[str] = set()

        def __init__(self):
            self.installed = []

        def install(self, reaction, installed_site, member_index):
            self.installed.append((reaction, installed_site, member_index))

    index = Index()
    updated, _ = recompute_module.recompute_affected_sites(
        nx.Graph(),
        [site],
        {frozenset({10})},
        None,
        {},
        rxn_index=index,
        **_settings(),
    )

    assert evaluated == [(site, 1)]
    assert updated == [new_one]
    assert site.applicable_reactions == [old_zero, new_one]
    assert index.installed == [(new_one, site, 1)]
