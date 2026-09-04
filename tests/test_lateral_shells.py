"""Focused tests for configurable lateral-interaction shell depth."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from autokmc.core.constants import LATERAL_SHELLS_DEFAULT
from autokmc.kmc import engine as engine_module
from autokmc.kmc.initialization import initialise_runtime
from autokmc.kmc.models import (
    KMCChannels,
    KMCResumeState,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.reactions import adsorption, bond, diffusion


def test_kmc_settings_use_configured_lateral_shells_for_refresh_radius():
    assert KMCSettings(
        temperature=500.0,
        n_steps=1,
    ).max_n_shells == LATERAL_SHELLS_DEFAULT
    assert KMCSettings(
        temperature=500.0,
        n_steps=1,
        lateral_shells=3,
    ).max_n_shells == 3
    assert KMCSettings(
        temperature=500.0,
        n_steps=1,
        lateral_interactions=False,
        lateral_shells=3,
    ).max_n_shells == 0


def test_legacy_engine_adapter_preserves_lateral_shell_depth(monkeypatch):
    captured = {}

    class Result:
        @staticmethod
        def to_legacy_dict():
            return {}

    def run(request, *, functions):
        captured["settings"] = request.settings
        captured["functions"] = functions
        return Result()

    sentinel = object()
    monkeypatch.setattr(engine_module, "run_kmc", run)
    monkeypatch.setattr(engine_module, "default_kmc_functions", lambda: sentinel)

    engine_module.run_kmc_steps(
        nx.Graph(),
        [],
        None,
        {},
        temperature=500.0,
        n_steps=0,
        lateral_shells=4,
        verbose=False,
    )

    assert captured["settings"].lateral_shells == 4
    assert captured["settings"].max_n_shells == 4
    assert captured["functions"] is sentinel


def test_initial_sweep_receives_lateral_shell_depth():
    calls = []

    def compute_adsorption(*_args, **kwargs):
        calls.append(kwargs)
        return []

    initialise_runtime(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(
            temperature=500.0,
            n_steps=0,
            lateral_shells=3,
            verbose=False,
        ),
        KMCChannels(),
        KMCThermochemistry(),
        KMCResumeState(),
        rng=1,
        compute_adsorption=compute_adsorption,
    )

    assert calls[0]["lateral_shells"] == 3


def test_adsorption_classifier_receives_lateral_shell_depth(monkeypatch):
    calls = []
    site = SimpleNamespace(
        reactant="[O]",
        iso_class=0,
        applicable_reactions=[],
    )
    monkeypatch.setattr(adsorption, "is_clique_blocked", lambda *_args: False)

    def classify(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stable=False)

    monkeypatch.setattr(
        adsorption,
        "check_adsorbate_site_lateral",
        classify,
    )

    reaction = adsorption.get_applicable_reaction_for_member(
        nx.Graph(),
        site,
        0,
        None,
        {"[O]": 0.0},
        temperature=500.0,
        lateral_shells=2,
    )

    assert reaction is None
    assert calls == [{"n_shells": 2, "ignore_lateral": False, "include_all_occupied": False}]


def test_diffusion_classifier_receives_lateral_shell_depth(monkeypatch):
    calls = []
    site = SimpleNamespace(
        iso_class=0,
        applicable_reactions=[],
    )
    monkeypatch.setattr(
        diffusion,
        "is_diffusion_applicable",
        lambda *_args: (True, "a_to_b"),
    )

    def classify(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stable=False)

    monkeypatch.setattr(
        diffusion,
        "check_diffusion_site_lateral",
        classify,
    )

    reaction = diffusion.get_applicable_diffusion_for_member(
        nx.Graph(),
        site,
        0,
        None,
        temperature=500.0,
        lateral_shells=2,
    )

    assert reaction is None
    assert calls == [{"n_shells": 2, "ignore_lateral": False, "include_all_occupied": False}]


def test_bond_classifier_receives_lateral_shell_depth(monkeypatch):
    calls = []
    site = SimpleNamespace(
        iso_class=0,
        applicable_reactions=[],
    )
    monkeypatch.setattr(
        bond,
        "is_bond_applicable",
        lambda *_args: (True, "couple"),
    )

    def classify(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stable=False)

    monkeypatch.setattr(
        bond,
        "check_bond_site_lateral",
        classify,
    )

    reaction = bond.get_applicable_bond_reaction_for_member(
        nx.Graph(),
        site,
        0,
        None,
        temperature=500.0,
        lateral_shells=2,
    )

    assert reaction is None
    assert calls == [{"n_shells": 2, "ignore_lateral": False, "include_all_occupied": False}]
