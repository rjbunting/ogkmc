"""Regressions for stable KMC identities and typed workflow compatibility."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.io.config import FreeEnergyCfg, KMCCfg, OutputCfg, RunConfig
from autokmc.kmc.index import _ReactionIndex
from autokmc.kmc.initialization import normalise_channels
from autokmc.kmc.models import (
    BondGrowthOptions,
    KMCChannels,
    KMCFunctions,
    KMCRunResult,
    KMCRuntime,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.kmc.network import DynamicNetworkExpander, ExpansionChanges
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.bond import BondReactionSite, BondReactionTemplate
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.identity import member_identifier, site_identifier
from autokmc.utils.telemetry import RuntimeTelemetry, telemetry_context
from autokmc.workflow.models import (
    ChannelRuntimeOptions,
    OutputSinks,
    PreparedNetwork,
    RunIdentity,
    SimulationContext,
    ThermoRuntime,
)
from autokmc.workflow.simulation import execute_kmc_stage


def _site(*, iso_class: int, member_nodes: list[list[int]]) -> AdsorbateSite:
    clique = frozenset({1})
    return AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[clique],
        positions=np.array([[0.0, 0.0, 1.0]]),
        iso_class=iso_class,
        members=[[clique] for _ in member_nodes],
        member_node_ids=member_nodes,
    )


def test_site_and_member_identity_ignore_iso_and_member_list_numbering():
    original = _site(iso_class=3, member_nodes=[[10], [20]])
    reconstructed = _site(iso_class=91, member_nodes=[[20], [10]])

    assert site_identifier(reconstructed) == site_identifier(original)
    assert member_identifier(reconstructed, 0) == member_identifier(original, 1)
    assert member_identifier(reconstructed, 1) == member_identifier(original, 0)

    index = _ReactionIndex([original])
    assert index.leaf_id(reconstructed, 0) == 1
    assert index.leaf_id(reconstructed, 1) == 0


def test_diffusion_identity_treats_endpoint_order_as_unordered():
    original = DiffusionSite(
        reactant="[O]",
        iso_class=2,
        member_node_ids=[([10], [20])],
    )
    reversed_endpoints = DiffusionSite(
        reactant="[O]",
        iso_class=77,
        member_node_ids=[([20], [10])],
    )

    assert site_identifier(reversed_endpoints) == site_identifier(original)
    assert member_identifier(reversed_endpoints, 0) == member_identifier(original, 0)


def test_bond_identity_exchanges_only_symmetric_reactant_roles():
    symmetric = BondReactionSite(
        template=BondReactionTemplate("[O]", "[O]", "O=O"),
        iso_class=1,
        member_node_ids=[([10], [20], [30, 31])],
    )
    symmetric_swapped = BondReactionSite(
        template=BondReactionTemplate("[O]", "[O]", "O=O"),
        iso_class=90,
        member_node_ids=[([20], [10], [30, 31])],
    )
    asymmetric = BondReactionSite(
        template=BondReactionTemplate("[O]", "[H]", "[OH]"),
        iso_class=2,
        member_node_ids=[([10], [20], [30, 31])],
    )
    asymmetric_swapped = BondReactionSite(
        template=BondReactionTemplate("[O]", "[H]", "[OH]"),
        iso_class=91,
        member_node_ids=[([20], [10], [30, 31])],
    )

    assert site_identifier(symmetric_swapped) == site_identifier(symmetric)
    assert member_identifier(symmetric_swapped, 0) == member_identifier(symmetric, 0)
    assert site_identifier(asymmetric_swapped) != site_identifier(asymmetric)
    assert member_identifier(asymmetric_swapped, 0) != member_identifier(asymmetric, 0)


def test_bond_site_preserves_pre_site_id_positional_constructor():
    """The appended stable-ID field must not rebind legacy positional data."""
    member_cliques = [
        ((frozenset({1}),), (frozenset({2}),), (frozenset({3}),))
    ]
    site = BondReactionSite(
        BondReactionTemplate("[O]", "[H]", "[OH]"),
        0,
        [],
        [],
        [],
        None,
        0,
        False,
        None,
        6.0,
        member_cliques,
    )

    assert site._member_cliques is member_cliques
    assert site.site_id == ""


def test_dynamic_expander_installs_and_samples_new_index_rates():
    existing_site = _site(iso_class=0, member_nodes=[[10]])
    new_site = _site(
        iso_class=1,
        member_nodes=[[20], [21], [22], [23]],
    )
    existing_reaction = SimpleNamespace(
        rate=1.0,
        site=existing_site,
        member_index=0,
    )
    new_reaction = SimpleNamespace(
        rate=3.0,
        site=new_site,
        member_index=3,
    )
    existing_site.applicable_reactions = [existing_reaction]
    new_site.applicable_reactions = [new_reaction]

    index = _ReactionIndex([existing_site])
    index.install_site(existing_site, existing_site.applicable_reactions)
    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=index,
        history=[],
        reaction_counts={},
        current_time_s=0.0,
        start_step=0,
    )
    expander = DynamicNetworkExpander(
        KMCSystem(nx.Graph(), [existing_site], None, {}),
        KMCSettings(temperature=500.0, n_steps=0, verbose=False),
        KMCChannels(),
        KMCThermochemistry(),
        KMCFunctions(
            compute_adsorption=lambda *_args, **_kwargs: [],
            recompute_affected=lambda *_args, **_kwargs: ([], []),
            expand_bond_network=lambda *_args, **_kwargs: [],
        ),
        runtime,
    )

    expander._extend_index([new_site], [], [], ExpansionChanges())

    assert index.n_total == 5
    assert index.reactions[0] is existing_reaction
    assert index.reactions[index.leaf_id(new_site, 3)] is new_reaction
    assert index.total_rate() == pytest.approx(4.0)
    assert index.sample(0.9) is new_reaction


@pytest.mark.parametrize(
    ("growth_frozen", "expected"),
    [
        (None, [2, 4]),
        ([9], [9]),
    ],
)
def test_dynamic_expander_resolves_run_frozen_indices_only_as_fallback(
    growth_frozen,
    expected,
):
    captured: dict = {}

    def expand_bond_network(*_args, **kwargs):
        captured.update(kwargs)
        return []

    graph = nx.Graph()
    graph.graph["bond_registry"] = {"adsorbate_sites": {}}
    system = KMCSystem(graph, [], None, {})
    settings = KMCSettings(
        temperature=500.0,
        n_steps=0,
        frozen_indices=[2, 4],
        verbose=False,
    )
    channels = KMCChannels(
        bond_growth_options=BondGrowthOptions(
            frozen_indices=growth_frozen,
        )
    )
    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=_ReactionIndex([]),
        history=[],
        reaction_counts={},
        current_time_s=0.0,
        start_step=0,
    )
    expander = DynamicNetworkExpander(
        system,
        settings,
        channels,
        KMCThermochemistry(),
        KMCFunctions(
            compute_adsorption=lambda *_args, **_kwargs: [],
            recompute_affected=lambda *_args, **_kwargs: ([], []),
            expand_bond_network=expand_bond_network,
        ),
        runtime,
    )

    expander.expand(SimpleNamespace(kind="adsorption"))

    assert captured["frozen_indices"] == expected


def test_dynamic_expander_skips_already_expanded_products():
    product = "[OH]"
    graph = nx.Graph()
    graph.graph["bond_registry"] = {
        "adsorbate_sites": {},
        "expanded_species": {product},
    }
    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=_ReactionIndex([]),
        history=[],
        reaction_counts={},
        current_time_s=0.0,
        start_step=0,
    )

    def unexpected_expansion(*_args, **_kwargs):
        raise AssertionError("already-expanded product was rescanned")

    expander = DynamicNetworkExpander(
        KMCSystem(graph, [], None, {}),
        KMCSettings(temperature=500.0, n_steps=0, verbose=False),
        KMCChannels(),
        KMCThermochemistry(),
        KMCFunctions(
            compute_adsorption=lambda *_args, **_kwargs: [],
            recompute_affected=lambda *_args, **_kwargs: ([], []),
            expand_bond_network=unexpected_expansion,
        ),
        runtime,
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="couple",
        site=SimpleNamespace(
            template=SimpleNamespace(
                smiles_a="[O]",
                smiles_b="[H]",
                smiles_c=product,
            )
        ),
    )

    assert expander.expand(reaction) == ExpansionChanges()


def test_legacy_bond_growth_runtime_keys_are_routed_or_stripped():
    free_energy_options = object()
    channels = KMCChannels(
        bond_growth_kwargs={
            "find_diffusion": True,
            "calculation_cache_root": "legacy-calculations",
            "free_energy_options": free_energy_options,
            "free_energy_temperature_k": 725.0,
            "verbose": True,
            "vib_cache_root": "legacy-vibrations",
        }
    )

    normalised, thermochemistry = normalise_channels(
        channels,
        KMCThermochemistry(),
    )

    assert normalised.bond_growth_options.find_diffusion is True
    assert normalised.bond_growth_kwargs == {"find_diffusion": True}
    assert thermochemistry.calculation_cache_root == "legacy-calculations"
    assert thermochemistry.free_energy_options is free_energy_options
    assert thermochemistry.vib_cache_root == "legacy-vibrations"
    assert thermochemistry.free_energy_temperature_k == pytest.approx(725.0)


@pytest.mark.parametrize("channel_name", ["diffusion_kwargs", "bond_kwargs"])
def test_legacy_channel_thermochemistry_keys_are_routed(channel_name):
    free_energy_options = object()
    channels = KMCChannels(**{
        channel_name: {
            "free_energy_options": free_energy_options,
            "vib_cache_root": "legacy-vibrations",
        }
    })

    _, thermochemistry = normalise_channels(channels, KMCThermochemistry())

    assert thermochemistry.free_energy_options is free_energy_options
    assert thermochemistry.vib_cache_root == "legacy-vibrations"


@pytest.mark.parametrize("isaac_export_enabled", [False, True])
def test_configured_workflow_calls_public_typed_run_boundary(
    tmp_path,
    monkeypatch,
    isaac_export_enabled,
):
    import autokmc.kmc.engine as engine_module
    import autokmc.workflow.simulation as simulation_module

    calls = []
    exports = []

    def fake_run_kmc(request):
        calls.append(request)
        return KMCRunResult(
            time_s=0.0,
            steps_executed=0,
            history=[],
            reaction_counts={},
            final_occupancy={},
        )

    monkeypatch.setattr(engine_module, "run_kmc", fake_run_kmc)

    def fake_export(cache_root, path):
        output_path = Path(path)
        output_path.write_text("[]\n", encoding="utf-8")
        exports.append((cache_root, output_path))
        return output_path

    monkeypatch.setattr(simulation_module, "write_isaac_export", fake_export)
    cfg = RunConfig(
        output=OutputCfg(
            dir=str(tmp_path),
            calculation_cache_enabled=False,
            isaac_export_enabled=isaac_export_enabled,
            trajectory_dump_every=0,
        ),
        kmc=KMCCfg(n_steps=0, log_every=0),
        free_energy=FreeEnergyCfg(enabled=False),
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / cfg.output.run_manifest_filename,
        run_id="typed-workflow-test",
    )
    context = SimulationContext(
        graph=nx.Graph(),
        network=PreparedNetwork(),
        calculator=None,
        frozen_indices=None,
        thermo=ThermoRuntime(None, str(tmp_path / "vibrations"), None),
        channels=ChannelRuntimeOptions(),
    )

    telemetry = RuntimeTelemetry()
    telemetry.increment("optimization.structure.calls")
    with telemetry_context(telemetry):
        result = execute_kmc_stage(
            cfg,
            identity,
            context,
            started_at=datetime.now(timezone.utc),
            verbose=False,
            telemetry=telemetry,
        )

    assert len(calls) == 1
    assert calls[0].settings.n_steps == 0
    assert calls[0].telemetry is telemetry
    assert result["steps_executed"] == 0
    assert result["termination_status"] == "complete"
    assert result["termination_reason"] == "no_steps_requested"
    assert result["simulated_time_s"] == 0.0
    assert result["performance"]["calculations"]["optimization_runs"] == 1
    diagnostics = json.loads(
        (
            tmp_path / "diagnostics" / "performance.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        diagnostics["telemetry"]["counters"]["optimization.structure.calls"]
        == 1
    )
    assert (
        diagnostics["telemetry"]["counters"]["persistence.event_sync.calls"]
        == 1
    )
    assert diagnostics["summary"] == result["performance"]
    assert diagnostics["wall_time_s"] == result["wall_time_s"]
    for timing_name in (
        "output.artifact_inventory.seconds",
        "output.performance_diagnostics.seconds",
        "output.summary.seconds",
    ):
        assert timing_name in diagnostics["telemetry"]["timings_s"]
        assert diagnostics["telemetry"]["timings_s"][timing_name] >= 0.0
    persisted = json.loads(
        (tmp_path / cfg.output.summary_filename).read_text(encoding="utf-8")
    )
    assert persisted["run"]["performance"] == result["performance"]
    if isaac_export_enabled:
        assert exports == [
            (None, tmp_path / cfg.output.isaac_export_filename)
        ]
        assert result["outputs"]["isaac_records"] == str(
            tmp_path / cfg.output.isaac_export_filename
        )
    else:
        assert exports == []
        assert result["outputs"]["isaac_records"] is None
        assert not (tmp_path / cfg.output.isaac_export_filename).exists()


def test_output_sink_cleanup_is_idempotent():
    calls: list[str] = []
    sinks = OutputSinks(
        reactions=SimpleNamespace(close=lambda: calls.append("reactions")),
        trajectory=SimpleNamespace(close=lambda: calls.append("trajectory")),
        summary=object(),
    )

    sinks.close()
    sinks.close()

    assert calls == ["trajectory", "reactions"]


def test_stage7_failure_persists_available_invalid_and_quarantine_metadata(
    tmp_path,
    monkeypatch,
):
    import autokmc.kmc.engine as engine_module

    def fail_run(_request):
        raise RuntimeError("stage 7 failed")

    monkeypatch.setattr(engine_module, "run_kmc", fail_run)
    cfg = RunConfig(
        output=OutputCfg(
            dir=str(tmp_path),
            calculation_cache_enabled=False,
            trajectory_dump_every=0,
        ),
        kmc=KMCCfg(n_steps=1, log_every=0),
        free_energy=FreeEnergyCfg(enabled=False),
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / cfg.output.run_manifest_filename,
        run_id="failed-workflow-test",
    )
    invalid_site = SimpleNamespace(
        lateral_classes=[SimpleNamespace(stable=False)]
    )
    network = PreparedNetwork(adsorbate_sites=[invalid_site])
    context = SimulationContext(
        graph=nx.Graph(),
        network=network,
        calculator=None,
        frozen_indices=None,
        thermo=ThermoRuntime(None, str(tmp_path / "vibrations"), None),
        channels=ChannelRuntimeOptions(),
    )
    quarantined = (
        tmp_path
        / "uncommitted_reactions"
        / "after_checkpoint_step_0"
        / "adsorption"
        / "species"
        / "iso0"
    )
    quarantined.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="stage 7 failed"):
        execute_kmc_stage(
            cfg,
            identity,
            context,
            started_at=datetime.now(timezone.utc),
            verbose=False,
            telemetry=RuntimeTelemetry(),
        )

    manifest = json.loads(identity.manifest_path.read_text(encoding="utf-8"))
    assert manifest["lifecycle"]["status"] == "failed"
    assert manifest["invalid_counts"] == {
        "adsorption": 1,
        "bond": 0,
        "diffusion": 0,
        "persisted_invalid_reactions": 0,
        "total": 1,
    }
    assert manifest["quarantine_locations"] == [
        (
            "uncommitted_reactions/after_checkpoint_step_0/"
            "adsorption/species/iso0"
        )
    ]


def test_stage7_enrichment_failure_uses_rich_failure_finalization(
    tmp_path,
    monkeypatch,
):
    import autokmc.workflow.simulation as simulation_module

    def fail_enrichment(*_args, **_kwargs):
        raise RuntimeError("manifest enrichment failed")

    monkeypatch.setattr(
        simulation_module,
        "enrich_run_manifest",
        fail_enrichment,
    )
    cfg = RunConfig(
        output=OutputCfg(
            dir=str(tmp_path),
            calculation_cache_enabled=False,
            trajectory_dump_every=0,
        ),
        kmc=KMCCfg(n_steps=1, log_every=0),
        free_energy=FreeEnergyCfg(enabled=False),
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / cfg.output.run_manifest_filename,
        run_id="failed-enrichment-test",
    )
    invalid_site = SimpleNamespace(
        lateral_classes=[SimpleNamespace(stable=False)]
    )
    context = SimulationContext(
        graph=nx.Graph(),
        network=PreparedNetwork(adsorbate_sites=[invalid_site]),
        calculator=None,
        frozen_indices=None,
        thermo=ThermoRuntime(None, str(tmp_path / "vibrations"), None),
        channels=ChannelRuntimeOptions(),
    )
    quarantined = (
        tmp_path
        / "uncommitted_reactions"
        / "after_checkpoint_step_0"
        / "adsorption"
        / "species"
        / "iso0"
    )
    quarantined.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="manifest enrichment failed"):
        execute_kmc_stage(
            cfg,
            identity,
            context,
            started_at=datetime.now(timezone.utc),
            verbose=False,
            telemetry=RuntimeTelemetry(),
        )

    manifest = json.loads(identity.manifest_path.read_text(encoding="utf-8"))
    assert manifest["lifecycle"]["status"] == "failed"
    assert manifest["lifecycle"]["termination_stage"] == "initializing"
    assert manifest["invalid_counts"] == {
        "adsorption": 1,
        "bond": 0,
        "diffusion": 0,
        "total": 1,
    }
    assert manifest["quarantine_locations"] == [
        (
            "uncommitted_reactions/after_checkpoint_step_0/"
            "adsorption/species/iso0"
        )
    ]
