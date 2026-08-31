"""Focused tests for the refactored configuration-to-runtime workflow."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import autokmc.io.resume_contract as resume_contract_module
from autokmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from autokmc.io.event_log import EventHistory, EventLogCommit, EventLogRecovery
from autokmc.io.calculators import CalculatorCfg, CalculatorConfigError
from autokmc.io.config import (
    AdsorptionCfg,
    BondCfg,
    CheckpointCfg,
    DiffusionCfg,
    FreeEnergyCfg,
    KMCCfg,
    OptimizationCfg,
    OutputCfg,
    ReactantCfg,
    RunConfig,
    StructureCfg,
)
from autokmc.io.resume_contract import (
    make_resume_contract,
    scientific_config_payload,
    verify_resume_contract,
)
from autokmc.workflow.models import RunIdentity, ThermoRuntime
from autokmc.workflow.network import SpeciesNetworkBuilder
from autokmc.workflow.runtime import (
    create_output_sinks,
    resolve_channel_runtime,
    resolve_kmc_resume,
    resolve_run_identity,
)


def _builder(cfg, tmp_path: Path, graph: nx.Graph, **overrides):
    values = {
        "cfg": cfg,
        "identity": RunIdentity(
            output_dir=tmp_path,
            manifest_path=tmp_path / "run_manifest.json",
            run_id="workflow-test",
        ),
        "graph": graph,
        "calculator_resource": None,
        "frozen_indices": None,
        "thermo_runtime": ThermoRuntime(
            options=None,
            vibration_cache_root=str(tmp_path / "vibrations"),
            calculation_cache_root=None,
        ),
        "verbose": False,
    }
    values.update(overrides)
    return SpeciesNetworkBuilder(**values)


def test_prepare_calculator_rejects_a_missing_construction_path():
    from autokmc.workflow.stages import prepare_calculator

    with pytest.raises(
        CalculatorConfigError,
        match=r"calculator\.import_path or calculator\.factory",
    ):
        prepare_calculator(RunConfig(calculator=CalculatorCfg()))


@pytest.mark.parametrize(
    ("configured_cap", "expected_cap"),
    [(4, 4), (None, None)],
)
def test_channel_runtime_propagates_anchor_clique_cap(
    configured_cap,
    expected_cap,
):
    cfg = RunConfig(
        adsorption=AdsorptionCfg(
            anchor_k_max=configured_cap,
            prune_fmax=0.021,
            prune_max_steps=111,
            endpoint_fmax=0.032,
            endpoint_max_steps=222,
        ),
        bond=BondCfg(
            enabled=True,
            prune_fmax=0.043,
            prune_max_steps=333,
            gas_precursor_relax=False,
            gas_precursor_distance=2.2,
        ),
    )

    runtime = resolve_channel_runtime(cfg, frozen_indices=None)

    assert runtime.adsorption.fmax == pytest.approx(0.032)
    assert runtime.adsorption.max_steps == 222
    assert runtime.bond is not None
    assert runtime.bond.gas_precursor_relax is False
    assert runtime.bond.gas_precursor_distance == pytest.approx(2.2)
    assert runtime.bond_growth is not None
    assert runtime.bond_growth.anchor_k_max == expected_cap
    assert runtime.bond_growth.adsorption_prune_fmax == pytest.approx(0.021)
    assert runtime.bond_growth.adsorption_prune_max_steps == 111
    assert runtime.bond_growth.bond_prune_fmax == pytest.approx(0.043)
    assert runtime.bond_growth.bond_prune_max_steps == 333


def test_channel_runtime_propagates_optimizer_choices():
    cfg = RunConfig(
        optimization=OptimizationCfg(
            optimizer="fire",
            optimizer_kwargs={"dt": 0.01, "maxstep": 0.05},
            neb_optimizer="mdmin",
            neb_optimizer_kwargs={"dt": 0.02, "maxstep": 0.04},
            neb_climb_optimizer="fire",
            neb_climb_optimizer_kwargs={
                "dt": 0.005,
                "dtmax": 0.02,
                "maxstep": 0.01,
                "downhill_check": False,
            },
            neb_method="aseneb",
            neb_band_eval="batched",
            neb_geometry_guard_multiplier=4.0,
            neb_intermediate_stagnation_steps=75,
            neb_intermediate_max_refinements=4,
            neb_intermediate_energy_tolerance=0.002,
            neb_intermediate_minimum_prominence=0.03,
        ),
        diffusion=DiffusionCfg(enabled=True),
        bond=BondCfg(enabled=True),
    )

    runtime = resolve_channel_runtime(cfg, frozen_indices=None)

    assert runtime.diffusion is not None
    assert runtime.diffusion.optimizer == "fire"
    assert runtime.diffusion.optimizer_kwargs == {
        "dt": pytest.approx(0.01),
        "maxstep": pytest.approx(0.05),
    }
    assert runtime.diffusion.neb_optimizer == "mdmin"
    assert runtime.diffusion.neb_optimizer_kwargs == {
        "dt": pytest.approx(0.02),
        "maxstep": pytest.approx(0.04),
    }
    assert runtime.diffusion.neb_climb_optimizer == "fire"
    assert runtime.diffusion.neb_climb_optimizer_kwargs == {
        "dt": pytest.approx(0.005),
        "dtmax": pytest.approx(0.02),
        "maxstep": pytest.approx(0.01),
        "downhill_check": False,
    }
    assert runtime.diffusion.neb_band_eval == "batched"
    assert runtime.diffusion.neb_method == "aseneb"
    assert runtime.diffusion.neb_geometry_guard_multiplier == pytest.approx(4.0)
    assert runtime.diffusion.neb_intermediate_stagnation_steps == 75
    assert runtime.diffusion.neb_intermediate_max_refinements == 4
    assert runtime.diffusion.neb_intermediate_energy_tolerance == pytest.approx(
        0.002
    )
    assert runtime.diffusion.neb_intermediate_minimum_prominence == pytest.approx(
        0.03
    )
    assert runtime.diffusion.image_spacing == pytest.approx(0.25)
    assert runtime.diffusion.min_images == 6
    assert runtime.diffusion.max_images == 8
    assert runtime.bond is not None
    assert runtime.bond.gas_precursor_distance == pytest.approx(2.5)
    assert runtime.bond.image_spacing == pytest.approx(0.25)
    assert runtime.bond.min_images == 6
    assert runtime.bond.max_images == 8
    assert runtime.bond.optimizer == "fire"
    assert runtime.bond.optimizer_kwargs == runtime.diffusion.optimizer_kwargs
    assert runtime.bond.neb_optimizer == "mdmin"
    assert (
        runtime.bond.neb_optimizer_kwargs
        == runtime.diffusion.neb_optimizer_kwargs
    )
    assert (
        runtime.bond.neb_climb_optimizer
        == runtime.diffusion.neb_climb_optimizer
    )
    assert (
        runtime.bond.neb_climb_optimizer_kwargs
        == runtime.diffusion.neb_climb_optimizer_kwargs
    )
    assert runtime.bond.neb_band_eval == "batched"
    assert runtime.bond.neb_method == "aseneb"
    assert runtime.bond.neb_geometry_guard_multiplier == pytest.approx(4.0)
    assert runtime.bond.neb_intermediate_stagnation_steps == 75
    assert runtime.bond.neb_intermediate_max_refinements == 4
    assert runtime.bond.neb_intermediate_energy_tolerance == pytest.approx(
        0.002
    )
    assert runtime.bond.neb_intermediate_minimum_prominence == pytest.approx(
        0.03
    )
    assert runtime.bond_growth is not None
    assert runtime.bond_growth.optimizer == "fire"
    assert runtime.bond_growth.optimizer_kwargs == {
        "dt": pytest.approx(0.01),
        "maxstep": pytest.approx(0.05),
    }


def test_channel_runtime_neb_modes_are_isolated():
    def resolve(mode: str):
        cfg = RunConfig(
            optimization=OptimizationCfg(neb_band_eval=mode),
            diffusion=DiffusionCfg(enabled=True),
            bond=BondCfg(enabled=True),
        )
        return resolve_channel_runtime(cfg, frozen_indices=None)

    images = resolve("images")
    batched = resolve("batched")

    assert images.diffusion is not None
    assert images.bond is not None
    assert batched.diffusion is not None
    assert batched.bond is not None
    assert images.diffusion.neb_band_eval == "images"
    assert images.bond.neb_band_eval == "images"
    assert batched.diffusion.neb_band_eval == "batched"
    assert batched.bond.neb_band_eval == "batched"


def test_network_builder_flattens_diffusion_channels(tmp_path, monkeypatch):
    import autokmc.sites.diffusion as diffusion_module

    graph = nx.Graph()
    sites = [SimpleNamespace(reactant="[O]")]
    first = object()
    second = object()

    def find_diffusion_sites(candidate_graph, candidate_sites, **kwargs):
        assert candidate_graph is graph
        assert candidate_sites == sites
        assert kwargs["max_hops"] == 3
        assert kwargs["prune_by_adsorption_pair"] is True
        return {"[O]": [first], "O=O": [second]}

    monkeypatch.setattr(
        diffusion_module,
        "find_diffusion_sites",
        find_diffusion_sites,
    )
    cfg = RunConfig(
        diffusion=DiffusionCfg(
            enabled=True,
            max_hops=3,
            prune_by_adsorption_pair=True,
        )
    )

    network = _builder(cfg, tmp_path, graph).prepare([], sites)

    assert network.adsorbate_sites == sites
    assert network.initial_adsorbate_sites == sites
    assert network.diffusion_sites == [first, second]
    assert network.bond_sites == []


def test_network_builder_keeps_stability_before_triple_pruning(
    tmp_path,
    monkeypatch,
):
    import autokmc.kmc.expansion as expansion_module
    import autokmc.sites.bond as bond_module
    import autokmc.workflow.network as network_module

    graph = nx.Graph()
    reactants = [
        SimpleNamespace(smiles="O"),
        SimpleNamespace(smiles="[H]"),
        SimpleNamespace(smiles="[OH]"),
    ]
    sites = [SimpleNamespace(reactant=item.smiles) for item in reactants]
    template = SimpleNamespace(
        smiles_a="O",
        smiles_b="[H]",
        smiles_c="[OH]",
    )
    first = SimpleNamespace(iso_class=4)
    second = SimpleNamespace(iso_class=9)
    calls: list[str] = []

    monkeypatch.setattr(
        network_module,
        "derive_configured_bond_templates",
        lambda *_args: [template],
    )

    def find_bond_sites(*_args, **kwargs):
        calls.append("enumerate")
        assert kwargs["prune_by_triple"] is False
        return [first, second]

    def prune_unstable(*_args, **_kwargs):
        calls.append("stability")
        return [second]

    def prune_triple(candidate_sites, **_kwargs):
        calls.append("triple")
        assert candidate_sites == [second]
        return candidate_sites

    def rebuild(*_args):
        calls.append("rebuild")

    def initialise(*_args, **kwargs):
        calls.append("registry")
        assert kwargs["bond_sites"] == [second]

    monkeypatch.setattr(bond_module, "find_bond_sites", find_bond_sites)
    monkeypatch.setattr(
        bond_module,
        "prune_unstable_bond_sites",
        prune_unstable,
    )
    monkeypatch.setattr(
        bond_module,
        "_prune_one_per_adsorption_triple",
        prune_triple,
    )
    monkeypatch.setattr(
        bond_module,
        "rebuild_bond_reverse_indexes",
        rebuild,
    )
    monkeypatch.setattr(
        expansion_module,
        "initialise_bond_registry",
        initialise,
    )

    cfg = RunConfig(
        reactants=[
            ReactantCfg(smiles=item.smiles, add_hydrogens=False)
            for item in reactants
        ],
        bond=BondCfg(
            enabled=True,
            prune_with_calculator=True,
            prune_by_triple=True,
        ),
    )
    network = _builder(
        cfg,
        tmp_path,
        graph,
        calculator_resource=object(),
    ).prepare(reactants, sites)

    assert calls == [
        "enumerate",
        "stability",
        "triple",
        "rebuild",
        "registry",
    ]
    assert network.bond_sites == [second]
    assert second.iso_class == 0
    assert graph.graph["bond_reaction_sites"] == [second]


def test_network_builder_restores_checkpoint_without_discovery(tmp_path):
    graph = nx.Graph()
    state = SimpleNamespace(
        reactants=[object()],
        adsorbate_sites=[object()],
        diffusion_sites=[object()],
        bond_sites=[object()],
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="resume-test",
        resume_state=state,
    )
    builder = SpeciesNetworkBuilder(
        cfg=object(),
        identity=identity,
        graph=graph,
        calculator_resource=None,
        frozen_indices=None,
        thermo_runtime=ThermoRuntime(None, "", None),
    )

    network = builder.prepare(["ignored"], ["ignored"])

    assert network.reactants == state.reactants
    assert network.adsorbate_sites == state.adsorbate_sites
    assert network.initial_adsorbate_sites == state.adsorbate_sites
    assert network.diffusion_sites == state.diffusion_sites
    assert network.bond_sites == state.bond_sites


def test_output_factory_closes_reaction_writer_on_partial_failure(
    tmp_path,
    monkeypatch,
):
    import autokmc.workflow.runtime as runtime_module

    created = SimpleNamespace(reactions=None)

    class Reactions:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            created.reactions = self

        def close(self):
            self.closed = True

    def fail_trajectory(*_args, **_kwargs):
        raise RuntimeError("trajectory construction failed")

    monkeypatch.setattr(runtime_module, "ReactionWriter", Reactions)
    monkeypatch.setattr(runtime_module, "TrajectoryWriter", fail_trajectory)
    cfg = RunConfig()
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="output-test",
    )

    with pytest.raises(RuntimeError, match="trajectory construction failed"):
        create_output_sinks(cfg, identity, {})

    assert created.reactions.closed is True


def test_output_factory_passes_checkpoint_step_to_trajectory_resume(
    tmp_path,
    monkeypatch,
):
    import autokmc.workflow.runtime as runtime_module

    captured = {}

    class Trajectory:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def close(self):
            return

    monkeypatch.setattr(runtime_module, "TrajectoryWriter", Trajectory)
    cfg = RunConfig()
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="output-resume-test",
        resume_state=SimpleNamespace(step=17),
    )

    sinks = create_output_sinks(cfg, identity, {})
    sinks.close()

    assert captured["append"] is True
    assert captured["resume_checkpoint_step"] == 17


def test_compact_checkpoint_history_is_restored_from_shared_event_recovery(
    tmp_path,
):
    event = {
        "step": 1,
        "time_s": 0.25,
        "kind": "adsorption",
        "reactant_smiles": "[O]",
        "iso_class": 2,
        "member_index": 3,
        "lateral_class": 4,
        "rate_hz": 5.0,
        "delta_e_ev": -0.2,
        "barrier_ev": 0.1,
    }
    recovery = EventLogRecovery()
    recovery.add(event)
    checkpoint = SimpleNamespace(
        step=1,
        time_s=0.25,
        history=[],
        reaction_counts={"adsorption": 1},
        rng_state={"kind": "python", "state": None},
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="resume-history-test",
        resume_state=checkpoint,
        event_commit=EventLogCommit(1, 100, recovery=recovery),
    )

    resume = resolve_kmc_resume(identity)

    assert resume.history == [
        (1, 0.25, "adsorption", 2, 3, 4, -0.2, 0.1, 5.0)
    ]


def test_saved_compact_checkpoint_resumes_history_from_committed_events(
    tmp_path,
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    event = {
        "step": 1,
        "time_s": 0.25,
        "kind": "adsorption",
        "reactant_smiles": "[O]",
        "iso_class": 2,
        "member_index": 3,
        "lateral_class": 4,
        "rate_hz": 5.0,
        "delta_e_ev": -0.2,
        "barrier_ev": 0.1,
        "run_id": "compact-resume-test",
    }
    encoded = (json.dumps(event) + "\n").encode("utf-8")
    (output_dir / "events.jsonl").write_bytes(encoded)
    state = make_checkpoint_state(
        step=1,
        time_s=0.25,
        graph=nx.Graph(),
        adsorbate_sites=[],
        reaction_counts={"adsorption": 1},
        committed_event_count=1,
        committed_event_offset=len(encoded),
        metadata={"run_id": "compact-resume-test"},
    )
    assert state.history == []
    checkpoint_path = save_checkpoint(tmp_path / "compact.pkl", state)
    cfg = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        checkpoint=CheckpointCfg(resume_from=str(checkpoint_path)),
    )

    identity = resolve_run_identity(cfg)
    resume = resolve_kmc_resume(identity)

    assert resume.history == [
        (1, 0.25, "adsorption", 2, 3, 4, -0.2, 0.1, 5.0)
    ]
    assert identity.event_commit is not None
    assert identity.event_commit.recovery is not None
    assert identity.event_commit.recovery.summary.n == 1


def test_missing_legacy_log_is_seeded_for_two_hop_compact_resume(tmp_path):
    output_dir = tmp_path / "run"
    legacy_history = [
        (1, 0.10, "adsorption", 0, 1, 2, -0.2, 0.1, 3.0),
        (2, 0.25, "desorption", 0, 1, 2, 0.2, 0.2, 4.0),
    ]
    legacy_state = make_checkpoint_state(
        step=2,
        time_s=0.25,
        graph=nx.Graph(),
        adsorbate_sites=[],
        history=legacy_history,
        reaction_counts={"adsorption": 1, "desorption": 1},
        metadata={"run_id": "two-hop-resume"},
    )
    legacy_state.schema_version = "2"
    legacy_checkpoint = save_checkpoint(tmp_path / "legacy.pkl", legacy_state)
    first_cfg = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        checkpoint=CheckpointCfg(resume_from=str(legacy_checkpoint)),
    )

    first_identity = resolve_run_identity(first_cfg)
    first_resume = resolve_kmc_resume(first_identity)

    assert isinstance(first_resume.history, EventHistory)
    assert first_resume.history.in_memory_count == 0
    assert list(first_resume.history) == legacy_history
    event_path = output_dir / "events.jsonl"
    seeded_rows = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in seeded_rows] == [1, 2]
    assert all(row["legacy_history_recovered"] is True for row in seeded_rows)

    third_event = {
        "step": 3,
        "time_s": 0.40,
        "kind": "adsorption",
        "reactant_smiles": "[O]",
        "iso_class": 3,
        "member_index": 4,
        "lateral_class": 5,
        "rate_hz": 6.0,
        "delta_e_ev": -0.3,
        "barrier_ev": 0.15,
        "run_id": "two-hop-resume",
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(third_event) + "\n")
    compact_state = make_checkpoint_state(
        step=3,
        time_s=0.40,
        graph=nx.Graph(),
        adsorbate_sites=[],
        reaction_counts={"adsorption": 2, "desorption": 1},
        committed_event_count=3,
        committed_event_offset=event_path.stat().st_size,
        metadata={"run_id": "two-hop-resume"},
    )
    assert compact_state.history == []
    compact_checkpoint = save_checkpoint(tmp_path / "compact-v4.pkl", compact_state)
    second_cfg = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        checkpoint=CheckpointCfg(resume_from=str(compact_checkpoint)),
    )

    second_identity = resolve_run_identity(second_cfg)
    second_resume = resolve_kmc_resume(second_identity)

    assert isinstance(second_resume.history, EventHistory)
    assert second_resume.history.in_memory_count == 0
    assert list(second_resume.history) == legacy_history + [
        (3, 0.40, "adsorption", 3, 4, 5, -0.3, 0.15, 6.0)
    ]


def _resume_checkpoint(tmp_path: Path, cfg: RunConfig) -> Path:
    state = make_checkpoint_state(
        step=0,
        time_s=0.0,
        graph=nx.Graph(),
        adsorbate_sites=[],
        committed_event_count=0,
        committed_event_offset=0,
        metadata={
            "run_id": "resume-contract-test",
            "resume_contract": make_resume_contract(cfg),
        },
    )
    return save_checkpoint(tmp_path / "checkpoint.pkl", state)


def test_resume_contract_allows_only_documented_operational_changes(tmp_path):
    output_dir = tmp_path / "run"
    original = RunConfig(
        output=OutputCfg(dir=str(output_dir), log_level="INFO"),
        kmc=KMCCfg(n_steps=10, log_every=1),
        checkpoint=CheckpointCfg(
            enabled=True,
            path=str(output_dir / "old.pkl"),
            every_n_steps=5,
        ),
    )
    checkpoint = _resume_checkpoint(tmp_path, original)
    resumed = RunConfig(
        output=OutputCfg(dir=str(output_dir), log_level="DEBUG"),
        kmc=KMCCfg(n_steps=500, log_every=25),
        checkpoint=CheckpointCfg(
            enabled=False,
            path=str(output_dir / "new.pkl"),
            every_n_steps=100,
            resume_from=str(checkpoint),
        ),
    )

    identity = resolve_run_identity(resumed)

    assert identity.run_id == "resume-contract-test"
    assert identity.resume_state is not None


def test_resume_contract_rejects_temperature_or_pressure_changes(tmp_path):
    output_dir = tmp_path / "run"
    original = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        kmc=KMCCfg(temperature_k=500.0),
        free_energy=FreeEnergyCfg(pressure_bar=1.0),
        checkpoint=CheckpointCfg(enabled=True),
    )
    checkpoint = _resume_checkpoint(tmp_path, original)
    resumed = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        kmc=KMCCfg(temperature_k=550.0),
        free_energy=FreeEnergyCfg(pressure_bar=0.5),
        checkpoint=CheckpointCfg(enabled=True, resume_from=str(checkpoint)),
    )
    events_path = output_dir / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    crash_tail = json.dumps({"step": 1, "run_id": "resume-contract-test"}) + "\n"
    events_path.write_text(crash_tail)

    with pytest.raises(ValueError, match=r"kmc.temperature_k.*free_energy.pressure_bar|free_energy.pressure_bar.*kmc.temperature_k"):
        resolve_run_identity(resumed)
    assert events_path.read_text() == crash_tail


@pytest.mark.parametrize(
    ("section", "field", "value", "changed_path"),
    [
        ("calculator", "kwargs", {"charge": 1}, "calculator.kwargs.charge"),
        (
            "free_energy",
            "vibration_displacement",
            0.02,
            "free_energy.vibration_displacement",
        ),
        ("diffusion", "n_images", 11, "diffusion.n_images"),
        ("bond", "neb_spring_k", 0.25, "bond.neb_spring_k"),
        ("kmc", "random_seed", 912, "kmc.random_seed"),
    ],
)
def test_resume_contract_rejects_other_scientific_setting_changes(
    section, field, value, changed_path
):
    original = RunConfig(
        reactants=[ReactantCfg(smiles="[O]", partial_pressure_bar=0.5)],
        calculator=CalculatorCfg(kwargs={"charge": 0}),
        diffusion=DiffusionCfg(enabled=True, n_images=7),
        bond=BondCfg(enabled=True, neb_spring_k=0.1),
        free_energy=FreeEnergyCfg(vibration_displacement=0.01),
        kmc=KMCCfg(random_seed=7),
    )
    contract = make_resume_contract(original)
    changed = deepcopy(original)
    setattr(getattr(changed, section), field, value)

    with pytest.raises(ValueError, match=changed_path.replace(".", r"\.")):
        verify_resume_contract(changed, contract)


def test_resume_contract_rejects_feed_changes():
    original = RunConfig(
        reactants=[ReactantCfg(smiles="[O]", partial_pressure_bar=0.5)],
    )
    changed = deepcopy(original)
    changed.reactants[0].partial_pressure_bar = 0.75

    with pytest.raises(ValueError, match=r"reactants\[0\]\.partial_pressure_bar"):
        verify_resume_contract(changed, make_resume_contract(original))


def test_resume_contract_hashes_local_model_contents_and_not_its_path(tmp_path):
    first_model = tmp_path / "first-model.bin"
    copied_model = tmp_path / "copied-model.bin"
    first_model.write_bytes(b"model-version-a")
    copied_model.write_bytes(first_model.read_bytes())
    original = RunConfig(
        calculator=CalculatorCfg(
            factory="example.calculator.from_checkpoint",
            factory_kwargs={"name_or_path": str(first_model)},
        ),
    )
    contract = make_resume_contract(original)

    copied = deepcopy(original)
    copied.calculator.factory_kwargs["name_or_path"] = str(copied_model)
    verify_resume_contract(copied, contract)

    first_model.write_bytes(b"model-version-b")  # same path and byte length
    with pytest.raises(
        ValueError,
        match=r"calculator\.factory_kwargs\.name_or_path\.sha256",
    ):
        verify_resume_contract(original, contract)


def test_resume_contract_hashes_file_structure_contents_not_its_path(tmp_path):
    first_structure = tmp_path / "first-structure.xyz"
    copied_structure = tmp_path / "copied-structure.xyz"
    first_structure.write_bytes(b"structure-version-a")
    copied_structure.write_bytes(first_structure.read_bytes())
    original = RunConfig(
        structure=StructureCfg(kind="file", path=str(first_structure)),
    )
    contract = make_resume_contract(original)

    copied = deepcopy(original)
    copied.structure.path = str(copied_structure)
    verify_resume_contract(copied, contract)

    first_structure.write_bytes(b"structure-version-b")
    with pytest.raises(ValueError, match=r"structure\.path\.sha256"):
        verify_resume_contract(original, contract)


def test_resume_contract_hashes_model_directories(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"first")
    cfg = RunConfig(
        calculator=CalculatorCfg(
            factory="example.calculator.from_checkpoint",
            # Deliberately use a backend-specific field name: every existing
            # artifact path nested under calculator kwargs is content-hashed.
            factory_kwargs={"custom_model_bundle": str(model_dir)},
        ),
    )
    contract = make_resume_contract(cfg)
    weights.write_bytes(b"other")

    with pytest.raises(
        ValueError,
        match=r"calculator\.factory_kwargs\.custom_model_bundle\.sha256",
    ):
        verify_resume_contract(cfg, contract)


def test_resume_contract_records_nested_calculator_package_versions(monkeypatch):
    monkeypatch.setattr(
        resume_contract_module,
        "_installed_package_distributions",
        lambda: {
            "outer_backend": ["outer-distribution"],
            "nested_backend": ["nested-distribution"],
            "imported_model": ["model-distribution"],
        },
    )
    versions = {
        "outer-distribution": "1.2.3",
        "nested-distribution": "4.5.6",
        "model-distribution": "7.8.9",
    }
    monkeypatch.setattr(
        resume_contract_module.importlib_metadata,
        "version",
        versions.__getitem__,
    )
    cfg = RunConfig(
        calculator=CalculatorCfg(
            factory="outer_backend.build",
            factory_kwargs={
                "predictor": {
                    "factory": "nested_backend:create",
                    "factory_kwargs": {
                        "model": {
                            "import_path": "imported_model.Model",
                            "kwargs": {},
                        },
                    },
                },
            },
        ),
    )

    payload = scientific_config_payload(cfg)

    assert payload["_software"]["calculator_distributions"] == versions


def test_resolve_run_identity_removes_events_after_checkpoint_commit(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    original = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        checkpoint=CheckpointCfg(enabled=True),
    )
    first = json.dumps({"step": 1, "run_id": "resume-contract-test"}) + "\n"
    tail = json.dumps({"step": 2, "run_id": "resume-contract-test"}) + "\n"
    events_path = output_dir / "events.jsonl"
    events_path.write_text(first + tail)
    state = make_checkpoint_state(
        step=1,
        time_s=0.1,
        graph=nx.Graph(),
        adsorbate_sites=[],
        committed_event_count=1,
        committed_event_offset=len(first.encode("utf-8")),
        metadata={
            "run_id": "resume-contract-test",
            "resume_contract": make_resume_contract(original),
        },
    )
    checkpoint = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    resumed = RunConfig(
        output=OutputCfg(dir=str(output_dir)),
        checkpoint=CheckpointCfg(enabled=True, resume_from=str(checkpoint)),
    )

    resolve_run_identity(resumed)

    assert events_path.read_text() == first
