"""Focused tests for typed KMC session boundaries."""

from __future__ import annotations

import networkx as nx
import numpy as np
from ase.io import read as ase_read

from autokmc.io.event_log import EventHistory
from autokmc.io.trajectory import TrajectoryWriter
from autokmc.kmc.index import _ReactionIndex
from autokmc.kmc.initialization import initialise_runtime, normalise_channels
from autokmc.kmc.models import (
    KMCChannels,
    KMCObservers,
    KMCResumeState,
    KMCRuntime,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.kmc.outputs import KMCOutputManager


def test_channel_normalisation_copies_inputs_and_resolves_cache_root():
    diffusion_sites = [object()]
    diffusion_kwargs = {
        "fmax": 0.1,
        "calculation_cache_root": "diffusion-cache",
        "free_energy_options": object(),
    }
    channels = KMCChannels(
        diffusion_sites=diffusion_sites,  # type: ignore[arg-type]
        diffusion_kwargs=diffusion_kwargs,
        bond_kwargs={"calculation_cache_root": "bond-cache", "n_images": 7},
    )

    normalised, thermochemistry = normalise_channels(
        channels,
        KMCThermochemistry(),
    )

    assert thermochemistry.calculation_cache_root == "diffusion-cache"
    assert normalised.diffusion_sites == diffusion_sites
    assert normalised.diffusion_sites is not diffusion_sites
    assert normalised.diffusion_kwargs == {"fmax": 0.1}
    assert normalised.bond_kwargs == {"n_images": 7}
    assert diffusion_kwargs["calculation_cache_root"] == "diffusion-cache"


def test_checkpoint_output_only_passes_force_for_forced_write():
    calls = []

    class CheckpointWriter:
        def maybe_write(self, **payload):
            calls.append(payload)

    graph = nx.Graph()
    system = KMCSystem(graph, [], None, {})
    channels = KMCChannels()
    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=_ReactionIndex([]),
        history=[],
        reaction_counts={},
        current_time_s=1.25,
        start_step=0,
    )
    outputs = KMCOutputManager(
        system,
        KMCSettings(temperature=500.0, n_steps=2),
        channels,
        KMCObservers(checkpoint_writer=CheckpointWriter()),
        runtime,
    )

    outputs.write_checkpoint(step=1)
    outputs.write_checkpoint(step=2, force=True)

    assert "force" not in calls[0]
    assert calls[1]["force"] is True
    assert calls[1]["time_s"] == 1.25
    assert calls[1]["rng_state"]["kind"] == "numpy"
    assert calls[1]["history"] == []


def test_checkpoint_syncs_event_prefix_before_serializing_state():
    calls = []

    class Trajectory:
        def sync_for_checkpoint(self):
            calls.append("trajectory_sync")

    class Reactions:
        def sync_for_checkpoint(self):
            calls.append("event_sync")
            return type("Commit", (), {"count": 7, "offset": 4096})()

    class CheckpointWriter:
        def should_write(self, *, step, force=False):
            return step == 4 or force

        def maybe_write(self, **payload):
            calls.append(payload)

    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=_ReactionIndex([]),
        history=[],
        reaction_counts={},
        current_time_s=1.25,
        start_step=0,
    )
    outputs = KMCOutputManager(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(temperature=500.0, n_steps=4),
        KMCChannels(),
        KMCObservers(
            reaction_writer=Reactions(),
            trajectory_writer=Trajectory(),
            checkpoint_writer=CheckpointWriter(),
        ),
        runtime,
    )

    outputs.write_checkpoint(step=3)
    outputs.write_checkpoint(step=4)

    assert calls[:2] == ["trajectory_sync", "event_sync"]
    assert calls[2]["committed_event_count"] == 7
    assert calls[2]["committed_event_offset"] == 4096
    assert "history" not in calls[2]


def test_output_manager_checks_trajectory_cadence_before_snapshot_creation():
    calls = []

    class Trajectory:
        def should_write(self, *, step):
            calls.append(("should_write", step))
            return False

        def maybe_write_snapshot(self, *_args, **_kwargs):
            raise AssertionError("skipped trajectory frame was materialised")

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
    outputs = KMCOutputManager(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(temperature=500.0, n_steps=1),
        KMCChannels(),
        KMCObservers(trajectory_writer=Trajectory()),
        runtime,
    )

    outputs.finish_step(step=1, reactions=[], invalid_diffusion_sites=[])

    assert calls == [("should_write", 1)]


def test_close_does_not_duplicate_a_checkpoint_already_written_at_final_step():
    writes = []

    class CheckpointWriter:
        def should_write(self, *, step, force=False):
            return force or step == 5

        def maybe_write(self, **payload):
            writes.append(payload)

    runtime = KMCRuntime(
        rng=np.random.default_rng(4),
        gas_energies={},
        gas_free_energies={},
        partial_pressures={},
        reaction_index=_ReactionIndex([]),
        history=[],
        reaction_counts={},
        current_time_s=1.0,
        start_step=0,
        steps_executed=5,
    )
    outputs = KMCOutputManager(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(temperature=500.0, n_steps=5),
        KMCChannels(),
        KMCObservers(checkpoint_writer=CheckpointWriter()),
        runtime,
    )

    outputs.write_checkpoint(step=5)
    outputs.close()

    assert len(writes) == 1
    assert writes[0]["step"] == 5


def test_zero_event_output_manager_marks_the_only_trajectory_frame_final(
    tmp_path,
    synth_graph,
):
    synth_graph.graph["run_id"] = "run-zero-event"
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
        steps_executed=0,
    )
    path = tmp_path / "kmc.extxyz"
    trajectory = TrajectoryWriter(path, dump_every=10)
    outputs = KMCOutputManager(
        KMCSystem(synth_graph, [], None, {}),
        KMCSettings(temperature=500.0, n_steps=0),
        KMCChannels(),
        KMCObservers(trajectory_writer=trajectory),
        runtime,
    )

    outputs.initialise()
    outputs.close()

    frames = ase_read(path, index=":")
    assert len(frames) == 1
    assert frames[0].info["kmc_step"] == 0
    assert frames[0].info["frame_kind"] == "final"
    assert frames[0].info["run_id"] == "run-zero-event"
    assert frames[0].info["simulated_time_s"] == 0.0
    assert frames[0].info["segment_start_step"] == 0


def test_initialise_runtime_keeps_lazy_resume_history_unmaterialized(tmp_path):
    history = EventHistory(
        tmp_path / "events.jsonl",
        committed_count=0,
        committed_offset=0,
    )

    runtime = initialise_runtime(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(temperature=500.0, n_steps=0, verbose=False),
        KMCChannels(),
        KMCThermochemistry(),
        KMCResumeState(history=history),
        rng=4,
        compute_adsorption=lambda *_args, **_kwargs: [],
    )

    assert runtime.history is history
    assert history.in_memory_count == 0
