"""Tests for TrajectoryWriter cadence (extended-XYZ format)."""

from __future__ import annotations

import pytest
from ase.io import read as ase_read
from ase.io import write as ase_write

import ogkmc.io.trajectory as trajectory_module
from ogkmc.io.schemas import (
    TRAJECTORY_ARTIFACT_TYPE,
    TRAJECTORY_METADATA_SCHEMA_VERSION,
)
from ogkmc.io.trajectory import TrajectoryWriter


def test_trajectory_writer_respects_dump_every(tmp_path, tiny_atoms):
    p = tmp_path / "kmc.extxyz"
    w = TrajectoryWriter(p, dump_every=5)
    assert w.enabled
    # Step 0 always written.
    assert w.maybe_write(tiny_atoms, step=0) is True
    # Steps 1..4 should not write.
    for s in range(1, 5):
        assert w.maybe_write(tiny_atoms, step=s) is False
    # Step 5 writes, 6 does not, 10 writes.
    assert w.maybe_write(tiny_atoms, step=5)  is True
    assert w.maybe_write(tiny_atoms, step=6)  is False
    assert w.maybe_write(tiny_atoms, step=10) is True
    w.close()

    frames = ase_read(p, index=":")
    assert len(frames) == 3
    assert frames[0].get_chemical_symbols() == tiny_atoms.get_chemical_symbols()
    # Each frame stamps its KMC step into atoms.info.
    assert frames[0].info.get("kmc_step") == 0
    assert frames[1].info.get("kmc_step") == 5
    assert frames[2].info.get("kmc_step") == 10


def test_trajectory_writer_disabled(tmp_path, tiny_atoms):
    p = tmp_path / "kmc.extxyz"
    w = TrajectoryWriter(p, dump_every=0)
    assert not w.enabled
    assert w.maybe_write(tiny_atoms, step=0) is False
    assert w.maybe_write(tiny_atoms, step=10) is False
    w.close()
    assert not p.exists()


def test_trajectory_writer_append_preserves_existing_frames(tmp_path, tiny_atoms):
    path = tmp_path / "kmc.extxyz"
    first = TrajectoryWriter(path, dump_every=1)
    first.maybe_write(tiny_atoms, step=0)
    first.close()

    resumed = TrajectoryWriter(path, dump_every=1, append=True)
    resumed.maybe_write(tiny_atoms, step=1)
    resumed.close()

    frames = ase_read(path, index=":")
    assert [frame.info["kmc_step"] for frame in frames] == [0, 1]


def test_final_snapshot_is_written_off_cadence_with_versioned_metadata(
    tmp_path,
    tiny_atoms,
):
    path = tmp_path / "kmc.extxyz"
    writer = TrajectoryWriter(path, dump_every=5)
    writer.maybe_write(
        tiny_atoms,
        step=0,
        metadata={
            "run_id": "run-trajectory",
            "simulated_time_s": 0.0,
            "frame_kind": "initial",
            "segment_start_step": 0,
        },
    )
    assert writer.maybe_write(tiny_atoms, step=3) is False
    assert writer.write_final_snapshot(
        lambda: tiny_atoms.copy(),
        step=3,
        metadata={
            "run_id": "run-trajectory",
            "simulated_time_s": 2.5,
            "event_id": "event-final",
            "segment_start_step": 0,
        },
    )
    assert not writer.write_final_snapshot(
        lambda: tiny_atoms.copy(),
        step=3,
        metadata={
            "run_id": "run-trajectory",
            "simulated_time_s": 2.5,
            "event_id": "event-final",
            "segment_start_step": 0,
        },
    )
    writer.close()

    frames = ase_read(path, index=":")
    assert [frame.info["kmc_step"] for frame in frames] == [0, 3]
    final = frames[-1].info
    assert final["artifact_type"] == TRAJECTORY_ARTIFACT_TYPE
    assert (
        str(final["trajectory_schema_version"])
        == TRAJECTORY_METADATA_SCHEMA_VERSION
    )
    assert final["run_id"] == "run-trajectory"
    assert final["simulated_time_s"] == 2.5
    assert final["event_id"] == "event-final"
    assert final["frame_kind"] == "final"
    assert final["segment_start_step"] == 0


def test_zero_event_final_snapshot_upgrades_initial_frame_without_duplicate(
    tmp_path,
    tiny_atoms,
):
    path = tmp_path / "kmc.extxyz"
    writer = TrajectoryWriter(path, dump_every=5)
    writer.maybe_write(
        tiny_atoms,
        step=0,
        metadata={
            "run_id": "run-zero",
            "simulated_time_s": 0.0,
            "frame_kind": "initial",
            "segment_start_step": 0,
        },
    )
    factory_calls = []
    assert writer.write_final_snapshot(
        lambda: factory_calls.append(True) or tiny_atoms.copy(),
        step=0,
        metadata={
            "run_id": "run-zero",
            "simulated_time_s": 0.0,
            "segment_start_step": 0,
        },
    )
    writer.close()

    frames = ase_read(path, index=":")
    assert len(frames) == 1
    assert frames[0].info["kmc_step"] == 0
    assert frames[0].info["frame_kind"] == "final"
    assert frames[0].info["run_id"] == "run-zero"
    assert factory_calls == []
    assert writer.n_frames == 1


def test_trajectory_checkpoint_syncs_file_then_parent(
    tmp_path,
    tiny_atoms,
    monkeypatch,
):
    path = tmp_path / "kmc.extxyz"
    writer = TrajectoryWriter(path, dump_every=1)
    writer.write(tiny_atoms, step=0)
    calls = []
    monkeypatch.setattr(
        trajectory_module.os,
        "fsync",
        lambda _descriptor: calls.append("file"),
    )
    monkeypatch.setattr(
        trajectory_module,
        "fsync_directory",
        lambda directory: calls.append(("directory", directory)),
    )

    writer.sync_for_checkpoint()

    # The constructor atomically published the file name already; appending a
    # frame changes file contents/length but not the parent directory entry.
    assert calls == ["file"]


def test_trajectory_resume_accepts_missing_legacy_file(tmp_path, tiny_atoms):
    path = tmp_path / "kmc.extxyz"

    resumed = TrajectoryWriter(
        path,
        dump_every=1,
        append=True,
        resume_checkpoint_step=3,
    )
    resumed.write(tiny_atoms, step=4)
    resumed.close()

    frames = ase_read(path, index=":")
    assert [frame.info["kmc_step"] for frame in frames] == [4]


def test_lazy_snapshot_skips_factory_and_does_not_copy_owned_atoms(
    tmp_path,
    tiny_atoms,
    monkeypatch,
):
    writer = TrajectoryWriter(tmp_path / "kmc.extxyz", dump_every=5)
    owned = tiny_atoms.copy()
    factory_calls = []
    written = []

    def factory():
        factory_calls.append(True)
        return owned

    monkeypatch.setattr(
        trajectory_module,
        "ase_write",
        lambda _path, atoms, **_kwargs: written.append(atoms),
    )

    assert writer.maybe_write_snapshot(factory, step=1) is False
    assert factory_calls == []
    assert writer.maybe_write_snapshot(factory, step=5) is True
    assert factory_calls == [True]
    assert written == [owned]
    assert written[0] is owned
    assert owned.info["kmc_step"] == 5


def test_trajectory_resume_removes_crash_tail_before_append(tmp_path, tiny_atoms):
    path = tmp_path / "kmc.extxyz"
    frames = []
    for step in (0, 2, 4):
        frame = tiny_atoms.copy()
        frame.info["kmc_step"] = step
        frames.append(frame)
    ase_write(path, frames, format="extxyz")

    resumed = TrajectoryWriter(
        path,
        dump_every=1,
        append=True,
        resume_checkpoint_step=2,
    )
    resumed.write(tiny_atoms, step=3)
    resumed.close()

    restored = ase_read(path, index=":")
    assert [frame.info["kmc_step"] for frame in restored] == [0, 2, 3]
    assert resumed.n_frames == 1


@pytest.mark.parametrize(
    ("step_value", "message"),
    [
        (None, "has no kmc_step"),
        ("not-a-step", "must be an integer"),
        (-1, "cannot be negative"),
    ],
)
def test_trajectory_resume_rejects_invalid_step_metadata(
    tmp_path,
    tiny_atoms,
    step_value,
    message,
):
    path = tmp_path / "kmc.extxyz"
    frame = tiny_atoms.copy()
    if step_value is not None:
        frame.info["kmc_step"] = step_value
    ase_write(path, frame, format="extxyz")
    original = path.read_bytes()

    with pytest.raises(ValueError, match=message):
        TrajectoryWriter(
            path,
            dump_every=1,
            append=True,
            resume_checkpoint_step=0,
        )

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((0, 2, 1), "not strictly increasing"),
        ((0, 3, 2), "follows an uncommitted crash-tail frame"),
    ],
)
def test_trajectory_resume_rejects_nonmonotonic_committed_prefix(
    tmp_path,
    tiny_atoms,
    steps,
    message,
):
    path = tmp_path / "kmc.extxyz"
    frames = []
    for step in steps:
        frame = tiny_atoms.copy()
        frame.info["kmc_step"] = step
        frames.append(frame)
    ase_write(path, frames, format="extxyz")
    original = path.read_bytes()

    with pytest.raises(ValueError, match=message):
        TrajectoryWriter(
            path,
            dump_every=1,
            append=True,
            resume_checkpoint_step=2,
        )

    assert path.read_bytes() == original
