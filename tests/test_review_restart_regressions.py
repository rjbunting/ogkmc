"""Regression coverage for rejected and interrupted resumes."""

from dataclasses import asdict
import json

from ase import Atoms
import networkx as nx
import pytest

from ogkmc.cli.pipeline import run_from_config
from ogkmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from ogkmc.io.config import RunConfig
from ogkmc.io.run_manifest import begin_run_manifest, finish_run_manifest
from ogkmc.io.trajectory import TrajectoryWriter, reconcile_trajectory


def test_resume_discards_incomplete_uncommitted_trajectory_frame(tmp_path):
    path = tmp_path / "trajectory.extxyz"
    writer = TrajectoryWriter(path)
    writer.write(Atoms("H", positions=[[0, 0, 0]]), step=1)
    writer.close()
    with path.open("a") as handle:
        handle.write("2\nProperties=species:S:1:pos:R:3 kmc_step=2\nH 0 0 0\n")
    resumed = TrajectoryWriter(path, append=True, resume_checkpoint_step=1)
    assert resumed.last_step == 1
    resumed.close()


def test_rejected_resume_preserves_completed_manifest(tmp_path):
    cfg = RunConfig()
    cfg.output.dir = str(tmp_path)
    cfg.bond.enabled = False
    cfg.free_energy.enabled = False
    cfg.checkpoint.resume_from = str(tmp_path / "checkpoint.pkl")
    manifest = tmp_path / cfg.output.run_manifest_filename
    begin_run_manifest(
        manifest, run_id="original-run", config_path=None, resolved_config=asdict(cfg)
    )
    finish_run_manifest(manifest, final_step=5, final_time_s=2, steps_executed=5, wall_time_s=30)
    before = json.loads(manifest.read_text())
    state = make_checkpoint_state(
        step=5, time_s=2, graph=nx.Graph(), adsorbate_sites=[], metadata={"run_id": "another-run"}
    )
    save_checkpoint(cfg.checkpoint.resume_from, state)
    try:
        run_from_config(cfg)
    except ValueError as exc:
        assert "run_id does not match" in str(exc)
    else:
        raise AssertionError("resume should have been rejected")
    assert json.loads(manifest.read_text()) == before


@pytest.mark.parametrize(
    "tail",
    [
        b"2",
        b"2\nProperties=spec",
        b"2\nProperties=species:S:1:pos:R:3 kmc_step=8\nH 0 0 0\n",
        b"\xff",
    ],
)
@pytest.mark.parametrize("checkpoint_step", [1, 7])
def test_checkpoint_byte_prefix_recovers_any_interrupted_append(tmp_path, tail, checkpoint_step):
    path = tmp_path / "trajectory.extxyz"
    writer = TrajectoryWriter(path)
    writer.write(Atoms("H"), step=1)
    writer.sync_for_checkpoint()
    committed = writer.committed_offset
    writer.close()
    before = path.read_bytes()
    path.write_bytes(before + tail)
    resumed = TrajectoryWriter(
        path, append=True, resume_checkpoint_step=checkpoint_step, resume_committed_offset=committed
    )
    assert resumed.last_step == 1
    assert path.read_bytes() == before
    resumed.close()


@pytest.mark.parametrize("exact", [False, True])
def test_recovery_preserves_file_when_committed_frame_is_corrupt(tmp_path, exact):
    path = tmp_path / "trajectory.extxyz"
    data = b"2\nProperties=species:S:1:pos:R:3 kmc_step=1\nH 0 0 0\n"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid committed"):
        reconcile_trajectory(path, checkpoint_step=1, committed_offset=len(data) if exact else None)
    assert path.read_bytes() == data


def test_checkpoint_recovery_rejects_missing_committed_bytes(tmp_path):
    path = tmp_path / "trajectory.extxyz"
    with pytest.raises(ValueError, match="committed trajectory is missing"):
        reconcile_trajectory(path, checkpoint_step=1, committed_offset=10)
    path.write_bytes(b"1\n")
    with pytest.raises(ValueError, match="shorter than its committed prefix"):
        reconcile_trajectory(path, checkpoint_step=1, committed_offset=10)
    assert path.read_bytes() == b"1\n"


@pytest.mark.parametrize("append_before_final", [False, True])
def test_final_frame_upgrade_keeps_previous_checkpoint_offset_valid(tmp_path, append_before_final):
    path = tmp_path / "trajectory.extxyz"
    writer = TrajectoryWriter(path)
    writer.write(Atoms("H"), step=0)
    writer.write(Atoms("H"), step=1, metadata={"frame_kind": "periodic"})
    writer.sync_for_checkpoint()
    committed = writer.committed_offset
    original = path.read_bytes()
    if append_before_final:
        writer.write(Atoms("H"), step=2)
    writer.write_final_snapshot(lambda: Atoms("H"), step=2 if append_before_final else 1)
    writer.close()
    if append_before_final:
        assert path.read_bytes()[: len(original)] == original
    else:
        assert path.stat().st_size == committed
    resumed = TrajectoryWriter(
        path, append=True, resume_checkpoint_step=1, resume_committed_offset=committed
    )
    assert resumed.last_step == 1
    assert path.stat().st_size == committed
    resumed.close()
