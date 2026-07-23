"""Tests for TrajectoryWriter cadence (extended-XYZ format)."""

from __future__ import annotations

from ase.io import read as ase_read

from autokmc.io.trajectory import TrajectoryWriter


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
