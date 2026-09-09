"""EXTXYZ columns remain complete for heterogeneous per-atom metadata."""

import io

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io.extxyz import key_val_str_to_dict

from autokmc.core.atom_metadata import apply_atom_metadata
from autokmc.io.extxyz import (
    STRING_ARRAYS_KEY, prepare_extxyz_atoms, read_atoms, write_extxyz,
)
from autokmc.io.persistence import _atomic_extxyz as write_reaction_atoms
from autokmc.io.calculation_cache import _atomic_extxyz as write_cache_atoms
from autokmc.io.trajectory import TrajectoryWriter
from autokmc.structure.loading import load_structure_file


def _slab_with_adsorbates():
    atoms = Atoms("PdHH", positions=[[0, 0, 0], [0, 0, 2], [0.7, 0, 2]],
                  cell=[10, 10, 20], pbc=[True, True, False])
    # Match pymatgen slab metadata propagated to two newly added adsorbates.
    apply_atom_metadata(atoms, [
        {"element": "Pd", "atom_arrays": {"bulk_wyckoff": "a", "bulk_equivalent": 0}},
        {"element": "H"}, {"element": "H"},
    ])
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_masses([106.42, 2.014, 1.008])
    # Place a numeric array after the missing string column to catch shifting
    # or silent truncation as well as outright read failures.
    atoms.new_array("later", np.array([13, 17, 19]))
    return atoms


def _assert_columns(path, n_frames):
    lines = path.read_text().splitlines()
    cursor = 0
    for _ in range(n_frames):
        count = int(lines[cursor])
        properties = key_val_str_to_dict(lines[cursor + 1])["Properties"].split(":")
        n_columns = sum(int(value) for value in properties[2::3])
        rows = lines[cursor + 2:cursor + 2 + count]
        assert len(rows) == count
        assert all(len(row.split()) == n_columns for row in rows)
        cursor += count + 2
    assert cursor == len(lines)


@pytest.mark.parametrize("writer", [write_extxyz, write_reaction_atoms, write_cache_atoms])
def test_empty_adsorbate_metadata_has_full_columns_and_preserves_results(tmp_path, writer):
    initial = _slab_with_adsorbates()
    optimized = initial.copy()
    optimized.calc = SinglePointCalculator(optimized, energy=-438.7340, forces=np.ones((3, 3)))
    path = tmp_path / "neb_path.extxyz"
    writer(path, [initial, optimized])
    _assert_columns(path, 2)
    public = ase_read(path, index=":")
    assert public[0].arrays["bulk_wyckoff"].tolist() == ["a", "_", "_"]
    assert public[0].arrays["later"].tolist() == [13, 17, 19]
    assert public[0].calc is None
    assert public[1].get_potential_energy() == -438.7340
    np.testing.assert_array_equal(public[1].get_forces(apply_constraint=False), np.ones((3, 3)))
    np.testing.assert_array_equal(public[0].get_masses(), initial.get_masses())
    assert public[0].constraints[0].get_indices().tolist() == [0]
    np.testing.assert_array_equal(public[0].positions, initial.positions)
    np.testing.assert_array_equal(public[0].cell, initial.cell)
    np.testing.assert_array_equal(public[0].pbc, initial.pbc)
    restored = read_atoms(path, index=":")
    for atoms in restored:
        for name in initial.arrays:
            np.testing.assert_array_equal(atoms.arrays[name], initial.arrays[name])
        assert STRING_ARRAYS_KEY not in atoms.info
    assert initial.arrays["bulk_wyckoff"].tolist() == ["a", "", ""]
    assert STRING_ARRAYS_KEY not in initial.info


@pytest.mark.parametrize("values", [
    np.array(["", "two words", "\t\n"]),
    np.array(["a\\b", 'quoted"value', "'single'"]),
    np.array(["_%20", "_", "x\x00y"]),
    np.array([b"a", b"", b"two words"]),
    np.array([b"a", b"\xff", b" "]),
    np.array(["a", None, 0], dtype=object),
    np.array([["a", ""], ["", "two words"], ["b", "c"]]),
    np.array([[""], ["a"], ["b"]]),
])
def test_string_arrays_round_trip_without_splitting_or_losing_values(tmp_path, values):
    atoms = _slab_with_adsorbates()
    atoms.new_array("labels", values)
    path = tmp_path / "labels.extxyz"
    write_extxyz(path, atoms)
    _assert_columns(path, 1)
    public = ase_read(path)
    assert public.arrays["later"].tolist() == [13, 17, 19]
    restored = read_atoms(path)
    assert restored.arrays["labels"].dtype == values.dtype
    np.testing.assert_array_equal(restored.arrays["labels"], values)
    # Rewriting an ASE-read snapshot must not double-escape or lose originals.
    write_extxyz(path, public)
    np.testing.assert_array_equal(read_atoms(path).arrays["labels"], values)


def test_string_conversion_does_not_invoke_a_live_calculator():
    atoms = _slab_with_adsorbates()

    class LiveCalculator:
        results = {}

        def get_potential_energy(self, *_args):
            pytest.fail("writer invoked a live calculator")

        def get_forces(self, *_args):
            pytest.fail("writer invoked a live calculator")

    atoms.calc = LiveCalculator()
    prepared = prepare_extxyz_atoms(atoms)
    assert prepared.calc is None
    write_extxyz(io.StringIO(), atoms)


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("bad_array", ["shape", "dtype", "name", "name_bracket"])
def test_bad_later_frame_cannot_truncate_or_partially_append_a_file(tmp_path, append, bad_array):
    good = _slab_with_adsorbates()
    bad = good.copy()
    if bad_array == "shape":
        bad.new_array("tensor", np.zeros((3, 2, 2)))
    elif bad_array == "dtype":
        bad.new_array("complex", np.zeros(3, dtype=complex))
    elif bad_array == "name":
        bad.new_array("two words", np.zeros(3))
    else:
        bad.new_array("broken[", np.zeros(3))
    path = tmp_path / "existing.extxyz"
    write_extxyz(path, good)
    previous = path.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        write_extxyz(path, [good, bad], append=append)
    assert path.read_bytes() == previous
    assert len(read_atoms(path, index=":")) == 1


def test_trajectory_strings_survive_append_resume_and_final_marker(tmp_path):
    atoms = _slab_with_adsorbates()
    path = tmp_path / "kmc.extxyz"
    writer = TrajectoryWriter(path, dump_every=1)
    writer.write(atoms, step=0)
    writer.write(atoms, step=1)
    writer.close()
    writer = TrajectoryWriter(path, dump_every=1, append=True)
    writer.write(atoms, step=2)
    writer.write_final_snapshot(lambda: atoms, step=2)
    writer.close()
    _assert_columns(path, 3)
    frames = read_atoms(path, index=":")
    assert [frame.info["kmc_step"] for frame in frames] == [0, 1, 2]
    assert frames[-1].info["frame_kind"] == "final"
    assert all(frame.arrays["bulk_wyckoff"].tolist() == ["a", "", ""] for frame in frames)


def test_structure_input_restores_original_per_atom_metadata(tmp_path):
    atoms = _slab_with_adsorbates()
    path = tmp_path / "input.extxyz"
    write_extxyz(path, atoms)
    loaded, _ = load_structure_file(path)
    assert loaded.arrays["bulk_wyckoff"].tolist() == ["a", "", ""]
    assert loaded.arrays["later"].tolist() == [13, 17, 19]
