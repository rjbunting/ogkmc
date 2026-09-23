"""Characterization tests for file-backed catalyst structures."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixBondLength
from ase.io import write

from autokmc.structure.loading import (
    load_structure_file,
    resolve_frozen_indices,
)
from autokmc.io.config import RunConfig, StructureCfg
from autokmc.workflow.models import (
    PreparedCalculator,
    PreparedStructure,
    RunIdentity,
)
from autokmc.workflow.stages import prepare_material_graph, prepare_structure


def test_extxyz_frame_loading_is_config_relative_and_records_source(tmp_path):
    config_dir = tmp_path / "config"
    structure_dir = config_dir / "structures"
    structure_dir.mkdir(parents=True)
    config_path = config_dir / "run.yaml"
    config_path.write_text("structure: {}\n", encoding="utf-8")
    structure_path = structure_dir / "catalyst.extxyz"

    first = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[4.0] * 3)
    second = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[6.0] * 3,
    )
    second.calc = SinglePointCalculator(second, energy=-1.25)
    write(structure_path, [first, second], format="extxyz")

    atoms, source = load_structure_file(
        "structures/catalyst.extxyz",
        format="extxyz",
        index=1,
        frozen_indices=[1],
        config_path=config_path,
    )

    assert atoms.get_chemical_formula() == "Pt2"
    assert atoms.calc is None
    assert atoms.info["frozen_indices"] == [1]
    assert source == {
        "kind": "file",
        "path": str(structure_path.resolve()),
        "format": "extxyz",
        "index": 1,
        "sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        "size_bytes": structure_path.stat().st_size,
        "chemical_formula": "Pt2",
        "frozen_indices": [1],
        "frozen_count": 1,
    }


def test_frozen_indices_union_info_and_constraints_then_allow_override():
    atoms = Atoms(
        "Pt3",
        positions=np.zeros((3, 3)),
        info={"frozen_indices": [0]},
        constraint=FixAtoms(indices=[1]),
    )

    assert resolve_frozen_indices(atoms) == [0, 1]
    assert atoms.info["frozen_indices"] == [0, 1]

    # An explicit empty list intentionally clears both metadata sources.
    assert resolve_frozen_indices(atoms, frozen_indices=[]) == []
    assert atoms.info["frozen_indices"] == []


def test_partial_ase_constraint_is_not_promoted_to_fully_frozen_atoms():
    atoms = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        info={"frozen_indices": [2]},
        constraint=FixBondLength(0, 1),
    )

    assert resolve_frozen_indices(atoms) == [2]
    assert atoms.info["frozen_indices"] == [2]


@pytest.mark.parametrize("frozen_indices", [[-1], [2]])
def test_frozen_indices_reject_out_of_range_values(frozen_indices):
    atoms = Atoms("Pt2", positions=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="valid range 0..1"):
        resolve_frozen_indices(atoms, frozen_indices=frozen_indices)


def test_file_workflow_bypasses_builders_and_structure_calculator(
    tmp_path,
    monkeypatch,
):
    import autokmc.structure as structure_module
    import autokmc.workflow.stages as stages_module

    structure_path = tmp_path / "input.extxyz"
    write(
        structure_path,
        Atoms("Pt", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("file structures must bypass builders/calculator acquisition")

    monkeypatch.setattr(stages_module, "acquire_calculator", forbidden)
    monkeypatch.setattr(structure_module, "build_surface", forbidden)
    monkeypatch.setattr(structure_module, "build_nanoparticle", forbidden)
    cfg = SimpleNamespace(
        structure=SimpleNamespace(
            kind="file",
            path=str(structure_path),
            format=None,
            index=-1,
            frozen_indices=None,
        )
    )
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="file-structure-test",
    )

    prepared = prepare_structure(
        cfg,
        identity,
        PreparedCalculator(resource=object(), primary=object()),
    )

    assert prepared.atoms is not None
    assert prepared.atoms.get_chemical_formula() == "Pt"
    assert prepared.frozen_indices is None
    assert prepared.structure_source is not None
    assert prepared.structure_source["path"] == str(structure_path.resolve())


def test_programmatic_file_config_resolves_relative_to_cwd(tmp_path, monkeypatch):
    structure_path = tmp_path / "input.extxyz"
    write(
        structure_path,
        Atoms("Pt", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3),
    )
    monkeypatch.chdir(tmp_path)

    atoms, source = load_structure_file("input.extxyz")

    assert len(atoms) == 1
    assert source["path"] == str(structure_path.resolve())


def test_file_workflow_rigidly_aligns_a_rotated_skew_slab(tmp_path):
    atoms = fcc111(
        "Cu",
        size=(3, 3, 4),
        vacuum=8.0,
        orthogonal=False,
    )
    atoms.rotate(25.0, "y", rotate_cell=True)
    original_gram = atoms.cell.array @ atoms.cell.array.T
    original_distances = atoms.get_all_distances(mic=False)
    atoms.set_constraint(FixAtoms(indices=[0]))
    cfg = RunConfig(structure=StructureCfg(kind="file"))
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="rotated-file-slab-test",
    )

    system = prepare_material_graph(
        cfg,
        identity,
        PreparedStructure(
            atoms=atoms,
            frozen_indices=None,
            structure_source={"kind": "file"},
        ),
    )

    assert system.surface_result.indices.tolist() == list(range(27, 36))
    np.testing.assert_allclose(
        atoms.cell.array @ atoms.cell.array.T,
        original_gram,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        atoms.get_all_distances(mic=False),
        original_distances,
        atol=1.0e-12,
    )
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-12)
    frame = system.structure_source["surface_frame"]
    assert frame["aligned_to_z"] is True
    assert frame["rotation_applied"] is True
    assert frame["periodic_connectivity_axes"] == [0, 1]


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        ([], "select exactly one frame"),
        (Atoms(), "empty Atoms"),
        (
            Atoms("Pt", positions=[[np.nan, 0.0, 0.0]]),
            "non-finite atomic positions",
        ),
    ],
)
def test_loader_rejects_invalid_ase_results(
    tmp_path,
    monkeypatch,
    loaded,
    message,
):
    import autokmc.structure.loading as loading_module

    structure_path = tmp_path / "input.xyz"
    structure_path.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(loading_module, "ase_read", lambda *_args, **_kwargs: loaded)

    with pytest.raises(ValueError, match=message):
        load_structure_file(structure_path)
