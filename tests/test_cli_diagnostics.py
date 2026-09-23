"""Safety, diagnostics, and CLI-boundary regression tests."""

from __future__ import annotations

from copy import deepcopy
import importlib
import logging
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from autokmc.cli.diagnostics import (
    PreflightError,
    doctor_report,
    preflight_config,
)
from autokmc.io.calculators import CalculatorCfg
from autokmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from autokmc.io.config import (
    CheckpointCfg,
    FreeEnergyCfg,
    KMCCfg,
    OutputCfg,
    ReactantCfg,
    RunConfig,
    StructureCfg,
)
from autokmc.io.resume_contract import make_resume_contract
from autokmc.kmc.models import KMCSettings
from autokmc.workflow.runtime import (
    OutputCollisionError,
    RunLockError,
    active_run_lock_owner,
    configured_run_lock,
)


def _valid_config(output_dir: Path) -> RunConfig:
    return RunConfig(
        output=OutputCfg(
            dir=str(output_dir),
            calculation_cache_enabled=False,
            trajectory_dump_every=0,
        ),
        reactants=[ReactantCfg(smiles="[O]", add_hydrogens=False)],
        calculator=CalculatorCfg(import_path="ase.calculators.emt.EMT"),
        free_energy=FreeEnergyCfg(enabled=False),
    )


def test_preflight_is_read_only_with_an_explicit_calculator(tmp_path):
    output_dir = tmp_path / "new-run"
    cfg = _valid_config(output_dir)

    result = preflight_config(cfg)

    assert result["status"] == "ok"
    assert result["checkpoint"]["mode"] == "fresh"
    assert result["calculator"]["target"] == "ase.calculators.emt.EMT"
    assert not output_dir.exists()


def test_preflight_rejects_a_missing_calculator_without_writing(tmp_path):
    output_dir = tmp_path / "new-run"
    cfg = _valid_config(output_dir)
    cfg.calculator = CalculatorCfg()

    with pytest.raises(
        PreflightError,
        match=r"calculator\.import_path or calculator\.factory",
    ):
        preflight_config(cfg)

    assert not output_dir.exists()


@pytest.mark.parametrize("relative_path", [True, False])
def test_preflight_reports_file_backed_structure(
    tmp_path,
    capsys,
    relative_path,
):
    from ase import Atoms
    from ase.io import write

    from autokmc.cli.main import _print_preflight

    config_dir = tmp_path / "config"
    structure_path = config_dir / "structures" / "catalyst.extxyz"
    structure_path.parent.mkdir(parents=True)
    write(
        structure_path,
        Atoms(
            "Pt2",
            positions=[[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        ),
        format="extxyz",
    )
    configured_path = (
        "structures/catalyst.extxyz"
        if relative_path
        else str(structure_path.resolve())
    )
    cfg = _valid_config(tmp_path / "new-run")
    cfg.structure = StructureCfg(
        kind="file",
        path=configured_path,
        index=-1,
        frozen_indices=[0],
    )

    result = preflight_config(
        cfg,
        config_path=config_dir / "quickstart.yaml",
    )

    assert result["structure"] == {
        "kind": "file",
        "path": str(structure_path.resolve()),
        "format": None,
        "index": -1,
        "sha256": result["structure"]["sha256"],
        "size_bytes": structure_path.stat().st_size,
        "chemical_formula": "Pt2",
        "frozen_indices": [0],
        "atom_count": 2,
        "frozen_count": 1,
    }
    assert len(result["structure"]["sha256"]) == 64
    _print_preflight(result)
    output = capsys.readouterr().out
    assert str(structure_path.resolve()) in output
    assert "index=-1" in output
    assert "atoms=2" in output
    assert "frozen=1" in output


def test_file_structure_calculator_probe_uses_loaded_composition(
    tmp_path,
    monkeypatch,
):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    observed: dict[str, str] = {}

    class RecordingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            assert atoms is not None
            observed["symbols"] = atoms.get_chemical_formula()
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
            }

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    structure_path = config_dir / "catalyst.extxyz"
    write(structure_path, Atoms("Pt2"), format="extxyz")
    cfg = _valid_config(tmp_path / "new-run")
    cfg.structure = StructureCfg(kind="file", path=structure_path.name)
    monkeypatch.setattr(
        "autokmc.io.calculators.build_calculator",
        lambda _cfg: RecordingCalculator(),
    )

    preflight_config(
        cfg,
        config_path=config_dir / "quickstart.yaml",
        check_calculator=True,
    )

    assert observed == {"symbols": "Pt"}


@pytest.mark.parametrize(
    ("filename", "contents", "frame_index", "error_pattern"),
    [
        ("missing.extxyz", None, -1, "structure file does not exist"),
        (
            "unreadable.extxyz",
            "this is not an extended XYZ structure\n",
            -1,
            "could not read structure file",
        ),
        ("single.extxyz", None, 8, "index=8"),
    ],
)
def test_preflight_rejects_invalid_file_backed_structure_before_calculator(
    tmp_path,
    monkeypatch,
    filename,
    contents,
    frame_index,
    error_pattern,
):
    from ase import Atoms
    from ase.io import write

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    structure_path = config_dir / filename
    if contents is not None:
        structure_path.write_text(contents, encoding="utf-8")
    elif filename != "missing.extxyz":
        write(structure_path, Atoms("Pt"), format="extxyz")

    cfg = _valid_config(tmp_path / "new-run")
    cfg.structure = StructureCfg(
        kind="file",
        path=filename,
        format="extxyz",
        index=frame_index,
    )
    monkeypatch.setattr(
        "autokmc.cli.diagnostics._resolve_calculator_target",
        lambda *_args, **_kwargs: pytest.fail(
            "calculator should not be resolved for an invalid structure"
        ),
    )

    with pytest.raises(PreflightError, match=error_pattern):
        preflight_config(
            cfg,
            config_path=config_dir / "quickstart.yaml",
        )


def test_preflight_optional_calculator_probe_requires_finite_results(tmp_path):
    output_dir = tmp_path / "new-run"
    result = preflight_config(
        _valid_config(output_dir),
        check_calculator=True,
    )

    assert result["calculator"]["check"]["forces_shape"] == [1, 3]
    assert not output_dir.exists()


def test_preflight_reports_import_and_fresh_output_failures(tmp_path):
    cfg = _valid_config(tmp_path / "run")
    cfg.calculator = CalculatorCfg(import_path="no_such_package.Calculator")
    Path(cfg.output.dir).mkdir()
    (Path(cfg.output.dir) / cfg.output.reactions_filename).write_text(
        "preserve me\n",
        encoding="utf-8",
    )

    with pytest.raises(PreflightError) as exc_info:
        preflight_config(cfg)

    message = str(exc_info.value)
    assert "output artifacts already exist" in message
    assert "could not be imported" in message


def test_preflight_rejects_unwritable_custom_checkpoint_destination(tmp_path):
    cfg = _valid_config(tmp_path / "run")
    non_directory_parent = tmp_path / "checkpoint-parent"
    non_directory_parent.write_text("preserve me\n", encoding="utf-8")
    cfg.checkpoint = CheckpointCfg(
        enabled=True,
        path=str(non_directory_parent / "checkpoint.pkl"),
    )

    with pytest.raises(PreflightError, match="checkpoint.path has a non-directory parent"):
        preflight_config(cfg)

    assert non_directory_parent.read_text(encoding="utf-8") == "preserve me\n"
    assert not Path(cfg.output.dir).exists()


def test_calculator_probe_uses_an_element_from_alloy_formulas():
    from autokmc.cli.diagnostics import _probe_symbol

    assert _probe_symbol("Pt3Ni") == "Pt"
    assert _probe_symbol({"Ni": 0.5, "Pt": 0.5}) == "Ni"


def test_preflight_validates_checkpoint_contract_and_allows_resume_outputs(
    tmp_path,
):
    output_dir = tmp_path / "run"
    source_cfg = _valid_config(output_dir)
    contract = make_resume_contract(source_cfg)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pkl",
        make_checkpoint_state(
            step=4,
            time_s=2.5,
            graph=nx.Graph(),
            adsorbate_sites=[],
            metadata={
                "run_id": "preflight-run",
                "resume_contract": contract,
            },
        ),
    )
    output_dir.mkdir()
    (output_dir / source_cfg.output.reactions_filename).write_text(
        '{"step": 1}\n',
        encoding="utf-8",
    )
    resume_cfg = deepcopy(source_cfg)
    resume_cfg.checkpoint = CheckpointCfg(resume_from=str(checkpoint))

    result = preflight_config(resume_cfg)

    assert result["checkpoint"]["mode"] == "resume"
    assert result["checkpoint"]["checkpoint_step"] == 4

    incompatible = deepcopy(resume_cfg)
    incompatible.kmc.temperature_k += 50.0
    with pytest.raises(PreflightError, match=r"kmc\.temperature_k"):
        preflight_config(incompatible)


def test_configured_run_lock_detects_reentry_and_releases(tmp_path):
    output_dir = tmp_path / "run"

    with configured_run_lock(output_dir):
        assert active_run_lock_owner(output_dir) is not None
        with pytest.raises(RunLockError):
            with configured_run_lock(output_dir):
                pass

    assert active_run_lock_owner(output_dir) is None


def test_run_from_config_holds_lock_around_complete_pipeline(
    tmp_path,
    monkeypatch,
):
    import autokmc.cli.pipeline as pipeline_module

    output_dir = tmp_path / "run"
    cfg = _valid_config(output_dir)

    def fake_pipeline(_cfg, *, config_path, telemetry):
        assert config_path == "config.yaml"
        assert telemetry is not None
        assert active_run_lock_owner(output_dir) is not None
        return {"status": "ok"}

    monkeypatch.setattr(pipeline_module, "_run_from_config", fake_pipeline)

    result = pipeline_module.run_from_config(
        cfg,
        config_path="config.yaml",
    )

    assert result == {"status": "ok"}
    assert active_run_lock_owner(output_dir) is None


def test_fresh_run_collision_refuses_before_pipeline_mutation(
    tmp_path,
    monkeypatch,
):
    import autokmc.cli.pipeline as pipeline_module

    cfg = _valid_config(tmp_path / "run")
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir()
    event_path = output_dir / cfg.output.reactions_filename
    event_path.write_text("do not overwrite\n", encoding="utf-8")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("pipeline ran despite an output collision")

    monkeypatch.setattr(pipeline_module, "_run_from_config", unexpected)

    with pytest.raises(OutputCollisionError):
        pipeline_module.run_from_config(cfg)

    assert event_path.read_text(encoding="utf-8") == "do not overwrite\n"


def test_cli_expected_errors_are_concise_and_debug_reraises(
    tmp_path,
    capsys,
):
    from autokmc.cli.main import main

    missing = tmp_path / "missing.yaml"
    assert main(["validate-config", str(missing)]) == 2
    captured = capsys.readouterr()
    assert "autokmc: error:" in captured.err
    assert "Traceback" not in captured.err

    with pytest.raises(FileNotFoundError):
        main(["--debug", "validate-config", str(missing)])


def test_cli_treats_exhausted_species_expansion_as_expected_error(
    tmp_path,
    monkeypatch,
    capsys,
):
    from autokmc.kmc.expansion import SpeciesExpansionError

    main_module = importlib.import_module("autokmc.cli.main")
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda _path: _valid_config(tmp_path / "run"),
    )

    def fail(*_args, **_kwargs):
        raise SpeciesExpansionError("backend retries exhausted")

    monkeypatch.setattr(main_module, "run_from_config", fail)

    assert main_module.main(["run", "config.yaml"]) == 2
    captured = capsys.readouterr()
    assert "backend retries exhausted" in captured.err
    assert "Traceback" not in captured.err

    with pytest.raises(SpeciesExpansionError):
        main_module.main(["--debug", "run", "config.yaml"])


def test_cli_report_forwards_options_and_prints_both_outputs(
    tmp_path,
    monkeypatch,
    capsys,
):
    main_module = importlib.import_module("autokmc.cli.main")

    calls = {}

    def fake_report(run_dir, **kwargs):
        calls["run_dir"] = run_dir
        calls.update(kwargs)
        return {
            "outputs": {
                "markdown": str(tmp_path / "report.md"),
                "html": str(tmp_path / "report.html"),
            }
        }

    monkeypatch.setattr(main_module, "generate_run_report", fake_report)

    result = main_module.main(
        [
            "report",
            str(tmp_path),
            "--manifest",
            "manifest.json",
            "--output-dir",
            str(tmp_path / "reports"),
            "--blocks",
            "4",
            "--allow-incomplete",
            "--no-refresh-analysis",
        ]
    )

    assert result == 0
    assert calls == {
        "run_dir": str(tmp_path),
        "manifest_filename": "manifest.json",
        "output_dir": str(tmp_path / "reports"),
        "n_blocks": 4,
        "strict": False,
        "refresh_analysis": False,
    }
    output = capsys.readouterr().out
    assert "report.md" in output
    assert "report.html" in output


def test_doctor_reports_runtime_and_packages_without_chemistry():
    report = doctor_report()

    assert report["runtime"]["python"]
    assert "ase" in report["packages"]
    assert "preflight" not in report


def test_cli_doctor_bootstrap_is_dependency_light_and_import_errors_are_friendly():
    repository_root = Path(__file__).resolve().parents[1]
    script = r"""
import builtins

blocked = {
    "ase",
    "jsonschema",
    "networkx",
    "numpy",
    "pymatgen",
    "rdkit",
    "scipy",
    "wulffpack",
}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition(".")[0] in blocked:
        raise ModuleNotFoundError(f"blocked scientific import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from autokmc.cli.main import main

doctor_rc = main(["doctor"])
analysis_rc = main(["analyze", "missing-run"])
print(f"doctor_rc={doctor_rc} analysis_rc={analysis_rc}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AutoKMC doctor:" in completed.stdout
    assert "analysis_rc=2" in completed.stdout
    assert "blocked scientific import" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_default_progress_is_periodic_and_info_is_not_detail_verbose():
    from autokmc.cli.pipeline import _progress_enabled, _verbose_enabled

    assert KMCCfg().log_every == 100
    assert KMCSettings(temperature=500.0, n_steps=1).log_every == 100
    assert _progress_enabled(logging.INFO) is True
    assert _verbose_enabled(logging.INFO) is False
    assert _progress_enabled(logging.DEBUG) is True
    assert _verbose_enabled(logging.DEBUG) is True
    assert _progress_enabled(logging.WARNING) is False
    assert KMCSettings(
        temperature=500.0,
        n_steps=1,
        verbose=False,
    ).progress_enabled is False
    assert KMCSettings(
        temperature=500.0,
        n_steps=1,
        progress=True,
        verbose=False,
    ).progress_enabled is True
