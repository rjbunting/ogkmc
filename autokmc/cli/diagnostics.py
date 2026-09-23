"""Read-only readiness checks for configured AutoKMC runs."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from autokmc import __version__ as autokmc_version

if TYPE_CHECKING:
    from autokmc.io.calculators import CalculatorCfg
    from autokmc.io.config import RunConfig


class PreflightError(RuntimeError):
    """Raised when a configured run is not safe or ready to start."""


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _check_output_writable(output_dir: Path) -> str | None:
    if output_dir.exists() and not output_dir.is_dir():
        return f"output.dir exists but is not a directory: {output_dir}"
    existing = _nearest_existing_path(output_dir)
    if not existing.exists():
        return f"no existing parent is available for output.dir: {output_dir}"
    if not existing.is_dir():
        return f"output.dir has a non-directory parent: {existing}"
    if not os.access(existing, os.W_OK | os.X_OK):
        return f"output.dir is not writable through existing parent: {existing}"
    return None


def _check_file_destination_writable(
    destination: Path,
    *,
    label: str,
) -> str | None:
    """Check whether an atomic file output can be published without writing."""
    if destination.exists() and destination.is_dir():
        return f"{label} exists but is a directory: {destination}"
    existing_parent = _nearest_existing_path(destination.parent)
    if not existing_parent.exists():
        return f"no existing parent is available for {label}: {destination}"
    if not existing_parent.is_dir():
        return f"{label} has a non-directory parent: {existing_parent}"
    if not os.access(existing_parent, os.W_OK | os.X_OK):
        return f"{label} is not writable through existing parent: {existing_parent}"
    return None


def _calculator_target(cfg: CalculatorCfg) -> tuple[str, str]:
    if cfg.factory:
        return "factory", str(cfg.factory)
    if cfg.import_path:
        return "import_path", str(cfg.import_path)
    raise PreflightError(
        "calculator must configure exactly one of "
        "calculator.import_path or calculator.factory"
    )


def _resolve_calculator_target(cfg: CalculatorCfg) -> tuple[str, str]:
    from autokmc.io.calculators import _resolve

    kind, target = _calculator_target(cfg)
    try:
        resolved = _resolve(target)
    except Exception as exc:
        raise PreflightError(
            f"calculator {kind} {target!r} could not be imported: {exc}"
        ) from exc
    if not callable(resolved):
        raise PreflightError(
            f"calculator {kind} {target!r} resolves to a non-callable "
            f"{type(resolved).__name__}"
        )
    return kind, target


def _checkpoint_readiness(cfg: RunConfig) -> tuple[dict[str, Any], list[str]]:
    from autokmc.io.checkpoint import load_checkpoint
    from autokmc.io.resume_contract import verify_resume_contract

    resume_from = cfg.checkpoint.resume_from
    if not resume_from:
        return {"mode": "fresh", "resume_from": None}, []

    checkpoint_path = Path(resume_from)
    if not checkpoint_path.is_file():
        raise PreflightError(
            f"checkpoint.resume_from is not a readable file: {checkpoint_path}"
        )
    try:
        checkpoint = load_checkpoint(checkpoint_path)
    except Exception as exc:
        raise PreflightError(
            f"checkpoint.resume_from could not be loaded: {checkpoint_path}: {exc}"
        ) from exc

    warnings: list[str] = []
    metadata = checkpoint.metadata
    if not isinstance(metadata, dict):
        raise PreflightError("checkpoint metadata is malformed")
    contract = metadata.get("resume_contract")
    if contract is None:
        warnings.append(
            "checkpoint has no scientific resume contract; compatibility "
            "cannot be fully verified"
        )
    elif not isinstance(contract, Mapping):
        raise PreflightError("checkpoint resume_contract metadata is malformed")
    else:
        try:
            verify_resume_contract(cfg, contract)
        except Exception as exc:
            raise PreflightError(str(exc)) from exc

    output_dir = Path(cfg.output.dir)
    manifest_path = output_dir / cfg.output.run_manifest_filename
    manifest_run_id: str | None = None
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise PreflightError(
                f"configured run manifest is not a file: {manifest_path}"
            )
        try:
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError(
                f"configured run manifest is unreadable: {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest_payload, dict):
            raise PreflightError(
                f"configured run manifest must contain a JSON object: "
                f"{manifest_path}"
            )
        raw_manifest_run_id = manifest_payload.get("run_id")
        if raw_manifest_run_id is not None:
            manifest_run_id = str(raw_manifest_run_id)

    raw_checkpoint_run_id = metadata.get("run_id")
    checkpoint_run_id = (
        None if raw_checkpoint_run_id is None else str(raw_checkpoint_run_id)
    )
    if (
        checkpoint_run_id
        and manifest_run_id
        and checkpoint_run_id != manifest_run_id
    ):
        raise PreflightError(
            "checkpoint run_id does not match the configured output manifest: "
            f"{checkpoint_run_id!r} != {manifest_run_id!r}"
        )

    return {
        "mode": "resume",
        "resume_from": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.step),
        "checkpoint_time_s": float(checkpoint.time_s),
        "run_id": checkpoint_run_id or manifest_run_id,
    }, warnings


def _structure_readiness(
    cfg: RunConfig,
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    """Resolve and parse a configured file-backed catalyst without writing."""
    structure_cfg = cfg.structure
    if structure_cfg.kind != "file":
        return {"kind": str(structure_cfg.kind)}

    from autokmc.structure.loading import load_structure_file

    structure_path = structure_cfg.path
    if not structure_path:
        raise PreflightError(
            "file-backed catalyst requires a non-empty structure.path"
        )
    try:
        atoms, source = load_structure_file(
            structure_path,
            format=structure_cfg.format,
            index=structure_cfg.index,
            frozen_indices=structure_cfg.frozen_indices,
            config_path=config_path,
        )
    except ValueError as exc:
        raise PreflightError(
            f"file-backed catalyst is not readable: {exc}"
        ) from exc

    return {
        **source,
        "atom_count": int(len(atoms)),
        "frozen_count": int(len(atoms.info.get("frozen_indices", ()))),
    }


def _probe_symbol(composition: Any) -> str:
    from ase.data import atomic_numbers
    from ase.formula import Formula

    if isinstance(composition, Mapping):
        candidates: list[Any] = list(composition)
    else:
        candidates = [composition]
    for candidate in candidates:
        text = str(candidate)
        if text in atomic_numbers:
            return text
        try:
            symbols = Formula(text)
        except (TypeError, ValueError):
            continue
        for symbol in symbols:
            if symbol in atomic_numbers:
                return symbol
    return "Cu"


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _exercise_calculator(
    cfg: RunConfig,
    *,
    probe_composition: Any | None = None,
) -> dict[str, float | list[int]]:
    from ase import Atoms
    import numpy as np

    from autokmc.io.calculators import (
        CalculatorPool,
        acquire_calculator,
        build_calculator,
    )

    calculator = None
    try:
        with tempfile.TemporaryDirectory(prefix="autokmc-preflight-") as temporary:
            with _working_directory(Path(temporary)):
                calculator = build_calculator(cfg.calculator)
                atoms = Atoms(
                    [
                        _probe_symbol(
                            cfg.structure.composition
                            if probe_composition is None
                            else probe_composition
                        )
                    ],
                    positions=[[0.0, 0.0, 0.0]],
                    cell=[8.0, 8.0, 8.0],
                    pbc=True,
                )
                with acquire_calculator(
                    calculator,
                    purpose="preflight energy/force check",
                ) as concrete:
                    atoms.calc = concrete
                    energy = float(atoms.get_potential_energy())
                    forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        raise PreflightError(
            f"calculator energy/force check failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if isinstance(calculator, CalculatorPool):
            calculator.shutdown(wait=True, cancel_futures=True)

    if not np.isfinite(energy):
        raise PreflightError(
            f"calculator energy/force check returned non-finite energy {energy!r}"
        )
    if forces.shape != (1, 3):
        raise PreflightError(
            "calculator energy/force check returned forces with unexpected "
            f"shape {forces.shape!r}; expected (1, 3)"
        )
    if not np.all(np.isfinite(forces)):
        raise PreflightError(
            "calculator energy/force check returned non-finite forces"
        )
    return {
        "energy_ev": energy,
        "max_force_ev_per_angstrom": float(
            np.linalg.norm(forces, axis=1).max(initial=0.0)
        ),
        "forces_shape": list(forces.shape),
    }


def preflight_config(
    cfg: RunConfig,
    *,
    config_path: str | Path | None = None,
    check_calculator: bool = False,
) -> dict[str, Any]:
    """Check run safety and readiness without creating configured outputs."""
    from autokmc.io.config import ConfigError
    from autokmc.io.config_validation import validate_config
    from autokmc.workflow.runtime import (
        active_run_lock_owner,
        managed_output_collisions,
    )

    try:
        validate_config(cfg, error_type=ConfigError)
    except ConfigError as exc:
        raise PreflightError(str(exc)) from exc

    output_dir = Path(cfg.output.dir)
    errors: list[str] = []
    structure_error = False
    try:
        structure = _structure_readiness(cfg, config_path=config_path)
    except PreflightError as exc:
        errors.append(str(exc))
        structure = {"kind": str(cfg.structure.kind)}
        structure_error = True

    owner = active_run_lock_owner(output_dir)
    if owner is not None:
        errors.append(
            f"another AutoKMC run is using output.dir {output_dir}: {owner}"
        )

    writable_error = _check_output_writable(output_dir)
    if writable_error:
        errors.append(writable_error)

    if cfg.checkpoint.enabled and cfg.checkpoint.path:
        checkpoint_error = _check_file_destination_writable(
            Path(cfg.checkpoint.path),
            label="checkpoint.path",
        )
        if checkpoint_error:
            errors.append(checkpoint_error)

    collisions = managed_output_collisions(cfg)
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        errors.append(
            "fresh-run output artifacts already exist: "
            f"{rendered}; choose a new output.dir or configure "
            "checkpoint.resume_from"
        )

    try:
        checkpoint, warnings = _checkpoint_readiness(cfg)
    except PreflightError as exc:
        errors.append(str(exc))
        checkpoint = {
            "mode": "resume" if cfg.checkpoint.resume_from else "fresh",
            "resume_from": cfg.checkpoint.resume_from,
        }
        warnings = []

    # A bad catalyst source is actionable without importing or probing a
    # calculator. Report it (and any independent filesystem issues) first.
    if structure_error:
        rendered = "\n".join(f"  - {message}" for message in errors)
        raise PreflightError(f"preflight failed:\n{rendered}")

    try:
        calculator_kind, calculator_target = _resolve_calculator_target(
            cfg.calculator
        )
    except PreflightError as exc:
        errors.append(str(exc))
        calculator_kind, calculator_target = _calculator_target(cfg.calculator)

    if errors:
        rendered = "\n".join(f"  - {message}" for message in errors)
        raise PreflightError(f"preflight failed:\n{rendered}")

    workers = cfg.calculator.max_workers or cfg.calculator.copies
    result: dict[str, Any] = {
        "status": "ok",
        "config_path": None if config_path is None else str(config_path),
        "output_dir": str(output_dir),
        "structure": structure,
        "checkpoint": checkpoint,
        "calculator": {
            "kind": calculator_kind,
            "target": calculator_target,
            "copies": int(cfg.calculator.copies),
            "max_workers": int(workers),
            "gpu_devices": list(cfg.calculator.gpu_devices or []),
        },
        "warnings": warnings,
    }
    if check_calculator:
        result["calculator"]["check"] = _exercise_calculator(
            cfg,
            probe_composition=(
                structure.get("chemical_formula")
                if structure.get("kind") == "file"
                else None
            ),
        )
    return result


_PACKAGE_MODULES = (
    ("autokmc", "autokmc"),
    ("ase", "ase"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("networkx", "networkx"),
    ("pymatgen", "pymatgen"),
    ("wulffpack", "wulffpack"),
    ("rdkit", "rdkit"),
    ("jsonschema", "jsonschema"),
)


def _package_versions() -> tuple[dict[str, str | None], list[str]]:
    packages: dict[str, str | None] = {}
    missing: list[str] = []
    for distribution, label in _PACKAGE_MODULES:
        if label == "autokmc":
            packages[label] = autokmc_version
            continue
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[label] = None
            missing.append(label)
    return packages, missing


def doctor_report(
    cfg: RunConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report runtime/package readiness without running chemistry."""
    packages, missing = _package_versions()
    issues = [f"required package is not installed: {name}" for name in missing]
    if sys.version_info < (3, 10):
        issues.append(
            "AutoKMC requires Python >= 3.10, found "
            f"{platform.python_version()}"
        )

    report: dict[str, Any] = {
        "status": "ok",
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "issues": issues,
    }
    if cfg is not None:
        try:
            report["preflight"] = preflight_config(
                cfg,
                config_path=config_path,
                check_calculator=False,
            )
        except PreflightError as exc:
            issues.append(str(exc))
    report["status"] = "ok" if not issues else "error"
    return report


__all__ = [
    "PreflightError",
    "doctor_report",
    "preflight_config",
]
