"""Scientific-configuration compatibility checks for checkpoint resumes.

Only settings listed in :data:`SAFE_RESUME_CONFIG_PATHS` may change between
the checkpoint-producing run and a continuation.  The allowlist is deliberately
small: a resumed invocation may request more steps, alter console logging, or
change how and where future checkpoints are written.  Every structure,
calculator, feed, thermochemistry, and reaction-channel setting remains part of
the scientific fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import autokmc as autokmc_package


RESUME_CONTRACT_SCHEMA_VERSION = "1"

_MODEL_ARTIFACT_FIELDS = frozenset({
    "checkpoint",
    "checkpoint_path",
    "model_checkpoint",
    "model",
    "model_file",
    "model_id",
    "model_path",
    "name_or_path",
    "param_file",
    "parameter_file",
    "path",
    "potential",
    "potential_file",
    "weights",
    "weights_path",
})

# These are operational controls that cannot alter the restored scientific
# state or the sequence of random draws.  In particular, ``kmc.n_steps`` means
# the number of *additional* events requested by a resumed invocation.
SAFE_RESUME_CONFIG_PATHS = frozenset({
    "checkpoint.enabled",
    "checkpoint.every_n_steps",
    "checkpoint.path",
    "checkpoint.resume_from",
    "kmc.log_every",
    "kmc.n_steps",
    "output.isaac_export_enabled",
    "output.isaac_export_filename",
    "output.log_level",
})


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _artifact_identity(value: Any) -> Any:
    """Represent a local model artifact by contents, not machine-local path."""
    if not isinstance(value, (str, Path)):
        return None
    candidate = Path(value).expanduser()
    try:
        if candidate.is_file():
            file_digest, size = _sha256_file(candidate)
            return {
                "artifact_kind": "file",
                "sha256": file_digest,
                "size_bytes": size,
            }
        if candidate.is_dir():
            directory_digest = hashlib.sha256()
            total_size = 0
            file_count = 0
            for path in sorted(
                (item for item in candidate.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(candidate).as_posix(),
            ):
                relative = path.relative_to(candidate).as_posix().encode("utf-8")
                file_digest, size = _sha256_file(path)
                directory_digest.update(len(relative).to_bytes(8, "big"))
                directory_digest.update(relative)
                directory_digest.update(bytes.fromhex(file_digest))
                directory_digest.update(size.to_bytes(8, "big"))
                total_size += size
                file_count += 1
            return {
                "artifact_kind": "directory",
                "sha256": directory_digest.hexdigest(),
                "size_bytes": total_size,
                "file_count": file_count,
            }
    except OSError:
        # Remote model aliases and temporarily unavailable paths remain part
        # of the literal config. They cannot be strengthened without backend
        # revision metadata, but must not make checkpoint creation fail.
        return None
    return None


def _normalise_existing_artifacts(value: Any) -> Any:
    """Content-hash every resolvable path nested in calculator kwargs."""
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_existing_artifacts(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalise_existing_artifacts(item) for item in value]
    artifact = _artifact_identity(value)
    return artifact if artifact is not None else value


def _autokmc_source_digest() -> str | None:
    """Hash installed AutoKMC Python sources to detect an in-place code change."""
    package_file = getattr(autokmc_package, "__file__", None)
    if not package_file:
        return None
    root = Path(package_file).resolve().parent
    try:
        paths = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
        digest = hashlib.sha256()
        for path in paths:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            file_digest, size = _sha256_file(path)
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(file_digest))
            digest.update(size.to_bytes(8, "big"))
        return digest.hexdigest()
    except OSError:
        return None


def _jsonable(value: Any, *, field_name: str = "") -> Any:
    """Return a deterministic JSON-compatible representation."""
    if field_name.lower() in _MODEL_ARTIFACT_FIELDS:
        artifact = _artifact_identity(value)
        if artifact is not None:
            return artifact
    if is_dataclass(value):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, field_name=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, field_name=field_name) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "run configuration contains a non-serializable value of type "
        f"{type(value).__name__}"
    )


@lru_cache(maxsize=1)
def _installed_package_distributions() -> Mapping[str, list[str]]:
    # A malformed, unrelated distribution can make Python 3.13 emit a
    # deprecation warning while building this global package map. It does not
    # affect the resolved versions used in the resume contract.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib_metadata.packages_distributions()


def _calculator_package_versions(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return installed distribution versions behind configured entry points."""
    calculator = payload.get("calculator")
    if not isinstance(calculator, Mapping):
        return {}

    entry_points: set[str] = set()

    def _collect_entry_points(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if (
                    str(key) in {"import_path", "factory"}
                    and isinstance(item, str)
                    and item
                ):
                    entry_points.add(item)
                _collect_entry_points(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect_entry_points(item)

    _collect_entry_points(calculator)
    module_roots = {
        str(path).partition(":")[0].partition(".")[0]
        for path in entry_points
    }
    try:
        distributions = _installed_package_distributions()
    except Exception:  # pragma: no cover - platform metadata failure
        distributions = {}
    versions: dict[str, str] = {}
    for module_root in sorted(module_roots):
        names = distributions.get(module_root, ()) or (module_root,)
        for name in sorted(set(names)):
            try:
                versions[str(name)] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue
    return versions


def _remove_path(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def scientific_config_payload(cfg: Any) -> dict[str, Any]:
    """Return the canonical resume-sensitive portion of *cfg*."""
    payload = _jsonable(cfg)
    if not isinstance(payload, dict):
        raise TypeError("run configuration must serialize to a mapping")
    calculator = payload.get("calculator")
    if isinstance(calculator, dict):
        for key in ("kwargs", "factory_kwargs"):
            if key in calculator:
                calculator[key] = _normalise_existing_artifacts(calculator[key])
    for path in SAFE_RESUME_CONFIG_PATHS:
        _remove_path(payload, path)
    payload["_software"] = {
        "autokmc_version": autokmc_package.__version__,
        "autokmc_source_sha256": _autokmc_source_digest(),
        "calculator_distributions": _calculator_package_versions(payload),
    }
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_resume_contract(cfg: Any) -> dict[str, Any]:
    """Build the versioned contract persisted in checkpoint metadata."""
    payload = scientific_config_payload(cfg)
    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "fingerprint": _fingerprint(payload),
        "scientific_config": payload,
        "safe_change_paths": sorted(SAFE_RESUME_CONFIG_PATHS),
    }


def _changed_paths(before: Any, after: Any, *, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_changed_paths(before[key], after[key], prefix=path))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        paths = []
        for index in range(max(len(before), len(after))):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if index >= len(before) or index >= len(after):
                paths.append(path)
            else:
                paths.extend(
                    _changed_paths(before[index], after[index], prefix=path)
                )
        return paths
    if before != after:
        return [prefix or "<root>"]
    return []


def verify_resume_contract(cfg: Any, stored: Mapping[str, Any]) -> None:
    """Raise when *cfg* changes a checkpoint's scientific configuration."""
    schema_version = str(stored.get("schema_version", ""))
    if schema_version != RESUME_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint resume contract schema_version="
            f"{schema_version!r} is unsupported; expected "
            f"{RESUME_CONTRACT_SCHEMA_VERSION!r}"
        )
    previous = stored.get("scientific_config")
    fingerprint = stored.get("fingerprint")
    if not isinstance(previous, Mapping) or not isinstance(fingerprint, str):
        raise ValueError("checkpoint resume contract is incomplete or malformed")
    if _fingerprint(previous) != fingerprint:
        raise ValueError("checkpoint resume contract fingerprint is corrupt")

    current = scientific_config_payload(cfg)
    current_fingerprint = _fingerprint(current)
    if current_fingerprint == fingerprint:
        return
    changed = _changed_paths(dict(previous), current)
    detail = ", ".join(changed[:8]) or "unknown fields"
    if len(changed) > 8:
        detail += f", and {len(changed) - 8} more"
    raise ValueError(
        "checkpoint scientific configuration does not match the resumed run; "
        f"changed resume-sensitive fields: {detail}. Safe changes are: "
        + ", ".join(sorted(SAFE_RESUME_CONFIG_PATHS))
    )


__all__ = [
    "RESUME_CONTRACT_SCHEMA_VERSION",
    "SAFE_RESUME_CONFIG_PATHS",
    "make_resume_contract",
    "scientific_config_payload",
    "verify_resume_contract",
]
