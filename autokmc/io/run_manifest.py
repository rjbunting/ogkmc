"""Durable lifecycle, provenance, and artifact metadata for configured runs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from autokmc import __version__
from autokmc.core.constants import (
    CALCULATION_CACHE_DIR,
    ISAAC_EXPORT_FILENAME,
    REACTIONS_DIR,
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
)
from autokmc.io._files import write_json_atomic
from autokmc.io.event_transitions import occupied_surface_states
from autokmc.io.performance import (
    PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH,
    PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION,
)
from autokmc.io.reaction_index import REACTION_INDEX_FILENAME
from autokmc.io.schemas import (
    EVENT_SCHEMA_VERSION,
    REACTION_DOCUMENT_SCHEMA_VERSION,
    REACTION_INDEX_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    TRAJECTORY_METADATA_SCHEMA_VERSION,
)
from autokmc.species.smiles import canonical_smiles


RUN_MANIFEST_SCHEMA_VERSION = "3"
RUN_LIFECYCLE_STATUSES = frozenset(
    {"preparing", "running", "complete", "stopped", "failed", "interrupted"}
)
TERMINAL_RUN_STATUSES = frozenset(
    {"complete", "stopped", "failed", "interrupted"}
)
_DIRECTORY_ENTRY_LIMIT = 10_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return write_json_atomic(path, payload, sort_keys=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run manifest {path} must contain a JSON object")
    return payload


def _config_metadata(
    config_path: str | None,
    resolved_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": config_path,
        "sha256": None,
        "content": None,
        "resolved": dict(resolved_config or {}),
    }
    if config_path:
        path = Path(config_path)
        if path.is_file():
            raw = path.read_bytes()
            metadata["sha256"] = hashlib.sha256(raw).hexdigest()
            metadata["content"] = raw.decode("utf-8")
    return metadata


def _segment(
    *,
    segment_index: int,
    initial_step: int,
    initial_time_s: float,
    now: str,
) -> dict[str, Any]:
    return {
        "segment_index": int(segment_index),
        "package_version": __version__,
        "started_utc": now,
        "ended_utc": None,
        "status": "preparing",
        "current_stage": "initializing",
        "stage_started_utc": now,
        "initial_step": int(initial_step),
        "initial_time_s": float(initial_time_s),
        "last_durable_step": int(initial_step),
        "last_durable_time_s": float(initial_time_s),
        "final_step": None,
        "final_time_s": None,
        "steps_executed": None,
        "termination_reason": None,
        "warnings": [],
    }


def _active_segment(payload: dict[str, Any]) -> dict[str, Any] | None:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    segment = segments[-1]
    return segment if isinstance(segment, dict) else None


def _close_stale_segment(payload: dict[str, Any], *, now: str) -> None:
    """Close an unterminated prior invocation before appending a resume."""
    segment = _active_segment(payload)
    if segment is None or segment.get("ended_utc") is not None:
        return
    result = payload.get("result")
    if isinstance(result, Mapping) and result.get("finished_utc"):
        segment.update(
            {
                "ended_utc": result["finished_utc"],
                "status": str(result.get("status") or "complete"),
                "termination_reason": str(
                    result.get("termination_reason")
                    or "legacy_manifest_completed"
                ),
                "final_step": result.get(
                    "final_step",
                    segment.get("last_durable_step"),
                ),
                "final_time_s": result.get(
                    "final_time_s",
                    segment.get("last_durable_time_s"),
                ),
                "steps_executed": result.get("steps_executed"),
            }
        )
        return
    segment["ended_utc"] = now
    segment["status"] = "interrupted"
    segment["termination_reason"] = "superseded_by_resume"
    if segment.get("final_step") is None:
        segment["final_step"] = segment.get("last_durable_step")
    if segment.get("final_time_s") is None:
        segment["final_time_s"] = segment.get("last_durable_time_s")
    warnings = payload.setdefault("warnings", [])
    warning = (
        "A prior invocation had no terminal manifest update and was marked "
        "interrupted when this resume segment started."
    )
    if warning not in warnings:
        warnings.append(warning)


def begin_run_manifest(
    path: str | Path,
    *,
    run_id: str | None,
    config_path: str | None,
    resolved_config: Mapping[str, Any] | None,
    events_filename: str = "events.jsonl",
    initial_step: int = 0,
    initial_time_s: float = 0.0,
    is_resume: bool = False,
) -> Path:
    """Publish the lifecycle record before any expensive configured stage."""
    manifest_path = Path(path)
    now = _utc_now()

    if is_resume and manifest_path.is_file():
        payload = _read_manifest(manifest_path)
        existing_run_id = str(payload.get("run_id") or "")
        if run_id and existing_run_id and existing_run_id != str(run_id):
            raise ValueError(
                "cannot append a run-manifest segment with a different run_id: "
                f"{run_id!r} != {existing_run_id!r}"
            )
        resolved_run_id = existing_run_id or str(run_id or uuid.uuid4())
        _close_stale_segment(payload, now=now)
        segments = payload.setdefault("segments", [])
        resumed_segment = _segment(
            segment_index=len(segments),
            initial_step=initial_step,
            initial_time_s=initial_time_s,
            now=now,
        )
        resumed_segment["config"] = _config_metadata(
            config_path,
            resolved_config,
        )
        segments.append(resumed_segment)
        payload.update(
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_id": resolved_run_id,
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "last_updated_utc": now,
                "files": {"events": str(events_filename)},
                "result": None,
            }
        )
        # ``package_version`` at the root identifies the original invocation.
        # Every segment records its own version so resumes after an upgrade do
        # not rewrite the provenance of earlier work.
        payload.setdefault("package_version", "unknown")
        payload["lifecycle"] = {
            "status": "preparing",
            "current_stage": "initializing",
            "started_utc": now,
            "stage_started_utc": now,
            "finished_utc": None,
            "termination_reason": None,
            "last_durable_step": int(initial_step),
            "last_durable_time_s": float(initial_time_s),
        }
        payload.setdefault("warnings", [])
        payload.setdefault("outputs", {})
        payload.setdefault("artifacts", {})
        payload.setdefault("invalid_counts", {})
        payload.setdefault("quarantine_locations", [])
        return _atomic_json(manifest_path, payload)

    resolved_run_id = str(run_id or uuid.uuid4())
    segment = _segment(
        segment_index=0,
        initial_step=initial_step,
        initial_time_s=initial_time_s,
        now=now,
    )
    segment["config"] = _config_metadata(config_path, resolved_config)
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "package_version": __version__,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "created_utc": now,
        "last_updated_utc": now,
        "config": _config_metadata(config_path, resolved_config),
        "files": {"events": str(events_filename)},
        "feed_reactants": [],
        "kmc": {},
        "catalyst": {},
        "initial_state": {
            "step": int(initial_step),
            "time_s": float(initial_time_s),
            "occupied_surface_states": [],
        },
        "lifecycle": {
            "status": "preparing",
            "current_stage": "initializing",
            "started_utc": now,
            "stage_started_utc": now,
            "finished_utc": None,
            "termination_reason": None,
            "last_durable_step": int(initial_step),
            "last_durable_time_s": float(initial_time_s),
        },
        "segments": [segment],
        "warnings": [],
        "outputs": {},
        "artifacts": {},
        "invalid_counts": {},
        "quarantine_locations": [],
        "result": None,
    }
    return _atomic_json(manifest_path, payload)


def update_run_manifest(
    path: str | Path,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    warning: str | None = None,
    last_durable_step: int | None = None,
    last_durable_time_s: float | None = None,
    termination_reason: str | None = None,
) -> Path:
    """Atomically update the active invocation's lifecycle fields."""
    if status is not None and status not in RUN_LIFECYCLE_STATUSES:
        raise ValueError(f"unsupported run lifecycle status {status!r}")
    manifest_path = Path(path)
    payload = _read_manifest(manifest_path)
    now = _utc_now()
    lifecycle = payload.setdefault("lifecycle", {})
    segment = _active_segment(payload)

    if status is not None:
        lifecycle["status"] = status
        if segment is not None:
            segment["status"] = status
    if current_stage is not None:
        if lifecycle.get("current_stage") != current_stage:
            lifecycle["stage_started_utc"] = now
        lifecycle["current_stage"] = str(current_stage)
        if segment is not None:
            if segment.get("current_stage") != current_stage:
                segment["stage_started_utc"] = now
            segment["current_stage"] = str(current_stage)
    if last_durable_step is not None:
        lifecycle["last_durable_step"] = int(last_durable_step)
        if segment is not None:
            segment["last_durable_step"] = int(last_durable_step)
    if last_durable_time_s is not None:
        lifecycle["last_durable_time_s"] = float(last_durable_time_s)
        if segment is not None:
            segment["last_durable_time_s"] = float(last_durable_time_s)
    if termination_reason is not None:
        lifecycle["termination_reason"] = str(termination_reason)
        if segment is not None:
            segment["termination_reason"] = str(termination_reason)
    if warning:
        warnings = payload.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(str(warning))
        if segment is not None:
            segment_warnings = segment.setdefault("warnings", [])
            if warning not in segment_warnings:
                segment_warnings.append(str(warning))
    payload["last_updated_utc"] = now
    return _atomic_json(manifest_path, payload)


def enrich_run_manifest(
    path: str | Path,
    *,
    graph: Any,
    adsorbate_sites: Iterable[Any],
    feed_reactants: Iterable[Mapping[str, Any]],
    temperature_k: float,
    random_seed: int | None,
    structure_kind: str,
    composition: Any,
    structure_source: Mapping[str, Any] | None = None,
    initial_step: int = 0,
    initial_time_s: float = 0.0,
) -> Path:
    """Add scientific provenance once the preparation stages have produced it."""
    manifest_path = Path(path)
    payload = _read_manifest(manifest_path)
    has_original_provenance = bool(payload.get("catalyst"))
    feeds = []
    for reactant in feed_reactants:
        raw = str(reactant.get("smiles", reactant.get("species", "")))
        feed: dict[str, Any] = {
            "species": canonical_smiles(raw),
            "input_smiles": raw,
            "partial_pressure_bar": float(reactant.get("partial_pressure_bar", 0.0)),
        }
        thermochemistry = reactant.get("thermochemistry")
        if isinstance(thermochemistry, Mapping) and thermochemistry:
            feed["thermochemistry"] = dict(thermochemistry)
        feeds.append(feed)
    feeds.sort(key=lambda item: item["species"])
    n_catalyst = sum(
        1
        for _, data in graph.nodes(data=True)
        if data.get("type") in {"surface", "bulk"}
    )
    n_surface = sum(
        1 for _, data in graph.nodes(data=True) if data.get("type") == "surface"
    )
    payload["feed_reactants"] = feeds
    payload["kmc"] = {
        "temperature_k": float(temperature_k),
        "random_seed": None if random_seed is None else int(random_seed),
    }
    existing_catalyst = payload.get("catalyst")
    catalyst = (
        dict(existing_catalyst)
        if has_original_provenance and isinstance(existing_catalyst, Mapping)
        else {
            "structure_kind": str(structure_kind),
            "composition": composition,
        }
    )
    catalyst.update(
        {
            "n_catalyst_atoms": int(n_catalyst),
            "n_surface_atoms": int(n_surface),
        }
    )
    if structure_source:
        catalyst["source"] = dict(structure_source)
    payload["catalyst"] = catalyst
    # A resume segment must not replace the original run's initial state.
    if not has_original_provenance:
        payload["initial_state"] = {
            "step": int(initial_step),
            "time_s": float(initial_time_s),
            "occupied_surface_states": occupied_surface_states(
                graph,
                adsorbate_sites,
            ),
        }
    payload["last_updated_utc"] = _utc_now()
    return _atomic_json(manifest_path, payload)


def start_run_manifest(
    path: str | Path,
    *,
    graph: Any,
    adsorbate_sites: Iterable[Any],
    feed_reactants: Iterable[Mapping[str, Any]],
    temperature_k: float,
    random_seed: int | None,
    structure_kind: str,
    composition: Any,
    config_path: str | None,
    structure_source: Mapping[str, Any] | None = None,
    events_filename: str = "events.jsonl",
    initial_step: int = 0,
    initial_time_s: float = 0.0,
    run_id: str | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    is_resume: bool = False,
) -> Path:
    """Compatibility wrapper that begins and enriches a manifest."""
    manifest_path = begin_run_manifest(
        path,
        run_id=run_id,
        config_path=config_path,
        resolved_config=resolved_config,
        events_filename=events_filename,
        initial_step=initial_step,
        initial_time_s=initial_time_s,
        is_resume=is_resume or initial_step > 0,
    )
    return enrich_run_manifest(
        manifest_path,
        graph=graph,
        adsorbate_sites=adsorbate_sites,
        feed_reactants=feed_reactants,
        temperature_k=temperature_k,
        random_seed=random_seed,
        structure_kind=structure_kind,
        composition=composition,
        structure_source=structure_source,
        initial_step=initial_step,
        initial_time_s=initial_time_s,
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.absolute().relative_to(root.absolute()))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _configured_value(
    cfg: Any | None,
    payload: Mapping[str, Any],
    section: str,
    name: str,
    default: Any,
) -> Any:
    if cfg is not None:
        section_value = getattr(cfg, section, None)
        if section_value is not None and hasattr(section_value, name):
            return getattr(section_value, name)
    config = payload.get("config", {})
    resolved = config.get("resolved", {}) if isinstance(config, Mapping) else {}
    section_value = (
        resolved.get(section, {}) if isinstance(resolved, Mapping) else {}
    )
    if isinstance(section_value, Mapping):
        return section_value.get(name, default)
    return default


def configured_artifact_descriptors(
    manifest_path: str | Path,
    *,
    cfg: Any | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return bounded expected-artifact descriptors, including missing files."""
    from autokmc.io.checkpoint import CHECKPOINT_SCHEMA_VERSION

    path = Path(manifest_path)
    root = path.parent
    manifest = dict(payload or (_read_manifest(path) if path.is_file() else {}))
    events_name = str(
        _configured_value(
            cfg,
            manifest,
            "output",
            "reactions_filename",
            REACTIONS_FILENAME,
        )
    )
    summary_name = str(
        _configured_value(
            cfg,
            manifest,
            "output",
            "summary_filename",
            SUMMARY_FILENAME,
        )
    )
    trajectory_name = str(
        _configured_value(
            cfg,
            manifest,
            "output",
            "trajectory_filename",
            TRAJECTORY_FILENAME,
        )
    )
    cache_name = str(
        _configured_value(
            cfg,
            manifest,
            "output",
            "calculation_cache_dir",
            CALCULATION_CACHE_DIR,
        )
    )
    isaac_name = str(
        _configured_value(
            cfg,
            manifest,
            "output",
            "isaac_export_filename",
            ISAAC_EXPORT_FILENAME,
        )
    )
    checkpoint_path = _configured_value(
        cfg,
        manifest,
        "checkpoint",
        "path",
        None,
    )
    if not checkpoint_path:
        checkpoint_path = root / "checkpoint.pkl"
    return {
        "run_manifest": {
            "path": path,
            "type": "run-manifest-json",
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "self_referential": True,
        },
        "events": {
            "path": root / events_name,
            "type": "event-log-jsonl",
            "schema_version": EVENT_SCHEMA_VERSION,
        },
        "summary": {
            "path": root / summary_name,
            "type": "run-summary-json",
            "schema_version": SUMMARY_SCHEMA_VERSION,
        },
        "performance_diagnostics": {
            "path": root / PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH,
            "type": "performance-diagnostics-json",
            "schema_version": PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION,
        },
        "trajectory": {
            "path": root / trajectory_name,
            "type": "trajectory-extxyz",
            "schema_version": TRAJECTORY_METADATA_SCHEMA_VERSION,
            "enabled": int(
                _configured_value(
                    cfg,
                    manifest,
                    "output",
                    "trajectory_dump_every",
                    10,
                )
                or 0
            )
            > 0,
        },
        "checkpoint": {
            "path": checkpoint_path,
            "type": "checkpoint-pickle",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "enabled": bool(
                _configured_value(
                    cfg,
                    manifest,
                    "checkpoint",
                    "enabled",
                    False,
                )
            ),
        },
        "calculation_cache": {
            "path": root / cache_name,
            "type": "calculation-cache-directory",
            "schema_version": "1",
            "enabled": bool(
                _configured_value(
                    cfg,
                    manifest,
                    "output",
                    "calculation_cache_enabled",
                    True,
                )
            ),
            "lookup_enabled": bool(
                _configured_value(
                    cfg,
                    manifest,
                    "output",
                    "calculation_cache_lookup_enabled",
                    False,
                )
            ),
        },
        "isaac_records": {
            "path": root / isaac_name,
            "type": "isaac-record-array-json",
            "schema_version": "1.05",
            "enabled": bool(
                _configured_value(
                    cfg,
                    manifest,
                    "output",
                    "isaac_export_enabled",
                    False,
                )
            ),
        },
        "reactions": {
            "path": root / REACTIONS_DIR,
            "type": "reaction-sidecar-directory",
            "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        },
        "reaction_index": {
            "path": root / REACTIONS_DIR / REACTION_INDEX_FILENAME,
            "type": "reaction-index-jsonl",
            "schema_version": REACTION_INDEX_SCHEMA_VERSION,
        },
        "invalid_adsorption": {
            "path": root / "diagnostics" / "invalid_adsorption",
            "type": "invalid-adsorption-diagnostics-directory",
            "schema_version": "1",
        },
        "invalid_diffusion": {
            "path": root / "diagnostics" / "invalid_diffusion",
            "type": "invalid-diffusion-diagnostics-directory",
            "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        },
        "invalid_bond": {
            "path": root / "diagnostics" / "invalid_bond",
            "type": "invalid-bond-diagnostics-directory",
            "schema_version": REACTION_DOCUMENT_SCHEMA_VERSION,
        },
    }


def configured_output_map(
    manifest_path: str | Path,
    *,
    cfg: Any | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the expected output map even when preparation stopped early."""
    path = Path(manifest_path)
    manifest = dict(payload or (_read_manifest(path) if path.is_file() else {}))
    descriptors = configured_artifact_descriptors(
        path,
        cfg=cfg,
        payload=manifest,
    )
    result = {
        name: (
            None
            if descriptor.get("enabled") is False
            else str(descriptor["path"])
        )
        for name, descriptor in descriptors.items()
    }
    result["reactions_dir"] = result.pop("reactions")
    return result


def _directory_summary(path: Path) -> dict[str, Any]:
    files = 0
    directories = 0
    symlinks = 0
    total_size = 0
    entries_seen = 0
    truncated = False
    errors: list[str] = []

    def record_error(error: OSError) -> None:
        if len(errors) < 10:
            errors.append(str(error))

    for current, directory_names, file_names in os.walk(
        path,
        topdown=True,
        onerror=record_error,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            entries_seen += 1
            if candidate.is_symlink():
                symlinks += 1
            else:
                directories += 1
                safe_directories.append(name)
            if entries_seen >= _DIRECTORY_ENTRY_LIMIT:
                truncated = True
                break
        directory_names[:] = safe_directories if not truncated else []
        if truncated:
            break
        for name in file_names:
            candidate = current_path / name
            entries_seen += 1
            if candidate.is_symlink():
                symlinks += 1
            else:
                files += 1
                try:
                    total_size += candidate.stat().st_size
                except OSError as exc:
                    if len(errors) < 10:
                        errors.append(f"{candidate.name}: {exc}")
            if entries_seen >= _DIRECTORY_ENTRY_LIMIT:
                truncated = True
                break
        if truncated:
            break
    return {
        "entry_count": entries_seen,
        "file_count": files,
        "directory_count": directories,
        "symlink_count": symlinks,
        "total_size_bytes": total_size,
        "truncated": truncated,
        "errors": errors,
    }


def build_artifact_inventory(
    output_dir: str | Path,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Describe canonical artifacts without recursively following symlinks."""
    root = Path(output_dir)
    inventory: dict[str, dict[str, Any]] = {}
    for name, descriptor in sorted(descriptors.items()):
        raw_path = descriptor.get("path")
        path = Path(raw_path) if raw_path is not None else None
        present = bool(
            path is not None and (path.exists() or path.is_symlink())
        )
        entry: dict[str, Any] = {
            "path": None if path is None else _display_path(path, root),
            "type": str(descriptor.get("type", "unknown")),
            "schema_version": descriptor.get("schema_version"),
            "present": present,
            "presence": present,
            "size_bytes": None,
            "sha256": None,
            "status": "missing",
        }
        if descriptor.get("self_referential") and path is not None and present:
            try:
                entry.update(
                    {
                        "kind": "file",
                        "size_bytes": path.stat().st_size,
                        "status": "partial",
                        "checksum_status": (
                            "omitted_self_referential_manifest"
                        ),
                    }
                )
            except OSError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)
            inventory[str(name)] = entry
            continue
        if not present or path is None:
            if descriptor.get("enabled") is False:
                entry["status"] = "disabled"
            inventory[str(name)] = entry
            continue
        try:
            if path.is_symlink():
                entry.update(
                    {
                        "kind": "symlink",
                        "target": os.readlink(path),
                        "status": "complete",
                    }
                )
            elif path.is_file():
                digest, size = _sha256_file(path)
                entry.update(
                    {
                        "kind": "file",
                        "size_bytes": size,
                        "sha256": digest,
                        "status": "complete",
                    }
                )
            elif path.is_dir():
                summary = _directory_summary(path)
                status = (
                    "truncated"
                    if summary["truncated"]
                    else "partial"
                    if summary["errors"]
                    else "complete"
                )
                entry.update(
                    {
                        "kind": "directory",
                        "size_bytes": summary["total_size_bytes"],
                        "status": status,
                        **summary,
                    }
                )
            else:
                entry["kind"] = "other"
                entry["status"] = "partial"
        except OSError as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
        inventory[str(name)] = entry
    return inventory


def discover_quarantine_locations(
    output_dir: str | Path,
    *,
    maximum: int = 100,
) -> list[str]:
    """Return bounded, output-relative paths to quarantined reaction leaves."""
    root = Path(output_dir)
    quarantine_root = root / "uncommitted_reactions"
    if not quarantine_root.is_dir():
        return []
    regular_leaves = {
        path
        for path in quarantine_root.glob("after_checkpoint_step_*/*/*/*")
        if path.is_dir()
        and path.relative_to(quarantine_root).parts[1:3]
        not in {
            ("diagnostics", "invalid_diffusion"),
            ("diagnostics", "invalid_bond"),
        }
    }
    invalid_diffusion_leaves = {
        path
        for path in quarantine_root.glob(
            "after_checkpoint_step_*/diagnostics/invalid_diffusion/*/*"
        )
        if path.is_dir()
    }
    invalid_bond_leaves = {
        path
        for path in quarantine_root.glob(
            "after_checkpoint_step_*/diagnostics/invalid_bond/*/*"
        )
        if path.is_dir()
    }
    leaves = sorted(
        regular_leaves | invalid_diffusion_leaves | invalid_bond_leaves
    )
    return [_display_path(path, root) for path in leaves[: max(0, maximum)]]


def finish_run_manifest(
    path: str | Path,
    *,
    final_step: int,
    final_time_s: float,
    steps_executed: int,
    status: str = "complete",
    termination_reason: str | None = None,
    wall_time_s: float | None = None,
    outputs: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    invalid_counts: Mapping[str, int] | None = None,
    quarantine_locations: Iterable[str] | None = None,
    performance: Mapping[str, Any] | None = None,
) -> Path:
    """Publish the terminal status and all durable configured-run outputs."""
    if status not in TERMINAL_RUN_STATUSES:
        raise ValueError(f"run status {status!r} is not terminal")
    manifest_path = Path(path)
    payload = _read_manifest(manifest_path)
    now = _utc_now()
    reason = termination_reason or (
        "requested_steps_completed" if status == "complete" else status
    )
    lifecycle = payload.setdefault("lifecycle", {})
    termination_stage = str(
        lifecycle.get("termination_stage")
        or lifecycle.get("current_stage")
        or "unknown"
    )
    lifecycle.update(
        {
            "status": status,
            "current_stage": "finished",
            "termination_stage": termination_stage,
            "finished_utc": now,
            "termination_reason": str(reason),
            "last_durable_step": int(final_step),
            "last_durable_time_s": float(final_time_s),
        }
    )
    segment = _active_segment(payload)
    if segment is not None:
        segment.update(
            {
                "ended_utc": now,
                "status": status,
                "current_stage": "finished",
                "termination_stage": termination_stage,
                "final_step": int(final_step),
                "final_time_s": float(final_time_s),
                "last_durable_step": int(final_step),
                "last_durable_time_s": float(final_time_s),
                "steps_executed": int(steps_executed),
                "termination_reason": str(reason),
            }
        )
    payload["last_updated_utc"] = now
    payload["result"] = {
        "finished_utc": now,
        "status": status,
        "termination_reason": str(reason),
        "termination_stage": termination_stage,
        "final_step": int(final_step),
        "final_time_s": float(final_time_s),
        "simulated_time_s": float(final_time_s),
        "wall_time_s": (
            None if wall_time_s is None else max(0.0, float(wall_time_s))
        ),
        "steps_executed": int(steps_executed),
    }
    if outputs is not None:
        payload["outputs"] = dict(outputs)
    if artifacts is not None:
        payload["artifacts"] = dict(artifacts)
    if invalid_counts is not None:
        payload["invalid_counts"] = {
            str(name): int(value) for name, value in invalid_counts.items()
        }
    if quarantine_locations is not None:
        payload["quarantine_locations"] = [
            str(location) for location in quarantine_locations
        ]
    if performance is not None:
        payload["performance"] = dict(performance)
    return _atomic_json(manifest_path, payload)


def fail_run_manifest(
    path: str | Path,
    error: BaseException | str,
    *,
    status: str = "failed",
    last_durable_step: int | None = None,
    last_durable_time_s: float | None = None,
    cfg: Any | None = None,
    invalid_counts: Mapping[str, int] | None = None,
    quarantine_locations: Iterable[str] | None = None,
) -> Path | None:
    """Best-effort terminal publication used while propagating an exception."""
    if status not in {"failed", "interrupted"}:
        raise ValueError("failure status must be 'failed' or 'interrupted'")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    payload = _read_manifest(manifest_path)
    lifecycle = payload.setdefault("lifecycle", {})
    step = int(
        lifecycle.get("last_durable_step", 0)
        if last_durable_step is None
        else last_durable_step
    )
    time_s = float(
        lifecycle.get("last_durable_time_s", 0.0)
        if last_durable_time_s is None
        else last_durable_time_s
    )
    reason = (
        str(error)
        if isinstance(error, str)
        else f"{type(error).__name__}: {error}"
    )
    descriptors = configured_artifact_descriptors(
        manifest_path,
        cfg=cfg,
        payload=payload,
    )
    artifacts = build_artifact_inventory(
        manifest_path.parent,
        descriptors,
    )
    outputs = configured_output_map(
        manifest_path,
        cfg=cfg,
        payload=payload,
    )
    return finish_run_manifest(
        manifest_path,
        final_step=step,
        final_time_s=time_s,
        steps_executed=max(
            0,
            step
            - int(
                (_active_segment(payload) or {}).get("initial_step", 0)
            ),
        ),
        status=status,
        termination_reason=reason,
        outputs=outputs,
        artifacts=artifacts,
        invalid_counts=invalid_counts,
        quarantine_locations=quarantine_locations,
    )


__all__ = [
    "RUN_LIFECYCLE_STATUSES",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "TERMINAL_RUN_STATUSES",
    "begin_run_manifest",
    "build_artifact_inventory",
    "configured_artifact_descriptors",
    "configured_output_map",
    "discover_quarantine_locations",
    "enrich_run_manifest",
    "fail_run_manifest",
    "finish_run_manifest",
    "start_run_manifest",
    "update_run_manifest",
]
