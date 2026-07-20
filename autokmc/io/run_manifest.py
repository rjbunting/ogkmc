"""Reproducible run metadata required by offline event analysis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from autokmc.core.constants import PERSISTENCE_SCHEMA_VERSION
from autokmc.io.event_transitions import occupied_surface_states
from autokmc.species.smiles import canonical_smiles


RUN_MANIFEST_SCHEMA_VERSION = "2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


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
    events_filename: str = "events.jsonl",
    initial_step: int = 0,
    initial_time_s: float = 0.0,
    run_id: str | None = None,
    resolved_config: Mapping[str, Any] | None = None,
) -> Path:
    """Create a new manifest or append a checkpoint-resume segment."""
    manifest_path = Path(path)
    if run_id is None and manifest_path.is_file():
        try:
            run_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id")
        except (OSError, json.JSONDecodeError, TypeError):
            run_id = None
    if not run_id:
        run_id = str(uuid.uuid4())
    segment = {
        "started_utc": _utc_now(),
        "initial_step": int(initial_step),
        "initial_time_s": float(initial_time_s),
    }
    if initial_step > 0 and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("run_id", str(run_id))
        payload.setdefault("segments", []).append(segment)
        payload["last_updated_utc"] = _utc_now()
        return _atomic_json(manifest_path, payload)

    feeds = []
    for reactant in feed_reactants:
        raw = str(reactant.get("smiles", reactant.get("species", "")))
        feeds.append(
            {
                "species": canonical_smiles(raw),
                "input_smiles": raw,
                "partial_pressure_bar": float(reactant.get("partial_pressure_bar", 0.0)),
            }
        )
    feeds.sort(key=lambda item: item["species"])
    n_catalyst = sum(
        1 for _, data in graph.nodes(data=True)
        if data.get("type") in {"surface", "bulk"}
    )
    n_surface = sum(
        1 for _, data in graph.nodes(data=True) if data.get("type") == "surface"
    )
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "event_schema_version": PERSISTENCE_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "last_updated_utc": _utc_now(),
        "config": _config_metadata(config_path, resolved_config),
        "files": {"events": str(events_filename)},
        "feed_reactants": feeds,
        "kmc": {
            "temperature_k": float(temperature_k),
            "random_seed": None if random_seed is None else int(random_seed),
        },
        "catalyst": {
            "structure_kind": str(structure_kind),
            "composition": composition,
            "n_catalyst_atoms": int(n_catalyst),
            "n_surface_atoms": int(n_surface),
        },
        "initial_state": {
            "step": int(initial_step),
            "time_s": float(initial_time_s),
            "occupied_surface_states": occupied_surface_states(graph, adsorbate_sites),
        },
        "segments": [segment],
        "result": None,
    }
    return _atomic_json(manifest_path, payload)


def finish_run_manifest(
    path: str | Path,
    *,
    final_step: int,
    final_time_s: float,
    steps_executed: int,
) -> Path:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["last_updated_utc"] = _utc_now()
    payload["result"] = {
        "finished_utc": _utc_now(),
        "final_step": int(final_step),
        "final_time_s": float(final_time_s),
        "steps_executed": int(steps_executed),
    }
    return _atomic_json(manifest_path, payload)


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "finish_run_manifest",
    "start_run_manifest",
]
