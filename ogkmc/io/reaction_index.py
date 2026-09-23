"""Compact, versioned index for discovered reaction definitions.

``events.jsonl`` stores time-dependent event data.  Static reaction metadata
is written once to ``reactions/index.jsonl`` and referenced by
``reaction_id``.  During a running job the index may contain compact
``stats`` updates; a clean close compacts it back to one definition row per
reaction.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from ogkmc.core.constants import (
    BOND_DESCRIPTION_FMT,
    DIFFUSION_DESCRIPTION_FMT,
    REACTION_DESCRIPTION_FMT,
)
from ogkmc.io._files import atomic_output_path, ensure_directory, fsync_directory
from ogkmc.io.reaction_layout import kind_subdir
from ogkmc.io.schemas import (
    REACTION_INDEX_ARTIFACT_TYPE,
    REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
    REACTION_INDEX_SCHEMA_VERSION,
    SUPPORTED_REACTION_INDEX_SCHEMA_VERSIONS,
)


REACTION_INDEX_FILENAME = "index.jsonl"
_STATIC_EVENT_FIELDS = ("description", "template", "reaction_dir", "gas_product")


def _event_description(
    event: Mapping[str, Any],
    *,
    template: Mapping[str, Any] | None,
) -> str | None:
    """Reconstruct the historical event description from compact row data."""
    delta = event.get("rate_delta_ev", event.get("delta_e_ev"))
    barrier = event.get("rate_barrier_ev", event.get("barrier_ev"))
    rate = event.get("rate_hz")
    if delta is None or barrier is None or rate is None:
        return None
    kind = str(event.get("kind", ""))
    subdir = kind_subdir(kind)
    common = {
        "iso": int(event.get("iso_class", -1)),
        "member": int(event.get("member_index", -1)),
        "lateral": int(event.get("lateral_class", -1)),
        "direction": str(event.get("direction") or ""),
        "delta_e": float(delta),
        "barrier": float(barrier),
        "rate": float(rate),
    }
    if subdir == "diffusion":
        description = DIFFUSION_DESCRIPTION_FMT.format(
            smiles=str(event.get("reactant_smiles", "")),
            **common,
        )
    elif subdir == "bond":
        static_template = dict(template or {})
        description = BOND_DESCRIPTION_FMT.format(
            smiles_a=str(static_template.get("smiles_a", "")),
            smiles_b=str(static_template.get("smiles_b", "")),
            smiles_c=str(static_template.get("smiles_c", "")),
            **common,
        )
    else:
        description = REACTION_DESCRIPTION_FMT.format(
            kind=kind,
            smiles=str(event.get("reactant_smiles", "")),
            **common,
        )
    if event.get("rate_energy_basis") == "free_energy":
        description = description.replace("ΔE=", "ΔG=").replace("Ea=", "G‡=")
    return description


def stable_reaction_id(
    kind: str,
    reactant_smiles: str,
    iso_class: int,
    lateral_class: int,
) -> str:
    """Return a direction-independent ID for one persisted reaction class."""
    identity = json.dumps(
        {
            "kind": kind_subdir(str(kind)),
            "reactant_smiles": str(reactant_smiles),
            "iso_class": int(iso_class),
            "lateral_class": int(lateral_class),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "reaction-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def reaction_id_from_event(event: Mapping[str, Any]) -> str:
    """Return an event's explicit ID or derive its legacy reaction-class ID."""
    explicit = event.get("reaction_id")
    if explicit:
        return str(explicit)
    return stable_reaction_id(
        str(event["kind"]),
        str(event.get("reactant_smiles", "")),
        int(event["iso_class"]),
        int(event["lateral_class"]),
    )


def stable_event_id(run_id: str | None, step: int) -> str:
    """Return a deterministic event ID scoped to one run and KMC step."""
    identity = f"{run_id or 'runless'}:{int(step)}"
    return "event-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _default_directions(kind: str) -> list[str]:
    subdir = kind_subdir(kind)
    if subdir == "adsorption":
        return ["adsorption", "desorption"]
    if subdir == "diffusion":
        return ["a_to_b", "b_to_a"]
    if subdir == "bond":
        return ["couple", "dissoc"]
    return [str(kind)]


def reaction_definition_from_document(
    payload: Mapping[str, Any],
    *,
    folder: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build one compact index definition from ``reaction.json`` metadata."""
    kind = str(payload.get("kind", ""))
    smiles = str(payload.get("reactant_smiles", ""))
    iso_class = int(payload.get("iso_class", -1))
    lateral_class = int(payload.get("lateral_class", -1))
    reaction_id = str(
        payload.get("reaction_id")
        or stable_reaction_id(kind, smiles, iso_class, lateral_class)
    )
    stats = dict(payload.get("stats") or {})
    definition = {
        field: payload.get(field)
        for field in ("description", "template", "gas_product")
        if payload.get(field) is not None
    }
    energy_bases = payload.get("rate_energy_bases") or []
    if isinstance(energy_bases, str):
        energy_bases = [energy_bases]
    return {
        "artifact_type": REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
        "schema_version": REACTION_INDEX_SCHEMA_VERSION,
        "record_type": "reaction",
        "reaction_id": reaction_id,
        "run_id": payload.get("run_id", run_id),
        "valid": bool(payload.get("valid", True)),
        "kind": kind_subdir(kind),
        "reactant_smiles": smiles,
        "iso_class": iso_class,
        "lateral_class": lateral_class,
        "directions": list(payload.get("kind_directions") or _default_directions(kind)),
        "discovery_step": int(payload.get("discovery_step", 0) or 0),
        "firing_count": int(stats.get("count", 0) or 0),
        "first_step": stats.get("first_step"),
        "last_step": stats.get("last_step"),
        "rate_energy_bases": sorted({str(value) for value in energy_bases if value}),
        "folder": str(folder),
        "definition": definition,
    }


def _merge_index_record(
    entries: dict[str, dict[str, Any]],
    record: Mapping[str, Any],
) -> None:
    record_type = str(record.get("record_type", "reaction"))
    if record_type == "header":
        return
    reaction_id = str(record.get("reaction_id") or "")
    if not reaction_id:
        raise ValueError("reaction index row has no reaction_id")
    if record_type == "stats":
        entry = entries.setdefault(
            reaction_id,
            {
                "artifact_type": REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
                "schema_version": REACTION_INDEX_SCHEMA_VERSION,
                "record_type": "reaction",
                "reaction_id": reaction_id,
            },
        )
        for key in (
            "firing_count",
            "first_step",
            "last_step",
            "rate_energy_bases",
        ):
            if key in record:
                entry[key] = record[key]
        return
    if record_type != "reaction":
        raise ValueError(f"unknown reaction index record_type={record_type!r}")
    previous = entries.get(reaction_id, {})
    merged = dict(previous)
    merged.update(dict(record))
    # A definition can follow an earlier stats update in a crash-recovered log.
    for key in ("firing_count", "first_step", "last_step", "rate_energy_bases"):
        if key in previous and key not in record:
            merged[key] = previous[key]
    entries[reaction_id] = merged


def load_reaction_index(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and fold a compacted or in-progress reaction index."""
    index_path = Path(path)
    if not index_path.is_file() or index_path.stat().st_size == 0:
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid reaction index row {line_number} in {index_path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"reaction index row {line_number} in {index_path} is not an object"
                )
            version = str(record.get("schema_version", ""))
            if version not in SUPPORTED_REACTION_INDEX_SCHEMA_VERSIONS:
                raise ValueError(
                    f"unsupported reaction index schema {version!r} on row "
                    f"{line_number} in {index_path}"
                )
            _merge_index_record(entries, record)
    return entries


def resolve_event_definition(
    event: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    require_definition: bool = False,
) -> dict[str, Any]:
    """Expand static fields on a compact event for compatibility consumers."""
    resolved = dict(event)
    if all(field in resolved for field in _STATIC_EVENT_FIELDS):
        return resolved
    reaction_id = resolved.get("reaction_id")
    definition = definitions.get(str(reaction_id)) if reaction_id else None
    if definition is None:
        if require_definition:
            raise ValueError(
                f"compact event {resolved.get('event_id', resolved.get('step'))!r} "
                f"references missing reaction_id={reaction_id!r}"
            )
        return resolved
    static = dict(definition.get("definition") or {})
    resolved.setdefault("template", static.get("template"))
    resolved.setdefault("gas_product", bool(static.get("gas_product", False)))
    resolved.setdefault("reaction_dir", definition.get("folder"))
    resolved.setdefault(
        "description",
        _event_description(resolved, template=resolved.get("template"))
        or static.get("description"),
    )
    return resolved


class ReactionIndexWriter:
    """Append efficient index updates and compact them on clean shutdown."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        initial_entries: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.path = Path(path)
        self.run_id = None if run_id is None else str(run_id)
        ensure_directory(self.path.parent)
        self.entries: dict[str, dict[str, Any]] = {}
        for entry in initial_entries:
            _merge_index_record(self.entries, entry)
        self._fp: TextIO | None = None
        self._pending_stats: set[str] = set()
        self._dirty = False
        self._compact()
        self._fp = self.path.open("a", encoding="utf-8")

    @property
    def n_valid(self) -> int:
        return sum(bool(entry.get("valid", True)) for entry in self.entries.values())

    @property
    def n_invalid(self) -> int:
        return len(self.entries) - self.n_valid

    def _header(self) -> dict[str, Any]:
        return {
            "artifact_type": REACTION_INDEX_ARTIFACT_TYPE,
            "schema_version": REACTION_INDEX_SCHEMA_VERSION,
            "record_type": "header",
            "run_id": self.run_id,
        }

    def _write_record(self, record: Mapping[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("reaction index writer has been closed")
        self._fp.write(json.dumps(dict(record), allow_nan=False, sort_keys=True) + "\n")
        self._dirty = True

    def register(self, entry: Mapping[str, Any]) -> str:
        reaction_id = str(entry["reaction_id"])
        normalized = dict(entry)
        normalized.setdefault("artifact_type", REACTION_INDEX_ENTRY_ARTIFACT_TYPE)
        normalized.setdefault("schema_version", REACTION_INDEX_SCHEMA_VERSION)
        normalized.setdefault("record_type", "reaction")
        existing = self.entries.get(reaction_id)
        if existing is not None:
            bases = {
                str(value)
                for value in (
                    list(existing.get("rate_energy_bases", []))
                    + list(normalized.get("rate_energy_bases", []))
                )
                if value
            }
            normalized["rate_energy_bases"] = sorted(bases)
        if existing == normalized:
            return reaction_id
        self.entries[reaction_id] = normalized
        self._write_record(normalized)
        return reaction_id

    def update_stats(
        self,
        reaction_id: str,
        *,
        count: int,
        first_step: int | None,
        last_step: int | None,
        rate_energy_basis: str | None = None,
    ) -> None:
        entry = self.entries[str(reaction_id)]
        entry["firing_count"] = int(count)
        entry["first_step"] = first_step
        entry["last_step"] = last_step
        bases = {str(value) for value in entry.get("rate_energy_bases", []) if value}
        if rate_energy_basis:
            bases.add(str(rate_energy_basis))
        entry["rate_energy_bases"] = sorted(bases)
        self._pending_stats.add(str(reaction_id))

    def sync(self) -> None:
        if self._fp is None:
            raise RuntimeError("reaction index writer has been closed")
        for reaction_id in sorted(self._pending_stats):
            entry = self.entries[reaction_id]
            self._write_record(
                {
                    "artifact_type": REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
                    "schema_version": REACTION_INDEX_SCHEMA_VERSION,
                    "record_type": "stats",
                    "reaction_id": reaction_id,
                    "firing_count": int(entry.get("firing_count", 0)),
                    "first_step": entry.get("first_step"),
                    "last_step": entry.get("last_step"),
                    "rate_energy_bases": list(entry.get("rate_energy_bases", [])),
                }
            )
        self._pending_stats.clear()
        if self._dirty:
            self._fp.flush()
            os.fsync(self._fp.fileno())
            self._dirty = False

    def _compact(self) -> None:
        with atomic_output_path(self.path) as temporary:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(self._header(), allow_nan=False, sort_keys=True) + "\n"
                )
                for reaction_id in sorted(self.entries):
                    handle.write(
                        json.dumps(
                            self.entries[reaction_id],
                            allow_nan=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )

    def close(self) -> None:
        if self._fp is None:
            return
        try:
            self.sync()
        finally:
            self._fp.close()
            self._fp = None
        self._compact()
        fsync_directory(self.path.parent)


__all__ = [
    "REACTION_INDEX_FILENAME",
    "ReactionIndexWriter",
    "load_reaction_index",
    "reaction_definition_from_document",
    "reaction_id_from_event",
    "resolve_event_definition",
    "stable_event_id",
    "stable_reaction_id",
]
