"""Graph-searchable reaction-result database with ISAAC records and extxyz assets.

Each database entry is an ISAAC v1.05 evidence record.  Structures are never
embedded in JSON: endpoint and transition-state geometries are immutable
``.extxyz`` assets, referenced by URI and SHA-256.  SQLite is only a rebuildable
search index; the record folders remain the source of truth.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import reduce
from importlib.resources import files as resource_files
import os
import sqlite3
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator, Mapping

import networkx as nx
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from autokmc.io.calculators import primary_calculator
from autokmc.io.reaction_graph import (
    normalise_reaction_graph,
    reaction_graph_from_payload,
    reaction_graph_hash,
    reaction_graph_payload,
    reaction_graphs_isomorphic,
)
from autokmc.utils.logging import get_logger


_log = get_logger(__name__)

ISAAC_RECORD_VERSION = "1.05"
REACTION_DATABASE_SCHEMA = "autokmc-reaction-database-v1"
_RECORD_FILENAME = "isaac_record.json"
_GRAPH_FILENAME = "reaction_graph.json"
_INDEX_FILENAME = "index.sqlite3"
_DATABASE_MANIFEST_FILENAME = "database_manifest.json"

_STATE_FILENAMES: dict[str, dict[str, str]] = {
    "adsorption": {
        "occupied": "occupied.extxyz",
        "unoccupied": "unoccupied.extxyz",
    },
    "diffusion": {
        "state_a": "state_a.extxyz",
        "state_b": "state_b.extxyz",
        "transition": "ts.extxyz",
    },
    "bond": {
        "state_ab": "state_ab.extxyz",
        "state_c": "state_c.extxyz",
        "transition": "ts.extxyz",
    },
}

_ISAAC_ROOT_FIELDS = {
    "isaac_record_version",
    "record_id",
    "record_type",
    "record_domain",
    "source_type",
    "timestamps",
    "sample",
    "system",
    "context",
    "measurement",
    "links",
    "assets",
    "descriptors",
    "computation",
    "attribution",
    "tags",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def initialise_calculation_database(
    root: str | os.PathLike[str],
    *,
    run_id: str,
) -> Path:
    """Link a calculation database to every run that has written into it."""
    root_path = _database_root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / _DATABASE_MANIFEST_FILENAME
    payload: dict[str, Any] = {
        "schema_version": "1",
        "created_utc": _utc_now(),
        "last_updated_utc": _utc_now(),
        "current_run_id": str(run_id),
        "run_ids": [str(run_id)],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        run_ids = [str(item) for item in existing.get("run_ids", [])]
        if str(run_id) not in run_ids:
            run_ids.append(str(run_id))
        payload = {
            **existing,
            "schema_version": "1",
            "last_updated_utc": _utc_now(),
            "current_run_id": str(run_id),
            "run_ids": run_ids,
        }
    _atomic_json(path, payload)
    return path


def _database_run_id(root: Path) -> str | None:
    path = root / _DATABASE_MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("current_run_id")
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return None if value is None else str(value)


def _jsonable(value: Any) -> Any:
    """Convert scientific Python values to deterministic JSON-safe values."""
    if isinstance(value, Atoms):
        return atoms_to_json(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "todict"):
        try:
            return _jsonable(value.todict())
        except Exception:
            pass
    return {
        "python_type": f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _metadata_jsonable(value: Any) -> Any:
    """JSON-safe metadata conversion that never embeds an atomic structure."""
    if isinstance(value, Atoms):
        return {
            "structure_embedded": False,
            "chemical_formula": value.get_chemical_formula(),
            "n_atoms": len(value),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_metadata_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    return _jsonable(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def atoms_to_json(atoms: Atoms) -> dict[str, Any]:
    """Return a compact deterministic geometry payload for cache-key hashing.

    This helper is retained as public API for callers that build keys.  The
    persisted reaction record does not embed this payload; it writes extxyz.
    """
    fixed: list[int] = []
    for constraint in getattr(atoms, "constraints", ()) or ():
        if isinstance(constraint, FixAtoms):
            fixed.extend(int(index) for index in constraint.get_indices())
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions_A": np.asarray(atoms.positions, dtype=float).round(10).tolist(),
        "cell_A": np.asarray(atoms.cell.array, dtype=float).round(10).tolist(),
        "pbc": [bool(value) for value in atoms.pbc],
        "fixed_indices": sorted(set(fixed)),
    }


def calculator_identity(calculator: Any) -> dict[str, Any]:
    """Return a stable calculator/method declaration for compatibility checks."""
    concrete = primary_calculator(calculator)
    identity: dict[str, Any] = {
        "class": f"{concrete.__class__.__module__}.{concrete.__class__.__qualname__}",
    }
    parameters = getattr(concrete, "parameters", None)
    if parameters:
        try:
            parameter_items = dict(parameters).items()
        except (TypeError, ValueError):
            parameter_items = ()
        method_parameters = {
            str(key): value
            for key, value in parameter_items
            if str(key).lower() not in {"device", "devices", "gpu", "gpu_devices"}
        }
        if method_parameters:
            identity["parameters"] = _jsonable(method_parameters)
    for name in ("model_name", "name_or_path", "checkpoint", "task_name"):
        value = getattr(concrete, name, None)
        if value not in (None, ""):
            identity[name] = _jsonable(value)
    return identity


def calculation_cache_key(
    *,
    kind: str,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> str:
    """Hash the exact calculation request for fast same-run lookup."""
    return _hash_json(
        {
            "schema": REACTION_DATABASE_SCHEMA,
            "kind": str(kind),
            "identity": identity,
            "parameters": parameters,
            "inputs": inputs,
        }
    )


def state_payload(
    atoms: Atoms,
    *,
    energy_ev: float,
    properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transient state payload; ``atoms`` is externalized on write."""
    if not isinstance(atoms, Atoms):
        raise TypeError("state atoms must be an ase.Atoms instance")
    return {
        "atoms": atoms.copy(),
        "energy_ev": float(energy_ev),
        "properties": _jsonable(dict(properties or {})),
    }


def make_calculation_record(
    *,
    kind: str,
    cache_key: str,
    operation: Mapping[str, Any],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    reaction_graph: nx.Graph,
    neb: Mapping[str, Any] | None = None,
    lateral_attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the transient source object consumed by the ISAAC writer."""
    kind = str(kind)
    if kind not in _STATE_FILENAMES:
        raise ValueError(f"unsupported reaction database kind: {kind!r}")
    missing = set(_STATE_FILENAMES[kind]) - set(states)
    if missing:
        raise ValueError(f"{kind} record is missing required states: {sorted(missing)}")
    return {
        "schema": REACTION_DATABASE_SCHEMA,
        "kind": kind,
        "cache_key": str(cache_key),
        "operation": dict(operation),
        "parameters": dict(parameters),
        "inputs": dict(inputs),
        "states": dict(states),
        "reaction_graph": normalise_reaction_graph(reaction_graph),
        "neb": None if neb is None else dict(neb),
        "lateral_attributes": dict(lateral_attributes or {}),
    }


def _operation_key(operation: Mapping[str, Any]) -> str:
    # Run-local class numbers and prose labels cannot be portable lookup keys.
    semantic_keys = ("reactant_smiles", "smiles_a", "smiles_b", "smiles_c")
    semantic = {
        key: operation[key]
        for key in semantic_keys
        if operation.get(key) not in (None, "")
    }
    if not semantic:
        semantic = {
            key: value
            for key, value in operation.items()
            if key not in {"iso_class", "lateral_class", "label", "reaction"}
        }
    return _hash_json(semantic)


def _record_id(cache_key: str, created_utc: str) -> str:
    """Create a standards-conformant ULID with deterministic entropy."""
    timestamp = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    milliseconds = int(timestamp.timestamp() * 1000)
    entropy = int.from_bytes(hashlib.sha256(cache_key.encode("utf-8")).digest()[:10], "big")
    value = (milliseconds << 80) | entropy
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    encoded = []
    for _ in range(26):
        encoded.append(alphabet[value & 31])
        value >>= 5
    return "".join(reversed(encoded))


def _database_root(root: str | os.PathLike[str]) -> Path:
    return Path(root).expanduser().resolve()


@contextmanager
def _connect(root: Path) -> Iterator[sqlite3.Connection]:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / _INDEX_FILENAME, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                operation_key TEXT NOT NULL,
                parameter_hash TEXT NOT NULL,
                graph_hash TEXT NOT NULL,
                record_path TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS records_graph_lookup
            ON records(kind, operation_key, parameter_hash, graph_hash)
            """
        )
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _safe_atoms_copy(atoms: Atoms, energy_ev: float | None = None) -> Atoms:
    snapshot = atoms.copy()
    snapshot.calc = None
    snapshot.info = {
        str(key): _jsonable(value)
        for key, value in getattr(atoms, "info", {}).items()
    }
    if energy_ev is not None:
        snapshot.info["autokmc_energy_ev"] = float(energy_ev)
    return snapshot


def _atomic_extxyz(path: Path, images: Atoms | list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        ase_write(tmp_name, images, format="extxyz")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _asset(
    *,
    asset_id: str,
    role: str,
    filename: str,
    path: Path,
    media_type: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "content_role": role,
        "uri": filename,
        "sha256": _sha256_file(path),
        "media_type": media_type,
        "notes": notes,
    }


def _descriptor(
    name: str,
    value: Any,
    *,
    kind: str = "theoretical_metric",
    unit: str | None = None,
    definition: str | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "source": "auto",
        "value": _jsonable(value),
        "uncertainty": {
            "sigma": None,
            "basis": "not_reported",
        },
    }
    if unit is not None:
        descriptor["unit"] = unit
        descriptor["uncertainty"]["unit"] = unit
    if definition:
        descriptor["definition"] = definition
    return descriptor


def _calculator_method(parameters: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    calculator = parameters.get("calculator", {})
    class_name = str(calculator.get("class", "unknown"))
    lowered = class_name.lower()
    if any(word in lowered for word in ("fairchem", "nequip", "mace", "allegro", "torch")):
        technique = "machine_learning_potential"
        family = "machine_learning"
    elif any(word in lowered for word in ("vasp", "espresso", "gpaw", "dft")):
        technique = "DFT"
        family = "DFT"
    else:
        # ISAAC has no generic atomistic-potential technique.  ``classical_MD``
        # is the closest controlled term for EMT, xTB and other non-DFT,
        # non-ML atomistic calculators; the precise class remains in method.code.
        technique = "classical_MD"
        family = "semi_empirical"
    method = {
        "family": family,
        "code": class_name,
        "notes": "Calculator declaration is mirrored in system.configuration.autokmc.calculator.",
    }
    return technique, method


def _material_from_graph(graph: nx.Graph) -> tuple[str, str]:
    counts = Counter(
        str(data.get("element"))
        for _, data in graph.nodes(data=True)
        if data.get("type") == "surface" and data.get("element")
    )
    elements = sorted(counts)
    if counts:
        divisor = reduce(math.gcd, counts.values())
        formula = "".join(
            element + (str(counts[element] // divisor) if counts[element] // divisor > 1 else "")
            for element in elements
        )
    else:
        formula = "unknown"
    name = "-".join(elements) + " surface" if elements else "unknown surface"
    return name, formula


def _state_descriptors(kind: str, states: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    descriptors = [
        _descriptor(
            f"reaction.energy.{name}",
            float(payload["energy_ev"]),
            unit="eV",
            definition=f"Potential energy of the {name} extxyz structure.",
        )
        for name, payload in states.items()
    ]
    energies = {name: float(payload["energy_ev"]) for name, payload in states.items()}
    if kind == "adsorption":
        descriptors.append(
            _descriptor(
                "reaction.energy_change",
                energies["occupied"] - energies["unoccupied"],
                unit="eV",
                definition="Occupied minus unoccupied local-structure potential energy.",
            )
        )
    elif kind == "diffusion":
        descriptors.extend(
            (
                _descriptor(
                    "reaction.activation_barrier_forward",
                    energies["transition"] - energies["state_a"],
                    unit="eV",
                ),
                _descriptor(
                    "reaction.activation_barrier_reverse",
                    energies["transition"] - energies["state_b"],
                    unit="eV",
                ),
                _descriptor(
                    "reaction.energy_change",
                    energies["state_b"] - energies["state_a"],
                    unit="eV",
                ),
            )
        )
    elif kind == "bond":
        descriptors.extend(
            (
                _descriptor(
                    "reaction.activation_barrier_forward",
                    energies["transition"] - energies["state_ab"],
                    unit="eV",
                ),
                _descriptor(
                    "reaction.activation_barrier_reverse",
                    energies["transition"] - energies["state_c"],
                    unit="eV",
                ),
                _descriptor(
                    "reaction.energy_change",
                    energies["state_c"] - energies["state_ab"],
                    unit="eV",
                ),
            )
        )
    return descriptors


def _package_version() -> str:
    try:
        return importlib_metadata.version("autokmc")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _build_isaac_record(
    source: Mapping[str, Any],
    *,
    record_id: str,
    created_utc: str,
    graph: nx.Graph,
    graph_hash: str,
    operation_key: str,
    parameter_hash: str,
    assets: list[dict[str, Any]],
    state_assets: Mapping[str, str],
    neb_asset: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    kind = str(source["kind"])
    operation = _metadata_jsonable(source["operation"])
    parameters = _metadata_jsonable(source["parameters"])
    inputs = _metadata_jsonable(source["inputs"])
    lateral_attributes = _metadata_jsonable(source.get("lateral_attributes", {}))
    states = source["states"]
    technique, method = _calculator_method(parameters)
    material_name, formula = _material_from_graph(graph)

    autokmc_configuration = {
        "database_schema": REACTION_DATABASE_SCHEMA,
        "matcher_schema": graph.graph.get("schema"),
        "kind": kind,
        "cache_key": source["cache_key"],
        "operation_key": operation_key,
        "parameter_hash": parameter_hash,
        "graph_hash": graph_hash,
        "operation": operation,
        "parameters": parameters,
        "inputs": inputs,
        "state_assets": dict(state_assets),
        "state_data": {
            name: {
                "energy_ev": float(payload["energy_ev"]),
                "properties": _metadata_jsonable(payload.get("properties", {})),
            }
            for name, payload in states.items()
        },
        "neb_asset": neb_asset,
        "neb_energies_ev": _jsonable((source.get("neb") or {}).get("energies_ev", [])),
        "lateral_attributes": lateral_attributes,
        "calculator": parameters.get("calculator", {}),
    }
    if run_id is not None:
        autokmc_configuration["run_id"] = str(run_id)

    descriptors = [
        _descriptor("autokmc.reaction_kind", kind, kind="categorical"),
        _descriptor("autokmc.graph_hash", graph_hash, kind="categorical"),
        *_state_descriptors(kind, states),
    ]
    record: dict[str, Any] = {
        "isaac_record_version": ISAAC_RECORD_VERSION,
        "record_id": record_id,
        "record_type": "evidence",
        "record_domain": "simulation",
        "source_type": "computation",
        "timestamps": {"created_utc": created_utc},
        "sample": {
            "material": {"name": material_name, "formula": formula},
            "sample_form": "surface model",
        },
        "system": {
            "domain": "computational",
            "technique": technique,
            "configuration": {"autokmc": autokmc_configuration},
        },
        "computation": {
            "method": method,
            "output_quantity": (
                "adsorption endpoint energies" if kind == "adsorption"
                else "activation barrier and endpoint energies"
            ),
            "corrections_applied": {
                "free_energy": bool(parameters.get("free_energy_enabled", False)),
            },
        },
        "assets": assets,
        "descriptors": {
            "outputs": [
                {
                    "label": "autokmc_reaction_result",
                    "generated_utc": created_utc,
                    "generated_by": {
                        "agent": "AutoKMC",
                        "version": _package_version(),
                    },
                    "descriptors": descriptors,
                }
            ]
        },
        "tags": ["autokmc-reaction-database", f"autokmc-{kind}"],
    }

    temperature = operation.get("temperature_k", parameters.get("temperature_k"))
    if isinstance(temperature, (int, float)) and math.isfinite(float(temperature)):
        record["context"] = {
            "environment": "in_silico",
            "temperature_K": float(temperature),
        }
    if kind in {"diffusion", "bond"}:
        transition_state: dict[str, Any] = {
            "method": "CI-NEB" if parameters.get("climb", True) else "NEB",
            "reaction": str(operation.get("reaction", operation.get("label", kind))),
        }
        if parameters.get("n_images") is not None:
            transition_state["images"] = int(parameters["n_images"])
        record["computation"]["transition_state"] = transition_state

    validate_isaac_record(record)
    return record


def validate_isaac_record(record: Mapping[str, Any]) -> None:
    """Validate against the vendored official ISAAC v1.05 JSON schema."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise RuntimeError("ISAAC validation requires the jsonschema package") from exc
    schema_path = resource_files("autokmc").joinpath("schema/isaac_record_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(record)), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"ISAAC schema validation failed at {location}: {error.message}")


def write_calculation_record(
    root: str | os.PathLike[str],
    kind: str,
    cache_key: str,
    record: Mapping[str, Any],
) -> Path:
    """Persist one immutable reaction result and update the SQLite index."""
    root_path = _database_root(root)
    kind = str(kind)
    if kind != record.get("kind") or cache_key != record.get("cache_key"):
        raise ValueError("record kind/cache_key do not match write target")
    graph = normalise_reaction_graph(record["reaction_graph"])
    graph_digest = reaction_graph_hash(graph)
    operation_digest = _operation_key(record["operation"])
    parameter_digest = _hash_json(record["parameters"])
    created_utc = _utc_now()
    record_id = _record_id(cache_key, created_utc)
    record_dir = root_path / "records" / record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    graph_path = record_dir / _GRAPH_FILENAME
    _atomic_json(graph_path, reaction_graph_payload(graph))
    assets.append(
        _asset(
            asset_id="reaction_graph",
            role="metadata_snapshot",
            filename=_GRAPH_FILENAME,
            path=graph_path,
            media_type="application/json",
            notes="Portable labelled graph used for full isomorphism confirmation.",
        )
    )

    state_assets: dict[str, str] = {}
    for state_name, filename in _STATE_FILENAMES[kind].items():
        payload = record["states"][state_name]
        atoms = payload.get("atoms")
        if not isinstance(atoms, Atoms):
            raise ValueError(f"state {state_name!r} does not contain ase.Atoms")
        state_path = record_dir / filename
        _atomic_extxyz(
            state_path,
            _safe_atoms_copy(atoms, float(payload["energy_ev"])),
        )
        asset_id = f"structure_{state_name}"
        state_assets[state_name] = asset_id
        assets.append(
            _asset(
                asset_id=asset_id,
                role="reduction_product",
                filename=filename,
                path=state_path,
                media_type="chemical/x-xyz",
                notes=f"Relaxed {state_name} structure required to reproduce this result.",
            )
        )

    neb_asset: str | None = None
    neb = record.get("neb") or {}
    path_atoms = neb.get("path_atoms", neb.get("path"))
    if path_atoms:
        if not all(isinstance(image, Atoms) for image in path_atoms):
            raise ValueError("NEB path must contain ase.Atoms images")
        energies = list(neb.get("energies_ev", []))
        images = [
            _safe_atoms_copy(image, energies[index] if index < len(energies) else None)
            for index, image in enumerate(path_atoms)
        ]
        neb_path = record_dir / "neb_path.extxyz"
        _atomic_extxyz(neb_path, images)
        neb_asset = "structure_neb_path"
        assets.append(
            _asset(
                asset_id=neb_asset,
                role="reduction_product",
                filename="neb_path.extxyz",
                path=neb_path,
                media_type="chemical/x-xyz",
                notes="Optional complete NEB image sequence.",
            )
        )

    isaac = _build_isaac_record(
        record,
        record_id=record_id,
        created_utc=created_utc,
        graph=graph,
        graph_hash=graph_digest,
        operation_key=operation_digest,
        parameter_hash=parameter_digest,
        assets=assets,
        state_assets=state_assets,
        neb_asset=neb_asset,
        run_id=_database_run_id(root_path),
    )
    record_path = record_dir / _RECORD_FILENAME
    # Assets are complete before the ISAAC record becomes visible.
    _atomic_json(record_path, isaac)

    with _connect(root_path) as connection:
        connection.execute(
            """
            INSERT INTO records(
                record_id, cache_key, kind, operation_key, parameter_hash,
                graph_hash, record_path, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                record_id=excluded.record_id,
                kind=excluded.kind,
                operation_key=excluded.operation_key,
                parameter_hash=excluded.parameter_hash,
                graph_hash=excluded.graph_hash,
                record_path=excluded.record_path,
                created_utc=excluded.created_utc
            """,
            (
                record_id,
                cache_key,
                kind,
                operation_digest,
                parameter_digest,
                graph_digest,
                str(record_path.relative_to(root_path)),
                created_utc,
            ),
        )
    return record_path


def _safe_asset_path(record_dir: Path, uri: str) -> Path:
    path = (record_dir / uri).resolve()
    if path.parent != record_dir.resolve():
        raise ValueError(f"asset URI escapes record directory: {uri!r}")
    return path


def _load_record_path(
    record_path: Path,
    *,
    query_graph: nx.Graph | None,
) -> dict[str, Any] | None:
    try:
        with record_path.open("r", encoding="utf-8") as handle:
            isaac = json.load(handle)
        validate_isaac_record(isaac)
        record_dir = record_path.parent
        assets_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
        for asset in isaac.get("assets", []):
            path = _safe_asset_path(record_dir, str(asset["uri"]))
            if not path.is_file() or _sha256_file(path) != asset["sha256"]:
                return None
            assets_by_id[str(asset["asset_id"])] = (asset, path)

        graph_asset = assets_by_id.get("reaction_graph")
        if graph_asset is None:
            return None
        with graph_asset[1].open("r", encoding="utf-8") as handle:
            stored_graph = reaction_graph_from_payload(json.load(handle))
        if query_graph is not None and not reaction_graphs_isomorphic(stored_graph, query_graph):
            return None

        configuration = isaac["system"]["configuration"]["autokmc"]
        stored_kind = str(configuration["kind"])
        if stored_kind not in _STATE_FILENAMES:
            return None
        if set(configuration.get("state_assets", {})) != set(_STATE_FILENAMES[stored_kind]):
            return None
        states: dict[str, Any] = {}
        for state_name, asset_id in configuration["state_assets"].items():
            asset_info = assets_by_id.get(str(asset_id))
            state_data = configuration["state_data"].get(state_name)
            if asset_info is None or state_data is None:
                return None
            atoms = ase_read(asset_info[1], index=0, format="extxyz")
            states[state_name] = {
                "atoms": atoms,
                "energy_ev": float(state_data["energy_ev"]),
                "properties": state_data.get("properties", {}),
            }

        loaded: dict[str, Any] = {
            "kind": stored_kind,
            "cache_key": configuration["cache_key"],
            "operation": configuration.get("operation", {}),
            "parameters": configuration.get("parameters", {}),
            "states": states,
            "lateral_attributes": configuration.get("lateral_attributes", {}),
            "reaction_graph": stored_graph,
            "isaac_record": isaac,
        }
        neb_asset_id = configuration.get("neb_asset")
        if neb_asset_id:
            neb_info = assets_by_id.get(str(neb_asset_id))
            if neb_info is None:
                return None
            loaded["neb"] = {
                "path": ase_read(neb_info[1], index=":", format="extxyz"),
                "energies_ev": configuration.get("neb_energies_ev", []),
            }
        return loaded
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        _log.debug("Rejected invalid reaction database record %s", record_path, exc_info=True)
        return None


def rebuild_calculation_index(root: str | os.PathLike[str]) -> int:
    """Rebuild SQLite exclusively from checksum-verified record folders."""
    root_path = _database_root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    record_paths = sorted((root_path / "records").glob(f"*/{_RECORD_FILENAME}"))
    with tempfile.TemporaryDirectory(prefix=".index-rebuild-", dir=root_path) as tmp_dir:
        temporary_root = Path(tmp_dir)
        with _connect(temporary_root) as connection:
            for record_path in record_paths:
                loaded = _load_record_path(record_path, query_graph=None)
                if loaded is None:
                    _log.warning("Skipping invalid reaction database record: %s", record_path)
                    continue
                isaac = loaded["isaac_record"]
                configuration = isaac["system"]["configuration"]["autokmc"]
                connection.execute(
                    """
                    INSERT INTO records(
                        record_id, cache_key, kind, operation_key, parameter_hash,
                        graph_hash, record_path, created_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        record_id=excluded.record_id,
                        kind=excluded.kind,
                        operation_key=excluded.operation_key,
                        parameter_hash=excluded.parameter_hash,
                        graph_hash=excluded.graph_hash,
                        record_path=excluded.record_path,
                        created_utc=excluded.created_utc
                    """,
                    (
                        str(isaac["record_id"]),
                        str(configuration["cache_key"]),
                        str(configuration["kind"]),
                        str(configuration["operation_key"]),
                        str(configuration["parameter_hash"]),
                        str(configuration["graph_hash"]),
                        str(record_path.relative_to(root_path)),
                        str(isaac["timestamps"]["created_utc"]),
                    ),
                )
            count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        os.replace(temporary_root / _INDEX_FILENAME, root_path / _INDEX_FILENAME)
    return count


def load_calculation_record(
    root: str | os.PathLike[str],
    kind: str,
    cache_key: str,
    *,
    reaction_graph: nx.Graph | None = None,
    operation: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Find a compatible result, then verify checksums and graph isomorphism."""
    root_path = _database_root(root)
    if not root_path.exists():
        return None
    index_path = root_path / _INDEX_FILENAME
    if not index_path.is_file() and any((root_path / "records").glob(f"*/{_RECORD_FILENAME}")):
        rebuild_calculation_index(root_path)
    query_graph = None if reaction_graph is None else normalise_reaction_graph(reaction_graph)
    rows: list[tuple[str]] = []
    try:
        with _connect(root_path) as connection:
            rows.extend(
                connection.execute(
                    "SELECT record_path FROM records WHERE cache_key = ?",
                    (str(cache_key),),
                ).fetchall()
            )
            if query_graph is not None and operation is not None and parameters is not None:
                graph_digest = reaction_graph_hash(query_graph)
                operation_digest = _operation_key(operation)
                parameter_digest = _hash_json(parameters)
                rows.extend(
                    connection.execute(
                        """
                        SELECT record_path FROM records
                        WHERE kind = ? AND operation_key = ?
                          AND parameter_hash = ? AND graph_hash = ?
                        ORDER BY created_utc DESC
                        """,
                        (str(kind), operation_digest, parameter_digest, graph_digest),
                    ).fetchall()
                )
    except sqlite3.Error:
        _log.warning("Reaction database index is invalid; rebuilding %s", index_path)
        rebuild_calculation_index(root_path)
        with _connect(root_path) as connection:
            rows.extend(
                connection.execute(
                    "SELECT record_path FROM records WHERE cache_key = ?",
                    (str(cache_key),),
                ).fetchall()
            )
            if query_graph is not None and operation is not None and parameters is not None:
                rows.extend(
                    connection.execute(
                        """
                        SELECT record_path FROM records
                        WHERE kind = ? AND operation_key = ?
                          AND parameter_hash = ? AND graph_hash = ?
                        ORDER BY created_utc DESC
                        """,
                        (
                            str(kind), _operation_key(operation),
                            _hash_json(parameters), reaction_graph_hash(query_graph),
                        ),
                    ).fetchall()
                )

    seen: set[str] = set()
    for (relative_path,) in rows:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        loaded = _load_record_path(
            root_path / relative_path,
            query_graph=query_graph,
        )
        if loaded is not None and loaded.get("kind") == str(kind):
            return loaded
    return None


def apply_cached_states(
    lateral_class: Any,
    record: Mapping[str, Any],
    state_mapping: Mapping[str, tuple[str, str]],
) -> bool:
    """Hydrate a lateral class from a verified database record."""
    states = record.get("states", {})
    pending: list[tuple[str, str, float, Atoms, Mapping[str, Any]]] = []
    for state_name, (energy_attr, atoms_attr) in state_mapping.items():
        state = states.get(state_name)
        if not state or not isinstance(state.get("atoms"), Atoms):
            return False
        pending.append(
            (
                energy_attr,
                atoms_attr,
                float(state["energy_ev"]),
                state["atoms"].copy(),
                state.get("properties", {}),
            )
        )

    for energy_attr, atoms_attr, energy, atoms, properties in pending:
        setattr(lateral_class, energy_attr, energy)
        setattr(lateral_class, atoms_attr, atoms)
        for name, value in properties.items():
            if value is not None:
                setattr(lateral_class, name, value)

    neb = record.get("neb")
    if neb:
        lateral_class.atoms_neb_path = [image.copy() for image in neb.get("path", [])]
        lateral_class.neb_path_energies = list(neb.get("energies_ev", []))
    for name, value in record.get("lateral_attributes", {}).items():
        if value is not None:
            setattr(lateral_class, name, value)
    lateral_class.stable = True
    if hasattr(lateral_class, "invalid_reason"):
        lateral_class.invalid_reason = None
    return True


def write_isaac_export(
    root: str | os.PathLike[str] | None,
    target: str | os.PathLike[str],
) -> Path | None:
    """Write a portable JSON array whose elements are ISAAC v1.05 records."""
    if root is None:
        return None
    root_path = _database_root(root)
    if not root_path.exists():
        return None
    target_path = Path(target).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for record_path in sorted((root_path / "records").glob(f"*/{_RECORD_FILENAME}")):
        loaded = _load_record_path(record_path, query_graph=None)
        if loaded is not None:
            exported = copy.deepcopy(loaded["isaac_record"])
            for asset in exported.get("assets", []):
                asset_path = _safe_asset_path(record_path.parent, str(asset["uri"]))
                asset["uri"] = Path(
                    os.path.relpath(asset_path, target_path.parent)
                ).as_posix()
            validate_isaac_record(exported)
            records.append(exported)
    _atomic_json(target_path, records)
    return target_path


__all__ = [
    "ISAAC_RECORD_VERSION",
    "REACTION_DATABASE_SCHEMA",
    "apply_cached_states",
    "atoms_to_json",
    "calculation_cache_key",
    "calculator_identity",
    "initialise_calculation_database",
    "load_calculation_record",
    "make_calculation_record",
    "rebuild_calculation_index",
    "state_payload",
    "validate_isaac_record",
    "write_calculation_record",
    "write_isaac_export",
]
