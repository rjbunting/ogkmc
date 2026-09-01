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
from functools import lru_cache, reduce
from importlib.resources import files as resource_files
import os
import sqlite3
import tempfile
import threading
import uuid
import warnings
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator, Mapping

import networkx as nx
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from autokmc.io._files import atomic_output_path, write_json_atomic
from autokmc.io.atoms import copy_atoms_with_results
from autokmc.io.calculators import (
    cached_calculator_scientific_identity,
    configured_calculator_identity,
    invalidate_calculator_identity,
    primary_calculator,
    stamp_calculator_scientific_identity,
)
from autokmc.io.reaction_graph import (
    normalise_reaction_graph,
    reaction_graph_from_payload,
    reaction_graph_hash,
    reaction_graph_payload,
    reaction_graphs_isomorphic,
)
from autokmc.utils.telemetry import increment, instrument
from autokmc.utils.logging import get_logger


_log = get_logger(__name__)

ISAAC_RECORD_VERSION = "1.05"
REACTION_DATABASE_SCHEMA = "autokmc-reaction-database-v2"
_RECORD_FILENAME = "isaac_record.json"
_GRAPH_FILENAME = "reaction_graph.json"
_INDEX_FILENAME = "index.sqlite3"
_DATABASE_MANIFEST_FILENAME = "database_manifest.json"
_GEOMETRY_FINGERPRINT_SCHEMA = "autokmc-local-geometry-v2"
_SCIENTIFIC_INPUT_FINGERPRINT_SCHEMA = "autokmc-scientific-input-v2"
_INPUT_FRAME_FINGERPRINT_SCHEMA = "autokmc-input-coordinate-frame-v2"

# These values identify an enumeration in one AutoKMC run, not a scientific
# calculation.  Keep this allowlist deliberately narrow: every other input is
# part of the portable scientific identity by default.
_RUN_LOCAL_INPUT_FIELDS = frozenset(
    {
        "a_node_ids",
        "b_node_ids",
        "c_node_ids",
        "iso_class",
        "lateral_class",
        "member_node_ids",
        "node_ids",
        "run_id",
        "site_id",
    }
)

# These fields are still represented in the scientific identity, but are
# canonicalized from ``operation`` so records that mirror them into ``inputs``
# match callers that provide them only as operation metadata.
_SEMANTIC_INPUT_FIELDS = (
    "reactant_smiles",
    "smiles_a",
    "smiles_b",
    "smiles_c",
)

_MODEL_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "model_checkpoint",
        "model",
        "model_file",
        "model_id",
        "model_name",
        "model_path",
        "name_or_path",
        "param_file",
        "parameter_file",
        "potential",
        "potential_file",
        "task_name",
        "weights",
        "weights_path",
    }
)
_DEVICE_PARAMETER_FIELDS = frozenset(
    {"device", "devices", "gpu", "gpu_device", "gpu_devices"}
)
_RUN_DEPENDENT_LATERAL_ATTRIBUTES = frozenset({"gas_pressure_bar"})
_THERMOCHEMISTRY_PARAMETER_FIELDS = frozenset(
    {"free_energy", "free_energy_enabled", "temperature_k"}
)
_THERMOCHEMISTRY_INPUT_FIELDS = frozenset(
    {
        "gas_entropy_ev_per_k",
        "gas_frequencies_ev",
        "gas_gibbs_energy_ev",
        "gas_imaginary_ev",
        "gas_zpe_ev",
    }
)
_PROCESS_LOCAL_IDENTITY = uuid.uuid4().hex
_CALCULATOR_IDENTITY_LOCK = threading.RLock()

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

_OPTIONAL_STATE_FILENAMES: dict[str, dict[str, str]] = {
    "diffusion": {
        "neb_refinement_initial": "neb_refinement_initial.extxyz",
        "neb_refinement_final": "neb_refinement_final.extxyz",
    },
    "bond": {
        "neb_refinement_initial": "neb_refinement_initial.extxyz",
        "neb_refinement_final": "neb_refinement_final.extxyz",
        "state_c_gas_reference": "state_c_gas_reference.extxyz",
        "gas_molecule": "gas_molecule.extxyz",
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
        "atom_arrays": _scientific_atom_arrays(atoms),
        "constraints": _constraint_identity(atoms),
        "info": _jsonable(dict(atoms.info)),
    }


def _model_directory_identity(path: Path) -> dict[str, Any]:
    """Content-hash a directory-valued model artifact recursively."""
    entries: list[dict[str, Any]] = []
    total_size = 0
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if not candidate.is_file():
            continue
        size = int(candidate.stat().st_size)
        total_size += size
        entries.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "sha256": _sha256_file(candidate),
                "size_bytes": size,
            }
        )
    return {
        "artifact_kind": "directory",
        "artifact_sha256": _hash_json(entries),
        "n_files": len(entries),
        "size_bytes": total_size,
    }


def _process_local_object_identity(value: Any) -> dict[str, Any]:
    """Return an identity that can never match an object from another process."""
    return {
        "python_type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "identity_scope": "process-local",
        "process_nonce": _PROCESS_LOCAL_IDENTITY,
        "object_id": id(value),
    }


def _model_identity_value(
    value: Any,
    *,
    artifact_cache: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Turn a model identifier or checkpoint path into portable identity data."""
    if isinstance(value, (str, os.PathLike)):
        candidate = Path(value).expanduser()
        try:
            if candidate.is_file():
                resolved = candidate.resolve()
                cache_key = f"file:{resolved}"
                if artifact_cache is not None and cache_key in artifact_cache:
                    return artifact_cache[cache_key]
                identity = {
                    "artifact_kind": "file",
                    "artifact_sha256": _sha256_file(resolved),
                    "size_bytes": int(resolved.stat().st_size),
                }
                if artifact_cache is not None:
                    artifact_cache[cache_key] = identity
                return identity
            if candidate.is_dir():
                resolved = candidate.resolve()
                cache_key = f"directory:{resolved}"
                if artifact_cache is not None and cache_key in artifact_cache:
                    return artifact_cache[cache_key]
                identity = _model_directory_identity(resolved)
                if artifact_cache is not None:
                    artifact_cache[cache_key] = identity
                return identity
        except OSError:
            # A logical remote model name remains a useful stable identifier.
            pass
    converted = _jsonable(value)
    if (
        isinstance(converted, Mapping)
        and set(converted) == {"python_type"}
    ):
        # Object identity is intentionally process-local.  Persisting only the
        # Python class would allow two opaque learned-model instances to share
        # a cache entry even though their weights cannot be inspected.
        return _process_local_object_identity(value)
    return converted


def _calculator_parameter_identity(
    value: Any,
    *,
    field_name: str = "",
    artifact_cache: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Normalise calculator parameters and content-hash local artifacts.

    Every resolvable path is hashed recursively, including backend-specific
    parameter/config names.  Unknown objects are process-local so opaque model
    instances cannot collapse to a shared class-only identity.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _calculator_parameter_identity(
                item,
                field_name=str(key),
                artifact_cache=artifact_cache,
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _DEVICE_PARAMETER_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [
            _calculator_parameter_identity(
                item,
                field_name=field_name,
                artifact_cache=artifact_cache,
            )
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        converted = [
            _calculator_parameter_identity(
                item,
                field_name=field_name,
                artifact_cache=artifact_cache,
            )
            for item in value
        ]
        return sorted(converted, key=_canonical_json)
    return _model_identity_value(value, artifact_cache=artifact_cache)


@lru_cache(maxsize=1)
def _installed_package_distributions() -> Mapping[str, list[str]]:
    """Return the import-package to distribution mapping without noisy metadata."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib_metadata.packages_distributions()


def _configured_entry_point_versions(
    configured: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve installed versions for every configured factory/import spec."""
    entry_points: set[str] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if (
                    str(key) in {"factory", "import_path"}
                    and isinstance(item, str)
                    and item
                ):
                    entry_points.add(item)
                _collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    _collect(configured)
    module_roots = {
        entry_point.partition(":")[0].partition(".")[0]
        for entry_point in entry_points
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


def calculator_identity(
    calculator: Any,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return a stable calculator/method declaration for compatibility checks.

    Device-placement parameters are intentionally excluded.  Model and
    checkpoint files are represented by their content digest rather than by a
    machine-local path, so copied identical artifacts match.  The verified
    identity is snapshotted on the loaded calculator/pool after the first call;
    this avoids rereading large model artifacts for every site calculation.

    A loaded calculator is assumed to keep immutable scientific parameters.
    Call with ``refresh=True`` (or call
    :func:`invalidate_calculator_identity`) after intentionally mutating a
    calculator or replacing an artifact that the same live instance consumes.
    """
    if refresh:
        invalidate_calculator_identity(calculator)
    cached = cached_calculator_scientific_identity(calculator)
    if cached is not None:
        return copy.deepcopy(cached)

    with _CALCULATOR_IDENTITY_LOCK:
        cached = cached_calculator_scientific_identity(calculator)
        if cached is not None:
            return copy.deepcopy(cached)

        configured = configured_calculator_identity(calculator)
        concrete = primary_calculator(calculator)
        artifact_cache: dict[str, dict[str, Any]] = {}
        identity: dict[str, Any] = {
            "class": (
                f"{concrete.__class__.__module__}."
                f"{concrete.__class__.__qualname__}"
            ),
        }
        entry_point_versions: dict[str, str] = {}
        if configured:
            identity["configured"] = _calculator_parameter_identity(
                configured,
                artifact_cache=artifact_cache,
            )
            entry_point_versions = _configured_entry_point_versions(configured)
            if entry_point_versions:
                identity["entry_point_distributions"] = entry_point_versions
        has_live_scientific_identity = False
        parameters = getattr(concrete, "parameters", None)
        if parameters:
            try:
                parameter_mapping = dict(parameters)
            except (TypeError, ValueError):
                parameter_mapping = {}
            method_parameters = {
                str(key): value
                for key, value in parameter_mapping.items()
                if str(key).lower() not in _DEVICE_PARAMETER_FIELDS
            }
            if method_parameters:
                identity["parameters"] = _calculator_parameter_identity(
                    method_parameters,
                    artifact_cache=artifact_cache,
                )
                has_live_scientific_identity = True
        for name in sorted(_MODEL_IDENTITY_FIELDS):
            value = getattr(concrete, name, None)
            if value not in (None, ""):
                identity[name] = _calculator_parameter_identity(
                    value,
                    field_name=name,
                    artifact_cache=artifact_cache,
                )
                has_live_scientific_identity = True
        if not has_live_scientific_identity and (
            not configured or not entry_point_versions
        ):
            # An unversioned construction recipe plus a class name cannot
            # prove that an opaque factory returned the same scientific
            # calculator.
            identity["opaque_instance"] = _process_local_object_identity(concrete)
        identity["method_digest_sha256"] = _hash_json(identity)
        stamp_calculator_scientific_identity(calculator, identity)
        return copy.deepcopy(identity)


def _fixed_atom_indices(atoms: Atoms) -> set[int]:
    fixed: set[int] = set()
    for constraint in getattr(atoms, "constraints", ()) or ():
        if isinstance(constraint, FixAtoms):
            fixed.update(int(index) for index in constraint.get_indices())
    return fixed


def _constraint_identity(atoms: Atoms) -> list[Any]:
    """Return deterministic declarations for all attached constraints."""
    constraints: list[Any] = []
    for constraint in getattr(atoms, "constraints", ()) or ():
        if hasattr(constraint, "todict"):
            try:
                constraints.append(_jsonable(constraint.todict()))
                continue
            except Exception:
                pass
        constraints.append(
            {
                "python_type": (
                    f"{constraint.__class__.__module__}."
                    f"{constraint.__class__.__qualname__}"
                )
            }
        )
    return constraints


def _scientific_atom_arrays(atoms: Atoms) -> dict[str, Any]:
    """Capture per-atom state that may alter a calculator result."""
    arrays: dict[str, Any] = {
        # ASE treats absent versions of these arrays as all-zero declarations;
        # canonicalize both representations to the same scientific state.
        "initial_charges": _jsonable(atoms.get_initial_charges()),
        "initial_magmoms": _jsonable(atoms.get_initial_magnetic_moments()),
        "tags": _jsonable(atoms.get_tags()),
    }
    builtins = {
        "numbers",
        "positions",
        "initial_charges",
        "initial_magmoms",
        "tags",
    }
    for name, value in sorted(atoms.arrays.items()):
        if name not in builtins:
            arrays[str(name)] = _jsonable(value)
    return arrays


def _atoms_geometry_signature(atoms: Atoms) -> dict[str, Any]:
    """Describe local geometry independent of origin, wrapping, and atom order."""
    fixed = _fixed_atom_indices(atoms)
    arrays = _scientific_atom_arrays(atoms)
    atom_attributes = [
        {
            "symbol": symbol,
            "fixed": index in fixed,
            "arrays": {
                name: values[index]
                for name, values in arrays.items()
            },
        }
        for index, symbol in enumerate(atoms.get_chemical_symbols())
    ]
    labels = [
        _canonical_json(attributes)
        for attributes in atom_attributes
    ]
    use_mic = bool(np.any(atoms.pbc))
    try:
        distances = np.asarray(atoms.get_all_distances(mic=use_mic), dtype=float)
    except (RuntimeError, ValueError):
        # Invalid periodic cells should not make cache persistence fail.  Their
        # Cartesian geometry still yields a conservative, non-portable match.
        distances = np.asarray(atoms.get_all_distances(mic=False), dtype=float)

    pairs = sorted(
        (
            min(labels[left], labels[right]),
            max(labels[left], labels[right]),
            round(float(distances[left, right]), 6),
        )
        for left in range(len(atoms))
        for right in range(left + 1, len(atoms))
    )
    atom_environments = sorted(
        (
            labels[index],
            sorted(
                (labels[other], round(float(distances[index, other]), 6))
                for other in range(len(atoms))
                if other != index
            ),
        )
        for index in range(len(atoms))
    )
    cell = np.asarray(atoms.cell.array, dtype=float)
    # A @ A.T retains every lattice-vector length and mutual angle while being
    # invariant to rigid rotation in Cartesian space.  Singular values alone
    # are insufficient because distinct Gram matrices can share a spectrum.
    cell_metric = np.asarray(cell @ cell.T, dtype=float).round(8).tolist()
    return {
        "labels": sorted(labels),
        "atom_environments": atom_environments,
        "pair_distances_A": pairs,
        "pbc": [bool(value) for value in atoms.pbc],
        "cell_metric_A2": cell_metric,
        "info": _jsonable(dict(atoms.info)),
    }


def _cached_geometry_signature(
    atoms: Atoms,
    cache: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    if cache is None:
        return _atoms_geometry_signature(atoms)
    object_id = id(atoms)
    signature = cache.get(object_id)
    if signature is None:
        signature = _atoms_geometry_signature(atoms)
        cache[object_id] = signature
    return signature


def _geometry_structures(
    value: Any,
    *,
    path: str = "inputs",
    geometry_cache: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(value, Atoms):
        return [
            {
                "path": path,
                "geometry": _cached_geometry_signature(value, geometry_cache),
            }
        ]
    if isinstance(value, Mapping):
        structures: list[dict[str, Any]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            structures.extend(
                _geometry_structures(
                    item,
                    path=f"{path}.{key}",
                    geometry_cache=geometry_cache,
                )
            )
        return structures
    if isinstance(value, (list, tuple)):
        structures = []
        for index, item in enumerate(value):
            structures.extend(
                _geometry_structures(
                    item,
                    path=f"{path}[{index}]",
                    geometry_cache=geometry_cache,
                )
            )
        return structures
    return []


def _input_frame_structures(
    value: Any,
    *,
    path: str = "inputs",
) -> list[dict[str, Any]]:
    """Return exact input coordinates used to guard structure hydration.

    Unlike :func:`_geometry_structures`, this representation deliberately
    retains origin, cell orientation, periodic image, and atom ordering.  A
    cached relaxed structure can only be returned unchanged when those frame
    details match the query.
    """
    if isinstance(value, Atoms):
        return [{"path": path, "coordinates": atoms_to_json(value)}]
    if isinstance(value, Mapping):
        structures: list[dict[str, Any]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            structures.extend(_input_frame_structures(item, path=f"{path}.{key}"))
        return structures
    if isinstance(value, (list, tuple)):
        structures = []
        for index, item in enumerate(value):
            structures.extend(_input_frame_structures(item, path=f"{path}[{index}]"))
        return structures
    return []


def _normalise_scientific_input(
    value: Any,
    *,
    geometry_cache: dict[int, dict[str, Any]] | None = None,
    excluded_fields: frozenset[str] = frozenset(),
) -> Any:
    """Replace structures with invariant geometry and retain other inputs."""
    if isinstance(value, Atoms):
        return {
            "atoms_geometry": _cached_geometry_signature(value, geometry_cache),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_scientific_input(
                item,
                geometry_cache=geometry_cache,
                excluded_fields=excluded_fields,
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if (
                str(key).lower() not in _RUN_LOCAL_INPUT_FIELDS
                and str(key).lower() not in excluded_fields
            )
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalise_scientific_input(
                item,
                geometry_cache=geometry_cache,
                excluded_fields=excluded_fields,
            )
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        converted = [
            _normalise_scientific_input(
                item,
                geometry_cache=geometry_cache,
                excluded_fields=excluded_fields,
            )
            for item in value
        ]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    return _jsonable(value)


def _scientific_input_payload(
    inputs: Mapping[str, Any],
    *,
    operation: Mapping[str, Any] | None,
    geometry_cache: dict[int, dict[str, Any]] | None = None,
    excluded_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    raw_inputs = {
        str(key): value
        for key, value in inputs.items()
        if (
            str(key) not in _SEMANTIC_INPUT_FIELDS
            and str(key).lower() not in excluded_fields
        )
    }
    operation = operation or {}
    semantics: dict[str, Any] = {}
    for key in _SEMANTIC_INPUT_FIELDS:
        value = operation.get(key)
        if value in (None, ""):
            value = inputs.get(key)
        if value not in (None, ""):
            semantics[key] = value
    return {
        "schema": _SCIENTIFIC_INPUT_FINGERPRINT_SCHEMA,
        "semantics": semantics,
        "inputs": _normalise_scientific_input(
            raw_inputs,
            geometry_cache=geometry_cache,
            excluded_fields=excluded_fields,
        ),
    }


def input_geometry_fingerprint(inputs: Mapping[str, Any]) -> str | None:
    """Hash every input structure using a translation/PBC-invariant signature."""
    structures = _geometry_structures(inputs)
    if not structures:
        return None
    return _hash_json(
        {
            "schema": _GEOMETRY_FINGERPRINT_SCHEMA,
            "structures": structures,
        }
    )


def scientific_input_fingerprint(
    inputs: Mapping[str, Any],
    *,
    operation: Mapping[str, Any] | None = None,
) -> str:
    """Hash every scientifically relevant input for portable cache matching.

    Atomic structures are replaced by translation-, wrapping-, rotation-, and
    atom-order-invariant geometry signatures.  Scalar and structured inputs
    such as charge, spin declarations, and gas energies remain in the digest.
    Only the explicitly enumerated run-local identifiers above are omitted.
    """
    return _hash_json(
        _scientific_input_payload(inputs, operation=operation)
    )


def input_coordinate_frame_fingerprint(inputs: Mapping[str, Any]) -> str | None:
    """Hash exact input coordinates for safe reuse of structure outputs."""
    structures = _input_frame_structures(inputs)
    if not structures:
        return None
    return _hash_json(
        {
            "schema": _INPUT_FRAME_FINGERPRINT_SCHEMA,
            "structures": structures,
        }
    )


@dataclass(frozen=True)
class _CalculationFingerprints:
    geometry: str | None
    scientific_input: str
    electronic_scientific_input: str
    input_frame: str | None


@dataclass
class CalculationFingerprintMemo:
    """Single-request carrier for expensive portable fingerprints.

    Stability workflows pass one memo from lookup through persistence.  The
    exact cache key binds the memo to the request, preventing accidental reuse
    for another calculation while avoiding a second O(A^2) geometry traversal
    after a portable miss.
    """

    _cache_key: str | None = None
    _fingerprints: _CalculationFingerprints | None = None

    def get(
        self,
        cache_key: str,
        inputs: Mapping[str, Any],
        *,
        operation: Mapping[str, Any] | None = None,
    ) -> _CalculationFingerprints:
        key = str(cache_key)
        if self._cache_key != key or self._fingerprints is None:
            self._fingerprints = _calculation_fingerprints(
                inputs,
                operation=operation,
            )
            self._cache_key = key
        return self._fingerprints


def _calculation_fingerprints(
    inputs: Mapping[str, Any],
    *,
    operation: Mapping[str, Any] | None = None,
) -> _CalculationFingerprints:
    """Compute all portable request fingerprints in one structure traversal."""
    geometry_cache: dict[int, dict[str, Any]] = {}
    structures = _geometry_structures(inputs, geometry_cache=geometry_cache)
    geometry = (
        None
        if not structures
        else _hash_json(
            {
                "schema": _GEOMETRY_FINGERPRINT_SCHEMA,
                "structures": structures,
            }
        )
    )
    scientific_input = _hash_json(
        _scientific_input_payload(
            inputs,
            operation=operation,
            geometry_cache=geometry_cache,
        )
    )
    electronic_scientific_input = _hash_json(
        _scientific_input_payload(
            inputs,
            operation=operation,
            geometry_cache=geometry_cache,
            excluded_fields=_THERMOCHEMISTRY_INPUT_FIELDS,
        )
    )
    frame_structures = _input_frame_structures(inputs)
    input_frame = (
        None
        if not frame_structures
        else _hash_json(
            {
                "schema": _INPUT_FRAME_FINGERPRINT_SCHEMA,
                "structures": frame_structures,
            }
        )
    )
    return _CalculationFingerprints(
        geometry=geometry,
        scientific_input=scientific_input,
        electronic_scientific_input=electronic_scientific_input,
        input_frame=input_frame,
    )


def _electronic_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Drop thermochemistry-only controls from endpoint/NEB identity."""
    return {
        str(key): value
        for key, value in parameters.items()
        if str(key).lower() not in _THERMOCHEMISTRY_PARAMETER_FIELDS
    }


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
        "atoms": copy_atoms_with_results(atoms),
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


_INDEX_SCHEMA_LOCK = threading.RLock()
_INITIALISED_INDEX_FILES: set[tuple[str, int, int]] = set()
_THREAD_CONNECTIONS = threading.local()
_INDEX_METADATA_COLUMNS: dict[str, str] = {
    "geometry_hash": "TEXT",
    "scientific_input_hash": "TEXT",
    "electronic_scientific_input_hash": "TEXT",
    "calculator_digest": "TEXT",
    "input_frame_hash": "TEXT",
    "electronic_parameter_hash": "TEXT",
}


@dataclass
class _ThreadConnection:
    connection: sqlite3.Connection
    file_identity: tuple[int, int]


def _index_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_dev), int(stat.st_ino)


def _thread_connection_map() -> dict[str, _ThreadConnection]:
    pid = os.getpid()
    if getattr(_THREAD_CONNECTIONS, "pid", None) != pid:
        # Never carry SQLite handles into a forked process.
        for item in getattr(_THREAD_CONNECTIONS, "connections", {}).values():
            try:
                item.connection.close()
            except sqlite3.Error:
                pass
        _THREAD_CONNECTIONS.pid = pid
        _THREAD_CONNECTIONS.connections = {}
    return _THREAD_CONNECTIONS.connections


def close_calculation_cache_connections(
    root: str | os.PathLike[str] | None = None,
) -> None:
    """Close reusable SQLite handles owned by the calling thread.

    Worker threads maintain independent connections.  Normal workflows can
    leave them open for the run; tests or applications that replace/delete a
    live database may call this explicit lifecycle hook.
    """
    connections = _thread_connection_map()
    if root is None:
        keys = list(connections)
    else:
        keys = [str(_database_root(root) / _INDEX_FILENAME)]
    for key in keys:
        item = connections.pop(key, None)
        if item is not None:
            item.connection.close()
    # This hook explicitly permits callers to replace or delete an index.
    # Linux may immediately reuse the old inode for the replacement, so the
    # path/device/inode schema cache cannot safely survive this lifecycle
    # boundary.
    with _INDEX_SCHEMA_LOCK:
        if root is None:
            _INITIALISED_INDEX_FILES.clear()
        else:
            index_path = keys[0]
            stale_entries = {
                entry
                for entry in _INITIALISED_INDEX_FILES
                if entry[0] == index_path
            }
            _INITIALISED_INDEX_FILES.difference_update(stale_entries)


def _backfill_index_metadata(root: Path, connection: sqlite3.Connection) -> None:
    """Populate newly added, rebuildable metadata columns from record JSON."""
    rows = connection.execute(
        """
        SELECT cache_key, record_path
        FROM records
        WHERE scientific_input_hash IS NULL
           OR electronic_scientific_input_hash IS NULL
           OR electronic_parameter_hash IS NULL
           OR calculator_digest IS NULL
           OR input_frame_hash IS NULL
        """
    ).fetchall()
    for cache_key, relative_path in rows:
        try:
            record_path = root / str(relative_path)
            isaac = json.loads(record_path.read_text(encoding="utf-8"))
            configuration = isaac["system"]["configuration"]["autokmc"]
            parameters = configuration.get("parameters", {})
            calculator = parameters.get("calculator")
            calculator_digest = (
                None if calculator is None else _hash_json(calculator)
            )
            electronic_parameter_hash = _hash_json(
                _electronic_parameters(parameters)
            )
            scientific_input_hash = configuration.get("scientific_input_hash")
            electronic_scientific_input_hash = configuration.get(
                "electronic_scientific_input_hash"
            )
            if electronic_scientific_input_hash is None:
                # Legacy records did not persist enough raw structure data to
                # derive this value safely.  A full scientific hash remains a
                # valid electronic hash only when no thermochemistry-only
                # inputs were present.
                raw_inputs = configuration.get("inputs", {})
                if not any(
                    key in raw_inputs for key in _THERMOCHEMISTRY_INPUT_FIELDS
                ):
                    electronic_scientific_input_hash = scientific_input_hash
            connection.execute(
                """
                UPDATE records
                SET geometry_hash = ?,
                    scientific_input_hash = ?,
                    electronic_scientific_input_hash = ?,
                    calculator_digest = ?,
                    input_frame_hash = ?,
                    electronic_parameter_hash = ?
                WHERE cache_key = ?
                """,
                (
                    configuration.get("geometry_hash"),
                    scientific_input_hash,
                    electronic_scientific_input_hash,
                    configuration.get("calculator_digest", calculator_digest),
                    configuration.get("input_frame_hash"),
                    electronic_parameter_hash,
                    str(cache_key),
                ),
            )
        except (OSError, TypeError, KeyError, json.JSONDecodeError):
            # The immutable record will be rejected if selected.  The index is
            # merely an accelerator and remains rebuildable.
            continue


def _ensure_index_schema(root: Path, connection: sqlite3.Connection) -> None:
    index_path = root / _INDEX_FILENAME
    identity = _index_file_identity(index_path)
    cache_key = (
        str(index_path),
        -1 if identity is None else identity[0],
        -1 if identity is None else identity[1],
    )
    if cache_key in _INITIALISED_INDEX_FILES:
        return
    with _INDEX_SCHEMA_LOCK:
        identity = _index_file_identity(index_path)
        cache_key = (
            str(index_path),
            -1 if identity is None else identity[0],
            -1 if identity is None else identity[1],
        )
        if cache_key in _INITIALISED_INDEX_FILES:
            return
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                operation_key TEXT NOT NULL,
                parameter_hash TEXT NOT NULL,
                graph_hash TEXT NOT NULL,
                geometry_hash TEXT,
                scientific_input_hash TEXT,
                electronic_scientific_input_hash TEXT,
                calculator_digest TEXT,
                input_frame_hash TEXT,
                electronic_parameter_hash TEXT,
                record_path TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(records)").fetchall()
        }
        for name, declaration in _INDEX_METADATA_COLUMNS.items():
            if name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE records ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS records_graph_lookup
            ON records(kind, operation_key, parameter_hash, graph_hash)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS records_portable_lookup
            ON records(
                kind, operation_key, parameter_hash, graph_hash,
                geometry_hash, scientific_input_hash, calculator_digest,
                input_frame_hash
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS records_electronic_lookup
            ON records(
                kind, operation_key, electronic_parameter_hash, graph_hash,
                geometry_hash, electronic_scientific_input_hash,
                calculator_digest, input_frame_hash
            )
            """
        )
        _backfill_index_metadata(root, connection)
        connection.commit()
        identity = _index_file_identity(index_path)
        _INITIALISED_INDEX_FILES.add(
            (
                str(index_path),
                -1 if identity is None else identity[0],
                -1 if identity is None else identity[1],
            )
        )


def _open_index_connection(
    root: Path,
    *,
    reuse: bool,
) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / _INDEX_FILENAME
    if not reuse:
        connection = sqlite3.connect(index_path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        _ensure_index_schema(root, connection)
        return connection

    key = str(index_path)
    connections = _thread_connection_map()
    existing = connections.get(key)
    current_identity = _index_file_identity(index_path)
    if (
        existing is not None
        and current_identity is not None
        and existing.file_identity == current_identity
    ):
        return existing.connection
    if existing is not None:
        try:
            existing.connection.close()
        finally:
            connections.pop(key, None)
    connection = sqlite3.connect(index_path, timeout=30.0)
    connection.execute("PRAGMA foreign_keys=ON")
    _ensure_index_schema(root, connection)
    identity = _index_file_identity(index_path)
    if identity is None:  # pragma: no cover - SQLite always creates the file
        connection.close()
        raise sqlite3.OperationalError(f"index was not created: {index_path}")
    connections[key] = _ThreadConnection(connection, identity)
    return connection


@contextmanager
def _connect(
    root: Path,
    *,
    reuse: bool = True,
) -> Iterator[sqlite3.Connection]:
    connection = _open_index_connection(root, reuse=reuse)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if not reuse:
            connection.close()


def _atomic_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value, sort_keys=True, transform=_jsonable)


def _safe_atoms_copy(atoms: Atoms, energy_ev: float | None = None) -> Atoms:
    snapshot = copy_atoms_with_results(atoms)
    snapshot.info = {
        str(key): _jsonable(value)
        for key, value in getattr(snapshot, "info", {}).items()
    }
    if energy_ev is not None:
        snapshot.info["autokmc_energy_ev"] = float(energy_ev)
    return snapshot


def _atomic_extxyz(path: Path, images: Atoms | list[Atoms]) -> None:
    with atomic_output_path(path) as temporary:
        ase_write(temporary, images, format="extxyz")


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
    electronic_parameter_hash: str,
    geometry_hash: str | None,
    scientific_input_hash: str,
    electronic_scientific_input_hash: str,
    input_frame_hash: str | None,
    calculator_digest: str | None,
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
        "electronic_parameter_hash": electronic_parameter_hash,
        "geometry_hash": geometry_hash,
        "scientific_input_hash": scientific_input_hash,
        "electronic_scientific_input_hash": electronic_scientific_input_hash,
        "input_frame_hash": input_frame_hash,
        "calculator_digest": calculator_digest,
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


@lru_cache(maxsize=1)
def _isaac_validator():
    """Compile the immutable vendored ISAAC schema once per process."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise RuntimeError("ISAAC validation requires the jsonschema package") from exc
    schema_path = resource_files("autokmc").joinpath("schema/isaac_record_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_isaac_record(record: Mapping[str, Any]) -> None:
    """Validate against the vendored official ISAAC v1.05 JSON schema."""
    validator = _isaac_validator()
    errors = sorted(validator.iter_errors(dict(record)), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"ISAAC schema validation failed at {location}: {error.message}")


@instrument("calculation_cache.write")
def write_calculation_record(
    root: str | os.PathLike[str],
    kind: str,
    cache_key: str,
    record: Mapping[str, Any],
    *,
    fingerprint_memo: CalculationFingerprintMemo | None = None,
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
    electronic_parameter_digest = _hash_json(
        _electronic_parameters(record["parameters"])
    )
    fingerprints = (
        _calculation_fingerprints(
            record["inputs"],
            operation=record["operation"],
        )
        if fingerprint_memo is None
        else fingerprint_memo.get(
            cache_key,
            record["inputs"],
            operation=record["operation"],
        )
    )
    calculator = record["parameters"].get("calculator")
    calculator_digest = None if calculator is None else _hash_json(calculator)
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
    state_filenames = dict(_STATE_FILENAMES[kind])
    state_filenames.update(
        {
            state_name: filename
            for state_name, filename in _OPTIONAL_STATE_FILENAMES.get(
                kind, {}
            ).items()
            if state_name in record["states"]
        }
    )
    for state_name, filename in state_filenames.items():
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
        electronic_parameter_hash=electronic_parameter_digest,
        geometry_hash=fingerprints.geometry,
        scientific_input_hash=fingerprints.scientific_input,
        electronic_scientific_input_hash=(
            fingerprints.electronic_scientific_input
        ),
        input_frame_hash=fingerprints.input_frame,
        calculator_digest=calculator_digest,
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
                graph_hash, geometry_hash, scientific_input_hash,
                electronic_scientific_input_hash, calculator_digest,
                input_frame_hash, electronic_parameter_hash, record_path,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                record_id=excluded.record_id,
                kind=excluded.kind,
                operation_key=excluded.operation_key,
                parameter_hash=excluded.parameter_hash,
                graph_hash=excluded.graph_hash,
                geometry_hash=excluded.geometry_hash,
                scientific_input_hash=excluded.scientific_input_hash,
                electronic_scientific_input_hash=excluded.electronic_scientific_input_hash,
                calculator_digest=excluded.calculator_digest,
                input_frame_hash=excluded.input_frame_hash,
                electronic_parameter_hash=excluded.electronic_parameter_hash,
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
                fingerprints.geometry,
                fingerprints.scientific_input,
                fingerprints.electronic_scientific_input,
                calculator_digest,
                fingerprints.input_frame,
                electronic_parameter_digest,
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
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        with record_path.open("r", encoding="utf-8") as handle:
            isaac = json.load(handle)
        validate_isaac_record(isaac)
        configuration = isaac["system"]["configuration"]["autokmc"]
        for name, expected in (expected_metadata or {}).items():
            if configuration.get(name) != expected:
                return None

        record_dir = record_path.parent
        assets_by_id: dict[str, tuple[dict[str, Any], Path]] = {}
        for asset in isaac.get("assets", []):
            path = _safe_asset_path(record_dir, str(asset["uri"]))
            assets_by_id[str(asset["asset_id"])] = (asset, path)

        graph_asset = assets_by_id.get("reaction_graph")
        if graph_asset is None:
            return None
        graph_descriptor, graph_path = graph_asset
        if (
            not graph_path.is_file()
            or _sha256_file(graph_path) != graph_descriptor["sha256"]
        ):
            return None
        with graph_asset[1].open("r", encoding="utf-8") as handle:
            stored_graph = reaction_graph_from_payload(json.load(handle))
        if query_graph is not None and not reaction_graphs_isomorphic(stored_graph, query_graph):
            return None

        # Only candidates that passed cheap indexed/configuration metadata and
        # authoritative graph isomorphism pay to verify and hydrate every
        # structure asset.
        for asset_id, (asset, path) in assets_by_id.items():
            if asset_id == "reaction_graph":
                continue
            if not path.is_file() or _sha256_file(path) != asset["sha256"]:
                return None

        stored_kind = str(configuration["kind"])
        if stored_kind not in _STATE_FILENAMES:
            return None
        state_asset_names = set(configuration.get("state_assets", {}))
        required_state_names = set(_STATE_FILENAMES[stored_kind])
        allowed_state_names = required_state_names | set(
            _OPTIONAL_STATE_FILENAMES.get(stored_kind, {})
        )
        if (
            not required_state_names.issubset(state_asset_names)
            or not state_asset_names.issubset(allowed_state_names)
        ):
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
            "parameter_hash": configuration.get("parameter_hash"),
            "electronic_parameter_hash": configuration.get(
                "electronic_parameter_hash"
            ),
            "geometry_hash": configuration.get("geometry_hash"),
            "scientific_input_hash": configuration.get("scientific_input_hash"),
            "electronic_scientific_input_hash": configuration.get(
                "electronic_scientific_input_hash"
            ),
            "input_frame_hash": configuration.get("input_frame_hash"),
            "calculator_digest": configuration.get("calculator_digest"),
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
    close_calculation_cache_connections(root_path)
    record_paths = sorted((root_path / "records").glob(f"*/{_RECORD_FILENAME}"))
    with tempfile.TemporaryDirectory(prefix=".index-rebuild-", dir=root_path) as tmp_dir:
        temporary_root = Path(tmp_dir)
        with _connect(temporary_root, reuse=False) as connection:
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
                        graph_hash, geometry_hash, scientific_input_hash,
                        electronic_scientific_input_hash, calculator_digest,
                        input_frame_hash, electronic_parameter_hash, record_path,
                        created_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        record_id=excluded.record_id,
                        kind=excluded.kind,
                        operation_key=excluded.operation_key,
                        parameter_hash=excluded.parameter_hash,
                        graph_hash=excluded.graph_hash,
                        geometry_hash=excluded.geometry_hash,
                        scientific_input_hash=excluded.scientific_input_hash,
                        electronic_scientific_input_hash=excluded.electronic_scientific_input_hash,
                        calculator_digest=excluded.calculator_digest,
                        input_frame_hash=excluded.input_frame_hash,
                        electronic_parameter_hash=excluded.electronic_parameter_hash,
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
                        configuration.get("geometry_hash"),
                        configuration.get("scientific_input_hash"),
                        configuration.get(
                            "electronic_scientific_input_hash"
                        ),
                        configuration.get("calculator_digest"),
                        configuration.get("input_frame_hash"),
                        configuration.get("electronic_parameter_hash")
                        or _hash_json(
                            _electronic_parameters(
                                configuration.get("parameters", {})
                            )
                        ),
                        str(record_path.relative_to(root_path)),
                        str(isaac["timestamps"]["created_utc"]),
                    ),
                )
            count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        os.replace(temporary_root / _INDEX_FILENAME, root_path / _INDEX_FILENAME)
    return count


@instrument("calculation_cache.lookup")
def load_calculation_record(
    root: str | os.PathLike[str],
    kind: str,
    cache_key: str,
    *,
    reaction_graph: nx.Graph | None = None,
    operation: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    allow_electronic_match: bool = False,
    fingerprint_memo: CalculationFingerprintMemo | None = None,
) -> dict[str, Any] | None:
    """Find a verified exact or portable calculation result.

    Exact cache keys retain their original behavior.  Portable graph-search
    fallback additionally requires a normalized identity for every scientific
    input.  Structure-bearing results require the query's exact input
    coordinate frame because persisted outputs cannot otherwise be mapped into
    translated, rotated, wrapped, or permuted query coordinates safely.
    """
    root_path = _database_root(root)
    if not root_path.exists():
        increment("calculation_cache.misses")
        return None
    index_path = root_path / _INDEX_FILENAME
    if (
        not index_path.is_file()
        and any((root_path / "records").glob(f"*/{_RECORD_FILENAME}"))
    ):
        rebuild_calculation_index(root_path)
    query_graph = (
        None
        if reaction_graph is None
        else normalise_reaction_graph(reaction_graph)
    )

    def _exact_paths() -> list[str]:
        with _connect(root_path) as connection:
            return [
                str(relative_path)
                for (relative_path,) in connection.execute(
                    "SELECT record_path FROM records WHERE cache_key = ?",
                    (str(cache_key),),
                ).fetchall()
            ]

    try:
        exact_paths = _exact_paths()
    except sqlite3.Error:
        _log.warning("Reaction database index is invalid; rebuilding %s", index_path)
        rebuild_calculation_index(root_path)
        exact_paths = _exact_paths()

    # Exact-key lookup is intentionally first.  It avoids every portable
    # O(A^2) geometry fingerprint on the common same-run hit path.
    for relative_path in exact_paths:
        loaded = _load_record_path(
            root_path / relative_path,
            query_graph=query_graph,
            expected_metadata={
                "kind": str(kind),
                "cache_key": str(cache_key),
            },
        )
        if loaded is not None:
            loaded["_cache_match"] = "exact"
            increment("calculation_cache.hits")
            increment("calculation_cache.exact_hits")
            return loaded

    if (
        query_graph is None
        or operation is None
        or parameters is None
        or inputs is None
    ):
        increment("calculation_cache.misses")
        return None

    graph_digest = reaction_graph_hash(query_graph)
    operation_digest = _operation_key(operation)
    parameter_digest = _hash_json(parameters)
    electronic_parameter_digest = _hash_json(
        _electronic_parameters(parameters)
    )
    calculator = parameters.get("calculator")
    calculator_digest = None if calculator is None else _hash_json(calculator)

    def _has_portable_prefix() -> bool:
        """Check cheap indexed identity before building O(A^2) fingerprints."""
        with _connect(root_path) as connection:
            if connection.execute(
                """
                SELECT 1 FROM records
                WHERE kind = ? AND operation_key = ?
                  AND parameter_hash = ? AND graph_hash = ?
                  AND calculator_digest IS ?
                LIMIT 1
                """,
                (
                    str(kind),
                    operation_digest,
                    parameter_digest,
                    graph_digest,
                    calculator_digest,
                ),
            ).fetchone():
                return True
            if not allow_electronic_match:
                return False
            return (
                connection.execute(
                    """
                    SELECT 1 FROM records
                    WHERE kind = ? AND operation_key = ?
                      AND electronic_parameter_hash = ? AND graph_hash = ?
                      AND calculator_digest IS ?
                    LIMIT 1
                    """,
                    (
                        str(kind),
                        operation_digest,
                        electronic_parameter_digest,
                        graph_digest,
                        calculator_digest,
                    ),
                ).fetchone()
                is not None
            )

    try:
        has_portable_prefix = _has_portable_prefix()
    except sqlite3.Error:
        _log.warning("Reaction database index is invalid; rebuilding %s", index_path)
        rebuild_calculation_index(root_path)
        has_portable_prefix = _has_portable_prefix()
    if not has_portable_prefix:
        increment("calculation_cache.misses")
        return None

    fingerprints = (
        _calculation_fingerprints(inputs, operation=operation)
        if fingerprint_memo is None
        else fingerprint_memo.get(
            cache_key,
            inputs,
            operation=operation,
        )
    )
    if fingerprints.geometry is None or fingerprints.input_frame is None:
        increment("calculation_cache.misses")
        return None

    def _portable_rows() -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        with _connect(root_path) as connection:
            rows.extend(
                (str(relative_path), "portable")
                for (relative_path,) in connection.execute(
                    """
                    SELECT record_path FROM records
                    WHERE kind = ? AND operation_key = ?
                      AND parameter_hash = ? AND graph_hash = ?
                      AND geometry_hash = ?
                      AND scientific_input_hash = ?
                      AND calculator_digest IS ?
                      AND input_frame_hash = ?
                    ORDER BY created_utc DESC
                    """,
                    (
                        str(kind),
                        operation_digest,
                        parameter_digest,
                        graph_digest,
                        fingerprints.geometry,
                        fingerprints.scientific_input,
                        calculator_digest,
                        fingerprints.input_frame,
                    ),
                ).fetchall()
            )
            if allow_electronic_match:
                rows.extend(
                    (str(relative_path), "electronic")
                    for (relative_path,) in connection.execute(
                        """
                        SELECT record_path FROM records
                        WHERE kind = ? AND operation_key = ?
                          AND electronic_parameter_hash = ?
                          AND graph_hash = ?
                          AND geometry_hash = ?
                          AND electronic_scientific_input_hash = ?
                          AND calculator_digest IS ?
                          AND input_frame_hash = ?
                        ORDER BY created_utc DESC
                        """,
                        (
                            str(kind),
                            operation_digest,
                            electronic_parameter_digest,
                            graph_digest,
                            fingerprints.geometry,
                            fingerprints.electronic_scientific_input,
                            calculator_digest,
                            fingerprints.input_frame,
                        ),
                    ).fetchall()
                )
        return rows

    try:
        rows = _portable_rows()
    except sqlite3.Error:
        _log.warning("Reaction database index is invalid; rebuilding %s", index_path)
        rebuild_calculation_index(root_path)
        rows = _portable_rows()

    seen = set(exact_paths)
    for relative_path, match_kind in rows:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        expected = {
            "kind": str(kind),
            "operation_key": operation_digest,
            "graph_hash": graph_digest,
            "geometry_hash": fingerprints.geometry,
            "calculator_digest": calculator_digest,
            "input_frame_hash": fingerprints.input_frame,
        }
        if match_kind == "portable":
            expected.update(
                {
                    "parameter_hash": parameter_digest,
                    "scientific_input_hash": fingerprints.scientific_input,
                }
            )
        else:
            expected.update(
                {
                    "electronic_parameter_hash": electronic_parameter_digest,
                    "electronic_scientific_input_hash": (
                        fingerprints.electronic_scientific_input
                    ),
                }
            )
        loaded = _load_record_path(
            root_path / relative_path,
            query_graph=query_graph,
            expected_metadata=expected,
        )
        if loaded is None:
            continue
        loaded["_cache_match"] = match_kind
        increment("calculation_cache.hits")
        increment(
            "calculation_cache.electronic_hits"
            if match_kind == "electronic"
            else "calculation_cache.portable_hits"
        )
        return loaded
    increment("calculation_cache.misses")
    return None


def apply_cached_states(
    lateral_class: Any,
    record: Mapping[str, Any],
    state_mapping: Mapping[str, tuple[str, str]],
    *,
    include_properties: bool = True,
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
                copy_atoms_with_results(state["atoms"]),
                state.get("properties", {}),
            )
        )

    for energy_attr, atoms_attr, energy, atoms, properties in pending:
        setattr(lateral_class, energy_attr, energy)
        setattr(lateral_class, atoms_attr, atoms)
        if include_properties:
            for name, value in properties.items():
                if (
                    value is not None
                    and str(name) not in _RUN_DEPENDENT_LATERAL_ATTRIBUTES
                ):
                    setattr(lateral_class, name, value)
        else:
            # An electronic-only match must not leave temperature-dependent
            # values from either the cached record or a previously populated
            # lateral object visible to the current request.
            for name in properties:
                if str(name) not in _RUN_DEPENDENT_LATERAL_ATTRIBUTES:
                    setattr(lateral_class, name, None)

    neb = record.get("neb")
    if neb:
        lateral_class.atoms_neb_path = [
            copy_atoms_with_results(image) for image in neb.get("path", [])
        ]
        lateral_class.neb_path_energies = list(neb.get("energies_ev", []))
    for state_name, atoms_attr in (
        ("neb_refinement_initial", "atoms_neb_refinement_initial"),
        ("neb_refinement_final", "atoms_neb_refinement_final"),
    ):
        state = states.get(state_name)
        if state and isinstance(state.get("atoms"), Atoms):
            setattr(
                lateral_class,
                atoms_attr,
                copy_atoms_with_results(state["atoms"]),
            )
    for name, value in record.get("lateral_attributes", {}).items():
        if (
            value is not None
            and str(name) not in _RUN_DEPENDENT_LATERAL_ATTRIBUTES
        ):
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
    "CalculationFingerprintMemo",
    "apply_cached_states",
    "atoms_to_json",
    "calculation_cache_key",
    "calculator_identity",
    "close_calculation_cache_connections",
    "initialise_calculation_database",
    "input_coordinate_frame_fingerprint",
    "input_geometry_fingerprint",
    "invalidate_calculator_identity",
    "load_calculation_record",
    "make_calculation_record",
    "rebuild_calculation_index",
    "scientific_input_fingerprint",
    "state_payload",
    "validate_isaac_record",
    "write_calculation_record",
    "write_isaac_export",
]
