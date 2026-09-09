"""Load user-supplied catalyst structures through ASE."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from autokmc.io.extxyz import read_atoms as ase_read


class StructureInputError(ValueError):
    """A configured catalyst structure cannot be loaded safely."""


def resolve_structure_path(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve a structure path relative to its config file or the current cwd."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise StructureInputError(
            "structure.path must be a non-empty path string"
        )
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = (
        Path.cwd()
        if config_path is None
        else Path(config_path).expanduser().resolve().parent
    )
    return (base / candidate).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_indices(
    values: Iterable[Any],
    *,
    origin: str,
) -> set[int]:
    indices: set[int] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise StructureInputError(
            f"{origin} must be a sequence of integer atom indices"
        ) from exc
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise StructureInputError(
                f"{origin} must contain only integer atom indices; "
                f"got {value!r}"
            )
        indices.add(int(value))
    return indices


def _constraint_indices(atoms: Atoms) -> set[int]:
    indices: set[int] = set()
    for position, constraint in enumerate(atoms.constraints):
        # Several partial/internal-coordinate constraints expose
        # ``get_indices()`` too (for example FixBondLengths and
        # FixCartesian).  Only FixAtoms guarantees that every Cartesian
        # degree of freedom for the reported atoms is frozen.
        if not isinstance(constraint, FixAtoms):
            continue
        getter = getattr(constraint, "get_indices", None)
        if not callable(getter):  # pragma: no cover - FixAtoms contract
            continue
        try:
            values = getter()
        except Exception as exc:
            raise StructureInputError(
                "could not read frozen atom indices from ASE constraint "
                f"{position} ({type(constraint).__name__}): {exc}"
            ) from exc
        indices.update(
            _normalise_indices(
                values,
                origin=(
                    f"ASE constraint {position} "
                    f"({type(constraint).__name__}).get_indices()"
                ),
            )
        )
    return indices


def resolve_frozen_indices(
    atoms: Atoms,
    *,
    frozen_indices: Sequence[int] | None = None,
) -> list[int]:
    """Resolve and validate the frozen atoms for a loaded structure.

    An explicitly configured list has precedence, including an empty list.
    Otherwise the result is the union of ``atoms.info["frozen_indices"]`` and
    indices held by ASE ``FixAtoms`` constraints.  Partial-coordinate and
    internal-coordinate constraints are deliberately not promoted to fully
    frozen atoms.
    """
    if frozen_indices is not None:
        resolved = _normalise_indices(
            frozen_indices,
            origin="structure.frozen_indices",
        )
    else:
        info_values = atoms.info.get("frozen_indices", ())
        if info_values is None:
            info_values = ()
        resolved = _normalise_indices(
            info_values,
            origin="atoms.info['frozen_indices']",
        )
        resolved.update(_constraint_indices(atoms))

    invalid = sorted(index for index in resolved if index < 0 or index >= len(atoms))
    if invalid:
        raise StructureInputError(
            "frozen atom indices are outside the loaded structure's valid "
            f"range 0..{len(atoms) - 1}: {invalid}"
        )
    result = sorted(resolved)
    atoms.info["frozen_indices"] = result
    return result


def load_structure_file(
    path: str | Path,
    *,
    format: str | None = None,
    index: int = -1,
    frozen_indices: Sequence[int] | None = None,
    config_path: str | Path | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    """Load and validate one ASE-readable catalyst structure.

    Relative paths are interpreted relative to ``config_path`` when supplied,
    and relative to the current working directory for programmatic configs.
    The returned provenance dictionary records the resolved input used for the
    run.
    """
    resolved_path = resolve_structure_path(path, config_path=config_path)
    if not resolved_path.exists():
        raise StructureInputError(
            f"structure file does not exist: {resolved_path}"
        )
    if not resolved_path.is_file():
        raise StructureInputError(
            f"structure path is not a regular file: {resolved_path}"
        )
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise StructureInputError(
            f"structure.index must be an integer frame index; got {index!r}"
        )

    if format is not None and not isinstance(format, str):
        raise StructureInputError(
            f"structure.format must be a string or null; got {format!r}"
        )
    ase_format = None if format is None else format.strip()
    if ase_format == "":
        raise StructureInputError(
            "structure.format must be a non-empty string or null"
        )
    try:
        loaded = ase_read(
            str(resolved_path),
            index=int(index),
            format=ase_format,
        )
    except Exception as exc:
        format_label = "auto-detected" if ase_format is None else repr(ase_format)
        raise StructureInputError(
            f"could not read structure file {resolved_path} with ASE "
            f"(format={format_label}, index={int(index)}): {exc}"
        ) from exc

    if not isinstance(loaded, Atoms):
        raise StructureInputError(
            f"ASE returned {type(loaded).__name__} for structure file "
            f"{resolved_path}; select exactly one frame with structure.index"
        )
    if len(loaded) == 0:
        raise StructureInputError(
            f"structure file contains an empty Atoms object: {resolved_path}"
        )

    positions = np.asarray(loaded.get_positions(), dtype=float)
    cell = np.asarray(loaded.get_cell(), dtype=float)
    if not np.all(np.isfinite(positions)):
        raise StructureInputError(
            f"structure file contains non-finite atomic positions: {resolved_path}"
        )
    if not np.all(np.isfinite(cell)):
        raise StructureInputError(
            f"structure file contains non-finite cell vectors: {resolved_path}"
        )

    # Loaded calculators may refer to serialized results or a backend that is
    # not valid in this process.  KMC always uses the separately configured
    # calculator resource.
    loaded.calc = None
    resolved_frozen_indices = resolve_frozen_indices(
        loaded,
        frozen_indices=frozen_indices,
    )

    try:
        size_bytes = resolved_path.stat().st_size
        sha256 = _sha256_file(resolved_path)
    except OSError as exc:
        raise StructureInputError(
            f"could not fingerprint structure file {resolved_path}: {exc}"
        ) from exc

    source = {
        "kind": "file",
        "path": str(resolved_path),
        "format": ase_format,
        "index": int(index),
        "sha256": sha256,
        "size_bytes": int(size_bytes),
        "chemical_formula": loaded.get_chemical_formula(),
        "frozen_indices": resolved_frozen_indices,
        "frozen_count": len(resolved_frozen_indices),
    }
    return loaded, source


__all__ = [
    "StructureInputError",
    "load_structure_file",
    "resolve_frozen_indices",
    "resolve_structure_path",
]
