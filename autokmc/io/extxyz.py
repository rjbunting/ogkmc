"""Complete EXTXYZ columns without changing live per-atom metadata.

ASE writes string columns verbatim but reads atom rows by whitespace splitting.
Empty strings therefore remove a column; whitespace can shift or truncate later
columns. Write safe tokens and retain the original values in the comment line
so AutoKMC's readers can restore them exactly.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib.parse import quote, quote_from_bytes

import numpy as np
from ase import Atoms
from ase.io import read as _ase_read, write as _ase_write

from autokmc.io.atoms import copy_atoms_with_results


STRING_ARRAYS_KEY = "autokmc_extxyz_string_arrays"


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported EXTXYZ string-array value: {type(value).__name__}")


def restore_string_arrays(atoms: Atoms) -> Atoms:
    """Restore original string values on a newly read or copied snapshot."""
    metadata = atoms.info.get(STRING_ARRAYS_KEY)
    if metadata is None:
        return atoms
    if not isinstance(metadata, str) or not metadata.startswith("v1:"):
        raise ValueError("unsupported AutoKMC EXTXYZ string-array metadata")
    encoded = metadata[3:]
    metadata = json.loads(base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True))
    restored = {}
    for name, entry in metadata["arrays"].items():
        dtype = np.dtype(entry["dtype"])
        shape = tuple(entry["shape"])
        if (
            name not in atoms.arrays or atoms.arrays[name].dtype.kind not in "USO"
            or dtype.kind not in "USO" or len(shape) not in (1, 2)
            or shape[0] != len(atoms) or any(size < 0 for size in shape)
            or int(np.prod(shape)) != atoms.arrays[name].size
        ):
            raise ValueError(f"invalid EXTXYZ string-array metadata for {name!r}")
        values = [
            bytes.fromhex(value["bytes_hex"]) if isinstance(value, dict) else value
            for value in entry["values"]
        ]
        restored[name] = np.asarray(values, dtype=dtype).reshape(shape)
    # Validate every entry before changing the snapshot.
    for name, restored_array in restored.items():
        atoms.arrays[name] = restored_array
    atoms.info.pop(STRING_ARRAYS_KEY)
    return atoms


def prepare_extxyz_atoms(atoms: Atoms) -> Atoms:
    """Detach results and make every string value a nonempty, safe token."""
    snapshot = restore_string_arrays(copy_atoms_with_results(atoms))
    originals = {}
    for name, array in list(snapshot.arrays.items()):
        if not name or any(
            char.isspace() or ord(char) < 32 or char in ':="\'\\{}[]' for char in name
        ):
            raise ValueError(f"invalid EXTXYZ property name: {name!r}")
        if (
            array.ndim not in (1, 2) or len(array) != len(snapshot)
            or (array.ndim == 2 and array.shape[1] == 0)
        ):
            raise ValueError(f"invalid EXTXYZ array shape for {name!r}: {array.shape}")
        if array.dtype.kind not in "USO":
            continue
        values = [_json_scalar(value) for value in array.flat]
        # This also rejects nonfinite object scalars before opening an output.
        json.dumps(values, allow_nan=False)
        tokens = []
        changed = array.dtype.kind == "S"
        for original in values:
            if isinstance(original, dict):
                token = quote_from_bytes(bytes.fromhex(original["bytes_hex"]), safe="") or "_"
            else:
                text = "" if original is None else str(original)
                unsafe = any(char.isspace() or ord(char) < 32 or char in '\\"\'' for char in text)
                token = "_" if not text else quote(text, safe="") if unsafe else text
            tokens.append(token)
            changed = changed or token != original
        if not changed:
            continue
        originals[name] = {"dtype": array.dtype.str, "shape": list(array.shape), "values": values}
        # Infer a new width; assigning into the original U1 array truncates
        # escaped strings and would reintroduce silent metadata corruption.
        snapshot.arrays[name] = np.asarray(tokens, dtype=str).reshape(array.shape)
    if originals:
        # ASE's comment parser consumes backslash escapes too. A base64 JSON
        # payload preserves newlines, quotes and literal backslashes exactly.
        payload = json.dumps({"arrays": originals}, allow_nan=False).encode("utf-8")
        # Unquoted '=' padding confuses ASE's key/value parser and can consume
        # the next field (including energy or pbc), so omit it on disk.
        snapshot.info[STRING_ARRAYS_KEY] = "v1:" + base64.b64encode(payload).decode("ascii").rstrip("=")
    return snapshot


def write_extxyz(filename, images, *, format="extxyz", append=False) -> None:
    """Serialize complete frames before touching an output file or stream."""
    if format != "extxyz":
        raise ValueError("write_extxyz only supports format='extxyz'")
    frames = [images] if isinstance(images, Atoms) else images
    prepared = [prepare_extxyz_atoms(atoms) for atoms in frames]
    buffer = io.StringIO()
    _ase_write(buffer, prepared, format="extxyz")
    content = buffer.getvalue()
    if hasattr(filename, "write"):
        filename.write(content)
    else:
        with Path(filename).open("a" if append else "w", encoding="utf-8") as destination:
            destination.write(content)


def read_atoms(*args, **kwargs):
    """Read any ASE format, restoring AutoKMC's escaped EXTXYZ string arrays."""
    images = _ase_read(*args, **kwargs)
    if isinstance(images, Atoms):
        return restore_string_arrays(images)
    return [restore_string_arrays(atoms) for atoms in images]
