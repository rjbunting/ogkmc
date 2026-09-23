"""Small, shared primitives for atomic on-disk writes."""

from __future__ import annotations

import errno
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


_UNSUPPORTED_DIRECTORY_SYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def fsync_directory(path: str | Path) -> None:
    """Persist directory-entry changes when the platform supports it.

    Unexpected I/O failures are propagated so callers cannot publish a
    checkpoint after an artifact failed to become durable.  Some platforms
    and filesystems do not implement directory ``fsync``; only those explicit
    "unsupported" errors are ignored.
    """
    if os.name == "nt":  # pragma: no cover - Windows cannot open directories.
        return

    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def ensure_directory(path: str | Path) -> Path:
    """Create *path* and durably publish every newly-created path component."""
    destination = Path(path)
    missing: list[Path] = []
    current = destination
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if current.exists() and not current.is_dir():
        raise NotADirectoryError(current)

    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        fsync_directory(directory)
        fsync_directory(directory.parent)
    return destination


def _fsync_file(path: Path) -> None:
    """Sync a closed regular file by path."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_path_atomic(source: str | Path, destination: str | Path) -> Path:
    """Atomically replace *destination* and sync both affected directories."""
    source_path = Path(source)
    destination_path = Path(destination)
    os.replace(source_path, destination_path)
    fsync_directory(destination_path.parent)
    if source_path.parent != destination_path.parent:
        fsync_directory(source_path.parent)
    return destination_path


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path, then durably publish it on success."""
    destination = Path(path)
    ensure_directory(destination.parent)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield temporary
        _fsync_file(temporary)
        replace_path_atomic(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    transform: Callable[[Any], Any] | None = None,
) -> Path:
    """Write an indented, finite JSON document without exposing partial data."""
    destination = Path(path)
    value = transform(payload) if transform is not None else payload
    with atomic_output_path(destination) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=sort_keys,
                allow_nan=False,
            )
            handle.write("\n")
    return destination


__all__ = [
    "atomic_output_path",
    "ensure_directory",
    "fsync_directory",
    "replace_path_atomic",
    "write_json_atomic",
]
