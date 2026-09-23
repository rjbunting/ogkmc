"""Focused tests for durable atomic filesystem helpers."""

from __future__ import annotations

from pathlib import Path

from ogkmc.io import _files


def test_atomic_output_syncs_file_before_replace_and_directory_after(
    tmp_path,
    monkeypatch,
):
    events: list[tuple[str, Path]] = []
    real_replace = _files.os.replace

    def sync_file(path: Path) -> None:
        assert path.is_file()
        events.append(("file_sync", path))

    def replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    def sync_directory(path: Path) -> None:
        events.append(("directory_sync", Path(path)))

    monkeypatch.setattr(_files, "_fsync_file", sync_file)
    monkeypatch.setattr(_files.os, "replace", replace)
    monkeypatch.setattr(_files, "fsync_directory", sync_directory)

    destination = tmp_path / "artifact.json"
    with _files.atomic_output_path(destination) as temporary:
        temporary.write_text('{"complete": true}\n', encoding="utf-8")

    assert destination.read_text(encoding="utf-8") == '{"complete": true}\n'
    assert [event for event, _path in events] == [
        "file_sync",
        "replace",
        "directory_sync",
    ]
    assert events[1][1] == destination
    assert events[2][1] == tmp_path


def test_ensure_directory_syncs_each_new_directory_entry(
    tmp_path,
    monkeypatch,
):
    synced: list[Path] = []
    monkeypatch.setattr(
        _files,
        "fsync_directory",
        lambda path: synced.append(Path(path)),
    )

    first = tmp_path / "first"
    second = first / "second"
    destination = second / "third"
    assert _files.ensure_directory(destination) == destination

    assert destination.is_dir()
    assert synced == [
        first,
        tmp_path,
        second,
        first,
        destination,
        second,
    ]


def test_replace_path_atomic_syncs_both_parent_directories(
    tmp_path,
    monkeypatch,
):
    source_parent = tmp_path / "source"
    destination_parent = tmp_path / "destination"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "reaction"
    source.mkdir()
    (source / "reaction.json").write_text("{}\n", encoding="utf-8")
    destination = destination_parent / "reaction"
    synced: list[Path] = []
    monkeypatch.setattr(
        _files,
        "fsync_directory",
        lambda path: synced.append(Path(path)),
    )

    assert _files.replace_path_atomic(source, destination) == destination

    assert not source.exists()
    assert (destination / "reaction.json").is_file()
    assert synced == [destination_parent, source_parent]
