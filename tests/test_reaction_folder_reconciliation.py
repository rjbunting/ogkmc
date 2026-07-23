"""Checkpoint-boundary tests for the per-reaction folder hierarchy."""

from __future__ import annotations

import json
from types import SimpleNamespace

from autokmc.io.config import OutputCfg, RunConfig
from autokmc.io.persistence import ReactionWriter, reconcile_event_log
from autokmc.io.reaction_index import load_reaction_index
from autokmc.workflow import runtime as runtime_module
from autokmc.workflow.models import RunIdentity


def _reaction_metadata_path(root, *, lateral_class: int):
    return (
        root
        / "reactions"
        / "adsorption"
        / "(C-)#(O+)"
        / f"iso0_lat{lateral_class}"
        / "reaction.json"
    )


def test_discovery_step_survives_batched_event_updates(tmp_path, make_reaction):
    reaction = make_reaction(lateral=2)
    writer = ReactionWriter(tmp_path, run_id="run-a")

    folder = writer.ensure_reaction(reaction, step=4)
    assert json.loads((folder / "reaction.json").read_text())["discovery_step"] == 4

    # Re-observing and then firing the reaction must not redefine discovery.
    writer.ensure_reaction(reaction, step=6)
    writer.record(
        step=8,
        time_s=1.0e-6,
        tau_s=1.0e-6,
        reaction=reaction,
    )
    writer.sync_for_checkpoint()
    writer.close()

    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["discovery_step"] == 4
    assert payload["stats"] == {"count": 1, "first_step": 8, "last_step": 8}


def test_invalid_diffusion_records_the_discovery_step(tmp_path):
    site = SimpleNamespace(iso_class=1, reactant="[O]")
    lateral = SimpleNamespace(lateral_class=2, invalid_reason="NEB failed")
    writer = ReactionWriter(tmp_path)

    folder = writer.write_invalid_diffusion(site, lateral, step=7)
    writer.close()

    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["valid"] is False
    assert payload["discovery_step"] == 7


def test_resume_recognizes_legacy_invalid_diffusion_folder(tmp_path):
    legacy = (
        tmp_path
        / "reactions"
        / "diffusion"
        / "(O)"
        / "diff_iso1_lat2"
    )
    legacy.mkdir(parents=True)
    (legacy / "reaction.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "kind": "diffusion",
                "iso_class": 1,
                "lateral_class": 2,
                "reactant_smiles": "[O]",
                "valid": False,
                "description": "legacy invalid diffusion",
                "stats": {
                    "count": 0,
                    "first_step": None,
                    "last_step": None,
                },
            }
        ),
        encoding="utf-8",
    )
    site = SimpleNamespace(iso_class=1, reactant="[O]")
    lateral = SimpleNamespace(lateral_class=2, invalid_reason="legacy failure")

    resumed = ReactionWriter(
        tmp_path,
        append=True,
        run_id="run-a",
        checkpoint_step=4,
    )

    assert resumed.n_invalid_reactions == 1
    assert resumed.write_invalid_diffusion(site, lateral, step=4) == legacy
    assert not (tmp_path / "diagnostics" / "invalid_diffusion").exists()
    resumed.close()
    [definition] = load_reaction_index(
        tmp_path / "reactions" / "index.jsonl"
    ).values()
    assert definition["valid"] is False
    assert definition["folder"] == "reactions/diffusion/(O)/diff_iso1_lat2"


def test_resume_quarantines_folders_discovered_after_checkpoint(
    tmp_path,
    make_reaction,
):
    committed = make_reaction(lateral=0)
    crash_tail = make_reaction(lateral=1)
    writer = ReactionWriter(tmp_path, run_id="run-a")
    writer.record(
        step=1,
        time_s=1.0e-6,
        tau_s=1.0e-6,
        reaction=committed,
    )
    event_commit = writer.sync_for_checkpoint()
    writer.record(
        step=2,
        time_s=2.0e-6,
        tau_s=1.0e-6,
        reaction=crash_tail,
    )
    writer.close()

    reconcile_event_log(
        tmp_path / "events.jsonl",
        checkpoint_step=1,
        committed_event_count=event_commit.count,
        committed_event_offset=event_commit.offset,
        run_id="run-a",
    )
    resumed = ReactionWriter(
        tmp_path,
        append=True,
        run_id="run-a",
        checkpoint_step=1,
    )

    committed_folder = _reaction_metadata_path(
        tmp_path,
        lateral_class=0,
    ).parent
    crash_tail_folder = _reaction_metadata_path(
        tmp_path,
        lateral_class=1,
    ).parent
    assert committed_folder.is_dir()
    assert not crash_tail_folder.exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [1]
    recovered = (
        tmp_path
        / "uncommitted_reactions"
        / "after_checkpoint_step_1"
        / "adsorption"
        / "(C-)#(O+)"
        / "iso0_lat1"
    )
    assert json.loads((recovered / "reaction.json").read_text())[
        "discovery_step"
    ] == 2
    assert resumed.n_unique_reactions == 1
    resumed.close()


def test_resume_quarantines_incomplete_reaction_leaf_without_metadata(tmp_path):
    incomplete = (
        tmp_path
        / "reactions"
        / "adsorption"
        / "(O)"
        / "iso4_lat2"
    )
    incomplete.mkdir(parents=True)
    partial_structure = incomplete / "occupied.extxyz"
    partial_structure.write_text("partial crash artifact", encoding="utf-8")

    resumed = ReactionWriter(
        tmp_path,
        append=True,
        run_id="run-a",
        checkpoint_step=3,
    )

    assert not incomplete.exists()
    recovered = (
        tmp_path
        / "uncommitted_reactions"
        / "after_checkpoint_step_3"
        / "adsorption"
        / "(O)"
        / "iso4_lat2"
    )
    assert (recovered / "occupied.extxyz").read_text(encoding="utf-8") == (
        "partial crash artifact"
    )
    assert resumed.n_unique_reactions == 0
    resumed.close()


def test_create_output_sinks_passes_resume_step_to_reaction_writer(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class Reactions:
        def __init__(self, output_dir, **kwargs):
            captured.update(kwargs)
            self.jsonl_path = output_dir / "events.jsonl"

        def close(self):
            pass

    class Trajectory:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runtime_module, "ReactionWriter", Reactions)
    monkeypatch.setattr(runtime_module, "TrajectoryWriter", Trajectory)
    cfg = RunConfig(output=OutputCfg(dir=str(tmp_path)))
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="run-a",
        resume_state=SimpleNamespace(step=9),
    )

    sinks = runtime_module.create_output_sinks(cfg, identity, {})
    sinks.reactions.close()
    sinks.trajectory.close()

    assert captured["append"] is True
    assert captured["checkpoint_step"] == 9


def test_create_output_sinks_seeds_resume_discovery_from_all_definitions(
    tmp_path,
    monkeypatch,
):
    class Reactions:
        def __init__(self, output_dir, **_kwargs):
            self.jsonl_path = output_dir / "events.jsonl"
            self.reaction_definitions = (
                {"reaction_id": "valid-unfired", "valid": True},
                {"reaction_id": "invalid-unfired", "valid": False},
            )

        def close(self):
            pass

    class Trajectory:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runtime_module, "ReactionWriter", Reactions)
    monkeypatch.setattr(runtime_module, "TrajectoryWriter", Trajectory)
    cfg = RunConfig(output=OutputCfg(dir=str(tmp_path)))
    identity = RunIdentity(
        output_dir=tmp_path,
        manifest_path=tmp_path / "run_manifest.json",
        run_id="run-a",
        resume_state=SimpleNamespace(step=0),
    )

    sinks = runtime_module.create_output_sinks(cfg, identity, {})
    totals = sinks.summary.to_dict()["totals"]
    sinks.close()

    assert totals["discovered_reactions"] == 2
    assert totals["discovered_valid_reactions"] == 1
    assert totals["discovered_invalid_reactions"] == 1
