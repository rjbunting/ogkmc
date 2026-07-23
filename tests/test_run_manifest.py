"""Run-manifest provenance and restart behavior."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import networkx as nx

from autokmc import __version__
from autokmc.io.run_manifest import (
    begin_run_manifest,
    build_artifact_inventory,
    discover_quarantine_locations,
    fail_run_manifest,
    finish_run_manifest,
    start_run_manifest,
    update_run_manifest,
)
from autokmc.io.schemas import EVENT_SCHEMA_VERSION


def _graph_and_site():
    graph = nx.Graph()
    graph.add_node(0, type="bulk", element="Pt")
    graph.add_node(1, type="surface", element="Pt")
    graph.add_node(10, type="adsorbate", element="C", occupied=True)
    site = SimpleNamespace(
        reactant="[C]=O",
        iso_class=2,
        member_node_ids=[[10]],
        members=[[frozenset({1})]],
    )
    return graph, site


def test_manifest_records_feed_initial_state_and_normalization(tmp_path):
    graph, site = _graph_and_site()
    config = tmp_path / "run.yaml"
    config.write_text("schema_version: '1'\n")
    path = tmp_path / "run_manifest.json"

    start_run_manifest(
        path,
        graph=graph,
        adsorbate_sites=[site],
        feed_reactants=[{
            "smiles": "[C]=O",
            "partial_pressure_bar": 1.0,
            "thermochemistry": {
                "symmetry_number": 1,
                "symmetry_number_source": "inferred",
                "point_group": "C*v",
                "symmetry_tolerance": 0.3,
            },
        }],
        temperature_k=500.0,
        random_seed=69,
        structure_kind="surface",
        composition="Pt",
        structure_source={
            "path": "/inputs/catalyst.extxyz",
            "format": "extxyz",
            "index": -1,
            "sha256": "abc123",
            "size_bytes": 42,
            "chemical_formula": "Pt2",
        },
        config_path=str(config),
        events_filename="events.jsonl",
        run_id="fd4912ce-ffb3-47d5-86b0-4ce5541c04a8",
        resolved_config={"kmc": {"temperature_k": 500.0}},
    )
    finish_run_manifest(path, final_step=100, final_time_s=2.5, steps_executed=100)

    payload = json.loads(path.read_text())
    assert payload["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert payload["schema_version"] == "3"
    assert payload["run_id"] == "fd4912ce-ffb3-47d5-86b0-4ce5541c04a8"
    assert payload["package_version"] == __version__
    assert payload["segments"][0]["package_version"] == __version__
    assert payload["config"]["content"] == "schema_version: '1'\n"
    assert payload["config"]["resolved"]["kmc"]["temperature_k"] == 500.0
    assert payload["feed_reactants"][0]["species"] == "[C]=O"
    assert payload["feed_reactants"][0]["thermochemistry"] == {
        "symmetry_number": 1,
        "symmetry_number_source": "inferred",
        "point_group": "C*v",
        "symmetry_tolerance": 0.3,
    }
    assert payload["catalyst"]["n_catalyst_atoms"] == 2
    assert payload["catalyst"]["n_surface_atoms"] == 1
    assert payload["catalyst"]["source"]["path"] == "/inputs/catalyst.extxyz"
    assert payload["catalyst"]["source"]["chemical_formula"] == "Pt2"
    initial = payload["initial_state"]["occupied_surface_states"]
    assert initial[0]["species"] == "[C]=O"
    assert initial[0]["surface_cliques"] == [[1]]
    assert payload["result"]["final_time_s"] == 2.5
    assert payload["result"]["simulated_time_s"] == 2.5
    assert payload["result"]["status"] == "complete"
    assert payload["lifecycle"]["status"] == "complete"
    assert payload["lifecycle"]["termination_reason"] == (
        "requested_steps_completed"
    )
    assert payload["segments"][0]["ended_utc"]


def test_resume_appends_segment_without_replacing_original_initial_state(tmp_path):
    graph, site = _graph_and_site()
    path = tmp_path / "run_manifest.json"
    kwargs = dict(
        path=path,
        graph=graph,
        adsorbate_sites=[site],
        feed_reactants=[{"smiles": "[C]=O", "partial_pressure_bar": 1.0}],
        temperature_k=500.0,
        random_seed=69,
        structure_kind="surface",
        composition="Pt",
        config_path=None,
    )
    start_run_manifest(**kwargs)
    finish_run_manifest(
        path,
        final_step=50,
        final_time_s=1.25,
        steps_executed=50,
    )
    original = json.loads(path.read_text())
    original["package_version"] = "0.1.0"
    original["segments"][0]["package_version"] = "0.1.0"
    path.write_text(json.dumps(original), encoding="utf-8")
    start_run_manifest(**kwargs, initial_step=50, initial_time_s=1.25)
    finish_run_manifest(
        path,
        final_step=75,
        final_time_s=1.75,
        steps_executed=25,
        status="stopped",
        termination_reason="zero_total_rate",
    )

    payload = json.loads(path.read_text())
    assert payload["initial_state"]["step"] == 0
    assert len(payload["segments"]) == 2
    assert payload["segments"][1]["initial_step"] == 50
    assert payload["segments"][0]["status"] == "complete"
    assert payload["segments"][1]["status"] == "stopped"
    assert payload["package_version"] == "0.1.0"
    assert payload["segments"][0]["package_version"] == "0.1.0"
    assert payload["segments"][1]["package_version"] == __version__
    assert all(segment["ended_utc"] for segment in payload["segments"])
    assert payload["run_id"]


def test_step_zero_resume_appends_manifest_segment(tmp_path):
    graph, site = _graph_and_site()
    path = tmp_path / "run_manifest.json"
    kwargs = dict(
        path=path,
        graph=graph,
        adsorbate_sites=[site],
        feed_reactants=[{"smiles": "[C]=O", "partial_pressure_bar": 1.0}],
        temperature_k=500.0,
        random_seed=69,
        structure_kind="surface",
        composition="Pt",
        config_path=None,
        run_id="step-zero-run",
    )
    start_run_manifest(**kwargs)
    finish_run_manifest(
        path,
        final_step=0,
        final_time_s=0.0,
        steps_executed=0,
        termination_reason="no_steps_requested",
    )
    start_run_manifest(**kwargs, initial_step=0, is_resume=True)

    payload = json.loads(path.read_text())
    assert payload["run_id"] == "step-zero-run"
    assert len(payload["segments"]) == 2
    assert payload["segments"][1]["initial_step"] == 0


def test_manifest_exists_while_preparing_and_tracks_failure(tmp_path):
    path = tmp_path / "run_manifest.json"

    begin_run_manifest(
        path,
        run_id="early-manifest",
        config_path=None,
        resolved_config={"kmc": {"n_steps": 10}},
        initial_step=4,
        initial_time_s=0.25,
    )
    update_run_manifest(
        path,
        current_stage="stage_2_structure",
        last_durable_step=5,
        last_durable_time_s=0.5,
        warning="test warning",
    )
    fail_run_manifest(path, RuntimeError("structure failed"))
    # The Stage-7 helper and outer configured pipeline may both report the
    # same exception; terminal rewrites must retain the original stage.
    fail_run_manifest(path, RuntimeError("structure failed"))

    payload = json.loads(path.read_text())
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["lifecycle"]["current_stage"] == "finished"
    assert payload["lifecycle"]["termination_stage"] == "stage_2_structure"
    assert payload["lifecycle"]["last_durable_step"] == 5
    assert payload["result"]["termination_reason"] == (
        "RuntimeError: structure failed"
    )
    assert payload["segments"][0]["status"] == "failed"
    assert payload["segments"][0]["termination_stage"] == "stage_2_structure"
    assert payload["segments"][0]["ended_utc"]
    assert payload["warnings"] == ["test warning"]
    assert payload["artifacts"]["run_manifest"]["status"] == "partial"
    assert payload["artifacts"]["events"]["status"] == "missing"
    assert payload["artifacts"]["isaac_records"]["status"] == "disabled"
    assert payload["artifacts"]["reaction_index"]["status"] == "missing"
    assert payload["artifacts"]["invalid_diffusion"]["status"] == "missing"


def test_artifact_inventory_hashes_files_and_summarizes_directories(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    events = output / "events.jsonl"
    events.write_bytes(b'{"step":1}\n')
    reactions = output / "reactions"
    reactions.mkdir()
    (reactions / "reaction.json").write_text("{}\n", encoding="utf-8")

    inventory = build_artifact_inventory(
        output,
        {
            "events": {
                "path": events,
                "type": "event-log-jsonl",
                "schema_version": "3",
            },
            "reactions": {
                "path": reactions,
                "type": "reaction-directory",
                "schema_version": "3",
            },
            "missing": {
                "path": output / "missing.json",
                "type": "json",
                "schema_version": "1",
            },
        },
    )

    assert inventory["events"] == {
        "path": "events.jsonl",
        "type": "event-log-jsonl",
        "schema_version": "3",
        "present": True,
        "presence": True,
        "size_bytes": len(b'{"step":1}\n'),
        "sha256": hashlib.sha256(b'{"step":1}\n').hexdigest(),
        "status": "complete",
        "kind": "file",
    }
    assert inventory["reactions"]["status"] == "complete"
    assert inventory["reactions"]["file_count"] == 1
    assert inventory["reactions"]["size_bytes"] == 3
    assert inventory["missing"]["status"] == "missing"


def test_quarantine_discovery_reports_invalid_diffusion_leaves(tmp_path):
    quarantine = tmp_path / "uncommitted_reactions" / "after_checkpoint_step_4"
    regular = quarantine / "adsorption" / "(O)" / "iso0_lat1"
    invalid_a = (
        quarantine
        / "diagnostics"
        / "invalid_diffusion"
        / "(O)"
        / "diff_iso0_lat2"
    )
    invalid_b = invalid_a.with_name("diff_iso0_lat3")
    for leaf in (regular, invalid_a, invalid_b):
        leaf.mkdir(parents=True)

    assert discover_quarantine_locations(tmp_path) == [
        "uncommitted_reactions/after_checkpoint_step_4/adsorption/(O)/iso0_lat1",
        (
            "uncommitted_reactions/after_checkpoint_step_4/diagnostics/"
            "invalid_diffusion/(O)/diff_iso0_lat2"
        ),
        (
            "uncommitted_reactions/after_checkpoint_step_4/diagnostics/"
            "invalid_diffusion/(O)/diff_iso0_lat3"
        ),
    ]
