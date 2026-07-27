"""Offline human-readable run reporting."""

from __future__ import annotations

import json

import pytest

from autokmc.analysis.run_report import generate_run_report
from autokmc.io.schemas import (
    EVENT_ARTIFACT_TYPE,
    EVENT_SCHEMA_VERSION,
    REACTION_INDEX_ARTIFACT_TYPE,
    REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
    REACTION_INDEX_SCHEMA_VERSION,
)


def test_generate_run_report_combines_coverage_flux_products_and_performance(
    tmp_path,
    monkeypatch,
):
    manifest = {
        "run_id": "report-run",
        "status": "completed",
        "termination": {"reason": "configured_steps_reached"},
        "files": {"events": "events.jsonl"},
        "catalyst": {"n_surface_atoms": 10},
        "initial_state": {
            "time_s": 0.0,
            "occupied_surface_states": [],
        },
        "result": {"final_time_s": 2.0},
        "artifacts": [
            {
                "path": "events.jsonl",
                "type": "event_log",
                "status": "complete",
                "size_bytes": 123,
                "sha256": "abcdef0123456789",
            }
        ],
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    surface = {
        "phase": "surface",
        "species": "[OH]",
        "placement_id": "placement-1",
    }
    gas = {"phase": "gas", "species": "[OH]"}
    events = [
        {
            "step": 1,
            "time_s": 0.5,
            "kind": "adsorption",
            "inputs": [gas],
            "outputs": [surface],
        },
        {
            "step": 2,
            "time_s": 1.5,
            "kind": "desorption",
            "inputs": [surface],
            "outputs": [gas],
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "performance.json").write_text(
        json.dumps(
            {
                "summary": {
                    "wall_time_s": 4.0,
                    "events_per_wall_second": 0.5,
                    "top_bottlenecks": [
                        {
                            "name": "neb",
                            "wall_time_s": 3.0,
                            "fraction_of_wall_time": 0.75,
                        }
                    ],
                }
            }
        )
    )

    def fake_analyze(run_dir, **kwargs):
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        rate_blocks = analysis_dir / "rate_blocks.csv"
        rate_blocks.write_text(
            "product,block,start_time_s,end_time_s,count,rate_hz\n"
            "[OH],0,0,1,0,0\n"
            "[OH],1,1,2,1,1\n"
        )
        return {
            "products": [
                {
                    "product": "[OH]",
                    "count": 1,
                    "rate_hz": 0.5,
                    "rate_ci95_low_hz": 0.01,
                    "rate_ci95_high_hz": 2.0,
                }
            ],
            "outputs": {"rate_blocks": str(rate_blocks)},
        }

    monkeypatch.setattr(
        "autokmc.analysis.run_report.analyze_run",
        fake_analyze,
    )

    report = generate_run_report(tmp_path, n_blocks=2)

    assert report["coverage"]["final_occupied_placements"] == 0
    assert report["coverage"]["time_average_occupied_placements"] == pytest.approx(
        0.5
    )
    assert report["directional_fluxes"]["net_rows"][0]["net_count"] == 0
    assert report["products"][0]["product"] == "[OH]"
    assert report["convergence"][0]["late_to_early_rate_ratio"] is None
    assert report["performance"]["top_bottlenecks"][0]["name"] == "neb"

    markdown = (tmp_path / "analysis" / "report.md").read_text()
    html = (tmp_path / "analysis" / "report.html").read_text()
    assert "Directional event fluxes" in markdown
    assert "Cumulative products" in markdown
    assert "Performance bottlenecks" in markdown
    assert "<title>AutoKMC run report</title>" in html
    assert "[OH]" in html


def test_generate_run_report_analyzes_compact_v3_events(tmp_path):
    run_id = "compact-report-run"
    reaction_id = "reaction-compact"
    surface = {
        "phase": "surface",
        "species": "[OH]",
        "placement_id": "placement-1",
    }
    gas = {"phase": "gas", "species": "[OH]"}
    manifest = {
        "schema_version": "3",
        "run_id": run_id,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "files": {"events": "events.jsonl"},
        "feed_reactants": [{"species": "[O]"}],
        "catalyst": {"n_surface_atoms": 4},
        "initial_state": {
            "step": 0,
            "time_s": 0.0,
            "occupied_surface_states": [surface],
        },
        "lifecycle": {
            "status": "complete",
            "termination_reason": "requested_steps_completed",
        },
        "result": {
            "status": "complete",
            "termination_reason": "requested_steps_completed",
            "final_step": 1,
            "final_time_s": 2.0,
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    event = {
        "artifact_type": EVENT_ARTIFACT_TYPE,
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": "event-compact",
        "reaction_id": reaction_id,
        "run_id": run_id,
        "step": 1,
        "time_s": 1.0,
        "tau_s": 1.0,
        "kind": "desorption",
        "reactant_smiles": "[OH]",
        "iso_class": 0,
        "member_index": 0,
        "lateral_class": 0,
        "rate_hz": 2.0,
        "delta_e_ev": 0.5,
        "barrier_ev": 0.5,
        "delta_g_ev": None,
        "barrier_g_ev": None,
        "rate_energy_basis": "electronic",
        "rate_delta_ev": 0.5,
        "rate_barrier_ev": 0.5,
        "inputs": [surface],
        "outputs": [gas],
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")
    reactions = tmp_path / "reactions"
    reactions.mkdir()
    index_rows = [
        {
            "artifact_type": REACTION_INDEX_ARTIFACT_TYPE,
            "schema_version": REACTION_INDEX_SCHEMA_VERSION,
            "record_type": "header",
            "run_id": run_id,
        },
        {
            "artifact_type": REACTION_INDEX_ENTRY_ARTIFACT_TYPE,
            "schema_version": REACTION_INDEX_SCHEMA_VERSION,
            "record_type": "reaction",
            "reaction_id": reaction_id,
            "run_id": run_id,
            "valid": True,
            "kind": "adsorption",
            "reactant_smiles": "[OH]",
            "iso_class": 0,
            "lateral_class": 0,
            "directions": ["adsorption", "desorption"],
            "discovery_step": 0,
            "firing_count": 1,
            "first_step": 1,
            "last_step": 1,
            "rate_energy_bases": ["electronic"],
            "folder": "reactions/adsorption/(OH)/iso0_lat0",
            "definition": {
                "description": "desorption of [OH]",
                "template": None,
                "gas_product": False,
            },
        },
    ]
    (reactions / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )

    report = generate_run_report(tmp_path, n_blocks=2)

    assert report["status"] == "complete"
    assert report["products"][0]["product"] == "[OH]"
    assert report["products"][0]["count"] == 1
    assert report["coverage"]["time_average_occupied_placements"] == pytest.approx(
        0.5
    )
    assert (tmp_path / "analysis" / "mechanisms.csv").is_file()
