"""Product-rate and backward-mechanism analysis from event schema v2."""

from __future__ import annotations

import csv
import json

import pytest

from autokmc.analysis.products import AnalysisError, analyze_run
from autokmc.cli.main import main as cli_main
from autokmc.core.constants import PERSISTENCE_SCHEMA_VERSION


def _surface(species: str, placement: str) -> dict:
    return {
        "phase": "surface",
        "species": species,
        "placement_id": placement,
        "site_iso_class": 0,
        "member_index": 0,
        "node_ids": [],
        "surface_cliques": [],
    }


def _gas(species: str) -> dict:
    return {"phase": "gas", "species": species}


def _event(step, time_s, kind, inputs, outputs, *, direction=None, template=None):
    return {
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "step": step,
        "time_s": time_s,
        "tau_s": 1.0,
        "kind": kind,
        "direction": direction,
        "inputs": inputs,
        "outputs": outputs,
        "template": template,
        "gas_product": False,
        "rate_hz": 123.0,
    }


def _write_run(tmp_path, events, *, final_time=10.0):
    manifest = {
        "schema_version": "1",
        "event_schema_version": PERSISTENCE_SCHEMA_VERSION,
        "files": {"events": "events.jsonl"},
        "feed_reactants": [
            {"species": "[C]=O", "partial_pressure_bar": 1.0},
            {"species": "O=O", "partial_pressure_bar": 0.2},
        ],
        "catalyst": {"n_surface_atoms": 5},
        "initial_state": {
            "step": 0,
            "time_s": 0.0,
            "occupied_surface_states": [],
        },
        "result": {"final_step": len(events), "final_time_s": final_time},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )


def _co2_mechanism_events(*, with_recross=False):
    co = _surface("[C]=O", "co")
    o2 = _surface("O=O", "o2")
    o1 = _surface("[O]", "o1")
    o2_atom = _surface("[O]", "o2-atom")
    o3 = _surface("[O]", "o3")
    co2 = _surface("O=C=O", "co2")
    dissociation = {"smiles_a": "[O]", "smiles_b": "[O]", "smiles_c": "O=O"}
    formation = {"smiles_a": "[C]=O", "smiles_b": "[O]", "smiles_c": "O=C=O"}
    events = [
        _event(1, 1.0, "adsorption", [_gas("[C]=O")], [co]),
        _event(2, 2.0, "adsorption", [_gas("O=O")], [o2]),
        _event(3, 3.0, "bond", [o2], [o1, o2_atom], direction="dissoc", template=dissociation),
        _event(4, 4.0, "diffusion", [o1], [o3], direction="a_to_b"),
        _event(5, 5.0, "bond", [co, o3], [co2], direction="couple", template=formation),
    ]
    if with_recross:
        events.extend(
            [
                _event(6, 5.5, "bond", [co2], [co, o3], direction="dissoc", template=formation),
                _event(7, 5.8, "bond", [co, o3], [co2], direction="couple", template=formation),
                _event(8, 6.0, "desorption", [co2], [_gas("O=C=O")]),
            ]
        )
    else:
        events.append(_event(6, 6.0, "desorption", [co2], [_gas("O=C=O")]))
    return events


def test_analysis_finds_product_rate_and_back_propagated_mechanism(tmp_path):
    _write_run(tmp_path, _co2_mechanism_events())

    result = analyze_run(tmp_path, n_blocks=2)

    assert result["products"] == [
        pytest.approx(
            {
                "product": "O=C=O",
                "count": 1,
                "start_time_s": 0.0,
                "end_time_s": 10.0,
                "duration_s": 10.0,
                "rate_hz": 0.1,
                "rate_ci95_low_hz": 0.002531780798428988,
                "rate_ci95_high_hz": 0.5571643390938898,
                "tof_per_surface_atom_s-1": 0.02,
            }
        )
    ]
    product_event = json.loads((tmp_path / "analysis" / "product_events.jsonl").read_text())
    mechanism = product_event["mechanism"]
    assert any("O=O* → [O]* + [O]*" in step for step in mechanism)
    assert any("[C]=O* + [O]* → O=C=O*" in step for step in mechanism)
    assert mechanism[-1] == "O=C=O* → O=C=O(g)"
    assert not any("diffusion" in step for step in mechanism)

    with (tmp_path / "analysis" / "mechanisms.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["count"] == "1"
    assert float(rows[0]["fraction"]) == pytest.approx(1.0)


def test_immediate_bond_recrossing_is_removed_from_mechanism(tmp_path):
    _write_run(tmp_path, _co2_mechanism_events(with_recross=True))
    analyze_run(tmp_path)
    product_event = json.loads((tmp_path / "analysis" / "product_events.jsonl").read_text())
    formation_steps = [
        step for step in product_event["mechanism"]
        if "[C]=O* + [O]* → O=C=O*" in step
    ]
    assert len(formation_steps) == 1


def test_feed_desorption_is_not_a_product(tmp_path):
    co = _surface("[C]=O", "co")
    events = [
        _event(1, 1.0, "adsorption", [_gas("[C]=O")], [co]),
        _event(2, 2.0, "desorption", [co], [_gas("[C]=O")]),
    ]
    _write_run(tmp_path, events)
    result = analyze_run(tmp_path)
    assert result["products"] == []
    assert (tmp_path / "analysis" / "product_events.jsonl").read_text() == ""


def test_inconsistent_surface_history_is_rejected(tmp_path):
    co2 = _surface("O=C=O", "co2")
    _write_run(
        tmp_path,
        [_event(1, 1.0, "desorption", [co2], [_gas("O=C=O")])],
    )
    with pytest.raises(AnalysisError, match="untracked surface state"):
        analyze_run(tmp_path)


def test_analysis_window_controls_observed_rate(tmp_path):
    events = _co2_mechanism_events()
    _write_run(tmp_path, events, final_time=10.0)
    result = analyze_run(tmp_path, start_time_s=5.0, end_time_s=7.0)
    assert result["products"][0]["count"] == 1
    assert result["products"][0]["rate_hz"] == pytest.approx(0.5)


@pytest.mark.parametrize("bounds", [
    {"end_time_s": 100.0}, {"start_time_s": -1.0},
    {"start_time_s": float("nan")}, {"end_time_s": float("inf")},
    {"start_time_s": 8.0, "end_time_s": 7.0},
])
def test_analysis_rejects_unobserved_or_invalid_exposure_without_replacing_results(tmp_path, bounds):
    _write_run(tmp_path, _co2_mechanism_events())
    analyze_run(tmp_path)
    rates_path = tmp_path / "analysis" / "product_rates.csv"
    previous = rates_path.read_bytes()
    with pytest.raises(AnalysisError):
        analyze_run(tmp_path, **bounds)
    assert rates_path.read_bytes() == previous


def test_analysis_window_cannot_precede_restarted_run_initial_time(tmp_path):
    _write_run(tmp_path, [])
    path = tmp_path / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["initial_state"]["time_s"] = 5.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(AnalysisError, match="recorded simulation interval"):
        analyze_run(tmp_path, start_time_s=0.0)
    assert analyze_run(tmp_path)["duration_s"] == 5.0


def test_unfinalized_run_bounds_and_rate_blocks_use_last_recorded_event(tmp_path):
    _write_run(tmp_path, _co2_mechanism_events(), final_time=None)
    with pytest.raises(AnalysisError, match="recorded simulation interval"):
        analyze_run(tmp_path, end_time_s=100.0)
    result = analyze_run(tmp_path, n_blocks=2)
    assert result["products"][0]["rate_hz"] == pytest.approx(1 / 6)
    with (tmp_path / "analysis" / "rate_blocks.csv").open() as handle:
        blocks = list(csv.DictReader(handle))
    assert sum(int(row["count"]) for row in blocks) == 1


def test_analyze_cli_rejects_extrapolated_end(tmp_path, capsys):
    _write_run(tmp_path, _co2_mechanism_events())
    assert cli_main(["analyze", str(tmp_path), "--end-time", "100"]) != 0
    assert "recorded simulation interval" in capsys.readouterr().err


def test_analyze_cli_writes_tables(tmp_path, capsys):
    _write_run(tmp_path, _co2_mechanism_events())
    assert cli_main(["analyze", str(tmp_path), "--blocks", "2"]) == 0
    assert "Analysis complete" in capsys.readouterr().out
    assert (tmp_path / "analysis" / "analysis_summary.json").is_file()


def test_analyze_accepts_configured_manifest_filename(tmp_path):
    _write_run(tmp_path, _co2_mechanism_events())
    (tmp_path / "run_manifest.json").rename(tmp_path / "custom_manifest.json")

    result = analyze_run(tmp_path, manifest_filename="custom_manifest.json")

    assert result["products"][0]["product"] == "O=C=O"


def test_failed_analysis_leaves_previous_outputs_unchanged(tmp_path):
    co2 = _surface("O=C=O", "missing")
    _write_run(
        tmp_path,
        [_event(1, 1.0, "desorption", [co2], [_gas("O=C=O")])],
    )
    destination = tmp_path / "analysis"
    destination.mkdir()
    previous = {}
    for filename in (
        "product_rates.csv", "product_events.jsonl", "mechanisms.csv",
        "rate_blocks.csv", "analysis_summary.json",
    ):
        content = f"previous-{filename}\n"
        (destination / filename).write_text(content)
        previous[filename] = content

    with pytest.raises(AnalysisError, match="untracked surface state"):
        analyze_run(tmp_path)

    assert {
        path.name: path.read_text() for path in destination.iterdir()
    } == previous
