"""Tests for ReactionSummary aggregation."""

from __future__ import annotations

import json
import math

import numpy as np

from ogkmc.species.smiles import smiles_to_dirname
from ogkmc.io.summary import ReactionSummary, make_run_meta
from ogkmc.io.schemas import SUMMARY_ARTIFACT_TYPE, SUMMARY_SCHEMA_VERSION


def test_summary_aggregates_per_type(make_reaction):
    s = ReactionSummary()
    s.add(make_reaction(kind="adsorption", iso=0, lateral=0,
                        delta_e=-0.2, barrier=0.1, rate=1e9), step=1)
    s.add(make_reaction(kind="adsorption", iso=0, lateral=0,
                        delta_e=-0.3, barrier=0.1, rate=2e9), step=2)
    s.add(make_reaction(kind="desorption", iso=0, lateral=0,
                        delta_e=+0.4, barrier=0.5, rate=3e7), step=3)

    out = s.to_dict(final_occupancy={0: 1})
    assert out["totals"]["reactions"] == 3
    assert out["totals"]["by_kind"] == {"adsorption": 2, "desorption": 1}
    assert len(out["by_reaction_type"]) == 2
    assert "production_summary" not in out

    ads = next(b for b in out["by_reaction_type"] if b["kind"] == "adsorption")
    assert ads["count"] == 2
    assert ads["reaction_dir"] == f"reactions/adsorption/{smiles_to_dirname('[C-]#[O+]')}/iso0_lat0"
    assert ads["direction"] == "adsorption"
    assert ads["rate_energy_basis"] == "unknown"
    assert ads["first_step"] == 1
    assert ads["last_step"]  == 2
    assert math.isclose(ads["rate_delta_ev"]["mean"], -0.25, rel_tol=1e-9)
    assert math.isclose(ads["rate_hz"]["mean"], 1.5e9, rel_tol=1e-9)
    assert math.isclose(ads["rate_hz"]["std"],
                        float(np.std([1e9, 2e9], ddof=0)), rel_tol=1e-9)
    assert out["final_occupancy"] == {"0": 1}


def test_summary_write_json(tmp_path, make_reaction):
    s = ReactionSummary()
    s.add(make_reaction(kind="adsorption"), step=1)
    p = s.write(tmp_path / "summary.json",
                run_meta=make_run_meta(temperature_k=500.0, n_steps_requested=10),
                final_occupancy={0: 1})
    payload = json.loads(p.read_text())
    assert payload["artifact_type"] == SUMMARY_ARTIFACT_TYPE
    assert payload["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert payload["run"]["temperature_k"] == 500.0
    assert payload["totals"]["reactions"] == 1


def test_summary_empty():
    s = ReactionSummary()
    out = s.to_dict()
    assert out["totals"]["reactions"] == 0
    assert out["by_reaction_type"] == []


def test_summary_uses_constant_space_online_accumulators(make_reaction):
    summary = ReactionSummary()
    values = []
    for step in range(1, 1001):
        rate = float(step * 3)
        values.append(rate)
        summary.add(
            make_reaction(
                kind="adsorption",
                delta_e=-0.2,
                barrier=0.1,
                rate=rate,
            ),
            step=step,
        )

    bucket = next(iter(summary._buckets.values()))
    assert bucket["rate"].count == 1000
    assert not hasattr(bucket["rate"], "append")
    output = summary.to_dict()["by_reaction_type"][0]["rate_hz"]
    assert math.isclose(output["mean"], float(np.mean(values)), rel_tol=1e-15)
    assert math.isclose(
        output["std"],
        float(np.std(values, ddof=0)),
        rel_tol=1e-15,
    )


def _event(
    *,
    step,
    direction,
    basis,
    rate_delta,
    rate_barrier,
    delta_e,
    barrier_e,
    delta_g,
    barrier_g,
):
    return {
        "run_id": "run-summary",
        "reaction_id": "reaction-shared",
        "step": step,
        "kind": "diffusion",
        "direction": direction,
        "reactant_smiles": "[O]",
        "iso_class": 3,
        "lateral_class": 4,
        "rate_energy_basis": basis,
        "rate_hz": float(step),
        "rate_delta_ev": rate_delta,
        "rate_barrier_ev": rate_barrier,
        "delta_e_ev": delta_e,
        "barrier_ev": barrier_e,
        "delta_g_ev": delta_g,
        "barrier_g_ev": barrier_g,
    }


def test_summary_separates_direction_basis_and_energy_statistics():
    summary = ReactionSummary(run_id="run-summary")
    summary.note_discovered("reaction-invalid", valid=False)
    summary.note_discovered("reaction-unfired", valid=True)
    summary.add_event(
        _event(
            step=1,
            direction="a_to_b",
            basis="electronic",
            rate_delta=0.2,
            rate_barrier=0.5,
            delta_e=0.2,
            barrier_e=0.5,
            delta_g=0.1,
            barrier_g=0.4,
        )
    )
    summary.add_event(
        _event(
            step=2,
            direction="b_to_a",
            basis="free_energy",
            rate_delta=-0.1,
            rate_barrier=0.3,
            delta_e=-0.2,
            barrier_e=0.3,
            delta_g=-0.1,
            barrier_g=0.3,
        )
    )

    output = summary.to_dict()
    assert output["run_id"] == "run-summary"
    assert output["totals"] == {
        "reactions": 2,
        "fired_events": 2,
        "by_kind": {"diffusion": 2},
        "by_direction": {"a_to_b": 1, "b_to_a": 1},
        "unique_reaction_types": 2,
        "fired_reaction_classes": 1,
        "discovered_reactions": 3,
        "discovered_valid_reactions": 2,
        "discovered_invalid_reactions": 1,
    }
    assert {
        (row["direction"], row["rate_energy_basis"])
        for row in output["by_reaction_type"]
    } == {("a_to_b", "electronic"), ("b_to_a", "free_energy")}
    forward = next(
        row
        for row in output["by_reaction_type"]
        if row["direction"] == "a_to_b"
    )
    assert forward["rate_delta_ev"]["mean"] == 0.2
    assert forward["electronic_energy"]["delta_e_ev"]["mean"] == 0.2
    assert forward["free_energy"]["delta_g_ev"]["mean"] == 0.1
