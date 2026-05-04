"""Tests for ReactionSummary aggregation."""

from __future__ import annotations

import json
import math

import numpy as np

from autokmc2.io.summary import ReactionSummary, make_run_meta


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

    ads = next(b for b in out["by_reaction_type"] if b["kind"] == "adsorption")
    assert ads["count"] == 2
    assert ads["reaction_dir"] == "reactions/adsorption/(C-)#(O+)/iso0_lat0"
    assert ads["first_step"] == 1
    assert ads["last_step"]  == 2
    assert math.isclose(ads["delta_e_ev"]["mean"], -0.25, rel_tol=1e-9)
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
    assert payload["run"]["temperature_k"] == 500.0
    assert payload["totals"]["reactions"] == 1


def test_summary_empty():
    s = ReactionSummary()
    out = s.to_dict()
    assert out["totals"]["reactions"] == 0
    assert out["by_reaction_type"] == []
