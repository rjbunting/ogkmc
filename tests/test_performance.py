"""Performance summary and raw-diagnostics persistence tests."""

from __future__ import annotations

import json

import pytest

from autokmc.io.performance import (
    PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION,
    build_performance_summary,
    write_performance_diagnostics,
)


def _snapshot():
    return {
        "counters": {
            "calculation_cache.hits": 8,
            "calculation_cache.misses": 2,
            "calculation_cache.write.calls": 3,
            "optimization.bulk.calls": 2,
            "optimization.structure.calls": 5,
            "neb.calls": 4,
        },
        "timings_s": {
            "kmc.session.seconds": 8.0,
            "neb.seconds": 3.0,
            "persistence.event_append.seconds": 0.4,
            "persistence.event_sync.seconds": 0.3,
            "persistence.metadata_flush.seconds": 0.2,
            "output.isaac_export.seconds": 0.1,
            "checkpoint.save.seconds": 0.5,
        },
        "gauges": {"kmc.last_step": 20.0},
    }


def test_performance_summary_reduces_raw_telemetry():
    summary = build_performance_summary(
        _snapshot(),
        wall_time_s=10.0,
        steps_executed=20,
        top_n=2,
    )

    assert summary["wall_time_s"] == 10.0
    assert summary["events_per_wall_second"] == 2.0
    assert summary["cache"] == {
        "hits": 8,
        "misses": 2,
        "hit_ratio": pytest.approx(0.8),
    }
    assert summary["calculations"] == {
        "optimization_runs": 7,
        "neb_runs": 4,
        "total_expensive_runs": 11,
        "calculation_record_writes": 3,
    }
    assert summary["overhead_s"] == {
        "output": pytest.approx(0.8),
        "checkpoint": pytest.approx(0.5),
        "output_and_checkpoint": pytest.approx(1.3),
    }
    assert [item["name"] for item in summary["top_bottlenecks"]] == [
        "kmc.session.seconds",
        "neb.seconds",
    ]


def test_raw_performance_is_written_to_versioned_diagnostics(tmp_path):
    snapshot = _snapshot()
    summary = build_performance_summary(
        snapshot,
        wall_time_s=10.0,
        steps_executed=20,
    )
    path = write_performance_diagnostics(
        tmp_path / "diagnostics" / "performance.json",
        run_id="performance-test",
        telemetry=snapshot,
        summary=summary,
        simulated_time_s=2.5,
        wall_time_s=10.0,
        steps_executed=20,
        termination_status="complete",
        termination_reason="requested_steps_completed",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["telemetry"] == snapshot
    assert payload["summary"] == summary
    assert payload["simulated_time_s"] == 2.5
    assert payload["termination"] == {
        "status": "complete",
        "reason": "requested_steps_completed",
    }
