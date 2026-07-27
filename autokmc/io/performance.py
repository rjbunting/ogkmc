"""Versioned run-performance diagnostics and concise summary metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autokmc.io._files import write_json_atomic


PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION = "1"
PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH = Path("diagnostics") / "performance.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _counter(snapshot: Mapping[str, Any], name: str) -> int:
    counters = snapshot.get("counters", {})
    if not isinstance(counters, Mapping):
        return 0
    return int(counters.get(name, 0) or 0)


def _timings(snapshot: Mapping[str, Any]) -> dict[str, float]:
    values = snapshot.get("timings_s", {})
    if not isinstance(values, Mapping):
        return {}
    return {
        str(name): float(seconds)
        for name, seconds in values.items()
        if isinstance(seconds, (int, float)) and float(seconds) >= 0.0
    }


def build_performance_summary(
    snapshot: Mapping[str, Any],
    *,
    wall_time_s: float,
    steps_executed: int,
    top_n: int = 5,
) -> dict[str, Any]:
    """Reduce raw telemetry to stable, user-facing run metrics."""
    wall_time = max(0.0, float(wall_time_s))
    steps = max(0, int(steps_executed))
    hits = _counter(snapshot, "calculation_cache.hits")
    misses = _counter(snapshot, "calculation_cache.misses")
    cache_lookups = hits + misses
    timings = _timings(snapshot)
    counters = snapshot.get("counters", {})
    optimization_runs = (
        sum(
            int(value or 0)
            for name, value in counters.items()
            if str(name).startswith("optimization.")
            and str(name).endswith(".calls")
        )
        if isinstance(counters, Mapping)
        else 0
    )
    neb_runs = _counter(snapshot, "neb.calls")

    checkpoint_overhead = timings.get("checkpoint.save.seconds", 0.0)
    output_overhead = sum(
        seconds
        for name, seconds in timings.items()
        if name.startswith("output.")
    )
    output_overhead += timings.get("persistence.event_append.seconds", 0.0)
    if "persistence.event_sync.seconds" in timings:
        # event_sync owns metadata_flush, so summing both would count the
        # nested flush interval twice.
        output_overhead += timings["persistence.event_sync.seconds"]
    else:
        # Preserve meaningful standalone instrumentation if a custom writer
        # invokes metadata flushing outside the built-in sync boundary.
        output_overhead += timings.get("persistence.metadata_flush.seconds", 0.0)
    bottlenecks = [
        {
            "name": name,
            "wall_time_s": seconds,
            "fraction_of_wall_time": (
                seconds / wall_time if wall_time > 0.0 else None
            ),
        }
        for name, seconds in sorted(
            timings.items(),
            key=lambda item: (-item[1], item[0]),
        )[: max(0, int(top_n))]
    ]

    return {
        "schema_version": PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION,
        "wall_time_s": wall_time,
        "events_per_wall_second": (
            float(steps) / wall_time if wall_time > 0.0 else None
        ),
        "cache": {
            "hits": hits,
            "misses": misses,
            "hit_ratio": (
                float(hits) / float(cache_lookups)
                if cache_lookups > 0
                else None
            ),
        },
        "calculations": {
            "optimization_runs": optimization_runs,
            "neb_runs": neb_runs,
            "total_expensive_runs": optimization_runs + neb_runs,
            "calculation_record_writes": _counter(
                snapshot,
                "calculation_cache.write.calls",
            ),
        },
        "overhead_s": {
            "output": output_overhead,
            "checkpoint": checkpoint_overhead,
            "output_and_checkpoint": output_overhead + checkpoint_overhead,
        },
        "top_bottlenecks": bottlenecks,
    }


def write_performance_diagnostics(
    path: str | Path,
    *,
    run_id: str,
    telemetry: Mapping[str, Any],
    summary: Mapping[str, Any],
    simulated_time_s: float,
    wall_time_s: float,
    steps_executed: int,
    termination_status: str,
    termination_reason: str,
) -> Path:
    """Persist the complete telemetry snapshot separately from ``summary.json``."""
    payload = {
        "schema_version": PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION,
        "kind": "autokmc-performance-diagnostics",
        "generated_utc": _utc_now(),
        "run_id": str(run_id),
        "termination": {
            "status": str(termination_status),
            "reason": str(termination_reason),
        },
        "steps_executed": int(steps_executed),
        "simulated_time_s": float(simulated_time_s),
        "wall_time_s": max(0.0, float(wall_time_s)),
        "summary": dict(summary),
        "telemetry": dict(telemetry),
    }
    return write_json_atomic(path, payload, sort_keys=True)


__all__ = [
    "PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH",
    "PERFORMANCE_DIAGNOSTICS_SCHEMA_VERSION",
    "build_performance_summary",
    "write_performance_diagnostics",
]
