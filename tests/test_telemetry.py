"""Run-scoped operational telemetry helpers."""

from __future__ import annotations

import importlib
import threading

import networkx as nx
import pytest

from ogkmc.io.calculators import CalculatorPool
from ogkmc.kmc.network import DynamicNetworkExpander
from ogkmc.utils.telemetry import (
    RuntimeTelemetry,
    current_telemetry,
    increment,
    instrument,
    set_gauge,
    telemetry_context,
    timed,
)


def test_telemetry_context_collects_and_isolates_metrics(monkeypatch):
    collector = RuntimeTelemetry()
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr("ogkmc.utils.telemetry.perf_counter", lambda: next(ticks))

    increment("outside")
    with telemetry_context(collector):
        increment("events", 2)
        set_gauge("active", 7)
        with timed("work.seconds"):
            pass

    assert collector.to_dict() == {
        "counters": {"events": 2},
        "timings_s": {"work.seconds": pytest.approx(0.25)},
        "gauges": {"active": 7.0},
    }


def test_instrument_counts_failures():
    collector = RuntimeTelemetry()

    @instrument("kernel")
    def fail() -> None:
        raise RuntimeError("boom")

    with telemetry_context(collector), pytest.raises(RuntimeError, match="boom"):
        fail()

    assert collector.counters == {"kernel.calls": 1, "kernel.failures": 1}
    assert collector.timings_s["kernel.seconds"] >= 0.0


def test_calculation_cache_reports_lookup_miss(tmp_path):
    from ogkmc.io.calculation_cache import load_calculation_record

    collector = RuntimeTelemetry()
    with telemetry_context(collector):
        assert load_calculation_record(tmp_path / "missing", "adsorption", "key") is None

    assert collector.counters["calculation_cache.lookup.calls"] == 1
    assert collector.counters["calculation_cache.misses"] == 1
    assert collector.timings_s["calculation_cache.lookup.seconds"] >= 0.0


@pytest.mark.parametrize(
    ("module_name", "worker_name", "compute_name"),
    [
        (
            "ogkmc.reactions.adsorption",
            "get_applicable_reactions",
            "compute_all_reactions",
        ),
        (
            "ogkmc.reactions.diffusion",
            "get_applicable_diffusions",
            "compute_all_diffusions",
        ),
        (
            "ogkmc.reactions.bond",
            "get_applicable_bond_reactions",
            "compute_all_bond_reactions",
        ),
    ],
)
def test_parallel_site_sweeps_propagate_run_telemetry(
    monkeypatch,
    module_name,
    worker_name,
    compute_name,
):
    module = importlib.import_module(module_name)
    collector = RuntimeTelemetry()
    barrier = threading.Barrier(2)
    worker_threads: set[int] = set()

    def worker(*_args, **_kwargs):
        assert current_telemetry() is collector
        worker_threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        increment("parallel.site_sweeps")
        return []

    monkeypatch.setattr(module, worker_name, worker)
    pool = CalculatorPool([object(), object()], max_workers=2)
    sites = [object(), object()]

    with telemetry_context(collector):
        compute = getattr(module, compute_name)
        if compute_name == "compute_all_reactions":
            compute(nx.Graph(), sites, pool, {}, temperature=500.0)
        else:
            compute(nx.Graph(), sites, pool, temperature=500.0)

    assert collector.counters["parallel.site_sweeps"] == 2
    assert len(worker_threads) == 2


def test_dynamic_network_expansion_reports_call_and_failure(monkeypatch):
    collector = RuntimeTelemetry()
    expander = object.__new__(DynamicNetworkExpander)

    def fail(*_args, **_kwargs):
        raise RuntimeError("expansion failed")

    monkeypatch.setattr(DynamicNetworkExpander, "_announce_event", fail)
    with telemetry_context(collector), pytest.raises(
        RuntimeError,
        match="expansion failed",
    ):
        expander.expand(object())

    assert collector.counters["kmc.expansion.calls"] == 1
    assert collector.counters["kmc.expansion.failures"] == 1
    assert collector.timings_s["kmc.expansion.seconds"] >= 0.0


def test_configured_pipeline_installs_telemetry_before_preparation(
    tmp_path,
    monkeypatch,
):
    import ogkmc.cli.pipeline as pipeline_module
    from ogkmc.io.config import OutputCfg, RunConfig

    def fake_pipeline(cfg, *, config_path, telemetry):
        assert isinstance(cfg, RunConfig)
        assert config_path == "run.yaml"
        assert current_telemetry() is telemetry
        increment("optimization.structure.calls")
        return {"performance": telemetry.to_dict()}

    monkeypatch.setattr(pipeline_module, "_run_from_config", fake_pipeline)

    result = pipeline_module.run_from_config(
        RunConfig(output=OutputCfg(dir=str(tmp_path / "run"))),
        config_path="run.yaml",
    )

    assert result["performance"]["counters"]["optimization.structure.calls"] == 1
