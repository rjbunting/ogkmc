"""Tests for calculator pool construction."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest

import autokmc.io.calculation_cache as calculation_cache
from autokmc.io.calculators import (
    CalculatorCfg,
    CalculatorConfigError,
    CalculatorPool,
    build_calculator,
    calculator_batch_active,
    calculator_batch_context,
    invalidate_calculator_identity,
)
from autokmc.io.calculation_cache import calculator_identity


class _OpaqueCalculatorWithToDict:
    def todict(self):
        return {"backend": "unverifiable"}


def test_build_calculator_pool_from_copies():
    calc = build_calculator(
        CalculatorCfg(import_path="ase.calculators.emt.EMT", copies=2)
    )
    assert isinstance(calc, CalculatorPool)
    assert len(calc) == 2
    with calc.acquire() as c0, calc.acquire() as c1:
        assert c0 is not c1


def test_build_calculator_returns_pool_for_single_copy():
    calc = build_calculator(
        CalculatorCfg(import_path="ase.calculators.emt.EMT")
    )
    assert isinstance(calc, CalculatorPool)
    assert len(calc) == 1


def test_calculator_pool_reuses_bounded_executor_and_shuts_down_cleanly():
    calc = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            copies=2,
            max_workers=99,
        )
    )
    assert isinstance(calc, CalculatorPool)
    assert calc.max_workers == 2
    executor = calc.executor
    assert calc.executor is executor
    assert [future.result() for future in (calc.submit(abs, -1), calc.submit(abs, -2))] == [
        1,
        2,
    ]

    calc.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        calc.submit(abs, -3)


def test_calculator_pool_rejects_nonpositive_worker_limit():
    with pytest.raises(CalculatorConfigError, match="must be positive"):
        CalculatorPool([object()], max_workers=0)
    with pytest.raises(CalculatorConfigError, match="must be positive"):
        CalculatorPool([object()], max_workers=-1)


def test_calculator_pool_rejects_duplicate_instances():
    calculator = object()
    with pytest.raises(
        CalculatorConfigError,
        match="independent calculator instances",
    ):
        CalculatorPool([calculator, calculator])


def test_calculator_pool_enforces_worker_limit_across_independent_executors():
    pool = CalculatorPool(
        [object(), object(), object(), object()],
        max_workers=2,
    )
    lock = threading.Lock()
    two_active = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0

    def use_calculator():
        nonlocal active, peak
        with pool.acquire():
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_active.set()
            assert release.wait(timeout=1.0)
            with lock:
                active -= 1

    with (
        ThreadPoolExecutor(max_workers=2) as first,
        ThreadPoolExecutor(max_workers=2) as second,
    ):
        futures = [
            first.submit(use_calculator),
            first.submit(use_calculator),
            second.submit(use_calculator),
            second.submit(use_calculator),
        ]
        assert two_active.wait(timeout=1.0)
        with lock:
            assert active == 2
        release.set()
        for future in futures:
            future.result(timeout=1.0)

    assert peak == 2


def test_calculator_pool_gather_drains_batch_before_propagating_error():
    pool = CalculatorPool([object(), object()])
    started = threading.Event()
    finished = threading.Event()

    def fail_after_sibling_starts():
        assert started.wait(timeout=1.0)
        raise RuntimeError("batch failure")

    def complete_sibling():
        started.set()
        time.sleep(0.02)
        finished.set()

    futures = [
        pool.submit(fail_after_sibling_starts),
        pool.submit(complete_sibling),
    ]
    with pytest.raises(RuntimeError, match="batch failure"):
        pool.gather(futures)

    assert finished.is_set()
    pool.shutdown()


def test_calculator_batch_context_is_nested_and_scoped():
    assert calculator_batch_active() is False
    with calculator_batch_context():
        assert calculator_batch_active() is True
        with calculator_batch_context():
            assert calculator_batch_active() is True
        assert calculator_batch_active() is True
    assert calculator_batch_active() is False


def test_acquire_many_requires_enough_independent_calculators():
    calc = build_calculator(
        CalculatorCfg(factory="types.SimpleNamespace", copies=2)
    )
    assert isinstance(calc, CalculatorPool)
    with calc.acquire_many(2, purpose="test batch") as calcs:
        assert len(calcs) == 2
        assert calcs[0] is not calcs[1]
    with pytest.raises(CalculatorConfigError, match="3 independent"):
        with calc.acquire_many(3, purpose="test batch"):
            pass


def test_acquire_many_cannot_exceed_worker_limit():
    pool = CalculatorPool([object(), object(), object()], max_workers=2)
    with pytest.raises(CalculatorConfigError, match="allows only 2 worker"):
        with pool.acquire_many(3, purpose="test batch"):
            pass


def test_acquire_many_reserves_batch_atomically():
    pool = CalculatorPool(
        [object(), object(), object(), object()],
        max_workers=2,
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_batch():
        with pool.acquire_many(2):
            first_entered.set()
            assert release_first.wait(timeout=1.0)

    def second_batch():
        assert first_entered.wait(timeout=1.0)
        with pool.acquire_many(2):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_batch)
        second = executor.submit(second_batch)
        assert first_entered.wait(timeout=1.0)
        assert second_entered.wait(timeout=0.05) is False
        release_first.set()
        first.result(timeout=1.0)
        second.result(timeout=1.0)

    assert second_entered.is_set()


def test_nested_acquisition_fails_instead_of_deadlocking_when_pool_is_full():
    pool = CalculatorPool([object(), object()], max_workers=1)
    with pool.acquire():
        with pytest.raises(CalculatorConfigError, match="nested request"):
            with pool.acquire(purpose="nested test"):
                pass

    with pool.acquire() as calculator:
        assert calculator in pool.calculators


def test_nested_acquisition_succeeds_when_capacity_is_immediately_available():
    pool = CalculatorPool([object(), object()], max_workers=2)
    with pool.acquire() as first:
        with pool.acquire() as second:
            assert first is not second


def test_build_calculator_pool_with_nested_factory_and_dotted_gpu_arg():
    calc = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={
                "predictor": {
                    "factory": "types.SimpleNamespace",
                    "factory_kwargs": {"device": "cpu"},
                },
                "task_name": "oc20",
            },
            gpu_devices=["cuda:0", "cuda:1"],
            gpu_device_arg="predictor.factory_kwargs.device",
        )
    )

    assert isinstance(calc, CalculatorPool)
    with calc.acquire() as c0, calc.acquire() as c1:
        assert c0.predictor.device == "cuda:0"
        assert c1.predictor.device == "cuda:1"
        assert c0.task_name == "oc20"


def test_build_fairchem_predictors_on_distinct_cuda_devices(monkeypatch):
    class FakeDevice:
        def __init__(self, value):
            text = str(value)
            device_type, separator, index = text.partition(":")
            self.type = device_type
            self.index = int(index) if separator else None

        def __str__(self):
            if self.index is None:
                return self.type
            return f"{self.type}:{self.index}"

    class FakeCuda:
        current = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

        def device(self, index):
            cuda = self

            class DeviceContext:
                def __enter__(self):
                    self.previous = cuda.current
                    cuda.current = int(index)

                def __exit__(self, exc_type, exc, traceback):
                    cuda.current = self.previous

            return DeviceContext()

    fake_cuda = FakeCuda()
    calls = []

    def get_predict_unit(name_or_path, *, device, workers, **kwargs):
        calls.append((name_or_path, device, workers, fake_cuda.current, kwargs))
        return SimpleNamespace(device=f"cuda:{fake_cuda.current}")

    fake_torch = ModuleType("torch")
    fake_torch.device = FakeDevice
    fake_torch.cuda = fake_cuda
    fake_pretrained = SimpleNamespace(get_predict_unit=get_predict_unit)
    fake_core = ModuleType("fairchem.core")
    fake_core.pretrained_mlip = fake_pretrained
    fake_core.FAIRChemCalculator = SimpleNamespace
    fake_fairchem = ModuleType("fairchem")
    fake_fairchem.core = fake_core
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "fairchem", fake_fairchem)
    monkeypatch.setitem(sys.modules, "fairchem.core", fake_core)

    pool = build_calculator(
        CalculatorCfg(
            factory="fairchem.core.FAIRChemCalculator",
            factory_kwargs={
                "predict_unit": {
                    "factory": "autokmc.io.fairchem.get_predict_unit_on_device",
                    "factory_kwargs": {
                        "name_or_path": "uma-s-1p2",
                        "device": "cuda",
                    },
                },
                "task_name": "oc20",
            },
            copies=4,
            gpu_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            gpu_device_arg="predict_unit.factory_kwargs.device",
            max_workers=4,
        )
    )

    assert [calc.predict_unit.device for calc in pool.calculators] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]
    assert [(call[1], call[2], call[3]) for call in calls] == [
        ("cuda", 1, 0),
        ("cuda", 1, 1),
        ("cuda", 1, 2),
        ("cuda", 1, 3),
    ]
    assert all(calc.task_name == "oc20" for calc in pool.calculators)


def test_built_calculator_identity_retains_factory_alias_and_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        calculation_cache,
        "_configured_entry_point_versions",
        lambda _configured: {"calculator-distribution": "1.0"},
    )
    first = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "model-revision-a", "device": "cpu"},
        )
    )
    second = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "model-revision-b", "device": "cuda"},
        )
    )

    first_identity = calculator_identity(first)
    second_identity = calculator_identity(second)

    assert first_identity["configured"]["factory"] == "types.SimpleNamespace"
    assert (
        first_identity["configured"]["factory_kwargs"]["model_alias"]
        == "model-revision-a"
    )
    assert "device" not in first_identity["configured"]["factory_kwargs"]
    assert (
        first_identity["method_digest_sha256"]
        != second_identity["method_digest_sha256"]
    )


def test_configured_entry_point_versions_include_nested_colon_specs(monkeypatch):
    monkeypatch.setattr(
        calculation_cache,
        "_installed_package_distributions",
        lambda: {
            "outer_backend": ["outer-distribution"],
            "nested_backend": ["nested-distribution"],
        },
    )
    versions = {
        "outer-distribution": "1.2.3",
        "nested-distribution": "4.5.6",
    }
    monkeypatch.setattr(
        calculation_cache.importlib_metadata,
        "version",
        versions.__getitem__,
    )

    resolved = calculation_cache._configured_entry_point_versions(
        {
            "factory": "outer_backend.build",
            "factory_kwargs": {
                "predictor": {
                    "factory": "nested_backend:create",
                    "factory_kwargs": {},
                },
            },
        }
    )

    assert resolved == versions


def test_versioned_opaque_factory_identity_tracks_version_not_device(
    monkeypatch,
):
    cpu = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "immutable-revision", "device": "cpu"},
        )
    )
    gpu = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "immutable-revision", "device": "cuda"},
        )
    )
    monkeypatch.setattr(
        calculation_cache,
        "_configured_entry_point_versions",
        lambda _configured: {"calculator-distribution": "1.0"},
    )
    version_one = calculator_identity(cpu)
    same_version_gpu = calculator_identity(gpu)

    assert version_one == same_version_gpu
    assert "opaque_instance" not in version_one
    assert version_one["entry_point_distributions"] == {
        "calculator-distribution": "1.0",
    }

    monkeypatch.setattr(
        calculation_cache,
        "_configured_entry_point_versions",
        lambda _configured: {"calculator-distribution": "2.0"},
    )
    version_two = calculator_identity(cpu, refresh=True)

    assert (
        version_one["method_digest_sha256"]
        != version_two["method_digest_sha256"]
    )


def test_calculator_identity_snapshots_artifact_until_explicit_invalidation(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights-a")
    calculator = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"checkpoint": checkpoint},
        )
    )
    calls = 0
    original = calculation_cache._sha256_file

    def _counting_hash(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(calculation_cache, "_sha256_file", _counting_hash)
    first = calculator_identity(calculator)
    repeated = calculator_identity(calculator)
    assert repeated == first
    assert calls == 1

    checkpoint.write_bytes(b"weights-b")
    # The loaded calculator still represents its original weight snapshot.
    assert calculator_identity(calculator) == first
    assert calls == 1

    invalidate_calculator_identity(calculator)
    refreshed = calculator_identity(calculator)
    assert refreshed["method_digest_sha256"] != first["method_digest_sha256"]
    assert calls == 2


def test_unversioned_opaque_factory_identity_is_process_local(monkeypatch):
    monkeypatch.setattr(
        calculation_cache,
        "_configured_entry_point_versions",
        lambda _configured: {},
    )
    first = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "immutable-revision"},
        )
    )
    second = build_calculator(
        CalculatorCfg(
            factory="types.SimpleNamespace",
            factory_kwargs={"model_alias": "immutable-revision"},
        )
    )

    first_identity = calculator_identity(first)
    repeated = calculator_identity(first)
    second_identity = calculator_identity(second)

    assert first_identity == repeated
    assert (
        first_identity["opaque_instance"]["identity_scope"]
        == "process-local"
    )
    assert (
        first_identity["method_digest_sha256"]
        != second_identity["method_digest_sha256"]
    )


def test_unversioned_opaque_fallback_is_local_even_with_todict(monkeypatch):
    monkeypatch.setattr(
        calculation_cache,
        "_configured_entry_point_versions",
        lambda _configured: {},
    )
    first = _OpaqueCalculatorWithToDict()
    second = _OpaqueCalculatorWithToDict()
    declaration = {
        "construction": "factory",
        "factory": "unversioned_backend:create",
        "factory_kwargs": {},
    }
    first._autokmc_calculator_config_identity = declaration
    second._autokmc_calculator_config_identity = declaration

    first_identity = calculator_identity(first)
    second_identity = calculator_identity(second)

    assert first_identity["opaque_instance"]["identity_scope"] == "process-local"
    assert (
        first_identity["method_digest_sha256"]
        != second_identity["method_digest_sha256"]
    )
