"""Tests for calculator pool construction."""

from __future__ import annotations

import pytest

from autokmc.io.calculators import (
    CalculatorCfg,
    CalculatorConfigError,
    CalculatorPool,
    build_calculator,
)


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
