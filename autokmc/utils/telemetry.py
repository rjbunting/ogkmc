"""Lightweight, run-scoped operational telemetry.

The scientific kernels remain usable as ordinary functions.  When a KMC run
installs a :class:`RuntimeTelemetry` collector, the helpers in this module
accumulate counters, gauges, and wall-clock timings without introducing a
global mutable singleton or a logging dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class RuntimeTelemetry:
    """Thread-safe counters and accumulated wall-clock timings for one run."""

    counters: dict[str, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + int(amount)

    def add_time(self, name: str, seconds: float) -> None:
        with self._lock:
            self.timings_s[name] = self.timings_s.get(name, 0.0) + float(seconds)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = float(value)

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Return a deterministic JSON-ready snapshot."""
        with self._lock:
            return {
                "counters": dict(sorted(self.counters.items())),
                "timings_s": {
                    name: float(value)
                    for name, value in sorted(self.timings_s.items())
                },
                "gauges": {
                    name: float(value)
                    for name, value in sorted(self.gauges.items())
                },
            }


_CURRENT: ContextVar[RuntimeTelemetry | None] = ContextVar(
    "autokmc_runtime_telemetry",
    default=None,
)


def current_telemetry() -> RuntimeTelemetry | None:
    """Return the collector installed for the current execution context."""
    return _CURRENT.get()


@contextmanager
def telemetry_context(collector: RuntimeTelemetry) -> Iterator[RuntimeTelemetry]:
    """Install *collector* for nested scientific and persistence calls."""
    token: Token[RuntimeTelemetry | None] = _CURRENT.set(collector)
    try:
        yield collector
    finally:
        _CURRENT.reset(token)


def increment(name: str, amount: int = 1) -> None:
    collector = current_telemetry()
    if collector is not None:
        collector.increment(name, amount)


def set_gauge(name: str, value: float) -> None:
    collector = current_telemetry()
    if collector is not None:
        collector.set_gauge(name, value)


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Accumulate elapsed wall time under *name* when telemetry is active."""
    collector = current_telemetry()
    if collector is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        collector.add_time(name, perf_counter() - started)


def instrument(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Count and time calls to a synchronous function."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            increment(f"{name}.calls")
            try:
                with timed(f"{name}.seconds"):
                    return function(*args, **kwargs)
            except BaseException:
                increment(f"{name}.failures")
                raise

        return wrapped

    return decorate


__all__ = [
    "RuntimeTelemetry",
    "current_telemetry",
    "increment",
    "instrument",
    "set_gauge",
    "telemetry_context",
    "timed",
]
