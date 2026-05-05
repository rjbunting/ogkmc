"""Rate totals and stochastic sampling helpers for KMC."""

from __future__ import annotations

import random
from typing import Iterable, Protocol, TypeVar

import numpy as np


class Rated(Protocol):
    rate: float


ReactionT = TypeVar("ReactionT", bound=Rated)


class _RateSegmentTree:
    """Sum-segment tree over a fixed-size rate vector."""

    __slots__ = ("_n", "_size", "_tree")

    def __init__(self, n: int):
        self._n = int(n)
        size = 1
        while size < max(1, self._n):
            size *= 2
        self._size = size
        self._tree = np.zeros(2 * size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def __len__(self) -> int:
        return self._n

    def update(self, i: int, value: float) -> None:
        """Set leaf *i* to *value* clipped at zero, then propagate sums."""
        if i < 0 or i >= self._n:
            raise IndexError(f"leaf index {i} out of range [0, {self._n})")
        pos = self._size + i
        self._tree[pos] = max(0.0, float(value))
        pos //= 2
        while pos:
            self._tree[pos] = self._tree[2 * pos] + self._tree[2 * pos + 1]
            pos //= 2

    def sample(self, u: float) -> int:
        """Return the leaf id whose prefix sum first exceeds ``u * total``."""
        total = self._tree[1]
        if total <= 0.0:
            return -1
        u = min(max(float(u), 0.0), float(np.nextafter(1.0, 0.0)))
        target = u * float(total)
        pos = 1
        while pos < self._size:
            left = self._tree[2 * pos]
            if target < left:
                pos = 2 * pos
            else:
                target -= left
                pos = 2 * pos + 1
        return min(pos - self._size, self._n - 1)


def total_rate(reactions: Iterable[Rated]) -> float:
    """Sum positive rates over a reaction iterable."""
    rates = np.fromiter(
        (r.rate if r.rate > 0.0 else 0.0 for r in reactions),
        dtype=np.float64,
    )
    return float(rates.sum()) if rates.size else 0.0


def sample_tau(q_total: float, rng: random.Random | np.random.Generator) -> float:
    """KMC time increment sampled from ``Exp(q_total)``."""
    if q_total <= 0.0:
        return float("inf")
    u = float(rng.random())
    if u <= 0.0:
        u = float(np.nextafter(0.0, 1.0))
    return float(np.log(1.0 / u) / q_total)


def choose_reaction(
    reactions: list[ReactionT],
    rng: random.Random | np.random.Generator,
) -> tuple[ReactionT | None, int | None, float]:
    """Pick one reaction with probability proportional to its positive rate."""
    if not reactions:
        return None, None, 0.0
    rates = np.fromiter(
        (r.rate if r.rate > 0.0 else 0.0 for r in reactions),
        dtype=np.float64,
    )
    total = float(rates.sum())
    if total <= 0.0:
        return None, None, 0.0

    target = float(rng.random()) * total
    idx = int(np.searchsorted(np.cumsum(rates), target, side="right"))
    if idx >= len(reactions):
        idx = len(reactions) - 1
    return reactions[idx], idx, total


__all__ = ["_RateSegmentTree", "total_rate", "sample_tau", "choose_reaction"]
