"""Rate totals and stochastic sampling helpers for KMC."""

from __future__ import annotations

import random
from typing import Iterable, Protocol, TypeVar

import numpy as np


class Rated(Protocol):
    rate: float


ReactionT = TypeVar("ReactionT", bound=Rated)


class _RateSegmentTree:
    """Sum-segment tree over a rate vector with amortised incremental growth."""

    __slots__ = ("_n", "_size", "_tree")

    def __init__(self, n: int):
        self._n = int(n)
        size = 1
        while size < max(1, self._n):
            size *= 2
        self._size = size
        self._tree: np.ndarray = np.zeros(2 * size, dtype=np.float64)

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

    def build(self, values: Iterable[float]) -> None:
        """Replace all leaf values and construct the tree in linear time."""
        rates = np.fromiter(
            (max(0.0, float(value)) for value in values),
            dtype=np.float64,
        )
        if rates.size != self._n:
            raise ValueError(
                f"expected {self._n} rates, received {rates.size}"
            )
        self._tree.fill(0.0)
        if self._n:
            self._tree[self._size : self._size + self._n] = rates
        for pos in range(self._size - 1, 0, -1):
            self._tree[pos] = self._tree[2 * pos] + self._tree[2 * pos + 1]

    def update_many(self, updates: Iterable[tuple[int, float]]) -> None:
        """Set several leaves while propagating each affected parent once."""
        parents: set[int] = set()
        for raw_index, value in updates:
            index = int(raw_index)
            if index < 0 or index >= self._n:
                raise IndexError(
                    f"leaf index {index} out of range [0, {self._n})"
                )
            pos = self._size + index
            self._tree[pos] = max(0.0, float(value))
            if pos > 1:
                parents.add(pos // 2)

        while parents:
            next_parents: set[int] = set()
            for pos in parents:
                self._tree[pos] = (
                    self._tree[2 * pos] + self._tree[2 * pos + 1]
                )
                if pos > 1:
                    next_parents.add(pos // 2)
            parents = next_parents

    def grow(self, n: int) -> None:
        """Extend the addressable leaf range to *n*, preserving all rates.

        Capacity grows geometrically.  Most dynamic network expansions only
        expose previously-unused leaves and therefore cost O(1); a bulk tree
        copy/reduction occurs only when the power-of-two capacity is crossed.
        """
        new_n = int(n)
        if new_n < self._n:
            raise ValueError(
                f"cannot shrink rate tree from {self._n} to {new_n} leaves"
            )
        if new_n == self._n:
            return
        if new_n <= self._size:
            self._n = new_n
            return

        old_n = self._n
        old_values = self._tree[self._size : self._size + old_n].copy()
        new_size = self._size
        while new_size < new_n:
            new_size *= 2
        new_tree: np.ndarray = np.zeros(2 * new_size, dtype=np.float64)
        new_tree[new_size : new_size + old_n] = old_values
        for pos in range(new_size - 1, 0, -1):
            new_tree[pos] = new_tree[2 * pos] + new_tree[2 * pos + 1]
        self._n = new_n
        self._size = new_size
        self._tree = new_tree

    def sample(self, u: float) -> int:
        """Return the leaf id whose prefix sum first exceeds ``u * total``."""
        total = self._tree[1]
        if total <= 0.0:
            return -1
        u = min(max(float(u), 0.0), float(np.nextafter(1.0, 0.0)))
        # ``u < 1`` is not by itself enough: multiplication can still round
        # back up to ``total``.  Keep the target strictly inside the sampled
        # interval so the descent cannot walk into zero-capacity tail leaves.
        target = min(
            u * float(total),
            float(np.nextafter(total, 0.0)),
        )
        original_target = target
        pos = 1
        while pos < self._size:
            left = self._tree[2 * pos]
            right = self._tree[2 * pos + 1]
            if left > 0.0 and (target < left or right <= 0.0):
                # If roundoff put the target on/above a parent whose right
                # subtree is empty, clamp it back inside the nonempty left
                # subtree before continuing.
                target = min(target, float(np.nextafter(left, 0.0)))
                pos = 2 * pos
            elif right > 0.0:
                target = max(0.0, target - left)
                target = min(target, float(np.nextafter(right, 0.0)))
                pos = 2 * pos + 1
            else:  # pragma: no cover - defensive guard against corrupt trees
                return -1

        index = pos - self._size
        if index < self._n and self._tree[pos] > 0.0:
            return index

        # The guarded descent above should always land on a positive real
        # leaf.  Retain a numerically robust, rare fallback so callers never
        # interpret an internal floating-point boundary as ``total_rate=0``.
        running = 0.0
        last_positive = -1
        for candidate, weight in enumerate(
            self._tree[self._size : self._size + self._n]
        ):
            if weight <= 0.0:
                continue
            last_positive = candidate
            running += float(weight)
            if original_target < running:
                return candidate
        return last_positive


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
