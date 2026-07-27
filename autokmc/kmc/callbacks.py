"""Protocols for optional KMC callback-style collaborators."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class CalculatorLike(Protocol):
    """Minimum ASE-calculator surface consumed by scientific kernels."""

    def get_potential_energy(
        self,
        atoms=None,
        force_consistent: bool = False,
    ) -> float: ...


class CalculatorPoolLike(Protocol):
    """Typed calculator-pool resource accepted by parallel kernels."""

    @property
    def primary(self) -> CalculatorLike: ...

    def __len__(self) -> int: ...

    def acquire(self) -> AbstractContextManager[CalculatorLike]: ...


class ReactionWriterLike(Protocol):
    def ensure_reaction(
        self,
        reaction,
        *,
        step: int = 0,
        gas_energies: dict | None = None,
        gas_free_energies: dict | None = None,
    ): ...

    def write_invalid_diffusion(
        self,
        site,
        lateral_class,
        *,
        step: int = 0,
    ): ...

    def record(
        self,
        *,
        step: int,
        time_s: float,
        tau_s: float,
        reaction,
        gas_energies: dict | None = None,
        gas_free_energies: dict | None = None,
        **kwargs,
    ): ...


class TrajectoryWriterLike(Protocol):
    append: bool

    @property
    def enabled(self) -> bool: ...

    def maybe_write(self, atoms, *, step: int) -> bool: ...

    def close(self) -> None: ...


class SummaryCollectorLike(Protocol):
    def add(self, reaction, *, step: int) -> None: ...


class CheckpointWriterLike(Protocol):
    """Checkpoint sink used by periodic and forced final writes."""

    def maybe_write(self, **payload: Any): ...


__all__ = [
    "CalculatorLike",
    "CalculatorPoolLike",
    "CheckpointWriterLike",
    "ReactionWriterLike",
    "TrajectoryWriterLike",
    "SummaryCollectorLike",
]
