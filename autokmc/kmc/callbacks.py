"""Protocols for optional KMC callback-style collaborators."""

from __future__ import annotations

from typing import Protocol


class ReactionWriterLike(Protocol):
    def ensure_reaction(
        self,
        reaction,
        *,
        step: int = 0,
        gas_energies: dict | None = None,
        gas_free_energies: dict | None = None,
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
    @property
    def enabled(self) -> bool: ...

    def maybe_write(self, atoms, *, step: int) -> bool: ...


class SummaryCollectorLike(Protocol):
    def add(self, reaction, *, step: int) -> None: ...


class ProductTrackerLike(Protocol):
    def seed_from_graph(
        self,
        G,
        adsorbate_sites,
        *,
        step: int = 0,
        time_s: float = 0.0,
    ) -> None: ...

    def record_event(
        self,
        G,
        reaction,
        *,
        step: int,
        time_s: float,
        tau_s: float,
    ): ...


__all__ = [
    "ReactionWriterLike",
    "TrajectoryWriterLike",
    "SummaryCollectorLike",
    "ProductTrackerLike",
]
