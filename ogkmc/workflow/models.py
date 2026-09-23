"""Typed state exchanged between workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ogkmc.kmc.models import (
    AdsorptionChannelOptions,
    BondChannelOptions,
    BondGrowthOptions,
    DiffusionChannelOptions,
    KMCResumeState,
)


@dataclass(frozen=True)
class RunIdentity:
    """Resolved identity and restart state for one configured invocation."""

    output_dir: Path
    manifest_path: Path
    run_id: str
    resume_state: Any | None = None
    event_commit: Any | None = None

    @property
    def is_resume(self) -> bool:
        return self.resume_state is not None


@dataclass
class PreparedSystem:
    """Catalyst structure and graph produced by the structure stages."""

    graph: Any
    atoms: Any | None
    frozen_indices: list[int] | None
    surface_result: Any | None = None
    structure_source: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedCalculator:
    """Calculator resource plus its representative concrete calculator."""

    resource: Any
    primary: Any


@dataclass
class PreparedStructure:
    """Built structure state before surface classification and graph creation."""

    atoms: Any | None
    frozen_indices: list[int] | None
    structure_source: dict[str, Any] | None = None


@dataclass
class PreparedNetwork:
    """Species and reaction-site registries handed to the KMC engine."""

    reactants: list[Any] = field(default_factory=list)
    adsorbate_sites: list[Any] = field(default_factory=list)
    initial_adsorbate_sites: list[Any] = field(default_factory=list)
    diffusion_sites: list[Any] = field(default_factory=list)
    bond_sites: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class ThermoRuntime:
    """Resolved thermochemistry and calculation-cache paths."""

    options: Any
    vibration_cache_root: str
    calculation_cache_root: str | None
    calculation_cache_lookup_enabled: bool = False


@dataclass(frozen=True)
class ChannelRuntimeOptions:
    """Typed home for the keyword groups consumed by reaction channels."""

    adsorption: AdsorptionChannelOptions = field(
        default_factory=AdsorptionChannelOptions
    )
    diffusion: DiffusionChannelOptions | None = None
    bond: BondChannelOptions | None = None
    bond_growth: BondGrowthOptions | None = None


@dataclass
class OutputSinks:
    """Output collaborators with one shared lifecycle."""

    reactions: Any
    trajectory: Any
    summary: Any
    checkpoint: Any | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close every owned handle, tolerating repeated finalization."""
        if self._closed:
            return
        close_trajectory = getattr(self.trajectory, "close", None)
        close_reactions = getattr(self.reactions, "close", None)
        try:
            if callable(close_trajectory):
                close_trajectory()
        finally:
            if callable(close_reactions):
                close_reactions()
        self._closed = True


@dataclass(frozen=True)
class SimulationContext:
    """Prepared scientific inputs consumed by the KMC workflow stage."""

    graph: Any
    network: PreparedNetwork
    calculator: Any
    frozen_indices: list[int] | None
    thermo: ThermoRuntime
    channels: ChannelRuntimeOptions
    structure_source: dict[str, Any] | None = None
