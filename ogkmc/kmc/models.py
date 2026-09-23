"""Typed inputs, restart state, result, and runtime data for one KMC run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import random
from typing import Any, Callable, Iterable, Mapping, TypeAlias

import networkx as nx
import numpy as np

from ogkmc.core.constants import (
    BOND_ATOM_MATCHING,
    BOND_GAS_LIFT_HEIGHT,
    BOND_GAS_PRECURSOR_DISTANCE,
    BOND_GAS_PRECURSOR_RELAX,
    BOND_MATCHING_TRIALS,
    BOND_MAX_HOPS,
    BOND_NEB_INTERPOLATION,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    BOND_PRUNE_WITH_CALCULATOR,
    BOND_TOLERANCE,
    CO_FACTOR,
    CONTACT_FACTOR,
    DIFFUSION_MAX_HOPS,
    DIFFUSION_PRUNE_BY_ADS_PAIR,
    HULL_TOL,
    KABSCH_MAX_MAPPINGS,
    LATERAL_SHELLS_DEFAULT,
    MAX_PAIR_SHELLS,
    NEB_BAND_EVAL,
    NEB_CLIMB,
    NEB_FMAX,
    NEB_IMAGE_SPACING,
    NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    NEB_INTERMEDIATE_MAX_REFINEMENTS,
    NEB_INTERMEDIATE_MINIMUM_PROMINENCE,
    NEB_INTERMEDIATE_STAGNATION_STEPS,
    NEB_INTERPOLATION,
    NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER,
    NEB_MAX_IMAGES,
    NEB_METHOD,
    NEB_MAX_STEPS,
    NEB_MIN_IMAGES,
    NEB_N_IMAGES,
    NEB_SPRING_K,
    NL_MULT_DEFAULT,
    NN_DISTANCE,
    N_ADSORBATE_RESTARTS,
    N_SHELLS_DEFAULT,
    OPT_FACTOR,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
    RANDOM_SEED,
    REPULSION_WEIGHT,
    SITE_REPULSION_CUTOFF,
    STANDOFF_FACTOR,
)
from ogkmc.kmc.callbacks import (
    CalculatorLike,
    CalculatorPoolLike,
    CheckpointWriterLike,
    ReactionWriterLike,
    SummaryCollectorLike,
    TrajectoryWriterLike,
)
from ogkmc.io.event_log import EventHistory
from ogkmc.kmc.index import _ReactionIndex
from ogkmc.reactions.rates import DEFAULT_TRANSMISSION_COEFFICIENT
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import BondReactionSite
from ogkmc.sites.diffusion import DiffusionSite
from ogkmc.species.reactant import Reactant
from ogkmc.utils.telemetry import RuntimeTelemetry
from ogkmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER

RandomSource = random.Random | np.random.Generator
CalculatorResource: TypeAlias = CalculatorLike | CalculatorPoolLike | None
ReactantInput: TypeAlias = Reactant | Iterable[Reactant] | dict[Any, Any]
KMCEventRecord: TypeAlias = tuple[
    int,
    float,
    str,
    int,
    int,
    int,
    float,
    float,
    float,
]
KMCEventHistory: TypeAlias = list[KMCEventRecord] | EventHistory
PerformanceSnapshot: TypeAlias = dict[str, dict[str, int | float]]


def _options_from_mapping(cls, values: Mapping[str, Any] | None):
    payload = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise TypeError(
            f"unsupported {cls.__name__} option(s): {', '.join(unknown)}"
        )
    return cls(**payload)


@dataclass(frozen=True)
class AdsorptionChannelOptions:
    """Occupied/unoccupied endpoint controls for adsorption rates."""

    fmax: float = 0.05
    max_steps: int = 200

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> AdsorptionChannelOptions:
        return _options_from_mapping(cls, values)

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionChannelOptions:
    """Typed NEB and rate controls for the diffusion channel."""

    fmax: float = NEB_FMAX
    max_steps: int = NEB_MAX_STEPS
    n_images: int = NEB_N_IMAGES
    image_spacing: float | None = NEB_IMAGE_SPACING
    min_images: int = NEB_MIN_IMAGES
    max_images: int = NEB_MAX_IMAGES
    climb: bool = NEB_CLIMB
    spring_k: float = NEB_SPRING_K
    interpolation: str = NEB_INTERPOLATION
    nl_mult: float = NL_MULT_DEFAULT
    persist_neb_path: bool = False
    optimizer: str = DEFAULT_OPTIMIZER
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER
    neb_optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    neb_climb_optimizer: str | None = None
    neb_climb_optimizer_kwargs: dict[str, Any] | None = None
    neb_method: str = NEB_METHOD
    neb_band_eval: str = NEB_BAND_EVAL
    neb_geometry_guard_multiplier: float = (
        NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER
    )
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS
    neb_intermediate_max_refinements: int = NEB_INTERMEDIATE_MAX_REFINEMENTS
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    )
    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> DiffusionChannelOptions:
        return _options_from_mapping(cls, values)

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BondChannelOptions:
    """Typed NEB and atom-matching controls for bond reactions."""

    fmax: float = NEB_FMAX
    max_steps: int = NEB_MAX_STEPS
    n_images: int = NEB_N_IMAGES
    image_spacing: float | None = NEB_IMAGE_SPACING
    min_images: int = NEB_MIN_IMAGES
    max_images: int = NEB_MAX_IMAGES
    climb: bool = NEB_CLIMB
    spring_k: float = NEB_SPRING_K
    interpolation: str = BOND_NEB_INTERPOLATION
    atom_matching: str = BOND_ATOM_MATCHING
    matching_trials: int = BOND_MATCHING_TRIALS
    gas_precursor_relax: bool = BOND_GAS_PRECURSOR_RELAX
    gas_precursor_distance: float = BOND_GAS_PRECURSOR_DISTANCE
    nl_mult: float = NL_MULT_DEFAULT
    persist_neb_path: bool = False
    optimizer: str = DEFAULT_OPTIMIZER
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER
    neb_optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    neb_climb_optimizer: str | None = None
    neb_climb_optimizer_kwargs: dict[str, Any] | None = None
    neb_method: str = NEB_METHOD
    neb_band_eval: str = NEB_BAND_EVAL
    neb_geometry_guard_multiplier: float = (
        NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER
    )
    neb_intermediate_stagnation_steps: int = NEB_INTERMEDIATE_STAGNATION_STEPS
    neb_intermediate_max_refinements: int = NEB_INTERMEDIATE_MAX_REFINEMENTS
    neb_intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE
    neb_intermediate_minimum_prominence: float = (
        NEB_INTERMEDIATE_MINIMUM_PROMINENCE
    )
    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> BondChannelOptions:
        return _options_from_mapping(cls, values)

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BondGrowthOptions:
    """Typed controls for species and channel discovery after bond events."""

    find_diffusion: bool = False
    frozen_indices: list[int] | None = None
    bond_max_hops: int = BOND_MAX_HOPS
    nl_mult: float = NL_MULT_DEFAULT
    random_seed: int = RANDOM_SEED
    adsorption_prune_fmax: float = PRUNE_FMAX
    adsorption_prune_max_steps: int = PRUNE_MAX_STEPS
    bond_prune_fmax: float = PRUNE_FMAX
    bond_prune_max_steps: int = PRUNE_MAX_STEPS
    reactant_fmax: float = 0.05
    reactant_max_steps: int = 500
    anchor_k_max: int | None = None
    adsorbate_bond_tolerance: float = BOND_TOLERANCE
    adsorbate_n_shells_anchor: int | None = None
    adsorbate_n_shells_pair: int = N_SHELLS_DEFAULT
    co_bond_factor: float = CO_FACTOR
    anchor_bond_factor: float = OPT_FACTOR
    anchor_repulsion_weight: float = REPULSION_WEIGHT
    site_repulsion_cutoff: float | None = SITE_REPULSION_CUTOFF
    adsorbate_contact_factor: float = CONTACT_FACTOR
    adsorbate_standoff_factor: float = STANDOFF_FACTOR
    adsorbate_rotational_restarts: int = N_ADSORBATE_RESTARTS
    typical_neighbor_distance: float = NN_DISTANCE
    adsorbate_max_pair_shells: int = MAX_PAIR_SHELLS
    anchor_hull_tolerance: float = HULL_TOL
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE")
    include_ring_bonds: bool = False
    add_hydrogens: bool = True
    include_homo_coupling: bool = True
    include_dissociation: bool = True
    include_coupling: bool = True
    deduplicate_iso: bool = True
    auto_build_leaf_species: bool = True
    diffusion_max_hops: int = DIFFUSION_MAX_HOPS
    diffusion_n_shells_pair: int = N_SHELLS_DEFAULT
    diffusion_prune_by_ads_pair: bool | None = DIFFUSION_PRUNE_BY_ADS_PAIR
    bond_pair_n_shells: int = BOND_PAIR_N_SHELLS
    bond_prune_by_triple: bool = BOND_PRUNE_BY_TRIPLE
    bond_prune_with_calculator: bool = BOND_PRUNE_WITH_CALCULATOR
    gas_lift_height: float = BOND_GAS_LIFT_HEIGHT
    optimizer: str = DEFAULT_OPTIMIZER
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> BondGrowthOptions:
        return _options_from_mapping(cls, values)

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KMCSettings:
    """Scientific and loop settings shared by every KMC stage."""

    temperature: float
    n_steps: int
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT
    frozen_indices: list[int] | None = None
    log_every: int = 100
    progress: bool | None = None
    verbose: bool = True
    lateral_interactions: bool = True
    lateral_shells: int = LATERAL_SHELLS_DEFAULT
    optimizer: str = DEFAULT_OPTIMIZER
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def progress_enabled(self) -> bool:
        """Whether concise run progress should be printed.

        ``None`` preserves the historical direct-API behavior where
        ``verbose`` controlled both detailed diagnostics and progress.
        Configured runs set this explicitly so INFO can stay concise.
        """
        return self.verbose if self.progress is None else self.progress

    @property
    def max_n_shells(self) -> int:
        """Radius whose rates must be refreshed after an occupancy change."""
        return int(self.lateral_shells) if self.lateral_interactions else 0


@dataclass(init=False)
class KMCChannels:
    """Reaction-channel sites plus typed calculation controls.

    ``*_kwargs`` remain accepted only as a construction compatibility seam;
    every live session immediately stores the corresponding typed option
    dataclass.
    """

    adsorption_options: AdsorptionChannelOptions
    diffusion_sites: list[DiffusionSite]
    diffusion_options: DiffusionChannelOptions
    bond_sites: list[BondReactionSite]
    bond_options: BondChannelOptions
    bond_growth_options: BondGrowthOptions
    _diffusion_mapping_view: dict[str, Any]
    _bond_mapping_view: dict[str, Any]
    _bond_growth_mapping_view: dict[str, Any]
    _legacy_diffusion_reserved: dict[str, Any]
    _legacy_bond_reserved: dict[str, Any]
    _legacy_bond_growth_reserved: dict[str, Any]

    def __init__(
        self,
        adsorption_options: AdsorptionChannelOptions | None = None,
        diffusion_sites: Iterable[DiffusionSite] | None = None,
        diffusion_options: DiffusionChannelOptions | None = None,
        bond_sites: Iterable[BondReactionSite] | None = None,
        bond_options: BondChannelOptions | None = None,
        bond_growth_options: BondGrowthOptions | None = None,
        *,
        diffusion_kwargs: Mapping[str, Any] | None = None,
        bond_kwargs: Mapping[str, Any] | None = None,
        bond_growth_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if diffusion_options is not None and diffusion_kwargs is not None:
            raise TypeError("pass diffusion_options or diffusion_kwargs, not both")
        if bond_options is not None and bond_kwargs is not None:
            raise TypeError("pass bond_options or bond_kwargs, not both")
        if bond_growth_options is not None and bond_growth_kwargs is not None:
            raise TypeError("pass bond_growth_options or bond_growth_kwargs, not both")
        reserved = {
            "calculation_cache_root",
            "calculation_cache_lookup_enabled",
            "free_energy_options",
            "vib_cache_root",
        }
        growth_reserved = {
            "calculation_cache_root",
            "calculation_cache_lookup_enabled",
            "free_energy_options",
            "free_energy_temperature_k",
            "verbose",
            "vib_cache_root",
        }
        diffusion_values = dict(diffusion_kwargs or {})
        bond_values = dict(bond_kwargs or {})
        growth_values = dict(bond_growth_kwargs or {})
        self._legacy_diffusion_reserved = {
            key: diffusion_values.pop(key)
            for key in tuple(diffusion_values)
            if key in reserved
        }
        self._legacy_bond_reserved = {
            key: bond_values.pop(key)
            for key in tuple(bond_values)
            if key in reserved
        }
        self._legacy_bond_growth_reserved = {
            key: growth_values.pop(key)
            for key in tuple(growth_values)
            if key in growth_reserved
        }
        self.adsorption_options = adsorption_options or AdsorptionChannelOptions()
        self.diffusion_sites = list(diffusion_sites or [])
        self.diffusion_options = diffusion_options or DiffusionChannelOptions.from_mapping(
            diffusion_values
        )
        self.bond_sites = list(bond_sites or [])
        self.bond_options = bond_options or BondChannelOptions.from_mapping(bond_values)
        self.bond_growth_options = bond_growth_options or BondGrowthOptions.from_mapping(
            growth_values
        )
        self._diffusion_mapping_view = (
            self.diffusion_options.to_kwargs()
            if diffusion_options is not None
            else diffusion_values
        )
        self._bond_mapping_view = (
            self.bond_options.to_kwargs() if bond_options is not None else bond_values
        )
        self._bond_growth_mapping_view = (
            self.bond_growth_options.to_kwargs()
            if bond_growth_options is not None
            else growth_values
        )

    @property
    def diffusion_kwargs(self) -> dict[str, Any]:
        """Legacy mapping view; internal callers use ``diffusion_options``."""
        return dict(self._diffusion_mapping_view)

    @property
    def bond_kwargs(self) -> dict[str, Any]:
        """Legacy mapping view; internal callers use ``bond_options``."""
        return dict(self._bond_mapping_view)

    @property
    def bond_growth_kwargs(self) -> dict[str, Any]:
        """Legacy mapping view; internal callers use ``bond_growth_options``."""
        return dict(self._bond_growth_mapping_view)


@dataclass(frozen=True)
class KMCThermochemistry:
    """Free-energy and calculation-cache collaborators."""

    free_energy_options: Any = None
    vib_cache_root: str | None = None
    calculation_cache_root: str | None = None
    calculation_cache_lookup_enabled: bool | None = None
    # Compatibility override used only by legacy dynamic bond-growth calls.
    # Typed configured runs leave it unset and use ``KMCSettings.temperature``.
    free_energy_temperature_k: float | None = None


@dataclass(frozen=True)
class KMCObservers:
    """Optional typed persistence callbacks attached to a simulation."""

    reaction_writer: ReactionWriterLike | None = None
    trajectory_writer: TrajectoryWriterLike | None = None
    summary_collector: SummaryCollectorLike | None = None
    checkpoint_writer: CheckpointWriterLike | None = None


@dataclass(frozen=True)
class KMCResumeState:
    """Canonical state restored from a checkpoint before the next event."""

    step: int = 0
    time_s: float = 0.0
    history: KMCEventHistory = field(default_factory=list)
    reaction_counts: Mapping[str, int] = field(default_factory=dict)
    rng_state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class KMCFunctions:
    """Injectable scientific kernels used by :class:`KMCSession`."""

    compute_adsorption: Callable[..., Any]
    recompute_affected: Callable[..., Any]
    expand_bond_network: Callable[..., Any]


@dataclass
class KMCRuntime:
    """Mutable state advanced by the event loop."""

    rng: RandomSource
    gas_energies: dict[str, float]
    gas_free_energies: dict[str, float]
    partial_pressures: dict[str, float]
    reaction_index: _ReactionIndex
    history: KMCEventHistory
    reaction_counts: dict[str, int]
    current_time_s: float
    start_step: int
    steps_executed: int = 0


@dataclass(frozen=True)
class KMCSystem:
    """Graph, site registry, calculator, and feed species for one run."""

    graph: nx.Graph
    adsorbate_sites: list[AdsorbateSite]
    calculator: CalculatorResource
    reactants: ReactantInput

    def __post_init__(self) -> None:
        # Gas energies, Gibbs energies, pressures, and checkpoints all read
        # this feed. Retain a one-shot iterable before any consumer exhausts
        # it, so every consumer sees the same species and reservoir settings.
        if isinstance(self.reactants, Iterable) and not isinstance(
            self.reactants, (Reactant, dict, list, tuple)
        ):
            object.__setattr__(self, "reactants", list(self.reactants))


@dataclass(frozen=True)
class KMCRunRequest:
    """Complete typed invocation for one KMC session."""

    system: KMCSystem
    settings: KMCSettings
    channels: KMCChannels = field(default_factory=KMCChannels)
    thermochemistry: KMCThermochemistry = field(default_factory=KMCThermochemistry)
    observers: KMCObservers = field(default_factory=KMCObservers)
    resume: KMCResumeState = field(default_factory=KMCResumeState)
    rng: RandomSource | int | None = None
    telemetry: RuntimeTelemetry | None = field(default=None, compare=False)


@dataclass(frozen=True)
class KMCRunResult:
    """Typed scientific result and operational performance snapshot."""

    time_s: float
    steps_executed: int
    history: KMCEventHistory
    reaction_counts: dict[str, int]
    final_occupancy: dict[str, int]
    termination_status: str = "complete"
    termination_reason: str = "requested_steps_completed"
    wall_time_s: float | None = None
    performance: PerformanceSnapshot = field(default_factory=dict)

    @property
    def simulated_time_s(self) -> float:
        """Return the physical KMC time, distinct from elapsed wall time."""
        return float(self.time_s)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the historic mapping produced by :func:`run_kmc_steps`."""
        return {
            "time": self.time_s,
            "simulated_time_s": self.simulated_time_s,
            "wall_time_s": self.wall_time_s,
            "steps_executed": self.steps_executed,
            "history": self.history,
            "reaction_counts": self.reaction_counts,
            "final_occupancy": self.final_occupancy,
            "termination_status": self.termination_status,
            "termination_reason": self.termination_reason,
            "performance": self.performance,
        }


__all__ = [
    "AdsorptionChannelOptions",
    "BondChannelOptions",
    "BondGrowthOptions",
    "CalculatorResource",
    "DiffusionChannelOptions",
    "KMCChannels",
    "KMCEventRecord",
    "KMCEventHistory",
    "KMCFunctions",
    "KMCObservers",
    "KMCResumeState",
    "KMCRunRequest",
    "KMCRunResult",
    "KMCRuntime",
    "KMCSettings",
    "KMCSystem",
    "KMCThermochemistry",
    "PerformanceSnapshot",
    "RandomSource",
    "ReactantInput",
]
