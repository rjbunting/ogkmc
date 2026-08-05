"""
autokmc.io.config
==============
Configuration loader for the :mod:`autokmc` CLI.

Configs are TOML or YAML documents describing a single KMC run end-to-end:
the slab/nanoparticle, the reactant(s), the calculator (dynamically loaded
by dotted import path), the KMC parameters, and the output settings.

The schema is intentionally flat and dataclass-backed so that tab-completion
in IDEs surfaces every knob.  Missing fields fall back to defaults from
:mod:`autokmc.core.constants`.

A reference example lives at ``example/co_oxidation_pt111_uma_4gpu.yaml``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from autokmc.core.constants import (
    BOND_ATOM_MATCHING,
    BOND_GAS_LIFT_HEIGHT,
    BOND_MATCHING_TRIALS,
    BOND_MAX_HOPS,
    BOND_NEB_INTERPOLATION,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    BOND_PRUNE_WITH_CALCULATOR,
    BOND_TOLERANCE,
    CALCULATION_CACHE_DIR,
    CO_FACTOR,
    CONFIG_SCHEMA_VERSION,
    CONTACT_FACTOR,
    DEFAULT_OUTPUT_DIR,
    DIFFUSION_MAX_HOPS,
    DIFFUSION_PRUNE_BY_ADS_PAIR,
    HULL_TOL,
    ISAAC_EXPORT_FILENAME,
    KABSCH_MAX_MAPPINGS,
    LATERAL_SHELLS_DEFAULT,
    MAX_PAIR_SHELLS,
    NEB_BAND_EVAL,
    NEB_FMAX,
    NEB_CLIMB,
    NEB_INTERPOLATION,
    NEB_MAX_STEPS,
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
    RAYCAST_COVERAGE_THRESHOLD,
    RAYCAST_N_DISC_SAMPLE,
    REACTIONS_FILENAME,
    REPULSION_WEIGHT,
    RUN_MANIFEST_FILENAME,
    SITE_REPULSION_CUTOFF,
    STANDOFF_FACTOR,
    SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
    TRAJ_DUMP_EVERY,
)
from autokmc.io.calculators import CalculatorCfg
from autokmc.reactions.rates import DEFAULT_TRANSMISSION_COEFFICIENT
from autokmc.utils.optimizers import DEFAULT_NEB_OPTIMIZER, DEFAULT_OPTIMIZER


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutputCfg:
    dir: str = DEFAULT_OUTPUT_DIR
    reactions_filename:    str = REACTIONS_FILENAME
    summary_filename:      str = SUMMARY_FILENAME
    run_manifest_filename: str = RUN_MANIFEST_FILENAME
    trajectory_filename:   str = TRAJECTORY_FILENAME
    calculation_cache_enabled:   bool = True
    calculation_cache_lookup_enabled: bool = False
    calculation_cache_dir:       str = CALCULATION_CACHE_DIR
    isaac_export_enabled:        bool = False
    isaac_export_filename:       str = ISAAC_EXPORT_FILENAME
    trajectory_dump_every: int = TRAJ_DUMP_EVERY
    log_level: str = "INFO"


@dataclass
class ConstantsCfg:
    """Shared scientific and algorithmic controls.

    These defaults live in :mod:`autokmc.core.constants`; exposing them here
    makes each run's effective values explicit, validated, and reproducible.
    Channel-specific convergence and sampling controls remain in their
    corresponding configuration sections.
    """

    neighbor_list_multiplier: float = NL_MULT_DEFAULT
    co_bond_factor: float = CO_FACTOR
    anchor_bond_factor: float = OPT_FACTOR
    anchor_repulsion_weight: float = REPULSION_WEIGHT
    site_repulsion_cutoff: float | None = SITE_REPULSION_CUTOFF
    adsorbate_contact_factor: float = CONTACT_FACTOR
    adsorbate_standoff_factor: float = STANDOFF_FACTOR
    adsorbate_rotational_restarts: int = N_ADSORBATE_RESTARTS
    typical_neighbor_distance: float = NN_DISTANCE
    adsorbate_bond_tolerance: float = BOND_TOLERANCE
    anchor_hull_tolerance: float = HULL_TOL
    raycast_coverage_threshold: float = RAYCAST_COVERAGE_THRESHOLD
    raycast_disc_samples: int = RAYCAST_N_DISC_SAMPLE
    kabsch_max_mappings: int = KABSCH_MAX_MAPPINGS
    lateral_shells: int = LATERAL_SHELLS_DEFAULT


@dataclass
class OptimizationCfg:
    """ASE optimizer choices used by calculator-backed relaxations."""

    optimizer: str = DEFAULT_OPTIMIZER
    neb_optimizer: str = DEFAULT_NEB_OPTIMIZER
    #: NEB band evaluation mode: ``"images"`` (per-image calculator calls)
    #: or ``"batched"`` (whole band in one stacked model forward per
    #: optimizer step, when the calculator supports it).
    neb_band_eval: str = NEB_BAND_EVAL


@dataclass
class StructureCfg:
    kind: str = "surface"   # "surface", "nanoparticle", or "file"
    # File-backed structure knobs (used when kind == "file"):
    path: str | None = None
    format: str | None = None
    index: int = -1
    frozen_indices: list[int] | None = None
    composition: Any = "Cu"
    crystal_structure: str = "fcc"
    miller_index: tuple = (1, 1, 1)
    lattice_constant: Any = None
    min_slab_size: float = 8.0
    min_vacuum_size: float = 12.0
    goal_x: float = 12.0
    goal_y: float = 12.0
    n_freeze_layers: int = 2
    #: Which slab face ray casting should classify: top, bottom, or both.
    surface_side: str = "top"
    #: Scale applied to covalent-radius discs during slab ray casting.
    surface_radius_factor: float = 1.0
    #: Scale applied to covalent radii when classifying nanoparticle hull atoms.
    nanoparticle_hull_tolerance_factor: float = 0.5
    #: Force convergence criterion (eV/Å) for the slab/nanoparticle
    #: calculator-backed optimisation.  Default 0.05 eV/Å.
    fmax: float = 0.05
    #: Maximum number of optimizer steps for the slab/nanoparticle
    #: optimisation.  Default 1000.
    max_steps: int = 1000
    # Nanoparticle-only knobs (used when kind == "nanoparticle"):
    n_atoms: int | None = None
    surface_energies: dict | None = None
    surface_energy_facets: tuple = ((1, 1, 1), (1, 0, 0), (1, 1, 0))
    surface_energy_layers: int = 6
    surface_energy_vacuum: float = 10.0
    surface_energy_fmax: float | None = None
    surface_energy_max_steps: int | None = None
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class ReactantCfg:
    smiles: str
    add_hydrogens: bool = True
    relax_in_gas:  bool = True
    # ── Gas feed and thermochemistry ──────────────────────────────
    #: Partial pressure of the gas-phase reactant in bar.  Multiplies the
    #: adsorption rate so that ΔG / barriers stay at the 1-bar reference.
    #: This applies whether or not free-energy corrections are enabled.
    #: Default ``None`` → use ``free_energy.pressure_bar`` as the feed-wide
    #: fallback.
    partial_pressure_bar: float | None = None
    #: Symmetry number σ for IdealGasThermo (e.g. 2 for H₂, 12 for CH₄).
    #: Default ``None`` infers it from the final gas-phase geometry with
    #: pymatgen.  Retained as an override for unusual or distorted structures.
    symmetry_number: int | None        = None
    #: Spin S (number of unpaired electrons / 2).  Default ``None`` →
    #: ``free_energy.default_spin`` (0).
    spin:            float | None      = None
    #: ``"linear"`` / ``"nonlinear"`` / ``"monatomic"`` / ``None`` (auto).
    geometry:        str | None        = None


@dataclass
class AdsorbateSitesCfg:
    prune_stable_only: bool = True
    fmax:              float = PRUNE_FMAX
    max_steps:         int   = PRUNE_MAX_STEPS
    #: Hard cap on surface anchor-clique size.  Four covers atop, bridge,
    #: three-fold, and four-fold coordination while preventing combinatorial
    #: growth on unusually dense graphs.  Explicit ``null`` restores legacy
    #: unbounded enumeration.
    anchor_k_max:      int | None = 4
    #: Local graph depth for anchor-site isomorphism.  ``None`` selects it
    #: automatically from the molecular reach.
    n_shells_anchor:   int | None = None
    #: Local graph depth for multi-anchor placement isomorphism.
    pair_n_shells:     int = N_SHELLS_DEFAULT
    #: Maximum surface-graph path length retained for multi-anchor placements.
    max_pair_shells:   int = MAX_PAIR_SHELLS


@dataclass
class KMCCfg:
    temperature_k: float = 500.0
    n_steps: int = 1000
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT
    fmax: float = 0.05
    max_steps: int = 200
    log_every: int = 100
    random_seed: int = RANDOM_SEED
    lateral_interactions: bool = True


@dataclass
class DiffusionCfg:
    """Diffusion (NEB) channel knobs.

    When ``enabled=False`` (default) the KMC loop runs adsorption /
    desorption only — exactly as before this channel was added — so
    existing configs remain backwards-compatible.

    All NEB defaults come from :mod:`autokmc.core.constants` (``NEB_*``).
    """
    enabled:                   bool   = False
    max_hops:                  int    = DIFFUSION_MAX_HOPS
    n_shells_pair:             int    = N_SHELLS_DEFAULT
    prune_by_adsorption_pair:  bool   = DIFFUSION_PRUNE_BY_ADS_PAIR
    fmax:             float  = NEB_FMAX
    max_steps:        int    = NEB_MAX_STEPS
    n_images:         int    = NEB_N_IMAGES
    climb:            bool   = NEB_CLIMB
    spring_k:         float  = NEB_SPRING_K
    interpolation:    str    = NEB_INTERPOLATION
    persist_neb_path: bool   = False


@dataclass
class BondCfg:
    """Bond-changing reaction (A + B ⇌ C) channel knobs.

    When ``enabled=False`` (default) bond reactions are skipped entirely
    — backwards-compatible with configs written before this channel was
    wired up.

    When enabled, the CLI:

    1. Derives :class:`~autokmc.sites.bond.BondReactionTemplate`'s from
       every reactant SMILES via
       :func:`~autokmc.reactions.templates.derive_bond_templates`.
    2. Builds Reactant + adsorbate sites for any "leaf" species
       (fragments / coupling products) referenced by the templates that
       were not in ``reactants``.
    3. Calls :func:`~autokmc.sites.bond.find_bond_sites` to enumerate
       :class:`BondReactionSite` iso-classes on the live graph.
    4. Bootstraps :func:`~autokmc.kmc.expansion.initialise_bond_registry`
       so the on-the-fly growth machinery is ready for new species
       introduced by future bond-coupling events.

    The static enumeration result is stored on
    ``G.graph["bond_reaction_sites"]`` and passed into ``run_kmc_steps``.
    The KMC loop consumes both coupling and dissociation events, and can
    expand new adsorbate, diffusion, and bond-reaction series on the fly
    when a bond event produces a new surface species.  Bond-reaction folders
    go under ``reactions/bond/<A+B~C>/bond_iso{X}_lat{Y}/``.
    """
    enabled:                bool = False
    bond_max_hops:          int  = BOND_MAX_HOPS
    bond_types:             tuple = ("SINGLE", "DOUBLE", "TRIPLE")
    include_ring_bonds:     bool = False
    include_homo_coupling:  bool = True
    include_dissociation:   bool = True
    include_coupling:       bool = True
    deduplicate_iso:        bool = True
    gas_lift_height:        float = BOND_GAS_LIFT_HEIGHT
    # When True, every leaf species (fragment / coupling product) implied
    # by the templates that is *not* already in ``reactants`` is built and
    # has its adsorbate sites enumerated automatically.  When False, the
    # CLI raises if any template references a species without sites.
    auto_build_leaf_species: bool = True
    # ── Pruning ────────────────────────────────────────────────────────────
    #: BFS depth for the triple ego-graph used by Stage-2 iso-class pruning.
    pair_n_shells:           int  = BOND_PAIR_N_SHELLS
    #: Stage 2 — keep only the smallest-ego BondReactionSite per
    #: (frozenset({iso_a, iso_b}), iso_c) adsorption triple.  Mirrors
    #: ``diffusion.prune_by_adsorption_pair``.
    prune_by_triple:         bool = BOND_PRUNE_BY_TRIPLE
    #: Stage 1 — drop iso-classes whose A+B endpoint is bond-changing-
    #: unstable under a calculator relaxation (i.e. the reaction is not
    #: physically viable.
    prune_with_calculator:   bool = BOND_PRUNE_WITH_CALCULATOR
    #: Force convergence threshold for the Stage-1 endpoint relaxation.
    prune_fmax:              float = PRUNE_FMAX
    #: Maximum optimizer steps for the Stage-1 endpoint relaxation.
    prune_max_steps:         int  = PRUNE_MAX_STEPS
    # ── NEB knobs (consumed by ``check_bond_site_stability`` via the KMC loop)
    neb_fmax:                float = NEB_FMAX
    neb_max_steps:           int   = NEB_MAX_STEPS
    neb_n_images:            int   = NEB_N_IMAGES
    neb_climb:               bool  = NEB_CLIMB
    neb_spring_k:            float = NEB_SPRING_K
    neb_interpolation:       str   = BOND_NEB_INTERPOLATION
    atom_matching:           str   = BOND_ATOM_MATCHING
    matching_trials:         int   = BOND_MATCHING_TRIALS
    persist_neb_path:        bool  = False


@dataclass
class FreeEnergyCfg:
    """Free-energy / vibrational analysis knobs.

    When ``enabled=True`` (default) the CLI runs ASE
    :class:`~ase.vibrations.Vibrations` for every gas-phase reactant
    (→ :class:`~ase.thermochemistry.IdealGasThermo` at the simulation
    *T* and a fixed 1-bar standard state) and for every successful adsorbate
    relaxation in :func:`autokmc.sites.stability.adsorption.check_site_stability`
    (→ :class:`~ase.thermochemistry.HarmonicThermo`, vibrating only the
    reactive species — frozen slab atoms and frozen lateral-shell
    adsorbates contribute nothing).

    :attr:`pressure_bar` is the default reactant partial pressure.  A
    reactant-specific ``partial_pressure_bar`` overrides it.  Adsorption rates
    are multiplied by that resolved partial pressure (in bar) so the persisted
    ΔG / barriers stay at the fixed 1-bar standard-state reference.

    When ``enabled=False`` the entire pipeline runs on electronic energy
    only — strict superset of the pre-free-energy schema.
    """
    enabled                 : bool  = True
    #: Feed-wide partial-pressure fallback for reactants that do not declare
    #: ``partial_pressure_bar``.  This is not the thermodynamic standard-state
    #: pressure, which is fixed at 1 bar.
    pressure_bar            : float = 1.0
    vibration_displacement  : float = 0.01
    vibration_nfree         : int   = 2
    include_ts_vibrations   : bool  = True
    min_frequency_ev        : float = 0.0015
    #: Cartesian tolerance (Å) used by pymatgen's molecular point-group
    #: analyzer when a reactant does not provide ``symmetry_number``.
    symmetry_tolerance      : float = 0.3
    default_spin            : float = 0.0
    default_geometry        : str   = "auto"
    #: Optional persistent cache directory for ASE ``Vibrations`` JSON
    #: pickle files.  ``None`` (default) → an ephemeral per-call dir
    #: under the OS temp area is used and removed after analysis.
    cache_dir               : str | None = None


@dataclass
class CheckpointCfg:
    enabled: bool = False
    path: str | None = None
    every_n_steps: int = 1
    resume_from: str | None = None


@dataclass
class RunConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    output:           OutputCfg          = field(default_factory=OutputCfg)
    constants:        ConstantsCfg       = field(default_factory=ConstantsCfg)
    optimization:     OptimizationCfg    = field(default_factory=OptimizationCfg)
    structure:        StructureCfg       = field(default_factory=StructureCfg)
    reactants:        list[ReactantCfg]  = field(default_factory=list)
    calculator:       CalculatorCfg      = field(default_factory=CalculatorCfg)
    adsorbate_sites:  AdsorbateSitesCfg  = field(default_factory=AdsorbateSitesCfg)
    kmc:              KMCCfg             = field(default_factory=KMCCfg)
    diffusion:        DiffusionCfg       = field(default_factory=DiffusionCfg)
    bond:             BondCfg            = field(default_factory=BondCfg)
    free_energy:      FreeEnergyCfg      = field(default_factory=FreeEnergyCfg)
    checkpoint:       CheckpointCfg      = field(default_factory=CheckpointCfg)


# ---------------------------------------------------------------------------
# Dict ↔ dataclass coercion
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised on malformed config inputs."""


def _coerce(cls, value: Any, *, path: str = ""):
    """Recursively coerce a (nested) dict into the dataclass *cls*."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ConfigError(
            f"{path or cls.__name__}: expected mapping, got {type(value).__name__}"
        )

    # Explicit nested-dataclass map (avoids PEP-563 string-type headaches under
    # `from __future__ import annotations`).  Keyed by class ``__name__`` so
    # static analysers don't mis-flag dataclass-class keys as unhashable.
    nested_map: dict[str, dict[str, type]] = {
        "RunConfig": {
            "output":          OutputCfg,
            "constants":       ConstantsCfg,
            "optimization":    OptimizationCfg,
            "structure":       StructureCfg,
            "calculator":      CalculatorCfg,
            "adsorbate_sites": AdsorbateSitesCfg,
            "kmc":             KMCCfg,
            "diffusion":       DiffusionCfg,
            "bond":            BondCfg,
            "free_energy":     FreeEnergyCfg,
            "checkpoint":      CheckpointCfg,
        },
    }
    nested_for_cls = nested_map.get(cls.__name__, {})

    valid_names = {f.name for f in fields(cls)}
    unknown = set(value) - valid_names
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown keys {sorted(unknown)}; "
            f"valid keys: {sorted(valid_names)}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in value:
            continue
        v = value[f.name]
        nested_cls = nested_for_cls.get(f.name)
        if nested_cls is not None:
            kwargs[f.name] = _coerce(
                nested_cls, v, path=f"{path}.{f.name}".lstrip("."),
            )
        elif f.name == "reactants":
            if not isinstance(v, list):
                prefix = f"{path}." if path else ""
                raise ConfigError(f"{prefix}reactants must be a list")
            prefix = f"{path}." if path else ""
            kwargs[f.name] = [_coerce(ReactantCfg, item,
                                      path=f"{prefix}reactants[{i}]")
                              for i, item in enumerate(v)]
        elif f.name == "miller_index":
            if not isinstance(v, (list, tuple)):
                raise ConfigError("structure.miller_index must be a list or tuple")
            kwargs[f.name] = tuple(v)
        elif f.name == "bond_types":
            if not isinstance(v, (list, tuple)):
                raise ConfigError("bond.bond_types must be a list or tuple")
            kwargs[f.name] = tuple(v)
        elif f.name == "surface_energy_facets":
            if not isinstance(v, (list, tuple)):
                raise ConfigError(
                    "structure.surface_energy_facets must be a list or tuple"
                )
            try:
                kwargs[f.name] = tuple(tuple(facet) for facet in v)
            except TypeError as exc:
                raise ConfigError(
                    "structure.surface_energy_facets entries must be lists or tuples"
                ) from exc
        elif f.name == "frozen_indices":
            if v is not None and not isinstance(v, (list, tuple)):
                raise ConfigError(
                    "structure.frozen_indices must be a list, tuple, or null"
                )
            kwargs[f.name] = None if v is None else list(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _from_dict(d: dict[str, Any]) -> RunConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"top-level config must be a mapping, got {type(d).__name__}")
    cfg = _coerce(RunConfig, d)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: RunConfig) -> None:
    """Apply strict validation while retaining the historical helper."""
    from autokmc.io.config_validation import validate_config

    validate_config(cfg, error_type=ConfigError)

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> RunConfig:
    """Load a YAML or TOML config file into a :class:`RunConfig`.

    Format dispatch is by file extension:

    * ``.yaml`` / ``.yml`` → ``pyyaml`` (install via ``[cli]`` extra).
    * ``.toml``            → stdlib ``tomllib`` (Python ≥ 3.11) or ``tomli``.

    Raises
    ------
    ConfigError
        On unknown keys, wrong types, or unknown extensions.
    ImportError
        When the YAML/TOML reader is missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    ext = p.suffix.lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "YAML configs require pyyaml — install with `pip install pyyaml` "
                "or `pip install -e .[cli]`"
            ) from exc
        with p.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
    elif ext == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib
            with p.open("rb") as fp:
                data = tomllib.load(fp)
        else:  # pragma: no cover
            import tomli
            with p.open("rb") as fp:
                data = tomli.load(fp)
    else:
        raise ConfigError(f"unknown config extension {ext!r} (.yaml/.yml/.toml)")

    cfg = _from_dict(data or {})
    if cfg.schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema_version={cfg.schema_version!r} does not match "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if cfg.structure.path is not None:
        structure_path = Path(cfg.structure.path).expanduser()
        if not structure_path.is_absolute():
            structure_path = p.resolve().parent / structure_path
        cfg.structure.path = str(structure_path.resolve())
    return cfg
