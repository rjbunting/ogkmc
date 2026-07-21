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
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from autokmc.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_OUTPUT_DIR,
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,
    RUN_MANIFEST_FILENAME,
    TRAJECTORY_FILENAME,
    CALCULATION_CACHE_DIR,
    ISAAC_EXPORT_FILENAME,
    TRAJ_DUMP_EVERY,
    PRUNE_FMAX,
    PRUNE_MAX_STEPS,
    RANDOM_SEED,
    DIFFUSION_MAX_HOPS,
    DIFFUSION_PRUNE_BY_ADS_PAIR,
    NEB_FMAX,
    NEB_MAX_STEPS,
    NEB_N_IMAGES,
    NEB_CLIMB,
    NEB_SPRING_K,
    NEB_INTERPOLATION,
    N_SHELLS_DEFAULT,
    BOND_MAX_HOPS,
    BOND_PAIR_N_SHELLS,
    BOND_PRUNE_BY_TRIPLE,
    BOND_PRUNE_WITH_CALCULATOR,
    BOND_NEB_INTERPOLATION,
    BOND_ATOM_MATCHING,
    BOND_MATCHING_TRIALS,
    BOND_GAS_LIFT_HEIGHT,
    MAX_PAIR_SHELLS,
)
from autokmc.reactions.rates import DEFAULT_TRANSMISSION_COEFFICIENT
from autokmc.io.calculators import CalculatorCfg


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
    calculation_cache_dir:       str = CALCULATION_CACHE_DIR
    isaac_export_filename:       str = ISAAC_EXPORT_FILENAME
    trajectory_dump_every: int = TRAJ_DUMP_EVERY
    log_level: str = "INFO"


@dataclass
class StructureCfg:
    kind: str = "surface"   # "surface" or "nanoparticle"
    composition: Any = "Cu"
    crystal_structure: str = "fcc"
    miller_index: tuple = (1, 1, 1)
    lattice_constant: Any = None
    min_slab_size: float = 8.0
    min_vacuum_size: float = 12.0
    goal_x: float = 12.0
    goal_y: float = 12.0
    n_freeze_layers: int = 2
    #: Force convergence criterion (eV/Å) for the slab/nanoparticle
    #: LBFGS optimisation.  Default 0.05 eV/Å.
    fmax: float = 0.05
    #: Maximum number of LBFGS steps for the slab/nanoparticle
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
    # ── Thermochemistry (consumed when free_energy.enabled is true) ───────
    #: Partial pressure of the gas-phase reactant in bar.  Multiplies the
    #: adsorption rate so that ΔG / barriers stay at the 1-bar reference.
    #: Default ``None`` → fall back to ``free_energy.pressure_bar``.
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


@dataclass
class KMCCfg:
    temperature_k: float = 500.0
    n_steps: int = 1000
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT
    fmax: float = 0.05
    max_steps: int = 200
    log_every: int = 1
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
    surface_apsp_cutoff:    int  = MAX_PAIR_SHELLS
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
    #: physically viable).  Disabled automatically when no calculator is
    #: configured.
    prune_with_calculator:   bool = BOND_PRUNE_WITH_CALCULATOR
    #: Force convergence threshold for the Stage-1 endpoint relaxation.
    prune_fmax:              float = PRUNE_FMAX
    #: Maximum LBFGS steps for the Stage-1 endpoint relaxation.
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
    *T* and :attr:`pressure_bar`) and for every successful adsorbate
    relaxation in :func:`autokmc.sites.stability.adsorption.check_site_stability`
    (→ :class:`~ase.thermochemistry.HarmonicThermo`, vibrating only the
    reactive species — frozen slab atoms and frozen lateral-shell
    adsorbates contribute nothing).

    Adsorption rates are then multiplied by the reactant's partial
    pressure (in bar) so the persisted ΔG / barriers stay at the 1-bar
    reference.

    When ``enabled=False`` the entire pipeline runs on electronic energy
    only — strict superset of the pre-free-energy schema.
    """
    enabled                 : bool  = True
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
                raise ConfigError(f"{path}.reactants must be a list")
            kwargs[f.name] = [_coerce(ReactantCfg, item,
                                      path=f"{path}.reactants[{i}]")
                              for i, item in enumerate(v)]
        elif f.name == "miller_index":
            kwargs[f.name] = tuple(int(x) for x in v)
        elif f.name == "bond_types":
            kwargs[f.name] = tuple(str(x) for x in v)
        elif f.name == "surface_energy_facets":
            kwargs[f.name] = tuple(tuple(int(i) for i in facet) for facet in v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _from_dict(d: dict[str, Any]) -> RunConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"top-level config must be a mapping, got {type(d).__name__}")
    cfg = _coerce(RunConfig, d)
    _validate_config(cfg)
    return cfg


def _require_bool(value: Any, path: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{path} must be a boolean, got {value!r}")


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise ConfigError(f"{path} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value!r}")


def _require_number(value: Any, path: str, *, minimum: float | None = None,
                    strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path} must be finite, got {value!r}")
    if strictly_positive and number <= 0.0:
        raise ConfigError(f"{path} must be > 0, got {value!r}")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value!r}")


def _validate_config(cfg: RunConfig) -> None:
    """Apply strict type, enum, and physical range validation."""
    from autokmc.species.smiles import canonical_smiles

    _require_bool(cfg.output.calculation_cache_enabled, "output.calculation_cache_enabled")
    _require_int(cfg.output.trajectory_dump_every, "output.trajectory_dump_every", minimum=0)
    if str(cfg.output.log_level).upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ConfigError(f"output.log_level is invalid: {cfg.output.log_level!r}")

    if cfg.structure.kind not in {"surface", "nanoparticle"}:
        raise ConfigError("structure.kind must be 'surface' or 'nanoparticle'")
    if len(cfg.structure.miller_index) != 3:
        raise ConfigError("structure.miller_index must contain exactly three integers")
    for index, value in enumerate(cfg.structure.miller_index):
        _require_int(value, f"structure.miller_index[{index}]")
    for name in ("min_slab_size", "min_vacuum_size", "goal_x", "goal_y", "fmax"):
        _require_number(getattr(cfg.structure, name), f"structure.{name}", strictly_positive=True)
    for name, minimum in (("n_freeze_layers", 0), ("max_steps", 1),
                          ("surface_energy_layers", 1)):
        _require_int(getattr(cfg.structure, name), f"structure.{name}", minimum=minimum)
    if cfg.structure.n_atoms is not None:
        _require_int(cfg.structure.n_atoms, "structure.n_atoms", minimum=1)

    seen_reactants: dict[str, int] = {}
    for index, reactant in enumerate(cfg.reactants):
        prefix = f"reactants[{index}]"
        if not isinstance(reactant.smiles, str) or not reactant.smiles.strip():
            raise ConfigError(f"{prefix}.smiles must be a non-empty string")
        _require_bool(reactant.add_hydrogens, f"{prefix}.add_hydrogens")
        _require_bool(reactant.relax_in_gas, f"{prefix}.relax_in_gas")
        if reactant.partial_pressure_bar is not None:
            _require_number(reactant.partial_pressure_bar, f"{prefix}.partial_pressure_bar", minimum=0.0)
        if reactant.symmetry_number is not None:
            _require_int(reactant.symmetry_number, f"{prefix}.symmetry_number", minimum=1)
        if reactant.spin is not None:
            _require_number(reactant.spin, f"{prefix}.spin", minimum=0.0)
        if reactant.geometry not in {None, "auto", "linear", "nonlinear", "monatomic"}:
            raise ConfigError(f"{prefix}.geometry has unsupported value {reactant.geometry!r}")
        canonical = canonical_smiles(reactant.smiles)
        if canonical in seen_reactants:
            first = seen_reactants[canonical]
            raise ConfigError(
                f"{prefix}.smiles duplicates reactants[{first}].smiles after "
                f"canonicalisation ({canonical!r}); combine their feed settings "
                "into one reactant entry"
            )
        seen_reactants[canonical] = index

    _require_bool(cfg.adsorbate_sites.prune_stable_only, "adsorbate_sites.prune_stable_only")
    _require_number(cfg.adsorbate_sites.fmax, "adsorbate_sites.fmax", strictly_positive=True)
    _require_int(cfg.adsorbate_sites.max_steps, "adsorbate_sites.max_steps", minimum=1)

    _require_number(cfg.kmc.temperature_k, "kmc.temperature_k", strictly_positive=True)
    _require_int(cfg.kmc.n_steps, "kmc.n_steps", minimum=0)
    _require_number(cfg.kmc.transmission_coefficient, "kmc.transmission_coefficient", minimum=0.0)
    _require_number(cfg.kmc.fmax, "kmc.fmax", strictly_positive=True)
    _require_int(cfg.kmc.max_steps, "kmc.max_steps", minimum=1)
    _require_int(cfg.kmc.log_every, "kmc.log_every", minimum=0)
    _require_int(cfg.kmc.random_seed, "kmc.random_seed")
    _require_bool(cfg.kmc.lateral_interactions, "kmc.lateral_interactions")

    for name in (
        "enabled", "prune_by_adsorption_pair", "climb", "persist_neb_path",
    ):
        _require_bool(getattr(cfg.diffusion, name), f"diffusion.{name}")
    for name in (
        "enabled", "include_ring_bonds", "include_homo_coupling",
        "include_dissociation", "include_coupling", "deduplicate_iso",
        "auto_build_leaf_species", "prune_by_triple", "prune_with_calculator",
        "neb_climb", "persist_neb_path",
    ):
        _require_bool(getattr(cfg.bond, name), f"bond.{name}")
    for name, minimum in (("max_hops", 0), ("n_shells_pair", 0),
                          ("max_steps", 1), ("n_images", 1)):
        _require_int(getattr(cfg.diffusion, name), f"diffusion.{name}", minimum=minimum)
    for name in ("fmax", "spring_k"):
        _require_number(getattr(cfg.diffusion, name), f"diffusion.{name}", strictly_positive=True)
    if cfg.diffusion.interpolation not in {"linear", "idpp"}:
        raise ConfigError("diffusion.interpolation must be 'linear' or 'idpp'")

    for name, minimum in (
        ("bond_max_hops", 0), ("surface_apsp_cutoff", 0), ("pair_n_shells", 0),
        ("prune_max_steps", 1), ("neb_max_steps", 1), ("neb_n_images", 1),
        ("matching_trials", 0),
    ):
        _require_int(getattr(cfg.bond, name), f"bond.{name}", minimum=minimum)
    for name in ("gas_lift_height", "prune_fmax", "neb_fmax", "neb_spring_k"):
        _require_number(getattr(cfg.bond, name), f"bond.{name}", strictly_positive=True)
    if cfg.bond.neb_interpolation not in {"linear", "idpp"}:
        raise ConfigError("bond.neb_interpolation must be 'linear' or 'idpp'")
    if cfg.bond.atom_matching not in {"auto", "greedy", "hungarian", "reactant_index"}:
        raise ConfigError(f"bond.atom_matching is unsupported: {cfg.bond.atom_matching!r}")

    _require_bool(cfg.free_energy.enabled, "free_energy.enabled")
    _require_number(cfg.free_energy.pressure_bar, "free_energy.pressure_bar", strictly_positive=True)
    _require_number(cfg.free_energy.vibration_displacement, "free_energy.vibration_displacement", strictly_positive=True)
    _require_int(cfg.free_energy.vibration_nfree, "free_energy.vibration_nfree")
    if cfg.free_energy.vibration_nfree not in {2, 4}:
        raise ConfigError("free_energy.vibration_nfree must be 2 or 4")
    _require_bool(cfg.free_energy.include_ts_vibrations, "free_energy.include_ts_vibrations")
    _require_number(cfg.free_energy.min_frequency_ev, "free_energy.min_frequency_ev", minimum=0.0)
    _require_number(
        cfg.free_energy.symmetry_tolerance,
        "free_energy.symmetry_tolerance",
        strictly_positive=True,
    )
    _require_number(cfg.free_energy.default_spin, "free_energy.default_spin", minimum=0.0)
    if cfg.free_energy.default_geometry not in {"auto", "linear", "nonlinear", "monatomic"}:
        raise ConfigError("free_energy.default_geometry is unsupported")

    _require_bool(cfg.checkpoint.enabled, "checkpoint.enabled")
    _require_int(cfg.checkpoint.every_n_steps, "checkpoint.every_n_steps", minimum=1)
    for name in ("path", "resume_from"):
        value = getattr(cfg.checkpoint, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"checkpoint.{name} must be a non-empty path string")

    _require_int(cfg.calculator.copies, "calculator.copies", minimum=1)
    if cfg.calculator.max_workers is not None:
        _require_int(cfg.calculator.max_workers, "calculator.max_workers", minimum=1)


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
    return cfg
