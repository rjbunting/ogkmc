"""
autokmc2.io.config
==============
Configuration loader for the :mod:`autokmc2` CLI.

Configs are TOML or YAML documents describing a single KMC run end-to-end:
the slab/nanoparticle, the reactant(s), the calculator (dynamically loaded
by dotted import path), the KMC parameters, and the output settings.

The schema is intentionally flat and dataclass-backed so that tab-completion
in IDEs surfaces every knob.  Missing fields fall back to defaults from
:mod:`autokmc2.core.constants`.

A reference example lives at ``example/co_cu111_emt.yaml``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from autokmc2.core.constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_OUTPUT_DIR,
    REACTIONS_FILENAME,
    SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
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
    MAX_PAIR_SHELLS,
)
from autokmc2.reactions.rates import DEFAULT_TRANSMISSION_COEFFICIENT
from autokmc2.io.calculators import CalculatorCfg


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutputCfg:
    dir: str = DEFAULT_OUTPUT_DIR
    reactions_filename:    str = REACTIONS_FILENAME
    summary_filename:      str = SUMMARY_FILENAME
    trajectory_filename:   str = TRAJECTORY_FILENAME
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
    #: Default ``None`` → ``free_energy.default_symmetry_number`` (1).
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

    All NEB defaults come from :mod:`autokmc2.core.constants` (``NEB_*``).
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

    1. Derives :class:`~autokmc2.sites.bond.BondReactionTemplate`'s from
       every reactant SMILES via
       :func:`~autokmc2.reactions.templates.derive_bond_templates`.
    2. Builds Reactant + adsorbate sites for any "leaf" species
       (fragments / coupling products) referenced by the templates that
       were not in ``reactants``.
    3. Calls :func:`~autokmc2.sites.bond.find_bond_sites` to enumerate
       :class:`BondReactionSite` iso-classes on the live graph.
    4. Bootstraps :func:`~autokmc2.kmc.expansion.initialise_bond_registry`
       so the on-the-fly growth machinery is ready for new species
       introduced by future bond-coupling events.

    The static enumeration result is stored on
    ``G.graph["bond_reaction_sites"]``.  The ``run_kmc_steps`` driver
    does not yet consume bond reactions in the inner loop — they are
    persisted as discoverable metadata only.  Bond-reaction folders go
    under ``reactions/bond/bond_iso{X}_lat{Y}/``.
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
    neb_interpolation:       str   = NEB_INTERPOLATION
    persist_neb_path:        bool  = False


@dataclass
class FreeEnergyCfg:
    """Free-energy / vibrational analysis knobs.

    When ``enabled=True`` (default) the CLI runs ASE
    :class:`~ase.vibrations.Vibrations` for every gas-phase reactant
    (→ :class:`~ase.thermochemistry.IdealGasThermo` at the simulation
    *T* and :attr:`pressure_bar`) and for every successful adsorbate
    relaxation in :func:`autokmc2.sites.stability.adsorption.check_site_stability`
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
    min_frequency_cm        : float = 12.0
    default_symmetry_number : int   = 1
    default_spin            : float = 0.0
    default_geometry        : str   = "auto"
    #: Optional persistent cache directory for ASE ``Vibrations`` JSON
    #: pickle files.  ``None`` (default) → an ephemeral per-call dir
    #: under the OS temp area is used and removed after analysis.
    cache_dir               : str | None = None


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
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _from_dict(d: dict[str, Any]) -> RunConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"top-level config must be a mapping, got {type(d).__name__}")
    return _coerce(RunConfig, d)


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
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "YAML configs require pyyaml — install with `pip install pyyaml` "
                "or `pip install -e .[cli]`"
            ) from exc
        with p.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
    elif ext == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib  # type: ignore[import-not-found]
        else:  # pragma: no cover
            import tomli as tomllib  # type: ignore[import-not-found]
        with p.open("rb") as fp:
            data = tomllib.load(fp)
    else:
        raise ConfigError(f"unknown config extension {ext!r} (.yaml/.yml/.toml)")

    cfg = _from_dict(data or {})
    if cfg.schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema_version={cfg.schema_version!r} does not match "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    return cfg

