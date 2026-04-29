"""
autokmc.config
==============
Configuration loader for the :mod:`autokmc` CLI.

Configs are TOML or YAML documents describing a single KMC run end-to-end:
the slab/nanoparticle, the reactant(s), the calculator (dynamically loaded
by dotted import path), the KMC parameters, and the output settings.

The schema is intentionally flat and dataclass-backed so that tab-completion
in IDEs surfaces every knob.  Missing fields fall back to defaults from
:mod:`autokmc.constants`.

A reference example lives at ``example/co_cu111_emt.yaml``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from autokmc.constants import (
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
)
from autokmc.kmc_adsorption import DEFAULT_TRANSMISSION_COEFFICIENT


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
    # Nanoparticle-only knobs (used when kind == "nanoparticle"):
    n_atoms: int | None = None
    surface_energies: dict | None = None
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class ReactantCfg:
    smiles: str
    add_hydrogens: bool = True
    relax_in_gas:  bool = True


@dataclass
class CalculatorCfg:
    """Pluggable ASE calculator description.

    Two equivalent forms are supported:

    * ``import_path`` + ``kwargs`` — dotted path to the calculator class
      (e.g. ``"ase.calculators.emt.EMT"`` or
      ``"ase.calculators.vasp.Vasp"``).
    * ``factory`` + ``factory_kwargs`` — dotted path to a *callable* (no
      ``self``) that returns a calculator instance.  Useful for ML
      potentials whose constructor takes loaded weights, e.g.
      ``NequIPCalculator.from_compiled_model``.

    ``factory`` takes precedence when both are supplied.
    """
    import_path:     str | None = None
    kwargs:          dict       = field(default_factory=dict)
    factory:         str | None = None
    factory_kwargs:  dict       = field(default_factory=dict)


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

    All NEB defaults come from :mod:`autokmc.constants` (``NEB_*``).
    """
    enabled:                   bool   = True
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
class RunConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    output:           OutputCfg          = field(default_factory=OutputCfg)
    structure:        StructureCfg       = field(default_factory=StructureCfg)
    reactants:        list[ReactantCfg]  = field(default_factory=list)
    calculator:       CalculatorCfg      = field(default_factory=CalculatorCfg)
    adsorbate_sites:  AdsorbateSitesCfg  = field(default_factory=AdsorbateSitesCfg)
    kmc:              KMCCfg             = field(default_factory=KMCCfg)
    diffusion:        DiffusionCfg       = field(default_factory=DiffusionCfg)


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


# ---------------------------------------------------------------------------
# Calculator dynamic loader
# ---------------------------------------------------------------------------

def _resolve(dotted: str):
    """Resolve a dotted path like ``"pkg.mod.Class.method"`` to the live object.

    Works for any depth: module-only, module.Class, module.Class.classmethod,
    etc.  Tries progressively shorter module prefixes until ``importlib`` is
    happy, then chains ``getattr`` for the remaining parts.

    Examples
    --------
    ``"ase.calculators.emt.EMT"``
        → imports ``ase.calculators.emt``, returns ``EMT`` class.
    ``"nequip.ase.NequIPCalculator.from_compiled_model"``
        → imports ``nequip.ase``, returns
        ``NequIPCalculator.from_compiled_model`` bound method.
    """
    # Explicit "module:attr" escape hatch.
    if ":" in dotted:
        mod_name, rest = dotted.split(":", 1)
        mod = importlib.import_module(mod_name)
        obj = mod
        for part in rest.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError as exc:
                raise ConfigError(
                    f"cannot resolve {dotted!r}: {mod_name!r} has no {part!r}"
                ) from exc
        return obj

    parts = dotted.split(".")
    if len(parts) < 2:
        raise ConfigError(
            f"calculator path {dotted!r} must be dotted (e.g. pkg.mod.Class)"
        )

    # Walk from the longest possible module prefix down to the shortest,
    # stopping at the first successful import.
    last_err: Exception | None = None
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        attr_chain = parts[split:]
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            last_err = exc
            continue
        # Module imported — now walk any remaining attribute chain.
        obj = mod
        for attr in attr_chain:
            try:
                obj = getattr(obj, attr)
            except AttributeError as exc:
                raise ConfigError(
                    f"cannot resolve {dotted!r}: {obj!r} has no attribute {attr!r}"
                ) from exc
        return obj

    raise ConfigError(
        f"cannot import any module prefix of {dotted!r}. "
        f"Last import error: {last_err}"
    )


def build_calculator(cfg: CalculatorCfg):
    """Instantiate the ASE-compatible calculator described by *cfg*.

    Supports any calculator that follows the ASE calculator protocol:
    ``ase.calculators.emt.EMT``, ``ase.calculators.vasp.Vasp``,
    ``ase.calculators.cp2k.CP2K``, ``nequip.ase.NequIPCalculator``,
    ``mace.calculators.MACECalculator``, etc.

    Returns ``None`` when neither *import_path* nor *factory* is set —
    callers may then default to ASE EMT or skip calculator-dependent
    pipeline stages.
    """
    if cfg.factory:
        fn = _resolve(cfg.factory)
        return fn(**(cfg.factory_kwargs or {}))
    if cfg.import_path:
        cls = _resolve(cfg.import_path)
        return cls(**(cfg.kwargs or {}))
    return None


def calculator_meta(cfg: CalculatorCfg) -> dict[str, Any]:
    """Echo the calculator description into a JSON-friendly metadata dict."""
    return {
        "import_path":    cfg.import_path,
        "kwargs":         dict(cfg.kwargs or {}),
        "factory":        cfg.factory,
        "factory_kwargs": dict(cfg.factory_kwargs or {}),
    }

