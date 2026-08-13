"""Section-oriented validation for the public run configuration schema."""

from __future__ import annotations

import math
from pathlib import Path, PureWindowsPath
from typing import Any, NoReturn

from autokmc.core.constants import NEB_BAND_EVALS, NEB_METHODS
from autokmc.utils.optimizers import (
    NEB_OPTIMIZERS,
    REGULAR_OPTIMIZERS,
    normalize_optimizer_kwargs,
)


class _Validator:
    """Validate one resolved config while preserving public error messages."""

    def __init__(self, error_type: type[ValueError]) -> None:
        self.error_type = error_type

    def fail(self, message: str) -> NoReturn:
        raise self.error_type(message)

    def boolean(self, value: Any, path: str) -> None:
        if type(value) is not bool:
            self.fail(f"{path} must be a boolean, got {value!r}")

    def integer(
        self,
        value: Any,
        path: str,
        *,
        minimum: int | None = None,
    ) -> None:
        if type(value) is not int:
            self.fail(f"{path} must be an integer, got {value!r}")
        if minimum is not None and value < minimum:
            self.fail(f"{path} must be >= {minimum}, got {value!r}")

    def number(
        self,
        value: Any,
        path: str,
        *,
        minimum: float | None = None,
        strictly_positive: bool = False,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(f"{path} must be numeric, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            self.fail(f"{path} must be finite, got {value!r}")
        if strictly_positive and number <= 0.0:
            self.fail(f"{path} must be > 0, got {value!r}")
        if minimum is not None and number < minimum:
            self.fail(f"{path} must be >= {minimum}, got {value!r}")

    def string(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.fail(f"{path} must be a non-empty string, got {value!r}")
        if "\x00" in value:
            self.fail(f"{path} must not contain a null byte")
        return value

    def path_string(
        self,
        value: Any,
        path: str,
        *,
        relative: bool = False,
    ) -> Path:
        text = self.string(value, path)
        candidate = Path(text)
        if relative and (
            candidate.is_absolute() or PureWindowsPath(text).is_absolute()
        ):
            self.fail(f"{path} must be relative to output.dir, got {value!r}")
        if relative and (candidate == Path(".") or ".." in candidate.parts):
            self.fail(
                f"{path} must name a path inside output.dir without '..', "
                f"got {value!r}"
            )
        return candidate

    def calculator_value(self, value: Any, path: str) -> None:
        """Validate nested values consumed by the generic calculator loader."""
        if isinstance(value, dict):
            for key in value:
                if not isinstance(key, str) or not key:
                    self.fail(
                        f"{path} keys must be non-empty strings, got {key!r}"
                    )

            is_nested_spec = "factory" in value or "import_path" in value
            if is_nested_spec:
                factory = value.get("factory")
                import_path = value.get("import_path")
                if factory is not None:
                    self.string(factory, f"{path}.factory")
                if import_path is not None:
                    self.string(import_path, f"{path}.import_path")
                if factory and import_path:
                    self.fail(
                        f"{path}.factory and {path}.import_path are mutually "
                        "exclusive"
                    )
                if not factory and not import_path:
                    self.fail(
                        f"{path} must define a non-empty factory or import_path"
                    )
                args = value.get("args", ())
                if not isinstance(args, (list, tuple)):
                    self.fail(f"{path}.args must be a list or tuple")
                kwargs = value.get("kwargs", {})
                factory_kwargs = value.get("factory_kwargs", {})
                if not isinstance(kwargs, dict):
                    self.fail(f"{path}.kwargs must be a mapping")
                if not isinstance(factory_kwargs, dict):
                    self.fail(f"{path}.factory_kwargs must be a mapping")
                if factory and kwargs:
                    self.fail(f"{path}.kwargs is unused for a factory spec")
                if import_path and factory_kwargs:
                    self.fail(
                        f"{path}.factory_kwargs is unused for an import spec"
                    )
            for key, item in value.items():
                self.calculator_value(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.calculator_value(item, f"{path}[{index}]")

    def output(self, cfg: Any) -> None:
        self.path_string(cfg.dir, "output.dir")
        self.boolean(
            cfg.calculation_cache_enabled,
            "output.calculation_cache_enabled",
        )
        self.boolean(
            cfg.calculation_cache_lookup_enabled,
            "output.calculation_cache_lookup_enabled",
        )
        self.boolean(
            cfg.isaac_export_enabled,
            "output.isaac_export_enabled",
        )
        self.integer(
            cfg.trajectory_dump_every,
            "output.trajectory_dump_every",
            minimum=0,
        )
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if str(cfg.log_level).upper() not in valid_levels:
            self.fail(f"output.log_level is invalid: {cfg.log_level!r}")
        managed_names = {
            name: self.path_string(
                getattr(cfg, name),
                f"output.{name}",
                relative=True,
            )
            for name in (
                "reactions_filename",
                "summary_filename",
                "run_manifest_filename",
                "trajectory_filename",
                "calculation_cache_dir",
                "isaac_export_filename",
            )
        }
        for name in (
            "reactions_filename",
            "summary_filename",
            "run_manifest_filename",
            "trajectory_filename",
            "isaac_export_filename",
        ):
            raw = getattr(cfg, name)
            if raw.endswith(("/", "\\")):
                self.fail(f"output.{name} must name a file, got {raw!r}")
        reserved = {
            Path(".autokmc-run.lock"),
            Path("analysis"),
            Path("checkpoint.pkl"),
            Path("diagnostics"),
            Path("reactions"),
            Path("uncommitted_reactions"),
            Path("vib_cache"),
        }
        for name, path in managed_names.items():
            if path in reserved:
                self.fail(
                    f"output.{name} conflicts with AutoKMC's managed path "
                    f"{str(path)!r}"
                )
        items = list(managed_names.items())
        for left_index, (left_name, left_path) in enumerate(items):
            for right_name, right_path in items[left_index + 1:]:
                if (
                    left_path == right_path
                    or left_path in right_path.parents
                    or right_path in left_path.parents
                ):
                    self.fail(
                        f"output.{left_name} and output.{right_name} must "
                        "refer to distinct, non-overlapping paths"
                    )

    def constants(self, cfg: Any) -> None:
        """Validate shared scientific and algorithmic controls."""
        for name in (
            "neighbor_list_multiplier",
            "co_bond_factor",
            "anchor_bond_factor",
            "adsorbate_contact_factor",
            "typical_neighbor_distance",
        ):
            self.number(
                getattr(cfg, name),
                f"constants.{name}",
                strictly_positive=True,
            )
        for name in (
            "anchor_repulsion_weight",
            "adsorbate_standoff_factor",
            "adsorbate_bond_tolerance",
        ):
            self.number(
                getattr(cfg, name),
                f"constants.{name}",
                minimum=0.0,
            )
        if cfg.site_repulsion_cutoff is not None:
            self.number(
                cfg.site_repulsion_cutoff,
                "constants.site_repulsion_cutoff",
                strictly_positive=True,
            )
        self.number(
            cfg.anchor_hull_tolerance,
            "constants.anchor_hull_tolerance",
        )
        self.number(
            cfg.raycast_coverage_threshold,
            "constants.raycast_coverage_threshold",
            minimum=0.0,
        )
        if cfg.raycast_coverage_threshold > 1.0:
            self.fail(
                "constants.raycast_coverage_threshold must be <= 1, got "
                f"{cfg.raycast_coverage_threshold!r}"
            )
        for name, minimum in (
            ("adsorbate_rotational_restarts", 1),
            ("raycast_disc_samples", 1),
            ("kabsch_max_mappings", 1),
            ("lateral_shells", 0),
        ):
            self.integer(
                getattr(cfg, name),
                f"constants.{name}",
                minimum=minimum,
            )

    def optimization(self, cfg: Any) -> None:
        optimizer = self.string(cfg.optimizer, "optimization.optimizer").lower()
        if optimizer not in REGULAR_OPTIMIZERS:
            choices = ", ".join(sorted(REGULAR_OPTIMIZERS))
            self.fail(
                "optimization.optimizer must be one of "
                f"{choices}; got {cfg.optimizer!r}"
            )
        try:
            normalize_optimizer_kwargs(
                optimizer,
                cfg.optimizer_kwargs,
                allowed=REGULAR_OPTIMIZERS,
                setting="optimization.optimizer_kwargs",
            )
        except ValueError as exc:
            self.fail(str(exc))
        neb_optimizer = self.string(
            cfg.neb_optimizer,
            "optimization.neb_optimizer",
        ).lower()
        if neb_optimizer not in NEB_OPTIMIZERS:
            choices = ", ".join(sorted(NEB_OPTIMIZERS))
            self.fail(
                "optimization.neb_optimizer must be one of "
                f"{choices}; got {cfg.neb_optimizer!r}"
            )
        try:
            normalize_optimizer_kwargs(
                neb_optimizer,
                cfg.neb_optimizer_kwargs,
                allowed=NEB_OPTIMIZERS,
                setting="optimization.neb_optimizer_kwargs",
            )
        except ValueError as exc:
            self.fail(str(exc))
        neb_climb_optimizer = (
            neb_optimizer
            if cfg.neb_climb_optimizer is None
            else self.string(
                cfg.neb_climb_optimizer,
                "optimization.neb_climb_optimizer",
            ).lower()
        )
        if neb_climb_optimizer not in NEB_OPTIMIZERS:
            choices = ", ".join(sorted(NEB_OPTIMIZERS))
            self.fail(
                "optimization.neb_climb_optimizer must be one of "
                f"{choices}; got {cfg.neb_climb_optimizer!r}"
            )
        climb_kwargs = (
            cfg.neb_optimizer_kwargs
            if cfg.neb_climb_optimizer_kwargs is None
            else cfg.neb_climb_optimizer_kwargs
        )
        try:
            normalize_optimizer_kwargs(
                neb_climb_optimizer,
                climb_kwargs,
                allowed=NEB_OPTIMIZERS,
                setting="optimization.neb_climb_optimizer_kwargs",
            )
        except ValueError as exc:
            self.fail(str(exc))
        neb_method = self.string(
            cfg.neb_method,
            "optimization.neb_method",
        ).lower()
        if neb_method not in NEB_METHODS:
            choices = ", ".join(sorted(NEB_METHODS))
            self.fail(
                "optimization.neb_method must be one of "
                f"{choices}; got {cfg.neb_method!r}"
            )
        neb_band_eval = self.string(
            cfg.neb_band_eval,
            "optimization.neb_band_eval",
        ).lower()
        if neb_band_eval not in NEB_BAND_EVALS:
            choices = ", ".join(sorted(NEB_BAND_EVALS))
            self.fail(
                "optimization.neb_band_eval must be one of "
                f"{choices}; got {cfg.neb_band_eval!r}"
            )
        self.number(
            cfg.neb_geometry_guard_multiplier,
            "optimization.neb_geometry_guard_multiplier",
            strictly_positive=True,
        )

    def structure(self, cfg: Any) -> None:
        if cfg.kind not in {"surface", "nanoparticle", "file"}:
            self.fail(
                "structure.kind must be 'surface', 'nanoparticle', or 'file'"
            )
        self.integer(cfg.index, "structure.index")
        if cfg.kind == "file":
            self.path_string(cfg.path, "structure.path")
            if cfg.format is not None:
                self.string(cfg.format, "structure.format")
            if cfg.frozen_indices is not None:
                if not isinstance(cfg.frozen_indices, list):
                    self.fail(
                        "structure.frozen_indices must be a list or null"
                    )
                seen_frozen_indices: set[int] = set()
                for index, atom_index in enumerate(cfg.frozen_indices):
                    self.integer(
                        atom_index,
                        f"structure.frozen_indices[{index}]",
                        minimum=0,
                    )
                    if atom_index in seen_frozen_indices:
                        self.fail(
                            "structure.frozen_indices entries must be unique; "
                            f"found duplicate {atom_index}"
                        )
                    seen_frozen_indices.add(atom_index)
        else:
            for name in ("path", "format", "frozen_indices"):
                if getattr(cfg, name) is not None:
                    self.fail(
                        f"structure.{name} is only valid when "
                        "structure.kind='file'"
                    )
            if cfg.index != -1:
                self.fail(
                    "structure.index is only valid when structure.kind='file'"
                )
        if len(cfg.miller_index) != 3:
            self.fail(
                "structure.miller_index must contain exactly three integers"
            )
        for index, value in enumerate(cfg.miller_index):
            self.integer(value, f"structure.miller_index[{index}]")
        for name in (
            "min_slab_size",
            "min_vacuum_size",
            "goal_x",
            "goal_y",
            "fmax",
            "surface_radius_factor",
        ):
            self.number(
                getattr(cfg, name),
                f"structure.{name}",
                strictly_positive=True,
            )
        self.number(
            cfg.nanoparticle_hull_tolerance_factor,
            "structure.nanoparticle_hull_tolerance_factor",
            minimum=0.0,
        )
        if cfg.surface_side not in {"top", "bottom", "both"}:
            self.fail(
                "structure.surface_side must be 'top', 'bottom', or 'both'"
            )
        for name, minimum in (
            ("n_freeze_layers", 0),
            ("max_steps", 1),
            ("surface_energy_layers", 1),
        ):
            self.integer(
                getattr(cfg, name),
                f"structure.{name}",
                minimum=minimum,
            )
        if cfg.n_atoms is not None:
            self.integer(cfg.n_atoms, "structure.n_atoms", minimum=1)
        self.number(
            cfg.surface_energy_vacuum,
            "structure.surface_energy_vacuum",
            strictly_positive=True,
        )
        if cfg.surface_energy_fmax is not None:
            self.number(
                cfg.surface_energy_fmax,
                "structure.surface_energy_fmax",
                strictly_positive=True,
            )
        if cfg.surface_energy_max_steps is not None:
            self.integer(
                cfg.surface_energy_max_steps,
                "structure.surface_energy_max_steps",
                minimum=1,
            )
        if not isinstance(cfg.extra_kwargs, dict):
            self.fail(
                f"structure.extra_kwargs must be a mapping, got "
                f"{type(cfg.extra_kwargs).__name__}"
            )
        if cfg.surface_energies is not None and not isinstance(
            cfg.surface_energies,
            dict,
        ):
            self.fail("structure.surface_energies must be a mapping or null")
        if not isinstance(cfg.surface_energy_facets, (list, tuple)):
            self.fail("structure.surface_energy_facets must be a list or tuple")
        for facet_index, facet in enumerate(cfg.surface_energy_facets):
            if not isinstance(facet, (list, tuple)) or len(facet) != 3:
                self.fail(
                    f"structure.surface_energy_facets[{facet_index}] must "
                    "contain exactly three integers"
                )
            for value_index, value in enumerate(facet):
                self.integer(
                    value,
                    "structure.surface_energy_facets"
                    f"[{facet_index}][{value_index}]",
                )

    def reactants(self, reactants: list[Any]) -> None:
        if not reactants:
            self.fail("reactants must contain at least one species")

        from autokmc.utils.rdkit_logging import silence_rdkit_warnings

        silence_rdkit_warnings()
        from rdkit import Chem

        seen: dict[str, int] = {}
        for index, reactant in enumerate(reactants):
            prefix = f"reactants[{index}]"
            if not isinstance(reactant.smiles, str) or not reactant.smiles.strip():
                self.fail(f"{prefix}.smiles must be a non-empty string")
            self.boolean(reactant.add_hydrogens, f"{prefix}.add_hydrogens")
            self.boolean(reactant.relax_in_gas, f"{prefix}.relax_in_gas")
            if reactant.partial_pressure_bar is not None:
                self.number(
                    reactant.partial_pressure_bar,
                    f"{prefix}.partial_pressure_bar",
                    minimum=0.0,
                )
            if reactant.symmetry_number is not None:
                self.integer(
                    reactant.symmetry_number,
                    f"{prefix}.symmetry_number",
                    minimum=1,
                )
            if reactant.spin is not None:
                self.number(reactant.spin, f"{prefix}.spin", minimum=0.0)
            valid_geometry = {
                None,
                "auto",
                "linear",
                "nonlinear",
                "monatomic",
            }
            if reactant.geometry not in valid_geometry:
                self.fail(
                    f"{prefix}.geometry has unsupported value "
                    f"{reactant.geometry!r}"
                )
            try:
                molecule = Chem.MolFromSmiles(reactant.smiles)
            except Exception as exc:
                self.fail(
                    f"{prefix}.smiles could not be parsed by RDKit: "
                    f"{reactant.smiles!r} ({exc})"
                )
            if molecule is None:
                self.fail(
                    f"{prefix}.smiles could not be parsed by RDKit: "
                    f"{reactant.smiles!r}"
                )
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            if canonical in seen:
                first = seen[canonical]
                self.fail(
                    f"{prefix}.smiles duplicates reactants[{first}].smiles "
                    f"after canonicalisation ({canonical!r}); combine their "
                    "feed settings into one reactant entry"
                )
            seen[canonical] = index

    def adsorbate_sites(self, cfg: Any) -> None:
        self.boolean(
            cfg.prune_stable_only,
            "adsorbate_sites.prune_stable_only",
        )
        self.number(
            cfg.fmax,
            "adsorbate_sites.fmax",
            strictly_positive=True,
        )
        self.integer(
            cfg.max_steps,
            "adsorbate_sites.max_steps",
            minimum=1,
        )
        if cfg.anchor_k_max is not None:
            self.integer(
                cfg.anchor_k_max,
                "adsorbate_sites.anchor_k_max",
                minimum=1,
            )
        if cfg.n_shells_anchor is not None:
            self.integer(
                cfg.n_shells_anchor,
                "adsorbate_sites.n_shells_anchor",
                minimum=0,
            )
        self.integer(
            cfg.pair_n_shells,
            "adsorbate_sites.pair_n_shells",
            minimum=0,
        )
        self.integer(
            cfg.max_pair_shells,
            "adsorbate_sites.max_pair_shells",
            minimum=0,
        )

    def kmc(self, cfg: Any) -> None:
        self.number(
            cfg.temperature_k,
            "kmc.temperature_k",
            strictly_positive=True,
        )
        self.integer(cfg.n_steps, "kmc.n_steps", minimum=0)
        self.number(
            cfg.transmission_coefficient,
            "kmc.transmission_coefficient",
            minimum=0.0,
        )
        self.number(cfg.fmax, "kmc.fmax", strictly_positive=True)
        self.integer(cfg.max_steps, "kmc.max_steps", minimum=1)
        self.integer(cfg.log_every, "kmc.log_every", minimum=0)
        self.integer(cfg.random_seed, "kmc.random_seed", minimum=0)
        self.boolean(cfg.lateral_interactions, "kmc.lateral_interactions")

    def diffusion(self, cfg: Any) -> None:
        for name in (
            "enabled",
            "prune_by_adsorption_pair",
            "climb",
            "persist_neb_path",
        ):
            self.boolean(getattr(cfg, name), f"diffusion.{name}")
        for name, minimum in (
            ("max_hops", 0),
            ("n_shells_pair", 0),
            ("max_steps", 1),
            ("n_images", 1),
            ("min_images", 1),
            ("max_images", 1),
        ):
            self.integer(
                getattr(cfg, name),
                f"diffusion.{name}",
                minimum=minimum,
            )
        for name in ("fmax", "spring_k"):
            self.number(
                getattr(cfg, name),
                f"diffusion.{name}",
                strictly_positive=True,
            )
        if cfg.image_spacing is not None:
            self.number(
                cfg.image_spacing,
                "diffusion.image_spacing",
                strictly_positive=True,
            )
        if cfg.max_images < cfg.min_images:
            self.fail("diffusion.max_images must be >= diffusion.min_images")
        if cfg.interpolation not in {"linear", "idpp"}:
            self.fail("diffusion.interpolation must be 'linear' or 'idpp'")

    def bond(self, cfg: Any) -> None:
        for name in (
            "enabled",
            "include_ring_bonds",
            "include_homo_coupling",
            "include_dissociation",
            "include_coupling",
            "deduplicate_iso",
            "gas_precursor_relax",
            "auto_build_leaf_species",
            "prune_by_triple",
            "prune_with_calculator",
            "neb_climb",
            "persist_neb_path",
        ):
            self.boolean(getattr(cfg, name), f"bond.{name}")
        for name, minimum in (
            ("bond_max_hops", 0),
            ("pair_n_shells", 0),
            ("prune_max_steps", 1),
            ("neb_max_steps", 1),
            ("neb_n_images", 1),
            ("neb_min_images", 1),
            ("neb_max_images", 1),
            ("matching_trials", 0),
        ):
            self.integer(
                getattr(cfg, name),
                f"bond.{name}",
                minimum=minimum,
            )
        for name in (
            "gas_lift_height",
            "gas_precursor_distance",
            "prune_fmax",
            "neb_fmax",
            "neb_spring_k",
        ):
            self.number(
                getattr(cfg, name),
                f"bond.{name}",
                strictly_positive=True,
            )
        if cfg.neb_image_spacing is not None:
            self.number(
                cfg.neb_image_spacing,
                "bond.neb_image_spacing",
                strictly_positive=True,
            )
        if cfg.neb_max_images < cfg.neb_min_images:
            self.fail("bond.neb_max_images must be >= bond.neb_min_images")
        if cfg.neb_interpolation not in {"linear", "idpp"}:
            self.fail("bond.neb_interpolation must be 'linear' or 'idpp'")
        valid_matching = {"auto", "greedy", "hungarian", "reactant_index"}
        if cfg.atom_matching not in valid_matching:
            self.fail(
                f"bond.atom_matching is unsupported: {cfg.atom_matching!r}"
            )
        if not isinstance(cfg.bond_types, (list, tuple)):
            self.fail("bond.bond_types must be a list or tuple")
        valid_bond_types = {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
        seen_bond_types: set[str] = set()
        for index, bond_type in enumerate(cfg.bond_types):
            if not isinstance(bond_type, str) or bond_type not in valid_bond_types:
                self.fail(
                    f"bond.bond_types[{index}] must be one of "
                    f"{sorted(valid_bond_types)}, got {bond_type!r}"
                )
            if bond_type in seen_bond_types:
                self.fail(f"bond.bond_types contains duplicate {bond_type!r}")
            seen_bond_types.add(bond_type)

    def free_energy(self, cfg: Any) -> None:
        self.boolean(cfg.enabled, "free_energy.enabled")
        self.number(cfg.pressure_bar, "free_energy.pressure_bar", minimum=0.0)
        self.number(
            cfg.vibration_displacement,
            "free_energy.vibration_displacement",
            strictly_positive=True,
        )
        self.integer(cfg.vibration_nfree, "free_energy.vibration_nfree")
        if cfg.vibration_nfree not in {2, 4}:
            self.fail("free_energy.vibration_nfree must be 2 or 4")
        self.boolean(
            cfg.include_ts_vibrations,
            "free_energy.include_ts_vibrations",
        )
        self.number(
            cfg.min_frequency_ev,
            "free_energy.min_frequency_ev",
            minimum=0.0,
        )
        self.number(
            cfg.symmetry_tolerance,
            "free_energy.symmetry_tolerance",
            strictly_positive=True,
        )
        self.number(
            cfg.default_spin,
            "free_energy.default_spin",
            minimum=0.0,
        )
        valid_geometry = {"auto", "linear", "nonlinear", "monatomic"}
        if cfg.default_geometry not in valid_geometry:
            self.fail("free_energy.default_geometry is unsupported")
        if cfg.cache_dir is not None:
            self.path_string(cfg.cache_dir, "free_energy.cache_dir")

    def checkpoint(self, cfg: Any) -> None:
        self.boolean(cfg.enabled, "checkpoint.enabled")
        self.integer(
            cfg.every_n_steps,
            "checkpoint.every_n_steps",
            minimum=1,
        )
        for name in ("path", "resume_from"):
            value = getattr(cfg, name)
            if value is not None:
                self.path_string(value, f"checkpoint.{name}")

    def calculator(self, cfg: Any) -> None:
        self.integer(cfg.copies, "calculator.copies", minimum=1)
        if cfg.max_workers is not None:
            self.integer(
                cfg.max_workers,
                "calculator.max_workers",
                minimum=1,
            )
            if cfg.max_workers > cfg.copies:
                self.fail(
                    "calculator.max_workers must be <= calculator.copies "
                    f"({cfg.max_workers} > {cfg.copies})"
                )

        import_path = cfg.import_path
        factory = cfg.factory
        if import_path is not None:
            self.string(import_path, "calculator.import_path")
        if factory is not None:
            self.string(factory, "calculator.factory")
        if import_path and factory:
            self.fail(
                "calculator.import_path and calculator.factory are mutually "
                "exclusive; configure exactly one"
            )
        if not import_path and not factory:
            self.fail(
                "calculator must configure exactly one of "
                "calculator.import_path or calculator.factory"
            )
        if not isinstance(cfg.kwargs, dict):
            self.fail(
                f"calculator.kwargs must be a mapping, got "
                f"{type(cfg.kwargs).__name__}"
            )
        if not isinstance(cfg.factory_kwargs, dict):
            self.fail(
                f"calculator.factory_kwargs must be a mapping, got "
                f"{type(cfg.factory_kwargs).__name__}"
            )
        self.calculator_value(cfg.kwargs, "calculator.kwargs")
        self.calculator_value(
            cfg.factory_kwargs,
            "calculator.factory_kwargs",
        )
        if factory and cfg.kwargs:
            self.fail("calculator.kwargs is unused when calculator.factory is set")
        if import_path and cfg.factory_kwargs:
            self.fail(
                "calculator.factory_kwargs is unused when "
                "calculator.import_path is set"
            )
        gpu_device_arg = self.string(
            cfg.gpu_device_arg,
            "calculator.gpu_device_arg",
        )
        if any(not part for part in gpu_device_arg.split(".")):
            self.fail(
                "calculator.gpu_device_arg must be a dotted path without "
                f"empty components, got {gpu_device_arg!r}"
            )
        devices = cfg.gpu_devices
        if devices is not None and not isinstance(devices, (list, tuple)):
            self.fail("calculator.gpu_devices must be a list or tuple of strings")
        if devices is not None:
            normalized_devices: list[str] = []
            for index, device in enumerate(devices):
                normalized_devices.append(
                    self.string(device, f"calculator.gpu_devices[{index}]")
                )
            if len(set(normalized_devices)) != len(normalized_devices):
                self.fail("calculator.gpu_devices entries must be unique")
            if normalized_devices and len(normalized_devices) != cfg.copies:
                self.fail(
                    "calculator.gpu_devices must contain exactly "
                    "calculator.copies entries "
                    f"({len(normalized_devices)} != {cfg.copies})"
                )

    def run(self, cfg: Any) -> None:
        self.output(cfg.output)
        self.constants(cfg.constants)
        self.optimization(cfg.optimization)
        self.structure(cfg.structure)
        self.reactants(cfg.reactants)
        self.adsorbate_sites(cfg.adsorbate_sites)
        self.kmc(cfg.kmc)
        self.diffusion(cfg.diffusion)
        self.bond(cfg.bond)
        self.free_energy(cfg.free_energy)
        self.checkpoint(cfg.checkpoint)
        self.calculator(cfg.calculator)
        if cfg.checkpoint.enabled and cfg.checkpoint.path:
            checkpoint_path = Path(cfg.checkpoint.path).resolve()
            output_dir = Path(cfg.output.dir)
            managed = {
                name: (output_dir / getattr(cfg.output, name)).resolve()
                for name in (
                    "reactions_filename",
                    "summary_filename",
                    "run_manifest_filename",
                    "trajectory_filename",
                    "calculation_cache_dir",
                    "isaac_export_filename",
                )
            }
            for name, path in managed.items():
                if checkpoint_path == path:
                    self.fail(
                        "checkpoint.path must not overwrite "
                        f"output.{name}: {checkpoint_path}"
                    )


def validate_config(
    cfg: Any,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Validate every configuration section using the requested error type."""
    _Validator(error_type).run(cfg)


__all__ = ["validate_config"]
