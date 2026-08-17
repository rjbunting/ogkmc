"""Behavior-preserving preparation stages for configured runs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from autokmc.core.graph_state import set_frozen_indices, set_run_id
from autokmc.io.calculators import (
    CalculatorConfigError,
    acquire_calculator,
    build_calculator,
    primary_calculator,
)
from autokmc.workflow.models import (
    PreparedCalculator,
    PreparedStructure,
    PreparedSystem,
    RunIdentity,
)


def count_z_layers(atoms, *, tolerance: float = 0.35) -> int:
    """Count distinct Cartesian z layers for progress reporting."""
    z_values = sorted(float(z) for z in atoms.get_positions()[:, 2])
    if not z_values:
        return 0
    count = 1
    last = z_values[0]
    for value in z_values[1:]:
        if abs(value - last) > tolerance:
            count += 1
            last = value
    return count


def summarize_sites(sites: list) -> tuple[int, int]:
    """Return iso-class and concrete-member counts for progress reporting."""
    return len(sites), sum(
        len(getattr(site, "member_node_ids", ())) for site in sites
    )


def configured_adsorbate_site_kwargs(cfg) -> dict:
    """Resolve every configured adsorbate-site discovery control."""
    constants = cfg.constants
    settings = cfg.adsorbate_sites
    return {
        "bond_tolerance": constants.adsorbate_bond_tolerance,
        "n_shells_anchor": settings.n_shells_anchor,
        "n_shells_pair": settings.pair_n_shells,
        "anchor_k_max": settings.anchor_k_max,
        "co_factor": constants.co_bond_factor,
        "opt_factor": constants.anchor_bond_factor,
        "repulsion_weight": constants.anchor_repulsion_weight,
        "repulsion_cutoff": constants.site_repulsion_cutoff,
        "contact_factor": constants.adsorbate_contact_factor,
        "standoff_factor": constants.adsorbate_standoff_factor,
        "n_restarts": constants.adsorbate_rotational_restarts,
        "nn_distance": constants.typical_neighbor_distance,
        "max_pair_shells": settings.max_pair_shells,
        "hull_tolerance": constants.anchor_hull_tolerance,
        "kabsch_max_mappings": constants.kabsch_max_mappings,
        "nl_mult": constants.neighbor_list_multiplier,
        "prune_stable_only": settings.prune_stable_only,
        "prune_fmax": settings.fmax,
        "prune_max_steps": settings.max_steps,
        "optimizer": cfg.optimization.optimizer,
        "optimizer_kwargs": cfg.optimization.optimizer_kwargs,
    }


def prepare_calculator(cfg, *, verbose: bool = False) -> PreparedCalculator:
    """Build the explicitly configured calculator pool."""
    resource = build_calculator(cfg.calculator)
    calculator = primary_calculator(resource)
    if calculator is None:
        raise CalculatorConfigError(
            "calculator configuration did not construct a calculator"
        )
    elif verbose:
        print(f"[autokmc]   calculator ready: {type(calculator).__name__}")
    return PreparedCalculator(resource=resource, primary=calculator)


def prepare_structure(
    cfg,
    identity: RunIdentity,
    calculators: PreparedCalculator,
    *,
    config_path: str | None = None,
    verbose: bool = False,
) -> PreparedStructure:
    """Build the catalyst atoms or restore structure metadata."""
    structure_cfg = cfg.structure
    resume_state = identity.resume_state
    if resume_state is not None:
        return PreparedStructure(
            atoms=None,
            frozen_indices=resume_state.frozen_indices,
        )

    if structure_cfg.kind == "file":
        from autokmc.structure.loading import load_structure_file

        atoms, structure_source = load_structure_file(
            structure_cfg.path,
            format=structure_cfg.format,
            index=structure_cfg.index,
            frozen_indices=structure_cfg.frozen_indices,
            config_path=config_path,
        )
        frozen_indices = list(atoms.info["frozen_indices"]) or None
        if verbose:
            cell = atoms.get_cell()
            print(
                f"[autokmc]   structure loaded successfully: {len(atoms)} atoms, "
                f"formula={structure_source['chemical_formula']}, "
                f"cell=({np.linalg.norm(cell[0]):.2f}, "
                f"{np.linalg.norm(cell[1]):.2f}, "
                f"{np.linalg.norm(cell[2]):.2f}) Å"
            )
            if frozen_indices:
                print(
                    f"[autokmc]   frozen region: {len(frozen_indices)} "
                    "atom(s) fixed"
                )
            else:
                print("[autokmc]   frozen region: no fixed atoms")
        return PreparedStructure(
            atoms=atoms,
            frozen_indices=frozen_indices,
            structure_source=structure_source,
        )

    from autokmc.structure import build_nanoparticle, build_surface

    structure_kwargs = dict(structure_cfg.extra_kwargs or {})
    # ``kmc.random_seed`` is the single run-wide seed.  Keep ``extra_kwargs``
    # as an expert override seam for callers that intentionally need a
    # structure-only composition seed.
    structure_kwargs.setdefault("composition_seed", cfg.kmc.random_seed)

    with acquire_calculator(
        calculators.resource,
        purpose="structure construction",
    ) as calculator:
        if structure_cfg.kind == "surface":
            atoms = build_surface(
                composition=structure_cfg.composition,
                crystal_structure=structure_cfg.crystal_structure,
                miller_index=tuple(structure_cfg.miller_index),
                lattice_constant=structure_cfg.lattice_constant,
                min_slab_size=structure_cfg.min_slab_size,
                min_vacuum_size=structure_cfg.min_vacuum_size,
                goal_x=structure_cfg.goal_x,
                goal_y=structure_cfg.goal_y,
                n_freeze_layers=structure_cfg.n_freeze_layers,
                surface_radius_factor=structure_cfg.surface_radius_factor,
                raycast_coverage_threshold=(
                    cfg.constants.raycast_coverage_threshold
                ),
                raycast_disc_samples=cfg.constants.raycast_disc_samples,
                fmax=structure_cfg.fmax,
                max_steps=structure_cfg.max_steps,
                optimizer=cfg.optimization.optimizer,
                optimizer_kwargs=cfg.optimization.optimizer_kwargs,
                calculator=calculator,
                verbose=verbose,
                **structure_kwargs,
            )
        elif structure_cfg.kind == "nanoparticle":
            atoms = build_nanoparticle(
                composition=structure_cfg.composition,
                crystal_structure=structure_cfg.crystal_structure,
                lattice_constant=structure_cfg.lattice_constant,
                target_atoms=(
                    int(structure_cfg.n_atoms)
                    if structure_cfg.n_atoms
                    else 600
                ),
                surface_energies=structure_cfg.surface_energies,
                fmax=structure_cfg.fmax,
                max_steps=structure_cfg.max_steps,
                surface_energy_facets=structure_cfg.surface_energy_facets,
                surface_energy_layers=structure_cfg.surface_energy_layers,
                surface_energy_vacuum=structure_cfg.surface_energy_vacuum,
                surface_energy_fmax=structure_cfg.surface_energy_fmax,
                surface_energy_max_steps=structure_cfg.surface_energy_max_steps,
                optimizer=cfg.optimization.optimizer,
                optimizer_kwargs=cfg.optimization.optimizer_kwargs,
                calculator=calculator,
                verbose=verbose,
                **structure_kwargs,
            )
        else:
            raise ValueError(
                f"unknown structure.kind={structure_cfg.kind!r} "
                "(expected surface|nanoparticle)"
            )
        atoms.calc = None

    frozen_indices = list(atoms.info.get("frozen_indices", []) or []) or None
    if verbose:
        cell = atoms.get_cell()
        print(
            f"[autokmc]   structure built successfully: {len(atoms)} atoms, "
            f"{count_z_layers(atoms)} z-layer(s), "
            f"cell=({np.linalg.norm(cell[0]):.2f}, "
            f"{np.linalg.norm(cell[1]):.2f}, "
            f"{np.linalg.norm(cell[2]):.2f}) Å"
        )
        if frozen_indices:
            print(
                "[autokmc]   frozen region: requested bottom "
                f"{structure_cfg.n_freeze_layers} layer(s), "
                f"{len(frozen_indices)} atom(s) fixed"
            )
        else:
            print("[autokmc]   frozen region: no fixed atoms")
    return PreparedStructure(
        atoms=atoms,
        frozen_indices=frozen_indices,
    )


def prepare_material_graph(
    cfg,
    identity: RunIdentity,
    structure: PreparedStructure,
    *,
    verbose: bool = False,
) -> PreparedSystem:
    """Classify surface atoms and build, or restore, the material graph."""
    from autokmc.core.graph import build_graph
    from autokmc.structure import align_periodic_slab_frame, find_surface_atoms
    from autokmc.structure.surface import _pbc_connectivity_axes

    atoms = structure.atoms
    if identity.resume_state is not None:
        graph = identity.resume_state.graph
        surface_result = None
    else:
        if atoms is None:  # pragma: no cover - fresh preparation invariant
            raise RuntimeError("fresh structure preparation produced no atoms")
        constants = cfg.constants
        structure_settings = cfg.structure
        if structure_settings.kind == "file":
            connectivity_axes = _pbc_connectivity_axes(
                atoms,
                nl_mult=constants.neighbor_list_multiplier,
            )
            if int(np.count_nonzero(connectivity_axes)) == 2:
                frame_metadata = align_periodic_slab_frame(
                    atoms,
                    connectivity_axes,
                )
                source = dict(structure.structure_source or {})
                source["surface_frame"] = frame_metadata
                structure.structure_source = source
                if verbose and frame_metadata["rotation_applied"]:
                    print(
                        "[autokmc]   file-backed slab rigidly aligned: "
                        "surface normal → +z"
                    )
        surface_result = find_surface_atoms(
            atoms,
            nl_mult=constants.neighbor_list_multiplier,
            surf_radius_factor=structure_settings.surface_radius_factor,
            coverage_threshold=constants.raycast_coverage_threshold,
            n_disc_sample=constants.raycast_disc_samples,
            which=structure_settings.surface_side,
            hull_tol_factor=(
                structure_settings.nanoparticle_hull_tolerance_factor
            ),
            tag_atoms=True,
        )
        graph = build_graph(
            atoms,
            nl_mult=constants.neighbor_list_multiplier,
        )

    set_run_id(graph, identity.run_id)
    set_frozen_indices(graph, structure.frozen_indices)

    if verbose and surface_result is not None and atoms is not None:
        print(
            "[autokmc]   surface classification successful: "
            f"{len(surface_result.indices)}/{len(atoms)} surface atom(s), "
            f"method={surface_result.method}"
        )
        print(
            f"[autokmc]   graph ready: {graph.number_of_nodes()} node(s), "
            f"{graph.number_of_edges()} edge(s)"
        )
    elif verbose:
        print(
            f"[autokmc]   checkpoint graph restored: "
            f"{graph.number_of_nodes()} node(s), "
            f"{graph.number_of_edges()} edge(s)"
        )

    return PreparedSystem(
        graph=graph,
        atoms=atoms,
        frozen_indices=structure.frozen_indices,
        surface_result=surface_result,
        structure_source=structure.structure_source,
    )


def prepare_system(
    cfg,
    identity: RunIdentity,
    calculators: PreparedCalculator,
    *,
    config_path: str | None = None,
    verbose: bool = False,
) -> PreparedSystem:
    """Compatibility facade for callers that prepare both stages together."""
    structure = prepare_structure(
        cfg,
        identity,
        calculators,
        config_path=config_path,
        verbose=verbose,
    )
    return prepare_material_graph(
        cfg,
        identity,
        structure,
        verbose=verbose,
    )


def prepare_reactants(
    cfg,
    identity: RunIdentity,
    calculator_resource,
    thermo_runtime,
    *,
    pressure_resolver: Callable[..., float] | None = None,
    verbose: bool = False,
) -> list:
    """Build configured gas species or restore them from a checkpoint."""
    from autokmc.species.reactant import build_reactant

    if identity.resume_state is not None:
        reactants = list(identity.resume_state.reactants)
    else:
        reactants = []
        for reactant_cfg in cfg.reactants:
            if verbose:
                print(
                    f"[autokmc]   reactant {reactant_cfg.smiles!r}: "
                    "building 3D structure"
                )
            pressure = (
                pressure_resolver(reactant_cfg, cfg.free_energy)
                if pressure_resolver is not None
                else (
                    reactant_cfg.partial_pressure_bar
                    if reactant_cfg.partial_pressure_bar is not None
                    else cfg.free_energy.pressure_bar
                )
            )
            reactant = build_reactant(
                reactant_cfg.smiles,
                add_hydrogens=reactant_cfg.add_hydrogens,
                calculator=calculator_resource,
                relax=reactant_cfg.relax_in_gas,
                nl_mult=cfg.constants.neighbor_list_multiplier,
                random_seed=cfg.kmc.random_seed,
                free_energy_options=(
                    thermo_runtime.options if cfg.free_energy.enabled else None
                ),
                free_energy_temperature_k=cfg.kmc.temperature_k,
                partial_pressure_bar=float(pressure),
                symmetry_number=reactant_cfg.symmetry_number,
                spin=reactant_cfg.spin,
                geometry=reactant_cfg.geometry,
                vib_cache_root=thermo_runtime.vibration_cache_root,
                optimizer=cfg.optimization.optimizer,
                optimizer_kwargs=cfg.optimization.optimizer_kwargs,
            )
            reactants.append(reactant)
            if verbose:
                energy = getattr(reactant, "energy", float("nan"))
                print(
                    f"[autokmc]   reactant {reactant.smiles!r}: ready, "
                    f"{len(reactant.atoms)} atom(s), E={energy:.6f} eV"
                )

    if not reactants:
        raise ValueError("config.reactants is empty — supply at least one SMILES.")
    return reactants


def prepare_adsorbate_sites(
    cfg,
    identity: RunIdentity,
    graph,
    reactants: list,
    calculator_resource,
    frozen_indices: list[int] | None,
    *,
    verbose: bool = False,
) -> list:
    """Enumerate the feed-species adsorption sites used to seed the network."""
    from autokmc.sites.adsorbate import find_adsorbate_sites

    diagnostics_dir = identity.output_dir / "diagnostics"
    graph.graph["diagnostics_dir"] = str(diagnostics_dir)
    if identity.resume_state is not None:
        return list(identity.resume_state.adsorbate_sites)

    settings = cfg.adsorbate_sites
    site_kwargs = configured_adsorbate_site_kwargs(cfg)
    all_sites: list = []
    for reactant in reactants:
        if verbose:
            if settings.prune_stable_only and calculator_resource is not None:
                print(
                    f"[autokmc]   {reactant.smiles!r}: searching placements; "
                    "candidate iso-classes will each receive one stability "
                    "optimization during pruning"
                )
            else:
                print(
                    f"[autokmc]   {reactant.smiles!r}: searching placements "
                    "(stability pruning disabled)"
                )
        sites = find_adsorbate_sites(
            graph,
            reactant,
            calculator=calculator_resource,
            frozen_indices=frozen_indices,
            diagnostics_dir=str(diagnostics_dir),
            verbose=verbose,
            **site_kwargs,
        )
        all_sites.extend(sites)
        if verbose:
            n_iso, n_members = summarize_sites(sites)
            print(
                f"[autokmc]   {reactant.smiles!r}: {n_iso} stable adsorbate "
                f"iso-class(es), {n_members} member placement(s)"
            )
    return all_sites


__all__ = [
    "count_z_layers",
    "configured_adsorbate_site_kwargs",
    "prepare_adsorbate_sites",
    "prepare_calculator",
    "prepare_material_graph",
    "prepare_reactants",
    "prepare_structure",
    "prepare_system",
    "summarize_sites",
]
