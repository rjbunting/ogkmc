"""Config-driven pipeline orchestration for autokmc."""

from __future__ import annotations

import logging
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autokmc.io.calculators import (
    CalculatorPool,
    acquire_calculator,
    build_calculator,
    calculator_meta,
    primary_calculator,
)
from autokmc.io.checkpoint import CheckpointWriter, load_checkpoint
from autokmc.io.calculation_cache import (
    initialise_calculation_database,
    write_isaac_export,
)
from autokmc.io.config import RunConfig
from autokmc.io.persistence import ReactionWriter
from autokmc.io.summary import ReactionSummary, make_run_meta
from autokmc.io.run_manifest import finish_run_manifest, start_run_manifest
from autokmc.io.trajectory import TrajectoryWriter
from autokmc.utils.logging import get_logger

_log = get_logger(__name__)


def _verbose_enabled(log_level: int) -> bool:
    return log_level <= logging.INFO


def _stage(message: str, *, verbose: bool) -> None:
    if verbose:
        print(f"\n[autokmc] {message}")


def _count_z_layers(atoms, *, tol: float = 0.35) -> int:
    z_values = sorted(float(z) for z in atoms.get_positions()[:, 2])
    if not z_values:
        return 0
    n_layers = 1
    last = z_values[0]
    for z in z_values[1:]:
        if abs(z - last) > tol:
            n_layers += 1
            last = z
    return n_layers


def _summarise_adsorbate_sites(sites: list) -> tuple[int, int]:
    return len(sites), sum(len(getattr(site, "member_node_ids", ())) for site in sites)


def _derive_configured_bond_templates(reactant_configs, reactants, bond_cfg):
    """Derive templates while preserving each reactant's hydrogen policy."""
    from autokmc.reactions.templates import derive_bond_templates

    reactant_smiles = [reactant.smiles for reactant in reactants]
    templates = []
    if bond_cfg.include_dissociation:
        for reactant_cfg, reactant in zip(reactant_configs, reactants):
            templates.extend(derive_bond_templates(
                [reactant.smiles],
                include_dissociation  = True,
                include_coupling      = False,
                bond_types            = tuple(bond_cfg.bond_types),
                include_ring_bonds    = bond_cfg.include_ring_bonds,
                add_hydrogens         = reactant_cfg.add_hydrogens,
            ))
    if bond_cfg.include_coupling:
        templates.extend(derive_bond_templates(
            reactant_smiles,
            include_dissociation  = False,
            include_coupling      = True,
            include_homo_coupling = bond_cfg.include_homo_coupling,
        ))

    template_keys: set[tuple[str, str, str]] = set()
    unique_templates = []
    for template in templates:
        key = (template.smiles_a, template.smiles_b, template.smiles_c)
        if key in template_keys:
            continue
        template_keys.add(key)
        unique_templates.append(template)
    return unique_templates


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def run_from_config(cfg: RunConfig, *, config_path: str | None = None) -> dict:
    """Run the autokmc pipeline end-to-end from a :class:`RunConfig`.

    Returns the KMC summary dict (same shape as
    :func:`autokmc.kmc.engine.run_kmc_steps`) augmented with an
    ``outputs`` key listing the files written.
    """
    from autokmc.structure import build_surface, build_nanoparticle
    from autokmc.structure import find_surface_atoms
    from autokmc.core.graph import build_graph
    from autokmc.species.reactant import build_reactant
    from autokmc.sites.adsorbate import find_adsorbate_sites
    from autokmc.sites.diffusion import find_diffusion_sites
    from autokmc.sites.bond import (
        find_bond_sites,
        prune_unstable_bond_sites,
        rebuild_bond_reverse_indexes,
    )
    from autokmc.kmc.expansion import initialise_bond_registry
    from autokmc.kmc.engine import run_kmc_steps
    from autokmc.species.smiles import canonical_smiles

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / cfg.output.run_manifest_filename
    resume_state = (
        load_checkpoint(cfg.checkpoint.resume_from)
        if cfg.checkpoint.resume_from else None
    )
    manifest_run_id: str | None = None
    if cfg.checkpoint.resume_from and manifest_path.is_file():
        try:
            manifest_run_id = str(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id")
                or ""
            )
        except (OSError, json.JSONDecodeError, TypeError):
            manifest_run_id = None
    checkpoint_run_id = (
        None if resume_state is None else resume_state.metadata.get("run_id")
    )
    if (
        manifest_run_id and checkpoint_run_id is not None
        and manifest_run_id != str(checkpoint_run_id)
    ):
        raise ValueError(
            "checkpoint run_id does not match the output run manifest: "
            f"{checkpoint_run_id!r} != {manifest_run_id!r}"
        )
    run_id = manifest_run_id or (
        str(checkpoint_run_id) if checkpoint_run_id is not None else None
    )
    if not run_id:
        run_id = str(uuid.uuid4())

    log_level = getattr(logging, str(cfg.output.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    verbose_run = _verbose_enabled(log_level)

    started_at = datetime.now(timezone.utc)

    # 1. Calculator
    _stage("Stage 1/7: preparing calculator", verbose=verbose_run)
    calc_resource = build_calculator(cfg.calculator)
    calc = primary_calculator(calc_resource)
    if calc is None:
        _log.warning("No calculator configured — falling back to ASE EMT.")
        from ase.calculators.emt import EMT
        calc_resource = CalculatorPool([EMT()])
        calc = primary_calculator(calc_resource)
    elif verbose_run:
        print(f"[autokmc]   calculator ready: {type(calc).__name__}")

    # 2. Structure
    s = cfg.structure
    if resume_state is None:
        _stage(f"Stage 2/7: building {s.kind} structure", verbose=verbose_run)
    else:
        _stage("Stage 2/7: restoring structure and graph from checkpoint", verbose=verbose_run)
    with acquire_calculator(calc_resource, purpose="structure construction") as calc:
        if resume_state is not None:
            atoms = None
        elif s.kind == "surface":
            atoms = build_surface(
                composition       = s.composition,
                crystal_structure = s.crystal_structure,
                miller_index      = tuple(s.miller_index),
                lattice_constant  = s.lattice_constant,
                min_slab_size     = s.min_slab_size,
                min_vacuum_size   = s.min_vacuum_size,
                goal_x            = s.goal_x,
                goal_y            = s.goal_y,
                n_freeze_layers   = s.n_freeze_layers,
                fmax              = s.fmax,
                max_steps         = s.max_steps,
                calculator        = calc,
                verbose           = verbose_run,
                **(s.extra_kwargs or {}),
            )
        elif s.kind == "nanoparticle":
            atoms = build_nanoparticle(
                composition       = s.composition,
                crystal_structure = s.crystal_structure,
                lattice_constant  = s.lattice_constant,
                target_atoms      = int(s.n_atoms) if s.n_atoms else 600,
                surface_energies  = s.surface_energies,
                fmax              = s.fmax,
                max_steps         = s.max_steps,
                surface_energy_facets    = s.surface_energy_facets,
                surface_energy_layers    = s.surface_energy_layers,
                surface_energy_vacuum    = s.surface_energy_vacuum,
                surface_energy_fmax      = s.surface_energy_fmax,
                surface_energy_max_steps = s.surface_energy_max_steps,
                calculator        = calc,
                verbose           = verbose_run,
                **(s.extra_kwargs or {}),
            )
        else:
            raise ValueError(f"unknown structure.kind={s.kind!r} (expected surface|nanoparticle)")
        if atoms is not None:
            atoms.calc = None

    frozen_indices = (
        resume_state.frozen_indices
        if resume_state is not None
        else list(atoms.info.get("frozen_indices", []) or []) or None
    )
    if verbose_run and atoms is not None:
        cell = atoms.get_cell()
        print(
            f"[autokmc]   structure built successfully: {len(atoms)} atoms, "
            f"{_count_z_layers(atoms)} z-layer(s), "
            f"cell=({np.linalg.norm(cell[0]):.2f}, "
            f"{np.linalg.norm(cell[1]):.2f}, {np.linalg.norm(cell[2]):.2f}) Å"
        )
        if frozen_indices:
            print(
                f"[autokmc]   frozen region: requested bottom "
                f"{s.n_freeze_layers} layer(s), {len(frozen_indices)} atom(s) fixed"
            )
        else:
            print("[autokmc]   frozen region: no fixed atoms")

    # 3. Surface tagging + graph
    if resume_state is None:
        _stage("Stage 3/7: classifying surface atoms and building graph", verbose=verbose_run)
        surface_result = find_surface_atoms(atoms, tag_atoms=True)
        G = build_graph(atoms)
    else:
        G = resume_state.graph
        surface_result = None
    G.graph["run_id"] = run_id
    G.graph["frozen_indices"] = list(frozen_indices or [])
    if verbose_run and surface_result is not None and atoms is not None:
        n_surface = len(surface_result.indices)
        print(
            f"[autokmc]   surface classification successful: "
            f"{n_surface}/{len(atoms)} surface atom(s), "
            f"method={surface_result.method}"
        )
        print(
            f"[autokmc]   graph ready: {G.number_of_nodes()} node(s), "
            f"{G.number_of_edges()} edge(s)"
        )
    elif verbose_run:
        print(
            f"[autokmc]   checkpoint graph restored: {G.number_of_nodes()} "
            f"node(s), {G.number_of_edges()} edge(s)"
        )

    # 3b. Free-energy options + persistent vibration cache root.
    fe_cfg = cfg.free_energy
    from autokmc.thermo.free_energy import FreeEnergyOptions
    free_energy_options = FreeEnergyOptions(
        enabled                 = fe_cfg.enabled,
        pressure_bar            = fe_cfg.pressure_bar,
        vibration_displacement  = fe_cfg.vibration_displacement,
        vibration_nfree         = fe_cfg.vibration_nfree,
        include_ts_vibrations   = fe_cfg.include_ts_vibrations,
        min_frequency_ev        = fe_cfg.min_frequency_ev,
        symmetry_tolerance      = fe_cfg.symmetry_tolerance,
        default_spin            = fe_cfg.default_spin,
        default_geometry        = fe_cfg.default_geometry,
        cache_dir               = fe_cfg.cache_dir,
    )
    vib_cache_root: str | None = (
        fe_cfg.cache_dir if fe_cfg.cache_dir
        else str(out_dir / "vib_cache")
    )
    calculation_cache_root: str | None = (
        str(out_dir / cfg.output.calculation_cache_dir)
        if cfg.output.calculation_cache_enabled
        else None
    )
    if calculation_cache_root is not None:
        initialise_calculation_database(calculation_cache_root, run_id=run_id)

    # 4. Reactants
    if resume_state is None:
        _stage("Stage 4/7: building gas-phase reactants", verbose=verbose_run)
    reactants_built = (
        [] if resume_state is None else list(resume_state.reactants)
    )
    reactant_configs = cfg.reactants if resume_state is None else []
    for r in reactant_configs:
        if verbose_run:
            print(f"[autokmc]   reactant {r.smiles!r}: building 3D structure")
        rx = build_reactant(
            r.smiles,
            add_hydrogens             = r.add_hydrogens,
            calculator                = calc_resource,
            relax                     = r.relax_in_gas,
            free_energy_options       = free_energy_options if fe_cfg.enabled else None,
            free_energy_temperature_k = cfg.kmc.temperature_k,
            partial_pressure_bar      = (
                r.partial_pressure_bar if r.partial_pressure_bar is not None
                else fe_cfg.pressure_bar
            ),
            symmetry_number           = r.symmetry_number,
            spin                      = r.spin,
            geometry                  = r.geometry,
            vib_cache_root            = vib_cache_root,
        )
        reactants_built.append(rx)
        if verbose_run:
            e = getattr(rx, "energy", float("nan"))
            print(
                f"[autokmc]   reactant {rx.smiles!r}: ready, "
                f"{len(rx.atoms)} atom(s), E={e:.6f} eV"
            )

    if not reactants_built:
        raise ValueError("config.reactants is empty — supply at least one SMILES.")
    # 5. Adsorbate sites for every reactant
    if resume_state is None:
        _stage("Stage 5/7: enumerating and pruning adsorbate sites", verbose=verbose_run)
    asc = cfg.adsorbate_sites
    all_sites: list = (
        [] if resume_state is None else list(resume_state.adsorbate_sites)
    )
    reactants_to_enumerate = reactants_built if resume_state is None else []
    for rx in reactants_to_enumerate:
        if verbose_run:
            if asc.prune_stable_only and calc_resource is not None:
                print(
                    f"[autokmc]   {rx.smiles!r}: searching placements; "
                    "candidate iso-classes will each receive one stability "
                    "optimization during pruning"
                )
            else:
                print(
                    f"[autokmc]   {rx.smiles!r}: searching placements "
                    "(stability pruning disabled)"
                )
        sites = find_adsorbate_sites(
            G, rx,
            prune_stable_only = asc.prune_stable_only,
            calculator        = calc_resource,
            frozen_indices    = frozen_indices,
            prune_fmax        = asc.fmax,
            prune_max_steps   = asc.max_steps,
            verbose           = verbose_run,
        )
        all_sites.extend(sites)
        if verbose_run:
            n_iso, n_members = _summarise_adsorbate_sites(sites)
            print(
                f"[autokmc]   {rx.smiles!r}: {n_iso} stable adsorbate "
                f"iso-class(es), {n_members} member placement(s)"
            )

    # Snapshot of user-reactant-only sites to pass to the KMC loop.
    # Leaf species (coupling products, fragments) are added to all_sites below
    # so that find_bond_sites can enumerate A+B⇌C iso-classes, but they must
    # NOT enter the KMC segment tree from the start — their partial pressures
    # are 0 (produced on-surface only) and they should only be activated when
    # a bond reaction first produces them.  The on-the-fly expansion machinery
    # in kmc_simulation adds them at that point.
    kmc_initial_sites: list = list(all_sites)

    # 6. Persistence hooks
    _stage("Stage 6/7: preparing persistence and optional reaction channels", verbose=verbose_run)
    is_resume = bool(cfg.checkpoint.resume_from)
    reaction_writer = ReactionWriter(
        out_dir,
        reactions_filename = cfg.output.reactions_filename,
        calculator_meta    = calculator_meta(cfg.calculator),
        append             = is_resume,
        run_id             = run_id,
    )
    trajectory_writer = TrajectoryWriter(
        out_dir / cfg.output.trajectory_filename,
        dump_every = cfg.output.trajectory_dump_every,
        append = is_resume,
    )
    summary_collector = ReactionSummary()
    checkpoint_writer = None
    if cfg.checkpoint.enabled:
        checkpoint_path = cfg.checkpoint.path or str(out_dir / "checkpoint.pkl")
        checkpoint_writer = CheckpointWriter(
            checkpoint_path,
            every_n_steps=cfg.checkpoint.every_n_steps,
            metadata={"config_path": config_path, "run_id": run_id},
        )

    # 6b. Diffusion (NEB) sites — flat list across all SMILES
    diffusion_sites_flat: list = (
        [] if resume_state is None else list(resume_state.diffusion_sites)
    )
    diffusion_kwargs: dict | None = None
    d = cfg.diffusion
    if d.enabled:
        diffusion_kwargs = dict(
            fmax             = d.fmax,
            max_steps        = d.max_steps,
            n_images         = d.n_images,
            climb            = d.climb,
            spring_k         = d.spring_k,
            interpolation    = d.interpolation,
            persist_neb_path = d.persist_neb_path,
        )
        if resume_state is None:
            if verbose_run:
                print("[autokmc]   diffusion enabled: enumerating hop permutations")
            diff_by_smiles = find_diffusion_sites(
                G, all_sites,
                max_hops                 = d.max_hops,
                n_shells_pair            = d.n_shells_pair,
                prune_by_adsorption_pair = d.prune_by_adsorption_pair,
                verbose                  = verbose_run,
            )
            for _smiles, ds_list in diff_by_smiles.items():
                diffusion_sites_flat.extend(ds_list)
            if verbose_run:
                print(
                    f"[autokmc] Diffusion enabled: "
                    f"{sum(len(v) for v in diff_by_smiles.values())} "
                    f"DiffusionSite iso-class(es) across "
                    f"{len(diff_by_smiles)} SMILES."
                )

    # 6c. Bond-changing reactions (A + B ⇌ C).  When enabled, derive
    # templates from the user-supplied reactant SMILES, build sites for
    # any "leaf" species the templates reference, enumerate the bond
    # iso-classes on the live graph, and bootstrap the on-the-fly growth
    # registry so future coupling events can introduce new species.
    b = cfg.bond
    bond_sites: list = (
        [] if resume_state is None else list(resume_state.bond_sites)
    )
    if b.enabled and resume_state is None:
        reactant_smiles = [rx.smiles for rx in reactants_built]
        # Hydrogen materialisation is a per-reactant choice.
        templates = _derive_configured_bond_templates(
            cfg.reactants, reactants_built, b,
        )
        if verbose_run:
            print(
                f"[autokmc] Bond reactions enabled: derived "
                f"{len(templates)} template(s) from "
                f"{len(reactant_smiles)} reactant SMILES."
            )

        # Map species → Reactant and species → list[AdsorbateSite] for the
        # registry bootstrap.  Use canonical SMILES as the key so it lines
        # up with what the templates / registry use internally.
        reactant_by_smi: dict[str, object] = {
            canonical_smiles(rx.smiles): rx for rx in reactants_built
        }
        sites_by_smi: dict[str, list] = {}
        for s in all_sites:
            sites_by_smi.setdefault(canonical_smiles(s.reactant), []).append(s)

        # Build any leaf species (fragments / coupling products) that the
        # templates reference but the user did not list under `reactants`.
        if templates:
            leaves: list[str] = []
            for t in templates:
                for smi in (t.smiles_a, t.smiles_b, t.smiles_c):
                    cs = canonical_smiles(smi)
                    if cs and cs not in reactant_by_smi and cs not in leaves:
                        leaves.append(cs)
            if leaves and not b.auto_build_leaf_species:
                missing = sorted(leaves)
                raise ValueError(
                    f"bond.auto_build_leaf_species is False but the derived "
                    f"templates reference {len(missing)} species not in "
                    f"`reactants`: {missing}.  Add them to `reactants` or "
                    f"set `bond.auto_build_leaf_species: true`."
                )
            for cs in leaves:
                if verbose_run:
                    print(f"[autokmc]   leaf species: {cs!r} — building Reactant + adsorbate sites…")
                rx_leaf = build_reactant(
                    cs,
                    add_hydrogens             = False,
                    calculator                = calc_resource,
                    free_energy_options       = free_energy_options if fe_cfg.enabled else None,
                    free_energy_temperature_k = cfg.kmc.temperature_k,
                    # Leaf species are produced on-surface only — they are not
                    # present in the gas phase, so their partial pressure is 0.
                    # This ensures adsorption rate = 0 (they can only appear via
                    # a bond reaction, never from the gas phase).
                    partial_pressure_bar      = 0.0,
                    vib_cache_root            = vib_cache_root,
                )
                reactants_built.append(rx_leaf)
                reactant_by_smi[cs] = rx_leaf
                leaf_sites = find_adsorbate_sites(
                    G, rx_leaf,
                    prune_stable_only = asc.prune_stable_only,
                    calculator        = calc_resource,
                    frozen_indices    = frozen_indices,
                    prune_fmax        = asc.fmax,
                    prune_max_steps   = asc.max_steps,
                    verbose           = verbose_run,
                )
                all_sites.extend(leaf_sites)
                sites_by_smi.setdefault(cs, []).extend(leaf_sites)
                if verbose_run:
                    n_iso, n_members = _summarise_adsorbate_sites(leaf_sites)
                    print(
                        f"[autokmc]   leaf species {cs!r}: {n_iso} stable "
                        f"adsorbate iso-class(es), {n_members} member placement(s)"
                    )

        # Enumerate bond iso-classes on the live graph.  Empty templates
        # short-circuit to an empty list — find_bond_sites would still
        # raise on an empty adsorbate_sites list, but `all_sites` is
        # guaranteed non-empty here (asserted above when reactants exist).
        #
        # IMPORTANT: always pass prune_by_triple=False here so that the
        # ego-size "one per adsorption triple" pruning (Stage 2) never
        # runs before the calculator stability check (Stage 1).  If Stage 2
        # ran first it could discard a stable site in favour of a smaller-ego
        # one that subsequently fails Stage 1, leaving a triple with no
        # representative.  The correct order is:
        #   Stage 1 — optimise / stability-check ALL enumerated sites
        #   Stage 2 — keep the best-ego survivor per triple
        if templates:
            from autokmc.sites.bond import _prune_one_per_adsorption_triple

            bond_sites = find_bond_sites(
                G, all_sites, templates,
                max_hops            = b.bond_max_hops,
                surface_apsp_cutoff = b.surface_apsp_cutoff,
                deduplicate_iso     = b.deduplicate_iso,
                n_shells_pair       = b.pair_n_shells,
                prune_by_triple     = False,   # always defer to after Stage 1
                gas_species         = reactant_by_smi,
                gas_lift_height     = b.gas_lift_height,
                verbose             = verbose_run,
            )

            # Stage 1 — calculator-based A+B endpoint stability prune.
            # Runs on the full enumerated set so no viable site is discarded
            # before its stability has been assessed.
            if b.prune_with_calculator and calc_resource is not None and bond_sites:
                species_by_smi: dict = {
                    canonical_smiles(rx.smiles): rx for rx in reactants_built
                }
                bond_sites = prune_unstable_bond_sites(
                    G, bond_sites, species_by_smi, calc_resource,
                    frozen_indices = frozen_indices,
                    fmax           = b.prune_fmax,
                    max_steps      = b.prune_max_steps,
                    verbose        = verbose_run,
                )

            # Stage 2 — keep the smallest-ego BondReactionSite per
            # (frozenset({iso_a, iso_b}), iso_c) adsorption triple.
            # Always runs after Stage 1 so only stable survivors compete.
            if b.prune_by_triple and bond_sites:
                bond_sites = _prune_one_per_adsorption_triple(
                    bond_sites,
                    verbose=verbose_run,
                    prefix=" (post-stability)",
                )
                for new_idx, brs in enumerate(bond_sites):
                    brs.iso_class = new_idx
                G.graph["bond_reaction_sites"] = bond_sites
                rebuild_bond_reverse_indexes(G, bond_sites)

        # Bootstrap the on-the-fly registry so coupling events that
        # introduce a new species can extend the network mid-run via
        # ``expand_bond_sites_after_event``.
        initialise_bond_registry(
            G,
            reactants       = list(reactant_by_smi.values()),
            adsorbate_sites = sites_by_smi,
            templates       = templates,
            bond_sites      = bond_sites,
            # Mark the user-provided reactants as fully expanded so the
            # KMC loop does not attempt to re-derive their templates.
            # Leaf/product species built via auto_build_leaf_species are
            # intentionally NOT listed here — they will be expanded on the
            # fly the first time they appear on the surface as a product or
            # fragment.
            expanded_smiles = reactant_smiles,
        )

        if verbose_run:
            print(
                f"[autokmc] Bond reactions: {len(bond_sites)} "
                f"BondReactionSite iso-class(es) enumerated; "
                f"registry seeded with {len(reactant_by_smi)} species."
            )

    # Build the kwargs forwarded to the bond channel inside the KMC loop.
    bond_sites_for_kmc: list | None = None
    bond_kwargs: dict | None = None
    bond_growth_kwargs: dict | None = None
    if b.enabled:
        bond_sites_for_kmc = bond_sites
        bond_kwargs = dict(
            fmax             = b.neb_fmax,
            max_steps        = b.neb_max_steps,
            n_images         = b.neb_n_images,
            climb            = b.neb_climb,
            spring_k         = b.neb_spring_k,
            interpolation    = b.neb_interpolation,
            atom_matching    = b.atom_matching,
            matching_trials  = b.matching_trials,
            persist_neb_path = b.persist_neb_path,
        )
        bond_growth_kwargs = dict(
            find_diffusion              = d.enabled,
            # NOTE: verbose is passed explicitly by run_kmc_steps; do NOT
            # include it here or Python will raise "multiple values for
            # keyword argument 'verbose'" at every bond event.
            frozen_indices              = frozen_indices,
            bond_max_hops               = b.bond_max_hops,
            surface_apsp_cutoff         = b.surface_apsp_cutoff,
            bond_pair_n_shells          = b.pair_n_shells,
            bond_prune_by_triple        = b.prune_by_triple,
            bond_prune_with_calculator  = b.prune_with_calculator,
            prune_fmax                  = b.prune_fmax,
            prune_max_steps             = b.prune_max_steps,
            bond_types                  = tuple(b.bond_types),
            include_ring_bonds          = b.include_ring_bonds,
            include_homo_coupling       = b.include_homo_coupling,
            include_dissociation        = b.include_dissociation,
            include_coupling            = b.include_coupling,
            deduplicate_iso             = b.deduplicate_iso,
            auto_build_leaf_species     = b.auto_build_leaf_species,
            add_hydrogens               = False,
            gas_lift_height             = b.gas_lift_height,
            diffusion_max_hops          = d.max_hops,
            diffusion_n_shells_pair     = d.n_shells_pair,
            diffusion_prune_by_ads_pair = d.prune_by_adsorption_pair,
        )

    initial_step = 0
    initial_time_s = 0.0
    initial_history: list = []
    initial_reaction_counts: dict[str, int] = {}
    initial_rng_state: dict | None = None
    if cfg.checkpoint.resume_from:
        if resume_state is None:  # defensive; resume_from guarantees this above
            raise RuntimeError("checkpoint resume state was not loaded")
        state = resume_state
        G = state.graph
        reactants_built = list(state.reactants)
        all_sites = list(state.adsorbate_sites)
        kmc_initial_sites = list(state.adsorbate_sites)
        diffusion_sites_flat = list(state.diffusion_sites)
        bond_sites_for_kmc = list(state.bond_sites)
        frozen_indices = state.frozen_indices
        initial_step = int(state.step)
        initial_time_s = float(state.time_s)
        initial_history = list(state.history)
        initial_reaction_counts = dict(state.reaction_counts)
        initial_rng_state = getattr(state, "rng_state", None)
        G.graph["run_id"] = run_id
        G.graph["frozen_indices"] = list(frozen_indices or [])
        summary_collector = ReactionSummary.from_events(reaction_writer.jsonl_path)
        if verbose_run:
            print(
                f"[autokmc] Resuming from checkpoint {cfg.checkpoint.resume_from}: "
                f"step={initial_step}, t={initial_time_s:.4e} s"
            )

    # 7. KMC
    _stage("Stage 7/7: starting KMC simulation", verbose=verbose_run)
    k = cfg.kmc
    start_run_manifest(
        manifest_path,
        graph=G,
        adsorbate_sites=kmc_initial_sites,
        feed_reactants=[
            {
                "smiles": reactant.smiles,
                "partial_pressure_bar": reactant.partial_pressure_bar,
                "thermochemistry": dict(reactant.thermo_meta),
            }
            for reactant in reactants_built
            if canonical_smiles(reactant.smiles)
            in {canonical_smiles(item.smiles) for item in cfg.reactants}
        ],
        temperature_k=k.temperature_k,
        random_seed=k.random_seed,
        structure_kind=cfg.structure.kind,
        composition=cfg.structure.composition,
        config_path=config_path,
        events_filename=cfg.output.reactions_filename,
        initial_step=initial_step,
        initial_time_s=initial_time_s,
        run_id=run_id,
        resolved_config=asdict(cfg),
    )
    summary = run_kmc_steps(
        G, kmc_initial_sites, calc_resource,
        reactants                = reactants_built,
        temperature              = k.temperature_k,
        n_steps                  = k.n_steps,
        transmission_coefficient = k.transmission_coefficient,
        frozen_indices           = frozen_indices,
        fmax                     = k.fmax,
        max_steps                = k.max_steps,
        rng                      = k.random_seed,
        log_every                = k.log_every,
        verbose                  = verbose_run,
        lateral_interactions     = k.lateral_interactions,
        diffusion_sites          = diffusion_sites_flat,
        diffusion_kwargs         = diffusion_kwargs,
        bond_sites               = bond_sites_for_kmc,
        bond_kwargs              = bond_kwargs,
        bond_growth_kwargs       = bond_growth_kwargs,
        free_energy_options      = free_energy_options if fe_cfg.enabled else None,
        vib_cache_root           = vib_cache_root,
        calculation_cache_root   = calculation_cache_root,
        reaction_writer          = reaction_writer,
        trajectory_writer        = trajectory_writer,
        summary_collector        = summary_collector,
        checkpoint_writer        = checkpoint_writer,
        initial_step             = initial_step,
        initial_time_s           = initial_time_s,
        initial_history          = initial_history,
        initial_reaction_counts  = initial_reaction_counts,
        initial_rng_state        = initial_rng_state,
    )

    finished_at = datetime.now(timezone.utc)

    # 8. Final summary.json
    run_meta = make_run_meta(
        config_path       = config_path,
        temperature_k     = k.temperature_k,
        n_steps_requested = k.n_steps,
        steps_executed    = summary.get("steps_executed"),
        total_time_s      = summary.get("time"),
        random_seed       = k.random_seed,
        started_at        = started_at,
        finished_at       = finished_at,
    )
    summary_path = summary_collector.write(
        out_dir / cfg.output.summary_filename,
        run_meta        = run_meta,
        final_occupancy = summary.get("final_occupancy"),
    )
    isaac_export_path = write_isaac_export(
        calculation_cache_root,
        out_dir / cfg.output.isaac_export_filename,
    )
    reaction_writer.close()
    finish_run_manifest(
        manifest_path,
        final_step=initial_step + int(summary.get("steps_executed", 0) or 0),
        final_time_s=float(summary.get("time", initial_time_s) or initial_time_s),
        steps_executed=int(summary.get("steps_executed", 0) or 0),
    )

    summary["outputs"] = {
        "events":        str(reaction_writer.jsonl_path),
        "summary":       str(summary_path),
        "run_manifest":  str(manifest_path),
        "calculation_cache": calculation_cache_root,
        "isaac_records": (
            str(isaac_export_path) if isaac_export_path is not None else None
        ),
        "trajectory": (
            str(trajectory_writer.output_path)
            if trajectory_writer.enabled else None
        ),
        "reactions_dir": str(reaction_writer.reactions_root),
        "n_unique_reactions": reaction_writer.n_unique_reactions,
        "checkpoint": (
            str(checkpoint_writer.last_path or checkpoint_writer.path)
            if checkpoint_writer is not None else None
        ),
    }
    return summary


__all__ = ["run_from_config"]
