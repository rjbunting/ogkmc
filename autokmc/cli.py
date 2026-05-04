"""
autokmc.cli
===========
Command-line interface for the autokmc pipeline.

Two subcommands:

* ``autokmc run CONFIG``             — run the full pipeline.
* ``autokmc validate-config CONFIG`` — parse the config and exit.

The CLI is **calculator-agnostic** — see :class:`autokmc.config.CalculatorCfg`
for the dynamic loading scheme that supports VASP, CP2K, EMT, NequIP, MACE,
or any other ASE-compatible calculator.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from autokmc.config import (
    RunConfig,
    load_config,
    build_calculator,
    calculator_meta,
)
from autokmc.logging_utils import get_logger
from autokmc.persistence import (
    ReactionWriter,
    TrajectoryWriter,
    ReactionSummary,
    make_run_meta,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def run_from_config(cfg: RunConfig, *, config_path: str | None = None) -> dict:
    """Run the autokmc pipeline end-to-end from a :class:`RunConfig`.

    Returns the KMC summary dict (same shape as
    :func:`autokmc.kmc_simulation.run_kmc_steps`) augmented with an
    ``outputs`` key listing the files written.
    """
    from autokmc import (
        build_surface,
        build_nanoparticle,
        find_surface_atoms,
        build_graph,
        build_reactant,
        find_adsorbate_sites,
        find_diffusion_sites,
        derive_bond_templates,
        find_bond_sites,
        prune_unstable_bond_sites,
        initialise_bond_registry,
        run_kmc_steps,
    )

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, str(cfg.output.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    started_at = datetime.now(timezone.utc)

    # 1. Calculator
    calc = build_calculator(cfg.calculator)
    if calc is None:
        _log.warning("No calculator configured — falling back to ASE EMT.")
        from ase.calculators.emt import EMT
        calc = EMT()

    # 2. Structure
    s = cfg.structure
    if s.kind == "surface":
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
            verbose           = log_level <= logging.INFO,
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
            calculator        = calc,
            verbose           = log_level <= logging.INFO,
            **(s.extra_kwargs or {}),
        )
    else:
        raise ValueError(f"unknown structure.kind={s.kind!r} (expected surface|nanoparticle)")

    frozen_indices = list(atoms.info.get("frozen_indices", []) or []) or None

    # 3. Surface tagging + graph
    find_surface_atoms(atoms, tag_atoms=True)
    G = build_graph(atoms)

    # 3b. Free-energy options + persistent vibration cache root.
    fe_cfg = cfg.free_energy
    from autokmc.free_energy import FreeEnergyOptions
    free_energy_options = FreeEnergyOptions(
        enabled                 = fe_cfg.enabled,
        pressure_bar            = fe_cfg.pressure_bar,
        vibration_displacement  = fe_cfg.vibration_displacement,
        vibration_nfree         = fe_cfg.vibration_nfree,
        include_ts_vibrations   = fe_cfg.include_ts_vibrations,
        min_frequency_cm        = fe_cfg.min_frequency_cm,
        default_symmetry_number = fe_cfg.default_symmetry_number,
        default_spin            = fe_cfg.default_spin,
        default_geometry        = fe_cfg.default_geometry,
        cache_dir               = fe_cfg.cache_dir,
    )
    vib_cache_root: str | None = (
        fe_cfg.cache_dir if fe_cfg.cache_dir
        else str(out_dir / "vib_cache")
    )

    # 4. Reactants
    reactants_built = []
    for r in cfg.reactants:
        rx = build_reactant(
            r.smiles,
            add_hydrogens             = r.add_hydrogens,
            calculator                = calc if r.relax_in_gas else None,
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

    if not reactants_built:
        raise ValueError("config.reactants is empty — supply at least one SMILES.")

    # 5. Adsorbate sites for every reactant
    asc = cfg.adsorbate_sites
    all_sites: list = []
    for rx in reactants_built:
        sites = find_adsorbate_sites(
            G, rx,
            prune_stable_only = asc.prune_stable_only,
            calculator        = calc,
            frozen_indices    = frozen_indices,
            prune_fmax        = asc.fmax,
            prune_max_steps   = asc.max_steps,
            verbose           = log_level <= logging.INFO,
        )
        all_sites.extend(sites)

    # Snapshot of user-reactant-only sites to pass to the KMC loop.
    # Leaf species (coupling products, fragments) are added to all_sites below
    # so that find_bond_sites can enumerate A+B⇌C iso-classes, but they must
    # NOT enter the KMC segment tree from the start — their partial pressures
    # are 0 (produced on-surface only) and they should only be activated when
    # a bond reaction first produces them.  The on-the-fly expansion machinery
    # in kmc_simulation adds them at that point.
    kmc_initial_sites: list = list(all_sites)

    # 6. Persistence hooks
    reaction_writer = ReactionWriter(
        out_dir,
        reactions_filename = cfg.output.reactions_filename,
        calculator_meta    = calculator_meta(cfg.calculator),
    )
    trajectory_writer = TrajectoryWriter(
        out_dir / cfg.output.trajectory_filename,
        dump_every = cfg.output.trajectory_dump_every,
    )
    summary_collector = ReactionSummary(
        reactant_smiles={rx.smiles for rx in reactants_built},
    )

    # 6b. Diffusion (NEB) sites — flat list across all SMILES
    diffusion_sites_flat: list = []
    diffusion_kwargs: dict | None = None
    d = cfg.diffusion
    if d.enabled:
        diff_by_smiles = find_diffusion_sites(
            G, all_sites,
            max_hops                 = d.max_hops,
            n_shells_pair            = d.n_shells_pair,
            prune_by_adsorption_pair = d.prune_by_adsorption_pair,
            verbose                  = log_level <= logging.INFO,
        )
        for smiles, ds_list in diff_by_smiles.items():
            diffusion_sites_flat.extend(ds_list)
        diffusion_kwargs = dict(
            fmax             = d.fmax,
            max_steps        = d.max_steps,
            n_images         = d.n_images,
            climb            = d.climb,
            spring_k         = d.spring_k,
            interpolation    = d.interpolation,
            persist_neb_path = d.persist_neb_path,
        )
        if log_level <= logging.INFO:
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
    if b.enabled:
        from autokmc.find_bond_sites import _canon_smiles  # type: ignore[attr-defined]

        reactant_smiles = [rx.smiles for rx in reactants_built]
        templates = derive_bond_templates(
            reactant_smiles,
            include_dissociation  = b.include_dissociation,
            include_coupling      = b.include_coupling,
            bond_types            = tuple(b.bond_types),
            include_ring_bonds    = b.include_ring_bonds,
            include_homo_coupling = b.include_homo_coupling,
        )
        if log_level <= logging.INFO:
            print(
                f"[autokmc] Bond reactions enabled: derived "
                f"{len(templates)} template(s) from "
                f"{len(reactant_smiles)} reactant SMILES."
            )

        # Map species → Reactant and species → list[AdsorbateSite] for the
        # registry bootstrap.  Use canonical SMILES as the key so it lines
        # up with what the templates / registry use internally.
        reactant_by_smi: dict[str, object] = {
            _canon_smiles(rx.smiles): rx for rx in reactants_built
        }
        sites_by_smi: dict[str, list] = {}
        for s in all_sites:
            sites_by_smi.setdefault(_canon_smiles(s.reactant), []).append(s)

        # Build any leaf species (fragments / coupling products) that the
        # templates reference but the user did not list under `reactants`.
        if templates:
            leaves: list[str] = []
            for t in templates:
                for smi in (t.smiles_a, t.smiles_b, t.smiles_c):
                    cs = _canon_smiles(smi)
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
                if log_level <= logging.INFO:
                    print(f"[autokmc]   leaf species: {cs!r} — building Reactant + adsorbate sites…")
                rx_leaf = build_reactant(
                    cs,
                    add_hydrogens             = False,
                    calculator                = calc,
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
                    calculator        = calc,
                    frozen_indices    = frozen_indices,
                    prune_fmax        = asc.fmax,
                    prune_max_steps   = asc.max_steps,
                    verbose           = log_level <= logging.INFO,
                )
                all_sites.extend(leaf_sites)
                sites_by_smi.setdefault(cs, []).extend(leaf_sites)

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
        bond_sites: list = []
        if templates:
            from autokmc.find_bond_sites import _prune_one_per_adsorption_triple

            bond_sites = find_bond_sites(
                G, all_sites, templates,
                max_hops            = b.bond_max_hops,
                surface_apsp_cutoff = b.surface_apsp_cutoff,
                deduplicate_iso     = b.deduplicate_iso,
                n_shells_pair       = b.pair_n_shells,
                prune_by_triple     = False,   # always defer to after Stage 1
                verbose             = log_level <= logging.INFO,
            )

            # Stage 1 — calculator-based A+B endpoint stability prune.
            # Runs on the full enumerated set so no viable site is discarded
            # before its stability has been assessed.
            if b.prune_with_calculator and calc is not None and bond_sites:
                species_by_smi: dict = {
                    _canon_smiles(rx.smiles): rx for rx in reactants_built
                }
                bond_sites = prune_unstable_bond_sites(
                    G, bond_sites, species_by_smi, calc,
                    frozen_indices = frozen_indices,
                    fmax           = b.prune_fmax,
                    max_steps      = b.prune_max_steps,
                    verbose        = log_level <= logging.INFO,
                )

            # Stage 2 — keep the smallest-ego BondReactionSite per
            # (frozenset({iso_a, iso_b}), iso_c) adsorption triple.
            # Always runs after Stage 1 so only stable survivors compete.
            if b.prune_by_triple and bond_sites:
                bond_sites = _prune_one_per_adsorption_triple(
                    bond_sites,
                    verbose=log_level <= logging.INFO,
                    prefix=" (post-stability)",
                )
                # Renumber iso_class and rebuild the clique reverse-index
                # to match the surviving set.
                G.graph["bond_clique_to_members"] = {}
                G.graph["bond_surface_node_to_members"] = {}
                rebuilt_idx = G.graph["bond_clique_to_members"]
                rebuilt_surf: dict = G.graph["bond_surface_node_to_members"]
                for new_idx, brs in enumerate(bond_sites):
                    brs.iso_class = new_idx
                    for m_idx, (cliques_a, cliques_b, cliques_c) in enumerate(
                        brs._member_cliques
                    ):
                        for clq in (*cliques_a, *cliques_b, *cliques_c):
                            rebuilt_idx.setdefault(clq, []).append(
                                (brs, m_idx)
                            )
                            for surf_id in clq:
                                rebuilt_surf.setdefault(
                                    int(surf_id), [],
                                ).append((brs, m_idx))
                G.graph["bond_reaction_sites"] = bond_sites

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

        if log_level <= logging.INFO:
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
            diffusion_max_hops          = d.max_hops,
            diffusion_n_shells_pair     = d.n_shells_pair,
            diffusion_prune_by_ads_pair = d.prune_by_adsorption_pair,
        )

    # 7. KMC
    k = cfg.kmc
    summary = run_kmc_steps(
        G, kmc_initial_sites, calc,
        reactants                = reactants_built,
        temperature              = k.temperature_k,
        n_steps                  = k.n_steps,
        transmission_coefficient = k.transmission_coefficient,
        frozen_indices           = frozen_indices,
        fmax                     = k.fmax,
        max_steps                = k.max_steps,
        rng                      = k.random_seed,
        log_every                = k.log_every,
        verbose                  = log_level <= logging.INFO,
        lateral_interactions     = k.lateral_interactions,
        diffusion_sites          = diffusion_sites_flat,
        diffusion_kwargs         = diffusion_kwargs,
        bond_sites               = bond_sites_for_kmc,
        bond_kwargs              = bond_kwargs,
        bond_growth_kwargs       = bond_growth_kwargs,
        free_energy_options      = free_energy_options if fe_cfg.enabled else None,
        vib_cache_root           = vib_cache_root,
        reaction_writer          = reaction_writer,
        trajectory_writer        = trajectory_writer,
        summary_collector        = summary_collector,
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
    reaction_writer.close()

    summary["outputs"] = {
        "events":        str(reaction_writer.jsonl_path),
        "summary":       str(summary_path),
        "trajectory": (
            str(trajectory_writer.output_path)
            if trajectory_writer.enabled else None
        ),
        "reactions_dir": str(reaction_writer.reactions_root),
        "n_unique_reactions": reaction_writer.n_unique_reactions,
    }
    return summary


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autokmc", description="autokmc CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline from a config file")
    p_run.add_argument("config", help="path to a .yaml/.yml/.toml config")

    p_val = sub.add_parser("validate-config", help="parse a config and exit 0/1")
    p_val.add_argument("config", help="path to a .yaml/.yml/.toml config")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.cmd == "validate-config":
        print(f"OK: {args.config} parsed successfully (schema={cfg.schema_version})")
        return 0

    if args.cmd == "run":
        summary = run_from_config(cfg, config_path=str(Path(args.config).resolve()))
        outputs = summary.get("outputs", {})
        n_unique = outputs.get("n_unique_reactions", 0)
        out_dir  = Path(outputs.get("events", "")).parent if outputs.get("events") else Path(cfg.output.dir)
        print(f"\n[autokmc] Run complete — output: {out_dir}")
        print(
            f"  steps={summary.get('steps_executed')}  "
            f"t={summary.get('time')!s} s  "
            f"events={summary.get('reaction_counts')}  "
            f"unique reactions discovered={n_unique}"
        )
        return 0

    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

