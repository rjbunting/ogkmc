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

    # 4. Reactants
    reactants_built = []
    for r in cfg.reactants:
        rx = build_reactant(
            r.smiles,
            add_hydrogens = r.add_hydrogens,
            calculator    = calc if r.relax_in_gas else None,
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
    summary_collector = ReactionSummary()

    # 6b. Diffusion (NEB) sites — flat list across all SMILES
    diffusion_sites_flat: list = []
    diffusion_kwargs: dict | None = None
    d = cfg.diffusion
    if d.enabled:
        diff_by_smiles = find_diffusion_sites(
            G, all_sites,
            max_hops      = d.max_hops,
            n_shells_pair = d.n_shells_pair,
            verbose       = log_level <= logging.INFO,
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

    # 7. KMC
    k = cfg.kmc
    summary = run_kmc_steps(
        G, all_sites, calc,
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

