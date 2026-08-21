"""KMC launch, persistence lifecycle, and run finalization."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from autokmc.io.calculation_cache import write_isaac_export
from autokmc.io.calculators import calculator_meta
from autokmc.io.performance import (
    PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH,
    build_performance_summary,
    write_performance_diagnostics,
)
from autokmc.io.persistence import (
    DIAGNOSTICS_DIR,
    INVALID_ADSORPTION_DIR,
    INVALID_BOND_DIR,
    INVALID_DIFFUSION_DIR,
)
from autokmc.io.run_manifest import (
    begin_run_manifest,
    build_artifact_inventory,
    configured_artifact_descriptors,
    discover_quarantine_locations,
    enrich_run_manifest,
    fail_run_manifest,
    finish_run_manifest,
    update_run_manifest,
)
from autokmc.io.summary import make_run_meta
from autokmc.kmc.models import (
    BondChannelOptions,
    BondGrowthOptions,
    DiffusionChannelOptions,
    KMCChannels,
    KMCObservers,
    KMCRunRequest,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.species.smiles import canonical_smiles
from autokmc.utils.telemetry import RuntimeTelemetry
from autokmc.workflow.models import (
    RunIdentity,
    SimulationContext,
)
from autokmc.workflow.runtime import create_output_sinks, resolve_kmc_resume


def _feed_reactants(cfg, reactants: list) -> list[dict]:
    """Return manifest records only for user-configured feed species."""
    feed_smiles = {canonical_smiles(item.smiles) for item in cfg.reactants}
    return [
        {
            "smiles": reactant.smiles,
            "partial_pressure_bar": reactant.partial_pressure_bar,
            "thermochemistry": dict(reactant.thermo_meta),
        }
        for reactant in reactants
        if canonical_smiles(reactant.smiles) in feed_smiles
    ]


def _termination(
    summary: dict[str, Any],
    *,
    n_steps_requested: int,
) -> tuple[str, str]:
    """Normalize typed and legacy completion results to explicit semantics."""
    status = str(summary.get("termination_status") or "")
    reason = str(summary.get("termination_reason") or "")
    executed = int(summary.get("steps_executed", 0) or 0)
    requested = int(n_steps_requested)
    if status == "success":
        status = "complete"
    if requested == 0:
        status, reason = "complete", "no_steps_requested"
    elif status == "complete" and executed < requested:
        status, reason = "stopped", "run_stopped_before_requested_steps"
    elif status not in {"complete", "stopped"}:
        if executed >= requested:
            status, reason = "complete", "requested_steps_completed"
        else:
            status, reason = "stopped", "legacy_adapter_stopped_early"
    elif not reason:
        reason = (
            "requested_steps_completed"
            if status == "complete"
            else "stopped_without_reason"
        )
    return status, reason


def _invalid_lateral_counts(network) -> dict[str, int]:
    """Count invalidated lateral classes retained in the prepared network."""

    def count(sites) -> int:
        return sum(
            1
            for site in sites
            for lateral in getattr(site, "lateral_classes", ())
            if getattr(lateral, "stable", None) is False
        )

    counts = {
        "adsorption": count(network.adsorbate_sites),
        "diffusion": count(network.diffusion_sites),
        "bond": count(network.bond_sites),
    }
    counts["total"] = sum(counts.values())
    return counts


def _output_paths(
    cfg,
    identity: RunIdentity,
    context: SimulationContext,
    sinks,
    *,
    summary_path: Path,
    performance_path: Path,
    isaac_export_path: Path | None,
) -> dict[str, Any]:
    """Return the canonical output map shared by results and the manifest."""
    checkpoint_path = None
    if sinks.checkpoint is not None:
        checkpoint_path = str(sinks.checkpoint.last_path or sinks.checkpoint.path)
    return {
        "events": str(sinks.reactions.jsonl_path),
        "summary": str(summary_path),
        "run_manifest": str(identity.manifest_path),
        "performance_diagnostics": str(performance_path),
        "calculation_cache": context.thermo.calculation_cache_root,
        "isaac_records": (
            str(isaac_export_path) if isaac_export_path is not None else None
        ),
        "trajectory": (
            str(sinks.trajectory.output_path)
            if sinks.trajectory.enabled
            else None
        ),
        "reactions_dir": str(sinks.reactions.reactions_root),
        "reaction_index": str(sinks.reactions.reaction_index_path),
        "invalid_adsorption": str(
            identity.output_dir / DIAGNOSTICS_DIR / INVALID_ADSORPTION_DIR
        ),
        "invalid_diffusion": str(
            identity.output_dir / DIAGNOSTICS_DIR / INVALID_DIFFUSION_DIR
        ),
        "invalid_bond": str(
            identity.output_dir / DIAGNOSTICS_DIR / INVALID_BOND_DIR
        ),
        "n_unique_reactions": sinks.reactions.n_unique_reactions,
        "checkpoint": checkpoint_path,
    }


def _artifact_descriptors(
    cfg,
    identity: RunIdentity,
    context: SimulationContext,
    sinks,
    *,
    summary_path: Path,
    performance_path: Path,
    isaac_export_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Return stable artifact types and their public schema versions."""
    descriptors = configured_artifact_descriptors(
        identity.manifest_path,
        cfg=cfg,
    )
    descriptors["events"]["path"] = sinks.reactions.jsonl_path
    descriptors["summary"]["path"] = summary_path
    descriptors["performance_diagnostics"]["path"] = performance_path
    descriptors["reactions"]["path"] = sinks.reactions.reactions_root
    descriptors["reaction_index"]["path"] = sinks.reactions.reaction_index_path
    if sinks.checkpoint is not None:
        descriptors["checkpoint"]["path"] = (
            sinks.checkpoint.last_path or sinks.checkpoint.path
        )
    if context.thermo.calculation_cache_root is not None:
        descriptors["calculation_cache"]["path"] = (
            context.thermo.calculation_cache_root
        )
    if isaac_export_path is not None:
        descriptors["isaac_records"]["path"] = isaac_export_path
    return descriptors


def _best_effort_failure(
    path: Path,
    error: BaseException,
    *,
    cfg,
    network=None,
    sinks=None,
    telemetry: RuntimeTelemetry | None,
) -> None:
    """Record a direct-stage failure without replacing the original error."""
    gauges = (
        telemetry.to_dict().get("gauges", {})
        if telemetry is not None
        else {}
    )
    invalid_counts = None
    if network is not None:
        try:
            invalid_counts = _invalid_lateral_counts(network)
            if sinks is not None:
                invalid_counts["persisted_invalid_reactions"] = (
                    sinks.reactions.n_invalid_reactions
                )
        except Exception:
            invalid_counts = None
    try:
        quarantine_locations = discover_quarantine_locations(path.parent)
    except Exception:
        quarantine_locations = None
    try:
        fail_run_manifest(
            path,
            error,
            status=(
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed"
            ),
            last_durable_step=(
                int(gauges["kmc.last_step"])
                if "kmc.last_step" in gauges
                else None
            ),
            last_durable_time_s=(
                float(gauges["kmc.time_s"])
                if "kmc.time_s" in gauges
                else None
            ),
            cfg=cfg,
            invalid_counts=invalid_counts,
            quarantine_locations=quarantine_locations,
        )
    except Exception:
        # The scientific or persistence exception that stopped the run is the
        # actionable failure. A secondary manifest-write error must not mask it.
        pass


def execute_kmc_stage(
    cfg,
    identity: RunIdentity,
    context: SimulationContext,
    *,
    started_at: datetime,
    config_path: str | None = None,
    verbose: bool = False,
    progress: bool | None = None,
    telemetry: RuntimeTelemetry | None = None,
    wall_started_s: float | None = None,
) -> dict:
    """Run KMC and publish all configured outputs as one managed stage."""
    # Resolve defaults at execution time so custom embeddings can still
    # replace individual scientific kernels through the established engine
    # seams without routing the workflow through the legacy 31-argument API.
    import autokmc.kmc.engine as engine_module

    resume = resolve_kmc_resume(identity)
    settings = cfg.kmc
    network = context.network
    thermo = context.thermo
    channels = context.channels
    progress_enabled = verbose if progress is None else progress

    sinks = None
    summary: dict[str, Any]
    try:
        if not identity.manifest_path.is_file():
            begin_run_manifest(
                identity.manifest_path,
                run_id=identity.run_id,
                config_path=config_path,
                resolved_config=asdict(cfg),
                events_filename=cfg.output.reactions_filename,
                initial_step=resume.step,
                initial_time_s=resume.time_s,
                is_resume=identity.is_resume,
            )
        enrich_run_manifest(
            identity.manifest_path,
            graph=context.graph,
            adsorbate_sites=network.initial_adsorbate_sites,
            feed_reactants=_feed_reactants(cfg, network.reactants),
            temperature_k=settings.temperature_k,
            random_seed=settings.random_seed,
            structure_kind=cfg.structure.kind,
            composition=(
                context.structure_source.get(
                    "chemical_formula",
                    cfg.structure.composition,
                )
                if context.structure_source
                else cfg.structure.composition
            ),
            structure_source=context.structure_source,
            initial_step=resume.step,
            initial_time_s=resume.time_s,
        )
        update_run_manifest(
            identity.manifest_path,
            status="running",
            current_stage="stage_7_kmc",
        )
        sinks = create_output_sinks(
            cfg,
            identity,
            calculator_meta(cfg.calculator),
            config_path=config_path,
        )
        request = KMCRunRequest(
            system=KMCSystem(
                graph=context.graph,
                adsorbate_sites=network.initial_adsorbate_sites,
                calculator=context.calculator,
                reactants=network.reactants,
            ),
            settings=KMCSettings(
                temperature=settings.temperature_k,
                n_steps=settings.n_steps,
                transmission_coefficient=settings.transmission_coefficient,
                frozen_indices=context.frozen_indices,
                log_every=settings.log_every,
                progress=progress_enabled,
                verbose=verbose,
                lateral_interactions=settings.lateral_interactions,
                lateral_shells=cfg.constants.lateral_shells,
                optimizer=cfg.optimization.optimizer,
                optimizer_kwargs=cfg.optimization.optimizer_kwargs,
            ),
            channels=KMCChannels(
                adsorption_options=channels.adsorption,
                diffusion_sites=network.diffusion_sites,
                diffusion_options=(
                    channels.diffusion or DiffusionChannelOptions()
                ),
                bond_sites=(
                    network.bond_sites
                    if identity.is_resume or cfg.bond.enabled
                    else []
                ),
                bond_options=channels.bond or BondChannelOptions(),
                bond_growth_options=(
                    channels.bond_growth or BondGrowthOptions()
                ),
            ),
            thermochemistry=KMCThermochemistry(
                free_energy_options=(
                    context.thermo.options if cfg.free_energy.enabled else None
                ),
                vib_cache_root=context.thermo.vibration_cache_root,
                calculation_cache_root=context.thermo.calculation_cache_root,
                calculation_cache_lookup_enabled=(
                    context.thermo.calculation_cache_lookup_enabled
                ),
            ),
            observers=KMCObservers(
                reaction_writer=sinks.reactions,
                trajectory_writer=sinks.trajectory,
                summary_collector=sinks.summary,
                checkpoint_writer=sinks.checkpoint,
            ),
            resume=resume,
            rng=settings.random_seed,
            telemetry=telemetry,
        )
        uses_legacy_adapter = (
            engine_module.run_kmc_steps
            is not engine_module._RUN_KMC_STEPS_COMPAT_ADAPTER
        )
        if uses_legacy_adapter:
            # Compatibility-only path for integrations that explicitly
            # replace the historical entry point. Normal configured runs
            # cross the public typed ``run_kmc(request)`` boundary below.
            summary = engine_module.run_kmc_steps(
                context.graph,
                network.initial_adsorbate_sites,
                context.calculator,
                reactants=network.reactants,
                temperature=settings.temperature_k,
                n_steps=settings.n_steps,
                transmission_coefficient=settings.transmission_coefficient,
                frozen_indices=context.frozen_indices,
                fmax=channels.adsorption.fmax,
                max_steps=channels.adsorption.max_steps,
                rng=settings.random_seed,
                log_every=settings.log_every,
                verbose=verbose,
                lateral_interactions=settings.lateral_interactions,
                lateral_shells=cfg.constants.lateral_shells,
                diffusion_sites=network.diffusion_sites,
                diffusion_kwargs=(
                    channels.diffusion.to_kwargs()
                    if channels.diffusion is not None
                    else None
                ),
                bond_sites=(
                    network.bond_sites
                    if identity.is_resume or cfg.bond.enabled
                    else None
                ),
                bond_kwargs=(
                    channels.bond.to_kwargs()
                    if channels.bond is not None
                    else None
                ),
                bond_growth_kwargs=(
                    channels.bond_growth.to_kwargs()
                    if channels.bond_growth is not None
                    else None
                ),
                free_energy_options=(
                    context.thermo.options
                    if cfg.free_energy.enabled
                    else None
                ),
                vib_cache_root=context.thermo.vibration_cache_root,
                calculation_cache_root=context.thermo.calculation_cache_root,
                calculation_cache_lookup_enabled=(
                    context.thermo.calculation_cache_lookup_enabled
                ),
                reaction_writer=sinks.reactions,
                trajectory_writer=sinks.trajectory,
                summary_collector=sinks.summary,
                checkpoint_writer=sinks.checkpoint,
                initial_step=resume.step,
                initial_time_s=resume.time_s,
                initial_history=cast(list[Any], resume.history),
                initial_reaction_counts=dict(resume.reaction_counts),
                initial_rng_state=(
                    dict(resume.rng_state)
                    if resume.rng_state is not None
                    else None
                ),
            )
        else:
            result = engine_module.run_kmc(request)
            summary = result.to_legacy_dict()

        # Finalize owned event and trajectory sinks before snapshotting output
        # telemetry. OutputSinks.close() is idempotent for failure cleanup.
        sinks.close()

        isaac_export_path = None
        if cfg.output.isaac_export_enabled:
            export_started = perf_counter()
            if telemetry is not None:
                telemetry.increment("output.isaac_export.calls")
            try:
                isaac_export_path = write_isaac_export(
                    context.thermo.calculation_cache_root,
                    identity.output_dir / cfg.output.isaac_export_filename,
                )
            finally:
                if telemetry is not None:
                    telemetry.add_time(
                        "output.isaac_export.seconds",
                        perf_counter() - export_started,
                    )

        termination_status, termination_reason = _termination(
            summary,
            n_steps_requested=settings.n_steps,
        )
        steps_executed = int(summary.get("steps_executed", 0) or 0)
        simulated_time_s = float(
            summary.get("simulated_time_s", summary.get("time", resume.time_s))
            or 0.0
        )
        performance_path = (
            identity.output_dir / PERFORMANCE_DIAGNOSTICS_RELATIVE_PATH
        )
        output_timings: dict[str, float] = {}

        def record_output_timing(name: str, started: float) -> None:
            elapsed = max(0.0, perf_counter() - started)
            if telemetry is not None:
                telemetry.add_time(name, elapsed)
            else:
                output_timings[name] = output_timings.get(name, 0.0) + elapsed

        def telemetry_snapshot() -> dict[str, Any]:
            if telemetry is not None:
                return telemetry.to_dict()
            existing = summary.get("performance") or {}
            snapshot = {
                "counters": dict(existing.get("counters") or {}),
                "timings_s": dict(existing.get("timings_s") or {}),
                "gauges": dict(existing.get("gauges") or {}),
            }
            timings = snapshot["timings_s"]
            for name, seconds in output_timings.items():
                timings[name] = float(timings.get(name, 0.0)) + seconds
            return snapshot

        def elapsed_wall(finished_at: datetime) -> float:
            return max(
                0.0,
                (
                    perf_counter() - wall_started_s
                    if wall_started_s is not None
                    else (finished_at - started_at).total_seconds()
                ),
            )

        def run_metadata(
            performance: dict[str, Any],
            *,
            finished_at: datetime,
            wall_time_s: float,
        ) -> dict[str, Any]:
            return make_run_meta(
                config_path=config_path,
                temperature_k=settings.temperature_k,
                n_steps_requested=settings.n_steps,
                steps_executed=steps_executed,
                total_time_s=simulated_time_s,
                random_seed=settings.random_seed,
                started_at=started_at,
                finished_at=finished_at,
                extra={
                    "simulated_time_s": simulated_time_s,
                    "wall_time_s": wall_time_s,
                    "termination_status": termination_status,
                    "termination_reason": termination_reason,
                    "performance": performance,
                    "performance_diagnostics": str(performance_path),
                },
            )

        # Publish a measured first pass so summary/diagnostic serialization and
        # the potentially expensive artifact scan are part of final wall time
        # and output overhead. The terminal pass below replaces these files
        # atomically with one internally consistent final snapshot.
        provisional_finished_at = datetime.now(timezone.utc)
        provisional_wall_time_s = elapsed_wall(provisional_finished_at)
        provisional_telemetry = telemetry_snapshot()
        provisional_performance = build_performance_summary(
            provisional_telemetry,
            wall_time_s=provisional_wall_time_s,
            steps_executed=steps_executed,
        )
        output_started = perf_counter()
        write_performance_diagnostics(
            performance_path,
            run_id=identity.run_id,
            telemetry=provisional_telemetry,
            summary=provisional_performance,
            simulated_time_s=simulated_time_s,
            wall_time_s=provisional_wall_time_s,
            steps_executed=steps_executed,
            termination_status=termination_status,
            termination_reason=termination_reason,
        )
        record_output_timing(
            "output.performance_diagnostics.seconds",
            output_started,
        )
        output_started = perf_counter()
        summary_path = sinks.summary.write(
            identity.output_dir / cfg.output.summary_filename,
            run_meta=run_metadata(
                provisional_performance,
                finished_at=provisional_finished_at,
                wall_time_s=provisional_wall_time_s,
            ),
            final_occupancy=summary.get("final_occupancy"),
        )
        record_output_timing("output.summary.seconds", output_started)
        outputs = _output_paths(
            cfg,
            identity,
            context,
            sinks,
            summary_path=summary_path,
            performance_path=performance_path,
            isaac_export_path=isaac_export_path,
        )
        summary["outputs"] = outputs

        invalid_counts = _invalid_lateral_counts(network)
        invalid_counts["persisted_invalid_reactions"] = (
            sinks.reactions.n_invalid_reactions
        )
        quarantine_locations = discover_quarantine_locations(identity.output_dir)
        if invalid_counts["total"]:
            update_run_manifest(
                identity.manifest_path,
                warning=(
                    f"{invalid_counts['total']} invalid lateral class(es) "
                    "remain recorded in the prepared network."
                ),
            )
        if quarantine_locations:
            update_run_manifest(
                identity.manifest_path,
                warning=(
                    f"{len(quarantine_locations)} uncommitted reaction "
                    "folder(s) were quarantined during resume."
                ),
            )
        descriptors = _artifact_descriptors(
            cfg,
            identity,
            context,
            sinks,
            summary_path=summary_path,
            performance_path=performance_path,
            isaac_export_path=isaac_export_path,
        )
        output_started = perf_counter()
        artifacts = build_artifact_inventory(
            identity.output_dir,
            descriptors,
        )
        record_output_timing("output.artifact_inventory.seconds", output_started)

        # This is the terminal measurement boundary: all scientific work and
        # one complete serialization/inventory pass are included. Only the
        # atomic publication of these measured values follows.
        finished_at = datetime.now(timezone.utc)
        wall_time_s = elapsed_wall(finished_at)
        raw_telemetry = telemetry_snapshot()
        performance = build_performance_summary(
            raw_telemetry,
            wall_time_s=wall_time_s,
            steps_executed=steps_executed,
        )
        write_performance_diagnostics(
            performance_path,
            run_id=identity.run_id,
            telemetry=raw_telemetry,
            summary=performance,
            simulated_time_s=simulated_time_s,
            wall_time_s=wall_time_s,
            steps_executed=steps_executed,
            termination_status=termination_status,
            termination_reason=termination_reason,
        )
        summary.update(
            {
                "time": simulated_time_s,
                "simulated_time_s": simulated_time_s,
                "wall_time_s": wall_time_s,
                "termination_status": termination_status,
                "termination_reason": termination_reason,
                "performance": performance,
            }
        )
        summary_path = sinks.summary.write(
            identity.output_dir / cfg.output.summary_filename,
            run_meta=run_metadata(
                performance,
                finished_at=finished_at,
                wall_time_s=wall_time_s,
            ),
            final_occupancy=summary.get("final_occupancy"),
        )
        # Refresh only the two files changed by terminal publication; the
        # expensive canonical inventory was already measured above.
        refreshed = build_artifact_inventory(
            identity.output_dir,
            {
                "performance_diagnostics": descriptors[
                    "performance_diagnostics"
                ],
                "summary": descriptors["summary"],
            },
        )
        artifacts.update(refreshed)
        finish_run_manifest(
            identity.manifest_path,
            final_step=resume.step + steps_executed,
            final_time_s=simulated_time_s,
            steps_executed=steps_executed,
            status=termination_status,
            termination_reason=termination_reason,
            wall_time_s=wall_time_s,
            outputs=outputs,
            artifacts=artifacts,
            invalid_counts=invalid_counts,
            quarantine_locations=quarantine_locations,
            performance=performance,
        )
        return summary
    except BaseException as exc:
        try:
            if sinks is not None:
                sinks.close()
        finally:
            _best_effort_failure(
                identity.manifest_path,
                exc,
                cfg=cfg,
                network=network,
                sinks=sinks,
                telemetry=telemetry,
            )
        raise
    finally:
        if sinks is not None:
            sinks.close()


__all__ = ["execute_kmc_stage"]
