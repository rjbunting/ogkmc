"""Config-driven pipeline orchestration for ogkmc."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ogkmc.io.config import FreeEnergyCfg, ReactantCfg, RunConfig
from ogkmc.io.calculators import CalculatorPool
from ogkmc.io.run_manifest import (
    begin_run_manifest,
    fail_run_manifest,
    update_run_manifest,
)
from ogkmc.workflow.network import (
    SpeciesNetworkBuilder,
    derive_configured_bond_templates,
)
from ogkmc.workflow.models import (
    PreparedCalculator,
    RunIdentity,
    SimulationContext,
)
from ogkmc.workflow.runtime import (
    configured_run_lock,
    ensure_run_output_available,
    resolve_channel_runtime,
    resolve_kmc_resume,
    resolve_run_identity,
    resolve_thermo_runtime,
)
from ogkmc.workflow.simulation import execute_kmc_stage
from ogkmc.workflow.stages import (
    prepare_adsorbate_sites,
    prepare_calculator,
    prepare_material_graph,
    prepare_reactants,
    prepare_structure,
)
from ogkmc.utils.telemetry import RuntimeTelemetry, telemetry_context


_log = logging.getLogger(__name__)


def _verbose_enabled(log_level: int) -> bool:
    return log_level <= logging.DEBUG


def _progress_enabled(log_level: int) -> bool:
    return log_level <= logging.INFO


def _stage(message: str, *, verbose: bool) -> None:
    if verbose:
        print(f"\n[ogkmc] {message}")


def _shutdown_calculator_resource(resource: object) -> None:
    """Release a pipeline-owned calculator pool without masking run results."""
    if not isinstance(resource, CalculatorPool):
        return
    try:
        resource.shutdown()
    except Exception:
        # KMC output finalization publishes the terminal manifest before control
        # returns here.  Cleanup failures must therefore remain operational
        # diagnostics rather than rewriting a scientifically complete run as
        # failed (or masking the original stage exception during unwinding).
        _log.exception("Could not shut down the calculator worker pool")


def _derive_configured_bond_templates(reactant_configs, reactants, bond_cfg):
    """Compatibility wrapper for the workflow-level template builder."""
    return derive_configured_bond_templates(
        reactant_configs,
        reactants,
        bond_cfg,
    )


def _resolved_partial_pressure_bar(
    reactant_cfg: ReactantCfg,
    free_energy_cfg: FreeEnergyCfg,
) -> float:
    """Resolve a reactant pressure, falling back to the feed-wide default."""
    if reactant_cfg.partial_pressure_bar is not None:
        return float(reactant_cfg.partial_pressure_bar)
    return float(free_energy_cfg.pressure_bar)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def run_from_config(cfg: RunConfig, *, config_path: str | None = None) -> dict:
    """Run the ogkmc pipeline end-to-end from a :class:`RunConfig`.

    Returns the KMC summary dict (same shape as
    :func:`ogkmc.kmc.engine.run_kmc_steps`) augmented with an
    ``outputs`` key listing the files written.
    """
    with configured_run_lock(cfg.output.dir):
        ensure_run_output_available(cfg)
        telemetry = RuntimeTelemetry()
        with telemetry_context(telemetry):
            return _run_from_config(
                cfg, config_path=config_path, telemetry=telemetry,
            )


@contextmanager
def _record_started_run_failure(cfg, telemetry):
    """Publish failures only after this invocation owns a manifest segment."""
    try:
        yield
    except BaseException as exc:
        gauges = telemetry.to_dict().get("gauges", {})
        try:
            fail_run_manifest(
                Path(cfg.output.dir) / cfg.output.run_manifest_filename,
                exc,
                status=(
                    "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, InterruptedError))
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
            )
        except Exception:
            _log.exception(
                "Could not publish terminal failure state to run manifest"
            )
        raise


def _run_from_config(
    cfg: RunConfig,
    *,
    config_path: str | None,
    telemetry: RuntimeTelemetry,
) -> dict:
    """Execute the configured stages inside one shared telemetry context."""
    wall_started_s = perf_counter()
    identity = resolve_run_identity(cfg)
    resume_state = identity.resume_state

    log_level = getattr(logging, str(cfg.output.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    verbose_run = _verbose_enabled(log_level)
    progress_run = _progress_enabled(log_level)

    started_at = datetime.now(timezone.utc)
    initial_step = 0 if resume_state is None else int(resume_state.step)
    initial_time_s = 0.0 if resume_state is None else float(resume_state.time_s)
    begin_run_manifest(
        identity.manifest_path,
        run_id=identity.run_id,
        config_path=config_path,
        resolved_config=asdict(cfg),
        events_filename=cfg.output.reactions_filename,
        initial_step=initial_step,
        initial_time_s=initial_time_s,
        is_resume=identity.is_resume,
    )

    with _record_started_run_failure(cfg, telemetry):
        # First, prepare the calculator used by every later stage.
        update_run_manifest(
            identity.manifest_path,
            status="preparing",
            current_stage="stage_1_calculator",
        )
        _stage("Stage 1/7: preparing calculator", verbose=progress_run)
        calculators = prepare_calculator(cfg, verbose=verbose_run)
        calc_resource = calculators.resource

        try:
            return _run_after_calculator_preparation(
                cfg,
                config_path=config_path,
                telemetry=telemetry,
                wall_started_s=wall_started_s,
                identity=identity,
                started_at=started_at,
                verbose_run=verbose_run,
                progress_run=progress_run,
                calculators=calculators,
            )
        finally:
            # The pipeline owns the calculator pool after Stage 1. Keep it open
            # through every preparation stage, and release it when the run ends.
            _shutdown_calculator_resource(calc_resource)


def _run_after_calculator_preparation(
    cfg: RunConfig,
    *,
    config_path: str | None,
    telemetry: RuntimeTelemetry,
    wall_started_s: float,
    identity: RunIdentity,
    started_at: datetime,
    verbose_run: bool,
    progress_run: bool,
    calculators: PreparedCalculator,
) -> dict:
    """Execute stages 2-7 while the caller owns the calculator lifetime."""
    resume_state = identity.resume_state
    calc_resource = calculators.resource

    # Next, build or restore the catalyst structure.
    update_run_manifest(
        identity.manifest_path,
        current_stage="stage_2_structure",
    )
    s = cfg.structure
    if resume_state is None:
        structure_action = (
            "loading catalyst structure"
            if s.kind == "file"
            else f"building {s.kind} structure"
        )
        _stage(f"Stage 2/7: {structure_action}", verbose=progress_run)
    else:
        _stage(
            "Stage 2/7: restoring structure and graph from checkpoint",
            verbose=progress_run,
        )
    structure = prepare_structure(
        cfg,
        identity,
        calculators,
        config_path=config_path,
        verbose=verbose_run,
    )
    update_run_manifest(
        identity.manifest_path,
        current_stage="stage_3_material_graph",
    )
    if resume_state is None:
        _stage(
            "Stage 3/7: classifying surface atoms and building graph",
            verbose=progress_run,
        )
    system = prepare_material_graph(
        cfg,
        identity,
        structure,
        verbose=verbose_run,
    )
    G = system.graph
    frozen_indices = system.frozen_indices

    # After the material graph is ready, resolve the free-energy settings and
    # the persistent vibration-cache location.
    thermo_runtime = resolve_thermo_runtime(cfg, identity)

    # The next stage builds the gas-phase reactants.
    update_run_manifest(
        identity.manifest_path,
        current_stage="stage_4_reactants",
    )
    if resume_state is None:
        _stage(
            "Stage 4/7: building gas-phase reactants",
            verbose=progress_run,
        )
    reactants_built = prepare_reactants(
        cfg,
        identity,
        calc_resource,
        thermo_runtime,
        pressure_resolver=_resolved_partial_pressure_bar,
        verbose=verbose_run,
    )

    # With the reactants built, enumerate and prune their adsorbate sites.
    update_run_manifest(
        identity.manifest_path,
        current_stage="stage_5_adsorbate_sites",
    )
    if resume_state is None:
        _stage(
            "Stage 5/7: enumerating and pruning adsorbate sites",
            verbose=progress_run,
        )
    all_sites = prepare_adsorbate_sites(
        cfg,
        identity,
        G,
        reactants_built,
        calc_resource,
        frozen_indices,
        verbose=verbose_run,
    )

    # The surface sites define the optional reaction channels and runtime
    # network prepared in this stage.
    update_run_manifest(
        identity.manifest_path,
        current_stage="stage_6_reaction_network",
    )
    _stage(
        "Stage 6/7: preparing optional reaction channels and outputs",
        verbose=progress_run,
    )
    channel_runtime = resolve_channel_runtime(cfg, frozen_indices)
    network = SpeciesNetworkBuilder(
        cfg=cfg,
        identity=identity,
        graph=G,
        calculator_resource=calc_resource,
        frozen_indices=frozen_indices,
        thermo_runtime=thermo_runtime,
        template_builder=_derive_configured_bond_templates,
        verbose=verbose_run,
    ).prepare(reactants_built, all_sites)

    resume = resolve_kmc_resume(identity)
    if cfg.checkpoint.resume_from:
        if resume_state is None:  # defensive; resume_from guarantees this above
            raise RuntimeError("checkpoint resume state was not loaded")
        if progress_run:
            print(
                f"[ogkmc] Resuming from checkpoint "
                f"{cfg.checkpoint.resume_from}: step={resume.step}, "
                f"t={resume.time_s:.4e} s"
            )

    # Finally, run KMC and finalize the managed outputs.
    update_run_manifest(
        identity.manifest_path,
        status="running",
        current_stage="stage_7_kmc",
    )
    _stage("Stage 7/7: starting KMC simulation", verbose=progress_run)
    context = SimulationContext(
        graph=G,
        network=network,
        calculator=calc_resource,
        frozen_indices=frozen_indices,
        thermo=thermo_runtime,
        channels=channel_runtime,
        structure_source=system.structure_source,
    )
    return execute_kmc_stage(
        cfg,
        identity,
        context,
        started_at=started_at,
        config_path=config_path,
        verbose=verbose_run,
        progress=progress_run,
        telemetry=telemetry,
        wall_started_s=wall_started_s,
    )


__all__ = ["run_from_config"]
