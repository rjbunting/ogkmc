"""Configuration-to-runtime translation with no scientific enumeration."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import logging
import os
import socket
import threading
import uuid
from pathlib import Path

from autokmc.io.calculation_cache import initialise_calculation_database
from autokmc.io.checkpoint import CheckpointWriter, load_checkpoint
from autokmc.io.event_log import reconcile_event_log
from autokmc.io.persistence import ReactionWriter
from autokmc.io.resume_contract import make_resume_contract, verify_resume_contract
from autokmc.io.summary import ReactionSummary
from autokmc.io.trajectory import TrajectoryWriter
from autokmc.thermo.free_energy import FreeEnergyOptions
from autokmc.workflow.models import (
    ChannelRuntimeOptions,
    KMCResumeState,
    OutputSinks,
    RunIdentity,
    ThermoRuntime,
)
from autokmc.kmc.models import (
    AdsorptionChannelOptions,
    BondChannelOptions,
    BondGrowthOptions,
    DiffusionChannelOptions,
)


_log = logging.getLogger(__name__)
RUN_LOCK_FILENAME = ".autokmc-run.lock"
_ACTIVE_RUN_LOCKS: set[str] = set()
_ACTIVE_RUN_LOCKS_GUARD = threading.Lock()


class OutputCollisionError(RuntimeError):
    """Raised when a fresh run would overwrite managed AutoKMC artifacts."""


class RunLockError(RuntimeError):
    """Raised when another configured run already owns an output directory."""


def managed_output_paths(cfg) -> tuple[Path, ...]:
    """Return paths owned by a configured run without touching the filesystem."""
    output_dir = Path(cfg.output.dir)
    relative_paths = (
        cfg.output.reactions_filename,
        cfg.output.summary_filename,
        cfg.output.run_manifest_filename,
        cfg.output.trajectory_filename,
        cfg.output.calculation_cache_dir,
        cfg.output.isaac_export_filename,
        "reactions",
        "uncommitted_reactions",
        "vib_cache",
        "analysis",
        "diagnostics",
        "checkpoint.pkl",
    )
    paths = {output_dir / str(relative) for relative in relative_paths}
    if cfg.checkpoint.enabled and cfg.checkpoint.path:
        paths.add(Path(cfg.checkpoint.path))
    return tuple(sorted(paths, key=lambda path: str(path)))


def managed_output_collisions(cfg) -> tuple[Path, ...]:
    """Return existing managed paths that make a fresh run unsafe."""
    if cfg.checkpoint.resume_from:
        return ()
    return tuple(path for path in managed_output_paths(cfg) if path.exists())


def ensure_run_output_available(cfg) -> None:
    """Refuse a fresh run that would mix with or truncate managed artifacts."""
    collisions = managed_output_collisions(cfg)
    if not collisions:
        return
    rendered = "\n".join(f"  - {path}" for path in collisions)
    raise OutputCollisionError(
        "fresh run refused because output artifacts already exist:\n"
        f"{rendered}\n"
        "Choose a new output.dir, or set checkpoint.resume_from to continue "
        "the existing run."
    )


def _acquire_os_file_lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        getattr(msvcrt, "locking")(
            handle.fileno(),
            getattr(msvcrt, "LK_NBLCK"),
            1,
        )
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_file_lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(
            handle.fileno(),
            getattr(msvcrt, "LK_UNLCK"),
            1,
        )
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_is_busy(exc: OSError) -> bool:
    return (
        isinstance(exc, BlockingIOError)
        or exc.errno in {errno.EACCES, errno.EAGAIN}
        or getattr(exc, "winerror", None) in {33, 36}
    )


def active_run_lock_owner(output_dir: str | Path) -> str | None:
    """Return lock metadata when another run is active, without creating files."""
    lock_path = Path(output_dir) / RUN_LOCK_FILENAME
    lock_key = str(lock_path.resolve())
    with _ACTIVE_RUN_LOCKS_GUARD:
        if lock_key in _ACTIVE_RUN_LOCKS:
            try:
                return lock_path.read_text(encoding="utf-8").strip() or "this process"
            except OSError:
                return "this process"
    if not lock_path.is_file():
        return None
    try:
        with lock_path.open("r+", encoding="utf-8") as handle:
            try:
                _acquire_os_file_lock(handle)
            except OSError as exc:
                if not _lock_is_busy(exc):
                    raise
                handle.seek(0)
                return handle.read().strip() or "another process"
            else:
                _release_os_file_lock(handle)
                return None
    except OSError:
        # An unreadable lock file is not proof of a live owner. The configured
        # run will surface the concrete filesystem error if execution proceeds.
        return None


@contextmanager
def configured_run_lock(output_dir: str | Path):
    """Hold an exclusive per-output lock for one complete configured run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / RUN_LOCK_FILENAME
    lock_key = str(lock_path.resolve())

    with _ACTIVE_RUN_LOCKS_GUARD:
        if lock_key in _ACTIVE_RUN_LOCKS:
            raise RunLockError(
                f"another AutoKMC run in this process already owns {directory}"
            )
        _ACTIVE_RUN_LOCKS.add(lock_key)

    handle = None
    acquired = False
    try:
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            _acquire_os_file_lock(handle)
        except OSError as exc:
            if not _lock_is_busy(exc):
                raise
            handle.seek(0)
            owner = handle.read().strip()
            detail = f" Lock owner: {owner}" if owner else ""
            raise RunLockError(
                f"another AutoKMC process is already using output directory "
                f"{directory}.{detail}"
            ) from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "output_dir": str(directory.resolve()),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        if handle is not None:
            try:
                if acquired:
                    _release_os_file_lock(handle)
            finally:
                handle.close()
        with _ACTIVE_RUN_LOCKS_GUARD:
            _ACTIVE_RUN_LOCKS.discard(lock_key)


def resolve_run_identity(cfg) -> RunIdentity:
    """Resolve output paths, checkpoint state, and the shared run UUID."""
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / cfg.output.run_manifest_filename
    resume_state = (
        load_checkpoint(cfg.checkpoint.resume_from)
        if cfg.checkpoint.resume_from
        else None
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
        manifest_run_id
        and checkpoint_run_id is not None
        and manifest_run_id != str(checkpoint_run_id)
    ):
        raise ValueError(
            "checkpoint run_id does not match the output run manifest: "
            f"{checkpoint_run_id!r} != {manifest_run_id!r}"
        )

    run_id = manifest_run_id or (
        str(checkpoint_run_id) if checkpoint_run_id is not None else None
    )
    resolved_run_id = run_id or str(uuid.uuid4())
    event_commit = None
    if resume_state is not None:
        stored_contract = resume_state.metadata.get("resume_contract")
        if stored_contract is None:
            _log.warning(
                "Checkpoint has no scientific resume fingerprint; accepting this "
                "legacy checkpoint with step-based event reconciliation."
            )
        elif not isinstance(stored_contract, dict):
            raise ValueError("checkpoint resume_contract metadata is malformed")
        else:
            verify_resume_contract(cfg, stored_contract)

        event_commit = reconcile_event_log(
            output_dir / cfg.output.reactions_filename,
            checkpoint_step=int(resume_state.step),
            committed_event_count=getattr(
                resume_state,
                "committed_event_count",
                None,
            ),
            committed_event_offset=getattr(
                resume_state,
                "committed_event_offset",
                None,
            ),
            run_id=resolved_run_id,
            legacy_history=getattr(resume_state, "history", None),
        )
    return RunIdentity(
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_id=resolved_run_id,
        resume_state=resume_state,
        event_commit=event_commit,
    )


def resolve_thermo_runtime(cfg, identity: RunIdentity) -> ThermoRuntime:
    """Translate config thermochemistry/cache fields once for all consumers."""
    fe_cfg = cfg.free_energy
    options = FreeEnergyOptions(
        enabled=fe_cfg.enabled,
        pressure_bar=fe_cfg.pressure_bar,
        vibration_displacement=fe_cfg.vibration_displacement,
        vibration_nfree=fe_cfg.vibration_nfree,
        include_ts_vibrations=fe_cfg.include_ts_vibrations,
        min_frequency_ev=fe_cfg.min_frequency_ev,
        symmetry_tolerance=fe_cfg.symmetry_tolerance,
        default_spin=fe_cfg.default_spin,
        default_geometry=fe_cfg.default_geometry,
        cache_dir=fe_cfg.cache_dir,
    )
    vibration_cache_root = (
        fe_cfg.cache_dir
        if fe_cfg.cache_dir
        else str(identity.output_dir / "vib_cache")
    )
    calculation_cache_root = (
        str(identity.output_dir / cfg.output.calculation_cache_dir)
        if cfg.output.calculation_cache_enabled
        else None
    )
    if calculation_cache_root is not None:
        initialise_calculation_database(
            calculation_cache_root,
            run_id=identity.run_id,
        )
    return ThermoRuntime(
        options=options,
        vibration_cache_root=vibration_cache_root,
        calculation_cache_root=calculation_cache_root,
        calculation_cache_lookup_enabled=bool(
            cfg.output.calculation_cache_lookup_enabled
        ),
    )


def resolve_channel_runtime(cfg, frozen_indices: list[int] | None) -> ChannelRuntimeOptions:
    """Build the typed reaction-channel option groups."""
    adsorption = AdsorptionChannelOptions(
        fmax=cfg.adsorption.endpoint_fmax,
        max_steps=cfg.adsorption.endpoint_max_steps,
    )
    diffusion = None
    if cfg.diffusion.enabled:
        d = cfg.diffusion
        diffusion = DiffusionChannelOptions(
            fmax=d.fmax,
            max_steps=d.max_steps,
            n_images=d.n_images,
            image_spacing=d.image_spacing,
            min_images=d.min_images,
            max_images=d.max_images,
            climb=d.climb,
            spring_k=d.spring_k,
            interpolation=d.interpolation,
            nl_mult=cfg.constants.neighbor_list_multiplier,
            persist_neb_path=d.persist_neb_path,
            optimizer=cfg.optimization.optimizer,
            optimizer_kwargs=cfg.optimization.optimizer_kwargs,
            neb_optimizer=cfg.optimization.neb_optimizer,
            neb_optimizer_kwargs=cfg.optimization.neb_optimizer_kwargs,
            neb_climb_optimizer=cfg.optimization.neb_climb_optimizer,
            neb_climb_optimizer_kwargs=(
                cfg.optimization.neb_climb_optimizer_kwargs
            ),
            neb_method=cfg.optimization.neb_method,
            neb_band_eval=cfg.optimization.neb_band_eval,
            neb_geometry_guard_multiplier=(
                cfg.optimization.neb_geometry_guard_multiplier
            ),
            neb_intermediate_stagnation_steps=(
                cfg.optimization.neb_intermediate_stagnation_steps
            ),
            neb_intermediate_energy_tolerance=(
                cfg.optimization.neb_intermediate_energy_tolerance
            ),
            neb_intermediate_minimum_prominence=(
                cfg.optimization.neb_intermediate_minimum_prominence
            ),
        )

    bond = None
    bond_growth = None
    if cfg.bond.enabled:
        b = cfg.bond
        d = cfg.diffusion
        bond = BondChannelOptions(
            fmax=b.neb_fmax,
            max_steps=b.neb_max_steps,
            n_images=b.neb_n_images,
            image_spacing=b.neb_image_spacing,
            min_images=b.neb_min_images,
            max_images=b.neb_max_images,
            climb=b.neb_climb,
            spring_k=b.neb_spring_k,
            interpolation=b.neb_interpolation,
            atom_matching=b.atom_matching,
            matching_trials=b.matching_trials,
            gas_precursor_relax=b.gas_precursor_relax,
            gas_precursor_distance=b.gas_precursor_distance,
            nl_mult=cfg.constants.neighbor_list_multiplier,
            persist_neb_path=b.persist_neb_path,
            optimizer=cfg.optimization.optimizer,
            optimizer_kwargs=cfg.optimization.optimizer_kwargs,
            neb_optimizer=cfg.optimization.neb_optimizer,
            neb_optimizer_kwargs=cfg.optimization.neb_optimizer_kwargs,
            neb_climb_optimizer=cfg.optimization.neb_climb_optimizer,
            neb_climb_optimizer_kwargs=(
                cfg.optimization.neb_climb_optimizer_kwargs
            ),
            neb_method=cfg.optimization.neb_method,
            neb_band_eval=cfg.optimization.neb_band_eval,
            neb_geometry_guard_multiplier=(
                cfg.optimization.neb_geometry_guard_multiplier
            ),
            neb_intermediate_stagnation_steps=(
                cfg.optimization.neb_intermediate_stagnation_steps
            ),
            neb_intermediate_energy_tolerance=(
                cfg.optimization.neb_intermediate_energy_tolerance
            ),
            neb_intermediate_minimum_prominence=(
                cfg.optimization.neb_intermediate_minimum_prominence
            ),
        )
        bond_growth = BondGrowthOptions(
            find_diffusion=d.enabled,
            frozen_indices=frozen_indices,
            bond_max_hops=b.bond_max_hops,
            nl_mult=cfg.constants.neighbor_list_multiplier,
            random_seed=cfg.kmc.random_seed,
            adsorbate_bond_tolerance=(
                cfg.constants.adsorbate_bond_tolerance
            ),
            adsorbate_n_shells_anchor=(
                cfg.adsorption.n_shells_anchor
            ),
            adsorbate_n_shells_pair=cfg.adsorption.pair_n_shells,
            co_bond_factor=cfg.constants.co_bond_factor,
            anchor_bond_factor=cfg.constants.anchor_bond_factor,
            anchor_repulsion_weight=(
                cfg.constants.anchor_repulsion_weight
            ),
            site_repulsion_cutoff=cfg.constants.site_repulsion_cutoff,
            adsorbate_contact_factor=(
                cfg.constants.adsorbate_contact_factor
            ),
            adsorbate_standoff_factor=(
                cfg.constants.adsorbate_standoff_factor
            ),
            adsorbate_rotational_restarts=(
                cfg.constants.adsorbate_rotational_restarts
            ),
            typical_neighbor_distance=(
                cfg.constants.typical_neighbor_distance
            ),
            adsorbate_max_pair_shells=(
                cfg.adsorption.max_pair_shells
            ),
            anchor_hull_tolerance=cfg.constants.anchor_hull_tolerance,
            kabsch_max_mappings=cfg.constants.kabsch_max_mappings,
            bond_pair_n_shells=b.pair_n_shells,
            bond_prune_by_triple=b.prune_by_triple,
            bond_prune_with_calculator=b.prune_with_calculator,
            adsorption_prune_fmax=cfg.adsorption.prune_fmax,
            adsorption_prune_max_steps=cfg.adsorption.prune_max_steps,
            bond_prune_fmax=b.prune_fmax,
            bond_prune_max_steps=b.prune_max_steps,
            anchor_k_max=cfg.adsorption.anchor_k_max,
            bond_types=tuple(b.bond_types),
            include_ring_bonds=b.include_ring_bonds,
            include_homo_coupling=b.include_homo_coupling,
            include_dissociation=b.include_dissociation,
            include_coupling=b.include_coupling,
            deduplicate_iso=b.deduplicate_iso,
            auto_build_leaf_species=b.auto_build_leaf_species,
            add_hydrogens=False,
            gas_lift_height=b.gas_lift_height,
            diffusion_max_hops=d.max_hops,
            diffusion_n_shells_pair=d.n_shells_pair,
            diffusion_prune_by_ads_pair=d.prune_by_adsorption_pair,
            optimizer=cfg.optimization.optimizer,
            optimizer_kwargs=cfg.optimization.optimizer_kwargs,
        )

    return ChannelRuntimeOptions(
        adsorption=adsorption,
        diffusion=diffusion,
        bond=bond,
        bond_growth=bond_growth,
    )


def create_output_sinks(
    cfg,
    identity: RunIdentity,
    calculator_metadata: dict,
    *,
    config_path=None,
) -> OutputSinks:
    """Create every output collaborator, cleaning up partial construction."""
    reactions = ReactionWriter(
        identity.output_dir,
        reactions_filename=cfg.output.reactions_filename,
        calculator_meta=calculator_metadata,
        append=identity.is_resume,
        run_id=identity.run_id,
        checkpoint_step=(
            int(identity.resume_state.step)
            if identity.resume_state is not None
            else None
        ),
        event_recovery=(
            identity.event_commit.recovery
            if identity.event_commit is not None
            else None
        ),
    )
    trajectory = None
    try:
        trajectory = TrajectoryWriter(
            identity.output_dir / cfg.output.trajectory_filename,
            dump_every=cfg.output.trajectory_dump_every,
            append=identity.is_resume,
            resume_checkpoint_step=(
                int(identity.resume_state.step)
                if identity.resume_state is not None
                else None
            ),
        )
        recovery = (
            identity.event_commit.recovery
            if identity.event_commit is not None
            else None
        )
        summary = ReactionSummary()
        if identity.is_resume:
            if recovery is not None and recovery.summary_complete:
                summary = recovery.summary
            else:
                summary = ReactionSummary.from_events(reactions.jsonl_path)
            for definition in getattr(reactions, "reaction_definitions", ()):
                summary.note_discovered(
                    str(definition["reaction_id"]),
                    valid=bool(definition.get("valid", True)),
                )
        checkpoint = None
        if cfg.checkpoint.enabled:
            checkpoint_path = cfg.checkpoint.path or str(
                identity.output_dir / "checkpoint.pkl"
            )
            checkpoint = CheckpointWriter(
                checkpoint_path,
                every_n_steps=cfg.checkpoint.every_n_steps,
                metadata={
                    "config_path": config_path,
                    "run_id": identity.run_id,
                    "resume_contract": make_resume_contract(cfg),
                },
            )
        return OutputSinks(
            reactions=reactions,
            trajectory=trajectory,
            summary=summary,
            checkpoint=checkpoint,
        )
    except BaseException:
        try:
            if trajectory is not None:
                trajectory.close()
        finally:
            reactions.close()
        raise


def resolve_kmc_resume(identity: RunIdentity) -> KMCResumeState:
    """Return normalized cumulative KMC state for fresh and resumed runs."""
    state = identity.resume_state
    if state is None:
        return KMCResumeState()
    history = state.history
    recovery = (
        identity.event_commit.recovery
        if identity.event_commit is not None
        else None
    )
    if recovery is not None and recovery.history_complete:
        # Schema-v4 checkpoints intentionally leave history empty; the exact
        # committed event prefix is already parsed during reconciliation.
        history = recovery.history
    return KMCResumeState(
        step=int(state.step),
        time_s=float(state.time_s),
        history=history,
        reaction_counts=dict(state.reaction_counts),
        rng_state=getattr(state, "rng_state", None),
    )


__all__ = [
    "OutputCollisionError",
    "RUN_LOCK_FILENAME",
    "RunLockError",
    "active_run_lock_owner",
    "configured_run_lock",
    "create_output_sinks",
    "ensure_run_output_available",
    "managed_output_collisions",
    "managed_output_paths",
    "resolve_channel_runtime",
    "resolve_kmc_resume",
    "resolve_run_identity",
    "resolve_thermo_runtime",
]
