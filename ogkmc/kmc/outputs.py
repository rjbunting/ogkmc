"""Persistence, trajectory, summary, and checkpoint handling for KMC."""

from __future__ import annotations

from collections.abc import Iterable

from ogkmc.io.event_transitions import reaction_transition
from ogkmc.io.reaction_index import stable_event_id
from ogkmc.kmc.models import KMCChannels, KMCObservers, KMCRuntime, KMCSettings, KMCSystem
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import BondReactionSite
from ogkmc.kmc.restart import capture_rng_state, reactants_for_checkpoint
from ogkmc.sites.diffusion import DiffusionSite


def persist_bare_neb_calculations(writer, channels: KMCChannels, *, step: int) -> None:
    """Flush internal reference calculations during normal and failed sweeps."""
    write_bare = getattr(writer, "write_bare_neb", None)
    if not callable(write_bare):
        return
    for kind, sites in (
        ("diffusion", channels.diffusion_sites),
        ("bond", channels.bond_sites),
    ):
        for site in sites:
            for lateral_class in site.lateral_classes:
                if (
                    getattr(lateral_class, "_seed_only", False)
                    and not getattr(lateral_class, "members", None)
                ):
                    write_bare(site, lateral_class, kind=kind, step=step)


class KMCOutputManager:
    """Keep optional output callbacks out of the scientific event loop."""

    def __init__(
        self,
        system: KMCSystem,
        settings: KMCSettings,
        channels: KMCChannels,
        observers: KMCObservers,
        runtime: KMCRuntime,
    ) -> None:
        self.system = system
        self.settings = settings
        self.channels = channels
        self.observers = observers
        self.runtime = runtime
        self._last_checkpoint_step: int | None = None
        self._last_event_id: str | None = None

    def persist_reactions(self, reactions: Iterable, *, step: int) -> None:
        writer = self.observers.reaction_writer
        collector = self.observers.summary_collector
        for reaction in reactions:
            if reaction is None:
                continue
            if writer is not None:
                writer.ensure_reaction(
                    reaction,
                    step=step,
                    gas_energies=self.runtime.gas_energies,
                    gas_free_energies=self.runtime.gas_free_energies,
                )
            discover = getattr(collector, "discover", None)
            if callable(discover):
                discover(reaction, valid=True)

    def persist_invalid_diffusion_sites(
        self,
        sites: Iterable[DiffusionSite],
        *,
        step: int,
    ) -> None:
        writer = self.observers.reaction_writer
        collector = self.observers.summary_collector
        for site in sites:
            for lateral_class in site.lateral_classes:
                seed_only = bool(
                    getattr(lateral_class, "_seed_only", False)
                    and not getattr(lateral_class, "members", None)
                )
                failed_numerically = bool(
                    getattr(lateral_class, "last_failure_reason", None)
                )
                composite = (
                    getattr(lateral_class, "direct_event_status", None)
                    == "composite"
                )
                if (
                    lateral_class.stable is False
                    or failed_numerically
                    or composite
                ) and not seed_only:
                    if writer is not None:
                        writer.write_invalid_diffusion(
                            site,
                            lateral_class,
                            step=step,
                        )
                    discover_invalid = getattr(
                        collector,
                        "discover_invalid_diffusion",
                        None,
                    )
                    if callable(discover_invalid):
                        discover_invalid(site, lateral_class)

    def persist_invalid_adsorption_sites(
        self,
        sites: Iterable[AdsorbateSite],
        *,
        step: int,
    ) -> None:
        """Persist failed occupied/unoccupied relaxations for diagnostics."""
        writer = self.observers.reaction_writer
        for site in sites:
            for lateral_class in site.lateral_classes:
                if lateral_class.stable is False:
                    write_invalid = getattr(
                        writer,
                        "write_invalid_adsorption",
                        None,
                    )
                    if callable(write_invalid):
                        write_invalid(site, lateral_class, step=step)

    def persist_invalid_bond_sites(
        self,
        sites: Iterable[BondReactionSite],
        *,
        step: int,
    ) -> None:
        """Persist failed bond endpoint/NEB calculations for diagnostics."""
        writer = self.observers.reaction_writer
        collector = self.observers.summary_collector
        for site in sites:
            for lateral_class in site.lateral_classes:
                seed_only = bool(
                    getattr(lateral_class, "_seed_only", False)
                    and not getattr(lateral_class, "members", None)
                )
                failed_numerically = bool(
                    getattr(lateral_class, "last_failure_reason", None)
                )
                composite = (
                    getattr(lateral_class, "direct_event_status", None)
                    == "composite"
                )
                if (
                    lateral_class.stable is False
                    or failed_numerically
                    or composite
                ) and not seed_only:
                    write_invalid = getattr(writer, "write_invalid_bond", None)
                    if callable(write_invalid):
                        write_invalid(site, lateral_class, step=step)
                    discover_invalid = getattr(
                        collector,
                        "discover_invalid_bond",
                        None,
                    )
                    if callable(discover_invalid):
                        discover_invalid(site, lateral_class)

    def initialise(self) -> None:
        """Materialise discovered reactions and the optional initial frame."""
        collector = self.observers.summary_collector
        set_run_id = getattr(collector, "set_run_id", None)
        if callable(set_run_id):
            reaction_writer = self.observers.reaction_writer
            set_run_id(
                self.system.graph.graph.get("run_id")
                or getattr(reaction_writer, "run_id", None)
            )
        self.persist_reactions(
            (
                reaction
                for reaction in self.runtime.reaction_index.reactions
                if reaction is not None
            ),
            step=self.runtime.start_step,
        )
        self.persist_invalid_diffusion_sites(
            self.channels.diffusion_sites,
            step=self.runtime.start_step,
        )
        self.persist_invalid_adsorption_sites(
            self.system.adsorbate_sites,
            step=self.runtime.start_step,
        )
        self.persist_invalid_bond_sites(
            self.channels.bond_sites,
            step=self.runtime.start_step,
        )
        persist_bare_neb_calculations(
            self.observers.reaction_writer,
            self.channels,
            step=self.runtime.start_step,
        )

        trajectory_writer = self.observers.trajectory_writer
        if (
            trajectory_writer is not None
            and not getattr(trajectory_writer, "append", False)
        ):
            self._write_trajectory(
                step=self.runtime.start_step,
                frame_kind="initial",
            )

    def _write_trajectory(
        self,
        *,
        step: int,
        frame_kind: str | None = None,
        force: bool = False,
    ) -> None:
        """Write one due frame without eagerly materialising skipped snapshots."""
        writer = self.observers.trajectory_writer
        if writer is None:
            return

        should_write = getattr(writer, "should_write", None)
        if not force and callable(should_write) and not should_write(step=step):
            return

        from ogkmc.io.atoms import atoms_from_graph

        metadata = {
            "run_id": self.system.graph.graph.get("run_id"),
            "simulated_time_s": float(self.runtime.current_time_s),
            "event_id": self._last_event_id,
            "frame_kind": (
                frame_kind
                or ("initial" if int(step) == self.runtime.start_step else "periodic")
            ),
            "segment_start_step": int(self.runtime.start_step),
        }
        metadata = {
            key: value for key, value in metadata.items() if value is not None
        }
        write_snapshot = getattr(
            writer,
            "write_final_snapshot" if force else "maybe_write_snapshot",
            None,
        )
        if callable(write_snapshot):
            kwargs: dict[str, object] = {"step": step}
            if getattr(writer, "supports_frame_metadata", False):
                kwargs["metadata"] = metadata
            write_snapshot(lambda: atoms_from_graph(self.system.graph), **kwargs)
            return
        # Compatibility path for third-party trajectory observers.
        if not force:
            writer.maybe_write(atoms_from_graph(self.system.graph), step=step)

    def capture_transition(self, reaction):
        """Capture pre-event state only when an event writer needs it."""
        if self.observers.reaction_writer is None:
            return None
        return reaction_transition(reaction)

    def record_event(
        self,
        reaction,
        *,
        step: int,
        time_s: float,
        tau_s: float,
        transition,
    ) -> None:
        writer = self.observers.reaction_writer
        record = None
        if writer is not None:
            record = writer.record(
                step=step,
                time_s=time_s,
                tau_s=tau_s,
                reaction=reaction,
                gas_energies=self.runtime.gas_energies,
                gas_free_energies=self.runtime.gas_free_energies,
                transition=transition,
            )
        collector = self.observers.summary_collector
        if collector is not None:
            add_event = getattr(collector, "add_event", None)
            if record is not None and callable(add_event):
                add_event(record.to_jsonable(include_static=False))
            else:
                collector.add(reaction, step=step)
        self._last_event_id = (
            getattr(record, "event_id", None)
            or stable_event_id(
                self.system.graph.graph.get("run_id"),
                int(step),
            )
        )

    def finish_step(
        self,
        *,
        step: int,
        reactions: Iterable,
        invalid_diffusion_sites: Iterable[DiffusionSite],
    ) -> None:
        """Persist newly discovered states, frame, and periodic checkpoint."""
        self.persist_reactions(reactions, step=step)
        self.persist_invalid_diffusion_sites(
            invalid_diffusion_sites,
            step=step,
        )
        self.persist_invalid_adsorption_sites(
            self.system.adsorbate_sites,
            step=step,
        )
        self.persist_invalid_bond_sites(
            self.channels.bond_sites,
            step=step,
        )
        persist_bare_neb_calculations(
            self.observers.reaction_writer,
            self.channels,
            step=step,
        )

        requested_final_step = self.runtime.start_step + int(self.settings.n_steps)
        self._write_trajectory(
            step=step,
            frame_kind=("final" if int(step) == requested_final_step else "periodic"),
        )
        self.write_checkpoint(step=step)

    def write_checkpoint(self, *, step: int, force: bool = False) -> None:
        writer = self.observers.checkpoint_writer
        if writer is None:
            return
        should_write = getattr(writer, "should_write", None)
        if callable(should_write) and not should_write(step=step, force=force):
            return

        committed_event_count = None
        committed_event_offset = None

        # A checkpoint is only authoritative after every output derived from
        # its state has reached durable storage.
        trajectory = self.observers.trajectory_writer
        sync_trajectory = getattr(trajectory, "sync_for_checkpoint", None)
        if callable(sync_trajectory):
            sync_trajectory()
        reaction_writer = self.observers.reaction_writer
        sync_events = getattr(reaction_writer, "sync_for_checkpoint", None)
        if callable(sync_events):
            commit = sync_events()
            committed_event_count = int(commit.count)
            committed_event_offset = int(commit.offset)
            self._mark_history_committed(commit)
        payload = dict(
            step=step,
            time_s=self.runtime.current_time_s,
            graph=self.system.graph,
            adsorbate_sites=self.system.adsorbate_sites,
            diffusion_sites=self.channels.diffusion_sites,
            bond_sites=self.channels.bond_sites,
            reactants=reactants_for_checkpoint(
                self.system.reactants,
                self.system.graph,
            ),
            frozen_indices=self.settings.frozen_indices,
            reaction_counts=self.runtime.reaction_counts,
            rng_state=capture_rng_state(self.runtime.rng),
        )
        if committed_event_count is not None:
            payload["committed_event_count"] = committed_event_count
            payload["committed_event_offset"] = committed_event_offset
        else:
            # Standalone API users may provide checkpoints without an event
            # writer. In that compatibility mode there is no reconstructible
            # source of truth, so retaining history is required for correctness.
            payload["history"] = list(self.runtime.history)
        trajectory_offset = getattr(trajectory, "committed_offset", None)
        if isinstance(trajectory_offset, int):
            payload["committed_trajectory_offset"] = trajectory_offset
        if force:
            result = writer.maybe_write(force=True, **payload)
        else:
            result = writer.maybe_write(**payload)
        # The built-in writer exposes should_write and may return a Path.
        # Supporting either convention keeps lightweight observer fakes and
        # third-party writers compatible.
        if result is not None or callable(should_write):
            self._last_checkpoint_step = int(step)

    def close(self) -> None:
        """Commit final persistence state and close the trajectory."""
        trajectory = self.observers.trajectory_writer
        final_step = self.runtime.start_step + self.runtime.steps_executed
        try:
            self._write_trajectory(
                step=final_step,
                frame_kind="final",
                force=True,
            )
            if (
                self.runtime.steps_executed > 0
                and self.observers.checkpoint_writer is not None
            ):
                if self._last_checkpoint_step != final_step:
                    self.write_checkpoint(step=final_step, force=True)
            else:
                reaction_writer = self.observers.reaction_writer
                sync_events = getattr(reaction_writer, "sync_for_checkpoint", None)
                if callable(sync_events):
                    self._mark_history_committed(sync_events())
        finally:
            if trajectory is not None:
                trajectory.close()

    def _mark_history_committed(self, commit) -> None:
        """Release a lazy history suffix once its event rows are durable."""
        mark_committed = getattr(self.runtime.history, "mark_committed", None)
        if callable(mark_committed):
            mark_committed(
                count=int(commit.count),
                offset=int(commit.offset),
            )


__all__ = ["KMCOutputManager"]
