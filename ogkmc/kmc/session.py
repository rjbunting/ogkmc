"""Typed KMC session orchestrator."""

from __future__ import annotations

from dataclasses import replace
import logging
from time import perf_counter

import numpy as np

from ogkmc.core.graph_state import get_n_occupied
from ogkmc.kmc.execute import execute_reaction
from ogkmc.kmc.initialization import initialise_runtime, normalise_channels
from ogkmc.kmc.models import (
    KMCFunctions,
    KMCRunRequest,
    KMCRunResult,
    KMCRuntime,
)
from ogkmc.kmc.network import DynamicNetworkExpander
from ogkmc.kmc.outputs import KMCOutputManager, persist_bare_neb_calculations
from ogkmc.kmc.restart import final_occupancy_by_species
from ogkmc.utils.telemetry import (
    RuntimeTelemetry,
    increment,
    set_gauge,
    telemetry_context,
    timed,
)


_log = logging.getLogger(__name__)


class KMCSession:
    """Own the lifecycle and mutable state of one BKL/Gillespie run."""

    def __init__(
        self,
        *,
        request: KMCRunRequest,
        functions: KMCFunctions,
    ) -> None:
        self.request = request
        self.system = request.system
        self.settings = request.settings
        self.channels, self.thermochemistry = normalise_channels(
            request.channels,
            request.thermochemistry,
        )
        self.observers = request.observers
        self.resume = request.resume
        self.functions = functions
        self.rng = request.rng
        self.telemetry = request.telemetry or RuntimeTelemetry()
        self.runtime: KMCRuntime | None = None

    def run(self) -> KMCRunResult:
        """Initialise channels, execute events, and return a typed result."""
        started = perf_counter()
        with telemetry_context(self.telemetry):
            increment("kmc.session.runs")
            with timed("kmc.session.seconds"):
                result = self._run_session()
        return replace(
            result,
            wall_time_s=max(0.0, perf_counter() - started),
            performance=self.telemetry.to_dict(),
        )

    def _run_session(self) -> KMCRunResult:
        """Execute the session while its telemetry context is installed."""
        try:
            runtime = initialise_runtime(
                self.system,
                self.settings,
                self.channels,
                self.thermochemistry,
                self.resume,
                rng=self.rng,
                compute_adsorption=self.functions.compute_adsorption,
            )
        except Exception as exc:
            self._persist_partial_outputs(
                step=int(self.resume.step),
                primary_error=exc,
            )
            raise
        self.runtime = runtime
        outputs = KMCOutputManager(
            self.system,
            self.settings,
            self.channels,
            self.observers,
            runtime,
        )
        expander = DynamicNetworkExpander(
            self.system,
            self.settings,
            self.channels,
            self.thermochemistry,
            self.functions,
            runtime,
        )
        outputs.initialise()

        termination_status = "complete"
        termination_reason = (
            "no_steps_requested"
            if int(self.settings.n_steps) == 0
            else "requested_steps_completed"
        )
        stop_step = runtime.start_step + int(self.settings.n_steps)
        for step in range(runtime.start_step + 1, stop_step + 1):
            try:
                stopped_reason = self._run_step(step, outputs, expander)
            except Exception as exc:
                self._persist_partial_outputs(
                    step=step,
                    primary_error=exc,
                )
                raise
            if stopped_reason is not None:
                termination_status = "stopped"
                termination_reason = stopped_reason
                break

        outputs.close()
        return self._summary(
            termination_status=termination_status,
            termination_reason=termination_reason,
        )

    def _persist_partial_outputs(
        self,
        *,
        step: int,
        primary_error: Exception,
    ) -> None:
        """Durably retain completed work and diagnostics before aborting.

        Initial reaction sweeps run before :class:`KMCOutputManager` exists,
        while recomputation deliberately publishes no event until all
        scientific updates succeed.  A failure in either phase must still
        preserve reactions that already completed and the last-known failed
        endpoint/NEB structures.
        """
        writer = self.observers.reaction_writer
        if writer is None:
            return
        try:
            n_completed = 0
            n_invalid = 0
            ensure_reaction = getattr(writer, "ensure_reaction", None)
            if callable(ensure_reaction):
                for site in (
                    *self.system.adsorbate_sites,
                    *self.channels.diffusion_sites,
                    *self.channels.bond_sites,
                ):
                    for reaction in (
                        getattr(site, "applicable_reactions", None) or []
                    ):
                        if reaction is not None:
                            ensure_reaction(
                                reaction,
                                step=int(step),
                            )
                            n_completed += 1

            for site in self.system.adsorbate_sites:
                write_invalid = getattr(
                    writer,
                    "write_invalid_adsorption",
                    None,
                )
                if not callable(write_invalid):
                    break
                for lateral_class in site.lateral_classes:
                    if lateral_class.stable is False:
                        write_invalid(site, lateral_class, step=int(step))
                        n_invalid += 1

            for method_name, sites in (
                ("write_invalid_diffusion", self.channels.diffusion_sites),
                ("write_invalid_bond", self.channels.bond_sites),
            ):
                write_invalid = getattr(writer, method_name, None)
                if not callable(write_invalid):
                    continue
                for site in sites:
                    for transition_lateral in site.lateral_classes:
                        seed_only = bool(
                            getattr(transition_lateral, "_seed_only", False)
                            and not getattr(transition_lateral, "members", None)
                        )
                        failed = (
                            transition_lateral.stable is False
                            or getattr(
                                transition_lateral,
                                "direct_event_status",
                                None,
                            )
                            == "composite"
                            or bool(
                                getattr(
                                    transition_lateral,
                                    "last_failure_reason",
                                    None,
                                )
                            )
                        )
                        if failed and not seed_only:
                            write_invalid(
                                site,
                                transition_lateral,
                                step=int(step),
                            )
                            n_invalid += 1

            persist_bare_neb_calculations(writer, self.channels, step=int(step))
            sync = getattr(writer, "sync_for_checkpoint", None)
            if callable(sync):
                sync()
            _log.info(
                "KMC partial-output flush: persisted %d completed reaction "
                "instance(s) and %d invalid/numerical lateral diagnostic(s) "
                "before re-raising %s",
                n_completed,
                n_invalid,
                type(primary_error).__name__,
            )
        except Exception as output_error:
            # BaseException.add_note was introduced in Python 3.11, while
            # OGKMC still supports Python 3.10.  Preserve the original
            # scientific failure even when the compatibility runtime cannot
            # attach the secondary persistence error.
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(
                    "OGKMC also failed while persisting partial reaction "
                    f"diagnostics: {type(output_error).__name__}: {output_error}"
                )

    def _run_step(
        self,
        step: int,
        outputs: KMCOutputManager,
        expander: DynamicNetworkExpander,
    ) -> str | None:
        runtime = self._runtime
        reaction_index = runtime.reaction_index
        total_rate = reaction_index.total_rate()
        if total_rate <= 0.0:
            if self.settings.progress_enabled:
                print(f"[KMC] Step {step}: total rate = 0 — stopping.")
            return "zero_total_rate"

        if isinstance(runtime.rng, np.random.Generator):
            selection_draw = float(runtime.rng.random())
            time_draw = float(runtime.rng.random())
        else:
            selection_draw = runtime.rng.random()
            time_draw = runtime.rng.random()
        if time_draw <= 0.0:
            time_draw = float(np.nextafter(0.0, 1.0))

        reaction = reaction_index.sample(selection_draw)
        if reaction is None:
            if self.settings.progress_enabled:
                print(f"[KMC] Step {step}: sampler returned None — stopping.")
            return "sampler_returned_none"

        tau_s = float(-np.log(time_draw) / total_rate)
        runtime.current_time_s += tau_s
        transition = outputs.capture_transition(reaction)
        affected_cliques = execute_reaction(self.system.graph, reaction)
        runtime.steps_executed += 1
        runtime.reaction_counts[reaction.kind] = (
            runtime.reaction_counts.get(reaction.kind, 0) + 1
        )
        if reaction.kind == "bond":
            subtype = "bond_" + getattr(reaction, "direction", "couple")
            runtime.reaction_counts[subtype] = runtime.reaction_counts.get(subtype, 0) + 1

        self._print_event(reaction, step, tau_s, total_rate)

        reactions, invalid_diffusion_sites = self.functions.recompute_affected(
            self.system.graph,
            self.system.adsorbate_sites,
            affected_cliques,
            self.system.calculator,
            runtime.gas_energies,
            temperature=self.settings.temperature,
            transmission_coefficient=self.settings.transmission_coefficient,
            frozen_indices=self.settings.frozen_indices,
            fmax=self.channels.adsorption_options.fmax,
            max_steps=self.channels.adsorption_options.max_steps,
            optimizer=self.settings.optimizer,
            optimizer_kwargs=self.settings.optimizer_kwargs,
            verbose=self.settings.verbose,
            max_n_shells=self.settings.max_n_shells,
            rxn_index=runtime.reaction_index,
            diffusion_sites=self.channels.diffusion_sites,
            diffusion_kwargs=self.channels.diffusion_options.to_kwargs(),
            bond_sites=self.channels.bond_sites,
            bond_kwargs=self.channels.bond_options.to_kwargs(),
            lateral_interactions=self.settings.lateral_interactions,
            lateral_shells=self.settings.lateral_shells,
            gas_g=runtime.gas_free_energies,
            partial_pressures=runtime.partial_pressures,
            free_energy_options=self.thermochemistry.free_energy_options,
            vib_cache_root=self.thermochemistry.vib_cache_root,
            calculation_cache_root=self.thermochemistry.calculation_cache_root,
            calculation_cache_lookup_enabled=bool(
                self.thermochemistry.calculation_cache_lookup_enabled
            ),
        )

        if reaction.kind == "bond":
            changes = expander.expand(reaction)
            reactions.extend(changes.reactions)
            invalid_diffusion_sites.extend(changes.invalid_diffusion_sites)

        # The event becomes externally visible only after every scientific
        # update needed for the next state has succeeded.  This prevents the
        # durable event stream from advancing past a resumable checkpoint when
        # recomputation or dynamic expansion raises.
        outputs.record_event(
            reaction,
            step=step,
            time_s=runtime.current_time_s,
            tau_s=tau_s,
            transition=transition,
        )
        runtime.history.append(
            (
                step,
                runtime.current_time_s,
                reaction.kind,
                reaction.site.iso_class,
                reaction.member_index,
                reaction.lateral_class.lateral_class,
                reaction.delta_e,
                reaction.barrier,
                reaction.rate,
            )
        )
        increment("kmc.events.committed")
        set_gauge("kmc.last_step", step)
        set_gauge("kmc.time_s", runtime.current_time_s)

        outputs.finish_step(
            step=step,
            reactions=reactions,
            invalid_diffusion_sites=invalid_diffusion_sites,
        )
        return None

    def _print_event(self, reaction, step: int, tau_s: float, total_rate: float) -> None:
        if not (
            self.settings.progress_enabled
            and self.settings.log_every
            and step % self.settings.log_every == 0
        ):
            return
        kind = reaction.kind
        if kind == "bond":
            direction = getattr(reaction, "direction", "")
            template = getattr(getattr(reaction, "site", None), "template", None)
            if template is not None and direction == "couple":
                label = (
                    "bond/couple  "
                    f"{template.smiles_a}+{template.smiles_b}→{template.smiles_c}"
                )
            elif template is not None:
                label = (
                    "bond/dissoc  "
                    f"{template.smiles_c}→{template.smiles_a}+{template.smiles_b}"
                )
            else:
                label = f"bond/{direction}"
        elif kind == "diffusion":
            label = "diffusion   "
        else:
            label = f"{kind:<11}"

        n_occupied = get_n_occupied(self.system.graph)
        print(
            f"[KMC] step {step:>5}  t = {self._runtime.current_time_s:.4e} s  "
            f"τ = {tau_s:.3e} s  Q = {total_rate:.3e} Hz  "
            f"{label}  iso={reaction.site.iso_class} "
            f"m={reaction.member_index} lat={reaction.lateral_class.lateral_class} "
            f"ΔE={reaction.delta_e:+.3f} eV  Ea={reaction.barrier:.3f} eV  "
            f"k={reaction.rate:.2e} Hz  occ={n_occupied}"
        )

    def _summary(
        self,
        *,
        termination_status: str,
        termination_reason: str,
    ) -> KMCRunResult:
        runtime = self._runtime
        final_occupancy = final_occupancy_by_species(self.system.adsorbate_sites)
        summary = KMCRunResult(
            time_s=runtime.current_time_s,
            steps_executed=runtime.steps_executed,
            history=runtime.history,
            reaction_counts=runtime.reaction_counts,
            final_occupancy=final_occupancy,
            termination_status=termination_status,
            termination_reason=termination_reason,
        )
        if self.settings.progress_enabled:
            print(
                f"\n[KMC] Done.  steps={runtime.steps_executed}  "
                f"t={runtime.current_time_s:.4e} s"
            )
            print(f"[KMC] Reaction counts: {runtime.reaction_counts}")
            print(f"[KMC] Final occupancy per species/iso-class: {final_occupancy}")
        return summary

    @property
    def _runtime(self) -> KMCRuntime:
        if self.runtime is None:  # pragma: no cover - internal lifecycle guard
            raise RuntimeError("KMC session has not been initialised")
        return self.runtime


__all__ = ["KMCSession"]
