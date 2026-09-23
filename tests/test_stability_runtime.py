"""Focused tests for shared stability/NEB execution boundaries."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import threading
from types import SimpleNamespace

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
import networkx as nx
import numpy as np
import pytest

from ogkmc.io.calculators import CalculatorConfigError, CalculatorPool
from ogkmc.io.calculation_cache import scientific_input_fingerprint
from ogkmc.sites.stability import adsorption as adsorption_module
from ogkmc.sites.stability import bond as bond_module
from ogkmc.sites.stability import diffusion as diffusion_module
from ogkmc.sites.stability import neb as neb_module
from ogkmc.sites.stability.bond import (
    BondTransitionStateInvalidError,
    _check_bond_ts_validity,
)
from ogkmc.sites.stability.diffusion import (
    TransitionStateInvalidError,
    _check_ts_validity,
)
from ogkmc.utils.telemetry import RuntimeTelemetry, telemetry_context


def _image(energy: float) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(energy),
        forces=np.zeros((len(atoms), 3)),
    )
    return atoms


@contextmanager
def _calculator_context(calculator, *, purpose):
    del purpose
    yield calculator


class _ConvergedOptimizer:
    def __init__(self, _neb, *, logfile):
        del logfile
        self.nsteps = 4

    def run(self, *, fmax, steps):
        del fmax, steps

    def converged(self) -> bool:
        return True


def test_shared_neb_selects_transition_and_detaches_images(monkeypatch):
    images = [_image(0.0), _image(0.5), _image(1.5), _image(0.2)]
    neb = SimpleNamespace(climb=None)
    optimizer_climb_states = []
    initial_paths = []

    class StageTrackingOptimizer(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile):
            optimizer_climb_states.append(stage_neb.climb)
            super().__init__(stage_neb, logfile=logfile)

    def band_factory(*_args, climb, spring_k, **_kwargs):
        assert climb is False
        assert spring_k == pytest.approx(5.0)
        neb.climb = climb
        return neb, images

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", StageTrackingOptimizer)
    telemetry = RuntimeTelemetry()

    with telemetry_context(telemetry):
        result = neb_module.run_neb(
            images[0],
            images[-1],
            calculator=object(),
            purpose="test NEB",
            n_images=2,
            interpolation="linear",
            spring_k=5.0,
            climb=True,
            frozen_indices=None,
            fmax=0.05,
            max_steps=20,
            verbose=False,
            not_converged_error=RuntimeError,
            persist_path=True,
            initial_path_callback=initial_paths.append,
            band_factory=band_factory,
        )

    assert result.energy_ts == pytest.approx(1.5)
    assert result.transition_index == 2
    assert result.n_interior == 2
    assert result.optimizer_steps == 8
    assert optimizer_climb_states == [False, True]
    assert result.path_energies == pytest.approx([0.0, 0.5, 1.5, 0.2])
    assert isinstance(result.atoms_ts.calc, SinglePointCalculator)
    assert result.atoms_ts.get_potential_energy() == pytest.approx(1.5)
    np.testing.assert_allclose(result.atoms_ts.get_forces(), 0.0)
    assert all(image.calc is None for image in images)
    assert all(
        isinstance(image.calc, SinglePointCalculator)
        for image in result.path_images or []
    )
    assert len(initial_paths) == 1
    assert len(initial_paths[0]) == len(images)
    assert all(image.calc is None for image in initial_paths[0])
    assert telemetry.counters["neb.calls"] == 1
    assert telemetry.timings_s["neb.seconds"] >= 0.0


def test_shared_neb_runs_ci_after_every_converged_ordinary_stage(monkeypatch):
    images = [_image(0.0), _image(0.45), _image(0.40)]
    neb = SimpleNamespace(climb=False)
    observed_stages = []

    class StageTrackingOptimizer(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile):
            observed_stages.append(stage_neb.climb)
            super().__init__(stage_neb, logfile=logfile)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", StageTrackingOptimizer)

    result = neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="CI-NEB stage sequence",
        n_images=1,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert observed_stages == [False, True]
    assert result.climb_performed is True


def test_highest_peak_uses_only_nearest_bracketing_minima():
    assert neb_module._highest_peak_minimum_bracket(
        [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5],
        minimum_prominence=0.01,
    ) == (4, 2, 5)
    assert (
        neb_module._highest_peak_minimum_bracket(
            [0.0, 0.4, 1.0, 0.3],
            minimum_prominence=0.01,
        )
        is None
    )


def test_converged_profile_refines_before_advancing_to_ci(monkeypatch):
    profiles = [
        [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5],
        [0.2, 0.8, 1.2, 0.3],
    ]
    bands = []

    def band_factory(initial, final, **_kwargs):
        band_index = len(bands)
        profile = profiles[min(band_index, len(profiles) - 1)]
        images = []
        for fraction, energy in zip(
            np.linspace(0.0, 1.0, len(profile)),
            profile,
        ):
            image = initial.copy()
            image.positions = (
                (1.0 - fraction) * initial.positions
                + fraction * final.positions
            )
            image.calc = SinglePointCalculator(image, energy=energy)
            images.append(image)
        neb = SimpleNamespace(climb=False, band_index=band_index)
        bands.append((neb, images))
        return neb, images

    optimizer_calls = []

    class ImmediatelyConvergedOptimizer:
        def __init__(self, stage_neb, *, logfile):
            del logfile
            self.stage_neb = stage_neb
            self.nsteps = 1
            optimizer_calls.append(
                (stage_neb.band_index, bool(stage_neb.climb))
            )

        def attach(self, _function, interval=1):
            assert interval == 1

        def run(self, *, fmax, steps):
            assert fmax == pytest.approx(0.05)
            assert steps == 20

        def converged(self):
            return True

    def relaxer(candidate, label):
        energy = 0.15 if "initial" in label else 0.25
        return candidate.copy(), energy

    refinements = []
    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", ImmediatelyConvergedOptimizer)

    result = neb_module.run_neb(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[6.0, 0.0, 0.0]]),
        calculator=object(),
        purpose="final converged-profile intermediate check",
        n_images=5,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        intermediate_stagnation_steps=100,
        intermediate_relaxer=relaxer,
        intermediate_refinement_callback=lambda *args: refinements.append(args),
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=band_factory,
    )

    assert len(bands) == 2
    assert optimizer_calls == [(0, False), (1, False), (1, True)]
    assert len(refinements) == 1
    assert refinements[0][2]["trigger"] == "converged_profile"
    assert (
        refinements[0][2]["policy"]
        == "highest_peak_nearest_minima_iterative_v4"
    )
    assert result.intermediate_refinement_count == 1
    assert result.intermediate_trigger == "converged_profile"
    assert result.intermediate_source_stage == "NEB pre-climb relaxation"
    assert result.intermediate_profile_energies == pytest.approx(profiles[0])
    assert result.climb_performed is True
    assert result.optimizer_steps == 3


def test_converged_ci_profile_is_refined_and_rerun(monkeypatch):
    ordinary_profile = [0.0, 0.4, 0.8, 1.2, 1.4, 1.0, 0.5]
    ci_profile = [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5]
    replacement_profile = [0.15, 0.8, 1.2, 0.25]
    bands = []

    def make_images(initial, final, profile):
        images = []
        for fraction, energy in zip(
            np.linspace(0.0, 1.0, len(profile)),
            profile,
        ):
            image = initial.copy()
            image.positions = (
                (1.0 - fraction) * initial.positions
                + fraction * final.positions
            )
            image.calc = SinglePointCalculator(image, energy=energy)
            images.append(image)
        return images

    def band_factory(initial, final, **_kwargs):
        band_index = len(bands)
        profile = ordinary_profile if band_index == 0 else replacement_profile
        neb = SimpleNamespace(climb=False, band_index=band_index)
        images = make_images(initial, final, profile)
        bands.append((neb, images))
        return neb, images

    optimizer_calls = []

    class CIProfileOptimizer:
        def __init__(self, stage_neb, *, logfile):
            del logfile
            self.stage_neb = stage_neb
            self.nsteps = 1
            optimizer_calls.append(
                (stage_neb.band_index, bool(stage_neb.climb))
            )

        def attach(self, _function, interval=1):
            assert interval == 1

        def run(self, *, fmax, steps):
            assert fmax == pytest.approx(0.05)
            assert steps == 20
            if self.stage_neb.band_index == 0 and self.stage_neb.climb:
                for image, energy in zip(bands[0][1], ci_profile):
                    image.calc = SinglePointCalculator(image, energy=energy)

        def converged(self):
            return True

    def relaxer(candidate, label):
        energy = 0.15 if "initial" in label else 0.25
        return candidate.copy(), energy

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", CIProfileOptimizer)

    result = neb_module.run_neb(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[6.0, 0.0, 0.0]]),
        calculator=object(),
        purpose="final converged CI-profile intermediate check",
        n_images=5,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        intermediate_stagnation_steps=100,
        intermediate_relaxer=relaxer,
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=band_factory,
    )

    assert len(bands) == 2
    assert optimizer_calls == [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]
    assert result.intermediate_refinement_count == 1
    assert result.intermediate_trigger == "converged_profile"
    assert result.intermediate_source_stage == "CI-NEB"
    assert result.intermediate_profile_energies == pytest.approx(ci_profile)
    assert result.climb_performed is True
    assert result.optimizer_steps == 4


def test_stalled_neb_refines_only_highest_peak_segment(monkeypatch):
    def energy_image(position: float, energy: float) -> Atoms:
        image = Atoms("H", positions=[[position, 0.0, 0.0]])
        image.calc = SinglePointCalculator(image, energy=energy)
        return image

    stalled_energies = [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5]
    stalled_images = [
        energy_image(float(index), energy)
        for index, energy in enumerate(stalled_energies)
    ]
    bands = []
    band_endpoints = []

    def band_factory(atoms_initial, atoms_final, **_kwargs):
        band_endpoints.append(
            (
                float(atoms_initial.positions[0, 0]),
                float(atoms_final.positions[0, 0]),
            )
        )
        neb = SimpleNamespace(climb=False)
        if not bands:
            images = stalled_images
        else:
            images = [
                energy_image(2.1, 0.15),
                energy_image(3.0, 0.90),
                energy_image(4.0, 1.25),
                energy_image(5.1, 0.25),
            ]
        bands.append((neb, images))
        return neb, images

    optimizer_calls = []

    class StagnationOptimizer:
        def __init__(self, stage_neb, *, logfile):
            del logfile
            self.stage_neb = stage_neb
            self.climb_state = bool(stage_neb.climb)
            self.nsteps = 0
            self.observers = []
            optimizer_calls.append(self)

        def attach(self, function, interval=1):
            assert interval == 1
            self.observers.append(function)

        def run(self, *, fmax, steps):
            assert fmax == pytest.approx(0.05)
            assert steps == 20
            if len(optimizer_calls) == 1:
                for step in range(1, 4):
                    self.nsteps = step
                    for observer in self.observers:
                        observer()
            else:
                self.nsteps = 4 if not self.stage_neb.climb else 2

        def converged(self) -> bool:
            return len(optimizer_calls) > 1

    relaxed_positions = []

    def relaxer(candidate: Atoms, label: str) -> tuple[Atoms, float]:
        position = float(candidate.positions[0, 0])
        relaxed_positions.append((position, label))
        optimized = candidate.copy()
        optimized.positions[0, 0] += 0.1
        energy = 0.15 if position == pytest.approx(2.0) else 0.25
        return optimized, energy

    initial_paths = []
    retained_refinements = []
    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", StagnationOptimizer)

    result = neb_module.run_neb(
        stalled_images[0],
        stalled_images[-1],
        calculator=object(),
        purpose="stalled highest-peak segment NEB",
        n_images=5,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        intermediate_stagnation_steps=2,
        intermediate_energy_tolerance=0.001,
        intermediate_minimum_prominence=0.01,
        intermediate_relaxer=relaxer,
        intermediate_refinement_callback=(
            lambda initial, final, metadata: retained_refinements.append(
                (initial, final, metadata)
            )
        ),
        verbose=False,
        not_converged_error=RuntimeError,
        persist_path=False,
        capture_path=True,
        initial_path_callback=initial_paths.append,
        band_factory=band_factory,
    )

    assert [position for position, _ in relaxed_positions] == [2.0, 5.0]
    assert band_endpoints == [(0.0, 6.0), (2.1, 5.1)]
    assert len(bands) == 2
    assert len(initial_paths) == 2
    assert len(retained_refinements) == 1
    assert retained_refinements[0][2]["other_segments_refined"] is False
    assert [optimizer.climb_state for optimizer in optimizer_calls] == [
        False,
        False,
        True,
    ]
    assert result.intermediate_refinement_performed is True
    assert result.intermediate_refinement_count == 1
    assert result.intermediate_max_refinements == 10
    assert result.intermediate_trigger == "energy_stagnation"
    assert result.intermediate_source_stage == "NEB pre-climb relaxation"
    assert result.intermediate_checkpoint_fmax is None
    assert result.intermediate_peak_index == 4
    assert result.intermediate_left_index == 2
    assert result.intermediate_right_index == 5
    assert result.intermediate_stalled_steps == 3
    assert result.intermediate_profile_energies == pytest.approx(stalled_energies)
    assert result.refinement_initial_energy == pytest.approx(0.15)
    assert result.refinement_final_energy == pytest.approx(0.25)
    assert result.energy_ts == pytest.approx(1.25)
    assert result.climb_performed is True
    assert result.optimizer_steps == 9
    assert result.path_images is not None
    assert [image.positions[0, 0] for image in result.path_images] == pytest.approx(
        [2.1, 3.0, 4.0, 5.1]
    )
    # A shorter lateral band can use the retained final segment, even with
    # successful-run path persistence disabled.
    lateral_initial = Atoms("OH", positions=[[8, 0, 0], [0, 0, 0]])
    lateral_final = Atoms("OH", positions=[[9, 0, 0], [6, 0, 0]])
    projected = neb_module.project_neb_path(
        result.path_images, lateral_initial, lateral_final,
        n_slab=0, n_lateral=1, n_images=1,
    )
    assert projected is not None
    np.testing.assert_allclose(projected[1].positions, [[8.5, 0, 0], [2.9, 0, 0]])


def test_stalled_neb_honors_multiple_refinement_limit(monkeypatch):
    energies = [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5]
    bands = []

    def band_factory(initial, final, **_kwargs):
        fractions = np.linspace(0.0, 1.0, len(energies))
        images = []
        for fraction, energy in zip(fractions, energies):
            image = initial.copy()
            image.positions = (
                (1.0 - fraction) * initial.positions
                + fraction * final.positions
            )
            image.calc = SinglePointCalculator(image, energy=energy)
            images.append(image)
        neb = SimpleNamespace(climb=False, band_index=len(bands))
        bands.append((neb, images))
        return neb, images

    optimizer_calls = []

    class IterativeStagnationOptimizer:
        def __init__(self, stage_neb, *, logfile):
            del logfile
            self.stage_neb = stage_neb
            self.nsteps = 0
            self.observers = []
            optimizer_calls.append((stage_neb.band_index, bool(stage_neb.climb)))

        def attach(self, function, interval=1):
            assert interval == 1
            self.observers.append(function)

        def run(self, *, fmax, steps):
            assert fmax == pytest.approx(0.05)
            assert steps == 20
            if self.stage_neb.band_index < 2 and not self.stage_neb.climb:
                for step in range(1, 4):
                    self.nsteps = step
                    for observer in self.observers:
                        observer()
            else:
                self.nsteps = 1

        def converged(self):
            return True

    def relaxer(candidate, label):
        energy = 0.15 if "initial" in label else 0.25
        return candidate.copy(), energy

    refinements = []
    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", IterativeStagnationOptimizer)

    result = neb_module.run_neb(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[6.0, 0.0, 0.0]]),
        calculator=object(),
        purpose="iterative stalled highest-peak segment NEB",
        n_images=5,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        intermediate_stagnation_steps=2,
        intermediate_max_refinements=2,
        intermediate_relaxer=relaxer,
        intermediate_refinement_callback=lambda *args: refinements.append(args),
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=band_factory,
    )

    assert len(bands) == 3
    assert optimizer_calls == [(0, False), (1, False), (2, False), (2, True)]
    assert len(refinements) == 2
    assert [item[2]["refinement_index"] for item in refinements] == [1, 2]
    assert all(item[2]["max_refinements"] == 2 for item in refinements)
    assert result.intermediate_refinement_performed is True
    assert result.intermediate_refinement_count == 2
    assert result.intermediate_max_refinements == 2


@pytest.mark.parametrize(
    ("rollback_stage", "max_steps", "replacement_rollback"),
    [
        (stage, steps, False)
        for stage in ("ordinary", "ci", "ci_restart")
        for steps in (3, 20)
    ] + [("ordinary", 20, True)],
)
def test_rollback_refines_minima_from_lowest_force_valid_band(
    monkeypatch,
    rollback_stage,
    max_steps,
    replacement_rollback,
):
    # The best force belongs to y=0.1, not the latest/lower-energy valid y=0.2.
    # The rejected y=2 frame has an even smaller force but violates spacing.
    profiles = {
        0.0: [0.0, 0.6, 0.9, 1.1, 0.8, 0.6, 0.5],
        0.1: [0.0, 0.8, 0.2, 0.9, 1.4, 0.3, 0.5],
        0.2: [0.0, 1.6, 0.05, 0.8, 0.4, 0.6, 0.5],
        2.0: [0.0, 2.0, 0.1, 0.5, 0.2, 0.7, 0.5],
    }

    class ProfileCalculator(Calculator):
        implemented_properties = ["energy"]

        def __init__(self, index):
            super().__init__()
            self.index = index

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            frame = round(float(atoms.positions[0, 1]), 1)
            self.results = {"energy": profiles[frame][self.index]}

    bands = []

    def band_factory(initial, final, *, n_images, **_kwargs):
        if not bands:
            images = [Atoms("H", positions=[[0.1 * i, 0.0, 0.0]]) for i in range(7)]
            for index, image in enumerate(images):
                image.calc = ProfileCalculator(index)
        else:
            assert n_images == 2
            images = []
            for fraction, energy in zip(np.linspace(0.0, 1.0, 4), [0.15, 0.1, 1.25, 0.25]):
                image = initial.copy()
                image.positions = (1.0 - fraction) * initial.positions + fraction * final.positions
                image.calc = SinglePointCalculator(image, energy=energy)
                images.append(image)
        neb = SimpleNamespace(climb=False, images=images, band_index=len(bands), force=0.4)
        neb.get_forces = lambda: np.asarray([[neb.force, 0.0, 0.0]])
        bands.append(neb)
        return neb, images

    stage_calls = []

    class RollbackOptimizer:
        def __init__(self, neb, *, logfile, maxstep=0.2):
            del logfile
            self.neb = neb
            self.nsteps = 0
            self.maxstep = maxstep
            self.observers = []
            stage_calls.append((neb.band_index, bool(neb.climb)))

        def attach(self, function, interval=1):
            assert interval == 1
            self.observers.append(function)

        def run(self, *, fmax, steps):
            assert fmax == 0.05
            replacement_attempt = stage_calls.count((1, False))
            retrying_replacement = (
                replacement_rollback and self.neb.band_index == 1
                and not self.neb.climb and replacement_attempt == 2
            )
            assert steps == (max_steps - 3 if retrying_replacement else max_steps)
            if retrying_replacement:
                assert self.maxstep == pytest.approx(0.1)
            should_rollback = self.neb.band_index == 0 and (
                (rollback_stage == "ordinary" and not self.neb.climb)
                or (rollback_stage != "ordinary" and self.neb.climb)
            )
            should_rollback = should_rollback or (
                replacement_rollback and self.neb.band_index == 1
                and not self.neb.climb and replacement_attempt == 1
            )
            frames = [(0.0, 0.4), (0.1, 0.2), (0.2, 0.3), (2.0, 0.01)]
            if should_rollback:
                for step, (height, force) in enumerate(frames):
                    self.nsteps = step
                    self.neb.force = force
                    for image in self.neb.images[1:-1]:
                        image.positions[0, 1] = height
                    for observer in self.observers:
                        observer()
                pytest.fail("the distance guard should interrupt the stretched band")
            else:
                if self.neb.band_index == 1:
                    for image, energy in zip(self.neb.images, [0.15, 0.1, 1.25, 0.25]):
                        image.calc = SinglePointCalculator(image, energy=energy)
                self.neb.force = 0.01
                for observer in self.observers:
                    observer()
                self.nsteps = 1

        def converged(self):
            return True

    relaxed = []

    def relaxer(candidate, label):
        del label
        relaxed.append(candidate.positions.copy())
        return candidate.copy(), 0.15 if len(relaxed) == 1 else 0.25

    captured = []
    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", RollbackOptimizer)
    result = neb_module.run_neb(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.6, 0.0, 0.0]]),
        calculator=object(),
        purpose="rollback minimum inspection",
        n_images=5,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        start_climbing=rollback_stage == "ci_restart",
        frozen_indices=None,
        fmax=0.05,
        max_steps=max_steps,
        image_spacing=0.25,
        intermediate_stagnation_steps=100,
        intermediate_max_refinements=1,
        intermediate_min_images=2,
        intermediate_max_images=5,
        intermediate_relaxer=relaxer,
        intermediate_refinement_callback=lambda *args: captured.append(args),
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=band_factory,
    )

    assert len(bands) == 2
    assert len(relaxed) == 2
    np.testing.assert_allclose(relaxed, [[[0.2, 0.1, 0.0]], [[0.5, 0.1, 0.0]]])
    expected_stages = {
        "ordinary": [(0, False), (1, False), (1, True)],
        "ci": [(0, False), (0, True), (1, False), (1, True)],
        "ci_restart": [(0, True), (1, False), (1, True)],
    }
    expected = expected_stages[rollback_stage]
    if replacement_rollback:
        expected.insert(-1, (1, False))
    assert stage_calls == expected
    assert result.intermediate_trigger == "geometry_rollback"
    assert result.intermediate_source_stage == (
        "NEB pre-climb relaxation" if rollback_stage == "ordinary" else "CI-NEB"
    )
    assert result.intermediate_checkpoint_fmax == pytest.approx(0.2)
    assert result.intermediate_checkpoint_optimizer_steps == (2 if rollback_stage == "ci" else 1)
    assert result.intermediate_profile_energies == pytest.approx(profiles[0.1])
    assert (
        result.intermediate_peak_index,
        result.intermediate_left_index,
        result.intermediate_right_index,
    ) == (4, 2, 5)
    expected_steps = 6 if rollback_stage == "ci" else 5
    assert result.optimizer_steps == expected_steps + (3 if replacement_rollback else 0)
    assert result.climb_performed is True
    assert captured[0][2]["trigger"] == "geometry_rollback"
    assert captured[0][2]["checkpoint_fmax_ev_per_ang"] == pytest.approx(0.2)


def test_dynamic_neb_image_count_uses_maximum_mic_atom_displacement():
    initial = Atoms(
        "H2",
        positions=[[9.8, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    final = initial.copy()
    final.positions[0, 0] = 0.8  # 1.0 Å across the periodic boundary.
    final.positions[1, 0] = 4.5  # 0.5 Å.

    selection = neb_module.resolve_neb_image_count(
        initial,
        final,
        fixed_n_images=10,
        image_spacing=0.25,
        min_images=1,
        max_images=24,
    )

    assert selection.n_images == 3
    assert selection.n_frames == 5
    assert selection.max_endpoint_displacement == pytest.approx(1.0)
    assert selection.estimated_linear_spacing == pytest.approx(0.25)
    assert selection.limited_by == "distance"


def test_dynamic_neb_image_count_honours_bounds_and_fixed_mode():
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[4.0, 0.0, 0.0]])

    capped = neb_module.resolve_neb_image_count(
        initial,
        final,
        fixed_n_images=10,
        image_spacing=0.25,
        min_images=2,
        max_images=8,
    )
    fixed = neb_module.resolve_neb_image_count(
        initial,
        final,
        fixed_n_images=6,
        image_spacing=None,
        min_images=1,
        max_images=24,
    )

    assert capped.n_images == 8
    assert capped.n_frames == 10
    assert capped.limited_by == "maximum"
    assert capped.estimated_linear_spacing > 0.25
    assert fixed.n_images == 6
    assert fixed.n_frames == 8
    assert fixed.limited_by == "fixed"


def test_neb_band_gap_is_mic_aware_and_ignores_frozen_atoms():
    left = Atoms(
        "H2",
        positions=[[9.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    right = left.copy()
    right.positions[0, 0] = 0.2  # 0.4 Å through the periodic boundary.
    right.positions[1, 0] = 4.0  # Larger, but this atom is frozen.

    gap = neb_module._maximum_adjacent_image_displacement(
        [left, right],
        frozen_indices=[1],
    )

    assert gap.distance == pytest.approx(0.4)
    assert gap.left_image == 0
    assert gap.atom_index == 0


@pytest.mark.parametrize(
    ("guard_multiplier", "expected_optimizer_steps"),
    [(2.0, 4), (3.0, 5)],
)
@pytest.mark.parametrize("exhausted", [False, True])
def test_shared_neb_restores_best_valid_band_and_halves_fire_timestep(
    monkeypatch,
    caplog,
    guard_multiplier,
    expected_optimizer_steps,
    exhausted,
):
    energies = [0.0, 1.0, 0.0]
    images = [Atoms("H", positions=[[position, 0.0, 0.0]]) for position in (0.0, 0.10, 0.20)]

    class ConstantEnergyCalculator(Calculator):
        implemented_properties = ["energy"]

        def __init__(self, energy):
            super().__init__()
            self.energy = energy

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": self.energy}

    for image, energy in zip(images, energies):
        image.calc = ConstantEnergyCalculator(energy)
    inspected_positions = []
    original_bracket = neb_module._highest_peak_minimum_bracket

    def inspect_restored_band(profile, **kwargs):
        inspected_positions.append(float(images[1].positions[0, 0]))
        assert profile == energies
        return original_bracket(profile, **kwargs)

    class ControlledNEB:
        def __init__(self):
            self.climb = False
            self.force = 0.4

        def get_forces(self):
            return np.asarray([[self.force, 0.0, 0.0]])

    neb = ControlledNEB()
    attempts = []

    class DivergingFire:
        def __init__(
            self,
            stage_neb,
            *,
            logfile,
            dt,
            dtmax,
            maxstep,
            downhill_check,
        ):
            del logfile, downhill_check
            self.neb = stage_neb
            self.dt = float(dt)
            self.dtmax = float(dtmax)
            self.maxstep = float(maxstep)
            self.fdec = 0.5
            self.nsteps = 0
            self.observers = []
            self.attempt = len(attempts)
            attempts.append(
                {
                    "dt": self.dt,
                    "dtmax": self.dtmax,
                    "start": float(images[1].positions[0, 0]),
                }
            )

        def attach(self, function, interval=1):
            assert interval == 1
            self.observers.append(function)

        def _observe(self):
            for observer in self.observers:
                observer()

        def run(self, *, fmax, steps):
            del fmax
            if self.attempt == 0:
                # The lowest valid force is at x=0.15. A later, worse valid
                # band must not replace it before the band finally diverges.
                states = [
                    (0.10, 0.4),
                    (0.15, 0.2),
                    (0.12, 0.3),
                    # Valid at the 3x 0.25 Å limit; the former 2x guard
                    # would have restarted before reaching the next state.
                    (0.70, 0.25),
                    (0.90, 0.1),
                ]
                for index, (position, force) in enumerate(states):
                    images[1].positions[0, 0] = position
                    self.neb.force = force
                    self.nsteps = index
                    assert self.nsteps <= steps
                    self._observe()
            else:
                assert images[1].positions[0, 0] == pytest.approx(0.15)
                self.neb.force = 0.1
                self._observe()
                self.nsteps = 1
                for image, energy in zip(images, energies):
                    image.calc = SinglePointCalculator(image, energy=energy)

        def converged(self) -> bool:
            return self.attempt == 1

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "FIRE", DivergingFire)
    monkeypatch.setattr(neb_module, "_highest_peak_minimum_bracket", inspect_restored_band)

    expectation = (
        pytest.raises(RuntimeError, match="exhausted .*lowest-force valid band was restored")
        if exhausted else nullcontext()
    )
    with expectation:
        result = neb_module.run_neb(
            images[0],
            images[-1],
            calculator=object(),
            purpose="geometry recovery NEB",
            n_images=1,
            interpolation="linear",
            spring_k=1.0,
            climb=False,
            frozen_indices=None,
            fmax=0.05,
            max_steps=expected_optimizer_steps - 1 if exhausted else 20,
            optimizer="fire",
            optimizer_kwargs={
                "dt": 0.04,
                "dtmax": 0.20,
                "maxstep": 0.10,
                "downhill_check": False,
            },
            image_spacing=0.25,
            geometry_guard_multiplier=guard_multiplier,
            intermediate_stagnation_steps=100,
            verbose=False,
            not_converged_error=RuntimeError,
            band_factory=lambda *_args, **_kwargs: (neb, images),
        )

    expected_inspections = [0.15] if exhausted else [0.15, 0.15]
    assert inspected_positions == pytest.approx(expected_inspections)
    if exhausted:
        assert len(attempts) == 1
        assert images[1].positions[0, 0] == pytest.approx(0.15)
        return

    assert attempts == [
        {"dt": 0.04, "dtmax": 0.20, "start": 0.10},
        {"dt": 0.02, "dtmax": 0.10, "start": 0.15},
    ]
    assert result.optimizer_steps == expected_optimizer_steps
    assert result.intermediate_refinement_performed is False
    assert "Restored the lowest-force valid band" in caplog.text
    assert "dt=0.02, dtmax=0.1" in caplog.text


def test_shared_neb_uses_selected_optimizer(monkeypatch):
    images = [_image(0.0), _image(1.0), _image(0.0)]
    neb = SimpleNamespace(climb=False)
    selected = []

    class SelectedFire(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile):
            selected.append(stage_neb)
            super().__init__(stage_neb, logfile=logfile)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "FIRE", SelectedFire)

    result = neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="FIRE NEB",
        n_images=1,
        interpolation="linear",
        spring_k=0.1,
        climb=False,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        optimizer="fire",
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert selected == [neb]
    assert result.optimizer_steps == 4


def test_shared_neb_recovers_ordinary_fire_but_disables_ci_downhill(monkeypatch):
    images = [_image(0.0), _image(1.0), _image(0.0)]
    neb = SimpleNamespace(climb=False)
    captured = []

    class SelectedFire(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile, **kwargs):
            captured.append((stage_neb.climb, kwargs))
            self.dt = float(kwargs.get("dt", 0.1))
            self.fdec = float(kwargs.get("fdec", 0.5))
            self.downhill_check = bool(kwargs.get("downhill_check", False))
            super().__init__(stage_neb, logfile=logfile)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "FIRE", SelectedFire)

    neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="stable FIRE NEB",
        n_images=1,
        interpolation="linear",
        spring_k=0.1,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        optimizer="fire",
        optimizer_kwargs={
            "dt": 0.01,
            "dtmax": 0.05,
            "maxstep": 0.03,
            "downhill_check": True,
        },
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    ordinary_climb, ordinary_kwargs = captured[0]
    reset_callback = ordinary_kwargs.pop("position_reset_callback")
    assert ordinary_climb is False
    assert callable(reset_callback)
    assert ordinary_kwargs == {
        "dt": pytest.approx(0.01),
        "dtmax": pytest.approx(0.05),
        "maxstep": pytest.approx(0.03),
        "downhill_check": True,
    }
    assert captured[1] == (
        True,
        {
            "dt": pytest.approx(0.01),
            "dtmax": pytest.approx(0.05),
            "maxstep": pytest.approx(0.03),
            "downhill_check": False,
        },
    )


def test_shared_neb_switches_off_downhill_after_fire_dt_collapse(monkeypatch, caplog):
    images = [_image(0.0), _image(1.0), _image(0.0)]
    neb = SimpleNamespace(climb=False)
    selected = []

    class PlateauFire:
        def __init__(
            self,
            stage_neb,
            *,
            logfile,
            dt,
            downhill_check,
            position_reset_callback,
        ):
            del stage_neb, logfile
            self.dt = float(dt)
            self.fdec = 0.5
            self.downhill_check = downhill_check
            self.position_reset_callback = position_reset_callback
            self.nsteps = 0
            selected.append(self)

        def run(self, *, fmax, steps):
            del fmax, steps
            for _ in range(5):
                self.position_reset_callback(None, None, 1.0, 0.0)
                self.dt *= self.fdec
            self.nsteps = 5

        def converged(self) -> bool:
            return True

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "FIRE", PlateauFire)

    neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="plateau recovery NEB",
        n_images=1,
        interpolation="linear",
        spring_k=1.0,
        climb=False,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        optimizer="fire",
        optimizer_kwargs={"dt": 0.01, "downhill_check": True},
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert len(selected) == 1
    assert selected[0].downhill_check is False
    assert selected[0].dt == pytest.approx(0.01)
    assert "disabling it and restoring dt=0.01" in caplog.text


def test_shared_neb_uses_climbing_optimizer_override(monkeypatch):
    images = [_image(0.0), _image(1.0), _image(0.0)]
    neb = SimpleNamespace(climb=False)
    captured = []

    class SelectedFire(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile, **kwargs):
            captured.append(("fire", stage_neb.climb, kwargs))
            self.dt = float(kwargs.get("dt", 0.1))
            self.fdec = float(kwargs.get("fdec", 0.5))
            self.downhill_check = bool(kwargs.get("downhill_check", False))
            super().__init__(stage_neb, logfile=logfile)

    class SelectedMDMin(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile, **kwargs):
            captured.append(("mdmin", stage_neb.climb, kwargs))
            super().__init__(stage_neb, logfile=logfile)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "FIRE", SelectedFire)
    monkeypatch.setattr(neb_module, "MDMin", SelectedMDMin)

    neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="stage-specific optimizer NEB",
        n_images=1,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        optimizer="fire",
        optimizer_kwargs={"dt": 0.01, "downhill_check": True},
        climb_optimizer="mdmin",
        climb_optimizer_kwargs={"dt": 0.05, "maxstep": 0.01},
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    fire_name, fire_climb, fire_kwargs = captured[0]
    assert fire_name == "fire"
    assert fire_climb is False
    assert callable(fire_kwargs.pop("position_reset_callback"))
    assert fire_kwargs == {
        "dt": pytest.approx(0.01),
        "downhill_check": True,
    }
    assert captured[1] == (
        "mdmin",
        True,
        {"dt": pytest.approx(0.05), "maxstep": pytest.approx(0.01)},
    )


def test_shared_neb_climbing_restart_skips_ordinary_stage(monkeypatch):
    images = [_image(0.0), _image(1.0), _image(0.0)]
    neb = SimpleNamespace(climb=False)
    captured = []

    class SelectedMDMin(_ConvergedOptimizer):
        def __init__(self, stage_neb, *, logfile, **kwargs):
            captured.append(("mdmin", stage_neb.climb, kwargs))
            super().__init__(stage_neb, logfile=logfile)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "MDMin", SelectedMDMin)

    neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="CI restart",
        n_images=1,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        optimizer="fire",
        climb_optimizer="mdmin",
        start_climbing=True,
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert captured == [("mdmin", True, {})]


def test_shared_neb_stops_when_preclimb_stage_does_not_converge(monkeypatch):
    images = [_image(0.0), _image(0.5), _image(0.2)]
    neb = SimpleNamespace(climb=False)
    optimizer_climb_states = []
    initial_paths = []
    failed_paths = []

    class NonConvergedOptimizer:
        def __init__(self, stage_neb, *, logfile):
            del logfile
            optimizer_climb_states.append(stage_neb.climb)
            self.nsteps = 20

        def run(self, *, fmax, steps):
            assert fmax == pytest.approx(0.05)
            assert steps == 20
            images[1].positions[0, 0] = 0.75

        def converged(self) -> bool:
            return False

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", NonConvergedOptimizer)

    def legacy_band_factory(
        _atoms_initial,
        _atoms_final,
        *,
        n_images,
        interpolation,
        spring_k,
        climb,
        calculator,
        frozen_indices,
    ):
        del (
            n_images,
            interpolation,
            spring_k,
            climb,
            calculator,
            frozen_indices,
        )
        return neb, images

    with pytest.raises(RuntimeError, match="NEB pre-climb relaxation"):
        neb_module.run_neb(
            images[0],
            images[-1],
            calculator=object(),
            purpose="test NEB",
            n_images=1,
            interpolation="linear",
            spring_k=0.1,
            climb=True,
            frozen_indices=None,
            fmax=0.05,
            max_steps=20,
            verbose=False,
            not_converged_error=RuntimeError,
            initial_path_callback=initial_paths.append,
            failure_path_callback=failed_paths.append,
            band_factory=legacy_band_factory,
        )

    assert optimizer_climb_states == [False]
    assert len(initial_paths) == 1
    assert all(image.calc is None for image in initial_paths[0])
    assert len(failed_paths) == 1
    assert all(
        image.calc is None or isinstance(image.calc, SinglePointCalculator)
        for image in failed_paths[0]
    )
    assert failed_paths[0][1].positions[0, 0] == pytest.approx(0.75)
    assert neb.climb is False
    assert all(image.calc is None for image in images)


def test_make_neb_band_requires_one_concrete_calculator():
    pool = CalculatorPool([object(), object()], max_workers=2)
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    try:
        with pytest.raises(CalculatorConfigError, match="one concrete calculator"):
            neb_module.make_neb_band(
                initial,
                final,
                n_images=3,
                interpolation="linear",
                spring_k=0.1,
                climb=False,
                calculator=pool,
                frozen_indices=None,
            )
    finally:
        pool.shutdown()


@pytest.mark.parametrize(
    "method",
    ["improvedtangent", "aseneb", "eb", "spline", "string"],
)
def test_make_neb_band_uses_selected_ase_method(method):
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    neb, _ = neb_module.make_neb_band(
        initial,
        final,
        n_images=2,
        interpolation="linear",
        spring_k=0.1,
        climb=False,
        calculator=object(),
        frozen_indices=None,
        neb_method=method,
    )

    assert neb.method == method


def test_make_neb_band_rejects_unknown_method():
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="neb_method must be one of"):
        neb_module.make_neb_band(
            initial,
            final,
            n_images=2,
            interpolation="linear",
            spring_k=0.1,
            climb=False,
            calculator=object(),
            frozen_indices=None,
            neb_method="unknown",
        )


def test_idpp_starts_from_linear_band_without_shared_artifacts(
    monkeypatch,
    tmp_path,
):
    observed = {}

    def fake_idpp(neb, traj="idpp.traj", log="idpp.log", mic=False):
        observed["positions"] = [
            np.asarray(image.positions, dtype=float).copy() for image in neb.images
        ]
        observed["traj"] = traj
        observed["log"] = log
        observed["mic"] = mic

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(neb_module, "_idpp_interpolate", fake_idpp)
    calculator = object()
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[3.0, 0.0, 0.0]])

    _, images = neb_module.make_neb_band(
        initial,
        final,
        n_images=2,
        interpolation="idpp",
        spring_k=0.1,
        climb=False,
        calculator=calculator,
        frozen_indices=None,
    )

    observed_x = [positions[0, 0] for positions in observed["positions"]]
    assert observed_x == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert [image.positions[0, 0] for image in images] == pytest.approx(observed_x)
    assert observed["traj"] is None
    assert observed["log"] is None
    assert observed["mic"] is True
    assert not (tmp_path / "idpp.log").exists()
    assert not (tmp_path / "idpp.traj").exists()


def test_nonfinite_idpp_output_restores_linear_band_and_calculators(monkeypatch):
    replacement_calculators = []

    def nonfinite_idpp(neb, **_kwargs):
        for image in neb.images:
            replacement = object()
            replacement_calculators.append(replacement)
            image.calc = replacement
        for image in neb.images[1:-1]:
            image.positions[:] = np.nan

    monkeypatch.setattr(neb_module, "_idpp_interpolate", nonfinite_idpp)
    calculator = object()
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[3.0, 0.0, 0.0]])

    _, images = neb_module.make_neb_band(
        initial,
        final,
        n_images=2,
        interpolation="idpp",
        spring_k=0.1,
        climb=False,
        calculator=calculator,
        frozen_indices=None,
    )

    assert [image.positions[0, 0] for image in images] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert all(np.isfinite(image.positions).all() for image in images)
    assert all(image.calc is calculator for image in images)
    assert all(
        image.calc is not replacement
        for image, replacement in zip(
            images,
            replacement_calculators,
        )
    )


def test_make_neb_band_uses_compatible_seed_without_interpolation(monkeypatch):
    def fail_idpp(*_args, **_kwargs):
        raise AssertionError("seeded NEB bands must not be re-interpolated")

    monkeypatch.setattr(neb_module, "_idpp_interpolate", fail_idpp)
    calculator = object()
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[3.0, 0.0, 0.0]])
    seed = [
        Atoms("H", positions=[[9.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.4, 0.5, 0.0]]),
        Atoms("H", positions=[[2.5, -0.2, 0.0]]),
        Atoms("H", positions=[[-9.0, 0.0, 0.0]]),
    ]

    _, images = neb_module.make_neb_band(
        initial,
        final,
        n_images=2,
        interpolation="idpp",
        spring_k=0.1,
        climb=False,
        calculator=calculator,
        frozen_indices=None,
        initial_path=seed,
    )

    np.testing.assert_allclose(images[0].positions, initial.positions)
    np.testing.assert_allclose(images[1].positions, seed[1].positions)
    np.testing.assert_allclose(images[2].positions, seed[2].positions)
    np.testing.assert_allclose(images[-1].positions, final.positions)
    assert all(image.calc is calculator for image in images)


def test_run_neb_can_capture_path_without_public_persistence(monkeypatch):
    images = [_image(0.0), _image(0.7), _image(0.1)]
    neb = SimpleNamespace(climb=False)

    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", _ConvergedOptimizer)
    result = neb_module.run_neb(
        images[0],
        images[-1],
        calculator=object(),
        purpose="bare warm-start asset",
        n_images=1,
        interpolation="linear",
        spring_k=0.1,
        climb=False,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        verbose=False,
        not_converged_error=RuntimeError,
        persist_path=False,
        capture_path=True,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert result.path_energies == pytest.approx([0.0, 0.7, 0.1])
    assert result.path_images is not None
    assert len(result.path_images) == 3
    assert all(
        isinstance(image.calc, SinglePointCalculator)
        for image in result.path_images
    )


def test_diffusion_endpoint_failure_retains_last_geometry(monkeypatch):
    from ogkmc.structure import StructureOptimisationError

    def fail_optimisation(atoms, **_kwargs):
        failed = atoms.copy()
        failed.positions[0, 0] = 2.0
        raise StructureOptimisationError(
            "forced endpoint failure",
            failed,
            converged=False,
            steps=5,
        )

    monkeypatch.setattr(
        "ogkmc.structure.optimise_structure",
        fail_optimisation,
    )
    with pytest.raises(diffusion_module.EndpointStabilityError) as caught:
        diffusion_module._relax_endpoint(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            calculator=object(),
            fmax=0.05,
            max_steps=5,
            optimizer="lbfgs",
            frozen_indices=None,
            nl_mult=1.0,
            n_slab=0,
            n_lat=0,
            n_mig=1,
            G=nx.Graph(),
            self_node_ids=frozenset(),
            self_node_order=[],
            state_label="endpoint_a",
            verbose=False,
        )

    assert caught.value.state_label == "endpoint_a"
    assert caught.value.atoms.calc is None
    assert caught.value.atoms.positions[0, 0] == pytest.approx(2.0)


def test_bond_endpoint_failure_retains_last_geometry(monkeypatch):
    from ogkmc.structure import StructureOptimisationError

    def fail_optimisation(atoms, **_kwargs):
        failed = atoms.copy()
        failed.positions[0, 0] = 3.0
        raise StructureOptimisationError(
            "forced endpoint failure",
            failed,
            converged=None,
            steps=2,
        )

    monkeypatch.setattr(
        "ogkmc.structure.optimise_structure",
        fail_optimisation,
    )
    with pytest.raises(bond_module.BondEndpointStabilityError) as caught:
        bond_module._relax_bond_endpoint(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            calculator=object(),
            fmax=0.05,
            max_steps=5,
            optimizer="lbfgs",
            frozen_indices=None,
            nl_mult=1.0,
            n_slab=0,
            n_lat=0,
            n_react=1,
            G=nx.Graph(),
            self_groups=[],
            state_label="endpoint_ab",
            verbose=False,
        )

    assert caught.value.state_label == "endpoint_ab"
    assert caught.value.atoms.calc is None
    assert caught.value.atoms.positions[0, 0] == pytest.approx(3.0)


@pytest.mark.parametrize("n_images", [1, 2, 3, 5])
def test_project_neb_path_transfers_bare_curvature_with_mic(n_images):
    cell = [10.0, 10.0, 10.0]
    source_initial = Atoms(
        "CuCuH",
        positions=[
            [9.8, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [8.8, 0.0, 0.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )
    source_middle = Atoms(
        "CuCuH",
        positions=[
            [0.1, 0.4, 0.0],
            [3.0, -0.2, 0.0],
            [0.0, 0.6, 0.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )
    source_final = Atoms(
        "CuCuH",
        positions=[
            [0.2, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )
    target_initial = Atoms(
        "CuCuOH",
        positions=[
            [1.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [9.0, 2.0, 0.0],
            [7.5, 1.0, 0.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )
    target_final = Atoms(
        "CuCuOH",
        positions=[
            [3.0, 1.0, 0.0],
            [7.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [9.5, 1.0, 0.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )

    projected = neb_module.project_neb_path(
        [source_initial, source_middle, source_final],
        target_initial,
        target_final,
        n_slab=2,
        n_lateral=1,
        n_images=n_images,
    )

    assert projected is not None
    assert len(projected) == n_images + 2
    assert all(len(image) == len(target_initial) for image in projected)
    assert np.array_equal(projected[0].positions, target_initial.positions)
    assert np.array_equal(projected[-1].positions, target_final.positions)
    for index, image in enumerate(projected[1:-1], start=1):
        fraction = index / (n_images + 1)
        curvature_weight = 2.0 * min(fraction, 1.0 - fraction)
        expected = target_initial.positions + fraction * np.array(
            [[2, 0, 0], [2, 0, 0], [2, 2, 0], [2, 0, 0]]
        )
        expected += curvature_weight * np.array(
            [[0.1, 0.4, 0], [0, -0.2, 0], [0, 0, 0], [0.2, 0.6, 0]]
        )
        np.testing.assert_allclose(image.positions, expected, atol=1e-14)
    assert all(image.calc is None for image in projected)


@pytest.mark.parametrize("interpolation", ["linear", "idpp"])
def test_project_neb_path_uses_standard_interpolation_for_neighbours(interpolation):
    source = [
        Atoms("CuH", positions=[[0, 0, 0], [x, bend, 2]], cell=[10, 10, 10])
        for x, bend in [(1, 0), (2, 0.7), (3, 0)]
    ]
    initial = Atoms(
        "CuOH", positions=[[0, 0, 0], [2, 1, 2], [1, 0, 2]], cell=[10, 10, 10]
    )
    final = initial.copy()
    final.positions[1:] = [[1, 2, 2], [3, 0, 2]]
    _, baseline = neb_module.make_neb_band(
        initial, final, n_images=3, interpolation=interpolation,
        spring_k=0.1, climb=False, calculator=None, frozen_indices=[0],
    )
    projected = neb_module.project_neb_path(
        source, initial, final, n_slab=1, n_lateral=1, n_images=3,
        interpolation=interpolation, frozen_indices=[0], spring_k=0.1,
    )
    assert projected is not None
    for image, interpolated in zip(projected, baseline):
        np.testing.assert_allclose(image.positions[1], interpolated.positions[1])
    np.testing.assert_allclose(
        [image.positions[-1, 1] for image in projected], [0, 0.35, 0.7, 0.35, 0]
    )
    assert all(image.calc is None for image in projected)
    if interpolation == "idpp":
        # Ensure this geometry actually distinguishes IDPP from linear motion.
        assert not np.allclose(baseline[2].positions[1], [1.5, 1.5, 2])


@pytest.mark.parametrize("channel", ["diffusion", "bond"])
@pytest.mark.parametrize("image_spacing", [None, 0.5])
def test_lateral_stability_passes_resampled_final_bare_band_to_neb(
    monkeypatch, channel, image_spacing,
):
    module = diffusion_module if channel == "diffusion" else bond_module
    graph = nx.Graph()
    graph.add_nodes_from((index, {"element": "H"}) for index in range(1, 5))
    endpoints = [SimpleNamespace(member_node_ids=[nodes]) for nodes in ([1], [2], [3, 4])]
    site = SimpleNamespace(
        iso_class=1, reactant="[H][H]", gas_product=False,
        template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"),
        member_node_ids=[([1, 2], [3, 4])],
        members=[(endpoints[0], 0, endpoints[1], 0)],
    )
    lateral = SimpleNamespace(n_shells=1, lateral_class=1)
    initial = Atoms(
        "CuOHH", positions=[[0, 0, 0], [5, 3, 2], [1, 0, 2], [1, 0.7, 2]],
        cell=[10, 10, 10],
    )
    final = initial.copy()
    final.positions[1, 1] += 1
    final.positions[2:, 0] += 3
    # A shortened bare band with one interior image; both lateral image-count
    # policies below select five interior images for the full endpoint pair.
    source = [initial[[0, 2, 3]] for _ in range(3)]
    source[1].positions[1:, 0] += 0.5
    source[1].positions[1:, 2] += 0.4
    source[-1].positions[1:, 0] += 1
    monkeypatch.setattr(module, "_member_clique_union", lambda *_args: frozenset({0}))
    if channel == "diffusion":
        def build(*_args, endpoint_position, **_kwargs):
            atoms = initial if endpoint_position == "a" else final
            return atoms.copy(), 1, 1, [2, 3], [1, 2]

        monkeypatch.setattr(module, "_build_diffusion_atoms", build)
        relax_name = "_relax_endpoint"
        check = module.check_diffusion_stability
    else:
        site.member_node_ids = [([1], [2], [3, 4])]
        site.members = [tuple(item for endpoint in endpoints for item in (endpoint, 0))]

        def build(*_args, endpoint, **_kwargs):
            atoms = initial if endpoint == "ab" else final
            return atoms.copy(), 1, 1, [2, 3], [1, 2], {}

        monkeypatch.setattr(module, "_build_bond_atoms", build)
        monkeypatch.setattr(module, "_ordered_endpoint_nodes", lambda _graph, nodes: nodes)
        monkeypatch.setattr(
            module, "_select_c_to_ab_mapping",
            lambda *_args, **_kwargs: ([3, 4], {"selected_method": "test"}),
        )
        relax_name = "_relax_bond_endpoint"
        check = module.check_bond_site_stability
    monkeypatch.setattr(module, relax_name, lambda atoms, **_kwargs: (atoms.copy(), 0.0))

    class BandInspected(Exception):
        pass

    def inspect_band(atoms_a, atoms_b, **kwargs):
        images = kwargs["initial_path"]
        assert images is not None
        assert kwargs["n_images"] == 5
        assert len(images) == 7
        assert lateral.neb_initialization == "bare_transfer"
        np.testing.assert_array_equal(images[0].positions, atoms_a.positions)
        np.testing.assert_array_equal(images[-1].positions, atoms_b.positions)
        np.testing.assert_allclose(images[3].positions[1], [5, 3.5, 2])
        np.testing.assert_allclose(images[3].positions[2:], [[2.5, 0, 2.4], [2.5, 0.7, 2.4]])
        raise BandInspected

    monkeypatch.setattr(module, "run_neb", inspect_band)
    with pytest.raises(BandInspected):
        check(
            graph, site, 0, lateral, object(), n_images=5,
            min_images=1,
            image_spacing=image_spacing, interpolation="linear",
            neb_seed_path=source, neb_seed_member_index=0,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source, _initial, _final: source.pop(),
        lambda source, _initial, _final: source[1].set_cell([11.0, 10.0, 10.0]),
        lambda source, _initial, _final: source[1].set_chemical_symbols("CuOH"),
        lambda _source, _initial, final: final.set_pbc([False, True, False]),
        lambda _source, _initial, final: final.set_chemical_symbols("CuCuNH"),
        lambda source, _initial, _final: source[1].positions.__setitem__(
            (0, 0),
            np.nan,
        ),
    ],
)
def test_project_neb_path_rejects_incompatible_inputs(mutate):
    source = [
        Atoms(
            "CuCuH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=[True, True, False],
        )
        for _ in range(3)
    ]
    initial = Atoms(
        "CuCuOH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, False],
    )
    final = initial.copy()
    mutate(source, initial, final)

    assert (
        neb_module.project_neb_path(
            source,
            initial,
            final,
            n_slab=2,
            n_lateral=1,
        )
        is None
    )


def test_neb_uses_only_one_pool_calculator_and_propagates_its_exception():
    class SelectivelyFailingCalculator:
        def __init__(self):
            self.calls = 0

        def get_forces(self, atoms):
            self.calls += 1
            x_position = float(atoms.positions[0, 0])
            if np.isclose(x_position, 0.5):
                raise RuntimeError("intentional interior-image failure")
            return -np.asarray(atoms.positions, dtype=float)

        def get_potential_energy(self, atoms, force_consistent=False):
            del force_consistent
            positions = np.asarray(atoms.positions, dtype=float)
            return 0.5 * float(np.einsum("ij,ij->", positions, positions))

    calculators = [SelectivelyFailingCalculator(), SelectivelyFailingCalculator()]
    pool = CalculatorPool(calculators, max_workers=2)
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    try:
        with pytest.raises(
            RuntimeError,
            match="intentional interior-image failure",
        ):
            neb_module.run_neb(
                initial,
                final,
                calculator=pool,
                purpose="failing single-calculator NEB",
                n_images=3,
                interpolation="linear",
                spring_k=0.1,
                climb=False,
                frozen_indices=None,
                fmax=0.05,
                max_steps=2,
                verbose=False,
                not_converged_error=RuntimeError,
            )

        with pool.acquire_many(2, purpose="failure recovery check") as calculators:
            assert len(calculators) == 2
        call_counts = [calculator.calls for calculator in pool.calculators]
        assert sum(count > 0 for count in call_counts) == 1
    finally:
        pool.shutdown()


def test_neb_uses_one_concrete_calculator_without_outer_batch(monkeypatch):
    images = [_image(0.0), _image(0.5), _image(1.5), _image(0.2)]
    neb = SimpleNamespace(climb=False)
    calculators = [object(), object()]
    pool = CalculatorPool(calculators)
    received = []
    monkeypatch.setattr(neb_module, "BFGS", _ConvergedOptimizer)

    def band_factory(*_args, calculator, **_kwargs):
        received.append(calculator)
        return neb, images

    result = neb_module.run_neb(
        images[0],
        images[-1],
        calculator=pool,
        purpose="isolated NEB",
        n_images=2,
        interpolation="linear",
        spring_k=0.1,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=band_factory,
    )

    assert result.energy_ts == pytest.approx(1.5)
    assert received[0] in calculators
    assert received[0] is not pool
    pool.shutdown()


def test_independent_nebs_can_lease_distinct_calculators(monkeypatch):
    class EnergyCalculator:
        def get_forces(self, atoms):
            return np.zeros_like(atoms.positions)

        def get_potential_energy(self, atoms, force_consistent=False):
            del force_consistent
            return float(atoms.info["energy"])

    barrier = threading.Barrier(2)
    received = []
    received_lock = threading.Lock()
    pool = CalculatorPool(
        [EnergyCalculator(), EnergyCalculator()],
        max_workers=2,
    )

    class ConcurrentOptimizer(_ConvergedOptimizer):
        def run(self, *, fmax, steps):
            del fmax, steps
            barrier.wait(timeout=5)

    def band_factory(*_args, calculator, **_kwargs):
        with received_lock:
            received.append(calculator)
        images = []
        for energy in (0.0, 0.5, 1.5, 0.2):
            image = Atoms("H", positions=[[energy, 0.0, 0.0]])
            image.info["energy"] = energy
            image.calc = calculator
            images.append(image)
        return SimpleNamespace(climb=False), images

    def run(label):
        initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        final = Atoms("H", positions=[[1.0, 0.0, 0.0]])
        return neb_module.run_neb(
            initial,
            final,
            calculator=pool,
            purpose=label,
            n_images=2,
            interpolation="linear",
            spring_k=0.1,
            climb=False,
            frozen_indices=None,
            fmax=0.05,
            max_steps=20,
            verbose=False,
            not_converged_error=RuntimeError,
            band_factory=band_factory,
        )

    monkeypatch.setattr(neb_module, "BFGS", ConcurrentOptimizer)
    try:
        results = pool.gather(
            [
                pool.submit(run, "independent NEB 1"),
                pool.submit(run, "independent NEB 2"),
            ]
        )
    finally:
        pool.shutdown()

    assert [result.energy_ts for result in results] == pytest.approx([1.5, 1.5])
    assert len(received) == 2
    assert len({id(calculator) for calculator in received}) == 2


@pytest.mark.parametrize(
    ("validator", "error", "energy_names", "extra"),
    [
        (
            _check_ts_validity,
            TransitionStateInvalidError,
            {"e_a": 0.0, "e_b": 0.2, "e_ts": float("nan")},
            {"n_mig": 1},
        ),
        (
            _check_bond_ts_validity,
            BondTransitionStateInvalidError,
            {"e_ab": 0.0, "e_c": 0.2, "e_ts": float("nan")},
            {"n_react": 1, "ts_index": 1, "n_interior": 1},
        ),
    ],
)
def test_transition_validators_reject_nonfinite_energy(
    validator,
    error,
    energy_names,
    extra,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(error, match="not finite"):
        validator(
            atoms,
            atoms.copy(),
            atoms.copy(),
            n_slab=0,
            n_lat=0,
            nl_mult=1.2,
            **energy_names,
            **extra,
        )


@pytest.mark.parametrize(
    ("e_a", "e_b", "e_ts"),
    [
        (0.0, -0.2, 0.0),
        (-0.2, 0.0, 0.0),
    ],
)
def test_diffusion_ts_validation_accepts_endpoint_like_low_barrier_path(
    e_a,
    e_b,
    e_ts,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    _check_ts_validity(
        atoms,
        atoms.copy(),
        atoms.copy(),
        n_slab=0,
        n_lat=0,
        n_mig=1,
        nl_mult=1.2,
        e_a=e_a,
        e_b=e_b,
        e_ts=e_ts,
    )


def _gas_reactant(**overrides):
    values = {
        "smiles": "CO",
        "atoms": Atoms(
            "CO",
            positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]],
        ),
        "energy": -14.0,
        "gibbs_energy": -13.6,
        "zpe": 0.13,
        "entropy": 0.001,
        "frequencies_ev": [0.10, 0.20],
        "imaginary_ev": [-0.01],
        "partial_pressure_bar": 1.0,
    }
    values.update(overrides)
    return type("GasReactant", (), values)()


def test_gas_product_endpoint_uses_periodic_reacting_centroid():
    cell = np.diag([10.0, 10.0, 20.0])
    atoms_ab = Atoms(
        "CuHH",
        positions=[
            [5.0, 5.0, 1.0],
            [9.8, 4.0, 2.0],
            [0.2, 4.0, 2.0],
        ],
        cell=cell,
        pbc=[True, True, False],
    )
    atoms_empty = atoms_ab[:1].copy()
    gas_reactant = SimpleNamespace(
        atoms=Atoms(
            "H2",
            positions=[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
        )
    )
    graph = nx.Graph()
    graph.add_node(11, element="H")
    graph.add_node(12, element="H")

    endpoint, diagnostics = bond_module._gas_product_neb_endpoint(
        atoms_empty=atoms_empty,
        atoms_ab=atoms_ab,
        n_slab=1,
        n_lat=0,
        n_react=2,
        react_nodes_ab=[11, 12],
        gas_reactant=gas_reactant,
        G=graph,
        lift_height=6.0,
    )

    gas_positions = endpoint.positions[1:]
    gas_centroid = gas_positions.mean(axis=0)
    assert gas_centroid[0] == pytest.approx(10.0)
    assert gas_centroid[1] == pytest.approx(4.0)
    assert gas_centroid[2] == pytest.approx(2.0 + diagnostics["selected_lift_height_ang"])
    assert np.linalg.norm(gas_positions[1] - gas_positions[0]) == pytest.approx(0.4)


def test_gas_product_endpoint_preserves_reactant_connectivity():
    atoms_ab = Atoms(
        "CuOHO",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 2.0],
        ],
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=[True, True, False],
    )
    atoms_empty = atoms_ab[:1].copy()

    gas_graph = nx.Graph()
    gas_graph.add_nodes_from((0, 1, 2))
    gas_graph.add_edge(0, 2)
    gas_graph.add_edge(1, 2)
    gas_reactant = SimpleNamespace(
        atoms=Atoms(
            "OHO",
            positions=[
                [0.2, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
        ),
        graph=gas_graph,
    )

    graph = nx.Graph()
    graph.add_node(11, element="O")
    graph.add_node(12, element="H")
    graph.add_node(13, element="O")
    graph.add_edge(11, 12, intra_adsorbate=True)

    endpoint, diagnostics = bond_module._gas_product_neb_endpoint(
        atoms_empty=atoms_empty,
        atoms_ab=atoms_ab,
        n_slab=1,
        n_lat=0,
        n_react=3,
        react_nodes_ab=[11, 12, 13],
        gas_reactant=gas_reactant,
        G=graph,
        lift_height=6.0,
        matching_trials=8,
    )

    assert diagnostics["gas_atom_order"] == [2, 1, 0]
    assert diagnostics["selected_method"] == "gas_product_connectivity_kabsch"
    assert diagnostics["alignment"]["connectivity_preserved"] is True
    assert endpoint.get_distance(1, 2, mic=True) == pytest.approx(1.0)


def test_gas_precursor_seed_is_lowered_to_requested_surface_distance():
    atoms = Atoms(
        "CuH2",
        positions=[
            [0.0, 5.0, 1.0],
            [9.8, 5.0, 8.0],
            [0.2, 5.0, 8.0],
        ],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=[True, True, False],
    )

    seed, diagnostics = bond_module._position_gas_precursor_seed(
        atoms,
        n_slab=1,
        n_lat=0,
        n_react=2,
        target_distance=1.8,
    )

    assert bond_module._minimum_gas_surface_distance(
        seed,
        n_slab=1,
        n_lat=0,
        n_react=2,
    ) == pytest.approx(1.8, abs=1.0e-10)
    assert diagnostics["precursor_initial_min_distance_ang"] > 1.8
    assert diagnostics["precursor_vertical_drop_ang"] > 0.0
    assert seed.get_distance(1, 2, mic=True) == pytest.approx(0.4)


def test_gas_precursor_relaxation_fixes_environment_and_keeps_molecule(
    monkeypatch,
):
    import ogkmc.structure as structure_module

    seed = Atoms(
        "CuH2",
        positions=[[0.0, 0.0, 0.0], [-0.37, 0.0, 1.8], [0.37, 0.0, 1.8]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=[True, True, False],
    )
    gas_reactant = SimpleNamespace(
        atoms=Atoms("H2", positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    )

    def fake_optimise(atoms, **kwargs):
        assert kwargs["fmax"] == pytest.approx(0.05)
        fixed = {
            int(index)
            for constraint in atoms.constraints
            if hasattr(constraint, "get_indices")
            for index in constraint.get_indices()
        }
        assert fixed == {0}
        result = atoms.copy()
        result.calc = SinglePointCalculator(
            result,
            energy=-3.0,
            forces=np.zeros((len(result), 3)),
        )
        return result

    monkeypatch.setattr(structure_module, "optimise_structure", fake_optimise)
    monkeypatch.setattr(bond_module, "acquire_calculator", _calculator_context)

    relaxed, energy, diagnostics = bond_module._relax_gas_precursor(
        seed,
        calculator=object(),
        gas_reactant=gas_reactant,
        gas_atom_order=[0, 1],
        target_distance=1.8,
        fmax=0.05,
        max_steps=100,
        optimizer="fire",
        optimizer_kwargs={"dt": 0.01},
        n_slab=1,
        n_lat=0,
        n_react=2,
        verbose=False,
    )

    assert energy == pytest.approx(-3.0)
    assert isinstance(relaxed.calc, SinglePointCalculator)
    np.testing.assert_allclose(relaxed.get_forces(), 0.0)
    assert diagnostics["precursor_relaxed"] is True
    assert diagnostics["precursor_environment_fixed"] is True
    assert diagnostics["precursor_bond_lengths"][0]["relaxed_ang"] == pytest.approx(0.74)


def test_bond_ts_validation_uses_physical_precursor_energy():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    diagnostic = _check_bond_ts_validity(
        atoms, atoms.copy(), atoms.copy(),
        n_slab=0, n_lat=0, n_react=1, nl_mult=1.2,
        e_ab=0.0, e_c=-2.0, e_c_path=0.2, e_ts=0.2,
        ts_index=1, n_interior=1,
    )
    assert diagnostic["status"] == "accepted_low_barrier"
    assert diagnostic["endpoint_matches"] == ["C"]
    assert diagnostic["energy_c_path_ev"] == pytest.approx(0.2)
    assert diagnostic["barrier_c_path_raw_ev"] == pytest.approx(0.0)


@pytest.mark.parametrize("ts_index", [1, 8, 10])
@pytest.mark.parametrize(
    ("e_ab", "e_c", "e_ts", "matches"),
    [
        (-438.9345, -438.7345, -438.7340, ["C"]),
        (-438.7345, -438.9345, -438.7340, ["AB"]),
        (-438.7345, -438.7345, -438.7345, ["AB", "C"]),
        (-438.9345, -438.7345, -438.7350, ["C"]),
        (0.0, -0.2, -0.1, []),
    ],
)
def test_bond_ts_validation_records_endpoint_like_energy(
    ts_index, e_ab, e_c, e_ts, matches,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    diagnostic = _check_bond_ts_validity(
        atoms, atoms.copy(), atoms.copy(),
        n_slab=0, n_lat=0, n_react=1, nl_mult=1.2,
        e_ab=e_ab, e_c=e_c, e_ts=e_ts, ts_index=ts_index, n_interior=10,
    )
    assert diagnostic["status"] == "accepted_low_barrier"
    assert diagnostic["endpoint_matches"] == matches
    assert diagnostic["transition_image_index"] == ts_index
    assert diagnostic["energy_ts_ev"] == e_ts
    assert diagnostic["kmc_barrier_floor_ev"] == pytest.approx(0.1)


def test_bond_ts_validation_leaves_resolved_barrier_without_diagnostic():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    assert _check_bond_ts_validity(
        atoms, atoms.copy(), atoms.copy(),
        n_slab=0, n_lat=0, n_react=1, nl_mult=1.2,
        e_ab=0.0, e_c=0.2, e_ts=0.5, ts_index=8, n_interior=10,
    ) is None


def test_bond_endpoint_like_energy_does_not_accept_third_species():
    # AB contains H2 + H, C contains a connected H3 chain; the TS has three
    # isolated atoms, so neither valid endpoint topology is preserved.
    atoms_ab = Atoms("H3", positions=[[0, 0, 0], [0.7, 0, 0], [3, 0, 0]])
    atoms_c = Atoms("H3", positions=[[0, 0, 0], [0.7, 0, 0], [1.4, 0, 0]])
    atoms_ts = Atoms("H3", positions=[[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    with pytest.raises(BondTransitionStateInvalidError, match="fragmented"):
        _check_bond_ts_validity(
            atoms_ts, atoms_ab, atoms_c,
            n_slab=0, n_lat=0, n_react=3, nl_mult=1.2,
            e_ab=0.0, e_c=0.2, e_ts=0.2005, ts_index=8, n_interior=10,
        )


@pytest.mark.parametrize("persist_neb_path", [False, True])
@pytest.mark.parametrize("higher_endpoint", ["AB", "C"])
def test_endpoint_like_bond_is_admitted_persisted_and_cached(
    monkeypatch, tmp_path, caplog, persist_neb_path, higher_endpoint,
):
    import json

    from ogkmc.io.persistence import ReactionWriter
    from ogkmc.reactions.bond import (
        _bond_energetics_cached, get_applicable_bond_reaction_for_member,
    )
    import ogkmc.reactions.bond as reaction_module
    from ogkmc.sites.bond import BondReactionLateral

    e_ab, e_c = -438.9345, -438.7345
    if higher_endpoint == "AB":
        e_ab, e_c = e_c, e_ab
    e_ts = -438.7340
    atoms_ab = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10])
    atoms_c = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[10, 10, 10])
    graph = nx.Graph()
    graph.add_nodes_from((index, {"element": "H"}) for index in range(1, 5))
    endpoints = [SimpleNamespace(member_node_ids=[nodes]) for nodes in ([1], [2], [3, 4])]
    site = SimpleNamespace(
        iso_class=0, gas_product=False,
        template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"),
        member_node_ids=[([1], [2], [3, 4])],
        members=[tuple(item for endpoint in endpoints for item in (endpoint, 0))],
    )
    lateral = BondReactionLateral(lateral_class=0, ego_graph=nx.Graph())
    monkeypatch.setattr(bond_module, "_member_clique_union", lambda *_args: frozenset({0}))
    monkeypatch.setattr(bond_module, "_ordered_endpoint_nodes", lambda _graph, nodes: nodes)
    monkeypatch.setattr(
        bond_module, "_select_c_to_ab_mapping",
        lambda *_args, **_kwargs: ([3, 4], {"selected_method": "test"}),
    )
    monkeypatch.setattr(
        bond_module, "_build_bond_atoms",
        lambda *_args, endpoint, **_kwargs: (
            (atoms_ab if endpoint == "ab" else atoms_c).copy(), 0, 0, [0, 1], [1, 2], {},
        ),
    )
    monkeypatch.setattr(bond_module, "normalise_reaction_graph", lambda *_args, **_kwargs: nx.Graph())
    monkeypatch.setattr(bond_module, "calculator_identity", lambda _calculator: {"class": "test.Calculator"})
    relax_calls = []
    neb_calls = []

    def relax(atoms, *, state_label, **_kwargs):
        relax_calls.append(state_label)
        energy = e_ab if state_label == "endpoint_ab" else e_c
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
        return atoms, energy

    def run_neb(initial, final, **kwargs):
        neb_calls.append(kwargs)
        assert kwargs["climb"] is True
        path = [initial] + [final.copy() for _ in range(10)] + [final]
        energies = [e_ab] + [e_ts] * 10 + [e_c]
        for atoms, energy in zip(path, energies):
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
        return neb_module.NEBRunResult(
            atoms_ts=path[8], energy_ts=e_ts, transition_index=8,
            n_interior=10, optimizer_steps=4, climb_performed=True,
            path_images=path, path_energies=energies,
        )

    monkeypatch.setattr(bond_module, "_relax_bond_endpoint", relax)
    monkeypatch.setattr(bond_module, "run_neb", run_neb)
    monkeypatch.setattr(reaction_module, "is_bond_applicable", lambda *_args: (True, "couple"))
    monkeypatch.setattr(reaction_module, "check_bond_site_lateral", lambda *_args, **_kwargs: lateral)
    cache_root = str(tmp_path / "cache")
    options = dict(
        temperature=500.0, lateral_interactions=False,
        n_images=10, image_spacing=None, persist_neb_path=persist_neb_path,
        calculation_cache_root=cache_root, calculation_cache_lookup_enabled=True,
    )
    reaction = get_applicable_bond_reaction_for_member(graph, site, 0, object(), **options)
    assert reaction is not None
    assert lateral.stable is True
    assert lateral.invalid_reason is None
    assert lateral.energy_ts == e_ts
    assert reaction.rate > 0.0
    assert reaction.barrier == pytest.approx(0.1 if higher_endpoint == "AB" else 0.3)
    reverse = _bond_energetics_cached(lateral, "dissoc", temperature=500.0)
    assert reverse[1] == pytest.approx(0.1 if higher_endpoint == "C" else 0.3)
    assert reaction.barrier - reverse[1] == pytest.approx(e_c - e_ab)
    assert "accepted for KMC" in caplog.text
    assert "marking as invalid" not in caplog.text
    diagnostic = lateral.ts_energy_diagnostic
    assert diagnostic["endpoint_matches"] == [higher_endpoint]
    assert diagnostic["transition_image_index"] == 8

    writer = ReactionWriter(tmp_path / "output")
    folder = writer.ensure_reaction(reaction, step=0)
    writer.close()
    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["valid"] is True
    assert payload["ts_energy_diagnostic"] == diagnostic
    assert payload["energies_ev"]["transition_raw"] == e_ts
    assert payload["barriers_ev"]["couple_kmc"] == pytest.approx(reaction.barrier)
    assert payload["barriers_ev"]["dissoc_kmc"] == pytest.approx(reverse[1])
    assert (folder / "ts.extxyz").is_file()
    assert (folder / "neb_path.extxyz").is_file() == persist_neb_path
    assert not (tmp_path / "output" / "diagnostics" / "invalid_bond").exists()

    # A fresh lateral class must recover the diagnostic from the real cache,
    # without needing the optional full NEB path or rerunning the calculations.
    lateral = BondReactionLateral(lateral_class=0, ego_graph=nx.Graph())
    restored = get_applicable_bond_reaction_for_member(graph, site, 0, object(), **options)
    assert restored is not None
    assert restored.lateral_class is lateral
    assert lateral.ts_energy_diagnostic == diagnostic
    assert lateral.stable is True
    assert lateral.energy_ts == e_ts
    assert restored.barrier == pytest.approx(reaction.barrier)
    assert relax_calls == ["endpoint_ab", "endpoint_c"]
    assert len(neb_calls) == 1


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("energy", -13.9),
        ("gibbs_energy", -13.5),
        ("zpe", 0.14),
        ("entropy", 0.002),
        ("frequencies_ev", [0.11, 0.20]),
        ("imaginary_ev", [-0.02]),
    ],
)
def test_gas_product_cache_identity_includes_consumed_thermochemistry(
    field,
    changed,
):
    baseline = _gas_reactant()
    modified = _gas_reactant(**{field: changed})

    first = bond_module._gas_product_cache_inputs(
        baseline,
        include_thermochemistry=True,
    )
    second = bond_module._gas_product_cache_inputs(
        modified,
        include_thermochemistry=True,
    )

    assert scientific_input_fingerprint(first) != scientific_input_fingerprint(second)


def test_gas_product_cache_identity_includes_atoms_but_not_live_pressure():
    baseline = _gas_reactant()
    moved_atoms = baseline.atoms.copy()
    moved_atoms.positions[1, 0] += 0.2

    first = bond_module._gas_product_cache_inputs(
        baseline,
        include_thermochemistry=True,
    )
    changed_geometry = bond_module._gas_product_cache_inputs(
        _gas_reactant(atoms=moved_atoms),
        include_thermochemistry=True,
    )
    changed_pressure = bond_module._gas_product_cache_inputs(
        _gas_reactant(partial_pressure_bar=4.0),
        include_thermochemistry=True,
    )

    assert scientific_input_fingerprint(first) != scientific_input_fingerprint(changed_geometry)
    assert scientific_input_fingerprint(first) == scientific_input_fingerprint(changed_pressure)
    assert "partial_pressure_bar" not in first


def test_gas_product_pressure_is_restamped_from_current_reactant():
    lateral = type(
        "Lateral",
        (),
        {"gas_product": True, "gas_pressure_bar": 99.0},
    )()
    site = type(
        "BondSite",
        (),
        {
            "gas_product": True,
            "gas_reactant": _gas_reactant(
                partial_pressure_bar=0.35,
            ),
        },
    )()

    bond_module._stamp_gas_product_runtime_state(lateral, site)

    assert lateral.gas_product is True
    assert lateral.gas_pressure_bar == pytest.approx(0.35)


def _thermochemistry_options():
    return SimpleNamespace(
        enabled=True,
        vibration_displacement=0.01,
        vibration_nfree=2,
        include_ts_vibrations=True,
        min_frequency_ev=1.0e-4,
        symmetry_tolerance=1.0e-3,
        default_spin=0.0,
        default_geometry="nonlinear",
    )


def _fake_harmonic_thermo(
    _atoms,
    indices,
    *,
    energy_ev,
    temperature_k,
    **_kwargs,
):
    correction = float(temperature_k) / 1000.0
    return {
        "g_corr_ev": correction,
        "g_total_ev": float(energy_ev) + correction,
        "zpe_ev": 0.1,
        "entropy_ev_per_k": 0.001,
        "frequencies_ev": [0.1],
        "imaginary_ev": [],
        "vib_indices": list(indices),
    }


def _electronic_record(states):
    return {
        "_cache_match": "electronic",
        "states": {
            name: {
                "atoms": atoms,
                "energy_ev": energy,
                "properties": {"stale_thermochemistry": -999.0},
            }
            for name, (atoms, energy) in states.items()
        },
    }


def test_calculation_database_is_write_only_by_default(monkeypatch):
    import ogkmc.structure as structure_module

    graph = nx.Graph()
    graph.add_node(1)
    site = SimpleNamespace(
        member_node_ids=[[1]],
        iso_class=4,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=2,
        ego_graph=nx.Graph(),
    )
    writes = []

    def _build(*_args, include_self, **_kwargs):
        if include_self:
            return Atoms("H2"), 1, 0, 1
        return Atoms("H"), 1, 0, 0

    def _optimise(atoms, **_kwargs):
        result = atoms.copy()
        energy = -2.0 if len(result) == 2 else -1.0
        result.calc = SinglePointCalculator(
            result,
            energy=energy,
            forces=np.zeros((len(result), 3)),
        )
        return result

    monkeypatch.setattr(adsorption_module, "_build_stability_atoms", _build)
    monkeypatch.setattr(
        adsorption_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        adsorption_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        adsorption_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: pytest.fail("cache lookup should be disabled"),
    )
    monkeypatch.setattr(
        adsorption_module,
        "_write_adsorption_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(adsorption_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(structure_module, "optimise_structure", _optimise)
    monkeypatch.setattr(adsorption_module, "_bond_set", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        adsorption_module,
        "_check_connectivity_stable",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adsorption_module,
        "_check_intended_coordination_stable",
        lambda *_args, **_kwargs: None,
    )

    result = adsorption_module.check_site_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
    )

    assert result == pytest.approx((-2.0, -1.0))
    assert lateral.stable is True
    assert len(writes) == 1


def test_adsorption_thermochemistry_reuses_cached_electronic_states(
    monkeypatch,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_node(1)
    site = SimpleNamespace(
        member_node_ids=[[1]],
        iso_class=4,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=2,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "occupied": (Atoms("H2"), -2.0),
            "unoccupied": (Atoms("H"), -1.0),
        }
    )
    writes = []
    thermo_calls = []

    def _build(*_args, include_self, **_kwargs):
        if include_self:
            return Atoms("H2"), 1, 0, 1
        return Atoms("H"), 1, 0, 0

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    def _load(*_args, **kwargs):
        assert kwargs["allow_electronic_match"] is True
        return record

    monkeypatch.setattr(adsorption_module, "_build_stability_atoms", _build)
    monkeypatch.setattr(
        adsorption_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        adsorption_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(adsorption_module, "load_calculation_record", _load)
    monkeypatch.setattr(
        adsorption_module,
        "_write_adsorption_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = adsorption_module.check_site_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        calculation_cache_lookup_enabled=True,
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=700.0,
    )

    assert result == pytest.approx((-2.0, -1.0))
    assert thermo_calls == [700.0, 700.0]
    assert lateral.g_occupied == pytest.approx(-1.3)
    assert lateral.stale_thermochemistry is None
    assert lateral.stable is True
    assert len(writes) == 1


def test_adsorption_thermochemistry_failure_keeps_cached_state_retryable(
    monkeypatch,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_node(1)
    site = SimpleNamespace(
        member_node_ids=[[1]],
        iso_class=4,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=2,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "occupied": (Atoms("H2"), -2.0),
            "unoccupied": (Atoms("H"), -1.0),
        }
    )

    def _build(*_args, include_self, **_kwargs):
        if include_self:
            return Atoms("H2"), 1, 0, 1
        return Atoms("H"), 1, 0, 0

    def _fail_thermochemistry(*_args, **_kwargs):
        raise RuntimeError("calculator worker failed")

    monkeypatch.setattr(adsorption_module, "_build_stability_atoms", _build)
    monkeypatch.setattr(
        adsorption_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        adsorption_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        adsorption_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _fail_thermochemistry,
    )

    with pytest.raises(RuntimeError, match="calculator worker failed"):
        adsorption_module.check_site_stability(
            graph,
            site,
            0,
            lateral,
            object(),
            calculation_cache_root="/tmp/test-cache",
            calculation_cache_lookup_enabled=True,
            free_energy_options=_thermochemistry_options(),
            free_energy_temperature_k=700.0,
        )

    assert lateral.stable is None
    assert lateral.energy_occupied == pytest.approx(-2.0)
    assert lateral.energy_unoccupied == pytest.approx(-1.0)


@pytest.mark.parametrize("capture_neb_path", [False, True])
def test_diffusion_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch, capture_neb_path,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2))
    endpoint_a = SimpleNamespace()
    endpoint_b = SimpleNamespace()
    site = SimpleNamespace(
        member_node_ids=[([1], [2])],
        members=[(endpoint_a, 0, endpoint_b, 0)],
        iso_class=5,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=3,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "state_a": (Atoms("H2"), -2.0),
            "state_b": (Atoms("H2"), -1.8),
            "transition": (Atoms("H2"), -1.0),
        }
    )
    record["neb"] = {
        "path": [Atoms("H2") for _ in range(3)],
        "energies_ev": [0.0, 0.5, 0.0],
    }
    writes = []
    thermo_calls = []

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    monkeypatch.setattr(
        diffusion_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        diffusion_module,
        "_build_diffusion_atoms",
        lambda *_args, **_kwargs: (Atoms("H2"), 1, 0, [1], [1]),
    )
    monkeypatch.setattr(
        diffusion_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        diffusion_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        diffusion_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        diffusion_module,
        "_write_diffusion_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        diffusion_module,
        "_relax_endpoint",
        lambda *_args, **_kwargs: pytest.fail("endpoint relaxation reran"),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = diffusion_module.check_diffusion_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        calculation_cache_lookup_enabled=True,
        capture_neb_path=capture_neb_path,
        n_images=5,
        image_spacing=None,
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=650.0,
    )

    assert result == pytest.approx((-2.0, -1.8, -1.0))
    assert thermo_calls == [650.0, 650.0, 650.0]
    assert lateral.g_ts == pytest.approx(-0.35)
    assert lateral.stale_thermochemistry is None
    assert lateral.stable is True
    assert len(writes) == 1

    if capture_neb_path:
        assert len(lateral._warm_start_neb_path) == 3
        assert lateral._warm_start_member_index == 0
        assert lateral.atoms_neb_path is None


def test_diffusion_thermochemistry_failure_keeps_cached_state_retryable(
    monkeypatch,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2))
    endpoint_a = SimpleNamespace()
    endpoint_b = SimpleNamespace()
    site = SimpleNamespace(
        member_node_ids=[([1], [2])],
        members=[(endpoint_a, 0, endpoint_b, 0)],
        iso_class=5,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=3,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "state_a": (Atoms("H2"), -2.0),
            "state_b": (Atoms("H2"), -1.8),
            "transition": (Atoms("H2"), -1.0),
        }
    )

    def _fail_thermochemistry(*_args, **_kwargs):
        raise RuntimeError("calculator worker failed")

    monkeypatch.setattr(
        diffusion_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        diffusion_module,
        "_build_diffusion_atoms",
        lambda *_args, **_kwargs: (Atoms("H2"), 1, 0, [1], [1]),
    )
    monkeypatch.setattr(
        diffusion_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        diffusion_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        diffusion_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _fail_thermochemistry,
    )

    with pytest.raises(RuntimeError, match="calculator worker failed"):
        diffusion_module.check_diffusion_stability(
            graph,
            site,
            0,
            lateral,
            object(),
            calculation_cache_root="/tmp/test-cache",
            calculation_cache_lookup_enabled=True,
            free_energy_options=_thermochemistry_options(),
            free_energy_temperature_k=650.0,
        )

    assert lateral.stable is None
    assert lateral.energy_a == pytest.approx(-2.0)
    assert lateral.energy_b == pytest.approx(-1.8)
    assert lateral.energy_ts == pytest.approx(-1.0)


@pytest.mark.parametrize("capture_neb_path", [False, True])
def test_bond_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch, capture_neb_path,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2, 3))
    endpoint_a = SimpleNamespace(member_node_ids=[[1]])
    endpoint_b = SimpleNamespace(member_node_ids=[[2]])
    endpoint_c = SimpleNamespace(member_node_ids=[[3]])
    template = SimpleNamespace(
        smiles_a="[H]",
        smiles_b="[H]",
        smiles_c="[H][H]",
    )
    site = SimpleNamespace(
        member_node_ids=[([1], [2], [3])],
        members=[(endpoint_a, 0, endpoint_b, 0, endpoint_c, 0)],
        iso_class=6,
        template=template,
        gas_product=False,
        gas_lift_height=6.0,
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=4,
        ego_graph=nx.Graph(),
        ts_energy_diagnostic={"status": "stale_previous_transition"},
    )
    record = _electronic_record(
        {
            "state_ab": (Atoms("H3"), -3.0),
            "state_c": (Atoms("H3"), -3.5),
            "transition": (Atoms("H3"), -2.0),
        }
    )
    record["neb"] = {
        "path": [Atoms("H3") for _ in range(3)],
        "energies_ev": [0.0, 0.5, 0.0],
    }
    writes = []
    thermo_calls = []

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    monkeypatch.setattr(
        bond_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        bond_module,
        "_build_bond_atoms",
        lambda *_args, **_kwargs: (
            Atoms("H3"),
            1,
            0,
            [1, 2],
            [1, 2],
            {},
        ),
    )
    monkeypatch.setattr(
        bond_module,
        "_ordered_endpoint_nodes",
        lambda _graph, node_ids: list(node_ids),
    )
    monkeypatch.setattr(
        bond_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        bond_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        bond_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        bond_module,
        "_write_bond_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        bond_module,
        "_relax_bond_endpoint",
        lambda *_args, **_kwargs: pytest.fail("endpoint relaxation reran"),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = bond_module.check_bond_site_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        calculation_cache_lookup_enabled=True,
        capture_neb_path=capture_neb_path,
        n_images=5,
        image_spacing=None,
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=600.0,
    )

    assert result == pytest.approx((-3.0, -3.5, -2.0))
    assert thermo_calls == [600.0, 600.0, 600.0]
    assert lateral.g_ab == pytest.approx(-2.4)
    assert lateral.stale_thermochemistry is None
    assert lateral.ts_energy_diagnostic is None
    assert lateral.stable is True
    assert len(writes) == 1

    if capture_neb_path:
        assert len(lateral._warm_start_neb_path) == 3
        assert lateral._warm_start_member_index == 0
        assert lateral.atoms_neb_path is None


def test_bond_thermochemistry_failure_keeps_cached_state_retryable(
    monkeypatch,
):
    import ogkmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2, 3))
    endpoint_a = SimpleNamespace(member_node_ids=[[1]])
    endpoint_b = SimpleNamespace(member_node_ids=[[2]])
    endpoint_c = SimpleNamespace(member_node_ids=[[3]])
    site = SimpleNamespace(
        member_node_ids=[([1], [2], [3])],
        members=[(endpoint_a, 0, endpoint_b, 0, endpoint_c, 0)],
        iso_class=6,
        template=SimpleNamespace(
            smiles_a="[H]",
            smiles_b="[H]",
            smiles_c="[H][H]",
        ),
        gas_product=False,
        gas_lift_height=6.0,
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=4,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "state_ab": (Atoms("H3"), -3.0),
            "state_c": (Atoms("H3"), -3.5),
            "transition": (Atoms("H3"), -2.0),
        }
    )

    def _fail_thermochemistry(*_args, **_kwargs):
        raise RuntimeError("calculator worker failed")

    monkeypatch.setattr(
        bond_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        bond_module,
        "_build_bond_atoms",
        lambda *_args, **_kwargs: (
            Atoms("H3"),
            1,
            0,
            [1, 2],
            [1, 2],
            {},
        ),
    )
    monkeypatch.setattr(
        bond_module,
        "_ordered_endpoint_nodes",
        lambda _graph, node_ids: list(node_ids),
    )
    monkeypatch.setattr(
        bond_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        bond_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        bond_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _fail_thermochemistry,
    )

    with pytest.raises(RuntimeError, match="calculator worker failed"):
        bond_module.check_bond_site_stability(
            graph,
            site,
            0,
            lateral,
            object(),
            calculation_cache_root="/tmp/test-cache",
            calculation_cache_lookup_enabled=True,
            free_energy_options=_thermochemistry_options(),
            free_energy_temperature_k=600.0,
        )

    assert lateral.stable is None
    assert lateral.energy_ab == pytest.approx(-3.0)
    assert lateral.energy_c == pytest.approx(-3.5)
    assert lateral.energy_ts == pytest.approx(-2.0)
