"""Focused tests for shared stability/NEB execution boundaries."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from types import SimpleNamespace

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import networkx as nx
import numpy as np
import pytest

from autokmc.io.calculators import CalculatorConfigError, CalculatorPool
from autokmc.io.calculation_cache import scientific_input_fingerprint
from autokmc.sites.stability import adsorption as adsorption_module
from autokmc.sites.stability import bond as bond_module
from autokmc.sites.stability import diffusion as diffusion_module
from autokmc.sites.stability import neb as neb_module
from autokmc.sites.stability.bond import (
    BondTransitionStateInvalidError,
    _check_bond_ts_validity,
)
from autokmc.sites.stability.diffusion import (
    TransitionStateInvalidError,
    _check_ts_validity,
)
from autokmc.utils.telemetry import RuntimeTelemetry, telemetry_context


def _image(energy: float) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=float(energy))
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
    assert result.atoms_ts.calc is None
    assert all(image.calc is None for image in images)
    assert all(image.calc is None for image in result.path_images or [])
    assert len(initial_paths) == 1
    assert len(initial_paths[0]) == len(images)
    assert all(image.calc is None for image in initial_paths[0])
    assert telemetry.counters["neb.calls"] == 1
    assert telemetry.timings_s["neb.seconds"] >= 0.0


@pytest.mark.parametrize(
    ("transition_energy", "expected_stages", "expected_skip"),
    [
        (0.45, [False], True),
        (0.50, [False, True], False),
    ],
)
def test_shared_neb_skips_ci_when_either_regular_barrier_is_below_floor(
    monkeypatch,
    caplog,
    transition_energy,
    expected_stages,
    expected_skip,
):
    # Forward barrier is 0.45/0.50 eV; reverse is 0.05/0.10 eV.
    images = [_image(0.0), _image(transition_energy), _image(0.40)]
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
        purpose="low-barrier NEB",
        n_images=1,
        interpolation="linear",
        spring_k=1.0,
        climb=True,
        frozen_indices=None,
        fmax=0.05,
        max_steps=20,
        barrier_endpoint_energies=(0.0, 0.40),
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert observed_stages == expected_stages
    assert result.climb_skipped_low_barrier is expected_skip
    assert result.climb_performed is (not expected_skip)
    assert result.regular_forward_barrier == pytest.approx(transition_energy)
    assert result.regular_reverse_barrier == pytest.approx(
        transition_energy - 0.40
    )
    if expected_skip:
        assert "Skipping CI-NEB" in caplog.text


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


def test_shared_neb_restores_best_valid_band_and_halves_fire_timestep(
    monkeypatch,
    caplog,
):
    energies = [0.0, 1.0, 0.0]
    images = [
        Atoms("H", positions=[[position, 0.0, 0.0]])
        for position in (0.0, 0.10, 0.20)
    ]

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
        max_steps=20,
        optimizer="fire",
        optimizer_kwargs={
            "dt": 0.04,
            "dtmax": 0.20,
            "maxstep": 0.10,
            "downhill_check": False,
        },
        image_spacing=0.25,
        verbose=False,
        not_converged_error=RuntimeError,
        band_factory=lambda *_args, **_kwargs: (neb, images),
    )

    assert attempts == [
        {"dt": 0.04, "dtmax": 0.20, "start": 0.10},
        {"dt": 0.02, "dtmax": 0.10, "start": 0.15},
    ]
    assert result.optimizer_steps == 4
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
    assert all(image.calc is None for image in failed_paths[0])
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
            np.asarray(image.positions, dtype=float).copy()
            for image in neb.images
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

    assert [image.positions[0, 0] for image in images] == pytest.approx(
        [0.0, 1.0, 2.0, 3.0]
    )
    assert all(np.isfinite(image.positions).all() for image in images)
    assert all(image.calc is calculator for image in images)
    assert all(image.calc is not replacement for image, replacement in zip(
        images,
        replacement_calculators,
    ))


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
    assert all(image.calc is None for image in result.path_images)


def test_diffusion_endpoint_failure_retains_last_geometry(monkeypatch):
    from autokmc.structure import StructureOptimisationError

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
        "autokmc.structure.optimise_structure",
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
    from autokmc.structure import StructureOptimisationError

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
        "autokmc.structure.optimise_structure",
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


def test_project_neb_path_transfers_bare_curvature_with_mic():
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
    )

    assert projected is not None
    assert len(projected) == 3
    assert all(len(image) == len(target_initial) for image in projected)
    assert np.array_equal(projected[0].positions, target_initial.positions)
    assert np.array_equal(projected[-1].positions, target_final.positions)
    np.testing.assert_allclose(
        projected[1].positions,
        [
            [2.1, 1.4, 0.0],
            [6.0, 0.8, 0.0],
            [10.0, 3.0, 0.0],
            [8.7, 1.6, 0.0],
        ],
    )
    assert all(image.calc is None for image in projected)


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

    assert neb_module.project_neb_path(
        source,
        initial,
        final,
        n_slab=2,
        n_lateral=1,
    ) is None


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
            {"n_react": 1},
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
            ts_index=1,
            n_interior=1,
            **energy_names,
            **extra,
        )


def test_transition_validators_allow_intentional_low_barrier_floor():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    _check_ts_validity(
        atoms,
        atoms.copy(),
        atoms.copy(),
        n_slab=0,
        n_lat=0,
        n_mig=1,
        nl_mult=1.2,
        e_a=0.0,
        e_b=0.4,
        e_ts=0.4,
        ts_index=1,
        n_interior=1,
        allow_barrier_floor=True,
    )
    _check_bond_ts_validity(
        atoms,
        atoms.copy(),
        atoms.copy(),
        n_slab=0,
        n_lat=0,
        n_react=1,
        nl_mult=1.2,
        e_ab=0.0,
        e_c=0.4,
        e_ts=0.4,
        ts_index=1,
        n_interior=1,
        allow_barrier_floor=True,
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
    assert gas_centroid[2] == pytest.approx(
        2.0 + diagnostics["selected_lift_height_ang"]
    )
    assert np.linalg.norm(gas_positions[1] - gas_positions[0]) == pytest.approx(0.4)


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
    import autokmc.structure as structure_module

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
    assert relaxed.calc is None
    assert diagnostics["precursor_relaxed"] is True
    assert diagnostics["precursor_environment_fixed"] is True
    assert diagnostics["precursor_bond_lengths"][0]["relaxed_ang"] == pytest.approx(0.74)


def test_bond_ts_validation_uses_physical_precursor_energy():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(
        BondTransitionStateInvalidError,
        match="physical endpoint C",
    ):
        _check_bond_ts_validity(
            atoms,
            atoms.copy(),
            atoms.copy(),
            n_slab=0,
            n_lat=0,
            n_react=1,
            nl_mult=1.2,
            e_ab=0.0,
            e_c=-2.0,
            e_c_path=0.2,
            e_ts=0.2,
            ts_index=1,
            n_interior=1,
        )


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

    assert (
        scientific_input_fingerprint(first)
        != scientific_input_fingerprint(changed_geometry)
    )
    assert (
        scientific_input_fingerprint(first)
        == scientific_input_fingerprint(changed_pressure)
    )
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
        {"gas_product": True, "gas_reactant": _gas_reactant(
            partial_pressure_bar=0.35,
        )},
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
    import autokmc.structure as structure_module

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
    import autokmc.thermo.free_energy as free_energy_module

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
    import autokmc.thermo.free_energy as free_energy_module

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


def test_diffusion_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

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
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=650.0,
    )

    assert result == pytest.approx((-2.0, -1.8, -1.0))
    assert thermo_calls == [650.0, 650.0, 650.0]
    assert lateral.g_ts == pytest.approx(-0.35)
    assert lateral.stale_thermochemistry is None
    assert lateral.stable is True
    assert len(writes) == 1


def test_diffusion_thermochemistry_failure_keeps_cached_state_retryable(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

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


def test_bond_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

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
    )
    record = _electronic_record(
        {
            "state_ab": (Atoms("H3"), -3.0),
            "state_c": (Atoms("H3"), -3.5),
            "transition": (Atoms("H3"), -2.0),
        }
    )
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
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=600.0,
    )

    assert result == pytest.approx((-3.0, -3.5, -2.0))
    assert thermo_calls == [600.0, 600.0, 600.0]
    assert lateral.g_ab == pytest.approx(-2.4)
    assert lateral.stale_thermochemistry is None
    assert lateral.stable is True
    assert len(writes) == 1


def test_bond_thermochemistry_failure_keeps_cached_state_retryable(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

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
