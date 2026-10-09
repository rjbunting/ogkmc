"""Periodic minima are candidates, independently relaxed before stagnation."""

from __future__ import annotations

import json
from inspect import signature
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import FIRE
from scipy.signal import find_peaks

from ogkmc.io.checkpoint import load_checkpoint, make_checkpoint_state, save_checkpoint
from ogkmc.io.calculators import CalculatorPool
from ogkmc.io.config import ConfigError, load_config
from ogkmc.io.persistence import ReactionWriter
from ogkmc.sites.stability import neb as neb_module
from ogkmc.sites.stability.intermediate_pruning import (
    CompositeDirectEventDetected,
    retain_periodic_minimum_check,
)
from ogkmc.structure.optimization import StructureOptimisationError
from ogkmc.workflow.runtime import resolve_channel_runtime


class AnalyticPotential(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, multiwell=True):
        super().__init__()
        self.multiwell = multiwell
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        x, y, z = atoms.positions[-1]
        if self.multiwell:
            segment = min(2, max(0, int(np.floor(x))))
            s = x - segment
            low, high = [(2, 0), (0, 0), (0, 1)][segment]
            amplitude = [1, 1, 4][segment]
            energy = low + (high - low) * (3 * s**2 - 2 * s**3)
            energy += amplitude * np.sin(np.pi * s) ** 2
            derivative = (high - low) * 6 * s * (1 - s)
            derivative += amplitude * np.pi * np.sin(2 * np.pi * s)
        else:
            energy = (x * x - 1) ** 2
            derivative = 4 * x * (x * x - 1)
        forces = np.zeros((len(atoms), 3))
        forces[-1] = [-derivative, -2 * y, -2 * z]
        self.results = {"energy": energy + y * y + z * z, "forces": forces}


def pose(x):
    atoms = Atoms(
        "CuH", positions=[[0, 0, -5], [x, 0, 0]], cell=[10, 10, 20], pbc=[True, True, False]
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_array("graph_node_id", np.array(["slab", "reacting"]))
    return atoms


def multiwell_seed(*, filled=False):
    seed = [pose(x) for x in np.linspace(0, 3, 13)]
    for i, frame in enumerate(seed[1:-1], 1):
        if filled:
            frame.calc = AnalyticPotential()
            energy = frame.get_potential_energy()
            x = frame.positions[-1, 0]
            target = 2 - x / 3 + 10 * np.sin(np.pi * x / 3)
            frame.positions[-1, 1] = np.sqrt(max(0, target - energy))
        else:
            frame.positions[-1, 1] = 0.2 * np.sin(np.pi * i / 12)
    return seed


def run(seed, calculator, **kwargs):
    options = dict(
        calculator=calculator,
        purpose="analytic periodic minimum test",
        n_images=len(seed) - 2,
        interpolation="linear",
        spring_k=1,
        climb=False,
        frozen_indices=[0],
        fmax=0.05,
        max_steps=1000,
        verbose=False,
        not_converged_error=RuntimeError,
        optimizer="fire",
        intermediate_optimizer="fire",
        intermediate_stagnation_steps=None,
        capture_path=True,
        initial_path=seed,
    )
    options.update(kwargs)
    return neb_module.run_neb(seed[0], seed[-1], **options)


def freeze_band(monkeypatch):
    instances = []

    class FrozenBand(FIRE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def step(self, *args, **kwargs):
            pass

    monkeypatch.setattr(neb_module, "FIRE", FrozenBand)
    return instances


@pytest.mark.parametrize("ci_only", [False, True])
def test_new_valley_is_relaxed_during_ordinary_and_ci_optimization(ci_only):
    seed = multiwell_seed(filled=True)
    energies = []
    for frame in seed:
        frame.calc = AnalyticPotential()
        energies.append(frame.get_potential_energy())
    assert not find_peaks(-np.array(energies), prominence=0.01)[0].size
    retained, failures, reviewed = [], [], []

    def classify(candidate, endpoint, metadata):
        assert np.linalg.norm(candidate.get_forces(), axis=1).max() <= 0.05
        assert metadata["optimizer_steps_at_detection"] > 0
        assert metadata["optimizer_steps_at_detection"] % 10 == 0
        assert metadata["trigger"] == "periodic_minimum_check"
        assert metadata["source_stage"] == ("CI-NEB" if ci_only else "NEB")
        np.testing.assert_array_equal(candidate.positions[0], seed[0].positions[0])
        np.testing.assert_array_equal(candidate.pbc, seed[0].pbc)
        reviewed.append(metadata)
        raise CompositeDirectEventDetected("analytic classified intermediate", {"reason": "test"})

    with pytest.raises(CompositeDirectEventDetected):
        run(
            seed,
            AnalyticPotential(),
            climb=ci_only,
            start_climbing=ci_only,
            intermediate_refinement_callback=classify,
            intermediate_minimum_callback=lambda *args: retained.append(args),
            failure_path_callback=failures.append,
        )
    assert len(reviewed) == len(retained) == len(failures) == 1
    source, relaxed, band, metadata = retained[0]
    assert metadata["status"] == "composite"
    assert metadata["max_force_ev_per_ang"] <= 0.05
    assert source.arrays["graph_node_id"].tolist() == ["slab", "reacting"]
    assert isinstance(relaxed.calc, SinglePointCalculator)
    for before, after in zip(band, failures[0]):
        np.testing.assert_array_equal(before.positions, after.positions)


def test_all_valleys_relaxed_once_and_live_optimizer_is_preserved(monkeypatch):
    instances = freeze_band(monkeypatch)
    retained, failures, reviewed = [], [], []
    seed = multiwell_seed()

    def classify(candidate, endpoint, metadata):
        assert np.linalg.norm(candidate.get_forces(), axis=1).max() <= 0.05
        reviewed.append(metadata["left_state_image_index"])

    with pytest.raises(RuntimeError, match="did not converge"):
        run(
            seed,
            AnalyticPotential(),
            max_steps=100,
            intermediate_refinement_callback=classify,
            intermediate_minimum_callback=lambda *args: retained.append(args),
            failure_path_callback=failures.append,
        )
    assert reviewed == [4, 8]
    assert len(instances) == 1 and instances[0].nsteps == 100
    assert len(retained) == 2
    assert all(args[-1]["status"] == "force_converged" for args in retained)
    for before, after in zip(seed, failures[0]):
        np.testing.assert_array_equal(before.positions, after.positions)


def test_periodic_relaxation_reuses_single_batched_calculator_lease():
    class BatchedAnalyticPotential(AnalyticPotential):
        def __init__(self):
            super().__init__()
            self.band_calls = 0
            self.band_calculator = AnalyticPotential()

        def evaluate_band(self, images):
            self.band_calls += 1
            return [
                (
                    self.band_calculator.get_potential_energy(image),
                    self.band_calculator.get_forces(image),
                )
                for image in images
            ]

    calculator = BatchedAnalyticPotential()
    pool = CalculatorPool([calculator], max_workers=1)
    retained = []

    def classify(candidate, endpoint, metadata):
        assert np.linalg.norm(candidate.get_forces(), axis=1).max() <= 0.05
        assert metadata["optimizer_steps_at_detection"] > 0
        raise CompositeDirectEventDetected("analytic classified intermediate", {"reason": "test"})

    try:
        with pytest.raises(CompositeDirectEventDetected):
            run(
                multiwell_seed(filled=True),
                pool,
                band_eval="batched",
                intermediate_refinement_callback=classify,
                intermediate_minimum_callback=lambda *args: retained.append(args),
            )
        with pool.acquire(purpose="verify NEB lease released") as leased:
            assert leased is calculator
    finally:
        pool.shutdown()
    assert calculator.band_calls > 1
    assert calculator.calls > 0
    assert len(retained) == 1 and retained[0][-1]["status"] == "composite"


def test_unconverged_finite_candidates_are_unstable_and_neb_continues(monkeypatch):
    instances = freeze_band(monkeypatch)
    retained = []

    def unstable(candidate, label):
        candidate.calc = SinglePointCalculator(
            candidate, energy=0, forces=np.ones((len(candidate), 3))
        )
        raise StructureOptimisationError(label, candidate, converged=False, steps=100)

    with pytest.raises(RuntimeError, match="NEB did not converge"):
        run(
            multiwell_seed(),
            AnalyticPotential(),
            max_steps=100,
            intermediate_relaxer=unstable,
            intermediate_refinement_callback=lambda *args: pytest.fail("Unstable candidate used"),
            intermediate_minimum_callback=lambda *args: retained.append(args),
        )
    assert instances[0].nsteps == 100
    assert len(retained) == 2
    assert all(args[-1]["status"] == "unstable" for args in retained)
    assert all(args[-1]["instability_code"] == "optimization_not_converged" for args in retained)


def test_calculator_failure_stops_neb_and_retains_failed_candidate(monkeypatch):
    instances = freeze_band(monkeypatch)
    retained = []

    def failure(candidate, label):
        raise ValueError("candidate calculator failure")

    with pytest.raises(RuntimeError, match="candidate calculator failure"):
        run(
            multiwell_seed(),
            AnalyticPotential(),
            intermediate_relaxer=failure,
            intermediate_minimum_callback=lambda *args: retained.append(args),
        )
    assert instances[0].nsteps == 0
    assert len(retained) == 1 and retained[0][-1]["status"] == "failed"


def test_pose_recheck_keeps_surface_position_atom_order_and_mic():
    original = pose(1)
    shifted = original.copy()
    shifted.positions[-1, 0] += 10
    assert neb_module._minimum_pose_matches(original, shifted, 1e-8)
    shifted.positions[-1, 0] += 0.2
    assert not neb_module._minimum_pose_matches(original, shifted, 0.1)
    swapped = original[::-1]
    assert not neb_module._minimum_pose_matches(original, swapped, 0.1)


def test_candidate_budget_and_disabled_checks(monkeypatch):
    freeze_band(monkeypatch)
    for interval, expected in [(10, 1), (None, 0)]:
        retained = []
        with pytest.raises(RuntimeError, match="NEB did not converge"):
            run(
                multiwell_seed(),
                AnalyticPotential(),
                max_steps=100,
                intermediate_check_interval=interval,
                intermediate_minimum_max_candidates=1,
                intermediate_minimum_callback=lambda *args: retained.append(args),
            )
        assert len(retained) == expected


def test_no_valley_observer_preserves_forces_path_and_calculator_calls(monkeypatch):
    trajectories = []

    class RecordedFIRE(FIRE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorded = []
            trajectories.append(self.recorded)

        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            self.recorded.append(
                (np.array([a.positions for a in self.atoms.images]), self.vel.copy(), self.dt)
            )

    monkeypatch.setattr(neb_module, "FIRE", RecordedFIRE)
    results, counts = [], []
    seed = [pose(x) for x in np.linspace(-1, 1, 9)]
    for i, frame in enumerate(seed[1:-1], 1):
        frame.positions[-1, 1] = 0.3 * np.sin(np.pi * i / 8)
    for interval in (None, 10):
        calculator = AnalyticPotential(multiwell=False)
        results.append(run(seed, calculator, intermediate_check_interval=interval))
        counts.append(calculator.calls)
    assert counts[0] == counts[1]
    assert results[1].minimum_check_count > 1 and not results[1].minimum_checks
    assert results[0].optimizer_steps == results[1].optimizer_steps
    assert results[0].path_energies == results[1].path_energies
    for before, after in zip(results[0].path_images, results[1].path_images):
        np.testing.assert_array_equal(before.positions, after.positions)
        np.testing.assert_array_equal(
            before.calc.results.get("forces"), after.calc.results.get("forces")
        )
    for before, after in zip(*trajectories):
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2] == after[2]


def test_check_geometries_and_unstable_outcome_survive_checkpoint_and_writer(tmp_path):
    source = pose(1)
    source.calc = SinglePointCalculator(source, energy=0, forces=np.zeros((2, 3)))
    lc = SimpleNamespace(lateral_class=0, stable=None, last_failure_reason="band failed")
    metadata = {"status": "unstable", "trigger": "periodic_minimum_check"}
    retain_periodic_minimum_check(lc, source, source, [source, source, source], metadata)
    graph = nx.Graph()
    graph.graph["minimum_diagnostics"] = lc
    state = make_checkpoint_state(step=0, time_s=0, graph=graph, adsorbate_sites=[])
    restored = load_checkpoint(save_checkpoint(tmp_path / "checkpoint.pkl", state))
    lc = restored.graph.graph["minimum_diagnostics"]
    assert lc.neb_minimum_checks == [metadata]
    assert isinstance(lc.atoms_neb_minimum_relaxed[0].calc, SinglePointCalculator)
    writer = ReactionWriter(tmp_path / "run")
    folder = writer.write_invalid_diffusion(SimpleNamespace(iso_class=0, reactant="[H]"), lc)
    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["neb_minimum_checks"] == [metadata]
    directory = folder / "minimum_checks"
    assert json.loads((directory / "results.json").read_text()) == [metadata]
    assert len(read(directory / "candidate_0000_source_band.extxyz", ":")) == 3
    relaxed = read(directory / "candidate_0000_relaxed.extxyz")
    np.testing.assert_array_equal(relaxed.positions, source.positions)
    assert relaxed.arrays["graph_node_id"].tolist() == ["slab", "reacting"]


@pytest.mark.parametrize("kind", ["diffusion", "bond"])
def test_minimum_diagnostics_roundtrip_through_calculation_cache(tmp_path, kind):
    from ogkmc.io.calculation_cache import (
        apply_cached_states,
        calculation_cache_key,
        load_calculation_record,
        make_calculation_record,
        minimum_check_geometry_records,
        state_payload,
        validate_isaac_record,
        write_calculation_record,
    )

    source = pose(1)
    source.calc = SinglePointCalculator(source, energy=0, forces=np.zeros((2, 3)))
    original = SimpleNamespace()
    metadata = {"status": "force_converged", "trigger": "periodic_minimum_check"}
    retain_periodic_minimum_check(original, source, source, [source, source, source], metadata)
    graph = nx.Graph()
    graph.add_node(0, element="Cu", type="surface")
    graph.add_node(1, element="H", type="adsorbate", reactant="[H]", reactant_index=0)
    graph.add_edge(0, 1, anchor_bond=True)
    parameters = {"neb_intermediate_check_interval": 10}
    inputs = {"initial": source}
    key = calculation_cache_key(kind=kind, identity={}, parameters=parameters, inputs=inputs)
    states = (
        ("state_a", "state_b", "transition")
        if kind == "diffusion"
        else ("state_ab", "state_c", "transition")
    )
    record = make_calculation_record(
        kind=kind,
        cache_key=key,
        operation={},
        parameters=parameters,
        inputs=inputs,
        reaction_graph=graph,
        states={name: state_payload(source, energy_ev=0) for name in states},
        lateral_attributes={"neb_minimum_checks": original.neb_minimum_checks},
        minimum_checks=minimum_check_geometry_records(original),
    )
    path = write_calculation_record(tmp_path, kind, key, record)
    validate_isaac_record(json.loads(path.read_text()))
    loaded = load_calculation_record(tmp_path, kind, key, reaction_graph=graph)
    assert loaded is not None and len(loaded["minimum_checks"]) == 1
    restored = SimpleNamespace()
    assert apply_cached_states(
        restored, loaded, {name: (f"energy_{name}", f"atoms_{name}") for name in states}
    )
    assert restored.neb_minimum_checks == [metadata]
    assert len(restored.atoms_neb_minimum_source_bands[0]) == 3
    np.testing.assert_array_equal(restored.atoms_neb_minimum_relaxed[0].positions, source.positions)
    assert restored.atoms_neb_minimum_relaxed[0].get_potential_energy() == 0
    # A corrupted diagnostic geometry invalidates its immutable cache record too.
    asset = path.parent / "minimum_candidate_0000_relaxed.extxyz"
    asset.write_text("broken")
    assert load_calculation_record(tmp_path, kind, key, reaction_graph=graph) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("neb_intermediate_check_interval", 0),
        ("neb_intermediate_check_interval", True),
        ("neb_intermediate_minimum_recheck_tolerance", -0.1),
        ("neb_intermediate_minimum_recheck_tolerance", ".nan"),
        ("neb_intermediate_minimum_max_candidates", 0),
        ("neb_intermediate_minimum_max_candidates", 1.5),
    ],
)
def test_configuration_validates_periodic_controls(tmp_path, key, value):
    config = tmp_path / "config.yaml"
    config.write_text(f"""schema_version: '1'
calculator:
  import_path: ase.calculators.emt.EMT
reactants:
- smiles: '[H]'
optimization:
  {key}: {value}
""")
    with pytest.raises(ConfigError, match=f"optimization.{key}"):
        load_config(config)


@pytest.mark.parametrize("interval", [None, 1, 10])
def test_controls_reach_both_channels_and_public_apis(tmp_path, interval):
    from ogkmc.reactions import bond, diffusion
    from ogkmc.sites.stability import bond as bond_stability, diffusion as diffusion_stability

    config = tmp_path / "config.yaml"
    config.write_text(f"""schema_version: '1'
calculator:
  import_path: ase.calculators.emt.EMT
reactants:
- smiles: '[H]'
optimization:
  neb_intermediate_check_interval: {"null" if interval is None else interval}
  neb_intermediate_minimum_recheck_tolerance: 0.2
  neb_intermediate_minimum_max_candidates: 7
diffusion:
  enabled: true
bond:
  enabled: true
""")
    cfg = load_config(config)
    runtime = resolve_channel_runtime(cfg, None)
    for channel in (runtime.diffusion, runtime.bond):
        settings = channel.to_kwargs()
        assert settings["neb_intermediate_check_interval"] == interval
        assert settings["neb_intermediate_minimum_recheck_tolerance"] == 0.2
        assert settings["neb_intermediate_minimum_max_candidates"] == 7
    for module in (bond, diffusion, bond_stability, diffusion_stability):
        funcs = [
            func
            for name, func in vars(module).items()
            if callable(func)
            and name.startswith(("check_", "get_"))
            and "neb_intermediate_stagnation_steps" in signature(func).parameters
        ]
        assert funcs
        for function in funcs:
            assert signature(function).parameters["neb_intermediate_check_interval"].default == 10


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("intermediate_check_interval", 0),
        ("intermediate_check_interval", True),
        ("intermediate_minimum_recheck_tolerance", -0.1),
        ("intermediate_minimum_recheck_tolerance", float("nan")),
        ("intermediate_minimum_max_candidates", 0),
        ("intermediate_minimum_max_candidates", True),
    ],
)
def test_direct_neb_api_validates_periodic_controls(key, value):
    with pytest.raises(ValueError, match=key):
        run(multiwell_seed(), AnalyticPotential(), **{key: value})
