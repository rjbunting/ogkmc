"""Batched NEB band evaluation: parity with the per-image path.

The batched mode must be a pure access-pattern change: same ASE NEB math,
same optimizer trajectory, same barriers — one stacked evaluation per step
instead of per-image calculator calls.  These tests prove that contract on
EMT, where both paths share identical arithmetic, so agreement is exact
rather than statistical.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from autokmc.io.calculators import CalculatorPool
from autokmc.sites.stability.band_eval import (
    BandEvaluationError,
    BandImageCalculator,
    CallableBandEvaluator,
    resolve_band_evaluator,
)
from autokmc.sites.stability.neb import (
    DEFAULT_NEB_BAND_EVAL,
    normalize_band_eval,
    run_neb,
)


class _NEBFailed(RuntimeError):
    pass


class _BandBatchingEMT:
    """EMT exposing the ``evaluate_band`` opt-in protocol.

    Evaluates images with plain EMT arithmetic (so results are bit-identical
    to the serial path) while recording how the band API is exercised.
    """

    def __init__(self):
        self._inner = EMT()
        self.band_calls = 0
        self.band_sizes: list[int] = []
        self.single_calls = 0

    def evaluate_band(self, images):
        self.band_calls += 1
        self.band_sizes.append(len(images))
        results = []
        for image in images:
            energy = float(self._inner.get_potential_energy(image))
            forces = np.asarray(self._inner.get_forces(image), dtype=float)
            results.append((energy, forces))
        return results

    # Single-structure fallback surface, so cache misses stay correct.
    def get_potential_energy(self, atoms, force_consistent=False):
        self.single_calls += 1
        return self._inner.get_potential_energy(atoms)

    def get_forces(self, atoms):
        self.single_calls += 1
        return self._inner.get_forces(atoms)


def _hop_endpoints() -> tuple:
    """A small O adatom fcc->hcp hop on Cu(111) with a frozen bottom layer."""
    initial = fcc111("Cu", size=(2, 2, 2), a=3.6, vacuum=6.0)
    frozen = [
        i for i, atom in enumerate(initial) if atom.tag == 2
    ]
    add_adsorbate(initial, "O", height=1.1, position="fcc")
    final = fcc111("Cu", size=(2, 2, 2), a=3.6, vacuum=6.0)
    add_adsorbate(final, "O", height=1.1, position="hcp")
    for atoms in (initial, final):
        atoms.set_constraint(FixAtoms(indices=frozen))
        atoms.calc = EMT()
    return initial, final, frozen


def _run(calculator, band_eval: str, *, frozen: list[int]):
    initial, final, _ = _hop_endpoints()
    return run_neb(
        initial,
        final,
        calculator=calculator,
        purpose="test NEB",
        n_images=3,
        interpolation="linear",
        spring_k=5.0,
        climb=True,
        frozen_indices=frozen,
        fmax=0.1,
        max_steps=200,
        verbose=False,
        not_converged_error=_NEBFailed,
        capture_path=True,
        band_eval=band_eval,
    )


def test_batched_matches_serial_exactly():
    _, _, frozen = _hop_endpoints()
    serial = _run(EMT(), "images", frozen=frozen)
    batching_calc = _BandBatchingEMT()
    batched = _run(batching_calc, "batched", frozen=frozen)

    assert batched.transition_index == serial.transition_index
    assert batched.optimizer_steps == serial.optimizer_steps
    assert batched.energy_ts == pytest.approx(serial.energy_ts, abs=1e-9)
    assert np.allclose(
        batched.path_energies, serial.path_energies, atol=1e-9
    )
    assert np.allclose(
        batched.atoms_ts.positions, serial.atoms_ts.positions, atol=1e-8
    )


def test_batched_evaluates_whole_band_once_per_step():
    _, _, frozen = _hop_endpoints()
    calc = _BandBatchingEMT()
    result = _run(calc, "batched", frozen=frozen)

    assert calc.band_calls > 0
    # First call carries endpoints (n_interior + 2); later calls only the
    # interior images that actually moved.
    assert calc.band_sizes[0] == result.n_interior + 2
    assert all(size <= result.n_interior for size in calc.band_sizes[1:])
    # One band evaluation per geometry change — repeated ``get_forces``
    # requests at unchanged positions are served from the primed caches, so
    # batched call counts track optimizer steps (both climb stages), not
    # ASE's raw force-request count.
    assert calc.band_calls <= result.optimizer_steps + 4
    # Nothing leaked through the single-structure fallback path.
    assert calc.single_calls == 0


def test_batched_neb_uses_only_one_calculator_from_pool():
    _, _, frozen = _hop_endpoints()
    calculators = [_BandBatchingEMT(), _BandBatchingEMT()]
    pool = CalculatorPool(calculators, max_workers=2)

    try:
        result = _run(pool, "batched", frozen=frozen)
    finally:
        pool.shutdown()

    assert result.path_energies is not None
    assert result.energy_ts == pytest.approx(
        result.path_energies[result.transition_index]
    )
    active = [calculator for calculator in calculators if calculator.band_calls]
    assert len(active) == 1
    assert active[0].single_calls == 0


def test_batched_honors_fix_atoms():
    _, _, frozen = _hop_endpoints()
    calc = _BandBatchingEMT()
    result = _run(calc, "batched", frozen=frozen)
    initial, _, _ = _hop_endpoints()
    assert np.allclose(
        result.atoms_ts.positions[frozen],
        initial.positions[frozen],
        atol=1e-12,
    )


def test_batched_falls_back_for_plain_calculator():
    _, _, frozen = _hop_endpoints()
    serial = _run(EMT(), "images", frozen=frozen)
    fallback = _run(EMT(), "batched", frozen=frozen)
    assert fallback.energy_ts == pytest.approx(serial.energy_ts, abs=1e-9)
    assert fallback.optimizer_steps == serial.optimizer_steps


def test_resolve_band_evaluator():
    assert resolve_band_evaluator(None) is None
    assert resolve_band_evaluator(EMT()) is None
    evaluator = resolve_band_evaluator(_BandBatchingEMT())
    assert isinstance(evaluator, CallableBandEvaluator)


def test_fairchem_evaluator_validates_before_conversion(monkeypatch):
    events = []

    atomic_data_module = ModuleType("fairchem.core.datasets.atomic_data")
    atomic_data_module.atomicdata_list_to_batch = lambda items: items
    for name in ("fairchem", "fairchem.core", "fairchem.core.datasets"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "fairchem.core.datasets.atomic_data",
        atomic_data_module,
    )

    class _Predictor:
        device = None

        def validate_atoms_data(self, atoms, task_name):
            events.append(("validate", task_name, atoms.info.get("spin")))
            atoms.info.setdefault("spin", 1)

        def predict(self, data_list):
            events.append(("predict", len(data_list)))
            return {
                "energy": np.arange(len(data_list), dtype=float),
                "forces": np.zeros(
                    (sum(len(atoms) for atoms in data_list), 3),
                    dtype=float,
                ),
            }

    def convert(atoms):
        events.append(("convert", atoms.info["spin"]))
        return atoms

    calculator = SimpleNamespace(
        predictor=_Predictor(),
        a2g=convert,
        task_name="omol",
    )
    images = [Atoms("H"), Atoms("H2")]
    evaluator = resolve_band_evaluator(calculator)

    assert evaluator is not None
    results = evaluator.evaluate_band(images)

    assert events == [
        ("validate", "omol", None),
        ("convert", 1),
        ("validate", "omol", None),
        ("convert", 1),
        ("predict", 2),
    ]
    assert [energy for energy, _forces in results] == [0.0, 1.0]
    assert all(image.info["spin"] == 1 for image in images)


def test_fairchem_evaluator_requires_validation_surface():
    calculator = SimpleNamespace(
        predictor=SimpleNamespace(predict=lambda _batch: {}),
        a2g=lambda atoms: atoms,
        task_name="oc20",
    )
    assert resolve_band_evaluator(calculator) is None


def test_band_evaluator_output_is_validated():
    class _BadShape:
        def evaluate_band(self, images):
            return [(0.0, np.zeros((1, 3)))] * len(images)

    initial, _, _ = _hop_endpoints()
    evaluator = CallableBandEvaluator(_BadShape())
    with pytest.raises(BandEvaluationError):
        evaluator.evaluate_band([initial])

    class _WrongCount:
        def evaluate_band(self, images):
            return []

    evaluator = CallableBandEvaluator(_WrongCount())
    with pytest.raises(BandEvaluationError):
        evaluator.evaluate_band([initial])


def test_band_image_calculator_cache_and_fallback():
    initial, _, _ = _hop_endpoints()
    inner = EMT()
    facade = BandImageCalculator(inner)
    energy = float(inner.get_potential_energy(initial))
    forces = np.asarray(inner.get_forces(initial), dtype=float)
    facade.store(initial, energy, forces)
    assert facade.get_potential_energy(initial) == pytest.approx(energy)
    assert np.allclose(facade.get_forces(initial), forces)
    # A different geometry misses the cache and falls back to the inner
    # calculator instead of serving stale values.
    moved = initial.copy()
    moved.positions[-1] += (0.1, 0.0, 0.0)
    assert facade.get_potential_energy(moved) != pytest.approx(energy)


def test_normalize_and_default_mode():
    assert normalize_band_eval(" Batched ") == "batched"
    with pytest.raises(ValueError):
        normalize_band_eval("turbo")
    with pytest.raises(ValueError):
        normalize_band_eval("")
    assert DEFAULT_NEB_BAND_EVAL == "images"


def test_config_accepts_and_validates_band_eval():
    from autokmc.io.config import OptimizationCfg

    assert OptimizationCfg().neb_band_eval == "images"
