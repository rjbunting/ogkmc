"""Regression tests for thermo and utility package surfaces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from ase import Atoms

import autokmc.thermo.free_energy as free_energy
from autokmc.io.calculators import CalculatorPool, calculator_batch_context


def test_thermo_and_utils_package_exports():
    from autokmc import thermo, utils

    assert thermo.FreeEnergyOptions is free_energy.FreeEnergyOptions
    assert thermo.compute_gas_thermo is free_energy.compute_gas_thermo
    assert thermo.compute_harmonic_thermo is free_energy.compute_harmonic_thermo
    assert utils.get_logger("") is logging.getLogger("autokmc")


def test_split_real_imag_ev_treats_negative_real_modes_as_imaginary():
    real_ev, imag_ev = free_energy._split_real_imag_ev(
        [
            0.150,
            -0.080,
            0.0004,
            1j * 0.090,
        ],
        min_frequency_ev=0.001,
    )

    assert real_ev == pytest.approx([0.150])
    assert imag_ev == pytest.approx([0.080, 0.0004, 0.090])


@pytest.mark.parametrize(
    ("atoms", "expected_number", "expected_point_group"),
    [
        (Atoms("He", positions=[[0.0, 0.0, 0.0]]), 1, "K_h"),
        (
            Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]),
            1,
            "C*v",
        ),
        (
            Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]),
            2,
            "D*h",
        ),
        (
            Atoms(
                "OH2",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.757, 0.586, 0.0],
                    [-0.757, 0.586, 0.0],
                ],
            ),
            2,
            "C2v",
        ),
    ],
)
def test_infer_rotational_symmetry_number(atoms, expected_number, expected_point_group):
    symmetry_number, point_group = free_energy._infer_rotational_symmetry_number(
        atoms,
        tolerance=0.3,
    )

    assert symmetry_number == expected_number
    assert point_group == expected_point_group


def test_gas_thermo_infers_and_records_rotational_symmetry(monkeypatch):
    captured: dict = {}

    def fake_vibrate(_atoms, indices, *, options, cache_dir, label):
        return [], [], []

    class FakeIdealGasThermo:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_gibbs_energy(self, *, temperature, pressure, verbose):
            return 1.25

        def get_ZPE_correction(self):
            return 0.05

        def get_entropy(self, *, temperature, pressure, verbose):
            return 0.001

    import ase.thermochemistry as ase_thermochemistry

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)
    monkeypatch.setattr(ase_thermochemistry, "IdealGasThermo", FakeIdealGasThermo)

    result = free_energy.compute_gas_thermo(
        Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]),
        energy_ev=1.0,
        temperature_k=300.0,
        pressure_bar=1.0,
        options=free_energy.FreeEnergyOptions(enabled=True, symmetry_tolerance=0.2),
    )

    assert captured["symmetrynumber"] == 2
    assert result["symmetry_number"] == 2
    assert result["symmetry_number_source"] == "inferred"
    assert result["point_group"] == "D*h"
    assert result["symmetry_tolerance"] == pytest.approx(0.2)


def test_explicit_gas_symmetry_number_bypasses_inference(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("automatic inference should not run for an explicit override")

    def fake_vibrate(_atoms, indices, *, options, cache_dir, label):
        return [], [], []

    class FakeIdealGasThermo:
        def __init__(self, **_kwargs):
            pass

        def get_gibbs_energy(self, *, temperature, pressure, verbose):
            return 1.0

        def get_ZPE_correction(self):
            return 0.0

        def get_entropy(self, *, temperature, pressure, verbose):
            return 0.0

    import ase.thermochemistry as ase_thermochemistry

    monkeypatch.setattr(free_energy, "_infer_rotational_symmetry_number", fail_if_called)
    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)
    monkeypatch.setattr(ase_thermochemistry, "IdealGasThermo", FakeIdealGasThermo)

    result = free_energy.compute_gas_thermo(
        Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]),
        energy_ev=1.0,
        temperature_k=300.0,
        pressure_bar=1.0,
        options=free_energy.FreeEnergyOptions(enabled=True),
        symmetry_number=2,
    )

    assert result["symmetry_number"] == 2
    assert result["symmetry_number_source"] == "explicit"
    assert result["point_group"] is None


def test_default_harmonic_cache_uses_temporary_directory(monkeypatch):
    seen: dict[str, Path] = {}

    def fake_vibrate(_atoms, indices, *, options, cache_dir, label):
        seen["cache_dir"] = Path(cache_dir)
        return [], [], []

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)

    result = free_energy.compute_harmonic_thermo(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        [0],
        energy_ev=1.0,
        temperature_k=300.0,
        options=free_energy.FreeEnergyOptions(enabled=True),
    )

    assert result["enabled"] is True
    assert seen["cache_dir"].name.startswith("autokmc_vib_")
    assert not seen["cache_dir"].exists()
    assert Path("_autokmc_vib_cache") != seen["cache_dir"]


def test_gas_thermo_forces_nonperiodic_vibration_snapshot(monkeypatch):
    seen: dict[str, tuple[bool, bool, bool]] = {}

    def fake_vibrate(atoms, indices, *, options, cache_dir, label):
        seen["pbc"] = tuple(bool(x) for x in atoms.pbc)
        return [], [], []

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)

    free_energy.compute_gas_thermo(
        Atoms(
            "He",
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 12.0,
            pbc=[True, True, True],
        ),
        energy_ev=0.0,
        temperature_k=300.0,
        pressure_bar=1.0,
        options=free_energy.FreeEnergyOptions(enabled=True),
        geometry="monatomic",
    )

    assert seen["pbc"] == (False, False, False)


def test_harmonic_thermo_promotes_mixed_slab_pbc(monkeypatch):
    seen: dict[str, tuple[bool, bool, bool]] = {}

    def fake_vibrate(atoms, indices, *, options, cache_dir, label):
        seen["pbc"] = tuple(bool(x) for x in atoms.pbc)
        return [], [], []

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)

    free_energy.compute_harmonic_thermo(
        Atoms(
            "CuO",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
            cell=np.eye(3) * 12.0,
            pbc=[True, True, False],
        ),
        [1],
        energy_ev=1.0,
        temperature_k=300.0,
        options=free_energy.FreeEnergyOptions(enabled=True),
    )

    assert seen["pbc"] == (True, True, True)


def test_explicit_harmonic_cache_is_preserved(monkeypatch, tmp_path):
    seen: dict[str, Path] = {}

    def fake_vibrate(_atoms, indices, *, options, cache_dir, label):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        seen["cache_dir"] = Path(cache_dir)
        return [], [], []

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)

    free_energy.compute_harmonic_thermo(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        [0],
        energy_ev=1.0,
        temperature_k=300.0,
        options=free_energy.FreeEnergyOptions(enabled=True),
        cache_dir=tmp_path,
    )

    assert seen["cache_dir"] == tmp_path
    assert tmp_path.exists()


def test_harmonic_vibration_indices_are_validated():
    with pytest.raises(ValueError, match="outside the structure"):
        free_energy.compute_harmonic_thermo(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            [1],
            energy_ev=1.0,
            temperature_k=300.0,
            options=free_energy.FreeEnergyOptions(enabled=True),
        )


def test_disabled_harmonic_thermo_remains_noop_for_unchecked_indices():
    result = free_energy.compute_harmonic_thermo(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        [1],
        energy_ev=1.0,
        temperature_k=300.0,
        options=free_energy.FreeEnergyOptions(enabled=False),
    )

    assert result["enabled"] is False
    assert result["vib_indices"] == [1]


def test_vibration_displacements_use_pool_and_reuse_completed_cache(tmp_path):
    class Tracker:
        def __init__(self):
            self.calls = 0
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        @contextmanager
        def call(self):
            with self.lock:
                self.calls += 1
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.01)
                yield
            finally:
                with self.lock:
                    self.active -= 1

    class HarmonicCalculator:
        def __init__(self, tracker):
            self.tracker = tracker

        def get_forces(self, atoms):
            with self.tracker.call():
                return -np.asarray(atoms.positions, dtype=float)

    tracker = Tracker()
    pool = CalculatorPool(
        [HarmonicCalculator(tracker), HarmonicCalculator(tracker)],
        max_workers=2,
    )
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    options = free_energy.FreeEnergyOptions(
        enabled=True,
        vibration_nfree=2,
    )

    first = free_energy._run_vibrations(
        atoms.copy(),
        [0],
        calculator=pool,
        options=options,
        cache_dir=tmp_path,
        label="parallel",
        purpose="test",
    )
    calls_after_first = tracker.calls
    second = free_energy._run_vibrations(
        atoms.copy(),
        [0],
        calculator=pool,
        options=options,
        cache_dir=tmp_path,
        label="parallel",
        purpose="test",
    )

    assert calls_after_first == 7
    assert tracker.maximum >= 2
    assert tracker.calls == calls_after_first
    assert np.asarray(first[2], dtype=complex) == pytest.approx(
        np.asarray(second[2], dtype=complex)
    )

    tracker.maximum = 0
    with calculator_batch_context():
        free_energy._run_vibrations(
            atoms.copy(),
            [0],
            calculator=pool,
            options=options,
            cache_dir=tmp_path,
            label="outer-batch",
            purpose="test",
        )
    assert tracker.maximum == 1
    pool.shutdown()


def test_vibration_cache_identity_includes_calculator_relevant_atom_arrays():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    options = free_energy.FreeEnergyOptions()

    def cache_label(candidate):
        return free_energy._vibration_cache_label(
            candidate,
            [0, 1],
            options=options,
            label="arrays",
            calculator=None,
        )

    baseline = cache_label(atoms)
    variants = []
    with_charges = atoms.copy()
    with_charges.set_initial_charges([0.1, -0.1])
    variants.append(with_charges)
    with_magmoms = atoms.copy()
    with_magmoms.set_initial_magnetic_moments([1.0, 0.0])
    variants.append(with_magmoms)
    with_tags = atoms.copy()
    with_tags.set_tags([1, 0])
    variants.append(with_tags)
    with_custom_array = atoms.copy()
    with_custom_array.new_array("calculator_state", np.array([2, 3]))
    variants.append(with_custom_array)

    assert all(cache_label(candidate) != baseline for candidate in variants)


def test_same_label_vibration_runs_are_serialized_and_reuse_cache(
    tmp_path,
    monkeypatch,
):
    class Tracker:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def record(self):
            with self.lock:
                self.calls += 1
            time.sleep(0.01)

    class HarmonicCalculator:
        def __init__(self, tracker):
            self.tracker = tracker

        def get_forces(self, atoms):
            self.tracker.record()
            return -np.asarray(atoms.positions, dtype=float)

    from ase.vibrations import Vibrations

    monkeypatch.setattr(
        Vibrations,
        "clean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("active displacement locks must not be deleted")
        ),
    )
    tracker = Tracker()
    pool = CalculatorPool(
        [HarmonicCalculator(tracker), HarmonicCalculator(tracker)],
        max_workers=2,
    )
    atoms = Atoms("H", positions=[[0.2, 0.0, 0.0]])
    options = free_energy.FreeEnergyOptions(
        enabled=True,
        vibration_nfree=2,
    )

    def run():
        return free_energy._run_vibrations(
            atoms.copy(),
            [0],
            calculator=pool,
            options=options,
            cache_dir=tmp_path,
            label="same-label",
            purpose="test",
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(run)
            second_future = executor.submit(run)
            first = first_future.result()
            second = second_future.result()
    finally:
        pool.shutdown()

    assert tracker.calls == 7
    assert np.asarray(first[2], dtype=complex) == pytest.approx(
        np.asarray(second[2], dtype=complex)
    )


def test_ephemeral_vibrations_skip_content_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        free_energy,
        "_vibration_cache_label",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral cache must not build a persistent identity")
        ),
    )
    labels = []

    def fake_vibrate(
        atoms,
        indices,
        *,
        options,
        cache_dir,
        label,
    ):
        del atoms, indices, options, cache_dir
        labels.append(label)
        return [], [], []

    monkeypatch.setattr(free_energy, "_vibrate", fake_vibrate)
    result = free_energy._run_vibrations(
        Atoms("H"),
        [0],
        calculator=None,
        options=free_energy.FreeEnergyOptions(),
        cache_dir=tmp_path,
        label="ephemeral",
        purpose="test",
        persistent_cache=False,
    )

    assert result == ([], [], [])
    assert labels == ["ephemeral"]
