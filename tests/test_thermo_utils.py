"""Regression tests for thermo and utility package surfaces."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from ase import Atoms

import autokmc.thermo.free_energy as free_energy


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
