"""Shared kinetic constants and Eyring-rate helpers."""

from __future__ import annotations

import math

from scipy import constants

from autokmc.core.constants import EA_MIN

#: Boltzmann constant in eV / K.
KB_EV: float = float(constants.physical_constants["Boltzmann constant in eV/K"][0])

#: Planck constant in eV · s.
H_EV_S: float = float(constants.physical_constants["Planck constant in eV s"][0])

#: Default transmission coefficient κ for the Eyring equation.
DEFAULT_TRANSMISSION_COEFFICIENT: float = 1.0

def _eyring_prefactor(
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float]:
    """Return ``(prefactor, kT)`` for the Eyring rate equation."""
    temperature = float(temperature)
    transmission_coefficient = float(transmission_coefficient)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0 K, got {temperature!r}")
    if not math.isfinite(transmission_coefficient) or transmission_coefficient < 0.0:
        raise ValueError(
            "transmission_coefficient must be finite and >= 0, got "
            f"{transmission_coefficient!r}"
        )
    kT = KB_EV * temperature
    prefactor = transmission_coefficient * kT / H_EV_S
    return float(prefactor), float(kT)


__all__ = [
    "KB_EV",
    "H_EV_S",
    "DEFAULT_TRANSMISSION_COEFFICIENT",
    "EA_MIN",
    "_eyring_prefactor",
]
