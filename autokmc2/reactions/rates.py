"""Shared kinetic constants and Eyring-rate helpers."""

from __future__ import annotations

#: Boltzmann constant in eV / K.
KB_EV: float = 8.617_333_262e-5

#: Planck constant in eV · s.
H_EV_S: float = 4.135_667_696e-15

#: Default transmission coefficient κ for the Eyring equation.
DEFAULT_TRANSMISSION_COEFFICIENT: float = 1.0

#: Minimum activation barrier floor in eV.
EA_MIN: float = 0.1


def _eyring_prefactor(
    temperature: float,
    transmission_coefficient: float = DEFAULT_TRANSMISSION_COEFFICIENT,
) -> tuple[float, float]:
    """Return ``(prefactor, kT)`` for the Eyring rate equation."""
    kT = KB_EV * float(temperature)
    prefactor = float(transmission_coefficient) * kT / H_EV_S
    return float(prefactor), float(kT)


__all__ = [
    "KB_EV",
    "H_EV_S",
    "DEFAULT_TRANSMISSION_COEFFICIENT",
    "EA_MIN",
    "_eyring_prefactor",
]

