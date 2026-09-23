"""Tests for nanoparticle helper utilities."""

from __future__ import annotations

from ogkmc.structure import normalise_surface_energies


def test_normalise_surface_energies_accepts_yaml_string_keys():
    out = normalise_surface_energies({
        "1 1 1": 0.69,
        "(1,0,0)": "0.83",
        "110": 0.97,
    })
    assert out == {
        (1, 1, 1): 0.69,
        (1, 0, 0): 0.83,
        (1, 1, 0): 0.97,
    }
