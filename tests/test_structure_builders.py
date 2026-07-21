"""Focused tests for shared structure-builder helpers."""

from collections import Counter

from ase import Atoms

from autokmc.structure.builders import _apply_composition


def test_apply_composition_uses_exact_constrained_largest_remainder_counts():
    atoms = Atoms("Pt10")
    composition = {"Pt": 0.40, "Pd": 0.25, "Au": 0.25, "Ag": 0.10}

    alloy = _apply_composition(atoms, composition, seed=17, verbose=False)

    assert Counter(alloy.get_chemical_symbols()) == {
        "Pt": 4,
        "Pd": 3,
        "Au": 2,
        "Ag": 1,
    }
