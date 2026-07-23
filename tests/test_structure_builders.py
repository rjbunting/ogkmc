"""Focused tests for shared structure-builder helpers."""

from collections import Counter

from ase import Atoms
from ase.calculators.emt import EMT
import pytest

from autokmc.io.calculators import CalculatorConfigError
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


def test_surface_builder_requires_an_explicit_calculator(monkeypatch):
    import autokmc.structure.slab as slab_module

    monkeypatch.setattr(slab_module, "_PMG_AVAILABLE", True)

    with pytest.raises(CalculatorConfigError, match="explicit calculator"):
        slab_module.build_surface(calculator=None)


def test_nanoparticle_helpers_require_an_explicit_calculator(monkeypatch):
    import autokmc.structure.nanoparticle as nanoparticle_module

    monkeypatch.setattr(nanoparticle_module, "_WULFF_AVAILABLE", True)

    with pytest.raises(CalculatorConfigError, match="explicit calculator"):
        nanoparticle_module.build_nanoparticle(calculator=None)
    with pytest.raises(CalculatorConfigError, match="explicit calculator"):
        nanoparticle_module.calculate_surface_energies(calculator=None)


def test_optimisation_requires_an_explicit_or_attached_calculator():
    from autokmc.structure.optimization import optimise_bulk, optimise_structure

    with pytest.raises(CalculatorConfigError, match="explicit calculator"):
        optimise_bulk("Cu", calculator=None)
    with pytest.raises(CalculatorConfigError, match="explicit calculator"):
        optimise_structure(Atoms("Cu"), calculator=None)

    atoms = Atoms("Cu")
    atoms.calc = EMT()
    relaxed = optimise_structure(atoms, calculator=None, verbose=False)

    assert isinstance(relaxed.calc, EMT)
