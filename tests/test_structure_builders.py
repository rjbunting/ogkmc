"""Focused tests for shared structure-builder helpers."""

from collections import Counter
from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT
import numpy as np
import pytest

from autokmc.io.calculators import CalculatorConfigError
from autokmc.io.config import load_config
from autokmc.structure.builders import _apply_composition
from autokmc.structure.slab import build_surface


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


@pytest.mark.parametrize("facet", ["111", "100"])
def test_pd_surface_examples_build_3x3_four_layer_slabs(facet):
    cfg = load_config(
        Path(__file__).parents[1]
        / "example"
        / f"h2_oxidation_pd{facet}_uma.yaml"
    )
    structure = cfg.structure

    atoms = build_surface(
        composition=structure.composition,
        crystal_structure=structure.crystal_structure,
        miller_index=structure.miller_index,
        lattice_constant=structure.lattice_constant,
        min_slab_size=structure.min_slab_size,
        min_vacuum_size=structure.min_vacuum_size,
        goal_x=structure.goal_x,
        goal_y=structure.goal_y,
        n_freeze_layers=structure.n_freeze_layers,
        calculator=EMT(),
        fmax=1.0e9,
        max_steps=1,
        verbose=False,
        **structure.extra_kwargs,
    )

    fractional_layers = np.unique(
        np.round(atoms.get_scaled_positions(wrap=False)[:, 2], decimals=8)
    )
    assert len(atoms) == 3 * 3 * 4
    assert len(fractional_layers) == 4
    assert len(atoms.info["frozen_indices"]) == 3 * 3 * 2
    if facet == "111":
        assert atoms.cell.angles()[2] == pytest.approx(60.0)


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
