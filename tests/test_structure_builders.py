"""Focused tests for shared structure-builder helpers."""

from collections import Counter
from pathlib import Path

from ase import Atoms
from ase.calculators.emt import EMT
import numpy as np
import pytest

from autokmc.core.graph import build_graph
from autokmc.io.calculators import CalculatorConfigError
from autokmc.io.config import load_config
from autokmc.sites.adsorbate import (
    _geometry_connectivity_mismatch,
    find_adsorbate_sites,
)
from autokmc.species.reactant import build_reactant
from autokmc.structure import find_surface_atoms
from autokmc.structure.builders import _apply_composition
from autokmc.structure.slab import build_surface
from autokmc.workflow.stages import configured_adsorbate_site_kwargs


def _build_example_surface(filename: str):
    cfg = load_config(
        Path(__file__).parents[1]
        / "example"
        / filename
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
    return cfg, atoms


def _build_pd_example_surface(facet: str):
    return _build_example_surface(f"h2_oxidation_pd{facet}_uma.yaml")


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


@pytest.mark.parametrize(
    "filename",
    [
        "h2_oxidation_pd111_uma.yaml",
        "h2_oxidation_pd100_uma.yaml",
        "h2_oxidation_pd111_dft.yaml",
        "co_adsorption_diffusion_cu111_uma.yaml",
        "all_options.yaml",
    ],
)
def test_surface_examples_build_4x4_four_layer_slabs(filename):
    cfg, atoms = _build_example_surface(filename)

    fractional_layers = np.unique(
        np.round(atoms.get_scaled_positions(wrap=False)[:, 2], decimals=8)
    )
    assert len(atoms) == 4 * 4 * 4
    assert len(fractional_layers) == 4
    assert len(atoms.info["frozen_indices"]) == 4 * 4 * 2
    if cfg.structure.miller_index == (1, 1, 1):
        np.testing.assert_allclose(atoms.cell.angles(), [90.0, 90.0, 60.0])
        normal = np.cross(atoms.cell[0], atoms.cell[1])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-12)


def test_skew_pd111_o_sites_are_local_and_connectivity_consistent():
    cfg, atoms = _build_pd_example_surface("111")
    constants = cfg.constants
    structure = cfg.structure
    find_surface_atoms(
        atoms,
        nl_mult=constants.neighbor_list_multiplier,
        surf_radius_factor=structure.surface_radius_factor,
        coverage_threshold=constants.raycast_coverage_threshold,
        n_disc_sample=constants.raycast_disc_samples,
        which=structure.surface_side,
        tag_atoms=True,
    )
    graph = build_graph(
        atoms,
        nl_mult=constants.neighbor_list_multiplier,
    )
    reactant = build_reactant("[O]", add_hydrogens=False, relax=False)
    site_kwargs = configured_adsorbate_site_kwargs(cfg)
    site_kwargs["prune_stable_only"] = False
    sites = find_adsorbate_sites(
        graph,
        reactant,
        calculator=None,
        verbose=False,
        **site_kwargs,
    )

    assert sorted(len(site.atom_cliques[0]) for site in sites) == [1, 2, 3, 3]
    assert len(graph.graph["raw_cliques"]["O"][3]) == 32
    assert all(
        _geometry_connectivity_mismatch(
            graph,
            site,
            reactant,
            site.positions,
            nl_mult=constants.neighbor_list_multiplier,
        )
        is None
        for site in sites
    )


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
