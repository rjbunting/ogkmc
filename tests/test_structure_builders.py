"""Focused tests for shared structure-builder helpers."""

from collections import Counter
from pathlib import Path

from ase import Atoms
from ase.build import fcc100, fcc111, make_supercell
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
    ("filename", "repeat"),
    [
        ("h2_oxidation_pd111_uma.yaml", 5),
        ("h2_oxidation_pd100_uma.yaml", 4),
        ("h2_oxidation_pd111_dft.yaml", 4),
        ("co_adsorption_diffusion_cu111_uma.yaml", 4),
        ("all_options.yaml", 4),
    ],
)
def test_surface_examples_build_expected_four_layer_slabs(filename, repeat):
    cfg, atoms = _build_example_surface(filename)

    fractional_layers = np.unique(
        np.round(atoms.get_scaled_positions(wrap=False)[:, 2], decimals=8)
    )
    atoms_per_layer = repeat**2
    assert len(atoms) == atoms_per_layer * 4
    assert len(fractional_layers) == 4
    assert len(atoms.info["frozen_indices"]) == atoms_per_layer * 2
    if cfg.structure.miller_index == (1, 1, 1):
        np.testing.assert_allclose(atoms.cell.angles(), [90.0, 90.0, 60.0])
        normal = np.cross(atoms.cell[0], atoms.cell[1])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-12)
    elif cfg.structure.miller_index == (1, 0, 0):
        np.testing.assert_allclose(atoms.cell.angles(), [90.0, 90.0, 90.0])


@pytest.mark.parametrize("facet", ["100", "111"])
@pytest.mark.parametrize("shear", [-2, -1, 1])
def test_surface_tiling_uses_shortest_in_plane_basis(monkeypatch, facet, shear):
    import autokmc.structure.slab as slab_module

    builder = fcc100 if facet == "100" else fcc111
    reference = builder("Pd", size=(1, 1, 4), a=3.89, vacuum=12.0, periodic=True)
    sheared = make_supercell(reference, [[1, 0, 0], [shear, 1, 0], [0, 0, 1]])
    # Primitive-cell generation may return any equivalent integer-sheared
    # basis. Reproduce that independently of the installed pymatgen version.
    pmg_slab = slab_module.AseAtomsAdaptor.get_structure(sheared)

    class ShearedSlabGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def get_slabs(self):
            return [pmg_slab]

    monkeypatch.setattr(slab_module, "SlabGenerator", ShearedSlabGenerator)
    _, atoms = _build_pd_example_surface(facet)
    repeat = 5 if facet == "111" else 4
    atoms_per_layer = repeat**2

    heights, layer_counts = np.unique(
        np.round(atoms.positions[:, 2], decimals=8), return_counts=True
    )
    assert len(atoms) == atoms_per_layer * 4
    assert len(heights) == 4
    np.testing.assert_array_equal(layer_counts, [atoms_per_layer] * 4)
    assert len(atoms.info["frozen_indices"]) == atoms_per_layer * 2
    np.testing.assert_allclose(
        atoms.cell.lengths()[:2],
        repeat * reference.cell.lengths()[:2],
    )
    np.testing.assert_allclose(atoms.cell[2], reference.cell[2])
    np.testing.assert_allclose(np.diff(heights), np.diff(np.unique(reference.positions[:, 2])))
    np.testing.assert_allclose(atoms.get_volume(), atoms_per_layer * reference.get_volume())
    # Square surfaces stay square; hexagonal surfaces retain their skew.
    expected_cosine = 0.0 if facet == "100" else 0.5
    np.testing.assert_allclose(
        abs(np.cos(np.deg2rad(atoms.cell.angles()[2]))), expected_cosine, atol=1.0e-12
    )


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
    assert len(graph.graph["raw_cliques"]["O"][3]) == 2 * 5**2
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
