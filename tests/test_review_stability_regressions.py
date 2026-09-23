"""Failure recovery and physical identity at stability calculation boundaries."""

from types import SimpleNamespace
import threading

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from ogkmc.io.calculators import CalculatorPool
from ogkmc.sites.adsorbate import AdsorbateSiteLateral
from ogkmc.sites.stability.adsorption import check_site_stability
from ogkmc.sites.stability.diffusion import NEBNotConvergedError
from ogkmc.sites.stability.intermediate_pruning import (
    CompositeDirectEventDetected,
    retain_refinement_and_maybe_suppress,
)
from ogkmc.thermo import free_energy


class Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, tracker=None):
        super().__init__()
        self.tracker = tracker

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.tracker is not None:
            with self.tracker.lock:
                self.tracker.calls += 1
                if self.tracker.calls == 2:
                    raise RuntimeError("transient force backend interruption")
        p = atoms.positions
        self.results = {"energy": float(0.5 * (p * p).sum()), "forces": -p.copy()}


@pytest.mark.parametrize("parallel", [False, True])
def test_interrupted_vibration_cache_recovers_and_reuses_completed_displacements(
    tmp_path, parallel
):
    tracker = SimpleNamespace(calls=0, lock=threading.Lock())
    calculators = [Harmonic(tracker), Harmonic(tracker)]
    calculator = CalculatorPool(calculators, max_workers=2) if parallel else calculators[0]
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    kwargs = dict(
        energy_ev=0.0,
        temperature_k=300.0,
        options=free_energy.FreeEnergyOptions(),
        calculator=calculator,
        cache_dir=tmp_path,
    )
    try:
        with pytest.raises(RuntimeError, match="transient force backend interruption"):
            free_energy.compute_harmonic_thermo(atoms, [0], **kwargs)
        assert any(p.stat().st_size == 0 for p in tmp_path.rglob("*.json"))
        recovered = free_energy.compute_harmonic_thermo(atoms, [0], **kwargs)
        calls_after_recovery = tracker.calls
        repeated = free_energy.compute_harmonic_thermo(atoms, [0], **kwargs)
        assert np.isfinite(recovered["g_total_ev"])
        assert recovered["g_corr_ev"] == repeated["g_corr_ev"]
        assert tracker.calls == calls_after_recovery
        assert not any(p.stat().st_size == 0 for p in tmp_path.rglob("*.json"))
    finally:
        if parallel:
            calculator.shutdown()


@pytest.mark.parametrize("kind", ["harmonic", "gas"])
def test_thermochemistry_accepts_an_attached_calculator(kind):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = Harmonic()
    atoms.calc = calculator
    kwargs = dict(energy_ev=0.0, temperature_k=300.0, options=free_energy.FreeEnergyOptions())
    if kind == "harmonic":
        function = lambda **extra: free_energy.compute_harmonic_thermo(
            atoms, [0], **kwargs, **extra
        )
    else:
        function = lambda **extra: free_energy.compute_gas_thermo(
            atoms, pressure_bar=1.0, **kwargs, **extra
        )
    attached = function()
    explicit = function(calculator=calculator)
    assert attached["g_total_ev"] == pytest.approx(explicit["g_total_ev"])
    assert atoms.calc is calculator


@pytest.mark.parametrize("masses, expected", [([1.0, 1.0], 2), ([2.0, 2.0], 2), ([1.0, 2.0], 1)])
def test_linear_rotational_symmetry_respects_isotopes(masses, expected):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], masses=masses)
    assert free_energy._infer_rotational_symmetry_number(atoms, tolerance=0.1)[0] == expected


@pytest.mark.parametrize("name, expected", [("H2O", 1), ("CH4", 3)])
def test_nonlinear_rotational_symmetry_respects_isotopes(name, expected):
    atoms = molecule(name)
    masses = atoms.get_masses()
    masses[np.flatnonzero(atoms.numbers == 1)[0]] = 2.014102
    atoms.set_masses(masses)
    assert free_energy._infer_rotational_symmetry_number(atoms, tolerance=0.1)[0] == expected


class Zero(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def test_fully_frozen_unoccupied_slab_is_a_valid_stability_endpoint():
    graph = nx.Graph(cell=np.eye(3) * 20)
    graph.add_node(0, type="surface", element="Cu", index=0, position=[0, 0, 0])
    graph.add_node(
        1,
        type="adsorbate",
        element="H",
        position=[0, 0, 1.5],
        clique=frozenset([0]),
        occupied=False,
        reactant_index=0,
        reactant="[H]",
        iso_class=0,
        siblings=[],
    )
    graph.add_edge(0, 1)
    site = SimpleNamespace(iso_class=0, member_node_ids=[[1]], reactant="[H]")
    lateral = AdsorbateSiteLateral(0, ego_graph=graph.copy())
    check_site_stability(graph, site, 0, lateral, Zero(), frozen_indices=[0], max_steps=2)
    assert lateral.stable is True
    assert lateral.energy_occupied == lateral.energy_unoccupied == 0.0


def test_composite_direct_event_control_exception_survives_neb_boundary(monkeypatch):
    from ogkmc.sites.stability import neb as nebmod

    class Converged:
        def __init__(self, atoms, logfile=None):
            self.nsteps = 0

        def run(self, *, fmax, steps):
            pass

        def converged(self):
            return True

        def attach(self, *args, **kwargs):
            pass

    monkeypatch.setattr(nebmod, "BFGS", Converged)

    def band_factory(initial, final, **kwargs):
        images = []
        for i, energy in enumerate([0.0, 0.8, 0.2, 1.0, 0.0]):
            image = Atoms("H", positions=[[i, 0, 0]])
            image.calc = SinglePointCalculator(image, energy=energy, forces=np.zeros((1, 3)))
            images.append(image)
        return SimpleNamespace(climb=False), images

    lateral = SimpleNamespace()

    def callback(initial, final, metadata):
        retain_refinement_and_maybe_suppress(
            lateral,
            initial,
            final,
            metadata,
            {"reason": "test_registered_intermediate"},
        )

    with pytest.raises(CompositeDirectEventDetected):
        nebmod.run_neb(
            Atoms("H", positions=[[0, 0, 0]]),
            Atoms("H", positions=[[4, 0, 0]]),
            calculator=None,
            purpose="regression",
            n_images=3,
            interpolation="linear",
            spring_k=0.1,
            climb=False,
            frozen_indices=None,
            fmax=0.05,
            max_steps=10,
            verbose=False,
            not_converged_error=NEBNotConvergedError,
            band_factory=band_factory,
            intermediate_stagnation_steps=2,
            intermediate_refinement_callback=callback,
            intermediate_relaxer=lambda atoms, label: (atoms, 0.2),
        )
    assert lateral.direct_event_status == "composite"
