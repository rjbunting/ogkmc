"""Focused tests for shared stability/NEB execution boundaries."""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from types import SimpleNamespace

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import networkx as nx
import numpy as np
import pytest

from autokmc.io.calculators import CalculatorPool, calculator_batch_context
from autokmc.io.calculation_cache import scientific_input_fingerprint
from autokmc.sites.stability import adsorption as adsorption_module
from autokmc.sites.stability import bond as bond_module
from autokmc.sites.stability import diffusion as diffusion_module
from autokmc.sites.stability import neb as neb_module
from autokmc.sites.stability.bond import (
    BondTransitionStateInvalidError,
    _check_bond_ts_validity,
)
from autokmc.sites.stability.diffusion import (
    TransitionStateInvalidError,
    _check_ts_validity,
)
from autokmc.utils.telemetry import RuntimeTelemetry, telemetry_context


def _image(energy: float) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=float(energy))
    return atoms


@contextmanager
def _calculator_context(calculator, *, purpose):
    del purpose
    yield calculator


class _ConvergedOptimizer:
    def __init__(self, _neb, *, logfile):
        del logfile
        self.nsteps = 4

    def run(self, *, fmax, steps):
        del fmax, steps

    def converged(self) -> bool:
        return True


def test_shared_neb_selects_transition_and_detaches_images(monkeypatch):
    images = [_image(0.0), _image(0.5), _image(1.5), _image(0.2)]
    monkeypatch.setattr(neb_module, "acquire_calculator", _calculator_context)
    monkeypatch.setattr(neb_module, "BFGS", _ConvergedOptimizer)
    telemetry = RuntimeTelemetry()

    with telemetry_context(telemetry):
        result = neb_module.run_neb(
            images[0],
            images[-1],
            calculator=object(),
            purpose="test NEB",
            n_images=2,
            interpolation="linear",
            spring_k=0.1,
            climb=True,
            frozen_indices=None,
            fmax=0.05,
            max_steps=20,
            verbose=False,
            not_converged_error=RuntimeError,
            persist_path=True,
            band_factory=lambda *_args, **_kwargs: (object(), images),
        )

    assert result.energy_ts == pytest.approx(1.5)
    assert result.transition_index == 2
    assert result.n_interior == 2
    assert result.optimizer_steps == 4
    assert result.path_energies == pytest.approx([0.0, 0.5, 1.5, 0.2])
    assert result.atoms_ts.calc is None
    assert all(image.calc is None for image in images)
    assert all(image.calc is None for image in result.path_images or [])
    assert telemetry.counters["neb.calls"] == 1
    assert telemetry.timings_s["neb.seconds"] >= 0.0


def test_neb_parallelizes_images_through_calculator_pool():
    class Tracker:
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        @contextmanager
        def call(self):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.02)
                yield
            finally:
                with self.lock:
                    self.active -= 1

    class HarmonicCalculator:
        def __init__(self, tracker):
            self.tracker = tracker

        def get_forces(self, atoms):
            with self.tracker.call():
                return -np.asarray(atoms.positions, dtype=float)

        def get_potential_energy(self, atoms, force_consistent=False):
            del force_consistent
            positions = np.asarray(atoms.positions, dtype=float)
            return 0.5 * float(np.einsum("ij,ij->", positions, positions))

    tracker = Tracker()
    pool = CalculatorPool(
        [HarmonicCalculator(tracker), HarmonicCalculator(tracker)],
        max_workers=2,
    )
    initial = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    final = Atoms("H", positions=[[1.0, 0.0, 0.0]])

    neb, images = neb_module.make_neb_band(
        initial,
        final,
        n_images=3,
        interpolation="linear",
        spring_k=0.1,
        climb=False,
        calculator=pool,
        frozen_indices=None,
    )
    forces = neb.get_forces()

    assert neb.parallel is True
    assert len({id(image.calc) for image in images}) == len(images)
    assert forces.shape == (3, 3)
    assert tracker.maximum >= 2


def test_neb_uses_one_concrete_calculator_inside_outer_batch(monkeypatch):
    images = [_image(0.0), _image(0.5), _image(1.5), _image(0.2)]
    calculators = [object(), object()]
    pool = CalculatorPool(calculators)
    received = []
    monkeypatch.setattr(neb_module, "BFGS", _ConvergedOptimizer)

    def band_factory(*_args, calculator, **_kwargs):
        received.append(calculator)
        return object(), images

    with calculator_batch_context():
        result = neb_module.run_neb(
            images[0],
            images[-1],
            calculator=pool,
            purpose="outer batch NEB",
            n_images=2,
            interpolation="linear",
            spring_k=0.1,
            climb=True,
            frozen_indices=None,
            fmax=0.05,
            max_steps=20,
            verbose=False,
            not_converged_error=RuntimeError,
            band_factory=band_factory,
        )

    assert result.energy_ts == pytest.approx(1.5)
    assert received[0] in calculators
    assert received[0] is not pool
    pool.shutdown()


@pytest.mark.parametrize(
    ("validator", "error", "energy_names", "extra"),
    [
        (
            _check_ts_validity,
            TransitionStateInvalidError,
            {"e_a": 0.0, "e_b": 0.2, "e_ts": float("nan")},
            {"n_mig": 1},
        ),
        (
            _check_bond_ts_validity,
            BondTransitionStateInvalidError,
            {"e_ab": 0.0, "e_c": 0.2, "e_ts": float("nan")},
            {"n_react": 1},
        ),
    ],
)
def test_transition_validators_reject_nonfinite_energy(
    validator,
    error,
    energy_names,
    extra,
):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(error, match="not finite"):
        validator(
            atoms,
            atoms.copy(),
            atoms.copy(),
            n_slab=0,
            n_lat=0,
            nl_mult=1.2,
            ts_index=1,
            n_interior=1,
            **energy_names,
            **extra,
        )


def _gas_reactant(**overrides):
    values = {
        "smiles": "CO",
        "atoms": Atoms(
            "CO",
            positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]],
        ),
        "energy": -14.0,
        "gibbs_energy": -13.6,
        "zpe": 0.13,
        "entropy": 0.001,
        "frequencies_ev": [0.10, 0.20],
        "imaginary_ev": [-0.01],
        "partial_pressure_bar": 1.0,
    }
    values.update(overrides)
    return type("GasReactant", (), values)()


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("energy", -13.9),
        ("gibbs_energy", -13.5),
        ("zpe", 0.14),
        ("entropy", 0.002),
        ("frequencies_ev", [0.11, 0.20]),
        ("imaginary_ev", [-0.02]),
    ],
)
def test_gas_product_cache_identity_includes_consumed_thermochemistry(
    field,
    changed,
):
    baseline = _gas_reactant()
    modified = _gas_reactant(**{field: changed})

    first = bond_module._gas_product_cache_inputs(
        baseline,
        include_thermochemistry=True,
    )
    second = bond_module._gas_product_cache_inputs(
        modified,
        include_thermochemistry=True,
    )

    assert scientific_input_fingerprint(first) != scientific_input_fingerprint(second)


def test_gas_product_cache_identity_includes_atoms_but_not_live_pressure():
    baseline = _gas_reactant()
    moved_atoms = baseline.atoms.copy()
    moved_atoms.positions[1, 0] += 0.2

    first = bond_module._gas_product_cache_inputs(
        baseline,
        include_thermochemistry=True,
    )
    changed_geometry = bond_module._gas_product_cache_inputs(
        _gas_reactant(atoms=moved_atoms),
        include_thermochemistry=True,
    )
    changed_pressure = bond_module._gas_product_cache_inputs(
        _gas_reactant(partial_pressure_bar=4.0),
        include_thermochemistry=True,
    )

    assert (
        scientific_input_fingerprint(first)
        != scientific_input_fingerprint(changed_geometry)
    )
    assert (
        scientific_input_fingerprint(first)
        == scientific_input_fingerprint(changed_pressure)
    )
    assert "partial_pressure_bar" not in first


def test_gas_product_pressure_is_restamped_from_current_reactant():
    lateral = type(
        "Lateral",
        (),
        {"gas_product": True, "gas_pressure_bar": 99.0},
    )()
    site = type(
        "BondSite",
        (),
        {"gas_product": True, "gas_reactant": _gas_reactant(
            partial_pressure_bar=0.35,
        )},
    )()

    bond_module._stamp_gas_product_runtime_state(lateral, site)

    assert lateral.gas_product is True
    assert lateral.gas_pressure_bar == pytest.approx(0.35)


def _thermochemistry_options():
    return SimpleNamespace(
        enabled=True,
        vibration_displacement=0.01,
        vibration_nfree=2,
        include_ts_vibrations=True,
        min_frequency_ev=1.0e-4,
        symmetry_tolerance=1.0e-3,
        default_spin=0.0,
        default_geometry="nonlinear",
    )


def _fake_harmonic_thermo(
    _atoms,
    indices,
    *,
    energy_ev,
    temperature_k,
    **_kwargs,
):
    correction = float(temperature_k) / 1000.0
    return {
        "g_corr_ev": correction,
        "g_total_ev": float(energy_ev) + correction,
        "zpe_ev": 0.1,
        "entropy_ev_per_k": 0.001,
        "frequencies_ev": [0.1],
        "imaginary_ev": [],
        "vib_indices": list(indices),
    }


def _electronic_record(states):
    return {
        "_cache_match": "electronic",
        "states": {
            name: {
                "atoms": atoms,
                "energy_ev": energy,
                "properties": {"stale_thermochemistry": -999.0},
            }
            for name, (atoms, energy) in states.items()
        },
    }


def test_adsorption_thermochemistry_reuses_cached_electronic_states(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_node(1)
    site = SimpleNamespace(
        member_node_ids=[[1]],
        iso_class=4,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=2,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "occupied": (Atoms("H2"), -2.0),
            "unoccupied": (Atoms("H"), -1.0),
        }
    )
    writes = []
    thermo_calls = []

    def _build(*_args, include_self, **_kwargs):
        if include_self:
            return Atoms("H2"), 1, 0, 1
        return Atoms("H"), 1, 0, 0

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    def _load(*_args, **kwargs):
        assert kwargs["allow_electronic_match"] is True
        return record

    monkeypatch.setattr(adsorption_module, "_build_stability_atoms", _build)
    monkeypatch.setattr(
        adsorption_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        adsorption_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(adsorption_module, "load_calculation_record", _load)
    monkeypatch.setattr(
        adsorption_module,
        "_write_adsorption_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = adsorption_module.check_site_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=700.0,
    )

    assert result == pytest.approx((-2.0, -1.0))
    assert thermo_calls == [700.0, 700.0]
    assert lateral.g_occupied == pytest.approx(-1.3)
    assert lateral.stale_thermochemistry is None
    assert len(writes) == 1


def test_diffusion_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2))
    endpoint_a = SimpleNamespace()
    endpoint_b = SimpleNamespace()
    site = SimpleNamespace(
        member_node_ids=[([1], [2])],
        members=[(endpoint_a, 0, endpoint_b, 0)],
        iso_class=5,
        reactant="[H]",
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=3,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "state_a": (Atoms("H2"), -2.0),
            "state_b": (Atoms("H2"), -1.8),
            "transition": (Atoms("H2"), -1.0),
        }
    )
    writes = []
    thermo_calls = []

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    monkeypatch.setattr(
        diffusion_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        diffusion_module,
        "_build_diffusion_atoms",
        lambda *_args, **_kwargs: (Atoms("H2"), 1, 0, [1], [1]),
    )
    monkeypatch.setattr(
        diffusion_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        diffusion_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        diffusion_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        diffusion_module,
        "_write_diffusion_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        diffusion_module,
        "_relax_endpoint",
        lambda *_args, **_kwargs: pytest.fail("endpoint relaxation reran"),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = diffusion_module.check_diffusion_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=650.0,
    )

    assert result == pytest.approx((-2.0, -1.8, -1.0))
    assert thermo_calls == [650.0, 650.0, 650.0]
    assert lateral.g_ts == pytest.approx(-0.35)
    assert lateral.stale_thermochemistry is None
    assert len(writes) == 1


def test_bond_thermochemistry_reuses_cached_endpoints_and_neb(
    monkeypatch,
):
    import autokmc.thermo.free_energy as free_energy_module

    graph = nx.Graph()
    graph.add_nodes_from((1, 2, 3))
    endpoint_a = SimpleNamespace(member_node_ids=[[1]])
    endpoint_b = SimpleNamespace(member_node_ids=[[2]])
    endpoint_c = SimpleNamespace(member_node_ids=[[3]])
    template = SimpleNamespace(
        smiles_a="[H]",
        smiles_b="[H]",
        smiles_c="[H][H]",
    )
    site = SimpleNamespace(
        member_node_ids=[([1], [2], [3])],
        members=[(endpoint_a, 0, endpoint_b, 0, endpoint_c, 0)],
        iso_class=6,
        template=template,
        gas_product=False,
        gas_lift_height=6.0,
    )
    lateral = SimpleNamespace(
        n_shells=1,
        lateral_class=4,
        ego_graph=nx.Graph(),
    )
    record = _electronic_record(
        {
            "state_ab": (Atoms("H3"), -3.0),
            "state_c": (Atoms("H3"), -3.5),
            "transition": (Atoms("H3"), -2.0),
        }
    )
    writes = []
    thermo_calls = []

    def _thermo(*args, **kwargs):
        thermo_calls.append(float(kwargs["temperature_k"]))
        return _fake_harmonic_thermo(*args, **kwargs)

    monkeypatch.setattr(
        bond_module,
        "_member_clique_union",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        bond_module,
        "_build_bond_atoms",
        lambda *_args, **_kwargs: (
            Atoms("H3"),
            1,
            0,
            [1, 2],
            [1, 2],
            {},
        ),
    )
    monkeypatch.setattr(
        bond_module,
        "_ordered_endpoint_nodes",
        lambda _graph, node_ids: list(node_ids),
    )
    monkeypatch.setattr(
        bond_module,
        "normalise_reaction_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        bond_module,
        "calculator_identity",
        lambda _calculator: {"class": "test.Calculator"},
    )
    monkeypatch.setattr(
        bond_module,
        "load_calculation_record",
        lambda *_args, **_kwargs: record,
    )
    monkeypatch.setattr(
        bond_module,
        "_write_bond_calculation_cache",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        bond_module,
        "_relax_bond_endpoint",
        lambda *_args, **_kwargs: pytest.fail("endpoint relaxation reran"),
    )
    monkeypatch.setattr(
        free_energy_module,
        "compute_harmonic_thermo",
        _thermo,
    )

    result = bond_module.check_bond_site_stability(
        graph,
        site,
        0,
        lateral,
        object(),
        calculation_cache_root="/tmp/test-cache",
        free_energy_options=_thermochemistry_options(),
        free_energy_temperature_k=600.0,
    )

    assert result == pytest.approx((-3.0, -3.5, -2.0))
    assert thermo_calls == [600.0, 600.0, 600.0]
    assert lateral.g_ab == pytest.approx(-2.4)
    assert lateral.stale_thermochemistry is None
    assert len(writes) == 1
