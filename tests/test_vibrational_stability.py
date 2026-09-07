"""Reject nonmetastable states using the vibrational work already requested."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from autokmc.core.graph import build_graph
from autokmc.sites.adsorbate import AdsorbateSite, AdsorbateSiteLateral
from autokmc.sites.bond import BondReactionLateral
from autokmc.sites.diffusion import DiffusionLateral
from autokmc.sites.stability.adsorption import SiteStabilityError, check_site_stability
from autokmc.sites.stability.bond import (
    BondEndpointStabilityError,
    BondTransitionStateInvalidError,
    _apply_bond_thermochemistry,
)
from autokmc.sites.stability.diffusion import (
    EndpointStabilityError,
    TransitionStateInvalidError,
    _apply_diffusion_thermochemistry,
)
from autokmc.thermo import free_energy


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        curvature = np.asarray(atoms.info.get("curvature", [1., 1., 1.]))
        self.results = {
            "energy": float(np.sum(.5 * curvature * atoms.positions**2)),
            "forces": -curvature * atoms.positions,
        }


def _state(curvature):
    atoms = Atoms("H", positions=[[0., 0., 0.]], cell=[20., 20., 20.], pbc=True)
    atoms.info["curvature"] = list(curvature)
    return atoms


@pytest.mark.parametrize(
    ("kind", "curvature", "accepted"),
    [
        ("minimum", [1., 1., 1.], True),
        ("minimum", [-1., 1., 1.], False),
        ("minimum", [-1., -1., 1.], False),
        ("transition_state", [-1., 1., 1.], True),
        ("transition_state", [1., 1., 1.], False),
        ("transition_state", [-1., -1., 1.], False),
        ("diffusion_transition_state", [1., 1., 1.], True),
        ("diffusion_transition_state", [-1., 1., 1.], True),
        ("diffusion_transition_state", [-1., -1., 1.], False),
    ],
)
def test_real_finite_difference_hessians_are_validated(kind, curvature, accepted):
    kwargs = dict(
        energy_ev=0., temperature_k=300., calculator=QuadraticCalculator(),
        stationary_point=kind,
    )
    if accepted:
        result = free_energy.compute_harmonic_thermo(_state(curvature), [0], **kwargs)
        assert np.isfinite(result["g_total_ev"])
        if kind == "transition_state":
            assert len(result["frequencies_ev"]) == 2
            assert len(result["imaginary_ev"]) == 1
    else:
        with pytest.raises(free_energy.VibrationalStabilityError, match="expected"):
            free_energy.compute_harmonic_thermo(_state(curvature), [0], **kwargs)


def test_noise_threshold_does_not_count_positive_modes_filtered_from_thermo(monkeypatch):
    raw = [0.0009, 0.001j, 0.1]
    monkeypatch.setattr(
        free_energy, "_run_vibrations",
        lambda *_args, **_kwargs: ([0.1], [0.0009, 0.001], raw),
    )
    options = free_energy.FreeEnergyOptions(
        min_frequency_ev=0.01, imaginary_mode_tolerance_ev=0.0015,
    )
    result = free_energy.compute_harmonic_thermo(
        _state([1, 1, 1]), [0], energy_ev=0., temperature_k=300., options=options,
    )
    assert result["frequencies_ev"] == [0.1]
    options.imaginary_mode_tolerance_ev = 0.0005
    with pytest.raises(free_energy.VibrationalStabilityError) as error:
        free_energy.compute_harmonic_thermo(
            _state([1, 1, 1]), [0], energy_ev=0., temperature_k=300., options=options,
        )
    assert error.value.imaginary_ev == [0.001]
    assert error.value.significant_imaginary_ev == [0.001]


def test_gas_hessian_is_validated_before_ideal_gas_mode_selection(monkeypatch):
    monkeypatch.setattr(
        free_energy, "_run_vibrations",
        lambda *_args, **_kwargs: ([0.2], [0.05], [0.2, 0.05j, 0., 0., 0., 0.]),
    )
    with pytest.raises(free_energy.VibrationalStabilityError, match="gas_H2"):
        free_energy.compute_gas_thermo(
            Atoms("H2", positions=[[0, 0, 0], [0, 0, .75]]),
            energy_ev=0., temperature_k=300., pressure_bar=1., symmetry_number=2,
        )


def _apply_channel(channel, lc, *, endpoint_curvature, ts_curvature, options=None):
    endpoint = _state(endpoint_curvature)
    ts = _state(ts_curvature)
    options = options or free_energy.FreeEnergyOptions()
    common = dict(
        atoms_ts=ts, energy_ts=0., n_slab=0, n_lateral=0,
        calculator=QuadraticCalculator(), free_energy_options=options,
        temperature_k=300., vib_cache_root=None,
    )
    if channel == "diffusion":
        _apply_diffusion_thermochemistry(
            lc, SimpleNamespace(reactant="[H]", iso_class=0),
            atoms_a=endpoint, atoms_b=_state([1, 1, 1]), energy_a=0., energy_b=0.,
            n_migrating=1, **common,
        )
    else:
        site = SimpleNamespace(
            iso_class=0,
            template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"),
        )
        _apply_bond_thermochemistry(
            lc, site, atoms_ab=endpoint, atoms_c=_state([1, 1, 1]),
            energy_ab=0., energy_c=0., n_reacting=1, gas_product=False, **common,
        )


@pytest.mark.parametrize(
    ("channel", "lateral_type", "endpoint_error", "ts_error"),
    [
        ("diffusion", DiffusionLateral, EndpointStabilityError, TransitionStateInvalidError),
        ("bond", BondReactionLateral, BondEndpointStabilityError, BondTransitionStateInvalidError),
    ],
)
def test_channels_preserve_rejected_modes_and_raise_scientific_invalid_errors(
    channel, lateral_type, endpoint_error, ts_error,
):
    lc = lateral_type(lateral_class=0)
    with pytest.raises(endpoint_error):
        _apply_channel(channel, lc, endpoint_curvature=[-1, 1, 1], ts_curvature=[-1, 1, 1])
    assert lc.stable is False
    assert "significant imaginary" in lc.invalid_reason
    suffix = "a" if channel == "diffusion" else "ab"
    assert len(getattr(lc, f"imaginary_{suffix}_ev")) == 1

    lc = lateral_type(lateral_class=0)
    with pytest.raises(ts_error):
        _apply_channel(channel, lc, endpoint_curvature=[1, 1, 1], ts_curvature=[-1, -1, 1])
    assert lc.stable is False
    assert len(lc.imaginary_ts_ev) == 2


@pytest.mark.parametrize("channel", ["diffusion", "bond"])
def test_disabled_ts_vibrations_preserve_existing_approximation(channel):
    lc = DiffusionLateral(0) if channel == "diffusion" else BondReactionLateral(0)
    _apply_channel(
        channel, lc, endpoint_curvature=[1, 1, 1], ts_curvature=[-1, -1, 1],
        options=free_energy.FreeEnergyOptions(include_ts_vibrations=False),
    )
    assert lc.g_ts is not None
    assert lc.imaginary_ts_ev == []


def test_disabled_free_energy_does_not_run_vibrations(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled thermochemistry must not add vibrations")

    monkeypatch.setattr(free_energy, "_run_vibrations", unexpected)
    result = free_energy.compute_harmonic_thermo(
        _state([-1, -1, 1]), [0], energy_ev=0., temperature_k=300.,
        options=free_energy.FreeEnergyOptions(enabled=False),
    )
    assert result["enabled"] is False


class AdsorptionSaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        centers = np.array([[2.6, 0, 0], [0, 0, 0], [0, 0, 1.7]])[:len(atoms)]
        curvature = np.ones((len(atoms), 3))
        if len(atoms) == 3:
            curvature[2] = [-1, -1, 1]
        delta = atoms.positions - centers
        self.results = {"energy": float(np.sum(.5 * curvature * delta**2)),
                        "forces": -curvature * delta}


def _adsorption_setup():
    atoms = Atoms("Cu2H", positions=[[2.6, 0, 0], [0, 0, 0], [0, 0, 1.7]],
                  cell=[20, 20, 20], pbc=True)
    atoms.arrays["surface"] = np.array([0, 1, 2])
    graph = build_graph(atoms)
    graph.nodes[2].update(clique=frozenset([1]), siblings=[], occupied=False,
                          reactant="[H]", reactant_index=0, iso_class=0)
    site = AdsorbateSite(
        reactant="[H]", n_atoms=1, atom_cliques=[frozenset([1])],
        positions=[[0, 0, 1.7]], iso_class=0,
        members=[[frozenset([1])]], member_node_ids=[[2]],
    )
    lc = AdsorbateSiteLateral(0, ego_graph=graph.subgraph([0, 1]).copy())
    return graph, site, lc


def test_force_converged_adsorption_saddle_is_rejected():
    graph, site, lc = _adsorption_setup()
    with pytest.raises(SiteStabilityError, match="2 significant imaginary"):
        check_site_stability(
            graph, site, 0, lc, AdsorptionSaddleCalculator(), max_steps=3,
            free_energy_options=free_energy.FreeEnergyOptions(),
            free_energy_temperature_k=300.,
        )
    assert lc.stable is False
    assert len(lc.imaginary_occupied_ev) == 2
    assert lc.atoms_occupied is not None


def test_unchecked_cached_thermochemistry_cannot_bypass_validation(tmp_path, monkeypatch):
    graph, site, permissive = _adsorption_setup()
    calculator = AdsorptionSaddleCalculator()
    kwargs = dict(
        max_steps=3, free_energy_temperature_k=300.,
        calculation_cache_root=str(tmp_path / "calculations"),
        calculation_cache_lookup_enabled=True, vib_cache_root=str(tmp_path / "vibrations"),
    )
    # Simulate a record written before the acceptance policy was fingerprinted.
    with monkeypatch.context() as old_policy:
        old_policy.setattr(free_energy, "vibrational_validation_parameters", lambda _opts: {})
        check_site_stability(
            graph, site, 0, permissive, calculator,
            free_energy_options=free_energy.FreeEnergyOptions(imaginary_mode_tolerance_ev=.1),
            **kwargs,
        )
    assert permissive.stable is True

    def no_new_relaxation(*_args, **_kwargs):
        raise AssertionError("expected compatible electronic states from the old cache")

    import autokmc.structure
    monkeypatch.setattr(autokmc.structure, "optimise_structure", no_new_relaxation)
    strict = AdsorbateSiteLateral(0, ego_graph=graph.subgraph([0, 1]).copy())
    with pytest.raises(SiteStabilityError, match="2 significant imaginary"):
        check_site_stability(
            graph, site, 0, strict, calculator,
            free_energy_options=free_energy.FreeEnergyOptions(), **kwargs,
        )
    assert strict.stable is False
    assert len(strict.imaginary_occupied_ev) == 2
