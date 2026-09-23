"""Thermochemical corrections and mode diagnostics for arbitrary spectra."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from ogkmc.core.graph import build_graph
from ogkmc.sites.adsorbate import AdsorbateSite, AdsorbateSiteLateral
from ogkmc.sites.bond import BondReactionLateral
from ogkmc.sites.diffusion import DiffusionLateral
from ogkmc.sites.stability.adsorption import check_site_stability
from ogkmc.sites.stability.bond import _apply_bond_thermochemistry
from ogkmc.sites.stability.diffusion import _apply_diffusion_thermochemistry
from ogkmc.thermo import free_energy

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



@pytest.mark.parametrize('curvature', [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, -1, -1]])
def test_finite_difference_spectra_produce_corrections_and_diagnostics(curvature):
    result = free_energy.compute_harmonic_thermo(
        _state(curvature), [0], energy_ev=0, temperature_k=300,
        calculator=QuadraticCalculator(),
    )
    n_real = sum(k > 0 for k in curvature)
    assert len(result['frequencies_ev']) == n_real
    assert len(result['imaginary_ev']) == 3 - n_real
    assert np.isfinite(result['g_total_ev'])
    if not n_real:
        assert result['g_corr_ev'] == result['zpe_ev'] == result['entropy_ev_per_k'] == 0


@pytest.mark.parametrize('channel', ['bond', 'diffusion'])
@pytest.mark.parametrize('ts_modes_ev', [[], [0.05335436458668682, 0.043818713155347515, 0.021693083727673677]])
def test_reaction_thermochemistry_records_endpoint_and_ts_spectra(channel, ts_modes_ev):
    lc = BondReactionLateral(0) if channel == 'bond' else DiffusionLateral(0)
    lc.stable = True
    conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
    ts_curvature = -np.square(np.asarray(ts_modes_ev) / conversion) * _state([1, 1, 1]).get_masses()[0]
    _apply_channel(
        channel, lc, endpoint_curvature=[-1, -1, 1],
        ts_curvature=ts_curvature if ts_modes_ev else [1, 1, 1],
    )
    assert lc.stable is True
    assert not lc.invalid_reason
    assert lc.imaginary_ts_ev == pytest.approx(ts_modes_ev)
    assert len(getattr(lc, 'imaginary_ab_ev' if channel == 'bond' else 'imaginary_a_ev')) == 2
    assert np.isfinite(lc.g_ts)


@pytest.mark.parametrize('symbols, positions', [
    ('H2', [[0, 0, 0], [0, 0, .75]]),
    ('OH2', [[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
])
def test_gas_thermo_handles_a_spectrum_without_real_modes(symbols, positions):
    atoms = Atoms(symbols, positions=positions)
    atoms.info['curvature'] = [-1, -1, -1]
    result = free_energy.compute_gas_thermo(
        atoms, energy_ev=0, temperature_k=300, pressure_bar=1,
        calculator=QuadraticCalculator(), symmetry_number=2,
    )
    assert result['frequencies_ev'] == []
    assert len(result['imaginary_ev']) == 3 * len(atoms)
    assert result['zpe_ev'] == 0
    assert np.isfinite(result['g_total_ev'])


def test_gas_thermo_retains_the_ideal_gas_mode_count(monkeypatch):
    monkeypatch.setattr(free_energy, '_run_vibrations',
                        lambda *_args, **_kwargs: ([.01, .02, .2], [.04], [.04j, 0, 0, .01, .02, .2]))
    result = free_energy.compute_gas_thermo(
        Atoms('H2', positions=[[0, 0, 0], [0, 0, .75]]),
        energy_ev=0, temperature_k=300, pressure_bar=1, symmetry_number=2,
    )
    assert result['zpe_ev'] == pytest.approx(.1)
    assert result['imaginary_ev'] == [.04]


def test_adsorption_spectrum_survives_cache_roundtrip(tmp_path, monkeypatch):
    graph, site, first = _adsorption_setup()
    calculator = AdsorptionSaddleCalculator()
    kwargs = dict(
        max_steps=3, free_energy_options=free_energy.FreeEnergyOptions(),
        free_energy_temperature_k=300,
        calculation_cache_root=str(tmp_path / 'calculations'),
        calculation_cache_lookup_enabled=True, vib_cache_root=str(tmp_path / 'vibrations'),
    )
    check_site_stability(graph, site, 0, first, calculator, **kwargs)
    assert first.stable is True
    assert len(first.imaginary_occupied_ev) == 2

    def no_new_relaxation(*_args, **_kwargs):
        raise AssertionError('expected reusable electronic states')

    import ogkmc.structure
    monkeypatch.setattr(ogkmc.structure, 'optimise_structure', no_new_relaxation)
    second = AdsorbateSiteLateral(0, ego_graph=graph.subgraph([0, 1]).copy())
    check_site_stability(graph, site, 0, second, calculator, **kwargs)
    assert second.stable is True
    assert second.imaginary_occupied_ev == pytest.approx(first.imaginary_occupied_ev)
    assert second.g_occupied == pytest.approx(first.g_occupied)


@pytest.mark.parametrize('mode', [float('nan'), float('inf'), complex(0, float('nan'))])
def test_nonfinite_spectra_remain_numerical_errors(mode):
    with pytest.raises(ValueError, match='Non-finite vibrational energy'):
        free_energy._split_real_imag_ev([mode], min_frequency_ev=.0015)
