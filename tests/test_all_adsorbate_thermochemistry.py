"""All molecules in a surface state share one vibrational Hessian."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import kB
from ase.vibrations.data import VibrationsData

from autokmc.core.graph import build_graph
from autokmc.reactions.adsorption import _energetics_cached
from autokmc.sites.adsorbate import AdsorbateSite, AdsorbateSiteLateral
from autokmc.sites.bond import BondReactionLateral
from autokmc.sites.diffusion import DiffusionLateral
from autokmc.sites.stability import adsorption as adsorption_stability
from autokmc.sites.stability.bond import _apply_bond_thermochemistry
from autokmc.sites.stability.diffusion import _apply_diffusion_thermochemistry
from autokmc.thermo import free_energy


def _harmonic_free_energy(atoms, hessian, temperature=300.):
    energies = np.asarray(VibrationsData.from_2d(atoms, hessian).get_energies()).real
    kt = kB * temperature
    return float(np.sum(.5 * energies + kt * np.log1p(-np.exp(-energies / kt))))


class CoupledAdsorbates(Calculator):
    """One smooth Hamiltonian with spectator stiffening and off-diagonal terms."""
    implemented_properties = ["energy", "forces"]
    centers = {"Cu": np.array([0., 0., 0.]), "H": np.array([0., 0., 2.]),
               "O": np.array([5., 0., 2.])}

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        symbols = atoms.get_chemical_symbols()
        q = np.array([p - self.centers[s] for s, p in zip(symbols, atoms.positions)])
        energy, forces = .5 * np.sum(q**2), -q
        if "H" in symbols and "O" in symbols:
            h, o = symbols.index("H"), symbols.index("O")
            qh2, qo2 = np.dot(q[h], q[h]), np.dot(q[o], q[o])
            factor = np.exp(-qo2)
            energy += 1.5 * qh2 * factor + .4 * np.dot(q[h], q[o])
            forces[h] -= 3 * q[h] * factor + .4 * q[o]
            forces[o] += 3 * qh2 * factor * q[o] - .4 * q[h]
        self.results = {"energy": float(energy), "forces": forces}


def _coupled_atoms(symbols):
    return Atoms(symbols, positions=[CoupledAdsorbates.centers[s] for s in symbols],
                 cell=[20, 20, 20], pbc=True)


def test_joint_hessian_restores_closed_cycle_and_includes_intermolecular_couplings():
    lateral = {}
    for key, reactive, spectators in (
        ("H", "H", []), ("O", "O", []), ("H|O", "H", ["O"]), ("O|H", "O", ["H"]),
    ):
        lc = AdsorbateSiteLateral(0, energy_occupied=0., energy_unoccupied=0.)
        adsorption_stability._apply_adsorption_thermochemistry(
            lc, SimpleNamespace(reactant=f"[{reactive}]", iso_class=0),
            atoms_occupied=_coupled_atoms(["Cu"] + spectators + [reactive]),
            atoms_unoccupied=_coupled_atoms(["Cu"] + spectators),
            energy_occupied=0., energy_unoccupied=0., n_slab_occupied=1,
            n_lateral_occupied=len(spectators), n_self_occupied=1,
            calculator=CoupledAdsorbates(), free_energy_options=free_energy.FreeEnergyOptions(),
            temperature_k=300., vib_cache_root=None,
        )
        lateral[key] = lc
    deltas, log_rate_ratio = [], 0.
    for key, occupied in (("H", False), ("O|H", False), ("H|O", True), ("O", True)):
        forward = _energetics_cached(lateral[key], 0., occupied, temperature=300., g_gas=0.)
        reverse = _energetics_cached(lateral[key], 0., not occupied, temperature=300., g_gas=0.)
        deltas.append(forward[0])
        log_rate_ratio += np.log(forward[2] / reverse[2])
    assert sum(deltas) == pytest.approx(0., abs=1e-12)
    assert np.exp(log_rate_ratio) == pytest.approx(1., abs=1e-12)
    assert lateral["O|H"].vib_indices_occupied == [1, 2]
    assert lateral["O|H"].vib_indices_unoccupied == [1]
    hessian = np.block([[4 * np.eye(3), .4 * np.eye(3)], [.4 * np.eye(3), np.eye(3)]])
    expected = _harmonic_free_energy(_coupled_atoms(["H", "O"]), hessian)
    assert lateral["O|H"].g_correction_occupied == pytest.approx(expected, abs=1e-8)
    separate_blocks = _harmonic_free_energy(_coupled_atoms(["H", "O"]), np.diag([4.] * 3 + [1.] * 3))
    assert abs(expected - separate_blocks) > 1e-4


class DiagonalCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q = atoms.positions - np.asarray(atoms.info["centers"])
        curvature = np.asarray(atoms.info["curvature"])
        self.results = {"energy": float(np.sum(.5 * curvature * q**2)), "forces": -curvature * q}


def _diagonal_atoms(symbols, curvatures):
    atoms = Atoms(symbols, cell=[30, 30, 30], pbc=True)
    atoms.positions = [[3. * i, 0., 2.] for i in range(len(atoms))]
    atoms.info["centers"] = atoms.positions.tolist()
    atoms.info["curvature"] = [list(np.broadcast_to(k, (3,))) for k in curvatures]
    return atoms


def test_unoccupied_spectator_spectrum_is_recorded():
    lc = AdsorbateSiteLateral(0)
    lc.stable = True
    adsorption_stability._apply_adsorption_thermochemistry(
        lc, SimpleNamespace(reactant="[H]", iso_class=0),
        atoms_occupied=_diagonal_atoms("CuOH", [1, 1, 1]),
        atoms_unoccupied=_diagonal_atoms("CuO", [1, [-1, 1, 1]]),
        energy_occupied=0., energy_unoccupied=0., n_slab_occupied=1,
        n_lateral_occupied=1, n_self_occupied=1, calculator=DiagonalCalculator(),
        free_energy_options=free_energy.FreeEnergyOptions(), temperature_k=300., vib_cache_root=None,
    )
    assert lc.stable is True
    assert lc.vib_indices_unoccupied == [1]
    assert len(lc.imaginary_unoccupied_ev) == 1


def test_diffusion_vibrates_spectator_and_migrant_together_in_every_state():
    lc = DiffusionLateral(0)
    _apply_diffusion_thermochemistry(
        lc, SimpleNamespace(reactant="[H]", iso_class=0),
        atoms_a=_diagonal_atoms("CuOH", [1, 1, 1]),
        atoms_b=_diagonal_atoms("CuOH", [1, 4, 1]),
        atoms_ts=_diagonal_atoms("CuOH", [1, 2, [-1, 1, 1]]),
        energy_a=0., energy_b=0., energy_ts=1., n_slab=1, n_lateral=1, n_migrating=1,
        calculator=DiagonalCalculator(), free_energy_options=free_energy.FreeEnergyOptions(),
        temperature_k=300., vib_cache_root=None,
    )
    assert lc.vib_indices_a == lc.vib_indices_b == lc.vib_indices_ts == [1, 2]
    assert len(lc.frequencies_a_ev) == len(lc.frequencies_b_ev) == 6
    assert len(lc.frequencies_ts_ev) == 5
    assert lc.g_b != pytest.approx(lc.g_a)


@pytest.mark.parametrize("gas_product", [False, True])
def test_bond_all_adsorbates_and_gas_remaining_surface_reference(gas_product):
    lc = BondReactionLateral(0)
    reference = _diagonal_atoms("CuN", [1, 9])
    lc.atoms_c_gas_reference = reference
    lc.energy_c_gas_reference = -4.
    gas = SimpleNamespace(energy=-3., gibbs_energy=-3.2, zpe=.05, entropy=.001,
                          frequencies_ev=[.1], imaginary_ev=[])
    site = SimpleNamespace(iso_class=0, gas_reactant=gas,
                           template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"))
    # A deliberately invalid lifted gas precursor must never provide gas C's
    # harmonic reference; the separate CuN surface is the equilibrium state.
    c_curvatures = [1, -1, -1, -1] if gas_product else [1, 9, 1, 1]
    _apply_bond_thermochemistry(
        lc, site, atoms_ab=_diagonal_atoms("CuNH2", [1, 4, 1, 1]),
        atoms_c=_diagonal_atoms("CuNH2", c_curvatures),
        atoms_ts=_diagonal_atoms("CuNH2", [1, 2, [-1, 1, 1], 1]),
        energy_ab=-10., energy_c=-7., energy_ts=-5., n_slab=1, n_lateral=1, n_reacting=2,
        gas_product=gas_product, calculator=DiagonalCalculator(),
        free_energy_options=free_energy.FreeEnergyOptions(), temperature_k=300., vib_cache_root=None,
    )
    assert lc.vib_indices_ab == lc.vib_indices_ts == [1, 2, 3]
    assert len(lc.frequencies_ab_ev) == 9
    assert len(lc.frequencies_ts_ev) == 8
    if gas_product:
        surface_correction = _harmonic_free_energy(reference[1:], 9 * np.eye(3))
        assert lc.g_c == pytest.approx(-7. + surface_correction - .2)
        assert lc.vib_indices_c == [1]
        assert len(lc.frequencies_c_ev) == 4
        assert lc.zpe_c > gas.zpe
        assert lc.entropy_c > gas.entropy
        assert lc.thermochemistry_c_components["remaining_surface"]["state"] == "state_c_gas_reference"
        assert lc.thermochemistry_c_components["remaining_surface"]["vib_indices"] == [1]
        assert lc.thermochemistry_c_components["isolated_gas"]["state"] == "gas_molecule"
        assert lc.thermochemistry_c_components["isolated_gas"]["vib_indices"] is None
    else:
        assert lc.vib_indices_c == [1, 2, 3]
        assert len(lc.frequencies_c_ev) == 9


def test_gas_product_records_remaining_surface_spectrum():
    lc = BondReactionLateral(0)
    lc.atoms_c_gas_reference = _diagonal_atoms("CuN", [1, [-1, 1, 1]])
    lc.energy_c_gas_reference = -4.
    gas = SimpleNamespace(energy=-3., gibbs_energy=-3.2)
    site = SimpleNamespace(iso_class=0, gas_reactant=gas,
                           template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"))
    lc.stable = True
    _apply_bond_thermochemistry(
        lc, site, atoms_ab=_diagonal_atoms("CuNH2", [1, 4, 1, 1]),
        atoms_c=_diagonal_atoms("CuNH2", [1, 4, 1, 1]),
        atoms_ts=_diagonal_atoms("CuNH2", [1, 2, [-1, 1, 1], 1]),
        energy_ab=-10., energy_c=-7., energy_ts=-5., n_slab=1, n_lateral=1, n_reacting=2,
        gas_product=True, calculator=DiagonalCalculator(),
        free_energy_options=free_energy.FreeEnergyOptions(), temperature_k=300., vib_cache_root=None,
    )
    assert lc.stable is True
    assert lc.vib_indices_c == [1]
    assert len(lc.imaginary_c_ev) == 1


class AdsorptionSpectatorCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        centers = np.array([[2.6, 0, 0], [0, 0, 0], [2.6, 0, 1.7], [0, 0, 1.7]])[:len(atoms)]
        q = atoms.positions - centers
        energy, forces = .5 * np.sum(q**2), -q
        if len(atoms) == 4:
            qo2 = np.dot(q[2], q[2])
            factor = np.exp(-np.dot(q[3], q[3]))
            energy += 1.5 * qo2 * factor
            forces[2] -= 3 * q[2] * factor
            forces[3] += 3 * qo2 * factor * q[3]
        self.results = {"energy": float(energy), "forces": forces}


@pytest.mark.parametrize("legacy_policy", ["reactive_atoms_v0", "all_adsorbates_v1"])
def test_legacy_cache_reuses_electronics_but_recomputes_local_free_energy(
    tmp_path, monkeypatch, legacy_policy,
):
    atoms = Atoms("Cu2OH", positions=[[2.6, 0, 0], [0, 0, 0], [2.6, 0, 1.7], [0, 0, 1.7]],
                  cell=[20, 20, 20], pbc=True)
    atoms.arrays["surface"] = np.array([1, 1, 2, 2])
    graph = build_graph(atoms)
    for nid, clique, symbol, occupied in ((2, 0, "O", True), (3, 1, "H", False)):
        graph.nodes[nid].update(clique=frozenset([clique]), siblings=[], occupied=occupied,
                                reactant=f"[{symbol}]", reactant_index=0, iso_class=0)
    site = AdsorbateSite(reactant="[H]", n_atoms=1, atom_cliques=[frozenset([1])], positions=[[0, 0, 1.7]],
                        iso_class=0, members=[[frozenset([1])]], member_node_ids=[[3]])
    old = AdsorbateSiteLateral(0, ego_graph=graph.copy())
    calculator = AdsorptionSpectatorCalculator()
    options = free_energy.FreeEnergyOptions()
    kwargs = dict(max_steps=3, free_energy_options=options, free_energy_temperature_k=300.,
                  calculation_cache_root=str(tmp_path / "calculations"), calculation_cache_lookup_enabled=True,
                  vib_cache_root=str(tmp_path / "vibrations"))
    def old_thermo(lc, _site, *, atoms_occupied, energy_occupied, energy_unoccupied, **_kwargs):
        result = free_energy.compute_harmonic_thermo(
            atoms_occupied, [3], energy_ev=energy_occupied, temperature_k=300.,
            calculator=calculator, options=options,
        )
        lc.g_occupied = result["g_total_ev"]
        lc.g_correction_occupied = result["g_corr_ev"]
        lc.g_unoccupied = energy_unoccupied
        lc.g_correction_unoccupied = 0.

    with monkeypatch.context() as legacy:
        legacy.setattr(free_energy, "SURFACE_VIBRATION_SUBSYSTEM", legacy_policy)
        legacy.setattr(adsorption_stability, "_apply_adsorption_thermochemistry", old_thermo)
        adsorption_stability.check_site_stability(graph, site, 0, old, calculator, **kwargs)
    assert old.stable is True

    def no_new_relaxation(*_args, **_kwargs):
        raise AssertionError("electronic cache should remain reusable")

    import autokmc.structure
    monkeypatch.setattr(autokmc.structure, "optimise_structure", no_new_relaxation)
    new = AdsorbateSiteLateral(0, ego_graph=graph.copy())
    adsorption_stability.check_site_stability(graph, site, 0, new, calculator, **kwargs)
    assert new.stable is True
    assert new.vib_indices_occupied == [2, 3]
    assert new.vib_indices_unoccupied == [2]
    assert new.g_occupied - new.g_unoccupied != pytest.approx(old.g_occupied - old.g_unoccupied)
