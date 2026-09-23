"""Focused regression tests for species construction helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from ase import Atoms

from ogkmc.species.bond_chemistry import _strip_dummy_atoms_from_smiles
from ogkmc.species.reactant import _smiles_to_atoms, find_anchor_atoms
from ogkmc.species.smiles import (
    canonical_atom_inventory_smiles,
    smiles_to_dirname,
)


def test_smiles_to_atoms_fallback_embedding_is_deterministic_and_checked(monkeypatch):
    calls = []

    class FakeAtom:
        def __init__(self, atomic_num=6):
            self._atomic_num = atomic_num

        def GetIdx(self):
            return 0

        def GetNumImplicitHs(self):
            return 0

        def GetNumExplicitHs(self):
            return 0

        def GetAtomicNum(self):
            return self._atomic_num

    class FakeMol:
        def GetAtoms(self):
            return [FakeAtom()]

    class FakeParams:
        def __init__(self):
            self.randomSeed = None
            self.useRandomCoords = False

    chem = ModuleType("rdkit.Chem")
    chem.SmilesParserParams = SimpleNamespace
    chem.MolFromSmiles = lambda _smiles, _params=None: FakeMol()
    chem.AddHs = lambda mol, onlyOnAtoms=None, explicitOnly=False: mol
    all_chem = ModuleType("rdkit.Chem.AllChem")
    all_chem.ETKDGv3 = FakeParams
    all_chem.EmbedParameters = FakeParams

    def embed(_mol, params):
        calls.append(params)
        return -1

    all_chem.EmbedMolecule = embed
    all_chem.MMFFOptimizeMolecule = lambda *_args, **_kwargs: None
    chem.AllChem = all_chem
    rdkit = ModuleType("rdkit")
    rdkit.Chem = chem

    monkeypatch.setitem(sys.modules, "rdkit", rdkit)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", all_chem)
    monkeypatch.setattr(
        "ogkmc.species.reactant._molecule_from_smiles", lambda *_a, **_k: FakeMol(),
    )

    with pytest.raises(ValueError, match="random fallback embedder failed"):
        _smiles_to_atoms("[C]", random_seed=12345)

    assert len(calls) == 2
    assert calls[0].randomSeed == calls[1].randomSeed == 12345
    assert calls[1].useRandomCoords is True


def test_dummy_stripping_preserves_radical_style_for_reactant_cleanup():
    assert _strip_dummy_atoms_from_smiles("[CH3]*") == "[H][C]([H])[H]"
    assert _strip_dummy_atoms_from_smiles("*[OH]") == "[H][O]"


def test_bond_smiles_canonicalization_preserves_explicit_atom_inventory():
    assert canonical_atom_inventory_smiles("[OH]") == "[H][O]"
    assert canonical_atom_inventory_smiles("[H][O]") == "[H][O]"
    assert canonical_atom_inventory_smiles("[H]O[O]") == "[H]O[O]"
    assert canonical_atom_inventory_smiles("[O]O") == "[O][O]"
    assert canonical_atom_inventory_smiles("[O]O", add_hydrogens=True) == "[H]O[O]"


def test_h_plus_o2_coupling_preserves_hydrogen_through_bond_templates():
    from ogkmc.sites.bond import (
        derive_coupling_templates,
        derive_dissociation_templates,
    )
    from ogkmc.species.reactant import build_reactant

    templates = derive_coupling_templates(
        ["[H]", "O=O"],
        include_homo=False,
        include_hetero=True,
    )

    assert len(templates) == 1
    template = templates[0]
    assert (
        template.smiles_a,
        template.smiles_b,
        template.smiles_c,
    ) == ("O=O", "[H]", "[H]O[O]")

    endpoints = [
        build_reactant(smiles, add_hydrogens=False, relax=False)
        for smiles in (
            template.smiles_a,
            template.smiles_b,
            template.smiles_c,
        )
    ]
    assert len(endpoints[0].atoms) + len(endpoints[1].atoms) == 3
    assert endpoints[2].atoms.get_chemical_formula() == "HO2"
    assert len(endpoints[2].atoms) == 3

    dissociation_templates = derive_dissociation_templates(
        template.smiles_c,
        add_hydrogens=False,
    )
    assert any(
        {candidate.smiles_a, candidate.smiles_b} == {"[H][O]", "[O]"}
        for candidate in dissociation_templates
    )
    for candidate in dissociation_templates:
        fragments = [
            build_reactant(smiles, add_hydrogens=False, relax=False)
            for smiles in (candidate.smiles_a, candidate.smiles_b)
        ]
        assert sum(len(fragment.atoms) for fragment in fragments) == 3


def test_species_package_exports_public_api():
    import ogkmc.species as species

    assert species.Reactant.__name__ == "Reactant"
    assert issubclass(species.ReactantConnectivityError, RuntimeError)
    assert callable(species.build_reactant)
    assert callable(species.get_all_fragments)
    assert callable(species.combine_fragments)
    assert species.smiles_to_dirname("[C]/[O]").startswith("(C)_(O)-")


def test_methane_anchor_atoms_exclude_buried_carbon():
    atoms = Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
    )

    assert find_anchor_atoms(SimpleNamespace(atoms=atoms)) == [1, 2, 3, 4]


def test_methyl_anchor_atoms_prefer_carbon_over_hydrogen():
    atoms = Atoms(
        "CH3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
    )

    assert find_anchor_atoms(SimpleNamespace(atoms=atoms)) == [0]


def test_io_uses_shared_smiles_dirname_helper():
    from ogkmc.io import persistence, summary

    label = "[C]/[O]↔[C]\\[O]"
    assert persistence._smiles_to_dirname(label) == smiles_to_dirname(label)
    assert summary._smiles_to_dirname(label) == smiles_to_dirname(label)


def test_rdkit_isolated_h_warning_is_suppressed(capfd):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from ogkmc.utils.rdkit_logging import silence_rdkit_warnings

    silence_rdkit_warnings()
    for _ in range(3):
        Chem.RemoveHs(Chem.MolFromSmiles("[H]"))

    _out, err = capfd.readouterr()
    assert "not removing hydrogen atom without neighbors" not in err


def test_gas_cache_dir_uses_safe_smiles_label(monkeypatch, tmp_path):
    import ogkmc.species.reactant as reactant_mod

    atoms = reactant_mod._smiles_to_atoms("[C]/[O]")

    class FakeCalc:
        pass

    def fake_optimise(atoms_arg, _calculator, **_kwargs):
        atoms_arg.calc = FakeCalc()

    captured = {}

    def fake_compute_gas_thermo(*_args, **kwargs):
        captured["cache_dir"] = kwargs["cache_dir"]
        return {
            "g_corr_ev": 0.0,
            "g_total_ev": 1.0,
            "zpe_ev": 0.0,
            "entropy_ev_per_k": 0.0,
            "frequencies_ev": [],
            "imaginary_ev": [],
            "geometry": "linear",
            "symmetry_number": 1,
            "symmetry_number_source": "inferred",
            "point_group": "C_inf_v",
            "symmetry_tolerance": 0.3,
            "spin": 0,
            "temperature_k": 500.0,
            "pressure_bar": 1.0,
        }

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", lambda *_a, **_k: atoms)
    monkeypatch.setattr(reactant_mod, "_optimise", fake_optimise)
    monkeypatch.setattr(atoms, "get_potential_energy", lambda: 1.0)
    monkeypatch.setattr(
        "ogkmc.thermo.free_energy.compute_gas_thermo",
        fake_compute_gas_thermo,
    )

    reactant = reactant_mod.build_reactant(
        "[C]/[O]",
        calculator=FakeCalc(),
        free_energy_options=SimpleNamespace(enabled=True),
        free_energy_temperature_k=500.0,
        vib_cache_root=str(tmp_path),
    )

    assert captured["cache_dir"].endswith(f"gas_{smiles_to_dirname('[C]/[O]')}")
    assert reactant.thermo_meta["symmetry_number_source"] == "inferred"
    assert reactant.thermo_meta["point_group"] == "C_inf_v"
    assert reactant.thermo_meta["symmetry_tolerance"] == pytest.approx(0.3)


def test_relax_false_still_computes_single_point_energy(monkeypatch):
    import ogkmc.species.reactant as reactant_mod

    atoms = reactant_mod._smiles_to_atoms("[O]")
    atoms.arrays["surface"] = [2]

    class FakeCalc:
        pass

    captured = {}

    def fake_smiles_to_atoms(*_args, **kwargs):
        captured.update(kwargs)
        return atoms

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", fake_smiles_to_atoms)
    monkeypatch.setattr(
        reactant_mod,
        "_optimise",
        lambda *_a, **_k: pytest.fail("relaxation should be skipped"),
    )
    monkeypatch.setattr(atoms, "get_potential_energy", lambda: -1.25)

    reactant = reactant_mod.build_reactant(
        "[O]",
        calculator=FakeCalc(),
        relax=False,
        random_seed=31415,
    )

    assert reactant.energy == pytest.approx(-1.25)
    assert captured["random_seed"] == 31415


def test_vasp_gas_evaluation_temporarily_enables_full_pbc(monkeypatch):
    import ogkmc.species.reactant as reactant_mod
    from ase.calculators.vasp import Vasp

    atoms = reactant_mod._smiles_to_atoms("[O]")
    observed_pbc = []
    calculator = Vasp(command="true")

    def fake_optimise(atoms_arg, calculator_arg, **_kwargs):
        assert calculator_arg is calculator
        observed_pbc.append(tuple(bool(value) for value in atoms_arg.pbc))

    def fake_energy():
        observed_pbc.append(tuple(bool(value) for value in atoms.pbc))
        return -1.25

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", lambda *_a, **_k: atoms)
    monkeypatch.setattr(reactant_mod, "_optimise", fake_optimise)
    monkeypatch.setattr(atoms, "get_potential_energy", fake_energy)

    reactant = reactant_mod.build_reactant(
        "[O]",
        calculator=calculator,
    )

    assert observed_pbc == [(True, True, True), (True, True, True)]
    assert tuple(bool(value) for value in reactant.atoms.pbc) == (
        False,
        False,
        False,
    )


def test_configured_spin_seeds_total_magnetic_moment(monkeypatch):
    import ogkmc.species.reactant as reactant_mod

    atoms = reactant_mod._smiles_to_atoms("O=O", add_hydrogens=False)

    class FakeCalc:
        pass

    def fake_optimise(atoms_arg, _calculator, **_kwargs):
        assert atoms_arg.get_initial_magnetic_moments().tolist() == [1.0, 1.0]

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", lambda *_a, **_k: atoms)
    monkeypatch.setattr(reactant_mod, "_optimise", fake_optimise)
    monkeypatch.setattr(atoms, "get_potential_energy", lambda: -9.0)

    reactant = reactant_mod.build_reactant(
        "O=O",
        calculator=FakeCalc(),
        add_hydrogens=False,
        spin=1.0,
    )

    assert reactant.atoms.get_initial_magnetic_moments().tolist() == [1.0, 1.0]
