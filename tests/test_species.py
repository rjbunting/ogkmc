"""Focused regression tests for species construction helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from ase import Atoms

from autokmc.species.bond_chemistry import _strip_dummy_atoms_from_smiles
from autokmc.species.reactant import _smiles_to_atoms, find_anchor_atoms
from autokmc.species.smiles import smiles_to_dirname


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
    chem.MolFromSmiles = lambda _smiles: FakeMol()
    chem.AddHs = lambda mol, onlyOnAtoms=None: mol
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

    with pytest.raises(ValueError, match="random fallback embedder failed"):
        _smiles_to_atoms("[C]")

    assert len(calls) == 2
    assert calls[0].randomSeed == calls[1].randomSeed
    assert calls[1].useRandomCoords is True


def test_dummy_stripping_preserves_radical_style_for_reactant_cleanup():
    assert _strip_dummy_atoms_from_smiles("[CH3]*") == "[CH3]"
    assert _strip_dummy_atoms_from_smiles("*[OH]") == "[OH]"


def test_species_package_exports_public_api():
    import autokmc.species as species

    assert species.Reactant.__name__ == "Reactant"
    assert callable(species.build_reactant)
    assert callable(species.get_all_fragments)
    assert callable(species.combine_fragments)
    assert species.smiles_to_dirname("[C]/[O]") == "(C)_(O)"


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
    from autokmc.io import persistence, summary

    label = "[C]/[O]↔[C]\\[O]"
    assert persistence._smiles_to_dirname(label) == smiles_to_dirname(label)
    assert summary._smiles_to_dirname(label) == smiles_to_dirname(label)


def test_rdkit_isolated_h_warning_is_suppressed(capfd):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from autokmc.utils.rdkit_logging import silence_rdkit_warnings

    silence_rdkit_warnings()
    for _ in range(3):
        Chem.RemoveHs(Chem.MolFromSmiles("[H]"))

    _out, err = capfd.readouterr()
    assert "not removing hydrogen atom without neighbors" not in err


def test_gas_cache_dir_uses_safe_smiles_label(monkeypatch, tmp_path):
    import autokmc.species.reactant as reactant_mod

    atoms = reactant_mod._smiles_to_atoms("[O]")
    atoms.arrays["surface"] = [2]

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
            "geometry": "monatomic",
            "symmetry_number": 1,
            "spin": 0,
            "temperature_k": 500.0,
            "pressure_bar": 1.0,
        }

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", lambda *_a, **_k: atoms)
    monkeypatch.setattr(reactant_mod, "_optimise", fake_optimise)
    monkeypatch.setattr(atoms, "get_potential_energy", lambda: 1.0)
    monkeypatch.setattr(
        "autokmc.thermo.free_energy.compute_gas_thermo",
        fake_compute_gas_thermo,
    )

    reactant_mod.build_reactant(
        "[C]/[O]",
        calculator=FakeCalc(),
        free_energy_options=SimpleNamespace(enabled=True),
        free_energy_temperature_k=500.0,
        vib_cache_root=str(tmp_path),
    )

    assert captured["cache_dir"].endswith("gas_(C)_(O)")


def test_relax_false_still_computes_single_point_energy(monkeypatch):
    import autokmc.species.reactant as reactant_mod

    atoms = reactant_mod._smiles_to_atoms("[O]")
    atoms.arrays["surface"] = [2]

    class FakeCalc:
        pass

    monkeypatch.setattr(reactant_mod, "_smiles_to_atoms", lambda *_a, **_k: atoms)
    monkeypatch.setattr(
        reactant_mod,
        "_optimise",
        lambda *_a, **_k: pytest.fail("relaxation should be skipped"),
    )
    monkeypatch.setattr(atoms, "get_potential_energy", lambda: -1.25)

    reactant = reactant_mod.build_reactant(
        "[O]", calculator=FakeCalc(), relax=False,
    )

    assert reactant.energy == pytest.approx(-1.25)
