"""
Tests for autokmc.structure
============================

Run with::

    pytest autokmc/tests/test_structure.py -v

All tests use EMT as the calculator so no external DFT code is required.
WulffPack and pymatgen are required for the nanoparticle / surface tests
respectively; those tests are skipped automatically if the libraries are
not installed.
"""

import math

import numpy as np
import pytest
from ase.calculators.emt import EMT

from autokmc.structure import (
    _apply_composition,
    _build_primitive_cell,
    _extract_lp,
    _get_bottom_layer_indices,
    _normalise_lp,
    _orthogonalise_slab,
    _parse_composition,
    _primary_element,
    _validate_crystal_structure,
    anneal_alloy,
    build_nanoparticle,
    build_surface,
    optimise_bulk,
    optimise_structure,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMT_CALC = EMT()

try:
    from wulffpack import SingleCrystal  # noqa: F401
    _WULFF_OK = True
except ImportError:
    _WULFF_OK = False

try:
    from pymatgen.core.surface import SlabGenerator  # noqa: F401
    _PMG_OK = True
except ImportError:
    _PMG_OK = False

requires_wulff = pytest.mark.skipif(not _WULFF_OK, reason="wulffpack not installed")
requires_pmg   = pytest.mark.skipif(not _PMG_OK,   reason="pymatgen not installed")


# ===========================================================================
# _parse_composition
# ===========================================================================

class TestParseComposition:

    def test_string_returns_pure(self):
        result = _parse_composition("Cu")
        assert result == {"Cu": 1.0}

    def test_dict_already_normalised(self):
        result = _parse_composition({"Cu": 0.7, "Pt": 0.3})
        assert abs(result["Cu"] - 0.7) < 1e-9
        assert abs(result["Pt"] - 0.3) < 1e-9

    def test_dict_unnormalised_integers(self):
        """Values like {Cu: 2, Pt: 1} should be normalised to fractions."""
        result = _parse_composition({"Cu": 2, "Pt": 1})
        assert abs(result["Cu"] - 2 / 3) < 1e-9
        assert abs(result["Pt"] - 1 / 3) < 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_dict_percentages(self):
        """Values like {Cu: 70, Pt: 30} should be normalised."""
        result = _parse_composition({"Cu": 70, "Pt": 30})
        assert abs(result["Cu"] - 0.7) < 1e-9
        assert abs(result["Pt"] - 0.3) < 1e-9

    def test_three_elements_sum_to_one(self):
        result = _parse_composition({"Cu": 0.5, "Pt": 0.3, "Au": 0.2})
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _parse_composition({"Cu": 0.0, "Pt": 0.0})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _parse_composition({"Cu": -1.0})


# ===========================================================================
# _primary_element
# ===========================================================================

class TestPrimaryElement:

    def test_single(self):
        assert _primary_element({"Cu": 1.0}) == "Cu"

    def test_picks_max(self):
        assert _primary_element({"Cu": 0.3, "Pt": 0.7}) == "Pt"

    def test_three_elements(self):
        assert _primary_element({"Cu": 0.2, "Au": 0.5, "Pt": 0.3}) == "Au"


# ===========================================================================
# _validate_crystal_structure
# ===========================================================================

class TestValidateCrystalStructure:

    @pytest.mark.parametrize("cs", ["fcc", "bcc", "hcp"])
    def test_valid(self, cs):
        _validate_crystal_structure(cs)  # should not raise

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="crystal_structure"):
            _validate_crystal_structure("diamond")

    def test_case_sensitivity(self):
        """The function expects lowercase; uppercase should raise."""
        with pytest.raises(ValueError):
            _validate_crystal_structure("FCC")


# ===========================================================================
# _normalise_lp
# ===========================================================================

class TestNormaliseLp:

    def test_float_fcc(self):
        lp = _normalise_lp(3.615, "fcc")
        assert lp == {"a": 3.615}

    def test_float_hcp_adds_c(self):
        a = 3.21
        lp = _normalise_lp(a, "hcp")
        assert "a" in lp and "c" in lp
        assert abs(lp["c"] / lp["a"] - math.sqrt(8 / 3)) < 1e-9

    def test_dict_passthrough(self):
        d = {"a": 3.21, "c": 5.21}
        assert _normalise_lp(d, "hcp") == d


# ===========================================================================
# _build_primitive_cell
# ===========================================================================

class TestBuildPrimitiveCell:

    @pytest.mark.parametrize("cs", ["fcc", "bcc", "hcp"])
    def test_returns_atoms(self, cs):
        from ase import Atoms
        lp = {"a": 3.5, "c": 5.5} if cs == "hcp" else {"a": 3.5}
        atoms = _build_primitive_cell("Cu", cs, lp)
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0

    def test_fcc_has_1_atom(self):
        atoms = _build_primitive_cell("Cu", "fcc", {"a": 3.615})
        assert len(atoms) == 1

    def test_hcp_has_2_atoms(self):
        # Ni used because EMT supports it; HCP structure gives 2 atoms regardless
        atoms = _build_primitive_cell("Ni", "hcp", {"a": 2.49, "c": 4.07})
        assert len(atoms) == 2


# ===========================================================================
# _extract_lp
# ===========================================================================

class TestExtractLp:

    def test_fcc_roundtrip(self):
        a_in = 3.615
        atoms = _build_primitive_cell("Cu", "fcc", {"a": a_in})
        lp = _extract_lp(atoms, "fcc")
        assert "a" in lp
        assert abs(lp["a"] - a_in) < 1e-3

    def test_bcc_roundtrip(self):
        # Cu in BCC: unphysical but geometrically valid; Fe excluded (no EMT)
        a_in = 2.87
        atoms = _build_primitive_cell("Cu", "bcc", {"a": a_in})
        lp = _extract_lp(atoms, "bcc")
        assert abs(lp["a"] - a_in) < 1e-3

    def test_hcp_roundtrip(self):
        # Ni in HCP: unphysical but geometrically valid; Ti excluded (no EMT)
        a_in, c_in = 2.49, 4.07
        atoms = _build_primitive_cell("Ni", "hcp", {"a": a_in, "c": c_in})
        lp = _extract_lp(atoms, "hcp")
        assert abs(lp["a"] - a_in) < 1e-3
        assert abs(lp["c"] - c_in) < 1e-3


# ===========================================================================
# _apply_composition
# ===========================================================================

class TestApplyComposition:

    def _make_cu(self, n=100):
        from ase.build import bulk, make_supercell
        atoms = make_supercell(bulk("Cu", "fcc", a=3.615), [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
        return atoms

    def test_pure_unchanged(self):
        atoms = self._make_cu()
        result = _apply_composition(atoms, {"Cu": 1.0}, verbose=False)
        syms = result.get_chemical_symbols()
        assert all(s == "Cu" for s in syms)

    def test_correct_count_binary(self):
        atoms = self._make_cu()        # 64 atoms
        n = len(atoms)
        result = _apply_composition(atoms, {"Cu": 0.75, "Pt": 0.25},
                                    seed=0, verbose=False)
        syms = np.array(result.get_chemical_symbols())
        n_pt = int(round(n * 0.25))
        assert (syms == "Pt").sum() == n_pt
        assert (syms == "Cu").sum() == n - n_pt

    def test_reproducible_with_seed(self):
        atoms = self._make_cu()
        r1 = _apply_composition(atoms, {"Cu": 0.7, "Pt": 0.3}, seed=42, verbose=False)
        r2 = _apply_composition(atoms, {"Cu": 0.7, "Pt": 0.3}, seed=42, verbose=False)
        assert r1.get_chemical_symbols() == r2.get_chemical_symbols()

    def test_different_seeds_differ(self):
        atoms = self._make_cu()
        r1 = _apply_composition(atoms, {"Cu": 0.7, "Pt": 0.3}, seed=1, verbose=False)
        r2 = _apply_composition(atoms, {"Cu": 0.7, "Pt": 0.3}, seed=2, verbose=False)
        assert r1.get_chemical_symbols() != r2.get_chemical_symbols()

    def test_three_element(self):
        atoms = self._make_cu()
        n = len(atoms)
        comp = {"Cu": 0.5, "Pt": 0.3, "Au": 0.2}
        result = _apply_composition(atoms, comp, seed=0, verbose=False)
        syms = np.array(result.get_chemical_symbols())
        assert set(syms).issubset({"Cu", "Pt", "Au"})
        total = sum((syms == s).sum() for s in comp)
        assert total == n


# ===========================================================================
# _get_bottom_layer_indices
# ===========================================================================

class TestGetBottomLayerIndices:

    def _fcc_slab(self):
        """Build a small Cu(111) slab for testing."""
        from ase.build import fcc111
        return fcc111("Cu", size=(2, 2, 4), vacuum=8.0)

    def test_returns_list(self):
        slab = self._fcc_slab()
        idx = _get_bottom_layer_indices(slab, 1, 3.615)
        assert isinstance(idx, list)

    def test_freeze_1_layer(self):
        slab = self._fcc_slab()   # 4 layers × 4 atoms = 16 atoms
        idx = _get_bottom_layer_indices(slab, 1, 3.615)
        assert len(idx) == 4

    def test_freeze_2_layers(self):
        slab = self._fcc_slab()
        idx = _get_bottom_layer_indices(slab, 2, 3.615)
        assert len(idx) == 8

    def test_frozen_atoms_are_lowest(self):
        slab = self._fcc_slab()
        idx = _get_bottom_layer_indices(slab, 1, 3.615)
        pos = slab.get_positions()
        z_frozen = pos[idx, 2]
        z_free   = np.delete(pos[:, 2], idx)
        assert z_frozen.max() < z_free.min() + 0.1


# ===========================================================================
# _orthogonalise_slab
# ===========================================================================

class TestOrthogonaliseSlab:

    def _cu111_slab(self):
        from pymatgen.core.surface import SlabGenerator
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.build import bulk
        from ase import Atoms as _Atoms
        atoms_bulk = bulk("Cu", "fcc", a=3.615)
        pmg = AseAtomsAdaptor.get_structure(atoms_bulk)
        slab_pmg = SlabGenerator(pmg, (1, 1, 1), 8.0, 12.0).get_slabs()[0]
        slab = AseAtomsAdaptor.get_atoms(slab_pmg)
        assert isinstance(slab, _Atoms)
        return slab

    @requires_pmg
    def test_diagonal_cell(self):
        """After orthogonalisation the cell must be diagonal."""
        slab = self._cu111_slab()
        ortho, _ = _orthogonalise_slab(slab)
        cell = np.array(ortho.get_cell())
        off_diag = cell - np.diag(np.diagonal(cell))
        assert np.abs(off_diag).max() < 1e-6

    @requires_pmg
    def test_returns_info_tuple(self):
        slab = self._cu111_slab()
        _, info = _orthogonalise_slab(slab)
        assert len(info) == 5   # (n1, n2, m1, m2, det)

    @requires_pmg
    def test_pbc_preserved(self):
        slab = self._cu111_slab()
        ortho, _ = _orthogonalise_slab(slab)
        assert all(ortho.get_pbc())


# ===========================================================================
# optimise_bulk
# ===========================================================================

class TestOptimiseBulk:

    @pytest.mark.parametrize("symbol,cs", [
        ("Cu", "fcc"),
        ("Cu", "bcc"),   # Cu replaces Fe – Fe has no EMT parameters
    ])
    def test_returns_atoms_and_lp(self, symbol, cs):
        from ase import Atoms
        atoms, lp = optimise_bulk(symbol, cs, calculator=EMT(), verbose=False)
        assert isinstance(atoms, Atoms)
        assert "a" in lp
        assert lp["a"] > 0

    def test_fcc_cu_lattice_constant(self):
        """EMT-relaxed Cu FCC lattice constant should be near 3.52 Å."""
        _, lp = optimise_bulk("Cu", "fcc", calculator=EMT(), verbose=False)
        assert 3.4 < lp["a"] < 3.7

    def test_invalid_structure_raises(self):
        with pytest.raises(ValueError):
            optimise_bulk("Cu", "diamond", calculator=EMT(), verbose=False)


# ===========================================================================
# optimise_structure
# ===========================================================================

class TestOptimiseStructure:

    def test_returns_copy(self):
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.615)
        result = optimise_structure(atoms, calculator=EMT(), verbose=False)
        # original should be unchanged (no calculator attached)
        assert atoms.calc is None

    def test_energy_decreases(self):
        from ase.build import bulk
        # Start with a compressed cell to ensure there's something to relax
        atoms = bulk("Cu", "fcc", a=3.3)
        atoms.calc = EMT()
        e_before = atoms.get_potential_energy()
        result = optimise_structure(atoms, calculator=EMT(), verbose=False)
        e_after = result.get_potential_energy()
        assert e_after <= e_before + 1e-6

    def test_default_calculator_is_emt(self):
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.615)
        result = optimise_structure(atoms, verbose=False)
        assert result.calc is not None

    def test_constraints_preserved(self):
        from ase.build import fcc111
        from ase.constraints import FixAtoms
        slab = fcc111("Cu", size=(2, 2, 4), vacuum=8.0)
        slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
        result = optimise_structure(slab, calculator=EMT(), verbose=False)
        assert result.constraints  # constraint list is non-empty


# ===========================================================================
# anneal_alloy  (placeholder)
# ===========================================================================

class TestAnnealAlloy:

    def test_raises_not_implemented(self):
        from ase.build import bulk, make_supercell
        atoms = make_supercell(bulk("Cu", "fcc", a=3.615),
                               [[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        with pytest.raises(NotImplementedError):
            anneal_alloy(atoms, calculator=EMT())


# ===========================================================================
# build_nanoparticle  (integration – requires wulffpack)
# ===========================================================================

@requires_wulff
class TestBuildNanoparticle:

    SE = {(1, 1, 1): 1.10, (1, 0, 0): 1.29, (1, 1, 0): 1.51}

    def test_pure_cu_returns_atoms(self):
        from ase import Atoms
        atoms = build_nanoparticle(
            composition="Cu",
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=100,
            calculator=EMT(),
            verbose=False,
        )
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0

    def test_pure_cu_all_copper(self):
        atoms = build_nanoparticle(
            composition="Cu",
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=100,
            calculator=EMT(),
            verbose=False,
        )
        syms = set(atoms.get_chemical_symbols())
        assert syms == {"Cu"}

    def test_binary_alloy_composition(self):
        atoms = build_nanoparticle(
            composition={"Cu": 0.7, "Pt": 0.3},
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=100,
            calculator=EMT(),
            verbose=False,
        )
        syms = np.array(atoms.get_chemical_symbols())
        assert (syms == "Pt").sum() > 0
        assert (syms == "Cu").sum() > 0

    def test_unnormalised_composition(self):
        """Composition given as percentages (not fractions) should work."""
        atoms = build_nanoparticle(
            composition={"Cu": 70, "Pt": 30},
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=100,
            calculator=EMT(),
            verbose=False,
        )
        assert len(atoms) > 0

    def test_invalid_crystal_structure_raises(self):
        with pytest.raises(ValueError):
            build_nanoparticle(
                composition="Cu",
                crystal_structure="diamond",
                lattice_constant=3.615,
                surface_energies=self.SE,
                target_atoms=50,
                calculator=EMT(),
                verbose=False,
            )

    def test_has_calculator_after_build(self):
        atoms = build_nanoparticle(
            composition="Cu",
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=50,
            calculator=EMT(),
            verbose=False,
        )
        assert atoms.calc is not None

    def test_energy_is_finite(self):
        atoms = build_nanoparticle(
            composition="Cu",
            crystal_structure="fcc",
            lattice_constant=3.615,
            surface_energies=self.SE,
            target_atoms=50,
            calculator=EMT(),
            verbose=False,
        )
        e = atoms.get_potential_energy()
        assert math.isfinite(e)


# ===========================================================================
# build_surface  (integration – requires pymatgen)
# ===========================================================================

@requires_pmg
class TestBuildSurface:

    def test_returns_atoms(self):
        from ase import Atoms
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0

    def test_pbc_is_true(self):
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        assert all(atoms.get_pbc())

    def test_cell_is_diagonal_after_ortho(self):
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            orthogonalise=True,
            calculator=EMT(),
            verbose=False,
        )
        cell = np.array(atoms.get_cell())
        off_diag = cell - np.diag(np.diagonal(cell))
        assert np.abs(off_diag).max() < 1e-4

    def test_skip_orthogonalise(self):
        """With orthogonalise=False the function should still complete."""
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            orthogonalise=False,
            calculator=EMT(),
            verbose=False,
        )
        assert len(atoms) > 0

    def test_freeze_constraint_applied(self):
        from ase.constraints import FixAtoms
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            n_freeze_layers=2,
            calculator=EMT(),
            verbose=False,
        )
        fixed = [c for c in atoms.constraints if isinstance(c, FixAtoms)]
        assert len(fixed) == 1

    def test_no_freeze_no_constraint(self):
        from ase.constraints import FixAtoms
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            n_freeze_layers=0,
            calculator=EMT(),
            verbose=False,
        )
        fixed = [c for c in atoms.constraints if isinstance(c, FixAtoms)]
        assert len(fixed) == 0

    def test_alloy_slab_has_both_elements(self):
        atoms = build_surface(
            composition={"Cu": 0.7, "Pt": 0.3},
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        syms = set(atoms.get_chemical_symbols())
        assert "Cu" in syms
        assert "Pt" in syms

    def test_unnormalised_composition(self):
        atoms = build_surface(
            composition={"Cu": 70, "Pt": 30},
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        assert len(atoms) > 0

    def test_energy_is_finite(self):
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        e = atoms.get_potential_energy()
        assert math.isfinite(e)

    def test_lateral_size_approximately_met(self):
        """Cell x and y dimensions should be ≥ goal_x / goal_y."""
        goal_x, goal_y = 10.0, 10.0
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=3.615,
            goal_x=goal_x,
            goal_y=goal_y,
            orthogonalise=True,
            calculator=EMT(),
            verbose=False,
        )
        cell = atoms.get_cell()
        assert np.linalg.norm(cell[0]) >= goal_x - 0.5
        assert np.linalg.norm(cell[1]) >= goal_y - 0.5

    @pytest.mark.parametrize("hkl", [(1, 0, 0), (1, 1, 0)])
    def test_different_miller_indices(self, hkl: tuple):
        atoms = build_surface(
            composition="Cu",
            crystal_structure="fcc",
            miller_index=hkl,  # type: ignore[arg-type]
            lattice_constant=3.615,
            goal_x=8.0,
            goal_y=8.0,
            calculator=EMT(),
            verbose=False,
        )
        assert len(atoms) > 0



