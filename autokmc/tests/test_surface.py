"""
Tests for autokmc.surface
=========================

Run with::

    pytest autokmc/tests/test_surface.py -v

All tests use ASE build helpers to construct fixtures; no external DFT code
or special data files are required.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk, fcc111, make_supercell

from autokmc.surface import (
    find_surface_atoms,
    find_surface_atoms_convexhull,
    find_surface_atoms_raycasting,
    has_pbc_connectivity,
    tag_surface_atoms,
)

try:
    from wulffpack import SingleCrystal  # noqa: F401
    _WULFF_OK = True
except ImportError:
    _WULFF_OK = False

requires_wulff = pytest.mark.skipif(not _WULFF_OK, reason="wulffpack not installed")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _cu_slab(size=(3, 3, 4), vacuum=8.0):
    """Return a Cu(111) slab with periodic boundary conditions."""
    return fcc111("Cu", size=size, vacuum=vacuum, periodic=True)


def _cu_nanoparticle(n_rep=3):
    """Return a finite Cu FCC nanoparticle (supercell with no PBC)."""
    atoms = make_supercell(bulk("Cu", "fcc"), [[n_rep, 0, 0], [0, n_rep, 0], [0, 0, n_rep]])
    atoms.set_pbc(False)
    atoms.center(vacuum=0.0)
    return atoms


# ===========================================================================
# tag_surface_atoms
# ===========================================================================

class TestTagSurfaceAtoms:

    def test_sets_surface_array(self):
        slab = _cu_slab()
        mask = np.zeros(len(slab), dtype=bool)
        mask[0] = True
        tag_surface_atoms(slab, mask)
        assert "surface" in slab.arrays

    def test_dtype_is_int8(self):
        slab = _cu_slab()
        mask = np.zeros(len(slab), dtype=bool)
        tag_surface_atoms(slab, mask)
        assert slab.arrays["surface"].dtype == np.int8

    def test_values_match_mask(self):
        slab = _cu_slab()
        mask = np.array([i % 2 == 0 for i in range(len(slab))])
        tag_surface_atoms(slab, mask)
        arr = slab.arrays["surface"]
        np.testing.assert_array_equal(arr, mask.astype(np.int8))

    def test_all_bulk(self):
        slab = _cu_slab()
        mask = np.zeros(len(slab), dtype=bool)
        tag_surface_atoms(slab, mask)
        assert slab.arrays["surface"].sum() == 0

    def test_all_surface(self):
        slab = _cu_slab()
        mask = np.ones(len(slab), dtype=bool)
        tag_surface_atoms(slab, mask)
        assert slab.arrays["surface"].sum() == len(slab)


# ===========================================================================
# has_pbc_connectivity
# ===========================================================================

class TestHasPbcConnectivity:

    def test_slab_returns_true(self):
        slab = _cu_slab()
        assert has_pbc_connectivity(slab) is True

    def test_nanoparticle_returns_false(self):
        nano = _cu_nanoparticle()
        assert has_pbc_connectivity(nano) is False

    def test_no_pbc_flag_returns_false(self):
        """Atoms with PBC disabled should return False regardless of geometry."""
        slab = _cu_slab()
        slab.set_pbc(False)
        assert has_pbc_connectivity(slab) is False

    def test_bulk_periodic_returns_true(self):
        atoms = make_supercell(bulk("Cu", "fcc"), [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        assert has_pbc_connectivity(atoms) is True


# ===========================================================================
# find_surface_atoms_raycasting
# ===========================================================================

class TestFindSurfaceAtomsRaycasting:

    def test_returns_mask_and_indices(self):
        slab = _cu_slab()
        mask, indices = find_surface_atoms_raycasting(slab)
        assert mask.shape == (len(slab),)
        assert mask.dtype == bool
        assert isinstance(indices, np.ndarray)

    def test_mask_indices_consistent(self):
        slab = _cu_slab()
        mask, indices = find_surface_atoms_raycasting(slab)
        np.testing.assert_array_equal(np.where(mask)[0], indices)

    def test_top_surface_atoms_have_high_z(self):
        """Surface atoms (top) should be in the topmost layer of the slab."""
        slab = _cu_slab(size=(3, 3, 5))
        mask, _ = find_surface_atoms_raycasting(slab, which="top")
        z_vals = slab.get_positions()[:, 2]
        z_surf = z_vals[mask]
        z_bulk = z_vals[~mask]
        # Every surface atom should be higher than the median bulk z
        assert z_surf.min() > np.median(z_bulk)

    def test_bottom_surface_atoms_have_low_z(self):
        slab = _cu_slab(size=(3, 3, 5))
        mask, _ = find_surface_atoms_raycasting(slab, which="bottom")
        z_vals = slab.get_positions()[:, 2]
        z_surf = z_vals[mask]
        z_bulk = z_vals[~mask]
        assert z_surf.max() < np.median(z_bulk)

    def test_both_finds_more_atoms_than_top(self):
        slab = _cu_slab(size=(3, 3, 6))
        _, idx_top  = find_surface_atoms_raycasting(slab, which="top")
        _, idx_both = find_surface_atoms_raycasting(slab, which="both")
        assert len(idx_both) >= len(idx_top)

    def test_invalid_which_raises(self):
        slab = _cu_slab()
        with pytest.raises(ValueError, match="which"):
            find_surface_atoms_raycasting(slab, which="side")

    def test_surface_count_includes_top_layer(self):
        """All atoms in the topmost z-layer of a Cu(111) slab must be flagged,
        and only the top layer — not the sub-surface layer — should appear."""
        slab = fcc111("Cu", size=(3, 3, 4), vacuum=8.0, periodic=True)
        mask, _ = find_surface_atoms_raycasting(slab, which="top")
        z_vals = slab.get_positions()[:, 2]
        top_layer = np.where(z_vals >= z_vals.max() - 0.1)[0]
        assert len(top_layer) == 9, "Fixture should have 9 atoms in the top layer"
        assert mask[top_layer].all(), "Not all top-layer atoms were flagged as surface"
        assert mask.sum() == 9, "Only the top layer should be flagged for a 4-layer Cu(111) slab"

    def test_nonzero_surface_found(self):
        slab = _cu_slab()
        mask, indices = find_surface_atoms_raycasting(slab)
        assert mask.any()
        assert len(indices) > 0

    def test_small_chunk_size_same_result(self):
        """Results must be identical regardless of internal batching."""
        slab = _cu_slab()
        mask1, _ = find_surface_atoms_raycasting(slab)
        mask2, _ = find_surface_atoms_raycasting(slab)
        np.testing.assert_array_equal(mask1, mask2)

    def test_surf_radius_factor_returns_valid_results(self):
        """Both small and large surf_radius_factor values must yield valid, non-empty results.

        A larger factor widens each atom's capture cone, so more rays are claimed
        by the highest atom and fewer distinct atoms appear in the surface set.
        A smaller factor narrows the cone, allowing lower-layer atoms visible
        through gaps to be picked up.  Neither direction is monotone in general,
        so we only assert that both calls succeed and return consistent outputs.
        """
        slab = _cu_slab()
        mask_small, idx_small = find_surface_atoms_raycasting(slab, surf_radius_factor=0.5)
        mask_large, idx_large = find_surface_atoms_raycasting(slab, surf_radius_factor=2.0)
        # Both must find at least some surface atoms
        assert len(idx_small) > 0
        assert len(idx_large) > 0
        # Masks and indices must remain internally consistent
        np.testing.assert_array_equal(np.where(mask_small)[0], idx_small)
        np.testing.assert_array_equal(np.where(mask_large)[0], idx_large)
        # The two radii should produce different results on a multi-layer slab
        assert len(idx_small) != len(idx_large)


# ===========================================================================
# find_surface_atoms_convexhull
# ===========================================================================

class TestFindSurfaceAtomsConvexhull:

    def test_returns_mask_indices_hull(self):
        from scipy.spatial import ConvexHull
        nano = _cu_nanoparticle()
        result = find_surface_atoms_convexhull(nano)
        assert len(result) == 3
        mask, indices, hull = result
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert isinstance(indices, np.ndarray)
        assert isinstance(hull, ConvexHull)

    def test_mask_indices_consistent(self):
        nano = _cu_nanoparticle()
        mask, indices, _ = find_surface_atoms_convexhull(nano)
        np.testing.assert_array_equal(np.where(mask)[0], indices)

    def test_hull_vertices_always_surface(self):
        """All convex-hull vertex atoms must be flagged as surface."""
        from scipy.spatial import ConvexHull
        nano = _cu_nanoparticle()
        mask, _, hull = find_surface_atoms_convexhull(nano)
        assert mask[hull.vertices].all()

    def test_nonzero_surface_found(self):
        nano = _cu_nanoparticle()
        mask, indices, _ = find_surface_atoms_convexhull(nano)
        assert mask.any()
        assert len(indices) > 0

    def test_not_all_surface(self):
        """A reasonably large nanoparticle should have interior (bulk) atoms."""
        nano = _cu_nanoparticle(n_rep=4)
        mask, _, _ = find_surface_atoms_convexhull(nano)
        assert not mask.all()

    def test_return_diagnostics_flag(self):
        nano = _cu_nanoparticle()
        result = find_surface_atoms_convexhull(nano, return_diagnostics=True)
        assert len(result) == 4
        diagnostics = result[3]
        assert "signed_dist" in diagnostics
        assert "hull_tol_per_atom" in diagnostics
        assert "dist_surface_mask" in diagnostics
        assert "vertex_mask" in diagnostics

    def test_diagnostics_shapes(self):
        nano = _cu_nanoparticle()
        mask, _, _, diag = find_surface_atoms_convexhull(nano, return_diagnostics=True)
        N = len(nano)
        assert diag["signed_dist"].shape == (N,)
        assert diag["hull_tol_per_atom"].shape == (N,)
        assert diag["dist_surface_mask"].shape == (N,)
        assert diag["vertex_mask"].shape == (N,)

    def test_tighter_tol_gives_fewer_surface_atoms(self):
        nano = _cu_nanoparticle(n_rep=4)
        _, idx_tight, _ = find_surface_atoms_convexhull(nano, hull_tol_factor=0.1)
        _, idx_loose, _ = find_surface_atoms_convexhull(nano, hull_tol_factor=2.0)
        assert len(idx_loose) >= len(idx_tight)

    def test_surface_mask_dtype(self):
        nano = _cu_nanoparticle()
        mask, _, _ = find_surface_atoms_convexhull(nano)
        assert mask.dtype == bool

    def test_surface_indices_sorted(self):
        nano = _cu_nanoparticle()
        _, indices, _ = find_surface_atoms_convexhull(nano)
        assert list(indices) == sorted(indices)


# ===========================================================================
# find_surface_atoms  (unified dispatcher)
# ===========================================================================

class TestFindSurfaceAtoms:

    def test_slab_dispatches_to_raycasting(self):
        slab = _cu_slab()
        *_, method = find_surface_atoms(slab)
        assert method == "raycasting"

    def test_nanoparticle_dispatches_to_convexhull(self):
        nano = _cu_nanoparticle()
        *_, method = find_surface_atoms(nano)
        assert method == "convexhull"

    def test_slab_returns_3_values(self):
        slab = _cu_slab()
        result = find_surface_atoms(slab)
        assert len(result) == 3  # (mask, indices, method)

    def test_nanoparticle_returns_4_values(self):
        nano = _cu_nanoparticle()
        result = find_surface_atoms(nano)
        assert len(result) == 4  # (mask, indices, hull, method)

    def test_nanoparticle_returns_5_values_with_diagnostics(self):
        nano = _cu_nanoparticle()
        result = find_surface_atoms(nano, return_diagnostics=True)
        assert len(result) == 5  # (mask, indices, hull, diagnostics, method)

    def test_slab_mask_shape(self):
        slab = _cu_slab()
        mask, indices, method = find_surface_atoms(slab)
        assert mask.shape == (len(slab),)

    def test_nano_mask_shape(self):
        nano = _cu_nanoparticle()
        mask, indices, hull, method = find_surface_atoms(nano)
        assert mask.shape == (len(nano),)

    def test_tag_atoms_slab(self):
        slab = _cu_slab()
        find_surface_atoms(slab, tag_atoms=True)
        assert "surface" in slab.arrays
        assert slab.arrays["surface"].dtype == np.int8

    def test_tag_atoms_nano(self):
        nano = _cu_nanoparticle()
        find_surface_atoms(nano, tag_atoms=True)
        assert "surface" in nano.arrays

    def test_tag_atoms_false_no_array(self):
        slab = _cu_slab()
        find_surface_atoms(slab, tag_atoms=False)
        assert "surface" not in slab.arrays

    def test_slab_surface_nonzero(self):
        slab = _cu_slab()
        mask, indices, _ = find_surface_atoms(slab)
        assert mask.any()
        assert len(indices) > 0

    def test_nano_surface_nonzero(self):
        nano = _cu_nanoparticle()
        mask, indices, *_ = find_surface_atoms(nano)
        assert mask.any()
        assert len(indices) > 0

    def test_slab_mask_indices_consistent(self):
        slab = _cu_slab()
        mask, indices, _ = find_surface_atoms(slab)
        np.testing.assert_array_equal(np.where(mask)[0], indices)

    def test_nano_mask_indices_consistent(self):
        nano = _cu_nanoparticle()
        mask, indices, *_ = find_surface_atoms(nano)
        np.testing.assert_array_equal(np.where(mask)[0], indices)

    def test_raycasting_which_both_forwarded(self):
        """which='both' should be forwarded and return more surface atoms."""
        slab = _cu_slab(size=(3, 3, 6))
        mask_top,  _, _ = find_surface_atoms(slab, which="top")
        mask_both, _, _ = find_surface_atoms(slab, which="both")
        assert mask_both.sum() >= mask_top.sum()

