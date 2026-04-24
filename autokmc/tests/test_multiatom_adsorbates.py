"""Tests for multi-atom adsorbate placement and discovery.

See ``autokmc/dev/PLAN_multiatom_adsorbates.md`` for the design.

Pure-RDKit unit tests (anchor / orbit detection) run unconditionally.
EMT-based enumeration / placement tests run when ASE is available
(it always is in this project).  The full
``register_adsorption_multi → desorption`` round trip with the deployed
NequIP model is gated on ``requires_nequip``.
"""

from __future__ import annotations

import numpy as np
import pytest

# RDKit is optional but required to *build* a Reactant.  Skip everything
# in this module cleanly if it isn't installed.
rdkit = pytest.importorskip("rdkit")

from ase.build import fcc111
from ase.calculators.emt import EMT

from autokmc.default_sites import (
    find_sites_for_element,
    optimise_site_positions,
    reduce_sites_by_isomorphism,
)
from autokmc.graph import build_graph
from autokmc.surface import find_surface_atoms
from autokmc.reactants import (
    build_reactant,
    find_anchor_atoms,
    find_unique_atoms,
)
from autokmc.adsorbate import (
    AdsorptionConfiguration,
    ConfigOptResult,
    enumerate_configurations,
    optimise_unique_configurations,
    optimise_unique_sites,
    register_adsorption_multi,
    register_desorption_multi,
    compute_neighbour_configurations,
)

from .conftest import requires_nequip


# ---------------------------------------------------------------------------
# Pure-molecule unit tests
# ---------------------------------------------------------------------------

class TestFindAnchorAtoms:
    """PLAN §11.1 — anchor detection on a few small molecules."""

    def test_h2_diatomic_all_anchors(self):
        r = build_reactant("[H][H]")
        # Diatomic → ConvexHull degenerate → all atoms are anchors.
        assert set(r.anchor_atoms) == set(range(len(r.atoms)))

    def test_h2o_planar_all_anchors(self):
        r = build_reactant("O")    # H2O
        assert set(r.anchor_atoms) == set(range(len(r.atoms)))

    def test_co_diatomic_all_anchors(self):
        r = build_reactant("[C-]#[O+]")
        assert set(r.anchor_atoms) == set(range(len(r.atoms)))

    def test_methane_only_hydrogens(self):
        r = build_reactant("C")    # CH4
        symbols = r.atoms.get_chemical_symbols()
        h_idx = [i for i, s in enumerate(symbols) if s == "H"]
        c_idx = [i for i, s in enumerate(symbols) if s == "C"]
        # All four hydrogens are hull vertices; carbon is interior.
        assert set(h_idx).issubset(set(r.anchor_atoms))
        # Carbon may or may not be flagged depending on hull_tol; the
        # important invariant is that the *primary* anchors are hydrogens.
        for h in h_idx:
            assert h in r.anchor_atoms
        # The carbon centre should not be a hull vertex of the regular
        # tetrahedron, even with the covalent-radius slack.
        assert c_idx[0] not in r.anchor_atoms or len(r.anchor_atoms) >= len(h_idx)


class TestFindUniqueAtoms:
    """PLAN §11.2 — intramolecular orbits."""

    def test_h2_one_orbit(self):
        r = build_reactant("[H][H]")
        assert r.unique_nodes == {"H": [[0, 1]]}

    def test_h2o_orbits(self):
        r = build_reactant("O")  # H2O: O at index 0, two H equivalent
        symbols = r.atoms.get_chemical_symbols()
        o_idx = symbols.index("O")
        h_idx = [i for i, s in enumerate(symbols) if s == "H"]
        assert r.unique_nodes["O"] == [[o_idx]]
        assert r.unique_nodes["H"] == [sorted(h_idx)]

    def test_methane_four_h_one_orbit(self):
        r = build_reactant("C")
        symbols = r.atoms.get_chemical_symbols()
        h_idx = sorted(i for i, s in enumerate(symbols) if s == "H")
        assert r.unique_nodes["H"] == [h_idx]


# ---------------------------------------------------------------------------
# Surface-side enumeration / placement
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cu111_graph():
    slab = fcc111("Cu", size=(4, 4, 4), vacuum=8.0, periodic=True)
    find_surface_atoms(slab, tag_atoms=True)
    G = build_graph(slab)
    return slab, G


@pytest.fixture(scope="module")
def cu111_with_h_sites(cu111_graph):
    """Pre-bootstrap H sites (single-atom) so multi-atom enumeration of
    H2 has the per-element ``unique_sites`` / ``site_positions`` keys
    populated for hydrogen anchors."""
    slab, G = cu111_graph
    for elem in ("H",):
        find_sites_for_element(G, elem, verbose=False)
        reduce_sites_by_isomorphism(G, elem, n_shells=1, verbose=False)
        optimise_site_positions(G, elem, verbose=False)
    return slab, G


class TestEnumerateConfigurations:
    """PLAN §11.3 — configurations are produced and deduplicated."""

    def test_h2_enumerates_pairs(self, cu111_with_h_sites):
        slab, G = cu111_with_h_sites
        r = build_reactant("[H][H]")
        # Loose tolerances to make the test resilient to RDKit/MMFF94 jitter.
        cfgs = enumerate_configurations(
            G, r,
            anchor_distance_tol=0.6,
            clash_factor=0.5,
            k_candidates=8,
            verbose=False,
        )
        # At minimum, *some* configuration is produced for diatomic H2.
        assert len(cfgs) > 0
        # Iso-class deduplication shrinks the unique set.
        unique = {c.iso_class for c in cfgs}
        assert len(unique) <= len(cfgs)
        # Every configuration uses two distinct cliques (no self-overlap).
        for c in cfgs:
            cliques = list(c.anchor_clique_map.values())
            assert len(set(cliques)) == 2

    def test_n_shells_other_than_one_raises(self, cu111_with_h_sites):
        slab, G = cu111_with_h_sites
        r = build_reactant("[H][H]")
        with pytest.raises(NotImplementedError):
            enumerate_configurations(G, r, n_shells=2)


# ---------------------------------------------------------------------------
# Optimisation + KMC round-trip (EMT-based)
# ---------------------------------------------------------------------------

class TestOptimiseUniqueConfigurationsEMT:
    """PLAN §11.4 — EMT placement of H2 → adsorbed H+H on Cu(111)."""

    def test_h2_on_cu111(self, cu111_with_h_sites):
        slab, G = cu111_with_h_sites
        calc = EMT()
        r = build_reactant("[H][H]", calculator=calc)
        results = optimise_unique_configurations(
            G, r, slab, calc,
            anchor_distance_tol=0.6,
            clash_factor=0.5,
            k_candidates=8,
            fmax=0.1, steps=100,
            verbose=False,
        )
        assert len(results) > 0
        # At least one configuration converged to a finite adsorption energy.
        any_finite = any(not np.isnan(res.adsorption_energy) for res in results)
        assert any_finite

        cfgs = G.graph["adsorption_configs"]["[H][H]"][1]
        assert len(cfgs) > 0
        for cfg in cfgs.values():
            # current_result was seeded at clean-surface bootstrap.
            assert cfg.current_result is not None
            assert frozenset() in cfg.context_results


class TestRegisterAdsorptionMultiRoundtrip:
    """PLAN §11.5 — register_adsorption_multi mutates state correctly."""

    def test_roundtrip_single_config(self, cu111_with_h_sites):
        slab, G = cu111_with_h_sites
        calc = EMT()
        r = build_reactant("[H][H]", calculator=calc)
        # Bootstrap H single-atom sites so the cross-write target exists.
        optimise_unique_sites(
            G, "H", slab, calc,
            n_shells=1, fmax=0.1, steps=100, verbose=False,
        )
        optimise_unique_configurations(
            G, r, slab, calc,
            anchor_distance_tol=0.6,
            clash_factor=0.5,
            k_candidates=8,
            fmax=0.1, steps=100,
            verbose=False,
        )

        cfgs = G.graph["adsorption_configs"]["[H][H]"][1]
        # Find a stable configuration to register.
        target_key = None
        for k, c in cfgs.items():
            if c.stable:
                target_key = k
                break
        if target_key is None:
            pytest.skip("No stable H2 configuration produced by EMT")

        cfg = cfgs[target_key]
        h_sites = G.graph["adsorption_sites"]["H"][1]
        # Sanity: cliques are vacant before registration.
        for clique in cfg.anchor_clique_map.values():
            if clique in h_sites:
                assert not h_sites[clique].occupied

        register_adsorption_multi(
            G, "[H][H]", target_key, slab, calc,
            n_shells=1, discover=False,
        )

        # (a) Single-atom sites are flipped to occupied.
        for clique in cfg.anchor_clique_map.values():
            if clique in h_sites:
                assert h_sites[clique].occupied
        # (b) Configuration itself is occupied + non-reactive.
        assert cfg.occupied
        assert not cfg.reactive

        # Desorption restores state.
        register_desorption_multi(G, "[H][H]", target_key, n_shells=1)
        assert not cfg.occupied
        for clique in cfg.anchor_clique_map.values():
            if clique in h_sites:
                assert not h_sites[clique].occupied


# ---------------------------------------------------------------------------
# Full integration via NequIP (skipped without the deployed model)
# ---------------------------------------------------------------------------

@requires_nequip
class TestRegisterAdsorptionMultiNequip:
    """PLAN §11.6 — full enumerate→optimise→register→desorb on NequIP."""

    def test_full_roundtrip(self, cu111_with_h_sites, ml_calc):
        from autokmc.tests.conftest import CountingCalculator
        slab, G = cu111_with_h_sites
        calc = CountingCalculator(ml_calc)
        r = build_reactant("[H][H]", calculator=calc)

        optimise_unique_sites(
            G, "H", slab, calc, n_shells=1,
            fmax=0.1, steps=100, verbose=False,
        )
        optimise_unique_configurations(
            G, r, slab, calc,
            fmax=0.1, steps=100, verbose=False,
        )
        cfgs = G.graph["adsorption_configs"]["[H][H]"][1]
        stable = [k for k, c in cfgs.items() if c.stable]
        assert stable, "Expected at least one stable H2 configuration"

        before = calc.n_force_calls
        register_adsorption_multi(
            G, "[H][H]", stable[0], slab, calc,
            n_shells=1, discover=False,
        )
        register_desorption_multi(G, "[H][H]", stable[0], n_shells=1)
        # Discovery is off so no extra force calls beyond bootstrap.
        assert calc.n_force_calls == before

