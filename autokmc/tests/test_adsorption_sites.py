"""Tests for the on-the-fly KMC adsorption-site machinery.

See ``autokmc/dev/PLAN_adsorption_sites.md`` for the design.

Tests 1, 2 and 4 are pure graph operations.  Test 3
(``TestDiscoverContextSiteNequip``) is the full end-to-end discovery
loop on Cu(111) + O using the deployed NequIP ML calculator from
``autokmc/dev/`` and is skipped when the model file or ``nequip`` are
missing.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from ase.build import fcc111

from autokmc.default_sites import (
    IsoClass,
    find_sites_for_element,
    optimise_site_positions,
    reduce_sites_by_isomorphism,
)
from autokmc.graph import build_graph
from autokmc.surface import find_surface_atoms
from autokmc.adsorbate import (
    AdsorptionSite,
    ConnectivityStatus,
    SiteOptResult,
    compute_neighbour_sites,
    discover_context_site,
    register_adsorption,
    register_desorption,
    update_reactive_flags,
)

from .conftest import requires_nequip


ELEMENT = "O"
N_SHELLS = 1


def _cu111_graph(size=(3, 3, 4)):
    slab = fcc111("Cu", size=size, vacuum=8.0, periodic=True)
    find_surface_atoms(slab, tag_atoms=True)
    G = build_graph(slab)
    find_sites_for_element(G, ELEMENT, verbose=False)
    reduce_sites_by_isomorphism(G, ELEMENT, n_shells=N_SHELLS, verbose=False)
    optimise_site_positions(G, ELEMENT, verbose=False)
    return G, slab


def _fake_site(iso, clique, *, stable=True, energy=-1.0):
    res = SiteOptResult(
        iso_class=iso,
        atoms_initial=None,  # type: ignore[arg-type]
        atoms_final=None,  # type: ignore[arg-type]
        energy=0.0,
        adsorption_energy=energy,
        converged=True,
        n_steps=0,
        connectivity=(
            ConnectivityStatus.OK if stable else ConnectivityStatus.MIGRATED
        ),
        displacement=0.0,
        ads_index=0,
        actual_clique=clique if stable else frozenset({-1}),
        matched_iso_class=None,
        surface_connectivity_changed=False,
        opt_log="",
    )
    site = AdsorptionSite(
        clique=clique,
        iso_class=iso,
        result=res,
        adsorption_energy=energy,
        ads_position=np.zeros(3),
        subgraph=None,
        stable=stable,
        reactive=stable,
        migrated_to=None if stable else frozenset({-1}),
        current_result=res,
    )
    site.context_results[frozenset()] = res
    return site


def _populate_synthetic_sites(G, *, stable=True):
    sites_by_k = G.graph["sites"][ELEMENT]
    unique_by_k = G.graph["unique_sites"][ELEMENT][N_SHELLS]
    iso_lookup = {
        m: iso
        for k, classes in unique_by_k.items()
        for iso in classes
        for m in iso.members
    }
    ads_sites = {}
    for k, cliques in sites_by_k.items():
        for clique in cliques:
            ads_sites[clique] = _fake_site(iso_lookup[clique], clique, stable=stable)
    G.graph.setdefault("adsorption_sites", {}).setdefault(ELEMENT, {})[N_SHELLS] = ads_sites
    compute_neighbour_sites(G, ELEMENT, N_SHELLS)
    return ads_sites


# ---------------------------------------------------------------------------
# 1. compute_neighbour_sites
# ---------------------------------------------------------------------------

class TestComputeNeighbourSites:

    def test_neighbours_share_at_least_one_surface_atom(self):
        G, _ = _cu111_graph()
        sites = _populate_synthetic_sites(G)
        for clique, site in sites.items():
            for nbr in site.neighbour_sites:
                assert clique & nbr
                assert nbr != clique

    def test_top_sites_have_twelve_neighbours_on_111(self):
        """Each surface atom on FCC(111) belongs to 6 bridges + 6 hollows
        (3 fcc + 3 hcp), giving 12 sites that share that atom."""
        G, _ = _cu111_graph(size=(4, 4, 4))
        sites = _populate_synthetic_sites(G)
        top_cliques = [c for c in sites if len(c) == 1]
        assert top_cliques
        for clique in top_cliques:
            assert len(sites[clique].neighbour_sites) == 12, (
                f"top {set(clique)} has "
                f"{len(sites[clique].neighbour_sites)} neighbours"
            )

    def test_hollow_to_hollow_three_edge_neighbours(self):
        G, _ = _cu111_graph(size=(4, 4, 4))
        sites = _populate_synthetic_sites(G)
        for c in [c for c in sites if len(c) == 3]:
            sharing_two = [
                n for n in sites[c].neighbour_sites
                if len(n) == 3 and len(c & n) == 2
            ]
            assert len(sharing_two) == 3


# ---------------------------------------------------------------------------
# 2. stable / migrated_to propagation
# ---------------------------------------------------------------------------

class TestStableUnstablePropagation:

    def _iso(self, members):
        return IsoClass(
            k=len(next(iter(members))),
            iso_class=0,
            n_shells=N_SHELLS,
            representative=next(iter(members)),
            members=list(members),
            centroid=np.zeros(3),
            ego_graph=nx.Graph(),
            position=np.zeros(3),
        )

    def test_stable_ok(self):
        iso = self._iso([frozenset({0, 1, 2})])
        s = _fake_site(iso, iso.representative, stable=True)
        assert s.stable is True
        assert s.reactive is True
        assert s.migrated_to is None

    def test_unstable_migrated(self):
        iso = self._iso([frozenset({0, 1, 2})])
        s = _fake_site(iso, iso.representative, stable=False)
        assert s.stable is False
        assert s.reactive is False
        assert s.migrated_to == frozenset({-1})

    def test_clean_surface_seeded_in_context_results(self):
        iso = self._iso([frozenset({0, 1, 2})])
        s = _fake_site(iso, iso.representative)
        assert frozenset() in s.context_results
        assert s.context_results[frozenset()] is s.current_result


# ---------------------------------------------------------------------------
# 3. End-to-end discovery loop with the deployed NequIP calculator
# ---------------------------------------------------------------------------

@requires_nequip
class TestDiscoverContextSiteNequip:

    @pytest.fixture(scope="class")
    def bootstrap(self, ml_calc):
        from autokmc.adsorbate import optimise_unique_sites
        slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0, periodic=True)
        find_surface_atoms(slab, tag_atoms=True)
        G = build_graph(slab)
        find_sites_for_element(G, ELEMENT, verbose=False)
        reduce_sites_by_isomorphism(G, ELEMENT, n_shells=N_SHELLS, verbose=False)
        optimise_site_positions(G, ELEMENT, verbose=False)
        optimise_unique_sites(
            G, ELEMENT, slab, ml_calc,
            n_shells=N_SHELLS, fmax=0.10, steps=200, verbose=False,
        )
        return G, slab

    def test_bootstrap_marks_sites(self, bootstrap):
        G, _ = bootstrap
        sites = G.graph["adsorption_sites"][ELEMENT][N_SHELLS]
        assert sites
        for s in sites.values():
            assert s.reactive == s.stable
            assert s.current_result is s.context_results[frozenset()]

    def test_register_adsorption_triggers_discovery(self, bootstrap, ml_calc):
        G, slab = bootstrap
        sites = G.graph["adsorption_sites"][ELEMENT][N_SHELLS]
        seed = next(
            c for c, s in sites.items()
            if s.stable and len(c) == 3 and not s.occupied
        )
        register_adsorption(
            G, ELEMENT, N_SHELLS, seed, slab, ml_calc,
            fmax=0.10, steps=100,
        )
        assert sites[seed].occupied is True
        assert sites[seed].reactive is False
        for nbr in sites[seed].neighbour_sites:
            n_site = sites[nbr]
            if not n_site.stable or n_site.occupied:
                continue
            cur_key = frozenset(
                c for c in n_site.neighbour_sites if sites[c].occupied
            )
            assert cur_key in n_site.context_results
            assert n_site.reactive is True

    def test_repeated_discovery_hits_cache(self, bootstrap):
        """After ``register_adsorption`` populates neighbours' caches,
        re-querying any vacant stable site must NOT invoke the calculator
        (per-site exact-key hit OR global iso-dedup hit) and must NOT
        grow the global cache."""
        G, slab = bootstrap
        sites = G.graph["adsorption_sites"][ELEMENT][N_SHELLS]
        ctx_cache = (
            G.graph.get("context_cache", {}).get(ELEMENT, {}).get(N_SHELLS, {})
        )
        before = sum(len(v) for v in ctx_cache.values())
        target = next(
            c for c, s in sites.items()
            if s.stable and not s.occupied
        )

        class _Forbid:
            def __getattr__(self, _):
                raise AssertionError("calculator invoked on a cache hit")

        discover_context_site(
            G, ELEMENT, N_SHELLS, target, slab, _Forbid(),
            fmax=0.10, steps=100,
        )
        after = sum(len(v) for v in ctx_cache.values())
        assert after == before


# ---------------------------------------------------------------------------
# 4. update_reactive_flags + register_desorption (pure graph ops)
# ---------------------------------------------------------------------------

class TestUpdateReactiveFlags:

    def test_adsorb_then_desorb_round_trip(self):
        G, _ = _cu111_graph()
        sites = _populate_synthetic_sites(G)
        seed = next(c for c in sites if len(c) == 3)

        # Manually mark seed occupied and recompute flags (no calculator path).
        sites[seed].occupied = True
        sites[seed].reactive = False
        flipped_on = update_reactive_flags(G, ELEMENT, N_SHELLS, seed)
        for c in sites[seed].neighbour_sites:
            assert sites[c].reactive is False
        assert flipped_on

        flipped_off = register_desorption(G, ELEMENT, N_SHELLS, seed)
        assert sites[seed].occupied is False
        assert sites[seed].reactive is True
        for c in sites[seed].neighbour_sites:
            assert sites[c].reactive is True
        assert flipped_off

