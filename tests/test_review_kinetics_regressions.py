"""Rare random draws, generator continuation, and low-pressure rate caching."""

import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from autokmc.kmc.engine import run_kmc_steps
from autokmc.kmc.restart import capture_rng_state, normalise_rng
from autokmc.kmc.sampling import sample_tau
from autokmc.reactions.adsorption import get_applicable_reaction_for_member
from autokmc.reactions.bond import _bond_energetics_cached
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral


def cached_surface():
    graph = nx.Graph()
    graph.add_node(0, type="surface", element="Pd")
    graph.add_node(
        10,
        type="adsorbate",
        element="O",
        reactant="[O]",
        iso_class=0,
        clique=frozenset({0}),
        is_bonded=True,
        occupied=False,
    )
    graph.add_edge(0, 10)
    site = AdsorbateSite(
        "[O]",
        1,
        [frozenset({0})],
        np.zeros((1, 3)),
        0,
        member_node_ids=[[10]],
        members=[[frozenset({0})]],
    )
    lateral = check_adsorbate_site_lateral(graph, site, 0)
    lateral.stable = True
    lateral.energy_occupied = -1.0
    lateral.energy_unoccupied = 0.0
    return graph, site


@pytest.mark.parametrize("name", ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"])
@pytest.mark.parametrize("existing", [False, True])
def test_numpy_checkpoint_restores_generator_type_and_continuation(name, existing):
    rng = np.random.Generator(getattr(np.random, name)(7))
    rng.random(11)
    state = capture_rng_state(rng)
    restored = normalise_rng(np.random.default_rng(3) if existing else None, state)
    assert type(restored.bit_generator) is type(rng.bit_generator)
    np.testing.assert_array_equal(rng.random(25), restored.random(25))


class ZeroRng(random.Random):
    def random(self):
        return 0.0


def test_zero_uniform_draw_yields_finite_time_in_sampler_and_session():
    assert sample_tau(1.0, ZeroRng()) == pytest.approx(-np.log(np.nextafter(0.0, 1.0)))
    graph, site = cached_surface()
    result = run_kmc_steps(
        graph,
        [site],
        None,
        {"[O]": 0.0},
        temperature=300.0,
        n_steps=1,
        rng=ZeroRng(),
        verbose=False,
    )
    assert result["steps_executed"] == 1
    assert np.isfinite(result["time"]) and result["time"] > 0.0


def adsorption_rate(graph, site, pressure):
    return get_applicable_reaction_for_member(
        graph,
        site,
        0,
        None,
        {"[O]": 0.0},
        temperature=300.0,
        partial_pressures={"[O]": pressure},
    ).rate


@pytest.mark.parametrize("pressures", [(0.0, 1e-10), (1e-10, 0.0), (1e-11, 2e-11)])
def test_adsorption_cache_keeps_distinct_small_pressures(pressures):
    graph, site = cached_surface()
    rates = [adsorption_rate(graph, site, pressure) for pressure in pressures]
    for pressure, cached in zip(pressures, rates):
        fresh_graph, fresh_site = cached_surface()
        fresh = adsorption_rate(fresh_graph, fresh_site, pressure)
        assert cached == fresh
        assert (cached > 0.0) == (pressure > 0.0)


@pytest.mark.parametrize("pressures", [(0.0, 1e-13), (1e-13, 0.0), (1e-14, 2e-14)])
def test_bond_cache_keeps_distinct_small_gas_pressures(pressures):
    lateral = SimpleNamespace(
        energy_ab=0.0, energy_c=0.0, energy_ts=1.0, gas_product=True, gas_pressure_bar=0.0
    )
    for pressure in pressures:
        lateral.gas_pressure_bar = pressure
        cached = _bond_energetics_cached(lateral, "dissoc", temperature=300.0)[2]
        fresh = SimpleNamespace(
            energy_ab=0.0, energy_c=0.0, energy_ts=1.0, gas_product=True, gas_pressure_bar=pressure
        )
        assert cached == _bond_energetics_cached(fresh, "dissoc", temperature=300.0)[2]
        assert (cached > 0.0) == (pressure > 0.0)
