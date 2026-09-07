"""Feed iterables must retain their thermodynamics through rates and restart."""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from autokmc.kmc.engine import run_kmc, run_kmc_steps
from autokmc.kmc.models import KMCRunRequest, KMCSettings, KMCSystem
from autokmc.kmc.restart import reactants_for_checkpoint
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.stability.adsorption import check_adsorbate_site_lateral
from autokmc.species.reactant import Reactant


def _run_feed(pressure, container, entrypoint="legacy"):
    graph = nx.Graph()
    graph.add_node(0, type="surface", element="Pd")
    graph.add_node(
        10, type="adsorbate", element="O", reactant="[O]", iso_class=0,
        clique=frozenset({0}), is_bonded=True, occupied=False,
    )
    graph.add_edge(0, 10)
    site = AdsorbateSite(
        "[O]", 1, [frozenset({0})], np.zeros((1, 3)), 0,
        member_node_ids=[[10]], members=[[frozenset({0})]],
    )
    lateral = check_adsorbate_site_lateral(graph, site, 0)
    lateral.stable = True
    lateral.energy_occupied = -1.0
    lateral.energy_unoccupied = 0.0
    lateral.g_occupied = -0.5
    lateral.g_unoccupied = 0.0
    reactant = Reactant(
        "[O]", Atoms("O"), nx.Graph(), energy=0.0,
        gibbs_energy=-0.75, partial_pressure_bar=pressure,
    )
    feed = container([reactant])
    if entrypoint == "legacy":
        run_kmc_steps(
            graph, [site], None, feed, temperature=300.0,
            n_steps=0, verbose=False,
        )
    else:
        system = KMCSystem(graph, [site], None, feed)
        run_kmc(KMCRunRequest(
            system=system,
            settings=KMCSettings(temperature=300.0, n_steps=0, verbose=False),
        ))
        # Checkpoint writing reads the retained system feed after all three
        # initialization lookups have consumed it.
        checkpoint_feed = reactants_for_checkpoint(system.reactants, graph)
        assert checkpoint_feed == [reactant]
        assert checkpoint_feed[0].partial_pressure_bar == pressure
        assert checkpoint_feed[0].gibbs_energy == -0.75
    reaction, = site.applicable_reactions
    return reaction.delta_e, reaction.barrier, reaction.rate


@pytest.mark.parametrize("pressure", [0.0, 0.2])
@pytest.mark.parametrize("entrypoint", ["legacy", "typed"])
def test_generator_preserves_gas_thermodynamics_rates_and_checkpoint(pressure, entrypoint):
    from_list = _run_feed(pressure, list, entrypoint)
    from_generator = _run_feed(
        pressure, lambda items: (item for item in items), entrypoint,
    )

    assert from_generator == pytest.approx(from_list)
    assert from_generator[0] == pytest.approx(0.25)
    assert from_generator[1] == pytest.approx(0.35)
    if pressure == 0.0:
        assert from_generator[2] == 0.0
    else:
        assert from_generator[2] > 0.0


@pytest.mark.parametrize("container", [tuple, lambda items: items[0]])
def test_repeatable_and_single_reactant_inputs_keep_existing_semantics(container):
    assert _run_feed(0.2, container) == pytest.approx(_run_feed(0.2, list))
