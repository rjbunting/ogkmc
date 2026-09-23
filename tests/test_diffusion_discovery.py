"""Product diffusion must reach the KMC index even without new bond chemistry."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from ogkmc.core.graph_state import (
    DIFFUSION_CLIQUE_TO_MEMBERS,
    DIFFUSION_SURFACE_NODE_TO_MEMBERS,
    get_bond_registry,
    get_diffusion_sites,
)
from ogkmc.kmc import expansion
from ogkmc.kmc.engine import default_kmc_functions
from ogkmc.kmc.execute import execute_reaction
from ogkmc.kmc.index import _ReactionIndex
from ogkmc.kmc.models import (
    BondGrowthOptions, KMCChannels, KMCRuntime, KMCSettings, KMCSystem,
    KMCThermochemistry,
)
from ogkmc.kmc.network import DynamicNetworkExpander
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import derive_dissociation_templates
from ogkmc.sites.diffusion import find_diffusion_sites
from ogkmc.sites.identity import site_identifier


def _surface(size=3):
    graph = nx.path_graph(size)
    for node in graph:
        graph.nodes[node].update(
            type="surface", element="Pd", position=(2.75 * node, 0.0, 0.0),
        )
    return graph


def _placements(graph, element="O", offset=10, occupied=(0, 1)):
    surface = [n for n, d in graph.nodes(data=True) if d["type"] == "surface"]
    cliques = [frozenset([node]) for node in surface]
    site = AdsorbateSite(
        reactant=f"[{element}]", n_atoms=1, atom_cliques=[cliques[0]],
        positions=np.array([[0.0, 0.0, 1.5]]), iso_class=0,
        members=[[c] for c in cliques],
        member_node_ids=[[offset + node] for node in surface],
    )
    site._member_cliques = [(c,) for c in cliques]
    for node, clique in zip(surface, cliques):
        graph.add_node(
            offset + node, type="adsorbate", element=element,
            reactant=site.reactant, iso_class=0, reactant_index=0,
            reactant_orbit=0, clique=clique, siblings=[],
            occupied=node in occupied, is_bonded=True,
            position=(2.75 * node, 0.0, 1.5),
        )
        graph.add_edge(offset + node, node)
    return site


def _seed(graph, site=None, expanded=False):
    template, = derive_dissociation_templates("O=O", add_hydrogens=False)
    species = ["O=O"] + (["[O]"] if site is not None else [])
    expansion.initialise_bond_registry(
        graph, reactants={s: SimpleNamespace(smiles=s) for s in species},
        adsorbate_sites={"O=O": [], **({"[O]": [site]} if site else {})},
        templates=[template], bond_sites=[],
        expanded_smiles=species if expanded else ["O=O"],
    )
    return SimpleNamespace(
        kind="bond", direction="dissoc", site=SimpleNamespace(template=template),
    )


def _expand(graph, event, **options):
    kwargs = dict(
        include_coupling=False, include_dissociation=True,
        find_diffusion=True, diffusion_max_hops=1,
        diffusion_prune_by_ads_pair=False,
    )
    kwargs.update(options)
    return expansion.expand_bond_sites_after_event(
        graph, event, calculator=None, **kwargs,
    )


@pytest.mark.parametrize("expanded", [False, True])
def test_atomic_fragments_gain_hops_without_new_bond_templates(expanded):
    graph = _surface()
    site = _placements(graph)
    event = _seed(graph, site, expanded)

    assert _expand(graph, event) == []
    hops = get_diffusion_sites(graph)["[O]"]
    assert sum(len(hop.members) for hop in hops) == 2
    assert "[O]" in get_bond_registry(graph)["expanded_species"]
    # Repeat entry must retain the channel objects and their calculated data.
    hops[0].applicable_reactions = [object()]
    original = hops[0].applicable_reactions
    assert _expand(graph, event) == []
    assert get_diffusion_sites(graph)["[O]"][0] is hops[0]
    assert hops[0].applicable_reactions is original


def test_newly_built_atomic_product_gains_hops(monkeypatch):
    graph = _surface()
    event = _seed(graph)
    built = []

    def build_species(smiles, **kwargs):
        built.append(smiles)
        return SimpleNamespace(smiles=smiles)

    monkeypatch.setattr(expansion, "build_reactant", build_species)
    monkeypatch.setattr(
        expansion, "find_adsorbate_sites",
        lambda graph, reactant, **kwargs: [_placements(graph)],
    )
    _expand(graph, event)
    assert built == ["[O]"]
    assert sum(len(s.members) for s in get_diffusion_sites(graph)["[O]"]) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_filtered_bond_templates_do_not_control_diffusion(enabled):
    graph = _surface()
    site = _placements(graph)
    event = _seed(graph, site)
    # O + O -> O2 exists, but O + O2 coupling requires an unavailable leaf.
    # Even if all new templates are filtered, O must have its own hops.
    _expand(
        graph, event, include_coupling=True, auto_build_leaf_species=False,
        find_diffusion=enabled,
    )
    assert bool(get_diffusion_sites(graph).get("[O]")) is enabled


def test_zero_hop_species_records_completed_discovery(monkeypatch):
    graph = _surface(size=1)
    event = _seed(graph, _placements(graph))
    _expand(graph, event)
    assert get_diffusion_sites(graph) == {"[O]": []}

    def unexpected_discovery(*args, **kwargs):
        raise AssertionError("completed zero-hop discovery repeated")

    monkeypatch.setattr(expansion, "find_diffusion_sites", unexpected_discovery)
    _expand(graph, event)


def test_diffusion_retry_preserves_existing_channels_and_indexes(monkeypatch):
    graph = _surface()
    old = find_diffusion_sites(
        graph, [_placements(graph, "C", 20, occupied=())], max_hops=1,
    )["[C]"]
    event = _seed(graph, _placements(graph))
    clique_index = graph.graph[DIFFUSION_CLIQUE_TO_MEMBERS]
    surface_index = graph.graph[DIFFUSION_SURFACE_NODE_TO_MEMBERS]
    calls = []

    def broken_discovery(graph, *args, **kwargs):
        calls.append(True)
        graph.graph["diffusion_sites"] = {"[O]": []}
        graph.graph[DIFFUSION_CLIQUE_TO_MEMBERS] = {}
        graph.graph[DIFFUSION_SURFACE_NODE_TO_MEMBERS] = {}
        raise RuntimeError("interrupted discovery")

    monkeypatch.setattr(expansion, "find_diffusion_sites", broken_discovery)
    with pytest.raises(expansion.SpeciesExpansionError, match="interrupted discovery"):
        _expand(graph, event)
    assert len(calls) == 3
    assert get_diffusion_sites(graph) == {"[C]": old}
    assert graph.graph[DIFFUSION_CLIQUE_TO_MEMBERS] == clique_index
    assert graph.graph[DIFFUSION_SURFACE_NODE_TO_MEMBERS] == surface_index
    assert "[O]" not in get_bond_registry(graph)["expanded_species"]

    monkeypatch.setattr(expansion, "find_diffusion_sites", find_diffusion_sites)
    _expand(graph, event)
    merged = get_diffusion_sites(graph)
    assert merged["[C]"][0] is old[0]
    assert merged["[O]"]
    # Both species must remain reachable through the local update indexes.
    for key in (DIFFUSION_CLIQUE_TO_MEMBERS, DIFFUSION_SURFACE_NODE_TO_MEMBERS):
        indexed_species = {
            site.reactant for entries in graph.graph[key].values()
            for site, _ in entries
        }
        assert indexed_species == {"[C]", "[O]"}


@pytest.mark.parametrize("already_discovered", [False, True])
def test_expanded_product_hops_become_selectable_kmc_events(
    monkeypatch, already_discovered,
):
    graph = _surface()
    site = _placements(graph)
    event = _seed(graph, site, expanded=True)
    if already_discovered:
        find_diffusion_sites(graph, [site], max_hops=1, prune_by_adsorption_pair=False)

    def controlled_neb(graph, site, member, lateral, calculator, **kwargs):
        lateral.energy_a = lateral.energy_b = 0.0
        lateral.energy_ts = 0.5
        lateral.stable = True
        return 0.0, 0.0, 0.5

    monkeypatch.setattr(
        "ogkmc.reactions.diffusion.check_diffusion_stability", controlled_neb,
    )
    channels = KMCChannels(bond_growth_options=BondGrowthOptions(
        find_diffusion=True, include_coupling=False, diffusion_max_hops=1,
        diffusion_prune_by_ads_pair=False,
    ))
    index = _ReactionIndex([site])
    runtime = KMCRuntime(
        rng=np.random.default_rng(4), gas_energies={}, gas_free_energies={},
        partial_pressures={}, reaction_index=index, history=[],
        reaction_counts={}, current_time_s=0.0, start_step=0,
    )
    expander = DynamicNetworkExpander(
        KMCSystem(graph, [site], None, {}),
        KMCSettings(temperature=500.0, n_steps=1, verbose=False), channels,
        KMCThermochemistry(), default_kmc_functions(), runtime,
    )
    changes = expander.expand(event)
    assert len(changes.reactions) == 1  # Two O atoms, one adjacent vacancy.
    assert index.total_rate() > 0
    assert {site_identifier(s) for s in channels.diffusion_sites} == index._diffusion_ids
    hop = index.sample(0.5)
    assert hop.kind == "diffusion"
    assert hop.site.reactant == "[O]"
    total_rate = index.total_rate()
    leaves = index.n_total
    assert expander.expand(event).reactions == []
    assert index.n_total == leaves
    assert index.total_rate() == total_rate
    assert index.sample(0.5) is hop
    execute_reaction(graph, hop)
    assert [graph.nodes[10 + n]["occupied"] for n in range(3)] == [True, False, True]


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_discovers_hops_after_materializing_bond_fragments(
    monkeypatch, tmp_path, enabled,
):
    from ogkmc.io.config import BondCfg, DiffusionCfg, ReactantCfg, RunConfig
    from ogkmc.species.reactant import build_reactant
    from ogkmc.workflow.models import RunIdentity, ThermoRuntime
    from ogkmc.workflow.network import SpeciesNetworkBuilder

    graph = _surface()
    feed = build_reactant("O=O", add_hydrogens=False)
    cfg = RunConfig(
        reactants=[ReactantCfg(smiles="O=O", add_hydrogens=False)],
        bond=BondCfg(enabled=True, include_coupling=False),
        diffusion=DiffusionCfg(enabled=enabled, max_hops=1, prune_by_adsorption_pair=False),
    )
    monkeypatch.setattr(
        "ogkmc.sites.adsorbate.find_adsorbate_sites",
        lambda graph, reactant, **kwargs: [_placements(graph, occupied=())],
    )
    # Bond geometry is irrelevant to whether the real O fragment builder's
    # materialized placements are included in the subsequent diffusion search.
    monkeypatch.setattr("ogkmc.sites.bond.find_bond_sites", lambda *a, **kw: [])
    builder = SpeciesNetworkBuilder(
        cfg=cfg, identity=RunIdentity(
            output_dir=tmp_path, manifest_path=tmp_path / "run_manifest.json",
            run_id="product-diffusion",
        ),
        graph=graph, calculator_resource=None, frozen_indices=None,
        thermo_runtime=ThermoRuntime(
            options=None, vibration_cache_root=str(tmp_path / "vibrations"),
            calculation_cache_root=None,
        ),
    )
    prepared = builder.prepare([feed], [])
    assert {r.smiles for r in prepared.reactants} == {"O=O", "[O]"}
    assert len(prepared.adsorbate_sites) == 1
    assert prepared.initial_adsorbate_sites == []
    assert sum(len(s.members) for s in prepared.diffusion_sites) == (2 if enabled else 0)
