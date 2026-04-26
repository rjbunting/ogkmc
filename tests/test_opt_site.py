from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import autokmc.opt_site as opt_site
from autokmc.cache import get_cache
from autokmc.find_multisite import AdsorbateSite


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 5.0
        self.results["forces"] = np.zeros((len(atoms), 3))


class MoveAdsorbateOptimizer:
    def __init__(self, atoms, logfile=None):
        self.atoms = atoms

    def run(self, fmax=0.05, steps=200):
        positions = self.atoms.get_positions()
        positions[-1] = np.array([0.0, 0.0, 2.0])
        self.atoms.set_positions(positions)


def _case():
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 10.0
    G.graph["pbc"] = np.array([False, False, False])
    G.add_node(
        0,
        element="Cu",
        position=np.array([0.0, 0.0, 0.0]),
        type="surface",
    )
    G.add_node(
        100,
        element="H",
        position=np.array([0.0, 0.0, 1.0]),
        type="anchor",
        optimised=False,
    )
    G.add_edge(0, 100, distance=1.0, offset=(0, 0, 0), anchor_bond=True)

    site = AdsorbateSite(
        smiles="[H]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[0.0, 0.0, 1.0]]),
        iso_class=0,
        members=[[frozenset({0})]],
        member_positions=[np.array([[0.0, 0.0, 1.0]])],
        member_node_ids=[[100]],
    )
    get_cache(G).adsorbate_sites["[H]"] = [site]

    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="H")
    reactant = SimpleNamespace(
        smiles="[H]",
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
        energy=1.0,
    )
    atoms_template = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    return G, site, reactant, atoms_template


def _run(G, reactant, atoms_template, *, optimizer=None, only_iso_classes=None):
    if optimizer is None:
        optimizer = MoveAdsorbateOptimizer
    return opt_site._run(
        G,
        "[H]",
        reactant,
        atoms_template,
        ConstantCalculator(),
        fmax=0.05,
        max_steps=1,
        optimizer=optimizer,
        clean_energy=1.0,
        gas_energy=1.0,
        nl_mult=1.0,
        only_iso_classes=only_iso_classes,
    )


def test_stable_result_commits_relaxed_geometry_and_graph_nodes(monkeypatch):
    G, site, reactant, atoms_template = _case()

    monkeypatch.setattr(
        opt_site,
        "_verify_relaxed",
        lambda *args, **kwargs: (
            "stable",
            "ok",
            G,
            [frozenset({0})],
            frozenset({0}),
            G.subgraph([0]).copy(),
        ),
    )
    monkeypatch.setattr(
        opt_site,
        "_propagate_to_members",
        lambda G, ms, relaxed_ads: (
            [relaxed_ads.copy()],
            [frozenset({0})],
            [G.subgraph([0]).copy()],
        ),
    )

    _run(G, reactant, atoms_template)

    assert site.stable is True
    assert site.adsorption_energy == pytest.approx(3.0)
    assert np.allclose(site.positions, [[0.0, 0.0, 2.0]])
    assert np.allclose(site.member_positions[0], [[0.0, 0.0, 2.0]])
    assert np.allclose(G.nodes[100]["position"], [0.0, 0.0, 2.0])
    assert G.nodes[100]["optimised"] is True
    assert G.edges[0, 100]["distance"] == pytest.approx(2.0)


@pytest.mark.parametrize("status", ["site_changed", "broken"])
def test_unstable_result_rolls_back_original_geometry(monkeypatch, status):
    G, site, reactant, atoms_template = _case()

    monkeypatch.setattr(
        opt_site,
        "_verify_relaxed",
        lambda *args, **kwargs: (
            status,
            "not stable",
            G if status == "site_changed" else None,
            [frozenset({0})] if status == "site_changed" else None,
            frozenset({0}) if status == "site_changed" else None,
            G.subgraph([0]).copy() if status == "site_changed" else None,
        ),
    )

    _run(G, reactant, atoms_template)

    assert site.stable is False
    assert site.adsorption_energy is None
    assert np.allclose(site.positions, [[0.0, 0.0, 1.0]])
    assert np.allclose(site.member_positions[0], [[0.0, 0.0, 1.0]])
    assert np.allclose(G.nodes[100]["position"], [0.0, 0.0, 1.0])
    assert G.nodes[100]["optimised"] is False


def test_optimizer_exception_rolls_back_original_geometry(monkeypatch):
    G, site, reactant, atoms_template = _case()

    def fail(self, fmax=0.05, steps=200):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(MoveAdsorbateOptimizer, "run", fail)
    _run(G, reactant, atoms_template)

    assert site.stable is False
    assert site.adsorption_energy is None
    assert np.allclose(site.positions, [[0.0, 0.0, 1.0]])
    assert np.allclose(site.member_positions[0], [[0.0, 0.0, 1.0]])


def test_only_iso_classes_skips_unselected_sites():
    G, site, reactant, atoms_template = _case()
    site.stable = True

    _run(G, reactant, atoms_template, only_iso_classes=[99])

    assert site.stable is True
    assert np.allclose(site.positions, [[0.0, 0.0, 1.0]])
    assert np.allclose(site.member_positions[0], [[0.0, 0.0, 1.0]])
