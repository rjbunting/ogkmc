from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from autokmc.core.graph import build_graph
from autokmc.sites.adsorbate import (
    AdsorbateSite,
    _adsorbate_edges_from_graph,
    _build_pruning_atoms,
    _geometry_connectivity_mismatch,
    _intended_adsorbate_edges,
    optimise_adsorbate_site_positions,
)


def _surface_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.graph["cell"] = np.eye(3) * 20.0
    graph.graph["pbc"] = np.array([True, True, False])
    graph.add_node(
        0,
        type="surface",
        element="Pt",
        position=np.array([0.0, 0.0, 0.0]),
        index=0,
    )
    graph.add_node(
        1,
        type="surface",
        element="Pt",
        position=np.array([2.75, 0.0, 0.0]),
        index=1,
    )
    graph.add_edge(0, 1)
    return graph


def _co_reactant() -> SimpleNamespace:
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="C")
    reactant_graph.add_node(1, element="O")
    reactant_graph.add_edge(0, 1)
    return SimpleNamespace(
        smiles="[C]=O",
        atoms=Atoms(
            "CO",
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]],
                dtype=float,
            ),
        ),
        graph=reactant_graph,
    )


def _co_site(positions) -> AdsorbateSite:
    return AdsorbateSite(
        reactant="[C]=O",
        n_atoms=2,
        atom_cliques=[frozenset({0}), None],
        positions=np.asarray(positions, dtype=float),
        iso_class=0,
        members=[[frozenset({0}), None]],
        member_node_ids=[],
    )


def _ase_connectivity_mismatch(graph, site, reactant, positions, nl_mult=1.2):
    original = np.asarray(site.positions, dtype=float).copy()
    try:
        site.positions = np.asarray(positions, dtype=float)
        atoms, n_slab, _n_ads, node_to_ase = _build_pruning_atoms(
            graph,
            site,
            list(reactant.atoms.get_chemical_symbols()),
        )
    finally:
        site.positions = original
    intended = _intended_adsorbate_edges(
        site,
        reactant,
        n_slab,
        node_to_ase,
    )
    actual_graph = build_graph(atoms, nl_mult=nl_mult)
    actual = _adsorbate_edges_from_graph(actual_graph, n_slab)
    missing = intended - actual
    extra = actual - intended
    return (missing, extra) if missing or extra else None


@pytest.mark.parametrize(
    "positions",
    [
        [[0.0, 0.0, 1.8], [0.0, 0.0, 2.95]],
        [[0.0, 0.0, 4.0], [0.0, 0.0, 5.15]],
        [[1.35, 0.0, 1.1], [1.35, 0.0, 2.25]],
        [[19.8, 0.0, 1.8], [19.8, 0.0, 2.95]],
    ],
)
def test_vector_connectivity_matches_ase_neighbor_list(positions):
    graph = _surface_graph()
    reactant = _co_reactant()
    site = _co_site(positions)

    expected = _ase_connectivity_mismatch(
        graph,
        site,
        reactant,
        positions,
    )
    actual = _geometry_connectivity_mismatch(
        graph,
        site,
        reactant,
        np.asarray(positions, dtype=float),
        nl_mult=1.2,
    )

    assert actual == expected


def test_connectivity_objective_analytical_jacobian(monkeypatch):
    graph = _surface_graph()
    reactant = _co_reactant()
    # Rotate the site away from the gas reference so R_base is nonidentity,
    # then probe near pi to exercise the non-small-angle SO(3) Jacobian.
    site = _co_site([[0.0, 0.0, 4.0], [1.15, 0.0, 4.0]])
    graph.graph["adsorbate_sites"] = {reactant.smiles: [site]}
    captured = {}

    def fake_minimize(fn, _x0, *, jac, **_kwargs):
        probe = np.array([0.11, -0.07, 0.05, 3.0, -0.2, 0.1])
        analytical = np.asarray(jac(probe), dtype=float)
        step = 1e-6
        numerical = np.empty(6)
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = step
            numerical[index] = (
                float(fn(probe + delta)) - float(fn(probe - delta))
            ) / (2.0 * step)
        captured["analytical"] = analytical
        captured["numerical"] = numerical
        return SimpleNamespace(fun=float(fn(probe)), x=probe)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    optimise_adsorbate_site_positions(
        graph,
        reactant.smiles,
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=1,
        max_iter=1,
    )

    assert captured["analytical"] == pytest.approx(
        captured["numerical"],
        rel=2e-5,
        abs=2e-5,
    )


def test_connectivity_objective_penalises_forbidden_graph_edge(monkeypatch):
    graph = _surface_graph()
    reactant = _co_reactant()
    # C is intentionally anchored to atom 0.  O is close enough to atom 1 to
    # create a forbidden surface bond under the graph neighbour cutoff.
    site = _co_site([[0.0, 0.0, 1.8], [2.3, 0.0, 1.0]])
    reactant.atoms.positions = site.positions.copy()
    graph.graph["adsorbate_sites"] = {reactant.smiles: [site]}
    captured = {}

    def fake_minimize(fn, x0, *, jac, **_kwargs):
        captured["energy"] = float(fn(x0))
        captured["gradient"] = np.asarray(jac(x0), dtype=float)
        return SimpleNamespace(fun=captured["energy"], x=np.asarray(x0))

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    optimise_adsorbate_site_positions(
        graph,
        reactant.smiles,
        reactant,
        restraint_weight=0.0,
        repulsion_weight=0.0,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=1,
        max_iter=1,
    )

    assert captured["energy"] > 0.0
    assert np.linalg.norm(captured["gradient"]) > 0.0


def test_connectivity_retry_warm_starts_each_orientation(monkeypatch):
    graph = _surface_graph()
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O")
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
    )
    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[10.0, 0.0, 0.0]]),
        iso_class=0,
        members=[[frozenset({0})]],
        member_node_ids=[],
    )
    graph.graph["adsorbate_sites"] = {reactant.smiles: [site]}
    starts = []
    results = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-10.0, 0.0, 1.8, 0.0, 0.0, 0.0]),
    ]

    def fake_minimize(fn, x0, **_kwargs):
        starts.append(np.asarray(x0, dtype=float).copy())
        result = results[len(starts) - 1]
        return SimpleNamespace(fun=float(fn(result)), x=result)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    optimise_adsorbate_site_positions(
        graph,
        reactant.smiles,
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=3,
        max_iter=1,
    )

    assert starts[0] == pytest.approx(np.zeros(6))
    assert starts[1] == pytest.approx(results[0])
    assert starts[2] == pytest.approx(results[1])


def test_nonfinite_optimizer_result_does_not_poison_warm_start(monkeypatch):
    graph = _surface_graph()
    reactant_graph = nx.Graph()
    reactant_graph.add_node(0, element="O")
    reactant = SimpleNamespace(
        smiles="[O]",
        atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
        graph=reactant_graph,
    )
    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[frozenset({0})],
        positions=np.array([[10.0, 0.0, 0.0]]),
        iso_class=0,
        members=[[frozenset({0})]],
        member_node_ids=[],
    )
    graph.graph["adsorbate_sites"] = {reactant.smiles: [site]}
    starts = []

    def fake_minimize(fn, x0, **_kwargs):
        starts.append(np.asarray(x0, dtype=float).copy())
        if len(starts) == 1:
            return SimpleNamespace(fun=np.nan, x=np.full(6, np.nan))
        result = np.array([-10.0, 0.0, 1.8, 0.0, 0.0, 0.0])
        return SimpleNamespace(fun=float(fn(result)), x=result)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    refined = optimise_adsorbate_site_positions(
        graph,
        reactant.smiles,
        reactant,
        n_restarts=1,
        try_flip=False,
        max_connectivity_attempts=2,
        max_iter=1,
    )

    assert len(refined) == 1
    assert len(starts) == 2
    assert starts[0] == pytest.approx(np.zeros(6))
    assert starts[1] == pytest.approx(np.zeros(6))
