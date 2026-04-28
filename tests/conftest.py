"""Shared pytest fixtures.

The bulk of the test suite is calculator-free — it builds tiny synthetic
``Reaction``-shaped objects and a hand-rolled ``nx.Graph`` so that the
persistence / summary / config / CLI plumbing can be exercised without
pulling in NequIP or relying on slow ML weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import pytest

from ase import Atoms


# ---------------------------------------------------------------------------
# Tiny atoms for trajectory & XYZ round-trips
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_atoms() -> Atoms:
    return Atoms(
        symbols=["Cu", "Cu", "C", "O"],
        positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 0, 2.0], [1.25, 0, 3.15]],
        cell=[10.0, 10.0, 20.0],
        pbc=[True, True, False],
    )


# ---------------------------------------------------------------------------
# Synthetic Reaction / AdsorbateSite / AdsorbateSiteLateral shaped objects
# ---------------------------------------------------------------------------

@dataclass
class _StubLateral:
    lateral_class:    int = 0
    energy_occupied:  float | None = -10.0
    energy_unoccupied: float | None = -8.5


@dataclass
class _StubSite:
    iso_class: int = 0
    reactant:  str = "[C-]#[O+]"
    member_node_ids: list[list[int]] = field(default_factory=list)


@dataclass
class _StubReaction:
    kind:          str
    site:          _StubSite
    member_index:  int
    lateral_class: _StubLateral
    delta_e:       float
    barrier:       float
    rate:          float


@pytest.fixture
def stub_site() -> _StubSite:
    return _StubSite(iso_class=0, reactant="[C-]#[O+]",
                     member_node_ids=[[100], [101]])


@pytest.fixture
def stub_lateral() -> _StubLateral:
    return _StubLateral(lateral_class=0,
                        energy_occupied=-10.0,
                        energy_unoccupied=-8.5)


@pytest.fixture
def stub_reaction(stub_site, stub_lateral) -> _StubReaction:
    return _StubReaction(
        kind          = "adsorption",
        site          = stub_site,
        member_index  = 0,
        lateral_class = stub_lateral,
        delta_e       = -0.2,
        barrier       = 0.1,
        rate          = 1.234e9,
    )


@pytest.fixture
def make_reaction(stub_site, stub_lateral):
    """Factory: produce reactions on different lateral_class / kind / iso."""
    def _make(*, kind="adsorption", iso=0, member=0, lateral=0,
              delta_e=-0.1, barrier=0.1, rate=1.0e9, smiles="[C-]#[O+]"):
        site = _StubSite(iso_class=iso, reactant=smiles,
                         member_node_ids=[[100 + member]])
        lc = _StubLateral(lateral_class=lateral,
                          energy_occupied=-10.0,
                          energy_unoccupied=-8.5)
        return _StubReaction(
            kind=kind, site=site, member_index=member, lateral_class=lc,
            delta_e=delta_e, barrier=barrier, rate=rate,
        )
    return _make


# ---------------------------------------------------------------------------
# Synthetic graph for atoms_from_graph() round-trips
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_graph() -> nx.Graph:
    G = nx.Graph()
    G.graph["cell"] = np.eye(3) * 10.0
    G.graph["pbc"]  = (True, True, False)
    # Two slab atoms (one bulk, one surface) and one occupied adsorbate.
    G.add_node(0, type="bulk",      element="Cu", index=0,
               position=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, type="surface",   element="Cu", index=1,
               position=np.array([2.5, 0.0, 0.0]))
    G.add_node(100, type="adsorbate", element="C", index=2,
               position=np.array([1.25, 0.0, 2.0]),
               occupied=True)
    return G

