"""Tests for checkpoint restart payloads."""

from __future__ import annotations

import networkx as nx
from ase import Atoms
from ase.calculators.emt import EMT

from autokmc.io.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    make_checkpoint_state,
    save_checkpoint,
)


def test_checkpoint_roundtrip_strips_calculators(tmp_path):
    G = nx.Graph()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = EMT()
    G.graph["atoms"] = atoms

    state = make_checkpoint_state(
        step=3,
        time_s=1.25,
        graph=G,
        adsorbate_sites=[],
        diffusion_sites=[],
        bond_sites=[],
        reactants=[],
        frozen_indices=[0],
        history=[(1, 0.1)],
        reaction_counts={"adsorption": 1},
    )
    path = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    loaded = load_checkpoint(path)

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.step == 3
    assert loaded.time_s == 1.25
    assert loaded.frozen_indices == [0]
    assert loaded.graph.graph["atoms"].calc is None
