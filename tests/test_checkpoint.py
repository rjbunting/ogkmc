"""Tests for checkpoint restart payloads."""

from __future__ import annotations

import networkx as nx
from types import SimpleNamespace
from ase import Atoms
from ase.calculators.emt import EMT

from autokmc.io.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    make_checkpoint_state,
    save_checkpoint,
)
from autokmc.utils.telemetry import RuntimeTelemetry, telemetry_context


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


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
        rng_state={"type": "numpy-generator", "state": {"counter": 4}},
        committed_event_count=3,
        committed_event_offset=712,
        metadata={"run_id": "run-123"},
    )
    path = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    loaded = load_checkpoint(path)

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.step == 3
    assert loaded.time_s == 1.25
    assert loaded.frozen_indices == [0]
    assert loaded.graph.graph["atoms"].calc is None
    assert loaded.rng_state == {"type": "numpy-generator", "state": {"counter": 4}}
    assert loaded.committed_event_count == 3
    assert loaded.committed_event_offset == 712
    assert loaded.metadata["run_id"] == "run-123"


def test_actual_checkpoint_save_reports_telemetry(tmp_path):
    state = make_checkpoint_state(
        step=1,
        time_s=0.1,
        graph=nx.Graph(),
        adsorbate_sites=[],
    )
    telemetry = RuntimeTelemetry()

    with telemetry_context(telemetry):
        save_checkpoint(tmp_path / "checkpoint.pkl", state)

    assert telemetry.counters["checkpoint.save.calls"] == 1
    assert telemetry.timings_s["checkpoint.save.seconds"] >= 0.0


def test_checkpoint_v2_migrates_without_claiming_an_event_commit(tmp_path):
    state = make_checkpoint_state(
        step=2,
        time_s=0.5,
        graph=nx.Graph(),
        adsorbate_sites=[],
        committed_event_count=2,
        committed_event_offset=128,
    )
    state.schema_version = "2"

    loaded = load_checkpoint(save_checkpoint(tmp_path / "legacy-v2.pkl", state))

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.committed_event_count is None
    assert loaded.committed_event_offset is None


def test_checkpoint_v3_migrates_with_commit_and_legacy_history_intact(tmp_path):
    state = make_checkpoint_state(
        step=2,
        time_s=0.5,
        graph=nx.Graph(),
        adsorbate_sites=[],
        history=[(1, 0.1), (2, 0.5)],
        committed_event_count=2,
        committed_event_offset=128,
    )
    state.schema_version = "3"

    loaded = load_checkpoint(save_checkpoint(tmp_path / "legacy-v3.pkl", state))

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.committed_event_count == 2
    assert loaded.committed_event_offset == 128
    assert loaded.history == [(1, 0.1), (2, 0.5)]


def test_checkpoint_preserves_hashable_keys_that_strip_to_dict(tmp_path):
    G = nx.Graph()
    key = HashableDict({"kind": "lat", "index": 0})
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = EMT()
    G.graph["cache"] = {key: atoms}

    state = make_checkpoint_state(
        step=4,
        time_s=2.0,
        graph=G,
        adsorbate_sites=[],
        diffusion_sites=[],
        bond_sites=[],
        reactants=[],
        frozen_indices=[],
        history=[],
        reaction_counts={},
    )
    path = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    loaded = load_checkpoint(path)

    [(loaded_key, loaded_atoms)] = loaded.graph.graph["cache"].items()
    assert isinstance(loaded_key, HashableDict)
    assert dict(loaded_key) == {"kind": "lat", "index": 0}
    assert loaded_atoms.calc is None


def test_checkpoint_preserves_shared_site_and_registry_identity(tmp_path):
    site = SimpleNamespace(iso_class=0, calc=EMT())
    diffusion = SimpleNamespace(site_a=site)
    reactant = SimpleNamespace(smiles="[O]", atoms=Atoms("O"))
    reactant.atoms.calc = EMT()
    G = nx.Graph()
    G.graph["adsorbate_clique_to_members"] = {frozenset({1}): [(site, 0)]}
    G.graph["bond_registry"] = {
        "species": {"[O]": reactant},
        "adsorbate_sites": {"[O]": [site]},
    }

    state = make_checkpoint_state(
        step=1,
        time_s=0.1,
        graph=G,
        adsorbate_sites=[site],
        diffusion_sites=[diffusion],
        bond_sites=[],
        reactants=[reactant],
    )
    loaded = load_checkpoint(save_checkpoint(tmp_path / "shared.pkl", state))

    restored_site = loaded.adsorbate_sites[0]
    assert loaded.graph.graph["adsorbate_clique_to_members"][frozenset({1})][0][0] is restored_site
    assert loaded.graph.graph["bond_registry"]["adsorbate_sites"]["[O]"][0] is restored_site
    assert loaded.diffusion_sites[0].site_a is restored_site
    assert loaded.graph.graph["bond_registry"]["species"]["[O]"] is loaded.reactants[0]
    assert restored_site.calc is None
    assert loaded.reactants[0].atoms.calc is None


def test_checkpoint_root_collections_never_alias_from_temporary_id_reuse(tmp_path):
    adsorbate = SimpleNamespace(kind="adsorbate")
    diffusion = SimpleNamespace(kind="diffusion")
    bond = SimpleNamespace(kind="bond")
    reactant = SimpleNamespace(kind="reactant")

    state = make_checkpoint_state(
        step=1,
        time_s=0.1,
        graph=nx.Graph(),
        adsorbate_sites=[adsorbate],
        diffusion_sites=[diffusion],
        bond_sites=[bond],
        reactants=[reactant],
    )
    loaded = load_checkpoint(save_checkpoint(tmp_path / "roots.pkl", state))

    assert [item.kind for item in loaded.adsorbate_sites] == ["adsorbate"]
    assert [item.kind for item in loaded.diffusion_sites] == ["diffusion"]
    assert [item.kind for item in loaded.bond_sites] == ["bond"]
    assert [item.kind for item in loaded.reactants] == ["reactant"]
    assert len({
        id(loaded.adsorbate_sites),
        id(loaded.diffusion_sites),
        id(loaded.bond_sites),
        id(loaded.reactants),
    }) == 4


def test_checkpoint_omits_lazily_reconstructible_graph_caches(tmp_path):
    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph.graph["surface_apsp"] = {
        "_cutoff": 10,
        "data": {1: {1: 0, 2: 1}, 2: {1: 1, 2: 0}},
    }
    graph.graph["_surface_shells_cache"] = {("seed", 10): {1, 2}}
    graph.graph["_clique_position_index_cache"] = {"positions": [1, 2]}
    graph.graph["_surface_atoms_array_cache"] = {"positions": [1, 2]}
    graph.graph["scientific_metadata"] = {"keep": True}

    state = make_checkpoint_state(
        step=1,
        time_s=0.1,
        graph=graph,
        adsorbate_sites=[],
    )
    loaded = load_checkpoint(save_checkpoint(tmp_path / "compact.pkl", state))

    assert "surface_apsp" not in loaded.graph.graph
    assert "_surface_shells_cache" not in loaded.graph.graph
    assert "_clique_position_index_cache" not in loaded.graph.graph
    assert "_surface_atoms_array_cache" not in loaded.graph.graph
    assert loaded.graph.graph["scientific_metadata"] == {"keep": True}
    # Preparing a checkpoint must never mutate the live simulation graph.
    assert "surface_apsp" in graph.graph
