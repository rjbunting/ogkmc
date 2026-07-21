"""Run-manifest provenance and restart behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx

from autokmc.core.constants import PERSISTENCE_SCHEMA_VERSION
from autokmc.io.run_manifest import finish_run_manifest, start_run_manifest


def _graph_and_site():
    graph = nx.Graph()
    graph.add_node(0, type="bulk", element="Pt")
    graph.add_node(1, type="surface", element="Pt")
    graph.add_node(10, type="adsorbate", element="C", occupied=True)
    site = SimpleNamespace(
        reactant="[C]=O",
        iso_class=2,
        member_node_ids=[[10]],
        members=[[frozenset({1})]],
    )
    return graph, site


def test_manifest_records_feed_initial_state_and_normalization(tmp_path):
    graph, site = _graph_and_site()
    config = tmp_path / "run.yaml"
    config.write_text("schema_version: '1'\n")
    path = tmp_path / "run_manifest.json"

    start_run_manifest(
        path,
        graph=graph,
        adsorbate_sites=[site],
        feed_reactants=[{
            "smiles": "[C]=O",
            "partial_pressure_bar": 1.0,
            "thermochemistry": {
                "symmetry_number": 1,
                "symmetry_number_source": "inferred",
                "point_group": "C*v",
                "symmetry_tolerance": 0.3,
            },
        }],
        temperature_k=500.0,
        random_seed=69,
        structure_kind="surface",
        composition="Pt",
        config_path=str(config),
        events_filename="events.jsonl",
        run_id="fd4912ce-ffb3-47d5-86b0-4ce5541c04a8",
        resolved_config={"kmc": {"temperature_k": 500.0}},
    )
    finish_run_manifest(path, final_step=100, final_time_s=2.5, steps_executed=100)

    payload = json.loads(path.read_text())
    assert payload["event_schema_version"] == PERSISTENCE_SCHEMA_VERSION
    assert payload["schema_version"] == "2"
    assert payload["run_id"] == "fd4912ce-ffb3-47d5-86b0-4ce5541c04a8"
    assert payload["config"]["content"] == "schema_version: '1'\n"
    assert payload["config"]["resolved"]["kmc"]["temperature_k"] == 500.0
    assert payload["feed_reactants"][0]["species"] == "[C]=O"
    assert payload["feed_reactants"][0]["thermochemistry"] == {
        "symmetry_number": 1,
        "symmetry_number_source": "inferred",
        "point_group": "C*v",
        "symmetry_tolerance": 0.3,
    }
    assert payload["catalyst"]["n_catalyst_atoms"] == 2
    assert payload["catalyst"]["n_surface_atoms"] == 1
    initial = payload["initial_state"]["occupied_surface_states"]
    assert initial[0]["species"] == "[C]=O"
    assert initial[0]["surface_cliques"] == [[1]]
    assert payload["result"]["final_time_s"] == 2.5


def test_resume_appends_segment_without_replacing_original_initial_state(tmp_path):
    graph, site = _graph_and_site()
    path = tmp_path / "run_manifest.json"
    kwargs = dict(
        path=path,
        graph=graph,
        adsorbate_sites=[site],
        feed_reactants=[{"smiles": "[C]=O", "partial_pressure_bar": 1.0}],
        temperature_k=500.0,
        random_seed=69,
        structure_kind="surface",
        composition="Pt",
        config_path=None,
    )
    start_run_manifest(**kwargs)
    start_run_manifest(**kwargs, initial_step=50, initial_time_s=1.25)

    payload = json.loads(path.read_text())
    assert payload["initial_state"]["step"] == 0
    assert len(payload["segments"]) == 2
    assert payload["segments"][1]["initial_step"] == 50
    assert payload["run_id"]
