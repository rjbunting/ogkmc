"""Round-trip tests for the ISAAC/extxyz reaction-result database."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import networkx as nx
import pytest
from ase import Atoms

import autokmc.io.calculation_cache as calculation_cache
from autokmc.io.calculation_cache import (
    apply_cached_states,
    calculation_cache_key,
    initialise_calculation_database,
    load_calculation_record,
    make_calculation_record,
    state_payload,
    validate_isaac_record,
    write_calculation_record,
    write_isaac_export,
)
from autokmc.io.reaction_graph import normalise_reaction_graph


def _graph(*, offset: int = 0, iso_class: int = 3, element: str = "Pt") -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(1 + offset, type="surface", element=element)
    graph.add_node(2 + offset, type="surface", element=element)
    graph.add_node(
        3 + offset,
        type="adsorbate",
        element="C",
        reactant="[C-]#[O+]",
        reactant_index=0,
        iso_class=iso_class,
    )
    graph.add_edge(1 + offset, 2 + offset)
    graph.add_edge(1 + offset, 3 + offset, anchor_bond=True)
    return normalise_reaction_graph(
        graph,
        endpoint_node_ids=(3 + offset,),
        endpoint_role="site",
    )


def _atoms(shift: float = 0.0) -> Atoms:
    return Atoms(
        "PtCO",
        positions=[[0, 0, 0], [0, 0, 1.8 + shift], [0, 0, 2.9 + shift]],
        cell=[8, 8, 8],
        pbc=True,
    )


def _adsorption_record(root, *, graph=None):
    graph = graph or _graph()
    operation = {
        "reactant_smiles": "[C-]#[O+]",
        "iso_class": 7,
        "lateral_class": 2,
        "temperature_k": 500.0,
    }
    parameters = {
        "fmax": 0.05,
        "max_steps": 100,
        "calculator": {"class": "fairchem.core.FAIRChemCalculator"},
    }
    key = calculation_cache_key(
        kind="adsorption",
        identity=operation,
        parameters=parameters,
        inputs={"occupied_initial": _atoms()},
    )
    record = make_calculation_record(
        kind="adsorption",
        cache_key=key,
        operation=operation,
        parameters=parameters,
        inputs={"reactant_smiles": "[C-]#[O+]"},
        states={
            "occupied": state_payload(
                _atoms(),
                energy_ev=-10.5,
                properties={"g_occupied": -10.2, "frequencies_occupied_ev": [0.1]},
            ),
            "unoccupied": state_payload(_atoms(0.1), energy_ev=-8.0),
        },
        reaction_graph=graph,
    )
    path = write_calculation_record(root, "adsorption", key, record)
    return key, operation, parameters, path


def test_writes_isaac_record_extxyz_assets_and_sqlite_index(tmp_path):
    root = tmp_path / "reaction_db"
    _, _, _, record_path = _adsorption_record(root)

    isaac = json.loads(record_path.read_text())
    validate_isaac_record(isaac)
    assert isaac["isaac_record_version"] == "1.05"
    assert isaac["sample"]["material"]["formula"] == "Pt"
    assert isaac["system"]["technique"] == "machine_learning_potential"
    assert {asset["uri"] for asset in isaac["assets"]} == {
        "reaction_graph.json",
        "occupied.extxyz",
        "unoccupied.extxyz",
    }
    assert all(len(asset["sha256"]) == 64 for asset in isaac["assets"])
    assert not any("atoms" in descriptor for descriptor in isaac["descriptors"]["outputs"][0]["descriptors"])
    assert "positions_A" not in json.dumps(isaac)

    with closing(sqlite3.connect(root / "index.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1


def test_graph_search_ignores_node_ids_and_run_local_iso_classes(tmp_path):
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)

    hit = load_calculation_record(
        root,
        "adsorption",
        "a-different-exact-key",
        reaction_graph=_graph(offset=100, iso_class=99),
        operation={"reactant_smiles": "[C-]#[O+]", "iso_class": 99},
        parameters=parameters,
    )
    assert hit is not None
    assert hit["states"]["occupied"]["energy_ev"] == pytest.approx(-10.5)
    assert hit["states"]["occupied"]["atoms"].get_chemical_symbols() == ["Pt", "C", "O"]


def test_graph_search_intentionally_ignores_geometry(tmp_path):
    """Geometry-blind matching is a documented method limitation."""
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)
    query = _graph(offset=20)
    for index, node in enumerate(query):
        query.nodes[node]["position"] = [100.0 * index, 0.0, 0.0]

    hit = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=query,
        operation={"reactant_smiles": "[C-]#[O+]", "iso_class": 8},
        parameters=parameters,
    )

    assert hit is not None


def test_graph_search_confirms_isomorphism_after_hash_lookup(tmp_path, monkeypatch):
    # Force a prefilter collision: the authoritative GraphMatcher must still
    # reject a Cu graph against a stored Pt graph.
    monkeypatch.setattr(calculation_cache, "reaction_graph_hash", lambda graph: "collision")
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)

    miss = load_calculation_record(
        root,
        "adsorption",
        "a-different-exact-key",
        reaction_graph=_graph(offset=50, iso_class=3, element="Cu"),
        operation={"reactant_smiles": "[C-]#[O+]"},
        parameters=parameters,
    )
    assert miss is None


def test_checksum_failure_rejects_database_hit(tmp_path):
    root = tmp_path / "reaction_db"
    key, operation, parameters, record_path = _adsorption_record(root)
    occupied = record_path.parent / "occupied.extxyz"
    occupied.write_bytes(occupied.read_bytes() + b"\n# tampered\n")

    assert load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
        operation=operation,
        parameters=parameters,
    ) is None


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_index_is_rebuilt_from_verified_record_folders(tmp_path, failure):
    root = tmp_path / "reaction_db"
    key, operation, parameters, _ = _adsorption_record(root)
    index = root / "index.sqlite3"
    if failure == "missing":
        index.unlink()
    else:
        index.write_bytes(b"not a sqlite database")

    hit = load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
        operation=operation,
        parameters=parameters,
    )

    assert hit is not None
    with closing(sqlite3.connect(index)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1


def test_database_and_isaac_record_link_to_run_id(tmp_path):
    root = tmp_path / "reaction_db"
    initialise_calculation_database(root, run_id="run-abc")
    _, _, _, record_path = _adsorption_record(root)

    manifest = json.loads((root / "database_manifest.json").read_text())
    isaac = json.loads(record_path.read_text())
    assert manifest["current_run_id"] == "run-abc"
    assert manifest["run_ids"] == ["run-abc"]
    assert isaac["system"]["configuration"]["autokmc"]["run_id"] == "run-abc"


def test_official_isaac_schema_rejects_invalid_enum(tmp_path):
    _, _, _, record_path = _adsorption_record(tmp_path / "reaction_db")
    isaac = json.loads(record_path.read_text())
    isaac["record_domain"] = "not-an-official-domain"

    with pytest.raises(ValueError, match="ISAAC schema validation failed"):
        validate_isaac_record(isaac)


def test_required_extxyz_states_cannot_be_omitted():
    with pytest.raises(ValueError, match="missing required states"):
        make_calculation_record(
            kind="adsorption",
            cache_key="abc",
            operation={"reactant_smiles": "[C-]#[O+]"},
            parameters={},
            inputs={},
            states={"occupied": state_payload(_atoms(), energy_ev=-1.0)},
            reaction_graph=_graph(),
        )


@pytest.mark.parametrize(
    ("kind", "states", "filenames"),
    [
        (
            "diffusion",
            {"state_a": -5.0, "state_b": -4.8, "transition": -4.0},
            {"state_a.extxyz", "state_b.extxyz", "ts.extxyz"},
        ),
        (
            "bond",
            {"state_ab": -7.0, "state_c": -8.0, "transition": -6.5},
            {"state_ab.extxyz", "state_c.extxyz", "ts.extxyz"},
        ),
    ],
)
def test_neb_reactions_write_required_states_and_optional_path(
    tmp_path, kind, states, filenames,
):
    root = tmp_path / kind
    operation = (
        {"reactant_smiles": "[C-]#[O+]"}
        if kind == "diffusion"
        else {"smiles_a": "[C-]#[O+]", "smiles_b": "[O]", "smiles_c": "O=C=O"}
    )
    parameters = {
        "n_images": 3,
        "climb": True,
        "calculator": {"class": "ase.calculators.emt.EMT"},
    }
    key = calculation_cache_key(
        kind=kind,
        identity=operation,
        parameters=parameters,
        inputs={"initial": _atoms()},
    )
    record = make_calculation_record(
        kind=kind,
        cache_key=key,
        operation=operation,
        parameters=parameters,
        inputs={},
        states={
            name: state_payload(_atoms(index * 0.05), energy_ev=energy)
            for index, (name, energy) in enumerate(states.items())
        },
        reaction_graph=_graph(),
        neb={
            "energies_ev": [-5.0, -4.0, -4.8],
            "path_atoms": [_atoms(0.0), _atoms(0.05), _atoms(0.1)],
        },
    )
    record_path = write_calculation_record(root, kind, key, record)
    isaac = json.loads(record_path.read_text())
    assert {asset["uri"] for asset in isaac["assets"]} == {
        "reaction_graph.json",
        "neb_path.extxyz",
        *filenames,
    }

    hit = load_calculation_record(root, kind, key, reaction_graph=_graph())
    assert hit is not None
    assert len(hit["neb"]["path"]) == 3


def test_verified_states_hydrate_lateral_class_and_export(tmp_path):
    root = tmp_path / "reaction_db"
    key, _, _, _ = _adsorption_record(root)
    hit = load_calculation_record(root, "adsorption", key, reaction_graph=_graph())
    lateral = SimpleNamespace(stable=None)

    assert apply_cached_states(
        lateral,
        hit,
        {
            "occupied": ("energy_occupied", "atoms_occupied"),
            "unoccupied": ("energy_unoccupied", "atoms_unoccupied"),
        },
    )
    assert lateral.stable is True
    assert lateral.energy_occupied == pytest.approx(-10.5)
    assert lateral.g_occupied == pytest.approx(-10.2)

    export_path = write_isaac_export(root, tmp_path / "isaac_records.json")
    exported = json.loads(export_path.read_text())
    assert len(exported) == 1
    assert exported[0]["record_id"] == hit["isaac_record"]["record_id"]
    assert all(
        asset["uri"].startswith("reaction_db/records/")
        for asset in exported[0]["assets"]
    )
