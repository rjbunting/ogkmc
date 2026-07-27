"""Round-trip tests for the ISAAC/extxyz reaction-result database."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

import autokmc.io.calculation_cache as calculation_cache
from autokmc.io.calculation_cache import (
    apply_cached_states,
    calculation_cache_key,
    calculator_identity,
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


def _adsorption_record(
    root,
    *,
    graph=None,
    parameters=None,
    occupied_initial=None,
    unoccupied_initial=None,
    input_metadata=None,
):
    graph = graph or _graph()
    operation = {
        "reactant_smiles": "[C-]#[O+]",
        "iso_class": 7,
        "lateral_class": 2,
        "temperature_k": 500.0,
    }
    parameters = parameters or {
        "fmax": 0.05,
        "max_steps": 100,
        "calculator": {"class": "fairchem.core.FAIRChemCalculator"},
    }
    occupied_initial = _atoms() if occupied_initial is None else occupied_initial
    unoccupied_initial = _atoms(0.1) if unoccupied_initial is None else unoccupied_initial
    inputs = {
        "occupied_initial": occupied_initial,
        "unoccupied_initial": unoccupied_initial,
        **dict(input_metadata or {}),
    }
    key = calculation_cache_key(
        kind="adsorption",
        identity=operation,
        parameters=parameters,
        inputs=inputs,
    )
    record = make_calculation_record(
        kind="adsorption",
        cache_key=key,
        operation=operation,
        parameters=parameters,
        inputs={**inputs, "reactant_smiles": "[C-]#[O+]"},
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
    configuration = isaac["system"]["configuration"]["autokmc"]
    assert configuration["scientific_input_hash"]
    assert configuration["input_frame_hash"]

    with closing(sqlite3.connect(root / "index.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
        indexed = connection.execute(
            """
            SELECT geometry_hash, scientific_input_hash,
                   electronic_scientific_input_hash, calculator_digest,
                   input_frame_hash, electronic_parameter_hash
            FROM records
            """
        ).fetchone()
        assert all(indexed)


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
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
    )
    assert hit is not None
    assert hit["states"]["occupied"]["energy_ev"] == pytest.approx(-10.5)
    assert hit["states"]["occupied"]["atoms"].get_chemical_symbols() == ["Pt", "C", "O"]


@pytest.mark.parametrize("transformation", ["translation_and_wrapping", "rotation"])
def test_graph_search_rejects_wrong_coordinate_frame_for_structure_outputs(
    tmp_path,
    transformation,
):
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)
    occupied = _atoms()
    unoccupied = _atoms(0.1)
    if transformation == "translation_and_wrapping":
        occupied.translate([3.4, -2.1, 1.3])
        unoccupied.translate([3.4, -2.1, 1.3])
        occupied.positions[2] += occupied.cell[0]
        unoccupied.positions[1] -= unoccupied.cell[1]
    else:
        occupied.rotate(37.0, "z", rotate_cell=True)
        unoccupied.rotate(37.0, "z", rotate_cell=True)

    # They are the same portable scientific geometry, but it would be unsafe
    # to return relaxed structures stored in the original coordinate frame.
    assert calculation_cache.input_geometry_fingerprint(
        {"occupied_initial": occupied, "unoccupied_initial": unoccupied}
    ) == calculation_cache.input_geometry_fingerprint(
        {"occupied_initial": _atoms(), "unoccupied_initial": _atoms(0.1)}
    )

    hit = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=20),
        operation={"reactant_smiles": "[C-]#[O+]", "iso_class": 8},
        parameters=parameters,
        inputs={
            "occupied_initial": occupied,
            "unoccupied_initial": unoccupied,
        },
    )

    assert hit is None


@pytest.mark.parametrize(
    ("field", "stored_value", "query_value"),
    [
        ("charge", 0, 1),
        ("gas_energy_ev", -1.25, -1.20),
    ],
)
def test_portable_lookup_rejects_changed_non_structure_scientific_input(
    tmp_path,
    field,
    stored_value,
    query_value,
):
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(
        root,
        input_metadata={field: stored_value},
    )

    miss = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=20),
        operation={"reactant_smiles": "[C-]#[O+]", "iso_class": 8},
        parameters=parameters,
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
            field: query_value,
        },
    )

    assert miss is None


def test_scientific_input_identity_ignores_only_run_local_ids():
    original = _atoms()
    translated = _atoms()
    translated.translate([2.0, -1.0, 0.5])

    stored = calculation_cache.scientific_input_fingerprint(
        {"initial": original, "charge": 0, "node_ids": [1, 2]},
    )
    equivalent = calculation_cache.scientific_input_fingerprint(
        {"initial": translated, "charge": 0, "node_ids": [101, 102]},
    )
    changed_charge = calculation_cache.scientific_input_fingerprint(
        {"initial": translated, "charge": 1, "node_ids": [101, 102]},
    )

    assert equivalent == stored
    assert changed_charge != stored


@pytest.mark.parametrize("attribute", ["initial_charges", "initial_magmoms", "tags", "custom"])
def test_geometry_identity_includes_calculator_relevant_atom_arrays(attribute):
    baseline = _atoms()
    changed = _atoms()
    if attribute == "initial_charges":
        changed.set_initial_charges([0.0, 0.25, -0.25])
    elif attribute == "initial_magmoms":
        changed.set_initial_magnetic_moments([0.0, 1.0, 0.0])
    elif attribute == "tags":
        changed.set_tags([0, 2, 0])
    else:
        changed.new_array("oxidation_state", np.array([0, 2, -2]))

    assert calculation_cache.input_geometry_fingerprint(
        {"initial": changed}
    ) != calculation_cache.input_geometry_fingerprint({"initial": baseline})

    reordered = changed[[2, 0, 1]]
    assert calculation_cache.input_geometry_fingerprint(
        {"initial": reordered}
    ) == calculation_cache.input_geometry_fingerprint({"initial": changed})


def test_geometry_identity_uses_full_lattice_metric_with_partial_pbc():
    diagonal = np.diag([3.0, 4.0, 5.0])
    angle = np.deg2rad(31.0)
    lattice_mixing = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    first = Atoms("Pt", positions=[[0.0, 0.0, 0.0]], cell=diagonal, pbc=[1, 0, 1])
    second = Atoms(
        "Pt",
        positions=[[0.0, 0.0, 0.0]],
        cell=lattice_mixing @ diagonal,
        pbc=[1, 0, 1],
    )

    # The cells share singular values, but not the full lattice Gram matrix.
    assert np.allclose(
        np.linalg.svd(first.cell.array, compute_uv=False),
        np.linalg.svd(second.cell.array, compute_uv=False),
    )
    assert calculation_cache.input_geometry_fingerprint(
        {"initial": first}
    ) != calculation_cache.input_geometry_fingerprint({"initial": second})


def test_geometry_identity_is_rotation_invariant_with_partial_pbc():
    atoms = Atoms(
        "PtCO",
        positions=[[0.1, 0.2, 0.3], [1.1, 0.5, 1.7], [2.0, 1.2, 2.4]],
        cell=[[4.0, 0.2, 0.0], [0.0, 5.0, 0.4], [0.3, 0.0, 6.0]],
        pbc=[True, False, True],
    )
    rotated = atoms.copy()
    rotated.rotate(43.0, "y", rotate_cell=True)

    assert calculation_cache.input_geometry_fingerprint(
        {"initial": rotated}
    ) == calculation_cache.input_geometry_fingerprint({"initial": atoms})


@pytest.mark.parametrize("displacement", [0.25, 100.0])
def test_graph_search_rejects_strained_or_displaced_geometry(tmp_path, displacement):
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)
    occupied = _atoms()
    occupied.positions[1, 2] += displacement

    miss = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=20),
        operation={"reactant_smiles": "[C-]#[O+]", "iso_class": 8},
        parameters=parameters,
        inputs={
            "occupied_initial": occupied,
            "unoccupied_initial": _atoms(0.1),
        },
    )

    assert miss is None


def test_exact_key_lookup_retains_authoritative_fast_path(tmp_path):
    root = tmp_path / "reaction_db"
    key, _, _, _ = _adsorption_record(root)
    distorted = _atoms()
    distorted.positions[1, 2] += 0.5

    hit = load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(offset=40),
        operation={"reactant_smiles": "[C-]#[O+]"},
        parameters={"calculator": {"class": "different.Calculator"}},
        inputs={"occupied_initial": distorted},
    )

    assert hit is not None


def test_exact_key_hit_skips_portable_geometry_fingerprints(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "reaction_db"
    key, _, _, _ = _adsorption_record(root)

    def _unexpected_fingerprints(*_args, **_kwargs):
        raise AssertionError("portable fingerprints should not run on an exact hit")

    monkeypatch.setattr(
        calculation_cache,
        "_calculation_fingerprints",
        _unexpected_fingerprints,
    )
    assert load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
    ) is not None


def test_write_reuses_each_expensive_geometry_signature(
    tmp_path,
    monkeypatch,
):
    calls = 0
    original = calculation_cache._atoms_geometry_signature

    def _counting_signature(atoms):
        nonlocal calls
        calls += 1
        return original(atoms)

    monkeypatch.setattr(
        calculation_cache,
        "_atoms_geometry_signature",
        _counting_signature,
    )
    _adsorption_record(tmp_path / "reaction_db")

    # occupied_initial and unoccupied_initial are each evaluated once, then
    # reused by geometry, scientific, and electronic-scientific digests.
    assert calls == 2


def test_empty_index_miss_then_write_fingerprints_only_once(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "reaction_db"
    root.mkdir()
    operation = {"reactant_smiles": "[C-]#[O+]"}
    parameters = {
        "fmax": 0.05,
        "calculator": {"class": "ase.calculators.emt.EMT"},
    }
    inputs = {
        "occupied_initial": _atoms(),
        "unoccupied_initial": _atoms(0.1),
    }
    key = calculation_cache_key(
        kind="adsorption",
        identity=operation,
        parameters=parameters,
        inputs=inputs,
    )
    calls = 0
    original = calculation_cache._atoms_geometry_signature

    def _counting_signature(atoms):
        nonlocal calls
        calls += 1
        return original(atoms)

    monkeypatch.setattr(
        calculation_cache,
        "_atoms_geometry_signature",
        _counting_signature,
    )
    memo = calculation_cache.CalculationFingerprintMemo()
    assert load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        fingerprint_memo=memo,
    ) is None
    assert calls == 0

    record = make_calculation_record(
        kind="adsorption",
        cache_key=key,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        states={
            "occupied": state_payload(_atoms(), energy_ev=-10.0),
            "unoccupied": state_payload(_atoms(0.1), energy_ev=-9.0),
        },
        reaction_graph=_graph(),
    )
    write_calculation_record(
        root,
        "adsorption",
        key,
        record,
        fingerprint_memo=memo,
    )
    assert calls == 2


def test_portable_miss_then_write_reuses_lookup_fingerprint_bundle(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "reaction_db"
    _, operation, parameters, _ = _adsorption_record(root)
    occupied = _atoms()
    occupied.positions[1, 2] += 0.3
    inputs = {
        "occupied_initial": occupied,
        "unoccupied_initial": _atoms(0.1),
    }
    key = calculation_cache_key(
        kind="adsorption",
        identity=operation,
        parameters=parameters,
        inputs=inputs,
    )
    calls = 0
    original = calculation_cache._atoms_geometry_signature

    def _counting_signature(atoms):
        nonlocal calls
        calls += 1
        return original(atoms)

    monkeypatch.setattr(
        calculation_cache,
        "_atoms_geometry_signature",
        _counting_signature,
    )
    memo = calculation_cache.CalculationFingerprintMemo()
    assert load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(offset=20),
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        fingerprint_memo=memo,
    ) is None
    assert calls == 2

    record = make_calculation_record(
        kind="adsorption",
        cache_key=key,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        states={
            "occupied": state_payload(_atoms(), energy_ev=-10.0),
            "unoccupied": state_payload(_atoms(0.1), energy_ev=-9.0),
        },
        reaction_graph=_graph(offset=20),
    )
    write_calculation_record(
        root,
        "adsorption",
        key,
        record,
        fingerprint_memo=memo,
    )
    assert calls == 2


def test_portable_index_rejects_metadata_before_asset_hydration(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "reaction_db"
    _, _, parameters, _ = _adsorption_record(root)
    calls = 0
    original = calculation_cache._load_record_path

    def _counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(calculation_cache, "_load_record_path", _counting_load)
    changed_parameters = {**parameters, "fmax": 0.025}
    assert load_calculation_record(
        root,
        "adsorption",
        "not-an-exact-key",
        reaction_graph=_graph(offset=20),
        operation={"reactant_smiles": "[C-]#[O+]"},
        parameters=changed_parameters,
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
    ) is None
    assert calls == 0


def test_electronic_match_ignores_only_thermochemistry_identity(tmp_path):
    root = tmp_path / "reaction_db"
    calculator = {"class": "ase.calculators.emt.EMT"}
    stored_parameters = {
        "fmax": 0.05,
        "max_steps": 100,
        "calculator": calculator,
        "free_energy_enabled": True,
        "temperature_k": 400.0,
        "free_energy": {"vibration_displacement": 0.01},
    }
    _adsorption_record(
        root,
        parameters=stored_parameters,
        input_metadata={"gas_gibbs_energy_ev": -1.0},
    )
    query_parameters = {
        **stored_parameters,
        "temperature_k": 700.0,
        "free_energy": {"vibration_displacement": 0.02},
    }
    query_inputs = {
        "occupied_initial": _atoms(),
        "unoccupied_initial": _atoms(0.1),
        "gas_gibbs_energy_ev": -0.8,
    }
    kwargs = {
        "reaction_graph": _graph(offset=20),
        "operation": {"reactant_smiles": "[C-]#[O+]"},
        "parameters": query_parameters,
        "inputs": query_inputs,
    }

    assert load_calculation_record(
        root,
        "adsorption",
        "different-key",
        **kwargs,
    ) is None
    hit = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        allow_electronic_match=True,
        **kwargs,
    )
    assert hit is not None
    assert hit["_cache_match"] == "electronic"


def test_isaac_validator_is_compiled_once():
    calculation_cache._isaac_validator.cache_clear()
    record = {
        "isaac_record_version": calculation_cache.ISAAC_RECORD_VERSION,
    }
    # Invalid records still exercise the cached compiled validator.
    with pytest.raises(ValueError):
        validate_isaac_record(record)
    with pytest.raises(ValueError):
        validate_isaac_record(record)
    info = calculation_cache._isaac_validator.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_record_without_new_portable_identity_remains_exact_key_only(tmp_path):
    root = tmp_path / "reaction_db"
    key, operation, parameters, record_path = _adsorption_record(root)
    isaac = json.loads(record_path.read_text())
    configuration = isaac["system"]["configuration"]["autokmc"]
    configuration.pop("scientific_input_hash")
    configuration.pop("input_frame_hash")
    record_path.write_text(json.dumps(isaac), encoding="utf-8")

    exact = load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
        operation=operation,
        parameters=parameters,
    )
    portable = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=10),
        operation=operation,
        parameters=parameters,
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
    )

    assert exact is not None
    assert portable is None


def test_calculator_identity_hashes_checkpoint_contents_not_local_path(tmp_path):
    checkpoint_a = tmp_path / "model-a.ckpt"
    checkpoint_b = tmp_path / "copied" / "model-b.ckpt"
    checkpoint_b.parent.mkdir()
    checkpoint_a.write_bytes(b"same learned weights")
    checkpoint_b.write_bytes(b"same learned weights")

    identity_a = calculator_identity(
        SimpleNamespace(parameters={"checkpoint": checkpoint_a, "device": "cuda:0"})
    )
    identity_b = calculator_identity(
        SimpleNamespace(parameters={"checkpoint": checkpoint_b, "device": "cpu"})
    )

    assert identity_a == identity_b
    artifact = identity_a["parameters"]["checkpoint"]
    assert artifact["artifact_sha256"]
    assert artifact["size_bytes"] == len(b"same learned weights")

    checkpoint_b.write_bytes(b"different learned weights")
    identity_changed = calculator_identity(
        SimpleNamespace(parameters={"checkpoint": checkpoint_b})
    )
    assert identity_changed["method_digest_sha256"] != identity_a["method_digest_sha256"]


def test_calculator_identity_rehashes_same_size_file_with_restored_mtime(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights-a")
    original_stat = checkpoint.stat()
    before = calculator_identity(
        SimpleNamespace(parameters={"checkpoint": checkpoint})
    )

    checkpoint.write_bytes(b"weights-b")
    os.utime(
        checkpoint,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = calculator_identity(
        SimpleNamespace(parameters={"checkpoint": checkpoint})
    )

    assert after["method_digest_sha256"] != before["method_digest_sha256"]


def test_calculator_identity_content_hashes_model_directories(tmp_path):
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "copied" / "model-b"
    for model in (model_a, model_b):
        (model / "submodule").mkdir(parents=True)
        (model / "config.json").write_text('{"layers": 4}', encoding="utf-8")
        (model / "submodule" / "weights.bin").write_bytes(b"learned weights")

    identity_a = calculator_identity(
        SimpleNamespace(parameters={"model_path": model_a, "device": "cuda:0"})
    )
    identity_b = calculator_identity(
        SimpleNamespace(parameters={"model_path": model_b, "device": "cpu"})
    )

    assert identity_a == identity_b
    artifact = identity_a["parameters"]["model_path"]
    assert artifact["artifact_kind"] == "directory"
    assert artifact["n_files"] == 2

    (model_b / "submodule" / "weights.bin").write_bytes(b"changed weights")
    changed = calculator_identity(
        SimpleNamespace(parameters={"model_path": model_b})
    )
    assert changed["method_digest_sha256"] != identity_a["method_digest_sha256"]


def test_calculator_identity_hashes_backend_specific_nested_paths(tmp_path):
    config = tmp_path / "backend-settings.yaml"
    config.write_text("cutoff: 5.0\n", encoding="utf-8")

    before = calculator_identity(
        SimpleNamespace(
            parameters={"backend": {"custom_config_location": str(config)}},
        )
    )
    artifact = before["parameters"]["backend"]["custom_config_location"]
    assert artifact["artifact_kind"] == "file"

    config.write_text("cutoff: 6.0\n", encoding="utf-8")
    after = calculator_identity(
        SimpleNamespace(
            parameters={"backend": {"custom_config_location": str(config)}},
        )
    )
    assert after["method_digest_sha256"] != before["method_digest_sha256"]


def test_calculator_identity_keeps_opaque_models_process_local_and_distinct():
    first_model = object()
    second_model = object()

    first = calculator_identity(
        SimpleNamespace(parameters={"model": first_model})
    )
    repeated = calculator_identity(
        SimpleNamespace(parameters={"model": first_model})
    )
    second = calculator_identity(
        SimpleNamespace(parameters={"model": second_model})
    )

    assert first == repeated
    assert first["parameters"]["model"]["identity_scope"] == "process-local"
    assert first["method_digest_sha256"] != second["method_digest_sha256"]


def test_portable_lookup_rejects_changed_checkpoint_artifact(tmp_path):
    root = tmp_path / "reaction_db"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"first model")
    stored_parameters = {
        "fmax": 0.05,
        "max_steps": 100,
        "calculator": calculator_identity(
            SimpleNamespace(parameters={"checkpoint": checkpoint})
        ),
    }
    _adsorption_record(root, parameters=stored_parameters)
    checkpoint.write_bytes(b"scientifically different model")
    query_parameters = {
        **stored_parameters,
        "calculator": calculator_identity(
            SimpleNamespace(parameters={"checkpoint": checkpoint})
        ),
    }

    miss = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=70),
        operation={"reactant_smiles": "[C-]#[O+]"},
        parameters=query_parameters,
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
    )

    assert miss is None


@pytest.mark.parametrize(
    ("stored_model", "query_model"),
    [("uma-s-1p2", "uma-m-1p1"), ("checkpoint-a", "checkpoint-b")],
)
def test_portable_lookup_rejects_distinct_explicit_model_identity(
    tmp_path, stored_model, query_model,
):
    root = tmp_path / "reaction_db"
    stored_parameters = {
        "fmax": 0.05,
        "max_steps": 100,
        "calculator": calculator_identity(
            SimpleNamespace(parameters={"name_or_path": stored_model})
        ),
    }
    _, _, _, _ = _adsorption_record(root, parameters=stored_parameters)
    query_parameters = {
        **stored_parameters,
        "calculator": calculator_identity(
            SimpleNamespace(parameters={"name_or_path": query_model})
        ),
    }

    miss = load_calculation_record(
        root,
        "adsorption",
        "different-key",
        reaction_graph=_graph(offset=80),
        operation={"reactant_smiles": "[C-]#[O+]"},
        parameters=query_parameters,
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
    )

    assert miss is None


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
        inputs={
            "occupied_initial": _atoms(),
            "unoccupied_initial": _atoms(0.1),
        },
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


def test_legacy_sqlite_index_is_migrated_and_metadata_backfilled(tmp_path):
    root = tmp_path / "reaction_db"
    key, operation, parameters, record_path = _adsorption_record(root)
    isaac = json.loads(record_path.read_text())
    configuration = isaac["system"]["configuration"]["autokmc"]
    calculation_cache.close_calculation_cache_connections(root)
    index = root / "index.sqlite3"
    index.unlink()
    with closing(sqlite3.connect(index)) as connection:
        connection.execute(
            """
            CREATE TABLE records (
                record_id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                operation_key TEXT NOT NULL,
                parameter_hash TEXT NOT NULL,
                graph_hash TEXT NOT NULL,
                record_path TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                isaac["record_id"],
                key,
                "adsorption",
                configuration["operation_key"],
                configuration["parameter_hash"],
                configuration["graph_hash"],
                str(record_path.relative_to(root)),
                isaac["timestamps"]["created_utc"],
            ),
        )
        connection.commit()

    assert load_calculation_record(
        root,
        "adsorption",
        key,
        reaction_graph=_graph(),
        operation=operation,
        parameters=parameters,
    ) is not None
    calculation_cache.close_calculation_cache_connections(root)
    with closing(sqlite3.connect(index)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(records)")
        }
        assert {
            "geometry_hash",
            "scientific_input_hash",
            "electronic_scientific_input_hash",
            "calculator_digest",
            "input_frame_hash",
            "electronic_parameter_hash",
        } <= columns
        metadata = connection.execute(
            """
            SELECT geometry_hash, scientific_input_hash,
                   electronic_scientific_input_hash, input_frame_hash
            FROM records WHERE cache_key = ?
            """,
            (key,),
        ).fetchone()
        assert all(metadata)


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


def test_cached_states_do_not_hydrate_run_dependent_gas_pressure():
    lateral = SimpleNamespace(stable=None, gas_pressure_bar=0.25)
    atoms = Atoms("H")

    assert apply_cached_states(
        lateral,
        {
            "states": {
                "state": {
                    "atoms": atoms,
                    "energy_ev": -1.0,
                    "properties": {"gas_pressure_bar": 98.0},
                },
            },
            "lateral_attributes": {
                "gas_product": True,
                "gas_pressure_bar": 99.0,
            },
        },
        {"state": ("energy", "atoms")},
    )
    assert lateral.gas_product is True
    assert lateral.gas_pressure_bar == pytest.approx(0.25)
