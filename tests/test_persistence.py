"""Tests for autokmc.io.persistence — ReactionWriter + atoms_from_graph."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read, write as ase_write

from autokmc.io.persistence import (
    EventLogCommit,
    ReactionWriter,
    reconcile_event_log,
)
from autokmc.io.reaction_index import (
    load_reaction_index,
    resolve_event_definition,
)
from autokmc.io.schemas import (
    EVENT_ARTIFACT_TYPE,
    EVENT_SCHEMA_VERSION,
    REACTION_DOCUMENT_SCHEMA_VERSION,
    REACTION_INDEX_SCHEMA_VERSION,
)
from autokmc.io.event_log import EventHistory
from autokmc.io.atoms import atoms_from_graph
from autokmc.io import persistence as persistence_module
from autokmc.utils.telemetry import RuntimeTelemetry, telemetry_context


def test_atoms_from_graph_includes_only_occupied_adsorbates(tmp_path, synth_graph):
    synth_graph.graph["schema"] = "test-graph-v1"
    synth_graph.graph["run_id"] = "run-123"
    synth_graph.graph["frozen_indices"] = [0]
    synth_graph.nodes[100].update(
        reactant="[C-]#[O+]", reactant_index=0,
        site_iso_class=2, site_member_index=1,
    )
    atoms = atoms_from_graph(synth_graph)
    syms = atoms.get_chemical_symbols()
    assert syms == ["Cu", "Cu", "C"]
    assert tuple(atoms.pbc) == (True, True, True)
    assert atoms.cell[0, 0] == 10.0
    assert atoms.arrays["graph_node_id"].tolist() == ["0", "1", "100"]
    assert atoms.arrays["node_type"].tolist() == ["bulk", "surface", "adsorbate"]
    assert atoms.arrays["reactant_smiles"].tolist()[-1] == "[C-]#[O+]"
    assert atoms.arrays["site_iso_class"].tolist()[-1] == "2"
    assert atoms.arrays["site_member_index"].tolist()[-1] == "1"
    assert atoms.info["autokmc_graph_schema"] == "test-graph-v1"
    assert atoms.info["run_id"] == "run-123"
    assert isinstance(atoms.constraints[0], FixAtoms)
    assert atoms.constraints[0].get_indices().tolist() == [0]
    path = tmp_path / "snapshot.extxyz"
    ase_write(path, atoms, format="extxyz")
    restored = ase_read(path, format="extxyz")
    assert restored.arrays["graph_node_id"].tolist() == ["0", "1", "100"]
    assert restored.arrays["reactant_smiles"].tolist()[-1] == "[C-]#[O+]"
    assert restored.info["run_id"] == "run-123"
    assert restored.arrays["frozen"].tolist() == [True, False, False]


def test_atoms_from_graph_excludes_unoccupied(synth_graph):
    synth_graph.nodes[100]["occupied"] = False
    atoms = atoms_from_graph(synth_graph)
    assert atoms.get_chemical_symbols() == ["Cu", "Cu"]


def test_atoms_from_graph_uses_legacy_iso_class_as_metadata_fallback(synth_graph):
    synth_graph.nodes[100]["iso_class"] = 7
    atoms = atoms_from_graph(synth_graph)

    assert atoms.arrays["site_iso_class"].tolist()[-1] == "7"


def test_reaction_writer_creates_per_lateral_class_folder(
    tmp_path, stub_reaction, tiny_atoms
):
    # Stamp the stub lateral class with the relaxed atoms (this is what
    # check_site_stability does in production).
    a_occ   = tiny_atoms.copy()
    a_unocc = tiny_atoms.copy()[:2]   # slab only
    stub_reaction.lateral_class.atoms_occupied   = a_occ
    stub_reaction.lateral_class.atoms_unoccupied = a_unocc

    w = ReactionWriter(tmp_path, calculator_meta={"import_path": "X.Y"})

    w.record(
        step=1, time_s=1.0e-6, tau_s=1.0e-6, reaction=stub_reaction,
        gas_energies={"[C-]#[O+]": -14.0},
    )
    w.close()

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    assert folder.is_dir()
    assert (folder / "occupied.extxyz").is_file()
    assert (folder / "unoccupied.extxyz").is_file()
    assert (folder / "reaction.json").is_file()

    jsonl = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1
    payload = json.loads(jsonl[0])
    assert payload["artifact_type"] == EVENT_ARTIFACT_TYPE
    assert payload["schema_version"] == EVENT_SCHEMA_VERSION
    assert payload["event_id"].startswith("event-")
    assert payload["reaction_id"].startswith("reaction-")
    assert payload["kind"] == "adsorption"
    assert payload["inputs"][0]["phase"] == "gas"
    assert payload["outputs"][0]["phase"] == "surface"
    assert payload["outputs"][0]["placement_id"].startswith("placement-")
    assert {
        "description",
        "reaction_dir",
        "template",
        "gas_product",
    }.isdisjoint(payload)

    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    resolved = resolve_event_definition(
        payload,
        definitions,
        require_definition=True,
    )
    assert resolved["reaction_dir"] == "reactions/adsorption/(C-)#(O+)/iso0_lat0"
    assert "ΔE" in resolved["description"]
    definition = definitions[payload["reaction_id"]]
    assert definition["schema_version"] == REACTION_INDEX_SCHEMA_VERSION
    assert definition["firing_count"] == 1
    assert definition["directions"] == ["adsorption", "desorption"]

    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert rxn_meta["schema_version"] == REACTION_DOCUMENT_SCHEMA_VERSION
    assert rxn_meta["reaction_id"] == payload["reaction_id"]
    assert rxn_meta["iso_class"] == 0
    assert rxn_meta["lateral_class"] == 0
    assert rxn_meta["energies_ev"]["occupied"]   == -10.0
    assert rxn_meta["energies_ev"]["unoccupied"] == -8.5
    assert rxn_meta["energies_ev"]["gas_phase"]  == -14.0
    assert rxn_meta["free_energies_ev"]["g_gas"] is None
    assert rxn_meta["stats"]["count"] == 1
    assert rxn_meta["calculator"]["import_path"] == "X.Y"

    a_occ_rt = ase_read(folder / "occupied.extxyz")
    assert a_occ_rt.get_chemical_symbols() == a_occ.get_chemical_symbols()


def test_reaction_writer_atomically_publishes_structure_files(
    tmp_path,
    stub_reaction,
    tiny_atoms,
    monkeypatch,
):
    stub_reaction.lateral_class.atoms_occupied = tiny_atoms.copy()
    stub_reaction.lateral_class.atoms_unoccupied = tiny_atoms.copy()[:2]
    published: list[Path] = []
    atomic_output_path = persistence_module.atomic_output_path

    @contextmanager
    def track_publication(path):
        with atomic_output_path(path) as temporary:
            yield temporary
        published.append(Path(path))

    monkeypatch.setattr(
        persistence_module,
        "atomic_output_path",
        track_publication,
    )

    writer = ReactionWriter(tmp_path)
    folder = writer.ensure_reaction(stub_reaction, step=0)
    writer.close()

    assert folder / "occupied.extxyz" in published
    assert folder / "unoccupied.extxyz" in published
    assert not list(folder.glob(".*.extxyz.*"))


def test_reaction_writer_persists_initial_structures_and_neb_paths(
    tmp_path,
    stub_reaction,
    tiny_atoms,
):
    initial = tiny_atoms.copy()
    optimized = tiny_atoms.copy()
    path_initial = [tiny_atoms.copy(), tiny_atoms.copy()]
    path_optimized = [tiny_atoms.copy(), tiny_atoms.copy()]

    adsorption_lateral = stub_reaction.lateral_class
    adsorption_lateral.atoms_occupied_initial = initial.copy()
    adsorption_lateral.atoms_unoccupied_initial = initial.copy()
    adsorption_lateral.atoms_occupied = optimized.copy()
    adsorption_lateral.atoms_unoccupied = optimized.copy()

    diffusion_lateral = SimpleNamespace(
        lateral_class=2,
        energy_a=-2.0,
        energy_b=-1.8,
        energy_ts=-1.0,
        atoms_a_initial=initial.copy(),
        atoms_b_initial=initial.copy(),
        atoms_a=optimized.copy(),
        atoms_b=optimized.copy(),
        atoms_ts=optimized.copy(),
        atoms_neb_path_initial=path_initial,
        atoms_neb_path=path_optimized,
    )
    diffusion_reaction = SimpleNamespace(
        kind="diffusion",
        direction="a_to_b",
        site=SimpleNamespace(iso_class=1, reactant="[O]"),
        member_index=0,
        lateral_class=diffusion_lateral,
        delta_e=0.2,
        barrier=1.0,
        rate=1.0,
    )

    bond_lateral = SimpleNamespace(
        lateral_class=3,
        energy_ab=-3.0,
        energy_c=-3.5,
        energy_ts=-2.0,
        atoms_ab_initial=initial.copy(),
        atoms_c_initial=initial.copy(),
        atoms_ab=optimized.copy(),
        atoms_c=optimized.copy(),
        atoms_ts=optimized.copy(),
        atoms_neb_path_initial=path_initial,
        atoms_neb_path=path_optimized,
    )
    bond_reaction = SimpleNamespace(
        kind="bond",
        direction="couple",
        site=SimpleNamespace(
            iso_class=2,
            gas_product=False,
            template=SimpleNamespace(
                smiles_a="[H]",
                smiles_b="[H]",
                smiles_c="[H][H]",
                bond_type="SINGLE",
                source="test",
            ),
        ),
        member_index=0,
        lateral_class=bond_lateral,
        delta_e=-0.5,
        barrier=1.0,
        rate=1.0,
    )

    writer = ReactionWriter(tmp_path)
    adsorption_folder = writer.ensure_reaction(stub_reaction, step=0)
    diffusion_folder = writer.ensure_reaction(diffusion_reaction, step=0)
    bond_folder = writer.ensure_reaction(bond_reaction, step=0)
    writer.close()

    assert (adsorption_folder / "occupied_initial.extxyz").is_file()
    assert (adsorption_folder / "unoccupied_initial.extxyz").is_file()
    assert (diffusion_folder / "state_a_initial.extxyz").is_file()
    assert (diffusion_folder / "state_b_initial.extxyz").is_file()
    assert (diffusion_folder / "neb_path_initial.extxyz").is_file()
    assert (diffusion_folder / "neb_path.extxyz").is_file()
    assert (bond_folder / "state_ab_initial.extxyz").is_file()
    assert (bond_folder / "state_c_initial.extxyz").is_file()
    assert (bond_folder / "neb_path_initial.extxyz").is_file()
    assert (bond_folder / "neb_path.extxyz").is_file()

    diffusion_payload = json.loads(
        (diffusion_folder / "reaction.json").read_text()
    )
    assert diffusion_payload["atoms"]["state_a_initial"] == (
        "state_a_initial.extxyz"
    )
    assert diffusion_payload["atoms"]["neb_path_initial"] == (
        "neb_path_initial.extxyz"
    )


def test_reaction_writer_reuses_folder_across_events(
    tmp_path, stub_reaction, tiny_atoms
):
    """Second event for the same (iso, lat) bumps the count, but the
    .extxyz files are written exactly once."""
    a_occ   = tiny_atoms.copy()
    a_unocc = tiny_atoms.copy()[:2]
    stub_reaction.lateral_class.atoms_occupied   = a_occ
    stub_reaction.lateral_class.atoms_unoccupied = a_unocc

    w = ReactionWriter(tmp_path)

    w.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    w.record(step=2, time_s=2e-6, tau_s=1e-6, reaction=stub_reaction)
    w.close()

    assert w.n_unique_reactions == 1
    assert w.n_written == 2

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert rxn_meta["stats"]["count"] == 2
    assert rxn_meta["stats"]["first_step"] == 1
    assert rxn_meta["stats"]["last_step"]  == 2

    jsonl = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 2


def test_event_and_reaction_ids_are_deterministic_and_index_compacts(
    tmp_path,
    stub_reaction,
):
    first_root = tmp_path / "first"
    first = ReactionWriter(first_root, run_id="run-a")
    event_1 = first.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    event_2 = first.record(
        step=2,
        time_s=2e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    first.close()

    replay = ReactionWriter(tmp_path / "replay", run_id="run-a")
    replay_event = replay.record(
        step=1,
        time_s=9e-6,
        tau_s=9e-6,
        reaction=stub_reaction,
    )
    replay.close()
    other_run = ReactionWriter(tmp_path / "other", run_id="run-b")
    other_event = other_run.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    other_run.close()

    assert event_1.event_id == replay_event.event_id
    assert event_1.event_id != event_2.event_id
    assert event_1.event_id != other_event.event_id
    assert event_1.reaction_id == event_2.reaction_id
    assert event_1.reaction_id == replay_event.reaction_id
    assert event_1.reaction_id == other_event.reaction_id

    index_rows = [
        json.loads(line)
        for line in (first_root / "reactions" / "index.jsonl").read_text().splitlines()
    ]
    assert [row["record_type"] for row in index_rows] == ["header", "reaction"]
    assert index_rows[1]["reaction_id"] == event_1.reaction_id
    assert index_rows[1]["firing_count"] == 2
    assert index_rows[1]["first_step"] == 1
    assert index_rows[1]["last_step"] == 2


def test_reaction_writer_batches_reaction_json_until_sync(
    tmp_path, stub_reaction
):
    writer = ReactionWriter(tmp_path, run_id="run-a")
    writer.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"

    before = json.loads((folder / "reaction.json").read_text())
    assert before["stats"]["count"] == 0

    commit = writer.sync_for_checkpoint()
    after = json.loads((folder / "reaction.json").read_text())
    assert after["stats"] == {"count": 1, "first_step": 1, "last_step": 1}
    assert after["rate_energy_bases"] == ["electronic"]
    index_rows = [
        json.loads(line)
        for line in (tmp_path / "reactions" / "index.jsonl").read_text().splitlines()
    ]
    assert [row["record_type"] for row in index_rows] == [
        "header",
        "reaction",
        "stats",
    ]
    assert commit.count == 1
    assert commit.offset == (tmp_path / "events.jsonl").stat().st_size
    writer.close()


def test_reaction_writer_reports_append_sync_and_metadata_flush_telemetry(
    tmp_path,
    stub_reaction,
):
    telemetry = RuntimeTelemetry()
    writer = ReactionWriter(tmp_path)

    with telemetry_context(telemetry):
        writer.record(
            step=1,
            time_s=1e-6,
            tau_s=1e-6,
            reaction=stub_reaction,
        )
        writer.sync_for_checkpoint()
        writer.close()

    assert telemetry.counters["persistence.event_append.calls"] == 1
    assert telemetry.counters["persistence.event_sync.calls"] == 2
    # close() observes that the explicit checkpoint sync already published all
    # pending metadata instead of rewriting it a second time.
    assert telemetry.counters["persistence.metadata_flush.calls"] == 1
    assert telemetry.timings_s["persistence.event_append.seconds"] >= 0.0
    assert telemetry.timings_s["persistence.event_sync.seconds"] >= 0.0
    assert telemetry.timings_s["persistence.metadata_flush.seconds"] >= 0.0


def test_event_recovery_is_shared_without_reparsing_jsonl(
    tmp_path,
    stub_reaction,
    monkeypatch,
):
    writer = ReactionWriter(tmp_path, run_id="run-a")
    writer.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    durable = writer.sync_for_checkpoint()
    writer.close()

    event_path = tmp_path / "events.jsonl"
    recovered = reconcile_event_log(
        event_path,
        checkpoint_step=1,
        committed_event_count=durable.count,
        committed_event_offset=durable.offset,
        run_id="run-a",
    )
    assert isinstance(recovered.recovery.history, EventHistory)
    assert recovered.recovery.history.in_memory_count == 0
    assert recovered.recovery.history == [
        (1, 1e-6, "adsorption", 0, 0, 0, -0.2, 0.1, 1234000000.0)
    ]
    assert recovered.recovery.summary.n == 1
    [(key, folder_state)] = recovered.recovery.reaction_states.items()
    assert key == ("adsorption", "(C-)#(O+)", 0, 0)
    assert folder_state.count == 1
    assert folder_state.rate_energy_bases == {"electronic"}

    path_open = Path.open

    def reject_event_read(path, mode="r", *args, **kwargs):
        if path == event_path and str(mode).startswith("r"):
            raise AssertionError("events.jsonl was parsed a second time")
        return path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_event_read)
    resumed = ReactionWriter(
        tmp_path,
        append=True,
        run_id="run-a",
        event_recovery=recovered.recovery,
    )
    assert resumed.n_written == 1
    resumed.close()


def test_event_recovery_collects_history_only_when_explicitly_requested(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "step": step,
            "time_s": float(step),
            "kind": "adsorption",
            "reactant_smiles": "[O]",
            "iso_class": 0,
            "member_index": 0,
            "lateral_class": 0,
            "rate_hz": 2.0,
            "delta_e_ev": -0.2,
            "barrier_ev": 0.1,
            "run_id": "run-a",
        }
        for step in range(1, 101)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    offset = path.stat().st_size

    lazy = reconcile_event_log(
        path,
        checkpoint_step=100,
        committed_event_count=100,
        committed_event_offset=offset,
        run_id="run-a",
    )
    assert isinstance(lazy.recovery.history, EventHistory)
    assert lazy.recovery.history.in_memory_count == 0
    assert len(lazy.recovery.history) == 100
    assert lazy.recovery.summary.n == 100
    assert lazy.recovery.history[-1][0] == 100

    eager = reconcile_event_log(
        path,
        checkpoint_step=100,
        committed_event_count=100,
        committed_event_offset=offset,
        run_id="run-a",
        collect_history=True,
    )
    assert isinstance(eager.recovery.history, list)
    assert len(eager.recovery.history) == 100

    continuation = dict(rows[-1])
    continuation["step"] = 101
    continuation["time_s"] = 101.0
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(continuation) + "\n")
    lazy.recovery.history.append(
        (101, 101.0, "adsorption", 0, 0, 0, -0.2, 0.1, 2.0)
    )
    assert lazy.recovery.history.in_memory_count == 1
    lazy.recovery.history.mark_committed(
        count=101,
        offset=path.stat().st_size,
    )
    assert lazy.recovery.history.in_memory_count == 0
    assert lazy.recovery.history[-1][0] == 101


def test_reaction_static_payload_is_built_once_per_discovered_class(
    tmp_path,
    stub_reaction,
    monkeypatch,
):
    build_payload = persistence_module.build_reaction_payload
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return build_payload(*args, **kwargs)

    monkeypatch.setattr(persistence_module, "build_reaction_payload", counted)
    writer = ReactionWriter(tmp_path)
    writer.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    writer.record(
        step=2,
        time_s=2e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    writer.close()

    assert calls == [True]


def test_checkpoint_reconciliation_truncates_crash_tail_and_repairs_stats(
    tmp_path, stub_reaction
):
    writer = ReactionWriter(tmp_path, run_id="run-a")
    writer.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    committed = writer.sync_for_checkpoint()
    writer.record(step=2, time_s=2e-6, tau_s=1e-6, reaction=stub_reaction)
    writer.close()

    restored = reconcile_event_log(
        tmp_path / "events.jsonl",
        checkpoint_step=1,
        committed_event_count=committed.count,
        committed_event_offset=committed.offset,
        run_id="run-a",
    )
    assert restored == committed
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1]

    resumed = ReactionWriter(tmp_path, append=True, run_id="run-a")
    resumed.close()
    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    metadata = json.loads((folder / "reaction.json").read_text())
    assert metadata["stats"] == {"count": 1, "first_step": 1, "last_step": 1}
    assert metadata["last_event"]["step"] == 1


def test_checkpoint_reconciliation_rejects_event_log_behind_checkpoint(
    tmp_path, stub_reaction
):
    writer = ReactionWriter(tmp_path, run_id="run-a")
    writer.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    commit = writer.sync_for_checkpoint()
    writer.close()

    with pytest.raises(ValueError, match="behind checkpoint step 2"):
        reconcile_event_log(
            tmp_path / "events.jsonl",
            checkpoint_step=2,
            committed_event_count=commit.count,
            committed_event_offset=commit.offset,
            run_id="run-a",
        )


def test_checkpoint_reconciliation_rejects_empty_v3_commit_after_step_zero(tmp_path):
    with pytest.raises(ValueError, match="step 7.*empty event prefix"):
        reconcile_event_log(
            tmp_path / "missing-events.jsonl",
            checkpoint_step=7,
            committed_event_count=0,
            committed_event_offset=0,
            run_id="run-a",
        )


@pytest.mark.parametrize(
    ("count", "offset", "match"),
    [
        (1, None, "count and offset must both be set"),
        (None, 10, "count and offset must both be set"),
        (-1, 0, "negative count/offset"),
        (0, -1, "negative count/offset"),
    ],
)
def test_checkpoint_reconciliation_rejects_malformed_commit_metadata(
    tmp_path, count, offset, match
):
    with pytest.raises(ValueError, match=match):
        reconcile_event_log(
            tmp_path / "missing-events.jsonl",
            checkpoint_step=0,
            committed_event_count=count,
            committed_event_offset=offset,
            run_id="run-a",
        )


def test_checkpoint_reconciliation_rejects_wrong_run_and_non_boundary_offset(tmp_path):
    path = tmp_path / "events.jsonl"
    row = json.dumps({"step": 1, "run_id": "other-run"})
    path.write_text(row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match checkpoint run_id"):
        reconcile_event_log(
            path,
            checkpoint_step=1,
            committed_event_count=1,
            committed_event_offset=path.stat().st_size,
            run_id="run-a",
        )

    path.write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match="does not end at a JSONL boundary"):
        reconcile_event_log(
            path,
            checkpoint_step=1,
            committed_event_count=1,
            committed_event_offset=path.stat().st_size,
            run_id="other-run",
        )


def test_legacy_reconciliation_rejects_newline_terminated_malformed_row(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"step": 1}\nnot-json\n')

    with pytest.raises(ValueError, match="invalid event line 2"):
        reconcile_event_log(
            path,
            checkpoint_step=1,
            committed_event_count=None,
            committed_event_offset=None,
        )


def test_legacy_runless_prefix_remains_resumable_after_v3_continuation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "kind": "legacy"})
            for step in (1, 2, 3)
        )
        + "\n",
        encoding="utf-8",
    )

    legacy_commit = reconcile_event_log(
        path,
        checkpoint_step=2,
        committed_event_count=None,
        committed_event_offset=None,
        run_id="run-a",
    )
    legacy_rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in legacy_rows] == [1, 2]
    assert all(row["run_id"] == "run-a" for row in legacy_rows)
    assert legacy_commit == EventLogCommit(count=2, offset=path.stat().st_size)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 3, "kind": "new", "run_id": "run-a"}) + "\n")
    v3_commit = EventLogCommit(count=3, offset=path.stat().st_size)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 4, "kind": "crash", "run_id": "run-a"}) + "\n")

    restored = reconcile_event_log(
        path,
        checkpoint_step=3,
        committed_event_count=v3_commit.count,
        committed_event_offset=v3_commit.offset,
        run_id="run-a",
    )
    assert restored == v3_commit
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [
        1,
        2,
        3,
    ]


@pytest.mark.parametrize("exact", [False, True])
def test_event_reconciliation_rejects_step_gaps_without_modifying_file(
    tmp_path, exact
):
    path = tmp_path / "events.jsonl"
    rows = [{"step": 1}, {"step": 3}]
    if exact:
        rows = [{**row, "run_id": "run-a"} for row in rows]
    original = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    path.write_bytes(original)

    with pytest.raises(ValueError, match="not consecutive"):
        reconcile_event_log(
            path,
            checkpoint_step=3,
            committed_event_count=2 if exact else None,
            committed_event_offset=len(original) if exact else None,
            run_id="run-a",
        )

    assert path.read_bytes() == original


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("bad_step", [-1, 0, 1.0, True, "1"])
def test_event_reconciliation_rejects_invalid_json_steps_without_modifying_file(
    tmp_path, exact, bad_step
):
    path = tmp_path / "events.jsonl"
    original = (json.dumps({"step": bad_step}) + "\n").encode()
    path.write_bytes(original)

    with pytest.raises(ValueError, match="has no valid step"):
        reconcile_event_log(
            path,
            checkpoint_step=1,
            committed_event_count=1 if exact else None,
            committed_event_offset=len(original) if exact else None,
            run_id="run-a",
        )

    assert path.read_bytes() == original


def test_reaction_writer_append_restores_counts_and_run_id(
    tmp_path, stub_reaction, tiny_atoms
):
    stub_reaction.lateral_class.atoms_occupied = tiny_atoms.copy()
    stub_reaction.lateral_class.atoms_unoccupied = tiny_atoms.copy()[:2]
    first = ReactionWriter(tmp_path, run_id="run-a")
    first.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    first.close()

    resumed = ReactionWriter(tmp_path, append=True, run_id="run-a")
    resumed.record(step=2, time_s=2e-6, tau_s=1e-6, reaction=stub_reaction)
    resumed.close()

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["run_id"] == "run-a" for row in rows)
    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    metadata = json.loads((folder / "reaction.json").read_text())
    assert metadata["stats"] == {"count": 2, "first_step": 1, "last_step": 2}


def test_reaction_index_preserves_rate_bases_across_resume_without_refiring(
    tmp_path,
    stub_reaction,
):
    first = ReactionWriter(tmp_path, run_id="run-a")
    event = first.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
    )
    first.close()

    before = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    assert before[event.reaction_id]["rate_energy_bases"] == ["electronic"]

    resumed = ReactionWriter(tmp_path, append=True, run_id="run-a")
    resumed.close()

    after = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    assert after[event.reaction_id]["rate_energy_bases"] == ["electronic"]
    metadata = json.loads(
        (
            tmp_path
            / "reactions"
            / "adsorption"
            / "(C-)#(O+)"
            / "iso0_lat0"
            / "reaction.json"
        ).read_text()
    )
    assert metadata["rate_energy_bases"] == ["electronic"]


def test_reaction_writer_warns_when_no_atoms(tmp_path, stub_reaction):
    """If check_site_stability hasn't stamped the atoms (stubs/tests),
    the writer still writes reaction.json + the events row."""
    w = ReactionWriter(tmp_path)
    w.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    w.close()

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    assert (folder / "reaction.json").is_file()
    assert not (folder / "occupied.extxyz").exists()
    assert not (folder / "unoccupied.extxyz").exists()


def test_reaction_writer_keeps_species_folders_separate(tmp_path, make_reaction):
    w = ReactionWriter(tmp_path)
    w.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=make_reaction(smiles="[C-]#[O+]", iso=0, lateral=0),
    )
    w.record(
        step=2,
        time_s=2e-6,
        tau_s=1e-6,
        reaction=make_reaction(smiles="[O]", iso=0, lateral=0),
    )
    w.close()

    assert (
        tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    ).is_dir()
    assert (
        tmp_path / "reactions" / "adsorption" / "(O)" / "iso0_lat0"
    ).is_dir()
    assert w.n_unique_reactions == 2


def test_reaction_writer_persists_gas_free_energy(tmp_path, stub_reaction):
    w = ReactionWriter(tmp_path)
    w.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
        gas_energies={"[C-]#[O+]": -14.0},
        gas_free_energies={"[C-]#[O+]": -13.5},
    )
    w.close()

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert rxn_meta["free_energies_ev"]["g_gas"] == -13.5


def test_reaction_writer_records_adsorption_free_energy_event_fields(
    tmp_path, stub_reaction
):
    stub_reaction.lateral_class.g_occupied = -9.75
    stub_reaction.lateral_class.g_unoccupied = -8.25
    stub_reaction.lateral_class.frequencies_occupied_ev = [0.012398]
    stub_reaction.lateral_class.imaginary_occupied_ev = [0.0030995]
    stub_reaction.delta_e = -0.25
    stub_reaction.barrier = 0.1

    w = ReactionWriter(tmp_path)
    w.record(
        step=1,
        time_s=1e-6,
        tau_s=1e-6,
        reaction=stub_reaction,
        gas_energies={"[C-]#[O+]": -1.0},
        gas_free_energies={"[C-]#[O+]": -1.25},
    )
    w.close()

    payload = json.loads((tmp_path / "events.jsonl").read_text())
    assert payload["delta_e_ev"] == -0.5
    assert payload["barrier_ev"] == 0.1
    assert payload["delta_g_ev"] == -0.25
    assert payload["barrier_g_ev"] == 0.1
    assert payload["rate_energy_basis"] == "free_energy"
    assert payload["rate_delta_ev"] == -0.25
    assert payload["rate_barrier_ev"] == 0.1
    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    resolved = resolve_event_definition(payload, definitions, require_definition=True)
    assert "ΔG" in resolved["description"]

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    rxn_meta = json.loads((folder / "reaction.json").read_text())
    vib = rxn_meta["vibrations"]["occupied"]
    assert set(vib) == {"real_ev", "imag_ev", "zpe_ev", "entropy_ev_per_k"}
    assert vib["real_ev"] == [0.012398]
    assert vib["imag_ev"] == [0.0030995]


def test_reaction_writer_records_diffusion_direction(tmp_path):
    site = SimpleNamespace(iso_class=3, reactant="[O]", member_node_ids=[[1], [2]])
    lateral = SimpleNamespace(
        lateral_class=4,
        energy_a=-10.0,
        energy_b=-9.8,
        energy_ts=-9.5,
    )
    reaction = SimpleNamespace(
        kind="diffusion",
        direction="b_to_a",
        site=site,
        member_index=1,
        lateral_class=lateral,
        delta_e=-0.2,
        barrier=0.3,
        rate=2.0e5,
    )

    w = ReactionWriter(tmp_path)
    rec = w.record(step=7, time_s=2e-6, tau_s=1e-6, reaction=reaction)
    w.close()

    payload = json.loads((tmp_path / "events.jsonl").read_text())
    assert rec.direction == "b_to_a"
    assert payload["direction"] == "b_to_a"
    assert payload["delta_e_ev"] == pytest.approx(-0.2)
    assert payload["barrier_ev"] == pytest.approx(0.3)
    assert payload["delta_g_ev"] is None
    assert payload["barrier_g_ev"] is None
    assert payload["rate_energy_basis"] == "electronic"
    assert payload["rate_delta_ev"] == pytest.approx(-0.2)
    assert payload["rate_barrier_ev"] == pytest.approx(0.3)
    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    resolved = resolve_event_definition(payload, definitions, require_definition=True)
    assert "dir=b_to_a" in resolved["description"]

    folder = tmp_path / "reactions" / "diffusion" / "(O)" / "diff_iso3_lat4"
    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert "dir=b_to_a" in rxn_meta["description"]
    assert rxn_meta["last_event"]["direction"] == "b_to_a"


def test_reaction_writer_records_bond_direction(tmp_path):
    template = SimpleNamespace(
        smiles_a="[C]",
        smiles_b="[O]",
        smiles_c="[C]=O",
        bond_type="DOUBLE",
        source="test",
    )
    site = SimpleNamespace(iso_class=5, template=template, member_node_ids=[[1, 2, 3]])
    lateral = SimpleNamespace(
        lateral_class=6,
        energy_ab=-10.0,
        energy_c=-11.0,
        energy_ts=-9.5,
    )
    reaction = SimpleNamespace(
        kind="bond",
        direction="couple",
        site=site,
        member_index=0,
        lateral_class=lateral,
        delta_e=-1.0,
        barrier=0.5,
        rate=3.0e4,
    )

    w = ReactionWriter(tmp_path)
    w.record(step=8, time_s=2e-6, tau_s=1e-6, reaction=reaction)
    w.close()

    payload = json.loads((tmp_path / "events.jsonl").read_text())
    assert payload["direction"] == "couple"
    assert payload["delta_e_ev"] == -1.0
    assert payload["barrier_ev"] == 0.5
    assert payload["delta_g_ev"] is None
    assert payload["barrier_g_ev"] is None
    assert payload["rate_energy_basis"] == "electronic"
    assert payload["rate_delta_ev"] == -1.0
    assert payload["rate_barrier_ev"] == 0.5
    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    resolved = resolve_event_definition(payload, definitions, require_definition=True)
    assert "dir=couple" in resolved["description"]


def test_invalid_diffusion_record_tolerates_missing_energies(tmp_path):
    ds = SimpleNamespace(iso_class=1, reactant="[O]")
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    lc = SimpleNamespace(
        lateral_class=2,
        invalid_reason="NEB failed early",
        atoms_a_initial=atoms.copy(),
        atoms_b_initial=atoms.copy(),
        atoms_neb_path_initial=[atoms.copy(), atoms.copy()],
    )

    w = ReactionWriter(tmp_path)
    folder = w.write_invalid_diffusion(ds, lc)
    w.close()

    assert folder == (
        tmp_path
        / "diagnostics"
        / "invalid_diffusion"
        / "(O)"
        / "diff_iso1_lat2"
    )
    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["valid"] is False
    assert payload["invalid_reason"] == "NEB failed early"
    assert payload["energies_ev"] == {
        "state_a": None,
        "state_b": None,
        "transition": None,
    }
    assert (folder / "state_a_initial.extxyz").is_file()
    assert (folder / "state_b_initial.extxyz").is_file()
    assert (folder / "neb_path_initial.extxyz").is_file()
    assert payload["atoms"]["state_a_initial"] == "state_a_initial.extxyz"
    assert payload["atoms"]["state_b_initial"] == "state_b_initial.extxyz"
    assert payload["atoms"]["neb_path_initial"] == "neb_path_initial.extxyz"
    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    definition = definitions[payload["reaction_id"]]
    assert definition["valid"] is False
    assert definition["folder"] == (
        "diagnostics/invalid_diffusion/(O)/diff_iso1_lat2"
    )


def test_reaction_writer_close_idempotent(tmp_path):
    w = ReactionWriter(tmp_path)
    w.close()
    w.close()
