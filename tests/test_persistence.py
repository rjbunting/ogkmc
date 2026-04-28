"""Tests for autokmc.persistence — ReactionWriter + atoms_from_graph."""

from __future__ import annotations

import json

from ase.io import read as ase_read

from autokmc.persistence import (
    ReactionWriter,
    atoms_from_graph,
    PERSISTENCE_SCHEMA_VERSION,
)


def test_atoms_from_graph_includes_only_occupied_adsorbates(synth_graph):
    atoms = atoms_from_graph(synth_graph)
    syms = atoms.get_chemical_symbols()
    assert syms == ["Cu", "Cu", "C"]
    assert tuple(atoms.pbc) == (True, True, False)
    assert atoms.cell[0, 0] == 10.0


def test_atoms_from_graph_excludes_unoccupied(synth_graph):
    synth_graph.nodes[100]["occupied"] = False
    atoms = atoms_from_graph(synth_graph)
    assert atoms.get_chemical_symbols() == ["Cu", "Cu"]


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

    folder = tmp_path / "reactions" / "iso0_lat0"
    assert folder.is_dir()
    assert (folder / "occupied.extxyz").is_file()
    assert (folder / "unoccupied.extxyz").is_file()
    assert (folder / "reaction.json").is_file()

    jsonl = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1
    payload = json.loads(jsonl[0])
    assert payload["schema_version"] == PERSISTENCE_SCHEMA_VERSION
    assert payload["kind"] == "adsorption"
    assert payload["reaction_dir"] == "reactions/iso0_lat0"
    assert "ΔE" in payload["description"]

    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert rxn_meta["iso_class"] == 0
    assert rxn_meta["lateral_class"] == 0
    assert rxn_meta["energies_ev"]["occupied"]   == -10.0
    assert rxn_meta["energies_ev"]["unoccupied"] == -8.5
    assert rxn_meta["energies_ev"]["gas_phase"]  == -14.0
    assert rxn_meta["stats"]["count"] == 1
    assert rxn_meta["calculator"]["import_path"] == "X.Y"

    a_occ_rt = ase_read(folder / "occupied.extxyz")
    assert a_occ_rt.get_chemical_symbols() == a_occ.get_chemical_symbols()


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

    folder = tmp_path / "reactions" / "iso0_lat0"
    rxn_meta = json.loads((folder / "reaction.json").read_text())
    assert rxn_meta["stats"]["count"] == 2
    assert rxn_meta["stats"]["first_step"] == 1
    assert rxn_meta["stats"]["last_step"]  == 2

    jsonl = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 2


def test_reaction_writer_warns_when_no_atoms(tmp_path, stub_reaction):
    """If check_site_stability hasn't stamped the atoms (stubs/tests),
    the writer still writes reaction.json + the events row."""
    w = ReactionWriter(tmp_path)
    w.record(step=1, time_s=1e-6, tau_s=1e-6, reaction=stub_reaction)
    w.close()

    folder = tmp_path / "reactions" / "iso0_lat0"
    assert (folder / "reaction.json").is_file()
    assert not (folder / "occupied.extxyz").exists()
    assert not (folder / "unoccupied.extxyz").exists()


def test_reaction_writer_close_idempotent(tmp_path):
    w = ReactionWriter(tmp_path)
    w.close()
    w.close()
