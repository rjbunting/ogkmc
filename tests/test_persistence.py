"""Tests for autokmc2.io.persistence — ReactionWriter + atoms_from_graph."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ase.io import read as ase_read

from autokmc2.io.persistence import (
    ReactionWriter,
    PERSISTENCE_SCHEMA_VERSION,
)
from autokmc2.io.atoms import atoms_from_graph


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

    folder = tmp_path / "reactions" / "adsorption" / "(C-)#(O+)" / "iso0_lat0"
    assert folder.is_dir()
    assert (folder / "occupied.extxyz").is_file()
    assert (folder / "unoccupied.extxyz").is_file()
    assert (folder / "reaction.json").is_file()

    jsonl = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1
    payload = json.loads(jsonl[0])
    assert payload["schema_version"] == PERSISTENCE_SCHEMA_VERSION
    assert payload["kind"] == "adsorption"
    assert payload["reaction_dir"] == "reactions/adsorption/(C-)#(O+)/iso0_lat0"
    assert "ΔE" in payload["description"]

    rxn_meta = json.loads((folder / "reaction.json").read_text())
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
    assert "dir=b_to_a" in payload["description"]

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
    assert "dir=couple" in payload["description"]


def test_invalid_diffusion_record_tolerates_missing_energies(tmp_path):
    ds = SimpleNamespace(iso_class=1, reactant="[O]")
    lc = SimpleNamespace(lateral_class=2, invalid_reason="NEB failed early")

    w = ReactionWriter(tmp_path)
    folder = w.write_invalid_diffusion(ds, lc)
    w.close()

    payload = json.loads((folder / "reaction.json").read_text())
    assert payload["valid"] is False
    assert payload["invalid_reason"] == "NEB failed early"
    assert payload["energies_ev"] == {
        "state_a": None,
        "state_b": None,
        "transition": None,
    }


def test_reaction_writer_close_idempotent(tmp_path):
    w = ReactionWriter(tmp_path)
    w.close()
    w.close()
