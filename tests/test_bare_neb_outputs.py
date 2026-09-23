"""Bare NEB references remain inspectable without becoming KMC reactions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from ogkmc.io.checkpoint import load_checkpoint, make_checkpoint_state, save_checkpoint
from ogkmc.io.persistence import ReactionWriter
from ogkmc.io.reaction_index import load_reaction_index
from ogkmc.io.run_manifest import discover_quarantine_locations
from ogkmc.kmc.index import _ReactionIndex
from ogkmc.kmc.models import (
    KMCChannels, KMCFunctions, KMCObservers, KMCRunRequest, KMCRuntime,
    KMCSettings, KMCSystem,
)
from ogkmc.kmc.outputs import KMCOutputManager
from ogkmc.kmc.session import KMCSession


def _bare_site(kind):
    band = []
    for z, energy in ((0.0, -2.0), (0.5, -1.0), (1.0, -1.5)):
        atoms = Atoms("H", positions=[[0.0, 0.0, z]])
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((1, 3)))
        band.append(atoms)
    lc = SimpleNamespace(
        lateral_class=0, members=[], _seed_only=True, stable=True,
        atoms_neb_path=None, atoms_ts=band[1], energy_ts=-1.0,
        _warm_start_neb_path=band, _warm_start_neb_energies=[-2.0, -1.0, -1.5],
        _warm_start_member_index=0,
    )
    endpoints = ("ab", "c") if kind == "bond" else ("a", "b")
    for endpoint, atoms in zip(endpoints, (band[0], band[-1])):
        setattr(lc, f"atoms_{endpoint}", atoms)
        setattr(lc, f"atoms_{endpoint}_initial", atoms.copy())
        setattr(lc, f"energy_{endpoint}", atoms.get_potential_energy())
    site = SimpleNamespace(
        iso_class=8, lateral_classes=[lc], applicable_reactions=[], reactant="[H]",
        template=SimpleNamespace(smiles_a="[H]", smiles_b="[H]", smiles_c="[H][H]"),
    )
    return site, lc


@pytest.mark.parametrize("kind", ["bond", "diffusion"])
def test_bare_output_keeps_private_band_and_does_not_register_reaction(tmp_path, kind):
    site, lc = _bare_site(kind)
    writer = ReactionWriter(tmp_path, run_id="bare-run")
    folder = writer.write_bare_neb(site, lc, kind=kind, step=3)
    assert folder.parent.parent == tmp_path / "diagnostics" / "bare_neb" / kind
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["diagnostic_status"] == "completed"
    assert payload["lateral_interactions"] is False
    assert payload["discovery_step"] == 3
    assert payload["run_id"] == "bare-run"
    assert payload["neb_path_energies_ev"] == [-2.0, -1.0, -1.5]
    for name, filename in payload["atoms"].items():
        frames = read(folder / filename, index=":")
        assert len(frames) == (3 if name == "neb_path" else 1)
    frames = read(folder / "neb_path.extxyz", index=":")
    assert [frame.get_potential_energy() for frame in frames] == [-2.0, -1.0, -1.5]
    assert frames[1].get_forces() == pytest.approx(np.zeros((1, 3)))

    timestamp = (folder / "neb_path.extxyz").stat().st_mtime_ns
    assert writer.write_bare_neb(site, lc, kind=kind, step=4) == folder
    assert (folder / "neb_path.extxyz").stat().st_mtime_ns == timestamp
    assert writer.n_unique_reactions == writer.n_valid_reactions == writer.n_invalid_reactions == 0
    writer.close()
    assert not (tmp_path / "calculation_cache").exists()
    assert not list((tmp_path / "reactions").glob("*/*/*/reaction.json"))
    assert not load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    assert (tmp_path / "events.jsonl").read_text() == ""


@pytest.mark.parametrize("kind", ["bond", "diffusion"])
@pytest.mark.parametrize("status", ["numerical_failure", "invalid", "composite_direct_event"])
def test_failed_bare_output_keeps_failed_public_band(tmp_path, kind, status):
    site, lc = _bare_site(kind)
    lc.stable = False if status == "invalid" else None
    lc.last_failure_reason = "NEB failed" if status == "numerical_failure" else None
    lc.invalid_reason = "endpoint changed" if status == "invalid" else None
    lc.direct_event_status = "composite" if status == "composite_direct_event" else None
    lc.direct_event_reason = "registered intermediate" if lc.direct_event_status else None
    lc.atoms_neb_path_initial = [atoms.copy() for atoms in lc._warm_start_neb_path]
    lc.atoms_neb_path = lc._warm_start_neb_path
    lc.neb_path_energies = lc._warm_start_neb_energies
    del lc._warm_start_neb_path
    writer = ReactionWriter(tmp_path)
    folder = writer.write_bare_neb(site, lc, kind=kind, step=2)
    writer.close()
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["diagnostic_status"] == status
    assert payload["failure_reason"]
    assert len(read(folder / "neb_path_initial.extxyz", index=":")) == 3
    assert len(read(folder / "neb_path.extxyz", index=":")) == 3


def test_uncomputed_and_live_classes_do_not_create_bare_diagnostics(tmp_path):
    site, lc = _bare_site("bond")
    writer = ReactionWriter(tmp_path)
    uncomputed = SimpleNamespace(_seed_only=True, members=[], stable=None, lateral_class=1)
    site.gas_product = True
    site.gas_reactant = SimpleNamespace(atoms=Atoms("H2"), energy=-1.0)
    assert writer.write_bare_neb(site, uncomputed, kind="bond") is None
    lc.members = [0]
    assert writer.write_bare_neb(site, lc, kind="bond") is None
    writer.close()
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize("kind", ["bond", "diffusion"])
def test_bare_output_is_backfilled_after_partial_flush(tmp_path, kind):
    site, lc = _bare_site(kind)
    partial = SimpleNamespace(
        _seed_only=True, members=[], stable=None, lateral_class=lc.lateral_class,
        atoms_ts=lc.atoms_ts,
    )
    writer = ReactionWriter(tmp_path)
    folder = writer.write_bare_neb(site, partial, kind=kind, step=2)
    assert json.loads((folder / "diagnostic.json").read_text())["diagnostic_status"] == "incomplete"
    writer.write_bare_neb(site, lc, kind=kind, step=3)
    writer.close()
    payload = json.loads((folder / "diagnostic.json").read_text())
    assert payload["diagnostic_status"] == "completed"
    assert payload["discovery_step"] == 2
    assert len(read(folder / "neb_path.extxyz", index=":")) == 3


def test_output_manager_writes_bare_calculations_at_initialisation_and_each_step(tmp_path):
    bond, _ = _bare_site("bond")
    diffusion, _ = _bare_site("diffusion")
    writer = ReactionWriter(tmp_path)
    channels = KMCChannels(bond_sites=[bond], diffusion_sites=[])
    outputs = KMCOutputManager(
        KMCSystem(nx.Graph(), [], None, {}),
        KMCSettings(temperature=500.0, n_steps=1),
        channels, KMCObservers(reaction_writer=writer),
        KMCRuntime(
            rng=np.random.default_rng(4), gas_energies={}, gas_free_energies={},
            partial_pressures={}, reaction_index=_ReactionIndex([]), history=[],
            reaction_counts={}, current_time_s=0.0, start_step=0,
        ),
    )
    outputs.initialise()
    assert len(list((tmp_path / "diagnostics" / "bare_neb").glob("*/*/*/diagnostic.json"))) == 1
    channels.diffusion_sites.append(diffusion)
    outputs.finish_step(step=1, reactions=[], invalid_diffusion_sites=[])
    paths = list((tmp_path / "diagnostics" / "bare_neb").glob("*/*/*/diagnostic.json"))
    assert sorted(json.loads(path.read_text())["discovery_step"] for path in paths) == [0, 1]
    assert writer.n_unique_reactions == 0
    outputs.close()
    writer.close()


def test_initial_sweep_failure_flushes_both_bare_channels(tmp_path, monkeypatch):
    bond, _ = _bare_site("bond")
    diffusion, lc = _bare_site("diffusion")
    lc.stable = None
    lc.last_failure_reason = "NEB failed"
    writer = ReactionWriter(tmp_path)
    session = KMCSession(
        request=KMCRunRequest(
            system=KMCSystem(nx.Graph(), [], None, {}),
            settings=KMCSettings(temperature=500.0, n_steps=0, verbose=False),
            channels=KMCChannels(bond_sites=[bond], diffusion_sites=[diffusion]),
            observers=KMCObservers(reaction_writer=writer),
        ),
        functions=KMCFunctions(
            compute_adsorption=lambda *_args, **_kwargs: [],
            recompute_affected=lambda *_args, **_kwargs: ([], []),
            expand_bond_network=lambda *_args, **_kwargs: [],
        ),
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("initial sweep failed")

    monkeypatch.setattr("ogkmc.kmc.session.initialise_runtime", fail)
    with pytest.raises(RuntimeError, match="initial sweep failed"):
        session.run()
    writer.close()
    paths = list((tmp_path / "diagnostics" / "bare_neb").glob("*/*/*/diagnostic.json"))
    assert sorted(json.loads(path.read_text())["diagnostic_status"] for path in paths) == [
        "completed", "numerical_failure",
    ]


@pytest.mark.parametrize("kind", ["bond", "diffusion"])
def test_bare_diagnostics_restore_from_checkpoint_and_quarantine_crash_tail(tmp_path, kind):
    site, lc = _bare_site(kind)
    state = make_checkpoint_state(
        step=2, time_s=0.0, graph=nx.Graph(), adsorbate_sites=[],
        **{f"{kind}_sites": [site]},
    )
    checkpoint = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    writer = ReactionWriter(tmp_path)
    committed = writer.write_bare_neb(site, lc, kind=kind, step=2)
    site.iso_class += 1
    crash_tail = writer.write_bare_neb(site, lc, kind=kind, step=3)
    incomplete = crash_tail.with_name(crash_tail.name + "_incomplete")
    incomplete.mkdir()
    writer.close()
    resumed = ReactionWriter(tmp_path, append=True, checkpoint_step=2)
    assert committed.is_dir()
    assert not crash_tail.exists()
    assert not incomplete.exists()
    quarantined = discover_quarantine_locations(tmp_path)
    assert len(quarantined) == 2
    assert all("/diagnostics/bare_neb/" in path for path in quarantined)
    restored_site = getattr(load_checkpoint(checkpoint), f"{kind}_sites")[0]
    restored_lc = restored_site.lateral_classes[0]
    resumed.write_bare_neb(restored_site, restored_lc, kind=kind, step=4)
    resumed.close()
    assert json.loads((committed / "diagnostic.json").read_text())["discovery_step"] == 2
    assert len(read(committed / "neb_path.extxyz", index=":")) == 3
    assert not load_reaction_index(tmp_path / "reactions" / "index.jsonl")
