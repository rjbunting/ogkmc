"""Old scientific results must not bypass corrected physical validation."""

from types import SimpleNamespace

import networkx as nx
import pytest

import ogkmc.io.resume_contract as resume_contract_module
from ogkmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from ogkmc.io.config import (
    BondCfg,
    CheckpointCfg,
    FreeEnergyCfg,
    OutputCfg,
    RunConfig,
)
from ogkmc.io.resume_contract import make_resume_contract
from ogkmc.workflow.runtime import resolve_run_identity


def _write_checkpoint(tmp_path, cfg, *, contract=None, bond_sites=None):
    metadata = {"run_id": "scientific-resume-validation"}
    if contract is not None:
        metadata["resume_contract"] = contract
    state = make_checkpoint_state(
        step=0,
        time_s=0.0,
        graph=nx.Graph(),
        adsorbate_sites=[],
        bond_sites=bond_sites,
        committed_event_count=0,
        committed_event_offset=0,
        metadata=metadata,
    )
    checkpoint = save_checkpoint(tmp_path / "checkpoint.pkl", state)
    cfg.checkpoint = CheckpointCfg(resume_from=str(checkpoint))
    output = tmp_path / "run"
    output.mkdir(exist_ok=True)
    events = output / "events.jsonl"
    events.write_text('{"step": 1, "run_id": "scientific-resume-validation"}\n')
    return events


@pytest.mark.parametrize("unsafe_state", ["bond_config", "bond_snapshot"])
def test_legacy_scientific_checkpoint_is_rejected_before_event_reconciliation(
    tmp_path, unsafe_state,
):
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=False),
        bond=BondCfg(enabled=unsafe_state == "bond_config"),
    )
    # The snapshot check applies even if the resumed config disables bonds:
    # network preparation would still restore these persisted templates.
    bond_sites = (
        [SimpleNamespace(lateral_classes=[])]
        if unsafe_state == "bond_snapshot" else []
    )
    events = _write_checkpoint(tmp_path, cfg, bond_sites=bond_sites)
    original = events.read_bytes()
    with pytest.raises(
        ValueError,
        match=r"no scientific resume fingerprint.*Start a fresh run.*new output directory",
    ):
        resolve_run_identity(cfg)
    assert events.read_bytes() == original


def test_modern_checkpoint_rejects_changed_thermochemistry_source(tmp_path, monkeypatch):
    # Exercise the real installed-source hashing routine without editing the
    # actual checkout or relying on a package version bump.
    package = tmp_path / "package"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("")
    thermo_source = package / "free_energy.py"
    thermo_source.write_text("SURFACE_VIBRATION_SUBSYSTEM = 'reactive_atoms_v0'\n")
    monkeypatch.setattr(resume_contract_module.ogkmc_package, "__file__", str(init))
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=True),
    )
    events = _write_checkpoint(tmp_path, cfg, contract=make_resume_contract(cfg))
    original = events.read_bytes()
    thermo_source.write_text("SURFACE_VIBRATION_SUBSYSTEM = 'all_adsorbates_v1'\n")
    with pytest.raises(ValueError, match=r"_software\.ogkmc_source_sha256"):
        resolve_run_identity(cfg)
    assert events.read_bytes() == original


def test_matching_modern_free_energy_and_bond_contract_can_resume(tmp_path):
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=True),
        bond=BondCfg(enabled=True),
    )
    _write_checkpoint(tmp_path, cfg, contract=make_resume_contract(cfg))
    identity = resolve_run_identity(cfg)
    assert identity.resume_state is not None
    assert identity.run_id == "scientific-resume-validation"



def test_legacy_free_energy_checkpoint_can_resume_without_bond_templates(tmp_path):
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / 'run')),
        free_energy=FreeEnergyCfg(enabled=True),
        bond=BondCfg(enabled=False),
    )
    _write_checkpoint(tmp_path, cfg, bond_sites=[])
    identity = resolve_run_identity(cfg)
    assert identity.resume_state is not None
    assert identity.run_id == 'scientific-resume-validation'
