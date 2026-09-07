"""Old scientific results must not bypass corrected physical validation."""

from dataclasses import asdict
from types import SimpleNamespace

import networkx as nx
import pytest

import autokmc.io.resume_contract as resume_contract_module
from autokmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from autokmc.io.config import (
    BondCfg,
    CheckpointCfg,
    FreeEnergyCfg,
    OutputCfg,
    RunConfig,
)
from autokmc.io.resume_contract import make_resume_contract
from autokmc.workflow.runtime import resolve_run_identity


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


@pytest.mark.parametrize("unsafe_state", ["free_energy", "bond_config", "bond_snapshot"])
def test_legacy_scientific_checkpoint_is_rejected_before_event_reconciliation(
    tmp_path, unsafe_state,
):
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=unsafe_state == "free_energy"),
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


def test_modern_checkpoint_rejects_changed_validation_source(tmp_path, monkeypatch):
    # Exercise the real installed-source hashing routine without editing the
    # actual checkout or relying on a package version bump.
    package = tmp_path / "package"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("")
    validation = package / "free_energy.py"
    validation.write_text("VIBRATIONAL_VALIDATION_VERSION = 0\n")
    monkeypatch.setattr(resume_contract_module.autokmc_package, "__file__", str(init))
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=True),
    )
    events = _write_checkpoint(tmp_path, cfg, contract=make_resume_contract(cfg))
    original = events.read_bytes()
    validation.write_text("VIBRATIONAL_VALIDATION_VERSION = 1\n")
    with pytest.raises(ValueError, match=r"_software\.autokmc_source_sha256"):
        resolve_run_identity(cfg)
    assert events.read_bytes() == original


@pytest.mark.parametrize("old_tolerance", [None, 0.002])
def test_modern_checkpoint_rejects_missing_or_changed_imaginary_tolerance(
    tmp_path, old_tolerance,
):
    cfg = RunConfig(
        output=OutputCfg(dir=str(tmp_path / "run")),
        free_energy=FreeEnergyCfg(enabled=True, imaginary_mode_tolerance_ev=0.0015),
    )
    previous_config = asdict(cfg)
    if old_tolerance is None:
        del previous_config["free_energy"]["imaginary_mode_tolerance_ev"]
    else:
        previous_config["free_energy"]["imaginary_mode_tolerance_ev"] = old_tolerance
    # Build the old contract from a serialized mapping: restoring today's
    # dataclass defaults must not fill in missing settings in the old record.
    contract = make_resume_contract(previous_config)
    events = _write_checkpoint(tmp_path, cfg, contract=contract)
    original = events.read_bytes()
    with pytest.raises(ValueError, match=r"free_energy\.imaginary_mode_tolerance_ev"):
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

