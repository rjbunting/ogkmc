"""Smoke tests for the autokmc CLI parser and config validation."""

from __future__ import annotations

import textwrap

from ase import Atoms
import networkx as nx
import pytest

from autokmc.cli.main import main as cli_main
from autokmc.cli.pipeline import (
    _derive_configured_bond_templates,
    _resolved_partial_pressure_bar,
    run_from_config,
)
from autokmc.io.checkpoint import make_checkpoint_state, save_checkpoint
from autokmc.io.config import (
    BondCfg,
    CheckpointCfg,
    FreeEnergyCfg,
    KMCCfg,
    OutputCfg,
    ReactantCfg,
    RunConfig,
)
from autokmc.species.reactant import Reactant


YAML_OK = """\
schema_version: "1"
output:
  dir: ./out
structure:
  kind: surface
  composition: Cu
  miller_index: [1, 1, 1]
reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
kmc:
  temperature_k: 500.0
  n_steps: 10
"""


def test_validate_config_ok(tmp_path, capsys):
    pytest.importorskip("yaml")
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(YAML_OK))
    rc = cli_main(["validate-config", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    assert "autokmc" in capsys.readouterr().out


def test_cli_package_exports_public_entrypoints():
    import autokmc.cli as cli

    assert cli.main is cli_main
    assert callable(cli.run_from_config)


def test_cli_no_args_errors():
    with pytest.raises(SystemExit):
        cli_main([])


def test_bond_template_derivation_preserves_per_reactant_hydrogen_policy(monkeypatch):
    from autokmc.reactions import templates as template_module

    calls = []

    def fake_derive(smiles, **kwargs):
        calls.append((list(smiles), kwargs))
        return []

    monkeypatch.setattr(template_module, "derive_bond_templates", fake_derive)
    configs = [
        ReactantCfg(smiles="C", add_hydrogens=False),
        ReactantCfg(smiles="O", add_hydrogens=True),
    ]
    reactants = [
        Reactant("C", Atoms("C"), nx.Graph()),
        Reactant("O", Atoms("O"), nx.Graph()),
    ]

    _derive_configured_bond_templates(configs, reactants, BondCfg(enabled=True))

    assert calls[0][0] == ["C"]
    assert calls[0][1]["add_hydrogens"] is False
    assert calls[1][0] == ["O"]
    assert calls[1][1]["add_hydrogens"] is True
    assert calls[2][0] == ["C", "O"]
    assert calls[2][1]["include_dissociation"] is False


def test_reactant_partial_pressure_uses_feed_default_unless_overridden():
    free_energy = FreeEnergyCfg(pressure_bar=0.4)

    inherited = ReactantCfg(smiles="[C]=O")
    overridden = ReactantCfg(smiles="O=O", partial_pressure_bar=0.2)

    assert _resolved_partial_pressure_bar(inherited, free_energy) == pytest.approx(0.4)
    assert _resolved_partial_pressure_bar(overridden, free_energy) == pytest.approx(0.2)


def test_resume_skips_fresh_structure_and_site_enumeration(tmp_path, monkeypatch):
    G = nx.Graph()
    reactant = Reactant("C", Atoms("C"), nx.Graph(), energy=0.0)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pkl",
        make_checkpoint_state(
            step=4,
            time_s=2.5,
            graph=G,
            adsorbate_sites=[],
            diffusion_sites=[],
            bond_sites=[],
            reactants=[reactant],
            history=[],
            reaction_counts={},
            metadata={"run_id": "resume-test"},
        ),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("fresh-run setup was called while resuming")

    import autokmc.structure as structure_module
    import autokmc.core.graph as graph_module
    import autokmc.species.reactant as reactant_module
    import autokmc.sites.adsorbate as adsorbate_module
    import autokmc.sites.diffusion as diffusion_module
    import autokmc.sites.bond as bond_module
    import autokmc.kmc.engine as engine_module

    for module, name in (
        (structure_module, "build_surface"),
        (structure_module, "build_nanoparticle"),
        (structure_module, "find_surface_atoms"),
        (graph_module, "build_graph"),
        (reactant_module, "build_reactant"),
        (adsorbate_module, "find_adsorbate_sites"),
        (diffusion_module, "find_diffusion_sites"),
        (bond_module, "find_bond_sites"),
    ):
        monkeypatch.setattr(module, name, unexpected)

    def fake_run(_graph, _sites, _calculator, **kwargs):
        assert _graph.graph["run_id"] == "resume-test"
        assert kwargs["initial_step"] == 4
        assert kwargs["initial_time_s"] == 2.5
        return {
            "time": 2.5,
            "steps_executed": 0,
            "history": [],
            "reaction_counts": {},
            "final_occupancy": {},
        }

    monkeypatch.setattr(engine_module, "run_kmc_steps", fake_run)
    cfg = RunConfig(
        output=OutputCfg(
            dir=str(tmp_path / "out"),
            trajectory_dump_every=0,
            calculation_cache_enabled=False,
        ),
        reactants=[ReactantCfg(smiles="C", add_hydrogens=False)],
        kmc=KMCCfg(n_steps=0),
        free_energy=FreeEnergyCfg(enabled=False),
        checkpoint=CheckpointCfg(resume_from=str(checkpoint)),
    )

    summary = run_from_config(cfg)

    assert summary["steps_executed"] == 0
