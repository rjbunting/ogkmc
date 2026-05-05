"""Smoke tests for the autokmc CLI parser and config validation."""

from __future__ import annotations

import textwrap

import pytest

from autokmc.cli.main import main as cli_main


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
