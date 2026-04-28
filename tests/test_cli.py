"""Smoke test for the autokmc CLI argument parser & validate-config path.

A full pipeline `autokmc run …` is not exercised here because the slab
build + ML stability checks are too heavy for a unit-test budget; the
end-to-end run is covered by the dev scripts under ``autokmc/dev/``.
"""

from __future__ import annotations

import textwrap

import pytest

from autokmc.cli import main as cli_main


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


def test_cli_no_args_errors():
    with pytest.raises(SystemExit):
        cli_main([])

