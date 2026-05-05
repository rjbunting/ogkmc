"""Tests for autokmc.io.config — load + dynamic calculator instantiation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from autokmc.io.config import (
    RunConfig,
    ConfigError,
    load_config,
)
from autokmc.io.calculators import (
    CalculatorCfg,
    build_calculator,
    calculator_meta,
)


def _write(tmp_path: Path, body: str, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


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


def test_load_yaml_ok(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK)
    cfg = load_config(p)
    assert isinstance(cfg, RunConfig)
    assert cfg.output.dir == "./out"
    assert cfg.structure.miller_index == (1, 1, 1)
    assert cfg.reactants[0].smiles == "[C-]#[O+]"
    assert cfg.calculator.import_path == "ase.calculators.emt.EMT"
    assert cfg.kmc.n_steps == 10
    assert cfg.diffusion.enabled is False


def test_load_toml_ok(tmp_path):
    body = """\
    schema_version = "1"

    [output]
    dir = "./out"

    [structure]
    kind = "surface"
    composition = "Cu"
    miller_index = [1, 1, 1]

    [[reactants]]
    smiles = "[C-]#[O+]"
    add_hydrogens = false

    [calculator]
    import_path = "ase.calculators.emt.EMT"

    [kmc]
    temperature_k = 500.0
    n_steps = 10
    """
    p = _write(tmp_path, body, name="cfg.toml")
    cfg = load_config(p)
    assert cfg.kmc.n_steps == 10


def test_unknown_key_raises(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK + "\nbogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_schema_version_mismatch(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, YAML_OK.replace('"1"', '"99"'))
    with pytest.raises(ConfigError):
        load_config(p)


def test_build_calculator_emt():
    cfg = CalculatorCfg(import_path="ase.calculators.emt.EMT", kwargs={})
    calc = build_calculator(cfg)
    assert calc is not None
    # Must look like an ASE calculator.
    assert hasattr(calc, "get_potential_energy")


def test_build_calculator_none():
    assert build_calculator(CalculatorCfg()) is None


def test_calculator_meta_roundtrip():
    cfg = CalculatorCfg(import_path="pkg.Foo", kwargs={"a": 1})
    meta = calculator_meta(cfg)
    assert meta["import_path"] == "pkg.Foo"
    assert meta["kwargs"] == {"a": 1}


def test_unknown_extension(tmp_path):
    p = tmp_path / "cfg.ini"
    p.write_text("[x]\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
