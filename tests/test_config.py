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
    CalculatorPool,
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
    assert isinstance(calc, CalculatorPool)
    with calc.acquire() as concrete:
        # Must look like an ASE calculator.
        assert hasattr(concrete, "get_potential_energy")


def test_build_calculator_none():
    assert build_calculator(CalculatorCfg()) is None


def test_calculator_meta_roundtrip():
    cfg = CalculatorCfg(
        import_path="pkg.Foo",
        kwargs={"a": 1},
        copies=2,
        gpu_devices=["cuda:0", "cuda:1"],
    )
    meta = calculator_meta(cfg)
    assert meta["import_path"] == "pkg.Foo"
    assert meta["kwargs"] == {"a": 1}
    assert meta["copies"] == 2
    assert meta["gpu_devices"] == ["cuda:0", "cuda:1"]


def test_load_new_checkpoint_and_parallel_fields(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, """
schema_version: "1"
output:
  dir: ./out
  calculation_cache_enabled: true
  calculation_cache_dir: calc_cache
  isaac_export_filename: isaac_upload.json
reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
calculator:
  import_path: ase.calculators.emt.EMT
  copies: 2
  gpu_devices: ["cuda:0", "cuda:1"]
  gpu_device_arg: device
  max_workers: 2
checkpoint:
  enabled: true
  path: ./out/checkpoint.pkl
  every_n_steps: 5
structure:
  kind: nanoparticle
  composition: Cu
  n_atoms: 55
  surface_energy_facets: [[1, 1, 1], [1, 0, 0]]
  surface_energy_layers: 4
""")
    cfg = load_config(p)
    assert cfg.calculator.copies == 2
    assert cfg.calculator.gpu_devices == ["cuda:0", "cuda:1"]
    assert cfg.checkpoint.enabled is True
    assert cfg.checkpoint.every_n_steps == 5
    assert cfg.structure.surface_energy_facets == ((1, 1, 1), (1, 0, 0))
    assert cfg.output.calculation_cache_enabled is True
    assert cfg.output.calculation_cache_dir == "calc_cache"
    assert cfg.output.isaac_export_filename == "isaac_upload.json"
    assert cfg.output.run_manifest_filename == "run_manifest.json"


def test_load_bond_matching_fields(tmp_path):
    pytest.importorskip("yaml")
    p = _write(tmp_path, """
schema_version: "1"
reactants:
  - smiles: "[OH]"
    add_hydrogens: false
bond:
  enabled: true
  neb_interpolation: idpp
  atom_matching: hungarian
  matching_trials: 12
  gas_lift_height: 4.5
""")
    cfg = load_config(p)
    assert cfg.bond.enabled is True
    assert cfg.bond.neb_interpolation == "idpp"
    assert cfg.bond.atom_matching == "hungarian"
    assert cfg.bond.matching_trials == 12
    assert cfg.bond.gas_lift_height == 4.5


def test_unknown_extension(tmp_path):
    p = tmp_path / "cfg.ini"
    p.write_text("[x]\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


@pytest.mark.parametrize(
    "fragment",
    [
        "diffusion:\n  enabled: 'false'\n",
        "bond:\n  neb_climb: 'true'\n",
        "kmc:\n  temperature_k: 0\n",
        "free_energy:\n  vibration_nfree: 3\n",
    ],
)
def test_strict_validation_rejects_coercible_types_and_invalid_ranges(tmp_path, fragment):
    pytest.importorskip("yaml")
    path = _write(
        tmp_path,
        "schema_version: '1'\nreactants:\n  - smiles: '[O]'\n" + fragment,
    )
    with pytest.raises(ConfigError):
        load_config(path)
