"""Configuration boundaries for imaginary-mode acceptance."""

from __future__ import annotations

import pytest

from autokmc.io.config import ConfigError, FreeEnergyCfg, load_config
from autokmc.thermo.free_energy import FreeEnergyOptions
from autokmc.workflow.models import RunIdentity
from autokmc.workflow.runtime import resolve_thermo_runtime


YAML_BASE = """\
reactants:
  - smiles: "[O]"
calculator:
  import_path: ase.calculators.emt.EMT
"""

TOML_BASE = """\
[[reactants]]
smiles = "[O]"
[calculator]
import_path = "ase.calculators.emt.EMT"
"""


@pytest.mark.parametrize("extension", ["yaml", "toml"])
@pytest.mark.parametrize("tolerance", [None, 0.0, 0.002])
def test_imaginary_tolerance_reaches_runtime(tmp_path, extension, tolerance):
    if extension == "yaml":
        text = YAML_BASE
        if tolerance is not None:
            text += f"free_energy:\n  imaginary_mode_tolerance_ev: {tolerance}\n"
    else:
        text = TOML_BASE
        if tolerance is not None:
            text += f"[free_energy]\nimaginary_mode_tolerance_ev = {tolerance}\n"
    path = tmp_path / f"config.{extension}"
    path.write_text(text)
    cfg = load_config(path)
    cfg.output.calculation_cache_enabled = False
    identity = RunIdentity(tmp_path, tmp_path / "manifest.json", "test")
    runtime = resolve_thermo_runtime(cfg, identity)

    expected = 0.0015 if tolerance is None else tolerance
    assert cfg.free_energy.imaginary_mode_tolerance_ev == pytest.approx(expected)
    assert runtime.options.imaginary_mode_tolerance_ev == pytest.approx(expected)
    assert runtime.options.min_frequency_ev == pytest.approx(0.0015)
    assert FreeEnergyCfg.imaginary_mode_tolerance_ev == pytest.approx(0.0015)
    assert FreeEnergyOptions.imaginary_mode_tolerance_ev == pytest.approx(0.0015)


@pytest.mark.parametrize(
    ("extension", "value"),
    [("yaml", value) for value in ("-0.001", ".nan", ".inf", "true", '"0.001"')]
    + [("toml", value) for value in ("-0.001", "nan", "inf", "true", '"0.001"')],
)
def test_rejects_invalid_imaginary_tolerance(tmp_path, extension, value):
    if extension == "yaml":
        text = YAML_BASE + f"free_energy:\n  imaginary_mode_tolerance_ev: {value}\n"
    else:
        text = TOML_BASE + f"[free_energy]\nimaginary_mode_tolerance_ev = {value}\n"
    path = tmp_path / f"config.{extension}"
    path.write_text(text)

    with pytest.raises(ConfigError, match="free_energy.imaginary_mode_tolerance_ev"):
        load_config(path)
