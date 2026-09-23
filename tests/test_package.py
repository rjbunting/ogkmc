"""Public entry points and resources of the renamed package."""

from __future__ import annotations

import json
from importlib.resources import files
import subprocess
import sys

import pytest

from ogkmc import __version__
from ogkmc.io.calculators import _resolve
from ogkmc.io.fairchem import get_predict_unit_on_device


@pytest.mark.parametrize("module", ["ogkmc", "ogkmc.cli"])
def test_module_entrypoints_identify_ogkmc(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == f"ogkmc {__version__}"
    assert result.stderr == ""

    help_result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "Online Graph Kinetic Monte Carlo" in help_result.stdout
    assert "usage: ogkmc" in help_result.stdout
    assert help_result.stderr == ""


def test_renamed_package_includes_runtime_resources():
    root = files("ogkmc")
    assert root.joinpath("py.typed").is_file()
    schema = json.loads(root.joinpath("schema/isaac_record_v1.json").read_text())
    assert schema["type"] == "object"


def test_configured_factory_resolves_in_new_namespace():
    assert _resolve("ogkmc.io.fairchem.get_predict_unit_on_device") is get_predict_unit_on_device
