"""Command-line interface package."""

from __future__ import annotations

from autokmc.cli.main import main
from autokmc.cli.pipeline import run_from_config

__all__ = ["main", "run_from_config"]
