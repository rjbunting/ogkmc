"""Command-line interface package."""

from __future__ import annotations

from autokmc2.cli.main import main
from autokmc2.cli.pipeline import run_from_config

__all__ = ["main", "run_from_config"]
