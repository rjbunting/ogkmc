"""Command-line interface package."""

from __future__ import annotations

from typing import Any

from ogkmc.cli.main import main


def run_from_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the scientific pipeline only when a configured run is requested."""
    from ogkmc.cli.pipeline import run_from_config as implementation

    return implementation(*args, **kwargs)

__all__ = ["main", "run_from_config"]
