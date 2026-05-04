"""Shared type aliases for structure builders."""

from __future__ import annotations

from typing import Dict, Union

# Composition: single element string, or {symbol: fraction} dict summing to 1.
Composition = Union[str, Dict[str, float]]

# Lattice parameters: float (a only) or {"a": ..., "c": ...} for HCP.
LatticeParams = Union[float, Dict[str, float], None]

__all__ = ["Composition", "LatticeParams"]
