"""Shared reaction typing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ReactionLike(Protocol):
    """Structural protocol shared by adsorption, diffusion, and bond events."""

    kind: str
    site: object
    member_index: int
    lateral_class: object
    delta_e: float
    barrier: float
    rate: float


@dataclass(frozen=True)
class ReactionSnapshot:
    """Serializable summary of a reaction event or discovered lateral class."""

    kind: str
    species_label: str
    iso_class: int
    member_index: int
    lateral_class: int
    rate: float
    delta_e: float
    barrier: float
    delta_g: float | None = None
    barrier_g: float | None = None


__all__ = ["ReactionLike", "ReactionSnapshot"]
