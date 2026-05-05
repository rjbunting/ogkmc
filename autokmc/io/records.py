"""Serializable persistence record models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ReactionRecord:
    """One persisted KMC event row.

    Heavy data lives in the per-lateral-class folder referenced by
    ``reaction_dir``; the row itself stays lightweight for JSONL streaming.
    """
    schema_version:  str
    step:            int
    time_s:          float
    tau_s:           float
    kind:            str
    reactant_smiles: str
    iso_class:       int
    member_index:    int
    lateral_class:   int
    rate_hz:         float
    delta_e_ev:      float
    barrier_ev:      float
    description:     str
    reaction_dir:    str
    direction:       str | None = None
    delta_g_ev:      float | None = None
    barrier_g_ev:    float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


__all__ = ["ReactionRecord"]
