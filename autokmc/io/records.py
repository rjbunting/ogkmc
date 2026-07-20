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
    delta_e_ev:      float | None
    barrier_ev:      float | None
    description:     str
    reaction_dir:    str
    inputs:          list[dict[str, Any]]
    outputs:         list[dict[str, Any]]
    template:        dict[str, Any] | None
    gas_product:     bool
    run_id:          str | None = None
    direction:       str | None = None
    delta_g_ev:      float | None = None
    barrier_g_ev:    float | None = None
    rate_energy_basis: str | None = None
    rate_delta_ev:     float | None = None
    rate_barrier_ev:   float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


__all__ = ["ReactionRecord"]
