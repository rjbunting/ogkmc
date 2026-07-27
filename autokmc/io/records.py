"""Serializable persistence record models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from autokmc.io.schemas import EVENT_ARTIFACT_TYPE


_STATIC_EVENT_FIELDS = frozenset(
    {"description", "reaction_dir", "template", "gas_product"}
)


@dataclass
class ReactionRecord:
    """One persisted KMC event row.

    Heavy data lives in the per-lateral-class folder referenced by
    ``reaction_dir``; the row itself stays lightweight for JSONL streaming.
    """
    schema_version: str
    event_id: str
    reaction_id: str
    step: int
    time_s: float
    tau_s: float
    kind: str
    reactant_smiles: str
    iso_class: int
    member_index: int
    lateral_class: int
    rate_hz: float
    delta_e_ev: float | None
    barrier_ev: float | None
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    artifact_type: str = EVENT_ARTIFACT_TYPE
    run_id: str | None = None
    direction: str | None = None
    delta_g_ev: float | None = None
    barrier_g_ev: float | None = None
    rate_energy_basis: str | None = None
    rate_delta_ev: float | None = None
    rate_barrier_ev: float | None = None
    # Compatibility representation.  New JSONL rows omit these static fields
    # and resolve them through ``reactions/index.jsonl`` by ``reaction_id``.
    description: str | None = None
    reaction_dir: str | None = None
    template: dict[str, Any] | None = None
    gas_product: bool | None = None

    def to_jsonable(self, *, include_static: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_static:
            for key in _STATIC_EVENT_FIELDS:
                d.pop(key, None)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


__all__ = ["ReactionRecord"]
