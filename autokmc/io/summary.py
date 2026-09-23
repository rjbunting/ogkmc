"""Run-summary persistence helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autokmc.io._files import write_json_atomic
from autokmc.io.reaction_index import (
	reaction_id_from_event,
	stable_reaction_id,
)
from autokmc.io.reaction_layout import (
	reaction_relative_dir as _reaction_relative_dir,
	reaction_smiles as _reaction_smiles,
)
from autokmc.io.schemas import SUMMARY_ARTIFACT_TYPE, SUMMARY_SCHEMA_VERSION
from autokmc.species.smiles import (  # noqa: F401 - compatibility re-export
	smiles_to_dirname as _smiles_to_dirname,
)


@dataclass(slots=True)
class _RunningStats:
	"""Constant-space population statistics using Welford's recurrence."""

	count: int = 0
	mean: float = 0.0
	m2: float = 0.0
	minimum: float = math.inf
	maximum: float = -math.inf

	def add(self, value: float) -> None:
		value = float(value)
		if not math.isfinite(value):
			raise ValueError("summary statistics contain non-finite values")
		self.count += 1
		delta = value - self.mean
		self.mean += delta / self.count
		self.m2 += delta * (value - self.mean)
		self.minimum = min(self.minimum, value)
		self.maximum = max(self.maximum, value)

	def to_dict(self) -> dict[str, float | None]:
		if self.count == 0:
			return {"mean": None, "std": None, "min": None, "max": None}
		# Round-off can make m2 a tiny negative number for nearly-identical
		# values; the mathematical population variance is non-negative.
		variance = max(0.0, self.m2 / self.count)
		return {
			"mean": float(self.mean),
			"std": float(math.sqrt(variance)),
			"min": float(self.minimum),
			"max": float(self.maximum),
		}


def _stats(values: list[float]) -> dict[str, float | None]:
	"""Compatibility helper returning population statistics for *values*."""
	stats = _RunningStats()
	for value in values:
		stats.add(value)
	return stats.to_dict()


_METRIC_NAMES = (
	"rate",
	"rate_delta",
	"rate_barrier",
	"electronic_delta",
	"electronic_barrier",
	"free_delta",
	"free_barrier",
)


def _new_bucket() -> dict[str, _RunningStats]:
	return {name: _RunningStats() for name in _METRIC_NAMES}


def _event_direction(kind: str, direction: Any) -> str | None:
	if direction not in (None, ""):
		return str(direction)
	if kind in {"adsorption", "desorption"}:
		return kind
	return None


def _energy_basis(event: dict[str, Any]) -> str:
	explicit = event.get("rate_energy_basis")
	if explicit:
		return str(explicit)
	rate_delta = event.get("rate_delta_ev", event.get("delta_e_ev"))
	rate_barrier = event.get("rate_barrier_ev", event.get("barrier_ev"))
	if (
		event.get("delta_g_ev") is not None
		and event.get("barrier_g_ev") is not None
		and rate_delta == event.get("delta_g_ev")
		and rate_barrier == event.get("barrier_g_ev")
	):
		return "free_energy"
	if event.get("delta_e_ev") is not None and event.get("barrier_ev") is not None:
		return "electronic"
	return "unknown"


class ReactionSummary:
	"""Direction- and energy-basis-aware run summary aggregator."""

	__slots__ = (
		"_buckets",
		"_total_by_kind",
		"_total_by_direction",
		"_n",
		"_first_step",
		"_last_step",
		"_run_id",
		"_discovered_valid_ids",
		"_discovered_invalid_ids",
		"_fired_reaction_ids",
	)

	def __init__(self, *, run_id: str | None = None):
		self._buckets: dict[tuple, dict[str, _RunningStats]] = {}
		self._total_by_kind: dict[str, int] = {}
		self._total_by_direction: dict[str, int] = {}
		self._n: int = 0
		self._first_step: dict[tuple, int] = {}
		self._last_step: dict[tuple, int] = {}
		self._run_id = None if run_id is None else str(run_id)
		self._discovered_valid_ids: set[str] = set()
		self._discovered_invalid_ids: set[str] = set()
		self._fired_reaction_ids: set[str] = set()

	@classmethod
	def from_events(cls, path: str | Path) -> "ReactionSummary":
		"""Reconstruct cumulative summary state from a v2 or v3 event log."""
		out = cls()
		path = Path(path)
		if not path.is_file():
			return out
		with path.open("r", encoding="utf-8") as handle:
			for line_number, line in enumerate(handle, start=1):
				if not line.strip():
					continue
				try:
					event = json.loads(line)
					out.add_event(event)
				except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
					raise ValueError(
						f"cannot resume summary from invalid event line {line_number}: {exc}"
					) from exc
		return out

	def set_run_id(self, run_id: str | None) -> None:
		if run_id is None:
			return
		resolved = str(run_id)
		if self._run_id is not None and self._run_id != resolved:
			raise ValueError(
				f"summary run_id mismatch: {resolved!r} != {self._run_id!r}"
			)
		self._run_id = resolved

	def note_discovered(self, reaction_id: str, *, valid: bool = True) -> None:
		reaction_id = str(reaction_id)
		if valid:
			self._discovered_valid_ids.add(reaction_id)
			self._discovered_invalid_ids.discard(reaction_id)
		else:
			self._discovered_invalid_ids.add(reaction_id)
			self._discovered_valid_ids.discard(reaction_id)

	def discover(self, reaction, *, valid: bool = True) -> str:
		smiles = _reaction_smiles(reaction)
		reaction_id = stable_reaction_id(
			str(reaction.kind),
			str(smiles),
			int(reaction.site.iso_class),
			int(reaction.lateral_class.lateral_class),
		)
		self.note_discovered(reaction_id, valid=valid)
		return reaction_id

	def discover_invalid_diffusion(self, site, lateral_class) -> str:
		reaction_id = stable_reaction_id(
			"diffusion",
			str(getattr(site, "reactant", "")),
			int(site.iso_class),
			int(lateral_class.lateral_class),
		)
		self.note_discovered(reaction_id, valid=False)
		return reaction_id

	def discover_invalid_bond(self, site, lateral_class) -> str:
		template = site.template
		smiles = (
			f"{template.smiles_a}+{template.smiles_b}"
			f"↔{template.smiles_c}"
		)
		reaction_id = stable_reaction_id(
			"bond",
			smiles,
			int(site.iso_class),
			int(lateral_class.lateral_class),
		)
		self.note_discovered(reaction_id, valid=False)
		return reaction_id

	def _add_values(
		self,
		key: tuple,
		*,
		values: dict[str, Any],
		step: int,
		reaction_id: str,
	) -> None:
		required = ("rate", "rate_delta", "rate_barrier")
		if any(
			values[name] is None or not math.isfinite(float(values[name]))
			for name in required
		):
			raise ValueError("event contains non-finite rate energetics")
		bucket = self._buckets.setdefault(key, _new_bucket())
		for name, value in values.items():
			if value is None:
				continue
			numeric = float(value)
			if not math.isfinite(numeric):
				raise ValueError(f"event contains non-finite {name}")
			bucket[name].add(numeric)
		kind, direction = str(key[0]), key[1]
		self._total_by_kind[kind] = self._total_by_kind.get(kind, 0) + 1
		if direction is not None:
			self._total_by_direction[str(direction)] = (
				self._total_by_direction.get(str(direction), 0) + 1
			)
		self._n += 1
		self._first_step.setdefault(key, int(step))
		self._last_step[key] = int(step)
		self._fired_reaction_ids.add(str(reaction_id))
		self.note_discovered(str(reaction_id), valid=True)

	def add_event(self, event: dict[str, Any]) -> None:
		"""Add one serialized event while retaining every available energy basis."""
		kind = str(event["kind"])
		direction = _event_direction(kind, event.get("direction"))
		smiles = str(event.get("reactant_smiles", ""))
		iso = int(event["iso_class"])
		lat = int(event["lateral_class"])
		basis = _energy_basis(event)
		key = (kind, direction, smiles, iso, lat, basis)
		reaction_id = reaction_id_from_event(event)
		self.set_run_id(event.get("run_id"))
		self._add_values(
			key,
			values={
				"rate": event["rate_hz"],
				"rate_delta": event.get("rate_delta_ev", event.get("delta_e_ev")),
				"rate_barrier": event.get(
					"rate_barrier_ev",
					event.get("barrier_ev"),
				),
				"electronic_delta": event.get("delta_e_ev"),
				"electronic_barrier": event.get("barrier_ev"),
				"free_delta": event.get("delta_g_ev"),
				"free_barrier": event.get("barrier_g_ev"),
			},
			step=int(event["step"]),
			reaction_id=reaction_id,
		)

	def add(self, reaction, *, step: int) -> None:
		"""Compatibility path for runs without a serialized event observer."""
		smiles = _reaction_smiles(reaction)
		kind = str(reaction.kind)
		direction = _event_direction(kind, getattr(reaction, "direction", None))
		basis = str(getattr(reaction, "rate_energy_basis", "unknown") or "unknown")
		iso = int(reaction.site.iso_class)
		lat = int(reaction.lateral_class.lateral_class)
		reaction_id = stable_reaction_id(kind, smiles, iso, lat)
		rate_delta = float(reaction.delta_e)
		rate_barrier = float(reaction.barrier)
		electronic_delta = getattr(reaction, "delta_e_ev", None)
		electronic_barrier = getattr(reaction, "barrier_ev", None)
		free_delta = getattr(reaction, "delta_g_ev", None)
		free_barrier = getattr(reaction, "barrier_g_ev", None)
		if basis == "electronic":
			electronic_delta = rate_delta if electronic_delta is None else electronic_delta
			electronic_barrier = (
				rate_barrier if electronic_barrier is None else electronic_barrier
			)
		elif basis == "free_energy":
			free_delta = rate_delta if free_delta is None else free_delta
			free_barrier = rate_barrier if free_barrier is None else free_barrier
		self._add_values(
			(kind, direction, str(smiles), iso, lat, basis),
			values={
				"rate": float(reaction.rate),
				"rate_delta": rate_delta,
				"rate_barrier": rate_barrier,
				"electronic_delta": electronic_delta,
				"electronic_barrier": electronic_barrier,
				"free_delta": free_delta,
				"free_barrier": free_barrier,
			},
			step=int(step),
			reaction_id=reaction_id,
		)

	def to_dict(
		self,
		*,
		run_meta: dict[str, Any] | None = None,
		final_occupancy: dict[str, int] | None = None,
	) -> dict[str, Any]:
		by_type: list[dict[str, Any]] = []
		for key, bucket in self._buckets.items():
			kind, direction, smiles, iso, lat, basis = key
			by_type.append({
				"reaction_id": stable_reaction_id(kind, smiles, iso, lat),
				"kind": kind,
				"direction": direction,
				"reactant_smiles": smiles,
				"iso_class": iso,
				"lateral_class": lat,
				"reaction_dir": _reaction_relative_dir(kind, iso, lat, smiles=smiles),
				"rate_energy_basis": basis,
				"count": bucket["rate"].count,
				"first_step": self._first_step[key],
				"last_step": self._last_step[key],
				"rate_hz": bucket["rate"].to_dict(),
				"rate_delta_ev": bucket["rate_delta"].to_dict(),
				"rate_barrier_ev": bucket["rate_barrier"].to_dict(),
				"electronic_energy": {
					"delta_e_ev": bucket["electronic_delta"].to_dict(),
					"barrier_ev": bucket["electronic_barrier"].to_dict(),
				},
				"free_energy": {
					"delta_g_ev": bucket["free_delta"].to_dict(),
					"barrier_g_ev": bucket["free_barrier"].to_dict(),
				},
			})
		by_type.sort(
			key=lambda item: (
				-item["count"],
				item["kind"],
				str(item["direction"]),
				item["iso_class"],
				item["lateral_class"],
				item["rate_energy_basis"],
			)
		)
		discovered_ids = self._discovered_valid_ids | self._discovered_invalid_ids
		return {
			"artifact_type": SUMMARY_ARTIFACT_TYPE,
			"schema_version": SUMMARY_SCHEMA_VERSION,
			"run_id": self._run_id,
			"run": dict(run_meta or {}),
			"totals": {
				# Compatibility alias retained for existing consumers.
				"reactions": self._n,
				"fired_events": self._n,
				"by_kind": dict(sorted(self._total_by_kind.items())),
				"by_direction": dict(sorted(self._total_by_direction.items())),
				"unique_reaction_types": len(self._buckets),
				"fired_reaction_classes": len(self._fired_reaction_ids),
				"discovered_reactions": len(discovered_ids),
				"discovered_valid_reactions": len(self._discovered_valid_ids),
				"discovered_invalid_reactions": len(self._discovered_invalid_ids),
			},
			"by_reaction_type": by_type,
			"final_occupancy": {
				str(key): int(value)
				for key, value in (final_occupancy or {}).items()
			},
		}

	def write(
		self,
		path: str | Path,
		*,
		run_meta: dict[str, Any] | None = None,
		final_occupancy: dict[str, int] | None = None,
	) -> Path:
		path = Path(path)
		payload = self.to_dict(run_meta=run_meta, final_occupancy=final_occupancy)
		return write_json_atomic(path, payload)

	@property
	def n(self) -> int:
		return self._n


def make_run_meta(
	*,
	config_path: str | None = None,
	temperature_k: float | None = None,
	n_steps_requested: int | None = None,
	steps_executed: int | None = None,
	total_time_s: float | None = None,
	random_seed: int | None = None,
	started_at: datetime | None = None,
	finished_at: datetime | None = None,
	extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
	def _iso(dt: datetime | None) -> str | None:
		return None if dt is None else dt.astimezone(timezone.utc).isoformat()

	out: dict[str, Any] = {
		"config_path": config_path,
		"started_at": _iso(started_at),
		"finished_at": _iso(finished_at),
		"temperature_k": temperature_k,
		"n_steps_requested": n_steps_requested,
		"steps_executed": steps_executed,
		"total_time_s": total_time_s,
		"random_seed": random_seed,
	}
	if extra:
		out.update(extra)
	return out


__all__ = ["ReactionSummary", "_stats", "make_run_meta"]
