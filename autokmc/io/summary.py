"""Run-summary persistence helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from autokmc.core.constants import (
	BOND_FOLDER_FMT,
	DIFFUSION_FOLDER_FMT,
	PERSISTENCE_SCHEMA_VERSION,
	REACTIONS_DIR,
)
from autokmc.species.smiles import smiles_to_dirname as _smiles_to_dirname


def _reaction_smiles(reaction) -> str:
	site = reaction.site
	smiles = getattr(site, "reactant", None)
	if smiles:
		return str(smiles)
	tpl = getattr(site, "template", None)
	if tpl is not None:
		return f"{tpl.smiles_a}+{tpl.smiles_b}↔{tpl.smiles_c}"
	return ""


def _kind_subdir(kind: str) -> str:
	return {"adsorption": "adsorption", "desorption": "adsorption"}.get(str(kind), str(kind))


def _kind_folder_name(sub: str, iso: int, lat: int) -> str:
	if sub == "diffusion":
		return DIFFUSION_FOLDER_FMT.format(iso=int(iso), lat=int(lat))
	if sub == "bond":
		return BOND_FOLDER_FMT.format(iso=int(iso), lat=int(lat))
	return f"iso{int(iso)}_lat{int(lat)}"


def _reaction_relative_dir(kind: str, iso: int, lat: int, smiles: str = "") -> str:
	sub = _kind_subdir(kind)
	species = _smiles_to_dirname(smiles) if smiles else "unknown"
	return f"{REACTIONS_DIR}/{sub}/{species}/{_kind_folder_name(sub, iso, lat)}"


def _stats(values: list[float]) -> dict[str, float | None]:
	if not values:
		return {"mean": None, "std": None, "min": None, "max": None}
	arr = np.asarray(values, dtype=float)
	if not np.all(np.isfinite(arr)):
		raise ValueError("summary statistics contain non-finite values")
	return {
		"mean": float(arr.mean()),
		"std": float(arr.std(ddof=0)),
		"min": float(arr.min()),
		"max": float(arr.max()),
	}


class ReactionSummary:
	"""Aggregator for per-reaction-type statistics."""

	__slots__ = (
		"_buckets", "_total_by_kind", "_n", "_first_step", "_last_step",
	)

	def __init__(self):
		self._buckets: dict[tuple, dict[str, list[float]]] = {}
		self._total_by_kind: dict[str, int] = {}
		self._n: int = 0
		self._first_step: dict[tuple, int] = {}
		self._last_step: dict[tuple, int] = {}

	@classmethod
	def from_events(cls, path: str | Path) -> "ReactionSummary":
		"""Reconstruct cumulative summary state from an append-only event log."""
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

	def add_event(self, event: dict[str, Any]) -> None:
		"""Add one serialized event, preserving the rate's actual energy basis."""
		key = (
			str(event["kind"]),
			str(event.get("reactant_smiles", "")),
			int(event["iso_class"]),
			int(event["lateral_class"]),
		)
		values = {
			"rate": event["rate_hz"],
			"delta_e": event.get("rate_delta_ev", event.get("delta_e_ev")),
			"barrier": event.get("rate_barrier_ev", event.get("barrier_ev")),
		}
		if any(value is None or not math.isfinite(float(value)) for value in values.values()):
			raise ValueError("event contains non-finite rate energetics")
		bucket = self._buckets.setdefault(key, {"rate": [], "delta_e": [], "barrier": []})
		for name, value in values.items():
			bucket[name].append(float(value))
		step = int(event["step"])
		self._total_by_kind[key[0]] = self._total_by_kind.get(key[0], 0) + 1
		self._n += 1
		self._first_step.setdefault(key, step)
		self._last_step[key] = step

	def add(self, reaction, *, step: int) -> None:
		smiles = _reaction_smiles(reaction)
		key = (
			str(reaction.kind),
			str(smiles),
			int(reaction.site.iso_class),
			int(reaction.lateral_class.lateral_class),
		)
		values = {
			"rate": float(reaction.rate),
			"delta_e": float(reaction.delta_e),
			"barrier": float(reaction.barrier),
		}
		if not all(math.isfinite(value) for value in values.values()):
			raise ValueError("reaction contains non-finite rate energetics")
		b = self._buckets.setdefault(key, {"rate": [], "delta_e": [], "barrier": []})
		for name, value in values.items():
			b[name].append(value)

		self._total_by_kind[key[0]] = self._total_by_kind.get(key[0], 0) + 1
		self._n += 1
		self._first_step.setdefault(key, int(step))
		self._last_step[key] = int(step)

	def to_dict(
		self,
		*,
		run_meta: dict[str, Any] | None = None,
		final_occupancy: dict[str, int] | None = None,
	) -> dict[str, Any]:
		by_type: list[dict[str, Any]] = []
		for key, b in self._buckets.items():
			kind, smiles, iso, lat = key
			by_type.append({
				"kind": kind,
				"reactant_smiles": smiles,
				"iso_class": iso,
				"lateral_class": lat,
				"reaction_dir": _reaction_relative_dir(kind, iso, lat, smiles=smiles),
				"count": len(b["rate"]),
				"first_step": self._first_step[key],
				"last_step": self._last_step[key],
				"rate_hz": _stats(b["rate"]),
				"delta_e_ev": _stats(b["delta_e"]),
				"barrier_ev": _stats(b["barrier"]),
			})
		by_type.sort(key=lambda d: (-d["count"], d["kind"], d["iso_class"], d["lateral_class"]))

		return {
			"schema_version": PERSISTENCE_SCHEMA_VERSION,
			"run": dict(run_meta or {}),
			"totals": {
				"reactions": self._n,
				"by_kind": dict(self._total_by_kind),
				"unique_reaction_types": len(self._buckets),
			},
			"by_reaction_type": by_type,
			"final_occupancy": {str(k): int(v) for k, v in (final_occupancy or {}).items()},
		}

	def write(
		self,
		path: str | Path,
		*,
		run_meta: dict[str, Any] | None = None,
		final_occupancy: dict[str, int] | None = None,
	) -> Path:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		payload = self.to_dict(run_meta=run_meta, final_occupancy=final_occupancy)
		fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as fp:
				json.dump(payload, fp, indent=2, allow_nan=False)
				fp.write("\n")
				fp.flush()
				os.fsync(fp.fileno())
			os.replace(temporary, path)
		except Exception:
			try:
				os.unlink(temporary)
			except FileNotFoundError:
				pass
			raise
		return path

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
