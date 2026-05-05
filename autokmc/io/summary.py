"""Run-summary persistence helpers."""

from __future__ import annotations

import json
import math
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


def _stats(values: list[float]) -> dict[str, float]:
	if not values:
		return {"mean": float("nan"), "std": float("nan"),
				"min": float("nan"), "max": float("nan")}
	arr = np.asarray(values, dtype=float)
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
		"_reactant_smiles",
	)

	def __init__(self, reactant_smiles: set[str] | None = None):
		self._buckets: dict[tuple, dict[str, list[float]]] = {}
		self._total_by_kind: dict[str, int] = {}
		self._n: int = 0
		self._first_step: dict[tuple, int] = {}
		self._last_step: dict[tuple, int] = {}
		self._reactant_smiles: frozenset[str] = frozenset(reactant_smiles or [])

	def add(self, reaction, *, step: int) -> None:
		smiles = _reaction_smiles(reaction)
		key = (
			str(reaction.kind),
			str(smiles),
			int(reaction.site.iso_class),
			int(reaction.lateral_class.lateral_class),
		)
		b = self._buckets.setdefault(key, {"rate": [], "delta_e": [], "barrier": []})
		b["rate"].append(float(reaction.rate))
		b["delta_e"].append(float(reaction.delta_e))
		b["barrier"].append(float(reaction.barrier))

		self._total_by_kind[key[0]] = self._total_by_kind.get(key[0], 0) + 1
		self._n += 1
		self._first_step.setdefault(key, int(step))
		self._last_step[key] = int(step)

	def _production_summary(self, run_meta: dict[str, Any] | None) -> dict[str, Any]:
		kmc_time: float | None = None
		if run_meta:
			t = run_meta.get("total_time_s")
			if t is not None and float(t) > 0:
				kmc_time = float(t)

		steps_executed: int | None = None
		if run_meta:
			se = run_meta.get("steps_executed")
			if se is not None:
				steps_executed = int(se)

		product_map: dict[str, list[dict[str, Any]]] = {}
		for key, b in self._buckets.items():
			kind, smiles, iso, lat = key
			if kind != "desorption":
				continue
			if self._reactant_smiles and smiles in self._reactant_smiles:
				continue
			product_map.setdefault(smiles, []).append({
				"iso_class": iso,
				"lateral_class": lat,
				"count": len(b["rate"]),
				"first_step": self._first_step[key],
				"last_step": self._last_step[key],
			})

		by_species: dict[str, Any] = {}
		total_count = 0
		for smiles in sorted(product_map):
			breakdowns = sorted(product_map[smiles], key=lambda d: (d["iso_class"], d["lateral_class"]))
			count = sum(d["count"] for d in breakdowns)
			total_count += count
			by_species[smiles] = {
				"desorption_count": count,
				"production_rate_hz": count / kmc_time if kmc_time is not None else None,
				"events_per_step": count / steps_executed if steps_executed and steps_executed > 0 else None,
				"iso_breakdown": breakdowns,
			}

		note = (
			"Desorption events of species not in the user-supplied reactants list "
			"(partial_pressure_bar=0 species created on-the-fly by bond coupling)."
			if self._reactant_smiles else
			"Desorption events of all species (no reactant set configured)."
		)
		return {
			"note": note,
			"kmc_time_s": kmc_time,
			"steps_executed": steps_executed,
			"reactant_smiles": sorted(self._reactant_smiles),
			"product_species": sorted(by_species),
			"by_species": by_species,
			"total_product_desorptions": total_count,
			"total_production_rate_hz": total_count / kmc_time if kmc_time is not None else None,
		}

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
			"production_summary": self._production_summary(run_meta),
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
		with path.open("w", encoding="utf-8") as fp:
			json.dump(payload, fp, indent=2)
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
