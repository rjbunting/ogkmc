"""Deterministic table builders for offline product analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import chi2


PRODUCT_RATE_FIELDS = (
    "product",
    "count",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "rate_hz",
    "rate_ci95_low_hz",
    "rate_ci95_high_hz",
    "tof_per_surface_atom_s-1",
)
MECHANISM_FIELDS = (
    "product",
    "mechanism_id",
    "count",
    "fraction",
    "rate_hz",
    "mechanism",
)
RATE_BLOCK_FIELDS = (
    "product",
    "block",
    "start_time_s",
    "end_time_s",
    "count",
    "rate_hz",
)


def build_product_rate_rows(
    product_counts: Mapping[str, int],
    *,
    start_time_s: float,
    end_time_s: float,
    duration_s: float,
    n_surface_atoms: int,
) -> list[dict[str, Any]]:
    """Reduce product counts into rate estimates and Garwood intervals."""
    rows: list[dict[str, Any]] = []
    for product, count in sorted(product_counts.items()):
        rate = count / duration_s
        count_low = 0.0 if count == 0 else 0.5 * chi2.ppf(0.025, 2 * count)
        count_high = 0.5 * chi2.ppf(0.975, 2 * (count + 1))
        rows.append(
            {
                "product": product,
                "count": count,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
                "rate_hz": rate,
                "rate_ci95_low_hz": count_low / duration_s,
                "rate_ci95_high_hz": count_high / duration_s,
                "tof_per_surface_atom_s-1": (
                    rate / n_surface_atoms if n_surface_atoms > 0 else None
                ),
            }
        )
    return rows


def build_mechanism_rows(
    mechanism_counts: Mapping[tuple[str, str], int],
    *,
    product_counts: Mapping[str, int],
    mechanism_steps: Mapping[tuple[str, str], list[str]],
    duration_s: float,
) -> list[dict[str, Any]]:
    """Reduce mechanism counts into fractions, rates, and serialized paths."""
    rows: list[dict[str, Any]] = []
    for (product, mechanism_id), count in sorted(mechanism_counts.items()):
        total = product_counts[product]
        rows.append(
            {
                "product": product,
                "mechanism_id": mechanism_id,
                "count": count,
                "fraction": count / total,
                "rate_hz": count / duration_s,
                "mechanism": json.dumps(mechanism_steps[(product, mechanism_id)]),
            }
        )
    return rows


def build_rate_block_rows(
    block_counts: Mapping[tuple[str, int], int],
    *,
    products: Iterable[str],
    start_time_s: float,
    duration_s: float,
    n_blocks: int,
) -> list[dict[str, Any]]:
    """Build fixed-width block rates for every observed product."""
    rows: list[dict[str, Any]] = []
    if n_blocks > 0:
        width = duration_s / n_blocks
        for product in sorted(products):
            for block in range(n_blocks):
                count = block_counts.get((product, block), 0)
                rows.append(
                    {
                        "product": product,
                        "block": block,
                        "start_time_s": start_time_s + block * width,
                        "end_time_s": start_time_s + (block + 1) * width,
                        "count": count,
                        "rate_hz": count / width,
                    }
                )
    return rows


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write one analysis table with stable columns and row order."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "MECHANISM_FIELDS",
    "PRODUCT_RATE_FIELDS",
    "RATE_BLOCK_FIELDS",
    "build_mechanism_rows",
    "build_product_rate_rows",
    "build_rate_block_rows",
    "write_csv",
]
