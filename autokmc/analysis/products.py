"""Product rates and backward-propagated mechanisms from ``events.jsonl``.

The KMC engine does not track products or lineage.  This analyzer reconstructs
the causal history of each occupied surface state from the explicit inputs and
outputs in event-schema v2.  A product is strictly a non-feed surface species
that leaves through a ``desorption`` event.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import weakref
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from scipy.stats import chi2

from autokmc.core.constants import (
    PERSISTENCE_SCHEMA_VERSION,
    REACTIONS_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from autokmc.species.smiles import canonical_smiles


class AnalysisError(RuntimeError):
    """Raised when an event log cannot support causal mechanism analysis."""


class _StagingDirectory:
    """Temporary output tree that also cleans up on exceptional exits."""

    def __init__(self, *, prefix: str, directory: Path):
        self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=directory))
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self.path, ignore_errors=True
        )

    def cleanup(self) -> None:
        self._finalizer()


@dataclass(frozen=True)
class _Lineage:
    event_id: str
    step: int
    species: str
    label: str
    parents: tuple["_Lineage", ...] = ()
    kind: str = ""
    direction: str | None = None
    template_key: tuple[str, str, str] | None = None


def _state_key(state: Mapping[str, Any]) -> tuple[str, str]:
    placement = state.get("placement_id")
    if not placement:
        raise AnalysisError(f"surface state has no placement_id: {state!r}")
    return canonical_smiles(state.get("species", "")), str(placement)


def _surface(states: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [state for state in states if state.get("phase") == "surface"]


def _gas(states: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [state for state in states if state.get("phase") == "gas"]


def _template_key(event: Mapping[str, Any]) -> tuple[str, str, str] | None:
    template = event.get("template") or {}
    if not all(template.get(key) for key in ("smiles_a", "smiles_b", "smiles_c")):
        return None
    return (
        canonical_smiles(template["smiles_a"]),
        canonical_smiles(template["smiles_b"]),
        canonical_smiles(template["smiles_c"]),
    )


def _event_label(event: Mapping[str, Any]) -> str:
    kind = str(event.get("kind", ""))
    direction = event.get("direction")
    inputs = event.get("inputs", [])
    outputs = event.get("outputs", [])

    def label(state: Mapping[str, Any]) -> str:
        suffix = "(g)" if state.get("phase") == "gas" else "*"
        return f"{canonical_smiles(state.get('species', ''))}{suffix}"

    left = " + ".join(label(state) for state in inputs) or "∅"
    right = " + ".join(label(state) for state in outputs) or "∅"
    if kind == "diffusion":
        return f"{left} → {right} [diffusion]"
    if kind == "bond" and direction:
        return f"{left} → {right} [{direction}]"
    return f"{left} → {right}"


def _matching_parents(
    outputs: list[Mapping[str, Any]], parents: tuple[_Lineage, ...]
) -> list[_Lineage] | None:
    remaining = list(parents)
    matched: list[_Lineage] = []
    for output in outputs:
        species = canonical_smiles(output.get("species", ""))
        index = next((i for i, parent in enumerate(remaining) if parent.species == species), None)
        if index is None:
            return None
        matched.append(remaining.pop(index))
    return matched if not remaining else None


def _bond_output_lineages(
    event: Mapping[str, Any],
    input_lineages: tuple[_Lineage, ...],
    outputs: list[Mapping[str, Any]],
    *,
    event_id: str,
    step: int,
) -> list[_Lineage]:
    """Create bond lineages while cancelling immediate reversible recrossings."""
    direction = str(event.get("direction", ""))
    template = _template_key(event)
    if direction == "couple" and len(input_lineages) == 2:
        first, second = input_lineages
        if (
            first.event_id == second.event_id
            and first.kind == second.kind == "bond"
            and first.direction == second.direction == "dissoc"
            and first.template_key == second.template_key == template
            and first.parents == second.parents
            and len(first.parents) == 1
        ):
            return [first.parents[0] for _ in outputs]
    if direction == "dissoc" and len(input_lineages) == 1:
        parent = input_lineages[0]
        if (
            parent.kind == "bond"
            and parent.direction == "couple"
            and parent.template_key == template
        ):
            restored = _matching_parents(outputs, parent.parents)
            if restored is not None:
                return restored

    label = _event_label(event)
    return [
        _Lineage(
            event_id=event_id,
            step=step,
            species=canonical_smiles(output.get("species", "")),
            label=label,
            parents=input_lineages,
            kind="bond",
            direction=direction,
            template_key=template,
        )
        for output in outputs
    ]


def _mechanism_steps(lineage: _Lineage, desorption: Mapping[str, Any]) -> list[str]:
    nodes: dict[str, tuple[int, str]] = {}
    stack = [lineage]
    while stack:
        node = stack.pop()
        if node.event_id in nodes:
            continue
        nodes[node.event_id] = (node.step, node.label)
        stack.extend(node.parents)
    ordered = [label for _, label in sorted(nodes.values(), key=lambda item: (item[0], item[1]))]
    ordered.append(_event_label(desorption))
    return ordered


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(
    run_dir: str | Path,
    *,
    manifest_filename: str | Path = RUN_MANIFEST_FILENAME,
    output_dir: str | Path | None = None,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    n_blocks: int = 10,
    strict: bool = True,
) -> dict[str, Any]:
    """Analyze one KMC run and write product/mechanism tables.

    Rates are observed event counts divided by KMC elapsed time.  They are not
    averages of the microscopic ``rate_hz`` propensities stored on events.
    """
    run_path = Path(run_dir).expanduser().resolve()
    manifest_path = Path(manifest_filename).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = run_path / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise AnalysisError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = run_path / manifest.get("files", {}).get("events", REACTIONS_FILENAME)
    if not events_path.is_file():
        raise AnalysisError(f"missing event log: {events_path}")
    if str(manifest.get("event_schema_version")) != str(PERSISTENCE_SCHEMA_VERSION):
        raise AnalysisError(
            "run manifest/event schema is not compatible with this analyzer: "
            f"expected {PERSISTENCE_SCHEMA_VERSION!r}, got "
            f"{manifest.get('event_schema_version')!r}"
        )
    if type(n_blocks) is not int or n_blocks < 0:
        raise AnalysisError(f"n_blocks must be a non-negative integer, got {n_blocks!r}")

    feed_species = {
        canonical_smiles(item.get("species", item.get("input_smiles", "")))
        for item in manifest.get("feed_reactants", [])
    }
    initial = manifest.get("initial_state", {})
    t_start = float(initial.get("time_s", 0.0) if start_time_s is None else start_time_s)
    result_meta = manifest.get("result") or {}
    configured_end = result_meta.get("final_time_s")
    end_value = configured_end if end_time_s is None else end_time_s
    t_end = None if end_value is None else float(end_value)
    end_known_during_scan = t_end is not None
    if t_end is not None and t_end <= t_start:
        raise AnalysisError("analysis end time must be greater than start time")

    final_destination = (
        Path(output_dir).expanduser().resolve() if output_dir else run_path / "analysis"
    )
    final_destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _StagingDirectory(
        prefix=f".{final_destination.name}-staging-",
        directory=final_destination.parent,
    )
    destination = staging.path
    product_events_path = destination / "product_events.jsonl"

    active: dict[tuple[str, str], _Lineage] = {}
    for index, state in enumerate(initial.get("occupied_surface_states", [])):
        key = _state_key(state)
        active[key] = _Lineage(
            event_id=f"initial-{index}",
            step=int(initial.get("step", 0)),
            species=key[0],
            label=f"initial {key[0]}*",
            kind="initial",
        )

    product_counts: Counter[str] = Counter()
    mechanism_counts: Counter[tuple[str, str]] = Counter()
    mechanism_steps: dict[tuple[str, str], list[str]] = {}
    block_counts: Counter[tuple[str, int]] = Counter()
    incomplete_events = 0
    last_time = float(initial.get("time_s", 0.0))
    n_events = 0

    with events_path.open("r", encoding="utf-8") as source, product_events_path.open(
        "w", encoding="utf-8"
    ) as product_handle:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            event_time = float(event["time_s"])
            if t_end is not None and event_time > t_end:
                break
            if str(event.get("schema_version")) != str(PERSISTENCE_SCHEMA_VERSION):
                raise AnalysisError(
                    f"event line {line_number} has incompatible schema "
                    f"{event.get('schema_version')!r}"
                )
            n_events += 1
            last_time = max(last_time, event_time)
            if "inputs" not in event or "outputs" not in event:
                raise AnalysisError(
                    f"event line {line_number} predates schema v2 and has no inputs/outputs"
                )
            event_id = f"event-{event.get('step', line_number)}-{line_number}"
            step = int(event.get("step", line_number))
            surface_inputs = _surface(event["inputs"])
            surface_outputs = _surface(event["outputs"])
            input_lineages: list[_Lineage] = []
            for state in surface_inputs:
                key = _state_key(state)
                lineage = active.pop(key, None)
                if lineage is None:
                    incomplete_events += 1
                    if strict:
                        raise AnalysisError(
                            f"event line {line_number} consumes an untracked surface state: {key}"
                        )
                    lineage = _Lineage(
                        event_id=f"unknown-{line_number}-{len(input_lineages)}",
                        step=step,
                        species=key[0],
                        label=f"unknown source of {key[0]}*",
                        kind="unknown",
                    )
                input_lineages.append(lineage)

            kind = str(event.get("kind", ""))
            if kind == "diffusion":
                if len(input_lineages) != 1 or len(surface_outputs) != 1:
                    raise AnalysisError(f"invalid diffusion transition on line {line_number}")
                output_lineages = input_lineages
            elif kind == "bond":
                output_lineages = _bond_output_lineages(
                    event,
                    tuple(input_lineages),
                    surface_outputs,
                    event_id=event_id,
                    step=step,
                )
            else:
                label = _event_label(event)
                output_lineages = [
                    _Lineage(
                        event_id=event_id,
                        step=step,
                        species=canonical_smiles(state.get("species", "")),
                        label=label,
                        parents=tuple(input_lineages),
                        kind=kind,
                        direction=event.get("direction"),
                    )
                    for state in surface_outputs
                ]
            for state, lineage in zip(surface_outputs, output_lineages):
                key = _state_key(state)
                if key in active and strict:
                    raise AnalysisError(
                        f"event line {line_number} overwrites an occupied surface state: {key}"
                    )
                active[key] = lineage

            # Strict product definition: a non-feed species leaves an occupied
            # surface state through a desorption event.
            if kind != "desorption" or event_time < t_start:
                continue
            gas_outputs = _gas(event["outputs"])
            if not gas_outputs or not input_lineages:
                continue
            for gas_output in gas_outputs:
                product = canonical_smiles(gas_output.get("species", ""))
                if product in feed_species:
                    continue
                lineage = next(
                    (item for item in input_lineages if item.species == product),
                    input_lineages[0],
                )
                steps = _mechanism_steps(lineage, event)
                fingerprint = " | ".join(steps)
                mechanism_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
                product_counts[product] += 1
                mechanism_counts[(product, mechanism_id)] += 1
                mechanism_steps.setdefault((product, mechanism_id), steps)
                if t_end is not None and n_blocks > 0:
                    fraction = (event_time - t_start) / (t_end - t_start)
                    block = min(n_blocks - 1, max(0, int(fraction * n_blocks)))
                    block_counts[(product, block)] += 1
                product_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "time_s": event_time,
                            "product": product,
                            "mechanism_id": mechanism_id,
                            "mechanism": steps,
                        }
                    )
                    + "\n"
                )

    if t_end is None:
        t_end = last_time
    duration = float(t_end - t_start)
    if duration <= 0.0:
        raise AnalysisError("event log has no positive KMC analysis duration")
    n_surface_atoms = int(manifest.get("catalyst", {}).get("n_surface_atoms", 0) or 0)
    if not end_known_during_scan and n_blocks > 0:
        with product_events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                fraction = (float(item["time_s"]) - t_start) / duration
                block = min(n_blocks - 1, max(0, int(fraction * n_blocks)))
                block_counts[(str(item["product"]), block)] += 1

    product_rows = []
    for product, count in sorted(product_counts.items()):
        rate = count / duration
        # Exact two-sided 95% Garwood interval for a Poisson event count.
        count_low = 0.0 if count == 0 else 0.5 * chi2.ppf(0.025, 2 * count)
        count_high = 0.5 * chi2.ppf(0.975, 2 * (count + 1))
        product_rows.append(
            {
                "product": product,
                "count": count,
                "start_time_s": t_start,
                "end_time_s": t_end,
                "duration_s": duration,
                "rate_hz": rate,
                "rate_ci95_low_hz": count_low / duration,
                "rate_ci95_high_hz": count_high / duration,
                "tof_per_surface_atom_s-1": (
                    rate / n_surface_atoms if n_surface_atoms > 0 else None
                ),
            }
        )
    _write_csv(
        destination / "product_rates.csv",
        [
            "product", "count", "start_time_s", "end_time_s", "duration_s",
            "rate_hz", "rate_ci95_low_hz", "rate_ci95_high_hz",
            "tof_per_surface_atom_s-1",
        ],
        product_rows,
    )

    mechanism_rows = []
    for (product, mechanism_id), count in sorted(mechanism_counts.items()):
        total = product_counts[product]
        mechanism_rows.append(
            {
                "product": product,
                "mechanism_id": mechanism_id,
                "count": count,
                "fraction": count / total,
                "rate_hz": count / duration,
                "mechanism": json.dumps(mechanism_steps[(product, mechanism_id)]),
            }
        )
    _write_csv(
        destination / "mechanisms.csv",
        ["product", "mechanism_id", "count", "fraction", "rate_hz", "mechanism"],
        mechanism_rows,
    )

    block_rows = []
    if n_blocks > 0:
        width = duration / n_blocks
        for product in sorted(product_counts):
            for block in range(n_blocks):
                count = block_counts[(product, block)]
                block_rows.append(
                    {
                        "product": product,
                        "block": block,
                        "start_time_s": t_start + block * width,
                        "end_time_s": t_start + (block + 1) * width,
                        "count": count,
                        "rate_hz": count / width,
                    }
                )
    _write_csv(
        destination / "rate_blocks.csv",
        ["product", "block", "start_time_s", "end_time_s", "count", "rate_hz"],
        block_rows,
    )

    summary = {
        "schema_version": "1",
        "run_dir": str(run_path),
        "event_schema_version": PERSISTENCE_SCHEMA_VERSION,
        "feed_species": sorted(feed_species),
        "start_time_s": t_start,
        "end_time_s": t_end,
        "duration_s": duration,
        "n_events_read": n_events,
        "n_surface_atoms": n_surface_atoms,
        "incomplete_events": incomplete_events,
        "products": product_rows,
        "mechanisms": mechanism_rows,
        "outputs": {
            "product_rates": str(final_destination / "product_rates.csv"),
            "product_events": str(final_destination / "product_events.jsonl"),
            "mechanisms": str(final_destination / "mechanisms.csv"),
            "rate_blocks": str(final_destination / "rate_blocks.csv"),
        },
    }
    summary["outputs"]["summary"] = str(final_destination / "analysis_summary.json")
    _atomic_json(destination / "analysis_summary.json", summary)
    final_destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        "product_rates.csv",
        "product_events.jsonl",
        "mechanisms.csv",
        "rate_blocks.csv",
        "analysis_summary.json",
    ):
        os.replace(destination / filename, final_destination / filename)
    staging.cleanup()
    return summary


__all__ = ["AnalysisError", "analyze_run"]
