"""Human-readable, offline reports for persisted AutoKMC runs."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from autokmc.analysis.products import AnalysisError, analyze_run
from autokmc.io._files import atomic_output_path, ensure_directory
from autokmc.io.reaction_index import (
    REACTION_INDEX_FILENAME,
    load_reaction_index,
    resolve_event_definition,
)
from autokmc.species.smiles import canonical_smiles


REPORT_SCHEMA_VERSION = "1"


class ReportError(RuntimeError):
    """A persisted run cannot be converted into a report."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ReportError(f"missing report input: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"could not read JSON report input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"JSON report input is not an object: {path}")
    return payload


def _artifact_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    file_key: str,
    fallback: str | Path,
    artifact_types: Sequence[str] = (),
) -> Path:
    files = manifest.get("files") or {}
    if isinstance(files, Mapping) and files.get(file_key):
        return run_dir / str(files[file_key])
    artifacts = manifest.get("artifacts") or []
    if isinstance(artifacts, Mapping):
        artifacts = list(artifacts.values())
    if isinstance(artifacts, list):
        wanted = {str(value) for value in artifact_types}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            artifact_type = str(
                artifact.get("type")
                or artifact.get("artifact_type")
                or artifact.get("kind")
                or ""
            )
            if artifact_type in wanted and artifact.get("path"):
                return run_dir / str(artifact["path"])
    return run_dir / fallback


def _surface(states: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [state for state in states if state.get("phase") == "surface"]


def _state_key(state: Mapping[str, Any]) -> str:
    placement = state.get("placement_id")
    if placement:
        return str(placement)
    fallback = {
        "species": canonical_smiles(str(state.get("species", ""))),
        "site_id": state.get("site_id"),
        "member_id": state.get("member_id"),
        "site_iso_class": state.get("site_iso_class"),
        "site_member_index": state.get("site_member_index"),
        "adsorbate_node_ids": state.get("adsorbate_node_ids"),
    }
    return json.dumps(fallback, sort_keys=True, separators=(",", ":"))


def _read_events(
    events_path: Path,
    reaction_index_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not events_path.is_file():
        raise ReportError(f"missing event log: {events_path}")
    warnings: list[str] = []
    try:
        definitions = load_reaction_index(reaction_index_path)
    except ValueError as exc:
        definitions = {}
        warnings.append(f"Reaction definitions could not be loaded: {exc}")
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReportError(
                    f"invalid event JSON on line {line_number} of "
                    f"{events_path}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ReportError(
                    f"event line {line_number} of {events_path} is not an object"
                )
            events.append(resolve_event_definition(raw, definitions))
    return events, warnings


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _coverage_metrics(
    manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    initial = manifest.get("initial_state") or {}
    initial_states = initial.get("occupied_surface_states") or []
    active = {
        _state_key(state): dict(state)
        for state in _surface(initial_states)
        if isinstance(state, Mapping)
    }
    initial_time = _finite_float(initial.get("time_s")) or 0.0
    last_time = initial_time
    occupancy_integral = 0.0
    species_integral: defaultdict[str, float] = defaultdict(float)
    peak_placements = len(active)
    transition_warnings = 0

    def integrate(until: float) -> None:
        nonlocal last_time, occupancy_integral
        if until < last_time:
            return
        duration = until - last_time
        occupancy_integral += duration * len(active)
        counts = Counter(
            canonical_smiles(str(state.get("species", "")))
            for state in active.values()
        )
        for species, count in counts.items():
            species_integral[species] += duration * count
        last_time = until

    for event in events:
        event_time = _finite_float(event.get("time_s"))
        if event_time is None:
            transition_warnings += 1
            continue
        integrate(event_time)
        for state in _surface(event.get("inputs") or []):
            if active.pop(_state_key(state), None) is None:
                transition_warnings += 1
        for state in _surface(event.get("outputs") or []):
            key = _state_key(state)
            if key in active:
                transition_warnings += 1
            active[key] = dict(state)
        peak_placements = max(peak_placements, len(active))

    result = manifest.get("result") or {}
    final_time = (
        _finite_float(result.get("final_time_s"))
        or _finite_float(manifest.get("simulated_time_s"))
        or last_time
    )
    integrate(max(last_time, final_time))
    duration = max(0.0, last_time - initial_time)
    n_surface_atoms = int(
        (manifest.get("catalyst") or {}).get("n_surface_atoms", 0) or 0
    )
    final_counts = Counter(
        canonical_smiles(str(state.get("species", "")))
        for state in active.values()
    )
    species = sorted(set(final_counts) | set(species_integral))
    rows = [
        {
            "species": name,
            "final_occupied_placements": int(final_counts.get(name, 0)),
            "final_placements_per_surface_atom": (
                float(final_counts.get(name, 0)) / n_surface_atoms
                if n_surface_atoms > 0
                else None
            ),
            "time_average_occupied_placements": (
                species_integral.get(name, 0.0) / duration
                if duration > 0.0
                else float(final_counts.get(name, 0))
            ),
        }
        for name in species
    ]
    return {
        "rows": rows,
        "initial_occupied_placements": len(_surface(initial_states)),
        "final_occupied_placements": len(active),
        "peak_occupied_placements": peak_placements,
        "time_average_occupied_placements": (
            occupancy_integral / duration if duration > 0.0 else len(active)
        ),
        "duration_s": duration,
        "n_surface_atoms": n_surface_atoms,
        "transition_warnings": transition_warnings,
    }


def _directional_fluxes(
    events: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> dict[str, Any]:
    directions: Counter[str] = Counter()
    for event in events:
        kind = str(event.get("kind", "unknown"))
        direction = event.get("direction")
        label = str(direction) if direction else kind
        directions[label] += 1

    rows = [
        {
            "direction": direction,
            "count": int(count),
            "observed_flux_hz": (
                float(count) / duration_s if duration_s > 0.0 else None
            ),
        }
        for direction, count in sorted(directions.items())
    ]
    pairs = (
        ("adsorption / desorption", "adsorption", "desorption"),
        ("bond couple / dissociate", "couple", "dissoc"),
        ("diffusion A→B / B→A", "a_to_b", "b_to_a"),
    )
    net_rows = []
    for label, forward, reverse in pairs:
        forward_count = int(directions.get(forward, 0))
        reverse_count = int(directions.get(reverse, 0))
        if forward_count == 0 and reverse_count == 0:
            continue
        net = forward_count - reverse_count
        net_rows.append(
            {
                "pair": label,
                "forward_count": forward_count,
                "reverse_count": reverse_count,
                "net_count": net,
                "net_flux_hz": net / duration_s if duration_s > 0.0 else None,
            }
        )
    return {"rows": rows, "net_rows": net_rows}


def _convergence_rows(rate_blocks_path: Path) -> list[dict[str, Any]]:
    if not rate_blocks_path.is_file():
        return []
    grouped: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    with rate_blocks_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                grouped[str(row["product"])].append(
                    (int(row["block"]), float(row["rate_hz"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    results: list[dict[str, Any]] = []
    for product, indexed_rates in sorted(grouped.items()):
        rates = [rate for _, rate in sorted(indexed_rates)]
        if not rates:
            continue
        midpoint = max(1, len(rates) // 2)
        first = fmean(rates[:midpoint])
        second = fmean(rates[midpoint:]) if rates[midpoint:] else first
        mean = fmean(rates)
        results.append(
            {
                "product": product,
                "blocks": len(rates),
                "mean_rate_hz": mean,
                "relative_block_stddev": (
                    pstdev(rates) / mean if len(rates) > 1 and mean > 0.0 else None
                ),
                "late_to_early_rate_ratio": (
                    second / first
                    if first > 0.0
                    else (1.0 if second == 0.0 else None)
                ),
            }
        )
    return results


def _performance_metrics(path: Path) -> dict[str, Any]:
    payload = _load_json(path, required=False)
    summary = payload.get("summary") or {}
    if not isinstance(summary, Mapping):
        summary = {}
    bottlenecks = summary.get("top_bottlenecks") or []
    if not isinstance(bottlenecks, list):
        bottlenecks = []
    return {
        "wall_time_s": summary.get("wall_time_s", payload.get("wall_time_s")),
        "events_per_wall_second": summary.get("events_per_wall_second"),
        "cache": summary.get("cache") or {},
        "overhead_s": summary.get("overhead_s") or {},
        "top_bottlenecks": [
            dict(item) for item in bottlenecks if isinstance(item, Mapping)
        ],
    }


def _status_value(manifest: Mapping[str, Any]) -> str:
    status = manifest.get("status")
    if isinstance(status, Mapping):
        return str(status.get("state") or status.get("status") or "unknown")
    if status:
        return str(status)
    lifecycle = manifest.get("lifecycle") or {}
    if isinstance(lifecycle, Mapping) and lifecycle.get("status"):
        return str(lifecycle["status"])
    result = manifest.get("result") or {}
    if isinstance(result, Mapping) and result.get("status"):
        return str(result["status"])
    return "completed" if manifest.get("result") else "incomplete"


def _termination_reason(manifest: Mapping[str, Any]) -> str:
    termination = manifest.get("termination")
    if isinstance(termination, Mapping):
        reason = termination.get("reason")
        if reason:
            return str(reason)
    result = manifest.get("result") or {}
    if isinstance(result, Mapping) and result.get("termination_reason"):
        return str(result["termination_reason"])
    lifecycle = manifest.get("lifecycle") or {}
    if isinstance(lifecycle, Mapping) and lifecycle.get("termination_reason"):
        return str(lifecycle["termination_reason"])
    return "not recorded"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.6g}"
    return str(value)


def _md_cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(columns: Sequence[tuple[str, str]], rows: Iterable[Mapping[str, Any]]) -> str:
    row_list = list(rows)
    if not row_list:
        return "_No data available._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_md_cell(row.get(key)) for key, _ in columns)
        + " |"
        for row in row_list
    ]
    return "\n".join([header, divider, *body])


def _html_table(columns: Sequence[tuple[str, str]], rows: Iterable[Mapping[str, Any]]) -> str:
    row_list = list(rows)
    if not row_list:
        return "<p class=\"empty\">No data available.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(_fmt(row.get(key)))}</td>"
            for key, _ in columns
        )
        + "</tr>"
        for row in row_list
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _artifact_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts") or []
    if isinstance(artifacts, Mapping):
        artifacts = list(artifacts.values())
    if not isinstance(artifacts, list):
        return []
    rows = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        rows.append(
            {
                "path": artifact.get("path"),
                "type": (
                    artifact.get("type")
                    or artifact.get("artifact_type")
                    or artifact.get("kind")
                ),
                "status": (
                    artifact.get("status")
                    or (
                        "present"
                        if artifact.get("present") is True
                        else "missing"
                        if artifact.get("present") is False
                        else None
                    )
                ),
                "size_bytes": artifact.get(
                    "size_bytes",
                    artifact.get("total_size_bytes"),
                ),
                "sha256": (
                    str(artifact.get("sha256", ""))[:12]
                    if artifact.get("sha256")
                    else None
                ),
            }
        )
    return rows


def _build_markdown(data: Mapping[str, Any]) -> str:
    coverage = data["coverage"]
    fluxes = data["directional_fluxes"]
    products = data["products"]
    performance = data["performance"]
    overview = [
        {"metric": "Run ID", "value": data.get("run_id")},
        {"metric": "Status", "value": data.get("status")},
        {"metric": "Termination", "value": data.get("termination_reason")},
        {"metric": "Events", "value": data.get("n_events")},
        {"metric": "Simulated time (s)", "value": data.get("simulated_time_s")},
        {"metric": "Wall time (s)", "value": performance.get("wall_time_s")},
        {
            "metric": "Final occupied placements",
            "value": coverage.get("final_occupied_placements"),
        },
        {
            "metric": "Time-average occupied placements",
            "value": coverage.get("time_average_occupied_placements"),
        },
    ]
    sections = [
        "# AutoKMC run report",
        "",
        f"Generated `{data['generated_utc']}` from `{data['run_dir']}`.",
        "",
        "## Run overview",
        "",
        _markdown_table((("metric", "Metric"), ("value", "Value")), overview),
        "",
        "## Surface coverage",
        "",
        (
            "Coverage below counts occupied adsorbate placements. "
            "The per-surface-atom value is a normalization, not a claim that "
            "every placement occupies exactly one catalyst atom."
        ),
        "",
        _markdown_table(
            (
                ("species", "Species"),
                ("final_occupied_placements", "Final placements"),
                (
                    "final_placements_per_surface_atom",
                    "Final placements / surface atom",
                ),
                (
                    "time_average_occupied_placements",
                    "Time-average placements",
                ),
            ),
            coverage.get("rows", []),
        ),
        "",
        "## Directional event fluxes",
        "",
        _markdown_table(
            (
                ("direction", "Direction"),
                ("count", "Count"),
                ("observed_flux_hz", "Observed flux (s⁻¹)"),
            ),
            fluxes.get("rows", []),
        ),
        "",
        "### Net directional balance",
        "",
        _markdown_table(
            (
                ("pair", "Pair"),
                ("forward_count", "Forward"),
                ("reverse_count", "Reverse"),
                ("net_count", "Net"),
                ("net_flux_hz", "Net flux (s⁻¹)"),
            ),
            fluxes.get("net_rows", []),
        ),
        "",
        "## Cumulative products",
        "",
        _markdown_table(
            (
                ("product", "Product"),
                ("count", "Cumulative count"),
                ("rate_hz", "Observed rate (s⁻¹)"),
                ("rate_ci95_low_hz", "95% CI low"),
                ("rate_ci95_high_hz", "95% CI high"),
            ),
            products,
        ),
        "",
        "## Product-rate convergence",
        "",
        _markdown_table(
            (
                ("product", "Product"),
                ("blocks", "Blocks"),
                ("mean_rate_hz", "Mean block rate (s⁻¹)"),
                ("relative_block_stddev", "Relative block σ"),
                ("late_to_early_rate_ratio", "Late / early rate"),
            ),
            data["convergence"],
        ),
        "",
        "## Performance bottlenecks",
        "",
        _markdown_table(
            (
                ("name", "Operation"),
                ("wall_time_s", "Accumulated time (s)"),
                ("fraction_of_wall_time", "Fraction of wall time"),
            ),
            performance.get("top_bottlenecks", []),
        ),
        "",
        "## Artifact inventory",
        "",
        _markdown_table(
            (
                ("path", "Path"),
                ("type", "Type"),
                ("status", "Status"),
                ("size_bytes", "Bytes"),
                ("sha256", "SHA-256 prefix"),
            ),
            data["artifacts"],
        ),
    ]
    warnings = list(data.get("warnings") or [])
    if warnings:
        sections.extend(
            [
                "",
                "## Report notes",
                "",
                *[f"- {warning}" for warning in warnings],
            ]
        )
    sections.append("")
    return "\n".join(sections)


def _build_html(data: Mapping[str, Any]) -> str:
    coverage = data["coverage"]
    fluxes = data["directional_fluxes"]
    products = data["products"]
    performance = data["performance"]
    overview = [
        {"metric": "Run ID", "value": data.get("run_id")},
        {"metric": "Status", "value": data.get("status")},
        {"metric": "Termination", "value": data.get("termination_reason")},
        {"metric": "Events", "value": data.get("n_events")},
        {"metric": "Simulated time (s)", "value": data.get("simulated_time_s")},
        {"metric": "Wall time (s)", "value": performance.get("wall_time_s")},
    ]
    warnings = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in data.get("warnings") or []
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoKMC run report</title>
<style>
:root {{ color-scheme: light dark; --panel:#f4f6f8; --line:#ccd3da; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --panel:#1c232b; --line:#394653; }}
}}
body {{ font: 15px/1.45 system-ui,sans-serif; margin:0; }}
main {{ max-width:1100px; margin:auto; padding:2rem; }}
h1 {{ margin-bottom:.2rem; }} h2 {{ margin-top:2rem; }}
.meta,.note,.empty {{ color:#64707d; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:.8rem; }}
.card {{ background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:.8rem 1rem; }}
table {{ width:100%; border-collapse:collapse; margin:.6rem 0 1rem; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line);
  padding:.45rem .55rem; vertical-align:top; }}
th {{ background:var(--panel); position:sticky; top:0; }}
.table-wrap {{ overflow:auto; }}
code {{ overflow-wrap:anywhere; }}
</style>
</head>
<body><main>
<h1>AutoKMC run report</h1>
<p class="meta">Generated {html.escape(str(data["generated_utc"]))} from
<code>{html.escape(str(data["run_dir"]))}</code>.</p>
<h2>Run overview</h2>
<div class="cards">{"".join(
    f'<div class="card"><strong>{html.escape(str(row["metric"]))}</strong>'
    f'<br>{html.escape(_fmt(row["value"]))}</div>' for row in overview
)}</div>
<h2>Surface coverage</h2>
<p class="note">Coverage counts occupied adsorbate placements. The
per-surface-atom value is a normalization, not a one-site assumption.</p>
<div class="table-wrap">{_html_table((
    ("species", "Species"),
    ("final_occupied_placements", "Final placements"),
    ("final_placements_per_surface_atom", "Final placements / surface atom"),
    ("time_average_occupied_placements", "Time-average placements"),
), coverage.get("rows", []))}</div>
<h2>Directional event fluxes</h2>
<div class="table-wrap">{_html_table((
    ("direction", "Direction"), ("count", "Count"),
    ("observed_flux_hz", "Observed flux (s⁻¹)"),
), fluxes.get("rows", []))}</div>
<h3>Net directional balance</h3>
<div class="table-wrap">{_html_table((
    ("pair", "Pair"), ("forward_count", "Forward"),
    ("reverse_count", "Reverse"), ("net_count", "Net"),
    ("net_flux_hz", "Net flux (s⁻¹)"),
), fluxes.get("net_rows", []))}</div>
<h2>Cumulative products</h2>
<div class="table-wrap">{_html_table((
    ("product", "Product"), ("count", "Cumulative count"),
    ("rate_hz", "Observed rate (s⁻¹)"),
    ("rate_ci95_low_hz", "95% CI low"),
    ("rate_ci95_high_hz", "95% CI high"),
), products)}</div>
<h2>Product-rate convergence</h2>
<div class="table-wrap">{_html_table((
    ("product", "Product"), ("blocks", "Blocks"),
    ("mean_rate_hz", "Mean block rate (s⁻¹)"),
    ("relative_block_stddev", "Relative block σ"),
    ("late_to_early_rate_ratio", "Late / early rate"),
), data["convergence"])}</div>
<h2>Performance bottlenecks</h2>
<div class="table-wrap">{_html_table((
    ("name", "Operation"), ("wall_time_s", "Accumulated time (s)"),
    ("fraction_of_wall_time", "Fraction of wall time"),
), performance.get("top_bottlenecks", []))}</div>
<h2>Artifact inventory</h2>
<div class="table-wrap">{_html_table((
    ("path", "Path"), ("type", "Type"), ("status", "Status"),
    ("size_bytes", "Bytes"), ("sha256", "SHA-256 prefix"),
), data["artifacts"])}</div>
{f"<h2>Report notes</h2><ul>{warnings}</ul>" if warnings else ""}
</main></body></html>
"""


def generate_run_report(
    run_dir: str | Path,
    *,
    manifest_filename: str | Path = "run_manifest.json",
    output_dir: str | Path | None = None,
    n_blocks: int = 10,
    strict: bool = True,
    refresh_analysis: bool = True,
) -> dict[str, Any]:
    """Generate self-contained Markdown and HTML reports from persisted data."""
    run_path = Path(run_dir).expanduser().resolve()
    manifest_path = Path(manifest_filename).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = run_path / manifest_path
    manifest = _load_json(manifest_path.resolve())

    events_path = _artifact_path(
        run_path,
        manifest,
        file_key="events",
        fallback="events.jsonl",
        artifact_types=("events", "event_log", "autokmc.events"),
    )
    reaction_index_path = _artifact_path(
        run_path,
        manifest,
        file_key="reaction_index",
        fallback=Path("reactions") / REACTION_INDEX_FILENAME,
        artifact_types=("reaction_index", "autokmc.reaction_index"),
    )
    performance_path = _artifact_path(
        run_path,
        manifest,
        file_key="performance",
        fallback=Path("diagnostics") / "performance.json",
        artifact_types=(
            "performance",
            "performance_diagnostics",
            "autokmc.performance",
        ),
    )
    events, warnings = _read_events(events_path, reaction_index_path)
    coverage = _coverage_metrics(manifest, events)
    if coverage["transition_warnings"]:
        warnings.append(
            f"{coverage['transition_warnings']} surface-state transitions "
            "could not be matched exactly while computing coverage."
        )

    analysis: dict[str, Any] = {}
    if refresh_analysis:
        try:
            analysis = analyze_run(
                run_path,
                manifest_filename=manifest_path,
                n_blocks=n_blocks,
                strict=strict,
            )
        except AnalysisError as exc:
            warnings.append(f"Product/mechanism analysis was unavailable: {exc}")
    if not analysis:
        analysis = _load_json(
            run_path / "analysis" / "analysis_summary.json",
            required=False,
        )
    products = [
        dict(row)
        for row in analysis.get("products", [])
        if isinstance(row, Mapping)
    ]
    analysis_outputs = analysis.get("outputs") or {}
    rate_blocks_path = Path(
        str(
            analysis_outputs.get(
                "rate_blocks",
                run_path / "analysis" / "rate_blocks.csv",
            )
        )
    )
    if not rate_blocks_path.is_absolute():
        rate_blocks_path = run_path / rate_blocks_path

    initial_time = _finite_float(
        (manifest.get("initial_state") or {}).get("time_s")
    ) or 0.0
    result = manifest.get("result") or {}
    final_time = (
        _finite_float(result.get("final_time_s"))
        or _finite_float(manifest.get("simulated_time_s"))
        or (max((_finite_float(event.get("time_s")) or 0.0 for event in events), default=0.0))
    )
    simulated_duration = max(0.0, final_time - initial_time)
    data = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "run_dir": str(run_path),
        "run_id": manifest.get("run_id"),
        "status": _status_value(manifest),
        "termination_reason": _termination_reason(manifest),
        "n_events": len(events),
        "simulated_time_s": final_time,
        "coverage": coverage,
        "directional_fluxes": _directional_fluxes(
            events,
            duration_s=simulated_duration,
        ),
        "products": products,
        "convergence": _convergence_rows(rate_blocks_path),
        "performance": _performance_metrics(performance_path),
        "artifacts": _artifact_rows(manifest),
        "warnings": warnings,
    }

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_path / "analysis"
    )
    ensure_directory(destination)
    markdown_path = destination / "report.md"
    html_path = destination / "report.html"
    with atomic_output_path(markdown_path) as temporary:
        temporary.write_text(_build_markdown(data), encoding="utf-8")
    with atomic_output_path(html_path) as temporary:
        temporary.write_text(_build_html(data), encoding="utf-8")
    data["outputs"] = {
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
    return data


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportError",
    "generate_run_report",
]
