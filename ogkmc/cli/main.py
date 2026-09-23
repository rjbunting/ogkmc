"""
ogkmc.cli.main
===========
Command-line interface for the ogkmc pipeline.

Seven subcommands:

* ``ogkmc run CONFIG``             — run the full pipeline.
* ``ogkmc validate-config CONFIG`` — parse the config and exit.
* ``ogkmc preflight CONFIG``       — check run readiness without writing.
* ``ogkmc doctor [CONFIG]``        — inspect runtime/package readiness.
* ``ogkmc analyze RUN_DIR``        — analyze products and mechanisms.
* ``ogkmc report RUN_DIR``         — write Markdown and HTML run reports.
* ``ogkmc rebuild-index DB_DIR``   — rebuild the reaction database index.

The CLI is **calculator-agnostic** — see :class:`ogkmc.io.config.CalculatorCfg`
for the dynamic loading scheme that supports VASP, CP2K, EMT, NequIP, MACE,
or any other ASE-compatible calculator.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from ogkmc import __version__ as ogkmc_version


# Keep console-script import and ``doctor`` usable when a scientific dependency
# is missing or has a broken binary installation.  Command implementations are
# imported only after ``main`` installs its friendly exception boundary.  These
# named wrappers also remain convenient monkeypatch seams for callers/tests.
def load_config(path: str | Path) -> Any:
    from ogkmc.io.config import load_config as implementation

    return implementation(path)


def preflight_config(cfg: Any, **kwargs: Any) -> dict[str, Any]:
    from ogkmc.cli.diagnostics import preflight_config as implementation

    return implementation(cfg, **kwargs)


def doctor_report(cfg: Any = None, **kwargs: Any) -> dict[str, Any]:
    from ogkmc.cli.diagnostics import doctor_report as implementation

    return implementation(cfg, **kwargs)


def run_from_config(cfg: Any, **kwargs: Any) -> dict[str, Any]:
    from ogkmc.cli.pipeline import run_from_config as implementation

    return implementation(cfg, **kwargs)


def analyze_run(run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    from ogkmc.analysis.products import analyze_run as implementation

    return implementation(run_dir, **kwargs)


def generate_run_report(run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    from ogkmc.analysis.run_report import generate_run_report as implementation

    return implementation(run_dir, **kwargs)


def rebuild_calculation_index(database_dir: str | Path) -> int:
    from ogkmc.io.calculation_cache import (
        rebuild_calculation_index as implementation,
    )

    return implementation(database_dir)


def _package_version() -> str:
    if ogkmc_version:
        return ogkmc_version
    try:
        return version("ogkmc")
    except PackageNotFoundError:
        return "0+unknown"

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ogkmc", description="OGKMC — Online Graph Kinetic Monte Carlo"
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="show Python tracebacks instead of concise CLI errors",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline from a config file")
    p_run.add_argument("config", help="path to a .yaml/.yml/.toml config")

    p_val = sub.add_parser("validate-config", help="parse a config and exit 0/1")
    p_val.add_argument("config", help="path to a .yaml/.yml/.toml config")

    p_preflight = sub.add_parser(
        "preflight",
        help="check config, outputs, checkpoint, and calculator readiness",
    )
    p_preflight.add_argument("config", help="path to a .yaml/.yml/.toml config")
    p_preflight.add_argument(
        "--check-calculator",
        action="store_true",
        help="construct the calculator and require finite probe energy/forces",
    )

    p_doctor = sub.add_parser(
        "doctor",
        help="report runtime and package readiness without running chemistry",
    )
    p_doctor.add_argument(
        "config",
        nargs="?",
        help="optional config to include in the readiness checks",
    )

    p_analyze = sub.add_parser(
        "analyze", help="post-process products, rates, and mechanisms from a run"
    )
    p_analyze.add_argument("run_dir", help="completed OGKMC run directory")
    p_analyze.add_argument(
        "--manifest",
        default="run_manifest.json",
        dest="manifest_filename",
        help="manifest filename relative to RUN_DIR, or an absolute path",
    )
    p_analyze.add_argument("--output-dir", default=None)
    p_analyze.add_argument("--start-time", type=float, default=None, dest="start_time_s")
    p_analyze.add_argument("--end-time", type=float, default=None, dest="end_time_s")
    p_analyze.add_argument("--blocks", type=int, default=10, dest="n_blocks")
    p_analyze.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="retain unknown lineage roots instead of rejecting an inconsistent log",
    )

    p_report = sub.add_parser(
        "report",
        help="generate Markdown and HTML reports from a persisted run",
    )
    p_report.add_argument("run_dir", help="completed OGKMC run directory")
    p_report.add_argument(
        "--manifest",
        default="run_manifest.json",
        dest="manifest_filename",
        help="manifest filename relative to RUN_DIR, or an absolute path",
    )
    p_report.add_argument("--output-dir", default=None)
    p_report.add_argument("--blocks", type=int, default=10, dest="n_blocks")
    p_report.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="retain unknown lineage roots in refreshed product analysis",
    )
    p_report.add_argument(
        "--no-refresh-analysis",
        action="store_false",
        dest="refresh_analysis",
        help="reuse persisted analysis instead of refreshing it from events",
    )

    p_rebuild = sub.add_parser(
        "rebuild-index", help="rebuild a reaction database SQLite index from ISAAC records"
    )
    p_rebuild.add_argument("database_dir", help="calculation_cache directory")

    return p


def _print_preflight(result: dict) -> None:
    checkpoint = result["checkpoint"]
    calculator = result["calculator"]
    structure = result.get("structure", {})
    print(f"OK: preflight passed for {result['output_dir']}")
    print(f"  mode: {checkpoint['mode']}")
    if structure.get("kind") == "file":
        format_label = structure.get("format") or "<auto>"
        print(
            f"  structure: {structure['path']} "
            f"(index={structure['index']}, "
            f"atoms={structure['atom_count']}, "
            f"frozen={structure['frozen_count']}, "
            f"format={format_label})"
        )
    if checkpoint.get("resume_from"):
        print(
            f"  checkpoint: {checkpoint['resume_from']} "
            f"(step={checkpoint.get('checkpoint_step')}, "
            f"time={checkpoint.get('checkpoint_time_s')} s)"
        )
    devices = calculator["gpu_devices"] or ["<unspecified>"]
    print(
        f"  calculator: {calculator['target']} "
        f"(copies={calculator['copies']}, "
        f"workers={calculator['max_workers']}, "
        f"devices={','.join(devices)})"
    )
    check = calculator.get("check")
    if check:
        print(
            "  calculator check: "
            f"energy={check['energy_ev']:.8g} eV, "
            f"max_force={check['max_force_ev_per_angstrom']:.8g} eV/Å"
        )
    for warning in result["warnings"]:
        print(f"  warning: {warning}")


def _print_doctor(result: dict) -> None:
    runtime = result["runtime"]
    print(
        "OGKMC doctor: "
        f"{result['status']} "
        f"(Python {runtime['python']}, {runtime['platform']})"
    )
    for name, package_version in result["packages"].items():
        rendered = package_version if package_version is not None else "MISSING"
        print(f"  {name}: {rendered}")
    for issue in result["issues"]:
        print(f"  issue: {issue}")
    if "preflight" in result:
        _print_preflight(result["preflight"])


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "validate-config":
        cfg = load_config(args.config)
        print(f"OK: {args.config} parsed successfully (schema={cfg.schema_version})")
        return 0

    if args.cmd == "preflight":
        cfg = load_config(args.config)
        result = preflight_config(
            cfg,
            config_path=str(Path(args.config).resolve()),
            check_calculator=args.check_calculator,
        )
        _print_preflight(result)
        return 0

    if args.cmd == "doctor":
        cfg = load_config(args.config) if args.config else None
        result = doctor_report(
            cfg,
            config_path=(
                None if args.config is None else str(Path(args.config).resolve())
            ),
        )
        _print_doctor(result)
        return 0 if result["status"] == "ok" else 2

    if args.cmd == "run":
        cfg = load_config(args.config)
        summary = run_from_config(cfg, config_path=str(Path(args.config).resolve()))
        outputs = summary.get("outputs", {})
        n_unique = outputs.get("n_unique_reactions", 0)
        out_dir = (
            Path(outputs.get("events", "")).parent
            if outputs.get("events")
            else Path(cfg.output.dir)
        )
        print(f"\n[ogkmc] Run complete — output: {out_dir}")
        print(
            f"  steps={summary.get('steps_executed')}  "
            f"t={summary.get('time')!s} s  "
            f"events={summary.get('reaction_counts')}  "
            f"unique reactions discovered={n_unique}"
        )
        return 0

    if args.cmd == "analyze":
        result = analyze_run(
            args.run_dir,
            manifest_filename=args.manifest_filename,
            output_dir=args.output_dir,
            start_time_s=args.start_time_s,
            end_time_s=args.end_time_s,
            n_blocks=args.n_blocks,
            strict=not args.allow_incomplete,
        )
        products = result.get("products", [])
        print(
            f"[ogkmc] Analysis complete — {len(products)} product species, "
            f"Δt={result.get('duration_s')} s"
        )
        for product in products:
            print(
                f"  {product['product']}: count={product['count']} "
                f"rate={product['rate_hz']:.6g} Hz"
            )
        print(f"  output: {Path(result['outputs']['summary']).parent}")
        return 0

    if args.cmd == "report":
        result = generate_run_report(
            args.run_dir,
            manifest_filename=args.manifest_filename,
            output_dir=args.output_dir,
            n_blocks=args.n_blocks,
            strict=not args.allow_incomplete,
            refresh_analysis=args.refresh_analysis,
        )
        print("[ogkmc] Report complete")
        print(f"  Markdown: {result['outputs']['markdown']}")
        print(f"  HTML: {result['outputs']['html']}")
        return 0

    if args.cmd == "rebuild-index":
        count = rebuild_calculation_index(args.database_dir)
        print(f"[ogkmc] Rebuilt reaction database index with {count} valid record(s)")
        return 0

    raise RuntimeError(f"unknown command {args.cmd!r}")


_EXPECTED_CLI_ERROR_NAMES = frozenset(
    {
        "AnalysisError",
        "CalculatorConfigError",
        "ConfigError",
        "OutputCollisionError",
        "PreflightError",
        "ReportError",
        "RunLockError",
        "SpeciesExpansionError",
        "StructureInputError",
    }
)


def _is_expected_cli_error(exc: Exception) -> bool:
    """Classify user/actionable failures without importing command modules."""
    return isinstance(exc, (FileNotFoundError, ImportError, PermissionError)) or (
        type(exc).__name__ in _EXPECTED_CLI_ERROR_NAMES
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("ogkmc: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        if _is_expected_cli_error(exc):
            print(f"ogkmc: error: {exc}", file=sys.stderr)
            return 2
        message = str(exc).strip() or "no additional detail"
        print(
            f"ogkmc: error: {type(exc).__name__}: {message}",
            file=sys.stderr,
        )
        print(
            "Run again with --debug before the subcommand for a traceback.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
