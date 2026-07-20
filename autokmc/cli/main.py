"""
autokmc.cli.main
===========
Command-line interface for the autokmc pipeline.

Four subcommands:

* ``autokmc run CONFIG``             — run the full pipeline.
* ``autokmc validate-config CONFIG`` — parse the config and exit.
* ``autokmc analyze RUN_DIR``        — analyze products and mechanisms.
* ``autokmc rebuild-index DB_DIR``   — rebuild the reaction database index.

The CLI is **calculator-agnostic** — see :class:`autokmc.io.config.CalculatorCfg`
for the dynamic loading scheme that supports VASP, CP2K, EMT, NequIP, MACE,
or any other ASE-compatible calculator.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from autokmc.cli.pipeline import run_from_config
from autokmc.io.config import load_config
from autokmc.analysis.products import analyze_run
from autokmc.io.calculation_cache import rebuild_calculation_index


def _package_version() -> str:
    try:
        return version("autokmc")
    except PackageNotFoundError:
        return "0+unknown"

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autokmc", description="autokmc CLI")
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline from a config file")
    p_run.add_argument("config", help="path to a .yaml/.yml/.toml config")

    p_val = sub.add_parser("validate-config", help="parse a config and exit 0/1")
    p_val.add_argument("config", help="path to a .yaml/.yml/.toml config")

    p_analyze = sub.add_parser(
        "analyze", help="post-process products, rates, and mechanisms from a run"
    )
    p_analyze.add_argument("run_dir", help="completed AutoKMC run directory")
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

    p_rebuild = sub.add_parser(
        "rebuild-index", help="rebuild a reaction database SQLite index from ISAAC records"
    )
    p_rebuild.add_argument("database_dir", help="calculation_cache directory")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "validate-config":
        cfg = load_config(args.config)
        print(f"OK: {args.config} parsed successfully (schema={cfg.schema_version})")
        return 0

    if args.cmd == "run":
        cfg = load_config(args.config)
        summary = run_from_config(cfg, config_path=str(Path(args.config).resolve()))
        outputs = summary.get("outputs", {})
        n_unique = outputs.get("n_unique_reactions", 0)
        out_dir  = Path(outputs.get("events", "")).parent if outputs.get("events") else Path(cfg.output.dir)
        print(f"\n[autokmc] Run complete — output: {out_dir}")
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
            f"[autokmc] Analysis complete — {len(products)} product species, "
            f"Δt={result.get('duration_s')} s"
        )
        for product in products:
            print(
                f"  {product['product']}: count={product['count']} "
                f"rate={product['rate_hz']:.6g} Hz"
            )
        print(f"  output: {Path(result['outputs']['summary']).parent}")
        return 0

    if args.cmd == "rebuild-index":
        count = rebuild_calculation_index(args.database_dir)
        print(f"[autokmc] Rebuilt reaction database index with {count} valid record(s)")
        return 0

    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
