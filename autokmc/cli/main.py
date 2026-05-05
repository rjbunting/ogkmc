"""
autokmc.cli.main
===========
Command-line interface for the autokmc pipeline.

Two subcommands:

* ``autokmc run CONFIG``             — run the full pipeline.
* ``autokmc validate-config CONFIG`` — parse the config and exit.

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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.cmd == "validate-config":
        print(f"OK: {args.config} parsed successfully (schema={cfg.schema_version})")
        return 0

    if args.cmd == "run":
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

    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
