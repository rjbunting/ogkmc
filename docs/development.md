# Development Guide

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cli,test,dev]"
```

Calculator-specific production dependencies, models, credentials, and GPU
runtimes are not installed by the base package.

## Checks

The local equivalents of CI are:

```bash
python -m compileall -q autokmc
ruff check autokmc tests
mypy --follow-imports=skip \
  autokmc/io/config.py \
  autokmc/io/checkpoint.py \
  autokmc/analysis/products.py
pytest --cov=autokmc --cov-report=term-missing --cov-fail-under=50
```

CI runs these checks on Python 3.10, 3.11, 3.12, and 3.13. Scientific
regressions should include focused tests, especially for restart persistence,
free-energy/electronic rate selection, database verification, and event-log
analysis.

## Package map

| Package | Responsibility |
| --- | --- |
| `autokmc/structure` | Slab/nanoparticle construction, surface tagging, and relaxation. |
| `autokmc/species` | SMILES handling, molecular structures, and bond chemistry. |
| `autokmc/sites` | Adsorption, diffusion, bond sites, lateral classes, and stability. |
| `autokmc/reactions` | Applicability, electronic/free energetics, and rates. |
| `autokmc/kmc` | Rate index, sampling, state mutation, network expansion, and checkpoints. |
| `autokmc/io` | Config, calculators, events, structures, manifests, ISAAC records, and database lookup. |
| `autokmc/analysis` | Offline product-rate and mechanism reconstruction. |
| `autokmc/thermo` | Ideal-gas and harmonic thermochemistry. |

## Schema conventions

There are independent versioned contracts:

- config schema version 1,
- event/persistence schema version 2,
- run-manifest schema version 2,
- checkpoint schema version 2,
- AutoKMC reaction-database schema v1,
- ISAAC record version 1.05.

Bump only the contract that actually changes. Readers should explicitly
migrate a compatible older version or reject it with a useful error. Never
write non-standard JSON `NaN`/`Infinity` values.

## Persistence rules

- Use atomic replacement for complete JSON documents and checkpoints.
- Treat `events.jsonl` as append-only.
- Restore counters before appending a resumed run.
- Store structures as `.extxyz` assets and checksum database assets.
- Keep SQLite disposable and rebuildable from verified records.
- Carry the run UUID through events, checkpoints, trajectories, manifests, and
  newly generated calculation records.

## Scientific integrity rules

- A calculator request must return finite energies.
- Required optimizations and NEBs must converge.
- Expected endpoint/NEB instability may invalidate one reaction class;
  unexpected exceptions must propagate.
- Both directions of a reversible process must derive barriers from a common
  effective transition-state level.
- Free-energy caches must include every setting that can alter the result and
  must be isolated by species/process.
- Geometry-insensitive reaction-graph fallback is intentional and should not
  be tightened without changing the scientific method and database contract.

## Adding a config option

1. Add it to the appropriate dataclass.
2. Add strict type/range/enum validation in `autokmc/io/config.py`.
3. Thread it through the pipeline to the consuming function.
4. Include it in cache compatibility parameters if it can change a computed
   structure or energy.
5. Add YAML/TOML parsing and invalid-value tests.
6. Update [Configuration Reference](configuration.md).

## Adding persisted event data

Update the event transition/record model, resume reconstruction if relevant,
analysis reader, schema version when compatibility breaks, and round-trip
tests. Product and mechanism tracking should remain offline unless the
scientific model itself changes.
