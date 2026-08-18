# Development Guide

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cli,test,dev]"
```

These commands install the base development environment. Install
calculator-specific production dependencies, models, credentials, and GPU
runtimes separately.

## Checks

After installing the environment, run the local equivalents of CI:

```bash
python -m compileall -q autokmc
ruff check autokmc tests
mypy --follow-imports=skip \
  autokmc/io/_files.py \
  autokmc/io/config.py \
  autokmc/io/checkpoint.py \
  autokmc/io/calculation_cache.py \
  autokmc/io/calculators.py \
  autokmc/io/config_validation.py \
  autokmc/io/event_log.py \
  autokmc/io/persistence.py \
  autokmc/io/reaction_graph.py \
  autokmc/io/resume_contract.py \
  autokmc/io/trajectory.py \
  autokmc/analysis/products.py \
  autokmc/core/graph_state.py \
  autokmc/sites/identity.py \
  autokmc/kmc \
  autokmc/workflow \
  autokmc/utils/telemetry.py
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
| `autokmc/workflow` | Config-to-runtime preparation stages, network assembly, and run lifecycle. |
| `autokmc/kmc/models.py` | Typed `KMCRunRequest`, `KMCRunResult`, canonical resume state, channel options, and callback boundaries. |
| `autokmc/sites/identity.py` | Stable site/member handles used by indexes, restart, and dynamic deduplication. |
| `autokmc/kmc` | Typed session, rate index, sampling, state mutation, network expansion, and checkpoints. |
| `autokmc/io` | Config, calculators, events, structures, manifests, ISAAC records, and database lookup. |
| `autokmc/analysis` | Offline product-rate and mechanism reconstruction. |
| `autokmc/thermo` | Ideal-gas and harmonic thermochemistry. |

## Schema conventions

There are independent versioned contracts:

- config schema version 1,
- event schema version 3,
- reaction-document schema version 3,
- reaction-index schema version 1,
- summary schema version 3,
- trajectory-metadata schema version 1,
- run-manifest schema version 3,
- checkpoint schema version 4,
- AutoKMC reaction-database schema v2,
- ISAAC record version 1.05.

Bump only the contract that actually changes. Readers should explicitly
migrate a compatible older version or reject it with a useful error. Never
write non-standard JSON `NaN`/`Infinity` values.

## Persistence rules

- Use atomic replacement for complete JSON documents and checkpoints.
- Treat `events.jsonl` as append-only during execution. A checkpoint records
  its committed event count and byte offset; resume may truncate only the
  uncommitted crash tail beyond that boundary.
- Rebuild reaction-folder counters from the reconciled event prefix before
  appending a resumed run. Batch `reaction.json` statistics at checkpoints and
  writer close rather than rewriting them for every event.
- Store structures as `.extxyz` assets and checksum database assets.
- Keep SQLite disposable and rebuildable from verified records.
- Carry the run UUID through events, checkpoints, trajectories, manifests, and
  newly generated calculation records.
- Persist a scientific-config fingerprint in every new checkpoint. Resume may
  change only `kmc.n_steps`, `kmc.log_every`, `output.log_level`, and the four
  `checkpoint.*` lifecycle fields; all scientific settings must match.

## Scientific integrity rules

- A calculator request must return finite energies.
- Required optimizations and NEBs must converge.
- Expected endpoint/NEB instability may invalidate one reaction class;
  unexpected exceptions must propagate.
- Both directions of a reversible process must derive barriers from a common
  effective transition-state level.
- Free-energy caches must include every setting that can alter the result and
  must be isolated by species/process.
- Portable reaction-database fallback must match the normalized scientific
  inputs and calculator/model digest as well as labelled graph topology.
  Structure-bearing results additionally require the exact input coordinate
  frame; never return persisted structures in a translated, rotated, wrapped,
  or permuted query frame without a verified mapping. Exact-key lookup remains
  the fast path. Records missing current compatibility evidence remain readable
  by an explicitly known exact key but are not eligible for portable fallback.

## Adding a config option

1. Add it to the appropriate dataclass.
2. Add strict type/range/enum validation in `autokmc/io/config_validation.py`.
3. Resolve it in the relevant `autokmc/workflow` stage and pass it to the
   consuming scientific function.
4. Include it in cache compatibility parameters if it can change a computed
   structure or energy.
5. Add YAML/TOML parsing and invalid-value tests.
6. Update [Configuration Reference](configuration.md).

## Adding persisted event data

First update the event transition and record model. Then update resume
reconstruction, when relevant, and the analysis reader. If compatibility
breaks, increment the affected schema version. Finally, add round-trip tests.
Product and mechanism tracking should remain offline unless the scientific
model itself changes.
