# AutoKMC Documentation

AutoKMC constructs and runs graph-based surface kinetic Monte Carlo models
from atomic structures and SMILES reactants. The workflow can enumerate
adsorption, desorption, diffusion, bond formation, and bond dissociation;
calculate endpoint and transition-state energetics; run KMC; and save enough
event and structural information for reproducible post-processing.

## Documentation map

- [Architecture and scientific workflow](architecture.md): pipeline stages,
  reaction-network expansion, rate evaluation, and failure behavior.
- [Configuration reference](configuration.md): every YAML/TOML section,
  defaults, accepted values, and validation rules.
- [Outputs, restart, and analysis](outputs-and-analysis.md): run artifacts,
  event schema, trajectories, checkpoint continuation, product rates, and
  backward-propagated mechanisms.
- [ISAAC reaction database](reaction-database.md): record layout, `.extxyz`
  assets, geometry/model-aware matching rules, and index recovery.
- [Development guide](development.md): package layout, tests, static checks,
  and schema-change conventions.

The repository [README](../README.md) remains the installation and quick-start
entry point. The production-style palladium example is:

- [`h2_oxidation_pd111_uma.yaml`](../example/h2_oxidation_pd111_uma.yaml)

## Command summary

```bash
autokmc validate-config CONFIG.yaml
autokmc preflight CONFIG.yaml
autokmc doctor [CONFIG.yaml]
autokmc run CONFIG.yaml
autokmc analyze RUN_DIR
autokmc report RUN_DIR
autokmc rebuild-index CALCULATION_CACHE_DIR
```

Use `autokmc COMMAND --help` for the current command-line arguments. Running
`python -m autokmc.cli ...` is equivalent to the installed `autokmc` command.
`preflight` performs read-only run-safety and calculator-import checks;
`--check-calculator` adds a finite energy/force probe. `doctor` reports
runtime/package readiness without chemistry. Expected failures are concise;
put the global `--debug` option before the command to show a traceback.

## Reproducibility model

AutoKMC separates three kinds of state:

1. The run directory records what the KMC trajectory actually did.
2. The checkpoint records the live simulation state needed to continue that
   same trajectory, including the random-number-generator state.
3. The reaction database records reusable scientific calculations and their
   checksum-verified structures.

Product identity, product rates, and mechanisms are not maintained in the
live KMC state. They are reconstructed from `run_manifest.json` and the
append-only `events.jsonl` log after the run.
