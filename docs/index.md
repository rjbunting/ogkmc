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
  assets, matching rules, index recovery, and the intentional
  geometry-insensitive fallback.
- [Development guide](development.md): package layout, tests, static checks,
  and schema-change conventions.

The repository [README](../README.md) remains the installation and quick-start
entry point. The two production-style platinum examples are:

- [`co_oxidation_pt111_uma_4gpu.yaml`](../example/co_oxidation_pt111_uma_4gpu.yaml)
- [`co_oxidation_ptnano_uma_4gpu.yaml`](../example/co_oxidation_ptnano_uma_4gpu.yaml)

## Command summary

```bash
autokmc validate-config CONFIG.yaml
autokmc run CONFIG.yaml
autokmc analyze RUN_DIR
autokmc rebuild-index CALCULATION_CACHE_DIR
```

Use `autokmc COMMAND --help` for the current command-line arguments. Running
`python -m autokmc.cli ...` is equivalent to the installed `autokmc` command.

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
