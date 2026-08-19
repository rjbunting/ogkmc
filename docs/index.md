# AutoKMC Documentation

AutoKMC constructs graph-based surface kinetic Monte Carlo models from atomic
structures and SMILES reactants. It first builds the catalyst, reactants, and
surface sites. It then enumerates adsorption, desorption, diffusion, bond
formation, and bond dissociation and calculates their required energetics.
Finally, it runs KMC and saves the event and structural information needed for
reproducible post-processing.

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

Start with the repository [README](../README.md) for installation and a first
run. Then use these production-style surface examples:

- [`h2_oxidation_pd111_uma.yaml`](../example/h2_oxidation_pd111_uma.yaml)
- [`h2_oxidation_pd100_uma.yaml`](../example/h2_oxidation_pd100_uma.yaml)
- [`co_adsorption_diffusion_cu111_uma.yaml`](../example/co_adsorption_diffusion_cu111_uma.yaml)

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
`python -m autokmc.cli ...` is equivalent to using the installed `autokmc`
command. First use `preflight` for read-only run-safety and calculator-import
checks. Add `--check-calculator` to require a finite energy/force probe. Use
`doctor` to report runtime and package readiness without running chemistry.
Expected failures are concise; put the global `--debug` option before the
command to show a traceback.

## Reproducibility model

AutoKMC separates three kinds of state, and each has a different purpose:

1. The run directory records what the KMC trajectory actually did.
2. The checkpoint records the live simulation state needed to continue that
   same trajectory, including the random-number-generator state.
3. The reaction database records reusable scientific calculations and their
   checksum-verified structures.

The live KMC state does not maintain product identity, product rates, or
mechanisms. After the run, AutoKMC reconstructs them from `run_manifest.json`
and the append-only `events.jsonl` log.
