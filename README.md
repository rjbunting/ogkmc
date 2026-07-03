# AutoKMC

AutoKMC builds and runs surface kinetic Monte Carlo simulations from atomic
structures. It is designed for catalysis workflows where adsorption,
desorption, diffusion, bond-forming, bond-breaking, thermochemistry, and KMC
outputs should be generated from one configuration file.

The package uses ASE-compatible calculators, so the same workflow can run with
simple local calculators for smoke tests or with machine-learning potentials
for production studies.

## What AutoKMC Does

AutoKMC can:

- Build periodic slabs or nanoparticles.
- Build gas-phase reactants from SMILES strings.
- Enumerate possible adsorbate placements.
- Prune unstable adsorbate and bond-reaction sites using calculator
  relaxations.
- Discover adsorption, desorption, diffusion, and bond-changing elementary
  steps.
- Run endpoint relaxations and CI-NEB transition-state searches.
- Compute KMC rates from electronic energies, with optional vibrational
  free-energy corrections.
- Run KMC trajectories with on-the-fly reaction-network expansion.
- Write publication-oriented product rates and mechanism traces.
- Cache expensive optimization results as reloadable JSON.
- Export calculation records in the ISAAC AI-ready scientific record format.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cli,test]"
```

The `cli` extra installs YAML config support. The `test` extra installs the
test runner.

Optional machine-learning calculator packages are installed separately. For
example:

```bash
python -m pip install -e ".[ml]"
```

Production configs may also require calculator-specific packages, model files,
GPU drivers, or login tokens. AutoKMC does not bundle those external models.

## Command Line

After installation, the main commands are:

```bash
autokmc validate-config path/to/config.yaml
autokmc run path/to/config.yaml
```

Without installing the console entry point, use:

```bash
python -m autokmc.cli.main validate-config path/to/config.yaml
python -m autokmc.cli.main run path/to/config.yaml
```

Validation only checks that the configuration can be parsed. It does not run
the expensive chemistry workflow.

## First Run

For a first test, start with a small number of KMC steps, no diffusion, no bond
chemistry, and no free-energy corrections. This checks the installation,
surface construction, gas-phase reactant build, adsorption site enumeration,
and KMC loop before you spend time on NEB calculations.

Create a file such as `quickstart.yaml`:

```yaml
schema_version: "1"

output:
  dir: ./runs/quickstart
  trajectory_dump_every: 10
  log_level: INFO

structure:
  kind: surface
  composition: Cu
  crystal_structure: fcc
  miller_index: [1, 1, 1]
  min_slab_size: 8.0
  min_vacuum_size: 12.0
  n_freeze_layers: 2

reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
    relax_in_gas: true
    partial_pressure_bar: 1.0
    geometry: linear

calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}

adsorbate_sites:
  prune_stable_only: true
  fmax: 0.05
  max_steps: 200

free_energy:
  enabled: false

diffusion:
  enabled: false

bond:
  enabled: false

kmc:
  temperature_k: 500.0
  n_steps: 100
  transmission_coefficient: 1.0
  fmax: 0.05
  max_steps: 200
  log_every: 10
  random_seed: 7
  lateral_interactions: true
```

Then run:

```bash
autokmc validate-config quickstart.yaml
autokmc run quickstart.yaml
```

The example files in `example/` are larger FAIR-Chem UMA workflows intended as
production-style starting points. They require the relevant FAIR-Chem
installation, model access, and GPU environment.

## Configuration Overview

AutoKMC configs are YAML or TOML files. The main sections are:

- `output`: output directory, filenames, trajectory cadence, calculation cache,
  ISAAC export, and log level.
- `structure`: slab or nanoparticle construction.
- `reactants`: gas-phase species, SMILES strings, pressures, and gas
  thermochemistry metadata.
- `calculator`: ASE calculator construction.
- `adsorbate_sites`: adsorption-site enumeration and stability pruning.
- `diffusion`: optional diffusion hops and CI-NEB settings.
- `bond`: optional bond-changing templates and CI-NEB settings.
- `free_energy`: optional vibrational thermochemistry.
- `kmc`: temperature, step count, random seed, and runtime controls.
- `checkpoint`: optional restart checkpoints.

### Calculator Configuration

For a directly importable ASE calculator:

```yaml
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
```

For a factory-style calculator:

```yaml
calculator:
  factory: fairchem.core.FAIRChemCalculator.from_model_checkpoint
  factory_kwargs:
    name_or_path: uma-s-1p2
    task_name: oc20
    device: cuda
  copies: 4
  max_workers: 4
```

`copies` creates a calculator pool. This is useful for independent relaxation
or NEB tasks when the calculator and hardware can support parallel work.

### Diffusion

Enable diffusion only after adsorption/desorption runs are behaving:

```yaml
diffusion:
  enabled: true
  max_hops: 1
  n_shells_pair: 1
  prune_by_adsorption_pair: true
  fmax: 0.05
  max_steps: 500
  n_images: 8
  climb: true
  spring_k: 0.1
  interpolation: idpp
  persist_neb_path: true
```

Each new diffusion lateral class can trigger two endpoint relaxations and a
CI-NEB calculation. These are often among the most expensive parts of a run.

### Bond-Changing Reactions

Bond reactions are generated from the reactant SMILES and represent
`A + B <=> C` style chemistry:

```yaml
bond:
  enabled: true
  bond_max_hops: 1
  bond_types: ["SINGLE", "DOUBLE", "TRIPLE"]
  include_dissociation: true
  include_coupling: true
  include_homo_coupling: true
  auto_build_leaf_species: true

  prune_by_triple: true
  prune_with_calculator: true
  prune_fmax: 0.05
  prune_max_steps: 500

  neb_fmax: 0.05
  neb_max_steps: 500
  neb_n_images: 8
  neb_climb: true
  neb_spring_k: 0.1
  neb_interpolation: idpp
  atom_matching: auto
  matching_trials: 8
  persist_neb_path: true
```

`atom_matching` controls how atoms in the initial and final bond-reaction
endpoints are paired before NEB interpolation. The default `auto` tries several
reasonable same-element mappings and keeps the lowest-displacement path.
`hungarian` uses global same-element assignment directly. `greedy` and
`reactant_index` are useful comparison modes.

### Free Energy

Free-energy corrections use ASE vibrational analysis. They can be expensive,
so it is often best to turn them on after the reaction network is behaving:

```yaml
free_energy:
  enabled: true
  pressure_bar: 1.0
  vibration_displacement: 0.01
  vibration_nfree: 2
  include_ts_vibrations: true
```

Gas-phase reactants can declare pressure, symmetry number, spin, and geometry
under `reactants`.

## Outputs

By default, outputs are written under `output.dir`.

Important files:

- `events.jsonl`: one JSON record per accepted KMC event.
- `summary.json`: run metadata, reaction counts, and final occupancy.
- `kmc.extxyz`: trajectory snapshots at the configured cadence.
- `products.json`: product counts, rates, selectivity, and first/last times.
- `product_timeseries.csv`: cumulative product formation over KMC time.
- `product_episodes.jsonl`: one product event per line, with its causal chain.
- `mechanism_summary.json`: mechanism fingerprints grouped by product.
- `isaac_records.json`: ISAAC AI-ready scientific record bundle.
- `checkpoint.pkl`: restart state when checkpointing is enabled.

Reaction folders live under `reactions/`:

```text
reactions/
  adsorption/<species>/isoX_latY/
    reaction.json
    occupied.extxyz
    unoccupied.extxyz
  diffusion/<species>/diff_isoX_latY/
    reaction.json
    state_a.extxyz
    state_b.extxyz
    ts.extxyz
    neb_path.extxyz
  bond/<A+B<->C>/bond_isoX_latY/
    reaction.json
    state_ab.extxyz
    state_c.extxyz
    ts.extxyz
    neb_path.extxyz
```

`reaction.json` stores the energies, barriers, vibrational data when present,
calculator metadata, and references to the structures written in that folder.

## Product Rates And Mechanisms

The product outputs are intended for publication analysis.

AutoKMC currently treats a gas product event as either:

- desorption of a species that was not listed in the initial reactants, or
- a bond-coupling event whose `C` endpoint is an explicit gas product and is
  not one of the initial reactants.

`products.json` gives the production rate in Hz for each product species:

```text
rate_hz = product_count / total_kmc_time_s
```

`product_episodes.jsonl` preserves the mechanism for each product event. It
stores the lineage of the produced species and the sequence of events that led
to it, such as adsorption, diffusion, bond coupling, and desorption.

## Calculation Cache

Optimization and NEB calculations are usually the slowest part of a run.
AutoKMC writes reloadable cache records by default:

```text
calculation_cache/
  adsorption/<hash>.json
  diffusion/<hash>.json
  bond/<hash>.json
```

Each cache record contains:

- a deterministic key based on the calculation inputs and settings,
- relaxed endpoint structures embedded as JSON,
- energies and optional thermochemistry,
- transition-state and NEB path data when available,
- AutoKMC metadata needed to repopulate the lateral class,
- an ISAAC v1.05 record representation.

On a later run, AutoKMC tries to load a matching complete cache record before
running the expensive calculation again. If the geometry, parameters, matching
settings, or relevant inputs change, the key changes and AutoKMC recomputes.

Configure the cache with:

```yaml
output:
  calculation_cache_enabled: true
  calculation_cache_dir: calculation_cache
  isaac_export_filename: isaac_records.json
```

The ISAAC export follows the public
[ISAAC AI-ready scientific record](https://github.com/ISAAC-DOE/isaac-ai-ready-record)
v1.05 schema. Numerical quantities are written as ISAAC descriptors; embedded
AutoKMC structures are referenced as assets.

## Checkpoint Restart

Checkpointing saves the live KMC state, including graph occupancy and discovered
sites:

```yaml
checkpoint:
  enabled: true
  path: ./runs/my_run/checkpoint.pkl
  every_n_steps: 10
  # resume_from: ./runs/my_run/checkpoint.pkl
```

Use checkpoints to resume a stopped simulation. Use the calculation cache to
avoid repeating expensive optimization and NEB calculations. They solve
different problems and are useful together.

## Suggested Workflow

1. Start with a small surface and one reactant.
2. Run adsorption/desorption only.
3. Inspect `summary.json`, `events.jsonl`, and `reactions/`.
4. Enable diffusion with a small number of KMC steps.
5. Enable bond reactions and keep `persist_neb_path: true` while debugging.
6. Turn on free-energy corrections once the network looks reasonable.
7. Use `products.json` and `mechanism_summary.json` for publication tables.
8. Keep `calculation_cache/` with the run artifacts so results can be reused
   and audited later.

## Development

Run tests with:

```bash
python -m pytest
```

Useful package areas:

- `autokmc/structure`: slab and nanoparticle builders.
- `autokmc/species`: SMILES parsing and reactant construction.
- `autokmc/sites`: adsorption, diffusion, bond-site, and stability logic.
- `autokmc/reactions`: reaction objects and rate calculations.
- `autokmc/kmc`: KMC state, sampling, execution, and on-the-fly expansion.
- `autokmc/io`: config loading, calculator construction, persistence, product
  outputs, calculation cache, and ISAAC export.
- `autokmc/thermo`: gas and adsorbate thermochemistry helpers.

## Troubleshooting

- If config validation fails, check for unknown keys or indentation errors.
- If a run is very slow, lower `kmc.n_steps`, disable diffusion/bond reactions,
  or reuse an existing `calculation_cache/`.
- If a NEB path is poor, try `interpolation: idpp`, `atom_matching: auto`, and a
  larger `matching_trials`.
- If no reactions are available, inspect adsorption-site pruning and gas-phase
  reactant energies.
- If product rates are zero, confirm that the product is not listed as an
  initial reactant and that desorption or gas-product bond events can occur.

## License

MIT
