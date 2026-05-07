# AutoKMC

AutoKMC builds surface kinetic Monte Carlo inputs and runs KMC simulations from
atomic structures. It is aimed at surface catalysis workflows where adsorption,
diffusion, bond-forming, bond-breaking, and thermochemistry data are generated
from ASE-compatible structures and calculators.

## What It Does

- Builds periodic slabs or nanoparticles from composition, structure, and size
  settings.
- Classifies surface atoms and constructs graph representations of the surface.
- Builds gas-phase reactants from SMILES strings.
- Enumerates and prunes adsorbate sites.
- Optionally enumerates diffusion and bond-changing reaction channels.
- Computes rates using electronic energies and optional vibrational
  free-energy corrections.
- Runs KMC trajectories and writes summaries, reaction records, and trajectory
  snapshots.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,cli]"
```

Optional ML calculator support is installed separately:

```bash
python -m pip install -e ".[ml]"
```

## Quick Start

Validate an example configuration:

```bash
python -m autokmc.cli.main validate-config example/co_cu111_allegro_no_diff.yaml
```

Run the full pipeline:

```bash
python -m autokmc.cli.main run example/co_cu111_allegro_no_diff.yaml
```

The CLI also exposes `run` and `validate-config` subcommands when installed as
an entry point.

## Configuration

Runs are controlled by YAML or TOML files. See the files in `example/` for
working starting points.

A typical config includes:

- `output`: output directory, filenames, trajectory cadence, and log level.
- `structure`: surface or nanoparticle construction settings.
- `reactants`: gas-phase species as SMILES strings.
- `calculator`: an ASE-compatible calculator or calculator factory.
- `adsorbate_sites`: adsorbate-site enumeration and stability pruning settings.
- `diffusion`: optional diffusion reaction enumeration and NEB settings.
- `bond`: optional bond-changing reaction enumeration and NEB settings.
- `free_energy`: optional vibrational thermochemistry settings.
- `kmc`: temperature, step count, RNG seed, and runtime KMC controls.

## Outputs

By default, run artifacts are written under the configured `output.dir`. The
pipeline writes a run summary, reaction records, trajectory snapshots, and
per-reaction folders containing relaxed structures and transition-state data
when those channels are enabled.

The exact filenames are configurable, but the default layout is organized
around:

- `summary.json`
- reaction records under `reactions/`
- trajectory snapshots
- optional vibration cache data

## Development

Run the test suite with:

```bash
python -m pytest
```

Useful project areas:

- `autokmc/structure`: slab and nanoparticle builders.
- `autokmc/species`: SMILES parsing and reactant construction.
- `autokmc/sites`: adsorption, diffusion, bond-site, and stability logic.
- `autokmc/reactions`: reaction objects and rate calculations.
- `autokmc/kmc`: KMC state, sampling, execution, and expansion.
- `autokmc/io`: config loading, calculator construction, persistence, and
  summaries.

## License

MIT

## Postscript
God forsook this repository long ago... we commit to main and say a prayer.
