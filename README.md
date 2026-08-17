# AutoKMC

AutoKMC prepares and runs surface kinetic Monte Carlo simulations from atomic
structures. It can build a slab or nanoparticle, or load any catalyst format
readable by ASE. It is designed for catalysis workflows where adsorption,
desorption, diffusion, bond-forming, bond-breaking, thermochemistry, and KMC
outputs should be generated from one configuration file.

The package uses ASE-compatible calculators, so the same workflow can run with
simple local calculators for smoke tests or with machine-learning potentials
for production studies.

## What AutoKMC Does

AutoKMC can:

- Build periodic slabs or nanoparticles, or load an existing atomic structure.
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
- Write a structured event log for downstream analysis.
- Reuse expensive optimization results from graph-searchable ISAAC records
  with checksum-verified `.extxyz` structures.
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
autokmc preflight path/to/config.yaml
autokmc doctor path/to/config.yaml
autokmc run path/to/config.yaml
autokmc analyze RUN_DIR
autokmc report RUN_DIR
autokmc rebuild-index CALCULATION_CACHE_DIR
```

Without installing the console entry point, use:

```bash
python -m autokmc.cli validate-config path/to/config.yaml
python -m autokmc.cli preflight path/to/config.yaml
python -m autokmc.cli doctor path/to/config.yaml
python -m autokmc.cli run path/to/config.yaml
python -m autokmc.cli analyze RUN_DIR
python -m autokmc.cli report RUN_DIR
python -m autokmc.cli rebuild-index CALCULATION_CACHE_DIR
```

Validation strictly checks types, ranges, reactant SMILES, bond types,
the required calculator construction declaration, worker/device consistency,
and managed output names. `preflight` additionally checks output collisions
and locks, checkpoint compatibility, file-backed catalyst readability,
calculator imports, and the effective copies/workers/devices without writing
configured outputs. Add
`--check-calculator` to construct the calculator in a temporary directory and
require a finite probe energy and force array. `doctor` reports Python,
required package, and optional config readiness without running chemistry.

Expected user/configuration failures are printed without a Python traceback
and return a stable nonzero status. Put the global `--debug` option before the
subcommand to re-enable tracebacks, for example
`autokmc --debug preflight CONFIG.yaml`.

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
  composition: Pt
  crystal_structure: fcc
  miller_index: [1, 1, 1]
  lattice_constant: 3.92
  # Four Pt(111) layers, repeated 3x3 in the primitive surface cell.
  min_slab_size: 8.0
  min_vacuum_size: 12.0
  goal_x: 8.0
  goal_y: 8.0
  extra_kwargs:
    orthogonalise: false
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
  anchor_k_max: 4

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
  log_every: 100
  random_seed: 7
  lateral_interactions: true
```

Then run:

```bash
autokmc validate-config quickstart.yaml
autokmc preflight quickstart.yaml --check-calculator
autokmc run quickstart.yaml
```

To use an existing catalyst instead, replace the `structure` section with:

```yaml
structure:
  kind: file
  path: ./structures/my-catalyst.extxyz
  index: -1
  # format: extxyz
  # frozen_indices: [0, 1, 2, 3]
```

Relative structure paths are resolved from the configuration file's directory,
not the shell's working directory. `format` is optional when ASE can infer it
from the filename, and `index: -1` selects the last frame. Use
`frozen_indices` when selected catalyst atoms must remain fixed. File-backed
catalysts are not rebuilt or relaxed. A periodic slab may use a skew cell and
may be arbitrarily rotated: AutoKMC rigidly rotates its detected surface normal
to Cartesian +z before site generation, without changing cell metrics or
interatomic geometry. The applied frame transform is recorded in the run
manifest. Provide the intended cell and periodic-boundary metadata for surface
classification.

[`example/all_options.yaml`](example/all_options.yaml) is a commented,
valid-as-written template with every configuration option and commented
alternatives for competing structure and calculator modes. The other example
files are larger FAIR-Chem UMA workflows intended as production-style starting
points. They require the relevant FAIR-Chem installation, model access, and
GPU environment.

## Documentation

The focused documentation is under [`docs/`](docs/index.md):

- [Architecture and scientific workflow](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Outputs, restart, and offline analysis](docs/outputs-and-analysis.md)
- [ISAAC reaction database](docs/reaction-database.md)
- [Development guide](docs/development.md)

The README gives the shortest path to a first run. The documentation pages are
the reference for complete option tables, persistence contracts, matching
semantics, and manuscript post-processing.

## Configuration Overview

AutoKMC configs are YAML or TOML files. The main sections are:

- `output`: output directory, filenames, trajectory cadence, reaction database,
  ISAAC export, and log level.
- `constants`: shared geometry, covalent-radius, site-search, isomorphism, and
  lateral-environment controls.
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

Every run must explicitly configure exactly one of `import_path` or `factory`.

For a directly importable ASE calculator:

```yaml
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
```

For a factory-style calculator:

```yaml
calculator:
  factory: fairchem.core.FAIRChemCalculator
  factory_kwargs:
    predict_unit:
      factory: autokmc.io.fairchem.get_predict_unit_on_device
      factory_kwargs:
        name_or_path: uma-s-1p2
        device: cuda
    task_name: oc20
  copies: 4
  gpu_devices: [cuda:0, cuda:1, cuda:2, cuda:3]
  gpu_device_arg: predict_unit.factory_kwargs.device
  max_workers: 4
```

`copies` creates a calculator pool. This is useful for independent relaxation
or NEB tasks when the calculator and hardware can support parallel work. Each
NEB leases exactly one calculator for its complete ordinary and climbing-image
lifecycle; calculator copies are never divided across images in one band.
The UMA example above creates one predictor per GPU so AutoKMC can run four
independent calculator tasks concurrently. FAIR-Chem's single-worker predictor
accepts `device: cuda`; `get_predict_unit_on_device` selects each configured
ordinal during construction and verifies that the predictor retained it.
The bundled FAIR-Chem device helper requires `workers: 1`, preserving the
one-calculator/one-device NEB policy.

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
  image_spacing: 0.25
  min_images: 6
  max_images: 8
  climb: true
  spring_k: 5.0
  interpolation: idpp
  persist_neb_path: true
```

Each new diffusion lateral class can trigger two endpoint relaxations, an
ordinary NEB relaxation, and then a climbing-image NEB refinement of the same
band. When either ordinary forward/reverse barrier is below the 0.1 eV KMC
floor, AutoKMC retains the ordinary band and skips the climbing stage. The rate
layer applies the floor through one common effective TS level so reversible
energy consistency is preserved. These calculations are often among the most
expensive parts of a run.

Effectively barrierless bands can oscillate above the strict channel `fmax`
because of their spring modes. Once the observed maximum NEB force is at or
below `optimization.neb_low_barrier_fmax` (default 0.1 eV/Å), AutoKMC checks
both raw directional barriers. If either is below 0.1 eV, the band is accepted
without reaching the strict force target. The reaction JSON records this as a
low-barrier early stop together with the observed force and both cutoffs.

When `image_spacing` (or bond `neb_image_spacing`) is set, it also guards the
optimized geometry: by default, no unfrozen atom may move more than three times
that distance between adjacent images. A violating step restores the lowest-force valid band
and starts a fresh optimizer inside the existing step budget. FIRE restarts
with halved `dt` and `dtmax`, preventing a stretched band from being retained.
Set `optimization.neb_geometry_guard_multiplier` to change the multiplier
without changing the image density.

With lateral interactions enabled, AutoKMC automatically uses the optimized
no-neighbour path as the initial band for a diffusion class containing a
neighbouring adsorbate. If that bare path has not been calculated yet, the bare
calculation runs first. The complete lateral band is still reoptimized, and a
failed or incompatible bare path falls back to the configured `interpolation`.

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
  neb_image_spacing: 0.25
  neb_min_images: 6
  neb_max_images: 8
  neb_climb: true
  neb_spring_k: 5.0
  neb_interpolation: idpp
  atom_matching: auto
  matching_trials: 8
  persist_neb_path: true
```

`atom_matching` controls how atoms in the initial and final bond-reaction
endpoints are paired before NEB interpolation. The default `auto` tries several
reasonable same-element mappings and keeps the lowest-displacement path.
With the default non-null image spacing, AutoKMC chooses the interior-image
count dynamically from the largest MIC-aware corresponding-atom displacement
between relaxed endpoints. The configured `n_images`/`neb_n_images` values are
fixed-count fallbacks used when the corresponding spacing is `null`.
`hungarian` uses global same-element assignment directly. `greedy` and
`reactant_index` are useful comparison modes.

Bond NEBs use the same automatic bare-first initialization as diffusion NEBs:
the no-neighbour lateral class is calculated on demand and its optimized band
seeds classes with neighbouring adsorbates. No additional configuration key is
required.

### Free Energy

Free-energy corrections use ASE vibrational analysis. They can be expensive,
so it is often best to turn them on after the reaction network is behaving:

```yaml
free_energy:
  enabled: true
  # Default partial pressure for reactants that omit partial_pressure_bar.
  pressure_bar: 1.0
  vibration_displacement: 0.01
  vibration_nfree: 2
  include_ts_vibrations: true
  symmetry_tolerance: 0.3
```

`free_energy.pressure_bar` is the feed-wide default partial pressure, including
when free-energy corrections are disabled. A reactant-level
`partial_pressure_bar` overrides it. Gas thermochemistry is always evaluated at
the fixed 1-bar standard state; the resolved partial pressure instead scales
the adsorption rate as the ideal-gas activity `p / p°`.

Gas-phase rotational symmetry numbers are inferred from the final molecular
coordinates with pymatgen and recorded with the detected point group in the
thermochemistry metadata and run manifest. Reactants can still declare
`symmetry_number` as an explicit override. Pressure, spin, and geometry remain
configurable under `reactants`; spin cannot generally be inferred from
molecular geometry.

## Outputs

By default, outputs are written under `output.dir`.

Important files:

- `events.jsonl`: one JSON record per accepted KMC event.
- `reactions/index.jsonl`: stable reaction ids and static network definitions
  referenced by compact event rows.
- `run_manifest.json`: a run UUID, the exact input text and resolved config,
  feed species, initial state, clock origin, and catalyst normalization
  metadata needed for reproducible analysis.
- `summary.json`: run metadata, reaction counts, final occupancy, and a
  concise run-scoped performance summary under `run.performance`.
- `diagnostics/performance.json`: versioned raw run telemetry and the matching
  concise performance summary.
- `kmc.extxyz`: trajectory snapshots with graph node/species/site identities
  and a per-atom frozen mask at the configured cadence, plus a guaranteed
  nonduplicate final frame.
- `diagnostics/invalid_adsorption/`: initial and last-known optimized structures
  for adsorption candidates pruned by the MLIP.
- `diagnostics/invalid_diffusion/`: failed diffusion endpoints and NEB bands
  kept outside the authoritative reaction network.
- `diagnostics/invalid_bond/`: failed bond-reaction endpoints and NEB bands.
- `isaac_records.json`: optional ISAAC AI-ready scientific record bundle.
- `checkpoint.pkl`: restart state when checkpointing is enabled.

A fresh configured run refuses to start when its managed output artifacts
already exist, so event logs, manifests, summaries, checkpoints, reaction
folders, and diagnostics are never silently mixed or overwritten. Choose a
new `output.dir`, or configure `checkpoint.resume_from` to continue the same
run. AutoKMC also holds a filesystem lock on `output.dir` for the entire run,
preventing two processes from writing there concurrently.

Reaction folders live under `reactions/`:

```text
reactions/
  index.jsonl
  adsorption/<species>/isoX_latY/
    reaction.json
    occupied_initial.extxyz
    occupied.extxyz
    unoccupied_initial.extxyz
    unoccupied.extxyz
  diffusion/<species>/diff_isoX_latY/
    reaction.json
    state_a_initial.extxyz
    state_a.extxyz
    state_b_initial.extxyz
    state_b.extxyz
    ts.extxyz
    neb_path_initial.extxyz       # when persist_neb_path is true
    neb_path.extxyz               # when persist_neb_path is true
  bond/<A+B<->C>/bond_isoX_latY/
    reaction.json
    state_ab_initial.extxyz
    state_ab.extxyz
    state_c_initial.extxyz
    state_c.extxyz
    state_c_gas_reference.extxyz # empty surface only, gas products
    gas_molecule.extxyz          # optimized gas molecule only, gas products
    ts.extxyz
    neb_path_initial.extxyz       # when persist_neb_path is true
    neb_path.extxyz               # when persist_neb_path is true
```

`reaction.json` stores the energies, barriers, vibrational data when present,
calculator metadata, and references to the structures written in that folder.
The `*_initial.extxyz` endpoint files are the structures before relaxation.
For gas-product bond reactions, `state_c_gas_reference.extxyz` is the relaxed
empty surface/lateral environment with no molecule in the vacuum, while
`gas_molecule.extxyz` is the separately optimized gas molecule. Recomputing
their independent energies reproduces the additive thermodynamic C state.
For diffusion and bond reactions, `neb_path_initial.extxyz` is the interpolated
band before NEB optimization and `neb_path.extxyz` is the optimized band.
Failed diffusion and bond candidates retain both files automatically, even
when `persist_neb_path` is false, with the latter containing the last-known
band at the point of failure.

## Post-Processing Events

Product rates and mechanisms are intentionally calculated after KMC. Event
schema v3 records compact dynamic rows with deterministic `event_id` and
stable `reaction_id` values, plus the canonical gas/surface inputs and outputs
of every fired reaction. Static descriptions, templates, reaction folders,
and gas-product flags live once in `reactions/index.jsonl`; built-in analysis
resolves them automatically. Expanded schema-v2 logs remain readable. The live
KMC graph does not store product lineage.

Analyze a completed run with:

```bash
autokmc analyze runs/h2_oxidation_pd111_uma
# Optional stationary-state window:
autokmc analyze RUN_DIR --start-time 1.0e-4 --end-time 5.0e-4 --blocks 20
# If output.run_manifest_filename was customized:
autokmc analyze RUN_DIR --manifest custom_manifest.json
```

Generate a readable report after the run with:

```bash
autokmc report RUN_DIR
autokmc report RUN_DIR --blocks 20 --output-dir RUN_DIR/analysis
autokmc report RUN_DIR --no-refresh-analysis
```

The report command writes `report.md` and a self-contained `report.html`.
By default it refreshes product/mechanism analysis from the event log first;
`--no-refresh-analysis` reuses persisted analysis files.

A product is defined strictly as a species not listed among the feed reactants
that leaves an occupied surface placement through a desorption event. For each
product desorption, the analyzer follows the consumed surface placement
backward through the events that formed it. Diffusion is collapsed, bond
formation branches into both precursor histories, and immediate reversible
bond recrossings are removed from the reported chemical mechanism.

The observed product rate is the number of product desorptions divided by the
selected KMC time window. It is not the microscopic `rate_hz` propensity on an
individual event. The analyzer writes:

```text
analysis/
  analysis_summary.json
  product_rates.csv
  product_events.jsonl
  mechanisms.csv
  rate_blocks.csv
```

`product_rates.csv` includes the per-simulation rate, an exact two-sided 95%
Garwood Poisson counting interval, and the rate normalized by the number of
classified surface atoms.
`mechanisms.csv` groups identical back-propagated mechanisms and reports their
counts, fractions, and observed rates. Logs written before event schema v2 do
not contain enough surface-state information for this analysis and must be
rerun.

## Reaction Database

Optimization and NEB calculations are usually the slowest part of a run.
AutoKMC writes graph-searchable reaction records by default:

```text
calculation_cache/
  index.sqlite3
  records/<record_id>/
    isaac_record.json
    reaction_graph.json
    occupied.extxyz         # adsorption
    unoccupied.extxyz
    state_a.extxyz          # diffusion
    state_b.extxyz
    state_ab.extxyz         # bond
    state_c.extxyz
    ts.extxyz               # diffusion and bond
    neb_path.extxyz         # optional
```

Each record contains:

- one ISAAC v1.05 evidence record as the source of truth,
- a portable labelled reaction graph used for database matching,
- required `.extxyz` endpoint and transition-state assets,
- SHA-256 checksums for every graph and structure asset,
- endpoint energies, barriers, and optional thermochemistry as ISAAC
  descriptors,
- method and optimizer/NEB settings used to establish compatibility,
- an optional multi-frame `neb_path.extxyz` when the path was retained.

On a later run, AutoKMC first checks the exact calculation key. It can then
search the SQLite index for the same reaction, calculator/settings, and graph
fingerprint. A candidate is accepted only after full labelled graph
isomorphism, normalized scientific-input comparison, calculator/model digest
comparison, and checksum verification. Atomic inputs use a geometry
fingerprint that is invariant to rigid translation, rotation, periodic
wrapping, and atom order but rejects strain and changes to initial charges,
magnetic moments, tags, custom atom arrays, or the full lattice metric; every
other scientific input, including scalar gas energies and charge, remains in
the identity. Because
relaxed structures and NEB paths are stored in the original coordinate frame,
those outputs are reused only when the query's exact input coordinates, cell,
periodic images, and atom ordering also match. Model checkpoint files are
matched by SHA-256 content rather than local path. Run-local node ids,
`iso_class`, and `lateral_class` numbers are excluded explicitly; chemical
identity, endpoint roles, elements, bond roles, topology, and all other inputs
are retained. Model files and directory-valued artifacts are rehashed from
their contents on every identity calculation. A missing, unreadable, modified,
or scientifically incompatible asset rejects the hit and the calculation is
recomputed.

If `index.sqlite3` is missing or corrupt, AutoKMC rebuilds it from verified
ISAAC record folders. You can also do this explicitly with
`autokmc rebuild-index CALCULATION_CACHE_DIR`.

The loaded primitive energies and structures repopulate the lateral class.
AutoKMC still computes the rate for the current KMC conditions, so temperature-
and pressure-dependent rates are not frozen into the database.

Configure the reaction database with:

```yaml
output:
  calculation_cache_enabled: true
  calculation_cache_lookup_enabled: false
  calculation_cache_dir: calculation_cache
  isaac_export_enabled: false
  isaac_export_filename: isaac_records.json
```

Set `isaac_export_enabled: true` to write the aggregate export. The calculation
cache remains enabled independently. Cache lookup is off by default, so new
database records are written for later ISAAC upload without first checking for
reusable records. Set `calculation_cache_lookup_enabled: true` to opt into
reuse. The ISAAC export follows the public
[ISAAC AI-ready scientific record](https://github.com/ISAAC-DOE/isaac-ai-ready-record)
v1.05 schema. Numerical quantities are written as ISAAC descriptors and
structures are external assets rather than embedded JSON. The configured
`isaac_export_filename` names a JSON array in which every element is one complete
ISAAC record; keep the corresponding `calculation_cache/records/` folders with
the export so its relative asset URIs remain reproducible.

## Checkpoint Restart

Checkpointing saves the live KMC state, including graph occupancy and discovered
sites, reaction counts, history, and random-number-generator state:

```yaml
checkpoint:
  enabled: true
  path: ./runs/my_run/checkpoint.pkl
  every_n_steps: 10
  # resume_from: ./runs/my_run/checkpoint.pkl
```

Use checkpoints to resume a stopped simulation. Use the reaction database to
avoid repeating expensive optimization and NEB calculations. They solve
different problems and are useful together.

Resume into the same `output.dir`. AutoKMC appends `events.jsonl` and
`kmc.extxyz`, restores reaction-folder counters and the RNG stream, rebuilds
the cumulative summary from the event log, and adds a continuation segment to
the existing run manifest. New checkpoints also verify that the structure,
calculator, feed, thermochemistry, and reaction-channel configuration is
unchanged. Resolvable calculator files/directories are compared by content,
and the contract records the AutoKMC source digest plus calculator-package
versions, so an in-place code or model change is rejected as well. Only the
additional step count, logging controls, and checkpoint lifecycle settings may
change. If a process stopped after appending an event
but before committing its checkpoint, resume removes that uncommitted JSONL
tail before continuing. It likewise removes `kmc.extxyz` frames beyond the
checkpoint step by atomic replacement before appending. Missing, non-integer,
or non-monotonic `kmc_step` metadata in the committed trajectory prefix stops
the restart without changing the file. Reaction folders record their immutable
discovery step; folders discovered after the restored checkpoint are moved to
`uncommitted_reactions/` for recovery instead of remaining in the active
network hierarchy. See
[Outputs, Restart, and Offline Analysis](docs/outputs-and-analysis.md#checkpoint-continuation).

## Suggested Workflow

1. Start with a small surface and one reactant.
2. Run adsorption/desorption only.
3. Inspect `summary.json`, `events.jsonl`, and `reactions/`.
4. Enable diffusion with a small number of KMC steps.
5. Enable bond reactions and keep `persist_neb_path: true` while debugging.
6. Turn on free-energy corrections once the network looks reasonable.
7. Run `autokmc preflight CONFIG.yaml` before committing an expensive job.
8. Run `autokmc analyze RUN_DIR` and `autokmc report RUN_DIR` for product
   rates, mechanisms, convergence, coverage, and performance summaries.
9. Keep `calculation_cache/` with the run artifacts so results can be reused
   and audited later.

## Development

Run tests with:

```bash
python -m pip install -e ".[cli,test,dev]"
python -m compileall -q autokmc
ruff check autokmc tests
mypy --follow-imports=skip \
  autokmc/io/_files.py autokmc/io/config.py autokmc/io/checkpoint.py \
  autokmc/io/calculation_cache.py autokmc/io/calculators.py \
  autokmc/io/config_validation.py \
  autokmc/io/event_log.py autokmc/io/persistence.py \
  autokmc/io/reaction_graph.py autokmc/io/resume_contract.py \
  autokmc/io/trajectory.py \
  autokmc/analysis/products.py autokmc/core/graph_state.py \
  autokmc/sites/identity.py autokmc/kmc autokmc/workflow \
  autokmc/utils/telemetry.py
pytest --cov=autokmc --cov-report=term-missing --cov-fail-under=50
```

Useful package areas:

- `autokmc/structure`: slab and nanoparticle builders.
- `autokmc/species`: SMILES parsing and reactant construction.
- `autokmc/sites`: adsorption, diffusion, bond-site, and stability logic.
- `autokmc/reactions`: reaction objects and rate calculations.
- `autokmc/kmc`: KMC state, sampling, execution, and on-the-fly expansion.
- `autokmc/io`: config loading, calculator construction, event persistence,
  reaction database, and ISAAC export.
- `autokmc/analysis`: offline product-rate and mechanism reconstruction.
- `autokmc/thermo`: gas and adsorbate thermochemistry helpers.

## Troubleshooting

- If config validation fails, check for unknown keys or indentation errors.
- If a run is very slow, lower `kmc.n_steps`, disable diffusion/bond reactions,
  or reuse an existing `calculation_cache/`.
- If a NEB path is poor, try `interpolation: idpp`, `atom_matching: auto`, and a
  larger `matching_trials`.
- If FIRE prints the same NEB energy and force for many steps with
  `downhill_check: true`, its rollback logic is collapsing `dt`. AutoKMC
  automatically disables the check and restores the initial timestep after
  five rollback halvings for ordinary NEB; CI-FIRE disables it immediately.
  See [FIRE downhill recovery for NEB](docs/configuration.md#fire-downhill-recovery-for-neb).
- A NEB that exhausts its optimizer steps is not evidence that the reaction is
  chemically impossible. AutoKMC preserves the failed band, keeps its lateral
  class retryable, and omits only that candidate from the current rate-index
  sweep so other valid KMC events can still run. Inspect or retry the saved
  path; do not replace a missing barrier with a fabricated value.
- If no reactions are available, inspect adsorption-site pruning and gas-phase
  reactant energies.
- If post-processed product rates are zero, inspect `events.jsonl` for a
  desorption of a non-feed surface species. Direct gas-forming bond events are
  intentionally not counted by the strict product definition.

## License

MIT
