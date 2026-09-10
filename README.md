# AutoKMC

AutoKMC prepares and runs surface kinetic Monte Carlo simulations from one
configuration file. It first builds a slab or nanoparticle, or loads a catalyst
from any format supported by ASE. It then generates reactants, surface sites,
and adsorption, desorption, diffusion, bond-forming, and bond-breaking channels.
Finally, it calculates the required energetics, runs KMC, and writes the results
needed for analysis and restart.

Every calculation uses an ASE-compatible calculator. The same workflow can
therefore use a simple local calculator for a smoke test or a machine-learning
potential for a production study.

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

First, `validate-config` checks types, ranges, reactant SMILES, bond types, the
calculator declaration, worker/device consistency, and managed output names.
Next, `preflight` checks output collisions and locks, checkpoint compatibility,
file-backed catalyst readability, calculator imports, and the effective
copies/workers/devices without writing configured outputs. Add
`--check-calculator` to construct the calculator in a temporary directory and
require a finite probe energy and force array. Use `doctor` to report Python,
required-package, and optional configuration readiness without running
chemistry.

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
  # Four Pt(111) layers, repeated 4x4 in the primitive surface cell.
  min_slab_size: 8.0
  min_vacuum_size: 12.0
  goal_x: 10.0
  goal_y: 10.0
  extra_kwargs:
    orthogonalise: false
  n_freeze_layers: 2

reactants:
  - smiles: "[C-]#[O+]"
    add_hydrogens: false
    relax_in_gas: true
    fmax: 0.05
    max_steps: 500
    partial_pressure_bar: 1.0
    geometry: linear

calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}

adsorption:
  prune_stable_only: true
  prune_fmax: 0.05
  prune_max_steps: 200
  endpoint_fmax: 0.05
  endpoint_max_steps: 200
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

Start with this README for installation and a first run. Then use the focused
pages for complete option tables, persistence contracts, database matching,
and manuscript post-processing.

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
- `adsorption`: adsorption-site enumeration, stability pruning, and runtime
  endpoint relaxations.
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

For each new diffusion lateral class, AutoKMC first relaxes both endpoints. It
then optimizes an ordinary NEB band and, when `climb` is enabled, always refines
the same band with a climbing image. Raw barrier height does not bypass this
optimization. The rate calculation separately applies its 0.1 eV floor through
one common effective transition-state level, which preserves reversible energy
consistency. These calculations are often among the most expensive parts of a
run.

If an ordinary NEB takes 100 optimizer steps without finding a lower
interior-image electronic energy, AutoKMC inspects that band's energy profile.
It selects only the two nearest minima bracketing the highest-energy image,
optimizes any selected interior states, and reruns the standard NEB workflow
for that one segment. Already-optimized original endpoints are reused. It
deliberately does not evaluate every adjacent-minimum segment. This is faster
but more aggressive; the resulting transition state is still referenced to
the original reaction endpoints when the forward and reverse barriers are
stored.

When `image_spacing` (or bond `neb_image_spacing`) is set, it also guards the
optimized geometry. By default, no unfrozen atom may move more than three times
that distance between adjacent images. If a step violates this limit, AutoKMC
restores the lowest-force valid band and immediately checks its electronic
energy profile for minima, without waiting for 100 stagnant steps. If a usable
bracket exists, it runs the one permitted highest-peak segment refinement;
this also applies to rollback during CI-NEB. Otherwise, it starts a fresh
optimizer inside the existing step budget. FIRE restarts with halved `dt` and
`dtmax`, preventing a stretched band from being retained. Set
`optimization.neb_geometry_guard_multiplier` to change the limit without
changing the image density.

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

`atom_matching` controls how the initial and final bond-reaction atoms are
paired before interpolation. Every unchanged bond within A and B must connect
the same atom indices in C; this prevents same-element atoms from exchanging
identities merely to shorten the path. The default `auto` mode tries several
connectivity-preserving mappings and then keeps the path with the smallest
displacement. With a non-null image spacing, AutoKMC uses the largest MIC-aware
displacement between corresponding atoms to choose the interior-image count.
If the spacing is `null`, it uses the configured fixed `n_images` or
`neb_n_images` value. `hungarian`, `greedy`, and `reactant_index` choose the
geometric seed, but fall back to a connectivity-preserving mapping if that seed
would change atom-index connectivity.

Bond NEBs use the same automatic bare-first initialization as diffusion NEBs:
the no-neighbour lateral class is calculated on demand and its optimized band
seeds classes with neighbouring adsorbates. No additional configuration key is
required.

### Free Energy

Free-energy corrections use one coupled ASE vibrational analysis over the
reacting adsorbate and the occupied neighboring molecules selected by
`constants.lateral_shells`, with catalyst atoms held fixed during the
displacements. The complete selected molecules participate, including atoms
without direct surface bonds. Both endpoints retain the selected neighbors;
transition states use the same local environment when TS vibrations are enabled.
These calculations can be expensive:

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

The lateral shell range controls endpoint structures, NEB structures, lateral
classification, and vibrational corrections in both electronic and free-energy
runs. For diffusion and bond reactions, the range starts from the union of
all endpoint anchor atoms, so both endpoints share the same selected neighbors.
Setting `kmc.lateral_interactions: false` omits surrounding occupied molecules;
free-energy corrections then cover only the reacting adsorbates and gas species.
Each event refreshes rates whose local environment or applicability can change.
Old free energies from the whole-surface or reactive-only policies are not
reused; compatible electronic calculations can supply structures for a new
local vibrational calculation.

The local lateral interaction range is an intentional approximation that truncates longer-range interactions. It can introduce small
reaction-energy errors and closed-cycle energy discrepancies. Increase
`constants.lateral_shells` with `kmc.lateral_interactions: true` until the
energies converge. In a CO/Cu(111) UMA example, extending one hop to two removed
a 6.93 meV cycle discrepancy by including all surrounding CO molecules. See
[lateral interaction range and the measured example](docs/architecture.md#lateral-interaction-range)
and the [configuration controls](docs/configuration.md#lateral-interaction-range).

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

Before a fresh run starts, AutoKMC checks for managed output artifacts. If any
already exist, the run stops instead of mixing or overwriting event logs,
manifests, summaries, checkpoints, reaction folders, or diagnostics. Choose a
new `output.dir`, or set `checkpoint.resume_from` to continue the same run.
During execution, AutoKMC also holds a filesystem lock on `output.dir` so a
second process cannot write there concurrently.

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
    neb_refinement_initial.extxyz # when stalled-path refinement is used
    neb_refinement_final.extxyz   # when stalled-path refinement is used
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
    neb_refinement_initial.extxyz # when stalled-path refinement is used
    neb_refinement_final.extxyz   # when stalled-path refinement is used
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
When stalled-path refinement is used, `neb_refinement_initial.extxyz` and
`neb_refinement_final.extxyz` are the most recently selected optimized
endpoints; the NEB path files then describe the final replacement highest-peak
segment.
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

A product is a species not listed among the feed reactants that leaves an
occupied surface placement through desorption. For each product, the analyzer
first finds the desorption event and then follows the consumed placement
backward through the events that formed it. Diffusion moves the same lineage and
is omitted. Bond formation follows both precursor histories. Finally, immediate
reversible bond recrossings are removed from the reported mechanism.

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

On a later run, AutoKMC first checks the exact calculation key. If that misses,
it searches the SQLite index for the same reaction, settings, calculator, and
graph fingerprint. It then verifies full labelled graph isomorphism, normalized
scientific inputs, the calculator/model digest, and every asset checksum before
accepting a candidate.

The atomic-input fingerprint is invariant to rigid translation, rotation,
periodic wrapping, and atom order. It still rejects strain and changes to
initial charges, magnetic moments, tags, custom atom arrays, or the full lattice
metric. Every other scientific input, including scalar gas energies and charge,
remains part of the identity. Model checkpoints are identified by SHA-256
content rather than their local path. Run-local node ids, `iso_class`, and
`lateral_class` values are excluded, while chemical identity, endpoint roles,
elements, bond roles, topology, and all other inputs are retained.

Relaxed structures and NEB paths remain in their original coordinate frame.
AutoKMC therefore reuses these assets only when the query has the same input
coordinates, cell, periodic images, and atom order. It rehashes model files and
directory-valued artifacts for every identity calculation. If an asset is
missing, unreadable, modified, or scientifically incompatible, AutoKMC rejects
the hit and recomputes the calculation.

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

Resume into the same `output.dir`. AutoKMC first verifies that the structure,
calculator, feed, thermochemistry, and reaction-channel configuration have not
changed. It compares resolvable calculator files and directories by content and
also checks the AutoKMC source digest and calculator-package versions. Only the
additional step count, logging controls, and checkpoint lifecycle settings may
change.

Next, AutoKMC reconciles the persisted outputs. It removes an uncommitted
`events.jsonl` tail and atomically removes `kmc.extxyz` frames beyond the
checkpoint step. Missing, non-integer, or non-monotonic `kmc_step` metadata in
the committed trajectory prefix stops the restart without changing the file.
Reaction folders record their immutable discovery step, so folders discovered
after the checkpoint move to `uncommitted_reactions/` instead of remaining in
the active network.

Finally, AutoKMC restores the reaction-folder counters and RNG stream, rebuilds
the cumulative summary from the event log, appends to `events.jsonl` and
`kmc.extxyz`, and adds a continuation segment to the existing run manifest. See
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
