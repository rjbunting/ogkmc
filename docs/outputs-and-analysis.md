# Outputs, Restart, and Offline Analysis

## Run directory

A successful run writes the following core artifacts under `output.dir`:

```text
RUN_DIR/
  run_manifest.json
  events.jsonl
  summary.json
  diagnostics/performance.json
  kmc.extxyz
  checkpoint.pkl                 # when enabled
  isaac_records.json             # when output.isaac_export_enabled is true
  calculation_cache/
  reactions/
    index.jsonl
  diagnostics/
    invalid_adsorption/
    invalid_diffusion/
    invalid_bond/
  analysis/                      # after `autokmc analyze`
```

JSON documents are written atomically where replacement is appropriate and
reject nonfinite numeric output. `events.jsonl` is append-only. Event rows may
remain in the process buffer between checkpoints; they are flushed and synced
before a checkpoint publishes its exact committed byte offset and again when
the writer closes.

A fresh configured run refuses any existing managed artifacts in its
`output.dir`; it never truncates or mixes an earlier event log, manifest,
summary, trajectory, checkpoint, reaction tree, cache, analysis, or diagnostics
tree. Use a new output directory for a new trajectory. A checkpoint resume is
the explicit continuation path and retains the reconciliation behavior below.
One filesystem lock is held for the complete configured run, so concurrent
processes cannot write the same output directory. `autokmc preflight CONFIG`
checks both conditions without creating configured outputs.

## `run_manifest.json`

Manifest schema version 3 records:

- a UUID shared with events, checkpoints, graph trajectory frames, and new
  ISAAC calculations,
- event schema version,
- exact config-file text, SHA-256, and resolved configuration,
- feed species and partial pressures,
- catalyst kind, composition, atom count, and surface-atom count,
- for a file-backed catalyst, the resolved source path, selected frame,
  requested format, file SHA-256 and byte size, chemical formula, and
  resolved frozen-atom indices and count,
- the initial occupied surface state,
- preparing/running/terminal lifecycle state and the current stage,
- one fully closed segment for each initial or resumed invocation,
- explicit termination reason, final durable step, simulated time, and wall
  time,
- warnings, invalid-record counts, and quarantine locations,
- the persisted output map and a typed artifact inventory with size and
  SHA-256 metadata for files.

The manifest is the normalization and provenance source used by offline
analysis. Keep it with `events.jsonl`.

## `summary.json`

Summary schema version 3 stores the run UUID, fired-event counts, discovered
valid and invalid reaction counts, final occupancy, and run metadata.
Reaction-type rows are separated by direction and rate-energy basis. Their
`rate_delta_ev` and `rate_barrier_ev` statistics describe the values used to
calculate rates; separate `electronic_energy` and `free_energy` statistics
preserve both physical bases when available. `run.performance` is a concise
operational view: wall time, event throughput, cache hit ratio,
optimization/NEB counts, output/checkpoint overhead, and top bottlenecks. The
complete `counters`, `timings_s`, and `gauges` snapshot is retained in the
versioned `diagnostics/performance.json` artifact.

## `events.jsonl`

Each compact event-schema-v3 line represents one fired reaction. Important
fields are:

| Field | Meaning |
| --- | --- |
| `artifact_type`, `schema_version` | Artifact identity and event schema version. |
| `run_id` | Run UUID. |
| `event_id` | Deterministic event identity scoped to the run and KMC step. |
| `reaction_id` | Stable reaction-class identity resolved through `reactions/index.jsonl`. |
| `step`, `time_s`, `tau_s` | Cumulative KMC step/time and sampled waiting time. |
| `kind` | `adsorption`, `desorption`, `diffusion`, or `bond`. |
| `direction` | Diffusion or bond direction when applicable. |
| `inputs`, `outputs` | Canonical gas/surface state transition. |
| `rate_hz` | Microscopic propensity used by KMC. |
| `rate_energy_basis` | `electronic` or `free_energy`. |
| `rate_delta_ev`, `rate_barrier_ev` | Energetics actually used for the rate. |
| `delta_e_ev`, `barrier_ev` | Electronic values when available. |
| `delta_g_ev`, `barrier_g_ev` | Free-energy values when available. |

A surface state includes canonical species, placement id, site/member ids,
adsorbate node ids, and occupied catalyst cliques. Those transitions are
sufficient to reconstruct lineage without adding product or mechanism state
to the live KMC engine. Static `description`, `template`, `reaction_dir`, and
`gas_product` values are stored once in the reaction index instead of being
repeated on every event. Built-in offline analysis resolves them
automatically. Expanded schema-v2 events remain readable for compatibility.

## Reaction folders

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
    neb_path_initial.extxyz      # optional
    neb_path.extxyz              # optional
  bond/<process>/bond_isoX_latY/
    reaction.json
    state_ab_initial.extxyz
    state_ab.extxyz
    state_c_initial.extxyz
    state_c.extxyz
    state_c_gas_reference.extxyz # empty surface only; gas products
    gas_molecule.extxyz          # optimized gas molecule; gas products
    ts.extxyz
    neb_path_initial.extxyz      # optional
    neb_path.extxyz              # optional
```

`reaction.json` contains discovery metadata, electronic/free energetics,
vibrational results, KMC barriers, calculator identity, validity, and
cumulative firing statistics. The `.extxyz` files preserve the structures
behind those values. Its immutable `discovery_step` records when the folder
first became part of the discovered network.
For a gas-product bond reaction, the thermodynamic C-state geometry is stored
as two independent calculation inputs: `state_c_gas_reference.extxyz` contains
the relaxed surface/lateral environment without a molecule in the vacuum, and
`gas_molecule.extxyz` contains only the optimized gas molecule. Their energies
sum to `energies_ev.state_c`; `state_c.extxyz` remains the molecular precursor
used as the NEB endpoint.
Endpoint files ending in `_initial.extxyz` contain the exact structures passed
to relaxation. When an optimization fails, its corresponding non-`_initial`
endpoint file contains the last-known atomic geometry. When
`persist_neb_path` is enabled,
`neb_path_initial.extxyz` contains the interpolated band before NEB
optimization and `neb_path.extxyz` contains the optimized band.

`reactions/index.jsonl` is the versioned network index. Each stable
`reaction_id` records validity, discovery step, supported directions, firing
count, observed rate-energy bases, sidecar folder, and a compact static
definition. Invalid diffusion candidates are excluded from the authoritative
reaction tree and written under
`diagnostics/invalid_diffusion/<species>/diff_isoX_latY/`; their invalid index
entries remain available for discovery accounting and diagnostics.
Invalid bond candidates are handled analogously under
`diagnostics/invalid_bond/<process>/bond_isoX_latY/`.
Failure folders retain every available initial, converged, or last-known
endpoint and automatically retain both the initial and final NEB bands,
regardless of the successful-run `persist_neb_path` setting.

Adsorption candidates rejected during MLIP pruning are written under
`diagnostics/invalid_adsorption/<species>/ads_isoX/`. Each folder contains
`initial.extxyz`, `optimized.extxyz` whenever relaxation started, and
`diagnostic.json` with the rejection reason and relevant force, energy, or
connectivity details.

## `kmc.extxyz`

The trajectory contains catalyst atoms and currently occupied adsorbate atoms.
Versioned frame metadata stores `kmc_step`, `frame_kind`, and, when available,
the run UUID, simulated time, causal event ID, graph schema, and continuation
segment start. A final frame is guaranteed even when the terminal step is not
on the periodic cadence. If a run fires no events, the existing step-zero frame
is marked final without adding a duplicate step.
Per-atom arrays include:

- `graph_node_id`
- `node_type`
- `reactant_smiles`
- `reactant_index`
- `site_iso_class`
- `site_member_index`
- `occupied`
- `frozen`

The explicit `frozen` mask is the portable constraint record in extended XYZ.
ASE also receives a `FixAtoms` constraint in the in-memory snapshot, but the
mask should be used when reconstructing constraints from the file.

## Checkpoint continuation

Schema-v4 checkpoints are compact restart snapshots. They contain the live
graph, all currently known sites/reactants, occupancy, reaction counts, frozen
indices, RNG state, a scientific-config fingerprint, and the exact committed
`events.jsonl` count and byte offset. The compatibility `history` field remains
present but is empty in routine checkpoints: committed event history is
reconstructed from `events.jsonl` instead of being copied into every snapshot.
Standalone API runs that attach a checkpoint writer without an event writer
retain history in the checkpoint because no reconstructible event source
exists.
Calculator objects are removed and rebuilt from the current config. Large,
reconstructible graph caches (`surface_apsp`, surface-shell BFS data, and
geometry lookup arrays) are also omitted and rebuilt lazily.

To continue the same run, keep `output.dir` unchanged and set `resume_from`:

```yaml
checkpoint:
  enabled: true
  path: ./runs/my_run/checkpoint.pkl
  every_n_steps: 100
  resume_from: ./runs/my_run/checkpoint.pkl
```

Continuation semantics are cumulative:

- `events.jsonl` is reconciled to the checkpoint's committed prefix, removing
  only a crash tail, and new events are then appended. Retained event steps
  must be positive, non-boolean JSON integers and consecutive; malformed
  histories are rejected before the file is changed. When a legacy prefix
  predates run UUIDs, its validated rows are atomically assigned the continuing
  run UUID so the next exact checkpoint can be resumed again. If the original
  legacy event log is missing but the checkpoint still contains public KMC
  history, AutoKMC first writes a canonical cumulative prefix marked
  `legacy_history_recovered`; this prevents a later compact checkpoint from
  losing the earlier public history. These synthetic rows preserve the fields
  present in the historic KMC tuple, but cannot recreate transition lineage or
  species labels that the legacy checkpoint never stored; strict offline
  lineage analysis may therefore still report the missing provenance,
- reaction folders discovered after the checkpoint are moved recoverably to
  `uncommitted_reactions/after_checkpoint_step_<N>/`, so their rolled-back
  scientific state cannot remain authoritative,
- `kmc.extxyz` is reconciled by `kmc_step`: frames beyond the checkpoint step
  are removed by atomic replacement before new frames are appended,
- reaction counts and first/last steps are restored,
- cumulative summary and reaction-folder counters are recovered during the
  same streaming validation pass,
- the stored RNG state resumes the same random stream,
- the run manifest adds a continuation segment under the same UUID.

Configured resume does not allocate one Python history tuple per committed
event. It carries an `events.jsonl`-backed history view and stores only the
not-yet-synced continuation suffix in memory; that suffix is released whenever
a checkpoint or final close makes the corresponding rows durable. Python
callers that explicitly need the historic list can iterate it or call
`list(result["history"])`; doing so streams the committed prefix on demand.

Trajectory reconciliation requires every readable frame to have a non-negative
integer `kmc_step`. The committed prefix must be strictly increasing, and a
committed frame cannot appear after a crash-tail frame. AutoKMC rejects those
ambiguous or malformed histories without modifying the original file. A
missing or empty trajectory remains valid for legacy runs that did not persist
trajectory frames.

Scientific settings must match the checkpoint. The explicit safe-change
allowlist is limited to `kmc.n_steps`, `kmc.log_every`, `output.log_level`, and
the `checkpoint.enabled`, `checkpoint.path`, `checkpoint.every_n_steps`, and
`checkpoint.resume_from` lifecycle fields. The fingerprint content-hashes
resolvable calculator artifacts, including directory-valued inputs, and records
the AutoKMC source digest, AutoKMC version, and installed calculator-package
versions. Legacy checkpoints without this fingerprint remain readable and use
conservative step-based event reconciliation. Their trajectory is still
reconciled from the checkpoint step. Legacy checkpoint history is retained for
compatibility and, when its original event log is unavailable, is promoted to
the canonical event prefix before the first continuation checkpoint is written.

A checkpoint restores one KMC trajectory. The reaction database serves a
different purpose: it reuses scientific calculations across trajectories.

## Product and mechanism analysis

Run analysis after KMC:

```bash
autokmc analyze RUN_DIR
autokmc analyze RUN_DIR --start-time 1e-4 --end-time 5e-4 --blocks 20
autokmc analyze RUN_DIR --manifest custom_manifest.json
```

Strict mode is the default. `--allow-incomplete` retains unknown lineage roots
instead of rejecting missing or inconsistent surface history.

### Product definition

A product is a non-feed surface species that leaves an occupied placement in a
`desorption` event. Direct gas-forming bond events are not counted by this
strict definition because the requested manuscript analysis starts from a
surface species desorbing.

### Rate definition

For a selected analysis window of duration `D`:

```text
observed product rate = number of product desorptions / D
TOF = observed product rate / number of classified surface atoms
```

The product-rate interval is the exact two-sided 95% Garwood Poisson interval.
It is not a normal approximation. `rate_blocks.csv` divides the selected window
into equal-duration blocks for stationarity inspection.

### Mechanism reconstruction

For each product desorption, the analyzer follows its consumed placement
backward through formation events. Diffusion relocates the same lineage and is
omitted from the chemical mechanism. Bond formation joins both precursor
histories. Immediate reversible bond recrossings cancel before mechanisms are
fingerprinted and grouped.

Analysis is transactional: all five outputs are staged first, and existing
analysis files remain unchanged if strict validation or writing fails.

```text
analysis/
  analysis_summary.json
  product_rates.csv
  product_events.jsonl
  mechanisms.csv
  rate_blocks.csv
```

## Human-readable run report

```bash
autokmc report RUN_DIR
autokmc report RUN_DIR --blocks 20 --output-dir RUN_DIR/analysis
autokmc report RUN_DIR --no-refresh-analysis
```

The report combines run status and termination reason, coverage, directional
flux, products, rate convergence, performance bottlenecks, and artifact
inventory. It writes `report.md` and a self-contained `report.html`. Product
analysis is refreshed from persisted events by default; use
`--no-refresh-analysis` to reuse the current analysis artifacts.
