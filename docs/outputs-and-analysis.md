# Outputs, Restart, and Offline Analysis

## Run directory

A successful run writes the following core artifacts under `output.dir`:

```text
RUN_DIR/
  run_manifest.json
  events.jsonl
  summary.json
  kmc.extxyz
  checkpoint.pkl                 # when enabled
  isaac_records.json
  calculation_cache/
  reactions/
  analysis/                      # after `autokmc analyze`
```

JSON documents are written atomically where replacement is appropriate and
reject nonfinite numeric output. `events.jsonl` is append-only and flushed
after each event.

## `run_manifest.json`

Manifest schema version 2 records:

- a UUID shared with events, checkpoints, graph trajectory frames, and new
  ISAAC calculations,
- event schema version,
- exact config-file text, SHA-256, and resolved configuration,
- feed species and partial pressures,
- catalyst kind, composition, atom count, and surface-atom count,
- the initial occupied surface state,
- one segment for each initial or resumed invocation,
- final step, time, and executed-step count when the run finishes.

The manifest is the normalization and provenance source used by offline
analysis. Keep it with `events.jsonl`.

## `events.jsonl`

Each event-schema-v2 line represents one fired reaction. Important fields are:

| Field | Meaning |
| --- | --- |
| `run_id` | Run UUID. |
| `step`, `time_s`, `tau_s` | Cumulative KMC step/time and sampled waiting time. |
| `kind` | `adsorption`, `desorption`, `diffusion`, or `bond`. |
| `direction` | Diffusion or bond direction when applicable. |
| `inputs`, `outputs` | Canonical gas/surface state transition. |
| `placement_id` | Stable identity for a concrete surface state within the run. |
| `rate_hz` | Microscopic propensity used by KMC. |
| `rate_energy_basis` | `electronic` or `free_energy`. |
| `rate_delta_ev`, `rate_barrier_ev` | Energetics actually used for the rate. |
| `delta_e_ev`, `barrier_ev` | Electronic values when available. |
| `delta_g_ev`, `barrier_g_ev` | Free-energy values when available. |
| `reaction_dir` | Relative path to the reaction sidecar folder. |

A surface state includes canonical species, placement id, site/member ids,
adsorbate node ids, and occupied catalyst cliques. Those transitions are
sufficient to reconstruct lineage without adding product or mechanism state
to the live KMC engine.

## Reaction folders

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
    neb_path.extxyz              # optional
  bond/<process>/bond_isoX_latY/
    reaction.json
    state_ab.extxyz
    state_c.extxyz
    ts.extxyz
    neb_path.extxyz              # optional
```

`reaction.json` contains discovery metadata, electronic/free energetics,
vibrational results, KMC barriers, calculator identity, validity, and
cumulative firing statistics. The `.extxyz` files preserve the structures
behind those values.

## `kmc.extxyz`

The trajectory contains catalyst atoms and currently occupied adsorbate atoms.
Each frame stores `kmc_step` and, when available, the run UUID and graph schema.
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

Checkpoints contain the live graph, all currently known sites/reactants,
occupancy, KMC history, reaction counts, frozen indices, and RNG state.
Calculator objects are removed and rebuilt from the current config.

To continue the same run, keep `output.dir` unchanged and set `resume_from`:

```yaml
checkpoint:
  enabled: true
  path: ./runs/my_run/checkpoint.pkl
  every_n_steps: 100
  resume_from: ./runs/my_run/checkpoint.pkl
```

Continuation semantics are cumulative:

- `events.jsonl` is appended rather than truncated,
- `kmc.extxyz` is appended without duplicating the initial frame,
- reaction counts and first/last steps are restored,
- `summary.json` is rebuilt from all event rows,
- the stored RNG state resumes the same random stream,
- the run manifest adds a continuation segment under the same UUID.

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
