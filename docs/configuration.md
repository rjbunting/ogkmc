# Configuration Reference

AutoKMC accepts YAML (`.yaml` or `.yml`) and TOML (`.toml`). The top-level
`schema_version` is currently `"1"`. Unknown keys, coercible string booleans,
nonfinite values, invalid enums, and out-of-range values are rejected.

Always validate before launching an expensive run:

```bash
autokmc validate-config CONFIG.yaml
```

The platinum examples in [`example/`](../example/) are the recommended
production-style starting points. Although some internal dataclasses retain
generic legacy defaults, production configurations should set the catalyst,
calculator, thermochemistry, and convergence settings explicitly.

## `output`

| Key | Default | Meaning |
| --- | --- | --- |
| `dir` | `autokmc_run` | Run directory. |
| `reactions_filename` | `events.jsonl` | Append-only event log name. |
| `summary_filename` | `summary.json` | Cumulative summary name. |
| `run_manifest_filename` | `run_manifest.json` | Reproducibility manifest name. |
| `trajectory_filename` | `kmc.extxyz` | Extended-XYZ trajectory name. |
| `trajectory_dump_every` | `10` | Write every Nth KMC step; `0` disables it. |
| `calculation_cache_enabled` | `true` | Enable the ISAAC reaction database. |
| `calculation_cache_dir` | `calculation_cache` | Database directory below `output.dir`. |
| `isaac_export_filename` | `isaac_records.json` | Portable ISAAC record-array export. |
| `log_level` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. |

## `structure`

| Key | Default | Meaning |
| --- | --- | --- |
| `kind` | `surface` | `surface` or `nanoparticle`. |
| `composition` | internal default `Cu` | Element/composition accepted by the structure builder. Set `Pt` for the supplied examples. |
| `crystal_structure` | `fcc` | Crystal structure passed to the builder. |
| `miller_index` | `[1, 1, 1]` | Three-index surface orientation. |
| `lattice_constant` | `null` | Explicit lattice parameter or builder-specific value; `null` requests calculator/reference handling. |
| `min_slab_size` | `8.0` Å | Minimum slab thickness. |
| `min_vacuum_size` | `12.0` Å | Minimum vacuum thickness. |
| `goal_x`, `goal_y` | `12.0` Å | Target lateral dimensions. |
| `n_freeze_layers` | `2` | Number of bottom layers to constrain. |
| `fmax` | `0.05` eV/Å | Structure relaxation threshold. |
| `max_steps` | `1000` | Maximum structure relaxation steps. |
| `n_atoms` | `null` | Approximate nanoparticle atom count. |
| `surface_energies` | `null` | Explicit facet energies; `null` calculates them when needed. |
| `surface_energy_facets` | `[[1,1,1],[1,0,0],[1,1,0]]` | Facets used for nanoparticle surface-energy calculations. |
| `surface_energy_layers` | `6` | Layers in facet calculations. |
| `surface_energy_vacuum` | `10.0` Å | Vacuum in facet calculations. |
| `surface_energy_fmax` | `null` | Optional facet-specific force threshold. |
| `surface_energy_max_steps` | `null` | Optional facet-specific step limit. |
| `extra_kwargs` | `{}` | Additional builder keyword arguments. |

Positive sizes and force thresholds are required. `n_freeze_layers` may be
zero. Unconverged structure optimization is an error.

## `reactants`

`reactants` is a list. `smiles` is required for each entry.

| Key | Default | Meaning |
| --- | --- | --- |
| `smiles` | required | Input SMILES; canonicalized internally. |
| `add_hydrogens` | `true` | Add implicit hydrogens during molecular construction. |
| `relax_in_gas` | `true` | Relax gas geometry. If false, a finite single-point energy is still computed. |
| `partial_pressure_bar` | `null` | Species pressure; falls back to `free_energy.pressure_bar`. Zero prevents gas adsorption. |
| `symmetry_number` | `null` | Ideal-gas symmetry number; otherwise uses the free-energy default. |
| `spin` | `null` | Spin value used by ideal-gas thermochemistry. |
| `geometry` | `null` | `auto`, `linear`, `nonlinear`, or `monatomic`. |

Feed reactants define which desorbing gas species are excluded from the strict
post-processing product definition. Species auto-built from bond templates
are surface-generated leaf species and receive zero gas pressure.

## `calculator`

Exactly one construction style is normally used:

```yaml
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
```

or:

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

| Key | Default | Meaning |
| --- | --- | --- |
| `import_path` | `null` | Dotted calculator class. |
| `kwargs` | `{}` | Class constructor arguments. |
| `factory` | `null` | Dotted factory callable; takes precedence over `import_path`. |
| `factory_kwargs` | `{}` | Factory arguments. |
| `copies` | `1` | Number of independent calculator instances. |
| `gpu_devices` | `null` | Optional device list assigned across copies. |
| `gpu_device_arg` | `device` | Constructor/factory argument that receives a device. |
| `max_workers` | `null` | Maximum parallel calculator tasks. |

If neither construction path is supplied, the CLI falls back to ASE EMT.
Production calculations should always configure the intended calculator
explicitly.

## `adsorbate_sites`

| Key | Default | Meaning |
| --- | --- | --- |
| `prune_stable_only` | `true` | Relax one representative per candidate class and retain stable classes. |
| `fmax` | `0.05` eV/Å | Pruning relaxation threshold. |
| `max_steps` | `500` | Pruning step limit. |

## `kmc`

| Key | Default | Meaning |
| --- | --- | --- |
| `temperature_k` | `500.0` K | KMC temperature; must be positive. |
| `n_steps` | `1000` | Steps to execute in this invocation; may be zero. |
| `transmission_coefficient` | `1.0` | Non-negative Eyring coefficient. |
| `fmax` | `0.05` eV/Å | Adsorption endpoint threshold during KMC discovery. |
| `max_steps` | `200` | Adsorption endpoint step limit. |
| `log_every` | `1` | Progress cadence; `0` disables step messages. |
| `random_seed` | `69` | Initial random seed. Checkpoint resume restores RNG state. |
| `lateral_interactions` | `true` | Reclassify local lateral environments after events. |

## `diffusion`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable diffusion channels. |
| `max_hops` | `0` | Maximum surface-graph separation used to form hop candidates. |
| `n_shells_pair` | `1` | Local graph depth used for pair classification. |
| `prune_by_adsorption_pair` | `true` | Keep one diffusion class per adsorption pair. |
| `fmax` | `0.01` eV/Å | NEB force threshold. |
| `max_steps` | `200` | NEB optimization limit. |
| `n_images` | `10` | Number of NEB images. |
| `climb` | `true` | Use climbing-image NEB. |
| `spring_k` | `0.1` | NEB spring constant. |
| `interpolation` | `linear` | `linear` or `idpp`. |
| `persist_neb_path` | `false` | Save the complete image sequence as `.extxyz`. |

## `bond`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable bond-changing channels. |
| `bond_max_hops` | `0` | Maximum surface separation for A/B candidates. |
| `surface_apsp_cutoff` | `10` | Surface all-pairs shortest-path cutoff. |
| `bond_types` | `["SINGLE", "DOUBLE", "TRIPLE"]` | Allowed RDKit bond types. |
| `include_ring_bonds` | `false` | Include ring-bond transformations. |
| `include_homo_coupling` | `true` | Permit A + A coupling templates. |
| `include_dissociation` | `true` | Generate C to A + B templates. |
| `include_coupling` | `true` | Generate A + B to C templates. |
| `deduplicate_iso` | `true` | Deduplicate graph-isomorphic classes. |
| `gas_lift_height` | `6.0` Å | Lift used for gas-product NEB endpoints. |
| `auto_build_leaf_species` | `true` | Build implied species absent from the feed list. |
| `pair_n_shells` | `1` | Ego-graph depth for triple pruning. |
| `prune_by_triple` | `true` | Keep the preferred stable class per adsorption triple. |
| `prune_with_calculator` | `true` | Relax and reject unstable A+B endpoints. |
| `prune_fmax` | `0.05` eV/Å | Bond-pruning threshold. |
| `prune_max_steps` | `500` | Bond-pruning step limit. |
| `neb_fmax` | `0.01` eV/Å | Bond NEB threshold. |
| `neb_max_steps` | `200` | Bond NEB step limit. |
| `neb_n_images` | `10` | Bond NEB images. |
| `neb_climb` | `true` | Use climbing-image NEB. |
| `neb_spring_k` | `0.1` | Bond NEB spring constant. |
| `neb_interpolation` | `idpp` | `linear` or `idpp`. |
| `atom_matching` | `auto` | `auto`, `greedy`, `hungarian`, or `reactant_index`. |
| `matching_trials` | `8` | Number of mapping trials used by automatic matching. |
| `persist_neb_path` | `false` | Save the complete bond NEB path. |

## `free_energy`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Enable vibrational/ideal-gas corrections. |
| `pressure_bar` | `1.0` bar | Default gas pressure and standard-state pressure input. |
| `vibration_displacement` | `0.01` Å | Finite-difference displacement. |
| `vibration_nfree` | `2` | Must be `2` or `4`. |
| `include_ts_vibrations` | `true` | Compute transition-state vibrations. |
| `min_frequency_ev` | `0.0015` eV | Low-frequency floor used by thermochemistry. |
| `default_symmetry_number` | `1` | Gas symmetry fallback. |
| `default_spin` | `0.0` | Gas spin fallback. |
| `default_geometry` | `auto` | `auto`, `linear`, `nonlinear`, or `monatomic`. |
| `cache_dir` | `null` | Persistent vibration-cache root; defaults below the run directory. |

Free-energy work can dominate runtime. The supplied platinum GPU examples
disable it intentionally for network-debug runs and can be switched on for
production thermochemistry.

## `checkpoint`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Write periodic checkpoints. |
| `path` | `null` | Checkpoint path; defaults to `OUTPUT_DIR/checkpoint.pkl`. |
| `every_n_steps` | `1` | Checkpoint cadence. |
| `resume_from` | `null` | Checkpoint to continue. |

On resume, event and trajectory files are appended, reaction-folder counters
are restored, cumulative summary state is reconstructed from `events.jsonl`,
and the saved NumPy or Python RNG state continues the random stream.
