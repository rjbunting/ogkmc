# Configuration Reference

AutoKMC reads YAML (`.yaml` or `.yml`) and TOML (`.toml`) configuration files.
Every file begins with the top-level `schema_version`, which is currently
`"1"`. The loader rejects unknown keys, coercible string booleans, nonfinite
values, invalid enums, and out-of-range values.

The adsorption controls now live in one top-level `adsorption` section.
Configs using the former `adsorbate_sites` section must rename it and replace
`fmax`/`max_steps` with `prune_fmax`/`prune_max_steps`. The former
`kmc.fmax`/`kmc.max_steps` controls are now
`adsorption.endpoint_fmax`/`adsorption.endpoint_max_steps`. These retired keys
are rejected rather than silently ignored.

Always validate before launching an expensive run:

```bash
autokmc validate-config CONFIG.yaml
autokmc preflight CONFIG.yaml
```

First, `validate-config` rejects empty feeds, invalid or duplicate canonical
SMILES, unsupported bond types, non-mapping calculator arguments, ambiguous
`import_path`/`factory` declarations, inconsistent calculator
copies/workers/devices, and unsafe or overlapping managed output names. Next,
`preflight` checks output collisions and active locks, checkpoint compatibility,
file-backed catalyst readability, calculator imports, and the effective
worker/device assignment. It does not write configured outputs. Add
`--check-calculator` to construct the calculator in a temporary directory and
require finite probe energy and forces.

Start with [`example/all_options.yaml`](../example/all_options.yaml) when you
need a complete input. It is valid as written, documents every public option,
and includes competing structure and calculator modes as commented
alternatives. Then use the palladium examples in [`example/`](../example/) as
production-style starting points. Some internal dataclasses retain generic
legacy defaults, so a production configuration should set its catalyst,
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
| `calculation_cache_lookup_enabled` | `false` | Reuse matching database records when `true`; the default write-only mode only generates database records. |
| `calculation_cache_dir` | `calculation_cache` | Database directory below `output.dir`. |
| `isaac_export_enabled` | `false` | Write a portable ISAAC record-array export at finalization. |
| `isaac_export_filename` | `isaac_records.json` | Export name used when `isaac_export_enabled` is true. |
| `log_level` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. INFO prints stage and periodic KMC progress; DEBUG also enables detailed site/network diagnostics. |

## `constants`

This section contains scientific and algorithmic values shared by several
stages. AutoKMC first resolves these values and then records them in the run
manifest. Controls used by only one channel, such as NEB image counts and force
thresholds, remain in that channel's section.

| Key | Default | Meaning |
| --- | --- | --- |
| `neighbor_list_multiplier` | `0.90` | Multiplier for ordinary ASE natural-cutoff bonds: $d_{ij} \le f(r_i+r_j)$. |
| `co_bond_factor` | `0.90` | Covalent-radius factor for simultaneous surface bonding to one adsorbate anchor: $d_{ij} \le f(2r_\mathrm{ads}+r_i+r_j)$. |
| `anchor_bond_factor` | `0.85` | Target anchor-bond factor: $d_\mathrm{ideal}=f(r_\mathrm{ads}+r_i)$. |
| `anchor_repulsion_weight` | `0.20` | Weight of the non-bonded $1/r^2$ term used while positioning anchors; zero disables it. |
| `site_repulsion_cutoff` | `10.0` Å | Radius around the clique centroid included in the repulsion sum; `null` includes all surface atoms. |
| `adsorbate_contact_factor` | `1.05` | Minimum rigid-body steric-contact factor: $R_\mathrm{min}=f(r_\mathrm{ads}+r_\mathrm{surface})$. |
| `adsorbate_standoff_factor` | `0.0` | Extra adsorbate lift along the outward surface normal in summed-covalent-radius units. |
| `adsorbate_rotational_restarts` | `6` | Rigid-body starting orientations tried during placement; one disables multi-start. |
| `typical_neighbor_distance` | `2.5` Å | Typical metal neighbor distance used to choose automatic anchor isomorphism depth. |
| `adsorbate_bond_tolerance` | `0.4` Å | Tolerance for matching molecular and surface anchor-pair distances. |
| `anchor_hull_tolerance` | `-0.2` Å | Signed-distance threshold for rejecting anchor centroids inside nanoparticle hulls. |
| `raycast_coverage_threshold` | `0.7` | Required exposed ray-disc fraction for slab surface classification; must be in `[0, 1]`. |
| `raycast_disc_samples` | `10` | Ray-disc sampling resolution per axis. |
| `kabsch_max_mappings` | `6969` | Maximum graph automorphisms examined during local Kabsch alignment. |
| `lateral_shells` | `0` | Surface-graph depth for lateral environments; zero requires sharing an anchor surface atom. |

The three covalent-radius factors serve different purposes:
`neighbor_list_multiplier` defines ordinary graph bonds, `co_bond_factor`
determines which multi-coordinate anchor cliques can exist, and
`structure.surface_radius_factor` changes the ray-casting discs used only for
slab surface classification.

## `optimization`

```yaml
optimization:
  optimizer: lbfgs
  optimizer_kwargs: {}
  neb_optimizer: bfgs
  neb_optimizer_kwargs: {}
  neb_climb_optimizer: null
  neb_climb_optimizer_kwargs: null
  neb_method: improvedtangent
  neb_band_eval: images
  neb_geometry_guard_multiplier: 3.0
  neb_intermediate_stagnation_steps: 100
  neb_intermediate_max_refinements: 10
  neb_intermediate_energy_tolerance: 0.001
  neb_intermediate_minimum_prominence: 0.01
```

`optimizer` controls calculator-backed ordinary relaxations, including
generated catalyst structures, gas-phase reactants, adsorbate and bond-site
pruning, adsorption stability, and NEB endpoint relaxation. Valid values are
`lbfgs`, `bfgs`, `fire`, and `mdmin`.

`optimizer_kwargs` is a YAML mapping forwarded to the selected ASE
optimizer constructor for every ordinary relaxation. AutoKMC supplies the
object being optimized and the logfile, so `atoms` and `logfile` cannot be
overridden. All other constructor keywords supported by the installed ASE
version are accepted and checked by `autokmc validate-config`.

`neb_optimizer` controls the ordinary diffusion and bond-reaction NEB band.
Valid values are `bfgs`, `fire`, and `mdmin`. `lbfgs` is intentionally excluded
because ASE does not recommend it for NEB. Defaults preserve the previous
behavior: `lbfgs` for ordinary relaxations and `bfgs` for NEB.

`neb_optimizer_kwargs` provides the same constructor-keyword interface for the
ordinary NEB optimizer. AutoKMC first optimizes the ordinary band and, whenever
the channel's climbing-image option is enabled, refines it with CI-NEB regardless
of the raw barrier height. CI-NEB uses `neb_climb_optimizer`; `null` reuses
`neb_optimizer`.
`neb_climb_optimizer_kwargs` supplies the climbing optimizer's constructor
keywords, while `null` reuses `neb_optimizer_kwargs`. Each stage creates a fresh
optimizer instance, so FIRE or MDMin velocity state does not pass from ordinary
NEB into CI-NEB. The calculation-cache identity includes every optimizer
choice and constructor mapping.

`neb_geometry_guard_multiplier` controls the geometric rollback threshold for
both diffusion and bond NEBs. The maximum adjacent-image atom displacement is
the multiplier times `diffusion.image_spacing` or `bond.neb_image_spacing`.
The default `3.0` therefore gives a 0.75 Å limit for the default 0.25 Å image
spacing. Changing this multiplier does not change the dynamically selected
number of images. It must be finite and greater than zero, and it is included
in calculation-cache identities.

The `neb_intermediate_*` controls define a deliberately narrow stalled-path
refinement. During the ordinary stage, AutoKMC tracks the lowest
interior-image electronic energy. After 100 consecutive optimizer steps
without a decrease of at least `neb_intermediate_energy_tolerance`, it finds
the highest-energy interior image and the nearest local minimum on each side.
An interior image counts as a minimum only when it lies below both neighboring
images by at least `neb_intermediate_minimum_prominence`; the original
endpoints also count as bounding minima.

A distance-guard rollback is a second, immediate trigger for the same check:
AutoKMC first restores the whole valid band with the lowest maximum NEB force
in the current optimizer stage, then evaluates that restored band's electronic
energy profile. It does not inspect the rejected stretched geometry or merely
the most recent valid frame. This check runs during either ordinary or CI-NEB
and does not wait for the 100-step stagnation threshold, even if rollback
consumes the last allowed stage step. A usable minimum bracket starts a fresh
ordinary/CI-NEB calculation with the normal per-stage budget. Otherwise, the
existing reduced-step rollback resumes within the remaining budget, or reports
non-convergence when that budget is exhausted.

If the two bounding states include at least one interior minimum, AutoKMC
optimizes the selected interior state or states with the ordinary `optimizer`
and runs the standard ordinary/CI-NEB workflow between that pair. A replacement
band is checked by the same stagnation and rollback rules and may be shortened
again, up to `neb_intermediate_max_refinements` times in one calculation
(default 10). It does not run NEBs between the other minima or refine the
remaining path segments. This is faster but can miss a competing barrier. The
final transition energy is nevertheless
combined with the original reaction endpoint energies (A/B for diffusion or
A+B/C for bond changes) for the stored forward and reverse energetics. The
most recently selected optimized states are written as
`neb_refinement_initial.extxyz` and `neb_refinement_final.extxyz`, and the
indices, stalled electronic-energy profile, and policy are stored in
`reaction.json` and the calculation cache. The metadata also distinguishes
`energy_stagnation` from `geometry_rollback` and records the source stage and,
for rollback, the checkpoint's maximum NEB force and optimizer step. It also
records the total refinement count and configured limit.

The retained ordinary transition energy remains the raw recorded value. At
rate construction, both reversible directions use the same effective level,
`max(E_ts, max(E_initial, E_final) + 0.1 eV)`. Thus the direction from the
higher-energy endpoint receives the 0.1 eV minimum while the opposite barrier
also includes the endpoint energy difference; this preserves detailed energy
consistency rather than independently forcing both directions to 0.1 eV.
Because the retained ordinary image is not a climbing-image stationary point,
AutoKMC also skips its TS vibrational calculation and uses the existing
average endpoint free-energy correction for that transition level.

`neb_method` selects ASE's NEB force and tangent formulation for both the
ordinary and climbing-image stages. Valid values are:

- `improvedtangent` (default): energy-weighted Henkelman--Jónsson tangents and
  tangential spring forces.
- `aseneb`: ASE's standard tangent, selected relative to the current
  highest-energy image.
- `eb`: the full elastic-band spring force, including perpendicular spring
  components.
- `spline`: spline-derived tangents plus ASE's spline-curvature spring force.
- `string`: spline-derived tangents with equal-arc-length redistribution after
  optimizer position updates instead of an explicit spring force.

The method is part of the calculation-cache identity. `spline` and `string`
fit splines through Cartesian image coordinates; for periodic paths, unwrap
cross-boundary atomic trajectories consistently before using them. Changing
the method can substantially change the projected force norm, so compare
methods from the same saved coordinates rather than treating their reported
`fmax` values as directly interchangeable.

For a conservative FIRE setup that limits bad initial geometries, configure
both the initial adaptive timestep and its upper bound; `dt` alone will grow
toward ASE's `dtmax`. `maxstep` adds a separate displacement cap:

```yaml
optimization:
  optimizer: fire
  optimizer_kwargs:
    dt: 0.01
    dtmax: 0.05
    maxstep: 0.05
    downhill_check: true
  neb_optimizer: fire
  neb_optimizer_kwargs:
    dt: 0.01
    dtmax: 0.05
    maxstep: 0.05
    downhill_check: true
  neb_climb_optimizer: mdmin
  neb_climb_optimizer_kwargs:
    dt: 0.05
    maxstep: 0.01
  neb_method: improvedtangent
```

### FIRE downhill recovery for NEB

ASE's FIRE `downhill_check` compares the maximum potential energy along the
band before accepting a step. NEB, however, follows projected physical and
spring forces, which are not the negative gradient of that maximum-energy
scalar. An ordinary NEB step can therefore be useful even when the maximum
image energy rises. Repeated rejection of such steps restores the same
coordinates, zeros the FIRE velocity, and multiplies `dt` by `fdec`; the
visible symptom is many identical energy and `fmax` lines while `dt` approaches
zero.

AutoKMC treats `downhill_check: true` for ordinary FIRE NEB as a temporary
preconditioner:

1. FIRE begins with downhill checking enabled and retains ASE's normal rollback
   behavior.
2. AutoKMC watches the rollback callback and the live FIRE timestep.
3. After five rollback halvings, AutoKMC disables downhill checking, restores
   the stage's initial `dt`, and continues with the velocity reset performed by
   FIRE. The change is internal; no second optimizer or YAML setting is needed.
4. A warning containing `disabling it and restoring dt=` records the switch.

If FIRE is selected for CI-NEB, AutoKMC disables downhill checking immediately
because the climbing image is intentionally driven uphill. A separate MDMin
climbing stage is the conservative configuration shown above. When
`downhill_check` is already `false`, AutoKMC does not install the recovery
behavior.

This recovery prevents a zero-timestep loop; it does not prove that a band is
chemically valid. Inspect the saved initial and optimized paths for image
continuity, overlaps, endpoint integrity, topology, energies, and forces before
accepting a barrier or transition state.

Numerical NEB non-convergence is likewise not a chemical stability result.
AutoKMC preserves the last-known band, leaves the lateral class undecided, and
omits only that reaction from the current rate-index sweep. Other valid
reactions remain available, so one exhausted optimizer cannot prevent the KMC
loop from starting. The undecided class is retried when its member is later
recomputed, and its failed structures are written as retryable diagnostics.
Endpoint or transition-state topology failures remain chemically invalid and
are excluded as before. If an optional bare-band warm start does not converge,
the target lateral NEB instead falls back to its configured interpolation while
the bare class stays undecided.

The supported algorithmic keywords depend on the selected optimizer and the
installed ASE version. Common controls are `maxstep` and `alpha` for BFGS;
`dt`, `maxstep`, `dtmax`, `Nmin`, `finc`, `fdec`, `astart`, `fa`, `a`, and
`downhill_check` for FIRE; and `dt` plus `maxstep` for MDMin. ASE lifecycle
keywords such as `restart` and `trajectory` are also forwarded, but a single
global path is reused by many relaxations and can collide in concurrent runs;
omit those keywords unless the path lifecycle is managed externally.

`neb_band_eval` controls how AutoKMC evaluates the band during each optimizer
step. The default `images` mode sends one image at a time through the calculator
leased by that NEB. The `batched` mode sends the whole band through one stacked
model call when the calculator supports it. This allows a single-GPU MLIP to
amortize dispatch and host-device overhead across the band. The NEB physics,
optimizer, and constraints do not change; only the force-evaluation access
pattern changes. An unsupported calculator logs a warning and uses `images`.

**When to use `batched`.** It helps when the calculator is a GPU machine-learned
potential and the barriers are real (multi-step NEBs). On UMA (`uma-s-1p2`) it
gives roughly 10x faster barriers with the barrier value unchanged to within
model round-off. It does nothing useful for near-instant CPU calculators such as
EMT. When a run is left on the `images` default but the calculator would support
batching, a one-time `INFO` hint is logged suggesting the flag.

**Which calculators batch.** FAIR-Chem calculators (UMA and its sibling
checkpoints) are batched automatically. Any other calculator is batched if it
exposes an `evaluate_band(images)` method returning one `(energy, forces)` pair
per image; otherwise the run falls back to `images`. To add a new model, either
give its calculator an `evaluate_band` method (the `CallableBandEvaluator` path)
or add a small evaluator class in `autokmc/sites/stability/band_eval.py` next to
`FairChemBandEvaluator`. The evaluator returns *raw* model forces per image;
`FixAtoms` and all NEB projections are applied afterward by the unchanged ASE
path, so a new backend only has to answer "energy and forces for these
structures in one call."

## `structure`

| Key | Default | Meaning |
| --- | --- | --- |
| `kind` | `surface` | `surface`, `nanoparticle`, or `file`. |
| `path` | `null` | Catalyst file for `kind: file`. Relative paths resolve from the configuration file's directory. |
| `format` | `null` | Optional ASE reader format. `null` lets ASE infer the format from `path`. |
| `index` | `-1` | Frame selected from a multi-frame catalyst file. ASE indexing is used; `-1` selects the last frame. |
| `frozen_indices` | `null` | Optional zero-based catalyst atom indices to constrain for `kind: file`. |
| `composition` | internal default `Cu` | Element/composition accepted by the structure builder. Set `Pt` for the supplied examples. |
| `crystal_structure` | `fcc` | Crystal structure passed to the builder. |
| `miller_index` | `[1, 1, 1]` | Three-index surface orientation. |
| `lattice_constant` | `null` | Explicit lattice parameter or builder-specific value; `null` requests calculator/reference handling. |
| `min_slab_size` | `8.0` Å | Minimum slab thickness. |
| `min_vacuum_size` | `12.0` Å | Minimum vacuum thickness. |
| `goal_x`, `goal_y` | `12.0` Å | Target lateral dimensions. |
| `n_freeze_layers` | `2` | Number of bottom layers to constrain. |
| `surface_side` | `top` | Slab face classified as exposed: `top`, `bottom`, or `both`. |
| `surface_radius_factor` | `1.0` | Covalent-radius multiplier for slab ray-casting discs. |
| `nanoparticle_hull_tolerance_factor` | `0.5` | Covalent-radius multiplier for nanoparticle hull-atom classification tolerance. |
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

Positive sizes and force thresholds are required for built structures.
`n_freeze_layers` may be zero. Unconverged structure optimization is an error.

Surface slabs use the shortest equivalent in-plane lattice vectors before
tiling to `goal_x` and `goal_y`, so an integer-sheared primitive cell from
pymatgen does not change the repeat counts. This basis reduction preserves
the layers and vacuum; `extra_kwargs.orthogonalise: false` still retains
naturally skewed surfaces such as FCC(111).

File-backed structures use any single-frame or multi-frame format supported by
the installed ASE version:

```yaml
structure:
  kind: file
  path: ./structures/catalyst.extxyz
  index: -1
  # format: extxyz
  frozen_indices: [0, 1, 2, 3]
```

AutoKMC does not rebuild or relax a file-backed catalyst. It first loads the
selected frame and detaches any serialized calculator. For a periodic slab, it
then detects the two connected lattice directions and, when necessary, applies
one rigid rotation that aligns the surface normal with Cartesian +z. This
preserves the cell Gram matrix, interatomic geometry, atom order, and frozen
atoms for skew and arbitrarily oriented cells. The catalyst provenance records
the original surface normal and the applied transform. Subsequent chemistry
uses the configured calculator.

The `path` is required, and the file must provide cell vectors and
periodic-boundary metadata suitable for surface classification. Explicit
`frozen_indices` are checked against the selected atom count and become the
portable frozen mask used by chemistry, KMC, checkpoints, and trajectory
output. Omit the field to combine `atoms.info["frozen_indices"]` with ASE
`FixAtoms` constraints from the selected frame. Other constraint types do not
mark atoms as fully frozen. Set `frozen_indices: []` to discard imported frozen
metadata. During preflight, AutoKMC resolves the source path, parses the frame,
and reports the resolved path, frame index, atom count, and frozen count before
it checks the calculator.

## `reactants`

`reactants` is a list. `smiles` is required for each entry.

| Key | Default | Meaning |
| --- | --- | --- |
| `smiles` | required | Input SMILES; canonicalized internally. |
| `add_hydrogens` | `true` | Add implicit hydrogens during molecular construction. |
| `relax_in_gas` | `true` | Relax gas geometry. If false, a finite single-point energy is still computed. |
| `fmax` | `0.05` eV/Å | Force threshold for the optional gas-phase relaxation. |
| `max_steps` | `500` | Optimizer step limit for the optional gas-phase relaxation. |
| `partial_pressure_bar` | `null` | Species partial pressure; when omitted, inherits `free_energy.pressure_bar`. Zero prevents gas adsorption. |
| `symmetry_number` | `null` | Optional ideal-gas symmetry-number override. By default it is inferred from the final gas geometry with pymatgen. |
| `spin` | `null` | Spin value used by ideal-gas thermochemistry. |
| `geometry` | `null` | `auto`, `linear`, `nonlinear`, or `monatomic`. |

Feed reactants define which desorbing gas species are excluded from the strict
post-processing product definition. Species auto-built from bond templates
are surface-generated leaf species and receive zero gas pressure.

## `calculator`

Exactly one calculator construction style is required:

```yaml
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
```

or:

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

| Key | Default | Meaning |
| --- | --- | --- |
| `import_path` | `null` | Dotted calculator class. |
| `kwargs` | `{}` | Class constructor arguments. |
| `factory` | `null` | Dotted factory callable; mutually exclusive with `import_path`. |
| `factory_kwargs` | `{}` | Factory arguments. |
| `copies` | `1` | Number of independent calculator instances. Independent site/NEB work and vibration displacements share this pool; one NEB holds one copy for its full lifecycle. |
| `gpu_devices` | `null` | Optional device list assigned across copies. |
| `gpu_device_arg` | `device` | Constructor/factory argument that receives a device. |
| `max_workers` | `null` | Maximum concurrent calculator tasks; defaults to the number of copies. |

These settings define two levels of concurrency. First, `copies` creates
independent calculator objects, and `max_workers` limits the number of
concurrent AutoKMC tasks. Separate NEBs may run concurrently, but each NEB keeps
one calculator for its complete lifecycle and does not distribute its images
across the pool. Second, calculator-specific factory arguments control
parallelism inside one calculator.

For FAIR-Chem UMA, configure one AutoKMC copy per GPU and pass each ordinal to
the nested `get_predict_unit_on_device` factory. The helper selects that ordinal
while constructing the `device: cuda` predictor and verifies the resolved
device. It requires one FAIR-Chem worker so each calculator remains on one
device. FAIR-Chem's own `workers` option distributes one predictor calculation
internally and should not be combined with per-GPU AutoKMC copies.

Every configuration must set exactly one of `calculator.import_path` or
`calculator.factory`. Omitting both is a validation error. EMT is used only
when selected explicitly.

`import_path` and `factory` are mutually exclusive. Their corresponding
argument fields must be mappings, `max_workers` cannot exceed `copies`, and a
non-empty `gpu_devices` list must contain one unique string per calculator
copy. The same mapping and callable-declaration checks apply to nested generic
factory specs inside calculator arguments.

Local files and directories nested anywhere in `kwargs` or `factory_kwargs`
are identified by their contents for cache and resume compatibility. Remote
model names cannot be inspected, so AutoKMC retains those aliases literally.
Use an immutable model revision, commit, or digest rather than a mutable alias
such as `latest`; otherwise a remote artifact could change without the local
cache or resume contract being able to detect it.

## `adsorption`

| Key | Default | Meaning |
| --- | --- | --- |
| `prune_stable_only` | `true` | Optimize one representative per candidate class with the potential and retain stable classes. Each representative first undergoes a fixed-slab rigid-molecule optimization, then the ordinary relaxed optimization. |
| `prune_fmax` | `0.05` eV/Å | Convergence threshold for each stable-site optimization stage. The rigid stage tests net translation and length-scaled torque; the relaxed stage tests free-atom forces. |
| `prune_max_steps` | `500` | Per-stage stable-site optimization limit; each candidate may use this many rigid steps followed by this many relaxed steps. |
| `endpoint_fmax` | `0.05` eV/Å | Occupied/unoccupied endpoint threshold used to construct adsorption/desorption rates. |
| `endpoint_max_steps` | `200` | Occupied/unoccupied endpoint step limit. |
| `anchor_k_max` | `4` | Maximum anchor clique size. Four covers atop, bridge, three-fold, and four-fold sites while bounding dense-graph enumeration; set to `null` for legacy unbounded enumeration. |
| `n_shells_anchor` | `null` | Anchor-environment graph depth; `null` selects it automatically from molecular reach. |
| `pair_n_shells` | `1` | Local graph depth used to classify multi-anchor molecular placements. |
| `max_pair_shells` | `10` | Maximum allowed surface-graph path length between anchors in one placement. |

## `kmc`

| Key | Default | Meaning |
| --- | --- | --- |
| `temperature_k` | `500.0` K | KMC temperature; must be positive. |
| `n_steps` | `1000` | Steps to execute in this invocation; may be zero. |
| `transmission_coefficient` | `1.0` | Non-negative Eyring coefficient. |
| `log_every` | `100` | Concise KMC progress cadence; `0` disables step messages. |
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
| `n_images` | `10` | Fixed interior-image count when `image_spacing` is `null`; total frames add two endpoints. |
| `image_spacing` | `0.25` Å | Enable dynamic image selection and limit every adjacent-image atom displacement to three times this value. |
| `min_images` | `6` | Minimum dynamically selected interior-image count, giving at least eight total frames. |
| `max_images` | `8` | Maximum dynamically selected interior-image count, giving at most ten total frames. |
| `climb` | `true` | Refine the ordinary NEB with a climbing image unless either raw directional barrier is below 0.1 eV. |
| `spring_k` | `5.0` eV/Å² | NEB spring constant used for both optimization stages. |
| `interpolation` | `linear` | `linear` or `idpp`. |
| `persist_neb_path` | `false` | Save both image sequences for successful runs; failed NEBs retain their initial and last-known bands automatically. |

Dynamic image selection begins after both endpoints are relaxed and their atoms
are paired. AutoKMC first finds `d_max`, the largest minimum-image displacement
between corresponding atoms. It then calculates
`ceil(d_max / image_spacing) - 1`, clips the result to
`min_images`/`max_images`, and adds the two endpoint frames. Set
`image_spacing: null` to use exactly `n_images`. If the maximum bound limits the
count, the persisted `estimated_linear_spacing_ang` can exceed the requested
target and `count_limited_by` becomes `maximum`. The bond channel follows the
same sequence through `neb_image_spacing`, `neb_min_images`, `neb_max_images`,
and `neb_n_images`.

Dynamic spacing also supplies a geometric guard during both the ordinary and
climbing-image stages. After every optimizer step, AutoKMC measures the
MIC-aware displacement of every unfrozen atom between adjacent images. If any
displacement exceeds `optimization.neb_geometry_guard_multiplier` times the
configured spacing, the step is rejected and
the lowest-force geometrically valid band from that stage is restored. A fresh
optimizer resets its velocities and continues within the original step budget.
For FIRE, both `dt` and `dtmax` are halved on every geometric restart. MDMin
halves `dt`; BFGS, which has no timestep, halves `maxstep`. Set the spacing to
`null` to use a fixed image count without this derived geometric limit.

## `bond`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable bond-changing channels. |
| `bond_max_hops` | `0` | Maximum surface separation for A/B candidates. |
| `bond_types` | `["SINGLE", "DOUBLE", "TRIPLE"]` | Allowed RDKit bond types. |
| `include_ring_bonds` | `false` | Include ring-bond transformations. |
| `include_homo_coupling` | `true` | Permit A + A coupling templates. |
| `include_dissociation` | `true` | Generate C to A + B templates. |
| `include_coupling` | `true` | Generate A + B to C templates. |
| `deduplicate_iso` | `true` | Deduplicate graph-isomorphic classes. |
| `gas_lift_height` | `6.0` Å | Staging lift used to align a gas product above the reacting site. |
| `gas_precursor_relax` | `true` | For gas products, relax an intact adsorbed molecular precursor before the bond NEB. The slab and lateral environment are fixed for this step. |
| `gas_precursor_distance` | `2.5` Å | Initial minimum molecule-to-slab distance for the precursor relaxation. |
| `auto_build_leaf_species` | `true` | Build implied species absent from the feed list. |
| `pair_n_shells` | `1` | Ego-graph depth for triple pruning. |
| `prune_by_triple` | `true` | Keep the preferred stable class per adsorption triple. |
| `prune_with_calculator` | `true` | Relax and reject unstable A+B endpoints. |
| `prune_fmax` | `0.05` eV/Å | Bond-pruning threshold. |
| `prune_max_steps` | `500` | Bond-pruning step limit. |
| `neb_fmax` | `0.01` eV/Å | Bond NEB threshold. |
| `neb_max_steps` | `200` | Bond NEB step limit. |
| `neb_n_images` | `10` | Fixed bond-NEB interior-image count when `neb_image_spacing` is `null`. |
| `neb_image_spacing` | `0.25` Å | Enable dynamic bond-NEB selection and limit every adjacent-image atom displacement to three times this value. |
| `neb_min_images` | `6` | Minimum dynamically selected bond-NEB interior-image count, giving at least eight total frames. |
| `neb_max_images` | `8` | Maximum dynamically selected bond-NEB interior-image count, giving at most ten total frames. |
| `neb_climb` | `true` | Refine the ordinary bond NEB with a climbing image unless either raw directional barrier is below 0.1 eV. |
| `neb_spring_k` | `5.0` eV/Å² | Bond NEB spring constant used for both optimization stages. |
| `neb_interpolation` | `idpp` | `linear` or `idpp`. |
| `atom_matching` | `auto` | Atom correspondence seed (`auto`, `greedy`, `hungarian`, or `reactant_index`); all modes preserve unchanged A/B bond connectivity. |
| `matching_trials` | `8` | Maximum connectivity-preserving mapping trials used to minimize endpoint displacement. |
| `persist_neb_path` | `false` | Save both bond NEB paths for successful runs; failed NEBs retain them automatically. |

For a gas-fed bond dissociation such as H2(g) to 2H*, AutoKMC does not
interpolate directly from a distant gas molecule to the dissociated adsorbates.
It first aligns the intact molecule above the reacting site and lowers it to
`gas_precursor_distance`. It then fixes the slab and lateral adsorbates and
relaxes the molecule. The relaxed precursor becomes the physical endpoint of
the bond NEB. For KMC thermodynamics, `energy_c` remains the sum of the empty
surface and gas-phase energies. The separate precursor energy is stored as
`energy_c_precursor` and `energies_ev.state_c_precursor` for diagnostics.

When lateral interactions are enabled, diffusion and bond channels
automatically retain the optimized no-neighbour NEB band as an internal
warm-start asset. Before evaluating a lateral class with a neighbouring
adsorbate, AutoKMC runs the corresponding bare calculation if no compatible
band exists, projects that band into the new endpoint layout, and reoptimizes
all images. If the bare calculation fails or its band is incompatible, the
channel uses its configured interpolation. This behavior is automatic and
does not change the output-only `persist_neb_path` setting.

## `free_energy`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Enable vibrational/ideal-gas corrections. |
| `pressure_bar` | `1.0` bar | Default partial pressure for every reactant that omits `partial_pressure_bar`; may be zero. |
| `vibration_displacement` | `0.01` Å | Finite-difference displacement. |
| `vibration_nfree` | `2` | Must be `2` or `4`. |
| `include_ts_vibrations` | `true` | Compute transition-state vibrations. |
| `min_frequency_ev` | `0.0015` eV | Low-frequency floor used by thermochemistry. |
| `symmetry_tolerance` | `0.3` Å | Cartesian tolerance used by pymatgen for molecular point-group and rotational-symmetry inference. |
| `default_spin` | `0.0` | Gas spin fallback. |
| `default_geometry` | `auto` | `auto`, `linear`, `nonlinear`, or `monatomic`. |
| `cache_dir` | `null` | Persistent vibration-cache root; defaults below the run directory. Completed content-addressed displacement calculations are reused after interruption. |

`free_energy.pressure_bar` is a feed-wide fallback despite its placement in the
`free_energy` section, and it applies even when `free_energy.enabled` is false.
Each explicit `reactants[].partial_pressure_bar` takes precedence. Gas-phase
thermochemistry always uses a fixed standard-state pressure of 1 bar; the
resolved partial pressure is applied separately to adsorption rates through
the ideal-gas activity `p / p°`.

Free-energy work can dominate runtime. The supplied platinum GPU examples
disable it intentionally for network-debug runs and can be switched on for
production thermochemistry. For multi-atom gas species, AutoKMC records the
inferred rotational symmetry number, point group, tolerance, and inference
source in the reactant thermochemistry metadata and run manifest. Set a
reactant-specific `symmetry_number` only when an explicit override is
scientifically necessary.

## `checkpoint`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Write periodic checkpoints. |
| `path` | `null` | Checkpoint path; defaults to `OUTPUT_DIR/checkpoint.pkl`. |
| `every_n_steps` | `1` | Checkpoint cadence. |
| `resume_from` | `null` | Checkpoint to continue. |

On resume, AutoKMC first reconciles the event and trajectory files to the
checkpoint. It atomically removes trajectory frames beyond the checkpoint step
and rejects malformed or non-monotonic committed `kmc_step` metadata. It then
restores the reaction-folder counters, reconstructs the cumulative summary from
`events.jsonl`, and continues with the saved NumPy or Python RNG state.
