# `autokmc2` Package Structure

Implementation status: in progress.

This document records the package restructuring from the old flat `autokmc` package into the explicit-domain `autokmc2` package. The goal is to preserve the current functional pipeline while making responsibilities explicit, reducing circular/import coupling, and creating a safer path for future reaction types such as transfer reactions and checkpoint/restart support.

## Final decisions

- The old flat `autokmc` import surface is not supported in the new package.
- User-facing and internal imports should be explicit domain imports, e.g. `autokmc2.structure`, `autokmc2.sites.adsorbate`, `autokmc2.kmc.engine`.
- Historical `autokmc/dev/` workflows are not part of the supported package.
- Raw `networkx.Graph.graph[...]` dictionaries remain exposed for now to keep development and notebook experimentation flexible.
- `autokmc2/` is the full new package. The old `autokmc/` tree is obsolete and excluded from packaging.

## Executive summary

The package is currently functional but largely flat. Most major concepts live as top-level modules in `autokmc/`, and several files mix data models, graph helpers, enumeration logic, stability checks, kinetic rate construction, persistence, and CLI orchestration.

The implemented restructuring moves toward domain subpackages:

- `structure/` for structure builders, slab/nanoparticle construction, optimization, and surface classification.
- `core/` for atom graphs, graph-state accessors, constants, and shared result models.
- `species/` for gas-phase reactants and molecular bond manipulation.
- `sites/` for static site enumeration and lateral/stability helpers.
- `reactions/` for adsorption, diffusion, bond-changing, rates, and reaction data models.
- `kmc/` for the KMC engine, state mutation, rate index, and event execution.
- `io/` for config, persistence, trajectories, summaries, and checkpointing.
- `thermo/` for vibrational/free-energy calculations.
- `cli/` for command-line parsing and pipeline orchestration.
- `utils/` for logging and small shared utilities.
- No compatibility facade or top-level shims are provided for old flat imports.

The most important design rule is that modules should depend inward/downward: low-level domain models and helpers should not import CLI, persistence, or the full package facade.

## Current structure: observed responsibilities

Current top-level package modules are mostly organized by development history rather than architectural boundaries.

| Current module | Main responsibility today | Structural issue |
| --- | --- | --- |
| `structure.py` | Build and optimize bulk, slabs, nanoparticles; alloy handling; Wulff and pymatgen integration. | Large builder/optimization module; many independent concerns in one file. |
| `surface.py` | Surface atom classification and tagging. | Reasonable boundary, belongs with core geometry/structure. |
| `graph.py` | Convert ASE `Atoms` to `networkx.Graph`; defines graph metadata convention. | Core module; graph-state conventions are used everywhere but not centralized as typed accessors. |
| `results.py` | Lightweight result dataclasses. | Core/shared models. |
| `reactants.py` | SMILES to 3-D `Reactant`, anchor atom metadata, uniqueness. | Belongs to species/chemistry domain. |
| `molecular_bond_changing.py` | Fragment/couple molecules and generate changed molecular species. | Belongs beside `reactants.py`, not next to KMC engine files. |
| `find_anchors.py` | Anchor-site enumeration plus many geometry/isomorphism/private helpers. | Core site-enumeration logic; helpers are imported privately by other modules. |
| `find_adsorbate_sites.py` | Adsorbate-site dataclasses, multi-atom placement enumeration, pruning and optimization glue. | Very large module; mixes data models, geometry helpers, graph mutation, and stability pruning. |
| `check_adsorbate_sites.py` | Lateral environment classification and adsorption stability calculations. | Uses shared lateral graph concepts that diffusion and bond checks also need. |
| `find_diffusion_sites.py` | Diffusion pair dataclasses and static pair enumeration. | Natural part of site/reaction domain; imports private adsorbate helpers. |
| `check_diffusion_sites.py` | Diffusion lateral classification, endpoint stability, NEB transition-state validation. | Similar shape to adsorbate/bond stability; should share abstractions. |
| `find_bond_sites.py` | Bond-reaction templates, bond reaction site dataclasses, static triple enumeration. | Mixes template chemistry and surface site enumeration; imports several private helpers. |
| `check_bond_sites.py` | Bond-reaction lateral classification, endpoint stability, NEB checks. | Similar to diffusion checks; should share transition-state/stability infrastructure. |
| `grow_bond_sites.py` | On-the-fly registry and expansion for new bond species. | KMC state/indexing concern plus site expansion concern. |
| `kmc_adsorption.py` | Adsorption/desorption reaction dataclass, rate constants, applicability, rate computation. | Reaction model/rate code, not engine code. Constants are imported by config. |
| `kmc_diffusion.py` | Diffusion reaction dataclass and applicability/rate computation. | Reaction model/rate code. |
| `kmc_bond.py` | Bond reaction dataclass and applicability/rate computation. | Reaction model/rate code. |
| `kmc_reactions.py` | Obsolete compatibility shim from the old package. | Archived only; not part of `autokmc2`. |
| `kmc_simulation.py` | Segment tree sampler, reaction indexing, KMC loop, reaction execution, incremental updates, persistence hooks. | Very large hub; should be split into engine, state mutation, indexing, and callbacks. |
| `free_energy.py` | Vibrational analysis and thermochemistry. | Clear domain module; should move under `thermo/`. |
| `config.py` | Config schema, parsing, dynamic calculator construction. | I/O/application layer; currently imports reaction constants. |
| `persistence.py` | Event JSONL, reaction folders, trajectories, summary helpers, graph-to-Atoms. | Large I/O module; knows details of all reaction object shapes. |
| `cli.py` | Argument parser plus full end-to-end pipeline driver. | CLI imports from package-level `autokmc`, causing broad eager imports and tighter coupling. |
| `constants.py` | Defaults and string formats. | Useful but broad; consider splitting domain constants later only if it improves clarity. |
| `logging_utils.py` | Logger helper. | Utility module. |
| `autokmc/dev/` | Exploratory scripts and workflows. | Removed from the supported package; promote maintained workflows to `examples/`. |
| `autokmc2/` | New explicit-domain package. | Full replacement for the old flat package. |

## Current coupling concerns

1. **Flat namespace hides boundaries.** Enumeration, stability checks, rate generation, engine execution, config, persistence, and CLI all sit at one level.
2. **Large hub modules.** `find_adsorbate_sites.py`, `kmc_simulation.py`, `find_bond_sites.py`, `persistence.py`, `structure.py`, and `cli.py` are large enough to make local changes risky.
3. **Private helper imports across domains.** Examples include helpers such as `_build_ego_graph`, `_kabsch_align_ego`, `_get_surface_apsp`, `_shortest_path_between_cliques`, `_member_clique_union`, and `_surface_bfs_shells` being reused from modules that conceptually should not own them.
4. **Shared mutable graph state is implicit.** Many stages read and write `G.graph[...]` keys such as `adsorbate_sites`, `diffusion_sites`, `bond_reaction_sites`, `occupied_by_clique`, shell caches, and cell/PBC metadata. This is powerful but fragile without centralized accessors or a schema.
5. **`__init__.py` eagerly imports almost everything.** The package facade currently re-exports a broad API, which increases import cost and raises the risk of circular imports. It also encourages internal modules such as `cli.py` to import from `autokmc` rather than from specific modules.
6. **Application concerns leak into domain code.** `config.py` imports `DEFAULT_TRANSMISSION_COEFFICIENT` from `kmc_adsorption.py`; `persistence.py` knows the shape of adsorption, diffusion, and bond reaction objects; `cli.py` orchestrates all scientific stages directly.
7. **Documentation drift.** `README.md` says there is no CLI and no tests, but `pyproject.toml` defines a CLI entry point and `tests/` exists. `pyproject.toml` points `readme = "AGENTS.md"`, but `AGENTS.md` is not present in the observed repository root.

## Target package layout

Recommended target tree:

```text
autokmc2/
  __init__.py                 # thin facade: version + curated stable exports only
  api.py                      # optional explicit public API for notebooks/scripts
  constants.py                # initially keep global defaults here
  py.typed

  structure/
    __init__.py
    builders.py               # common build helpers, composition/lattice utilities
    structure.py              # public build_surface/build_nanoparticle facade
    slab.py                   # slab-specific construction/orthogonalization
    nanoparticle.py           # nanoparticle/Wulff construction
    optimization.py           # optimise_bulk/optimise_structure/calculator copy policy
    surface.py                # surface atom classification/tagging

  core/
    __init__.py
    graph.py                  # ASE Atoms -> nx.Graph
    graph_state.py            # typed accessors for G.graph metadata and indexes
    constants.py              # shared package defaults
    results.py                # SurfaceClassification and similar core result models

  species/
    __init__.py
    reactant.py               # Reactant, build_reactant, anchor/unique atom helpers
    bond_chemistry.py         # fragment/couple molecules, CombinedSpecies, FragmentPair
    smiles.py                 # canonicalization/sanitization helpers if needed

  sites/
    __init__.py
    models.py                 # shared site/lateral protocols or base dataclasses
    geometry.py               # MIC, Kabsch, centroids, clique geometry helpers
    ego.py                    # ego-graph/isomorphism/fingerprint helpers
    anchors.py                # AnchorSite, find_anchor_sites
    adsorbate.py              # AdsorbateSite, find_adsorbate_sites
    adsorbate_refine.py       # rigid-body refinement and position pushing
    diffusion.py              # DiffusionSite, find_diffusion_sites
    bond.py                   # BondReactionSite and geometric triple enumeration
    lateral.py                # shared lateral-environment graph construction
    stability/
      __init__.py
      adsorption.py           # check_adsorbate_site_lateral/check_site_stability
      diffusion.py            # check_diffusion_site_lateral/check_diffusion_stability
      bond.py                 # check_bond_site_lateral/check_bond_site_stability
      neb.py                  # common NEB setup/validation helpers

  reactions/
    __init__.py
    models.py                 # common Reaction protocol/base fields if useful
    rates.py                  # KB_EV, H_EV_S, EA_MIN, Eyring helpers
    adsorption.py             # AdsorptionReaction + applicability/rate construction
    diffusion.py              # DiffusionReaction + applicability/rate construction
    bond.py                   # BondReaction + applicability/rate construction
    templates.py              # derive_dissociation/coupling/bond templates

  kmc/
    __init__.py
    sampling.py               # total_rate, sample_tau, choose_reaction, segment tree
    index.py                  # ReactionIndex / incremental leaf management
    state.py                  # occupancy, occupied_by_clique, affected-site lookup
    execute.py                # execute_reaction and graph mutation
    engine.py                 # run_kmc_steps main loop
    expansion.py              # on-the-fly species/bond-site growth currently in grow_bond_sites.py
    callbacks.py              # persistence/trajectory hooks or observer protocol

  thermo/
    __init__.py
    free_energy.py            # FreeEnergyOptions and thermo calculations
    vibrations.py             # optional split if free_energy.py keeps growing

  io/
    __init__.py
    config.py                 # dataclass config schema + load_config
    calculators.py            # dynamic ASE calculator construction + metadata
    persistence.py            # ReactionWriter or facade
    records.py                # ReactionRecord, serializable metadata models
    trajectory.py             # TrajectoryWriter
    summary.py                # ReactionSummary/make_run_meta
    atoms.py                  # atoms_from_graph
    checkpoint.py             # future checkpoint/restart support

  cli/
    __init__.py
    main.py                   # argparse entry point
    pipeline.py               # run_from_config orchestration

  utils/
    __init__.py
    logging.py

examples/
  configs/
  co_cu111_allegro.yaml
    ...
```

### No compatibility shims

The new package intentionally does **not** provide old flat imports. Code should be updated to explicit domain imports under `autokmc2.*`. The obsolete `kmc_reactions.py` compatibility shim from the old package is not part of `autokmc2`.

## Proposed dependency direction

Aim for this import direction:

```text
cli -> io/config + pipeline -> core/species/sites/reactions/kmc/thermo/io
io/persistence -> reaction/site protocols + core graph-to-Atoms helpers
kmc -> reactions + sites + core.graph_state
reactions -> sites.stability + species + thermo/rates
sites.stability -> sites models/helpers + core.graph_state + thermo
sites enumeration -> core graph helpers + species models
species -> external chemistry libraries only
core -> ASE/networkx/numpy/scipy/pymatgen/wulffpack, but not sites/reactions/kmc/io/cli
utils/constants -> no domain imports
```

Avoid these patterns:

- Internal modules importing from top-level `autokmc` facade.
- Config importing implementation constants from reaction modules.
- Persistence relying on many concrete reaction classes instead of a small serialization protocol.
- Site modules importing private helpers from sibling modules that merely happened to define them first.

## Suggested file moves and splits

### 1. Structure and graph

Move or split:

- `structure.py` -> `structure/structure.py`, `structure/slab.py`, `structure/nanoparticle.py`, `structure/optimization.py`, `structure/builders.py`.
- `surface.py` -> `structure/surface.py`.
- `graph.py` -> `core/graph.py`.
- `results.py` -> `core/results.py`.

Add `core/graph_state.py` to centralize graph metadata keys and occupancy indexes. This can start small with functions such as:

- `get_cell(G)` / `get_pbc(G)`
- `get_adsorbate_sites(G, smiles=None)`
- `get_diffusion_sites(G, smiles=None)`
- `get_bond_reaction_sites(G)`
- `get_occupied_by_clique(G)`
- `invalidate_surface_shell_cache(G)`

This should reduce direct `G.graph[...]` access over time.

### 2. Species chemistry

Move:

- `reactants.py` -> `species/reactant.py`.
- `molecular_bond_changing.py` -> `species/bond_chemistry.py`.

Move SMILES canonicalization currently embedded in `find_bond_sites.py` into `species/smiles.py` or `reactions/templates.py`.

### 3. Site enumeration

Move and split:

- `find_anchors.py` -> `sites/anchors.py` plus helper modules:
  - `sites/geometry.py` for `_get_cell`, MIC distance/unwrap, centroids, Kabsch alignment.
  - `sites/ego.py` for ego graph construction, fingerprints, and isomorphism helpers.
- `find_adsorbate_sites.py` ->
  - `sites/adsorbate.py` for `AdsorbateSite`, `AdsorbateSiteLateral`, and enumeration.
  - `sites/adsorbate_refine.py` for rigid-body optimization and graph position updates.
- `find_diffusion_sites.py` -> `sites/diffusion.py`.
- `find_bond_sites.py` -> split into:
  - `reactions/templates.py` for `BondReactionTemplate` and template derivation.
  - `sites/bond.py` for `BondReactionSite`, `BondReactionLateral`, and surface triple enumeration.

The key cleanup is to make shared private helpers public-within-package helpers under `sites/geometry.py`, `sites/ego.py`, and `sites/lateral.py` rather than importing `_private` names across modules.

### 4. Stability and transition-state checks

Move:

- `check_adsorbate_sites.py` -> `sites/stability/adsorption.py`.
- `check_diffusion_sites.py` -> `sites/stability/diffusion.py`.
- `check_bond_sites.py` -> `sites/stability/bond.py`.

Extract shared code into:

- `sites/lateral.py` for lateral ego-graph generation and lateral fingerprints.
- `sites/stability/neb.py` for NEB setup, interpolation, endpoint-collapse checks, TS validity, and optional path persistence.
- `sites/stability/connectivity.py` if adsorption/diffusion/bond checks share connectivity comparison logic.

### 5. Reaction construction and rates

Move:

- `kmc_adsorption.py` -> `reactions/adsorption.py`.
- `kmc_diffusion.py` -> `reactions/diffusion.py`.
- `kmc_bond.py` -> `reactions/bond.py`.

Extract constants/rate helpers:

- `KB_EV`, `H_EV_S`, `EA_MIN`, `DEFAULT_TRANSMISSION_COEFFICIENT` -> `reactions/rates.py` or `constants.py`.

This removes the config-layer dependency on `kmc_adsorption.py`.

### 6. KMC engine

Split `kmc_simulation.py` into:

- `kmc/sampling.py`: `total_rate`, `sample_tau`, `choose_reaction`, `_RateSegmentTree`.
- `kmc/index.py`: `_ReactionIndex` and leaf bookkeeping.
- `kmc/state.py`: occupancy toggles, `occupied_by_clique`, affected site/member lookup.
- `kmc/execute.py`: `execute_reaction`.
- `kmc/engine.py`: `run_kmc_steps`.
- `kmc/expansion.py`: registry and on-the-fly growth currently in `grow_bond_sites.py`.

No old flat KMC wrapper is provided; callers should import `autokmc2.kmc.engine` directly.

### 7. I/O, config, persistence, CLI

Move and split:

- `config.py` -> `io/config.py` and `io/calculators.py`.
- `persistence.py` -> `io/persistence.py`, `io/records.py`, `io/trajectory.py`, `io/summary.py`, `io/atoms.py`.
- `cli.py` -> `cli/main.py` and `cli/pipeline.py`.

`cli/pipeline.py` should import concrete modules directly rather than using a package-level facade.

For persistence, introduce a small reaction serialization protocol instead of checking many concrete reaction shapes. For example, each reaction type could expose or adapt to a `ReactionSnapshot` dataclass containing:

- `kind`
- `species_label` or reaction-process label
- `iso_class`
- `member_index`
- `lateral_class`
- `rate`
- `delta_e`
- `barrier`
- optional `delta_g` / `barrier_g`
- structure snapshots keyed by semantic names such as `occupied`, `unoccupied`, `state_a`, `state_b`, `ts`.

That lets `ReactionWriter` serialize uniformly.

### 8. Thermochemistry

Move:

- `free_energy.py` -> `thermo/free_energy.py`.

If it keeps growing, split ASE vibration orchestration into `thermo/vibrations.py` and keep thermodynamic correction calculations in `thermo/free_energy.py`.

### 9. Dev scripts and examples

Remove old `autokmc/dev/` scratch workflows from the import package. Promote any maintained workflows to tested examples under `examples/` instead:

```text
examples/
  configs/
  workflows/
  get_bond_reactions.py
  ...
```

These workflows are preserved for reference only, not as supported package modules.

## Recommended migration phases

### Phase 0: Public API decision

Final decision: use explicit `autokmc2.<domain>.<module>` imports only. No flat facade and no old-path compatibility shims.

### Phase 1: Mechanical package creation without shims

Create the new directories and move files with minimal internal edits. Do not leave top-level wrappers.

Goal: no behavior changes.

Recommended first moves:

1. `logging_utils.py` -> `utils/logging.py`.
2. `structure.py`, `surface.py` -> `structure/`.
3. `results.py`, `graph.py` -> `core/`.
4. `reactants.py`, `molecular_bond_changing.py` -> `species/`.
5. `free_energy.py` -> `thermo/`.
6. `config.py`, `persistence.py` -> `io/` wrappers without splitting internals yet.

### Phase 2: Extract shared helpers

Move cross-imported private helpers into real helper modules:

- `sites/geometry.py`
- `sites/ego.py`
- `sites/lateral.py`
- `sites/stability/neb.py`
- `core/graph_state.py`

This phase should include targeted unit tests for helper behavior before large algorithm modules are changed.

### Phase 3: Split the largest modules

Split these carefully, one at a time:

1. `find_adsorbate_sites.py`
2. `kmc_simulation.py`
3. `persistence.py`
4. `find_bond_sites.py`
5. `structure.py`
6. `cli.py`

For each split:

- Move dataclasses first.
- Move pure helpers second.
- Move state-mutating workflow functions last.
- Do not keep wrapper imports at old paths.
- Run tests after each split.

### Phase 4: Clean dependency direction

After files are moved:

- Replace internal old-package imports with explicit `autokmc2` domain imports.
- Move rate constants into `reactions.rates` so config does not import reaction implementation.
- Keep direct `G.graph[...]` access for now, per the development decision.
- Introduce a persistence serialization adapter/protocol.

### Phase 5: Documentation and packaging cleanup

Update:

- `README.md`: current CLI and tests now exist, so remove stale statements.
- `pyproject.toml`: `readme = "AGENTS.md"` should point to an existing file, probably `README.md`, unless `AGENTS.md` is restored.
- Examples/scripts paths after moving `autokmc/dev/`.
- Any docs that mention old module names.

### Phase 6: Old package cleanup

The old `autokmc/` package is obsolete and excluded from packaging. If filesystem deletion is needed, remove it after confirming all imports and workflows use `autokmc2`.

## Testing plan for the restructuring

Current tests mostly cover config, CLI validation, persistence, summaries, and trajectories. During restructuring, add tests in the same shape as the new package layout.

Suggested test tree:

```text
tests/
  core/
    test_graph.py
    test_surface.py
    test_graph_state.py
  species/
    test_reactant.py
    test_bond_chemistry.py
  sites/
    test_geometry.py
    test_ego.py
    test_anchors.py
    test_adsorbate_models.py
    test_diffusion_models.py
    test_bond_models.py
    stability/
      test_lateral.py
      test_neb_validation.py
  reactions/
    test_rates.py
    test_adsorption.py
    test_diffusion.py
    test_bond.py
    test_templates.py
  kmc/
    test_sampling.py
    test_index.py
    test_execute.py
  io/
    test_config.py
    test_persistence.py
    test_trajectory.py
    test_summary.py
  cli/
    test_main.py
    test_pipeline_smoke.py
  test_import_compat.py
```

High-priority additions:

1. **Import boundary tests** that confirm old flat modules are not provided and new explicit-domain imports work.
2. **Curated API tests** for selected package-level imports if a small facade is added later.
3. **Graph-state tests** for occupancy indexes and cache invalidation.
4. **Rate/sampling tests** for segment tree and `choose_reaction` edge cases.
5. **Persistence adapter tests** so adsorption/diffusion/bond records serialize through the same interface.
6. **CLI pipeline smoke test** with a tiny/cheap EMT-backed configuration if runtime permits.

## API recommendation

Use explicit domain imports:

1. **User-facing workflow imports**: stable functions from their domain modules, such as `autokmc2.structure` and `autokmc2.kmc.engine`.
2. **Contributor imports**: concrete submodules such as `autokmc2.sites.adsorbate.find_adsorbate_sites`.
3. **Internal helpers**: underscore-prefixed names inside subpackages, not imported across unrelated domains.

Example target user imports:

```text
from autokmc2.structure import (
    build_surface,
    find_surface_atoms,
)
from autokmc2.core.graph import build_graph
from autokmc2.species.reactant import build_reactant
from autokmc2.sites.anchors import find_anchor_sites
from autokmc2.sites.adsorbate import find_adsorbate_sites
from autokmc2.kmc.engine import run_kmc_steps
```

Example target internal imports:

```text
from autokmc2.core.graph_state import get_adsorbate_sites, get_occupied_by_clique
from autokmc2.sites.lateral import build_lateral_ego_graph
from autokmc2.reactions.rates import eyring_rate
```

## Specific cleanup opportunities

- Move `DEFAULT_TRANSMISSION_COEFFICIENT`, `KB_EV`, `H_EV_S`, and `EA_MIN` into `reactions/rates.py` or `constants.py`.
- Turn `ReactionWriter` into a serializer of a common reaction snapshot rather than a type-specific inspector.
- Centralize graph keys as constants or accessors. This will reduce silent key typos and make checkpointing easier.
- Stop importing from old `autokmc` paths inside `cli.py` and other internal modules.
- Move `autokmc/dev/` to `scripts/dev/` or `examples/workflows/`.
- Keep `autokmc2/` as the supported package root.
- Update documentation references to `AGENTS.md`, `REVIEW.md`, and `autokmc/todo.MD` if those files do not exist in the repository.
- Consider renaming `example/` to `examples/` unless external workflows depend on the current name.

## Clarification questions

1. **Public API stability:** Are users already importing from the current flat modules, or is this mainly internal research code where imports can be changed aggressively?
2. **Deprecation policy:** Should old module paths remain supported indefinitely, for one release cycle, or only during the refactor branch?
3. **Preferred user-facing API:** Should a small `autokmc2.api` facade be added later, or should users continue importing from explicit domain modules such as `autokmc2.sites.adsorbate`?
4. **Dev scripts:** Should `autokmc/dev/` be treated as private scratch scripts, promoted to tested examples, or moved to a separate repository/workflow folder?
5. **Graph state:** Are you open to wrapping `networkx.Graph` state in typed accessor/state objects, or should the code continue to expose raw `G.graph[...]` dictionaries for notebook convenience?
6. **Reaction extensibility:** Should the new structure explicitly reserve first-class space for future `transfer` reactions now, or wait until the design is clearer?
7. **Checkpointing:** Since `TODO.md` mentions checkpointing, should `io/checkpoint.py` be designed during this refactor, or postponed until after the package split?
8. **Documentation source of truth:** Should `README.md` become the canonical package overview, and should the missing/stale `AGENTS.md` references be removed or restored?
9. **Testing budget:** How much slow scientific workflow testing is acceptable in CI versus keeping only unit/smoke tests and running heavy EMT/ML workflows manually?
