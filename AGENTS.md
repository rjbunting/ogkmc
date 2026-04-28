# autokmc — AI agent guide

Surface KMC input generation + Gillespie/BKL engine. Pure-Python library
built on ASE `Atoms` + `networkx.Graph`. Calculator-agnostic (EMT, NequIP,
MACE, VASP, …).

## Pipeline (must run in this order)

`structure.build_surface | build_nanoparticle` → `surface.find_surface_atoms(atoms, tag_atoms=True)` (writes int8 `atoms.arrays["surface"]`: 0=bulk/1=surface/2=adsorbate) → `graph.build_graph(atoms)` → `reactants.build_reactant(smiles)` → `find_anchors.find_anchor_sites(G, element)` → `find_adsorbate_sites.find_adsorbate_sites(G, reactant)` → `find_adsorbate_sites.optimise_adsorbate_site_positions(...)` → `kmc_simulation.run_kmc_steps(...)`.

`autokmc/__init__.py` re-exports every public symbol; mirror it when adding new pipeline entry points. The same flow is driven from YAML/TOML by `cli.run_from_config` (see `example/co_cu111_allegro.yaml`).

## The graph is the single source of truth

Every stage reads/writes the same `nx.Graph G`:
- Node attrs: `element`, `position` (np.ndarray (3,)), `index` (orig ASE idx), `type ∈ {"bulk","surface","adsorbate","anchor"}`, `covalent_radius`. Adsorbate nodes additionally carry an `occupied: bool` toggled by `kmc_simulation.execute_reaction`.
- `"anchor"` nodes are *invisible bookkeeping* nodes added by `find_anchors`: they hold a clique of surface neighbours and per-iso/lat metadata. Skip them when reconstructing Atoms — see `persistence.atoms_from_graph` for the canonical filter (only `bulk`/`surface` + occupied `adsorbate`).
- `G.graph["cell"]` (3×3, rows = lattice vectors) and `G.graph["pbc"]` (3-bool, **derived from actual cross-image bonds**, not from `atoms.pbc`). Always read PBC from `G.graph["pbc"]` downstream — `build_graph` warns when user PBC and bond-derived PBC disagree (vacuum-gap bug).
- Edges carry `distance` (MIC) and `offset=(i,j,k)` of the partner image. Use this offset instead of recomputing MIC.

## Tunables: one source of truth

`autokmc/constants.py` is the **only** place magic numbers live (`NL_MULT_DEFAULT`, `CO_FACTOR`, `OPT_FACTOR`, `CONTACT_FACTOR`, `N_SHELLS_DEFAULT`, `BOND_TOLERANCE`, `LATERAL_SHELLS_DEFAULT`, `RANDOM_SEED=69`, schema versions, default filenames, …). Override per-call via explicit kwargs; do **not** hardcode the same value in another module. New tunables go here with a docstring naming every consumer.

## Iso-class / lateral-class identity

KMC reactions are deduplicated by `(iso_class, lateral_class)`:
- `iso_class` = graph-isomorphism equivalence of an `AdsorbateSite` (depth `N_SHELLS_DEFAULT=1` ego graph distinguishes fcc vs hcp hollows).
- `lateral_class` = fingerprint of lateral neighbours from `check_adsorbate_sites._lateral_fingerprint` at BFS depth `LATERAL_SHELLS_DEFAULT=0` (only co-bonded surface clique).
- `check_site_stability` runs **two** ML relaxations (occupied + unoccupied) per new lateral class and caches energies on the `AdsorbateSiteLateral`. Persist relaxed Atoms exactly **once** per `(iso, lat)` under `runs/<name>/reactions/iso{N}_lat{M}/{occupied,unoccupied}.extxyz` — subsequent firings only append to `events.jsonl`.

## Calculator plumbing

`config.CalculatorCfg` supports either `import_path`+`kwargs` (constructors: EMT, VASP, CP2K) or `factory`+`factory_kwargs` (classmethods: `nequip.ase.NequIPCalculator.from_compiled_model`, `mace.calculators.MACECalculator.from_model`). `factory` wins. `config._resolve` walks dotted paths from longest to shortest module prefix; use `module:attr` to force the split. `structure.optimise_structure` deep-copies the calculator on every call to preserve loaded ML weights.

## CLI / persistence

`autokmc run CONFIG.{yaml,toml}` → `cli.run_from_config`. Outputs (`constants.REACTIONS_FILENAME`, etc.):
```
runs/<name>/{events.jsonl, summary.json, kmc.extxyz, reactions/iso{N}_lat{M}/...}
```
Bump `PERSISTENCE_SCHEMA_VERSION` / `CONFIG_SCHEMA_VERSION` on any breaking schema change. `_safe_atoms_copy` strips stale calculator results before writing extxyz (NequIP/MACE caches frequently mismatch `len(atoms)`).

## Conventions

- `from __future__ import annotations` at top of every module.
- Loggers: `from autokmc.logging_utils import get_logger; _log = get_logger(__name__)` — never `logging.getLogger` directly. Library never installs a real handler (NullHandler only); the CLI calls `logging.basicConfig`.
- Public functions take keyword-only knobs (`*,`) defaulting to `constants.*`. Mutate `atoms.arrays` / `G.nodes` in place, return the structured result (`SurfaceClassification`, `AnchorSite`, `AdsorbateSite`, `Reaction`).
- Tests are calculator-free: `tests/conftest.py` builds `_StubReaction` / `_StubSite` / synthetic `nx.Graph`. Mirror that pattern — do **not** import NequIP from tests.

## Dev workflow

```bash
pip install -e ".[ml,cli,test,dev]"      # ml = nequip+torch (optional)
pytest -q                                 # tests/ (no CI yet)
autokmc validate-config example/co_cu111_allegro.yaml
autokmc run example/co_cu111_allegro.yaml
```

Notebooks/scratch scripts live in `autokmc/dev/` and are excluded from the
package build (`pyproject.toml` `[tool.setuptools.packages.find]`).

## Known sharp edges (see `TODO.md`)

- `find_adsorbate_sites`: rigid-molecule assumption; weakly-adsorbing molecules (CH₄) form no surface bonds and currently can't be activated.
- `find_anchors._optimise_position` slab branch assumes surface normal ‖ +z (true for `structure.build_surface` outputs only).
- KD-tree annulus prefilter in `find_adsorbate_sites._recurse` was reverted (MIC bug); current code is O(N) backtracking — preserve correctness over speed.
- Oxide adsorbates (reactive lattice O) not yet supported in `structure.py`.

