# autokmc — Agent Guide

`autokmc` is an automated kinetic-Monte-Carlo (KMC) setup pipeline for metal
surfaces and nanoparticles. Public API is re-exported from
`autokmc/__init__.py`; everything else under `autokmc/dev/` is throw-away
notebooks/scripts and is **excluded from the package** (see
`pyproject.toml` `[tool.setuptools.packages.find]`).

## Module layout

* **`autokmc/sites.py`** — single canonical entry point for *all* site
  discovery / classification / refinement. Imports from this module first.
  It is a façade over two implementation modules — `default_sites.py`
  (single-atom anchors, k-clique iso-classes) and `find_multisite.py`
  (multi-atom adsorbate placements) — both of which remain importable
  but should be treated as private.
* **`autokmc/site_graph.py`** — *deprecated stub*. The legacy
  "anchor + co-face" enumerator that used to live here was superseded
  by `find_multisite`; the file now just re-exports from `sites.py`
  with a `DeprecationWarning`.
* **`autokmc/opt_site.py`** — ML / calculator-driven refinement and
  propagation across iso-class members.

## Naming convention (April 2026 rename)

The public API was renamed from "multisite" to "adsorbate_site". Both
spellings still work, but new code must use the **new** names; the old
ones are kept as aliases (see `sites.py` and the legacy property on
`SiteCache`).

| Legacy name                                       | Preferred name                                      |
| ------------------------------------------------- | --------------------------------------------------- |
| `MultiSite`                                       | `AdsorbateSite`                                     |
| `find_multisites`                                 | `find_adsorbate_sites`                              |
| `find_multisites_for_reactant`                    | `find_adsorbate_sites_for_reactant`                 |
| `optimise_multisite_positions`                    | `optimise_adsorbate_site_positions`                 |
| `optimise_multisites_ml` (in `opt_site`)          | `optimise_adsorbate_sites_ml`                       |
| `seed_single_atom_multisites` (in `opt_site`)     | `seed_single_atom_adsorbate_sites`                  |
| `cache.multisites` / `G.graph["multisites"]`      | `cache.adsorbate_sites` / `G.graph["adsorbate_sites"]` |
| `N_MULTISITE_RESTARTS` (in `constants`)           | `N_ADSORBATE_RESTARTS`                              |

The `multisites` cache field is now a Python property aliasing
`adsorbate_sites`, and `_LEGACY_KEYS` in `cache.get_cache` aliases both
graph-level keys to the *same dict object*, so existing code that pokes
`G.graph["multisites"][smiles]` continues to mutate the typed cache.

## Pipeline (data flow)

The package is a linear pipeline; each stage reads/writes a single ASE
`Atoms` and a single `networkx.Graph`:

1. **Build geometry** — `structure.build_nanoparticle` (WulffPack) or
   `structure.build_surface` (pymatgen `SlabGenerator` + orthogonalisation
   + tiling). `build_surface` stores `atoms.info["frozen_indices"]` so the
   ML refinement stage can re-apply `FixAtoms` without re-detecting layers.
2. **Tag atom roles** — `surface.find_surface_atoms(atoms, tag_atoms=True)`
   auto-dispatches to `find_surface_atoms_raycasting` (slabs) or
   `find_surface_atoms_convexhull` (NPs) using `has_pbc_connectivity`
   (checks for cross-image bonds, **not** `atoms.pbc`). Writes int8
   `atoms.arrays["surface"]` with encoding `0=bulk, 1=surface, 2=adsorbate`
   (preserved in `.extxyz`). NP path also stashes
   `atoms.info["_hull_equations"]`.
3. **Build connectivity graph** — `graph.build_graph(atoms)` returns an
   `nx.Graph` whose nodes carry `element/position/index/type/covalent_radius`
   and edges carry `distance/offset` (MIC + cell-image tuple). `G.graph["pbc"]`
   is **derived from observed cross-image bonds**, not `atoms.pbc`.
4. **Site discovery (single-atom)** — `default_sites.find_sites_for_element`
   enumerates k-cliques (k=1…k_max) of an adsorbate-specific co-bonding
   graph, then `reduce_sites_by_isomorphism` collapses equivalents via
   ego-graph isomorphism, then `optimise_site_positions` does a
   calculator-free geometric optimisation. **Each accepted site is
   materialised on the graph as an *anchor node*** (`type="anchor"`,
   `element=<adsorbate>`, `covalent_radius=r_cov_ads`, plus the bonded
   `clique` frozenset and `k`) connected to every surface atom in its
   clique with an `anchor_bond=True` edge. `reduce_sites_by_isomorphism`
   stamps `iso_class`, `n_shells`, and `ego_subgraph` onto each anchor
   node and writes `IsoClass.member_node_ids` (the anchor ids belonging
   to that class). `optimise_site_positions` moves the anchor's
   `position`, refreshes its `anchor_bond` edge distances, and sets
   `optimised=True`. Anchor ids are also indexed in
   `cache.anchor_nodes[element][k]` (parallel to `cache.sites[element][k]`).
5. **Site discovery (multi-atom)** — `sites.find_adsorbate_sites`
   (still importable as `find_multisites`) places N-atom molecules
   using `reactants.build_reactant` (RDKit ETKDGv3 + MMFF94 → ASE →
   graph + automorphism orbits + convex-hull anchor mask).
   **Every member of every iso-class is materialised on the graph as a
   connected adsorbate-anchor subgraph**: N anchor nodes per placement
   (`type="anchor"`, `smiles=<reactant>`, `iso_class=<class>`,
   `reactant_index=<i>`, `is_bonded`, `clique`/`k`, `siblings` tuple of
   the other N−1 node ids) wired together with `intra_adsorbate=True`
   edges (mirroring `reactant.graph`); each *bonded* atom is also
   connected to its surface clique with `anchor_bond=True` edges.
   `AdsorbateSite.member_node_ids: list[list[int]]` and
   `AdsorbateSite.member_positions: list[np.ndarray]` are parallel to
   `AdsorbateSite.members`. Re-running `find_adsorbate_sites` for the
   same SMILES wipes prior adsorbate anchors via
   `_remove_adsorbate_anchor_nodes`. Refinement helpers
   (`optimise_adsorbate_site_positions` and any downstream Kabsch
   propagator) must call `push_member_positions_to_graph(G, ms,
   member_index)` after moving a member's Cartesians so the graph node
   positions and edge distances stay in lock-step with the cached
   `AdsorbateSite`.
6. **ML refinement** — `opt_site.optimise_adsorbate_sites_ml` (legacy
   alias `optimise_multisites_ml`) relaxes each iso-class with the
   user-supplied ASE calculator, then propagates the relaxed geometry
   to all members via Kabsch on n_shells=1 ego subgraphs.

## Cross-cutting conventions

- **All cached state lives on the graph**, in a single typed
  `cache.SiteCache` at `G.graph["autokmc"]`. Always access via
  `from autokmc.cache import get_cache; cache = get_cache(G)`. Legacy
  top-level keys (`G.graph["sites"]`, `["unique_sites"]`, ...) are the
  *same dict objects* as the typed attributes — keep them in sync if you
  add fields. Use `cache.invalidate(element=..., smiles=...)` rather than
  popping by hand.
- **All tunable magic numbers live in `autokmc/constants.py`**
  (`NL_MULT_DEFAULT`, `CO_FACTOR`, `OPT_FACTOR`, `STANDOFF_FACTOR`,
  `N_SHELLS_DEFAULT`, `BOND_TOLERANCE`, `HULL_TOL`,
  `N_ADSORBATE_RESTARTS`, …). Do not introduce new module-local
  constants; add a documented entry there and reference it via keyword
  default.
- **Logging, not `print`** — every module does `_log = get_logger(__name__)`
  from `autokmc.logging_utils`. Public functions accept `verbose: bool`
  for back-compat; implement it with `with verbose_scope(_log, verbose):`.
  Use `header(_log, "...")` / `divider(_log)` for the 58-char `===` banners
  (replicates the historical look). The package root attaches a
  `NullHandler`, so silence is the default.
- **Public results are dataclasses that iterate as legacy tuples** — see
  `results.SurfaceClassification`. When adding a new return type follow
  the same pattern (yield in the legacy tuple order from `__iter__`) so
  notebook callers do not break.
- **PBC is inferred from bonding, not from `atoms.pbc`.** Nanoparticles
  are wrapped in a periodic vacuum-padded box (`build_nanoparticle` sets
  `pbc=True`) and downstream code calls `has_pbc_connectivity` /
  inspects `G.graph["pbc"]`.
- **Anchor nodes are an extra `type` on the graph.** Both single-atom
  default sites (one anchor per clique) and multi-atom adsorbate sites
  (N connected anchors per placement, carrying a `smiles` attribute)
  use `type="anchor"`. Any new BFS / shell expansion through
  `G.neighbors(...)` must skip `type=="anchor"` (see
  `default_sites._build_clique_ego` and `find_multisite._clique_to_clique_max_hops`)
  — otherwise anchors act as 1-hop shortcuts between cliques. Use a
  filter such as `if G.nodes[m].get("type") == "anchor": continue`.
  Re-running `find_sites_for_element` / `find_multisites` for the same
  element / SMILES wipes the prior anchors via `_remove_anchor_nodes` /
  `_remove_adsorbate_anchor_nodes`; do **not** rely on
  `cache.invalidate(...)` to clean the graph (it only clears the
  registry).
- **Optional dependencies are guarded.** `wulffpack`, `pymatgen`, and
  `rdkit` are imported behind `try/except` with `_AVAILABLE` flags and
  raise `ImportError` with the exact `pip install …` hint when missing
  (see `structure.py`). Match this pattern for new optional backends.
- **Never mutate caller inputs.** `optimise_structure` returns a
  `atoms.copy()` and deep-copies the calculator (so e.g. a NequIP model
  path survives); follow this convention for any new relaxation helper.

## Developer workflow

- Install: `pip install -e ".[dev,all]"` (extras: `nanoparticle`, `slab`,
  `reactants`, `all`). Python ≥ 3.10.
- Tests: `pytest` (config in `pyproject.toml`, `testpaths = ["tests"]`,
  `addopts = "-ra"`). Tests stub `AdsorbateSite` / `_StubReactant` so the
  suite runs **without an ASE calculator** — mirror that pattern when
  adding new tests (see `tests/test_reactions.py`). Note: that file
  currently imports `autokmc.find_adsorbate_site` / `autokmc.reactions` /
  `TS_OFFSET_EV` which do **not** exist yet; treat as work-in-progress.
- Lint / type: `ruff check .` (line-length 100, ignores E501/E741) and
  `mypy autokmc` (lenient: `ignore_missing_imports`,
  `disallow_untyped_defs=False`).
- Exploratory work belongs in `autokmc/dev/` (notebooks + scripts);
  it is excluded from the wheel.
- Outstanding design notes are in `todo.MD`.

