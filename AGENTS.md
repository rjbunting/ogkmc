# AGENTS.md

Guidance for AI coding agents working on `autokmc`, a Python package for automated kinetic Monte Carlo setup on metal surfaces/nanoparticles.

## Big picture

- Core data flow is ASE `Atoms` → surface tagging → NetworkX graph → site enumeration → adsorbate placement/refinement.
- Build structures in `autokmc/structure.py`: `build_nanoparticle` uses WulffPack, `build_surface` uses pymatgen slabs, then ASE LBFGS relaxation; slabs store `atoms.info["frozen_indices"]` for later adsorbate relaxations.
- Classify surfaces in `autokmc/surface.py` before graphing: call `find_surface_atoms(atoms, tag_atoms=True)` so `atoms.arrays["surface"]` exists (`0=bulk`, `1=surface`, `2=adsorbate`).
- Build graphs with `autokmc/graph.py::build_graph`; nodes carry `element`, `position`, `index`, `type`, `covalent_radius`; edges carry MIC `distance` and `offset`; `G.graph["pbc"]` is inferred from cross-image bonds, not just `Atoms.pbc`.
- Site discovery is exposed via `autokmc/sites.py`; it re-exports single-atom logic from `default_sites.py` and current adsorbate-site logic from `find_multisite.py`.
- Reactants come from `autokmc/reactants.py::build_reactant`, which parses SMILES with RDKit, embeds/optimizes 3-D geometry, tags atoms as adsorbates, builds a molecular graph, and records symmetry/anchor metadata.
- ML/ASE calculator refinement lives in `autokmc/opt_site.py`; it mutates each adsorbate-site object with dynamic fields such as `stable`, `adsorption_energy`, `member_positions`, and `member_subgraphs`.

## State and naming conventions

- Always use `get_cache(G)` from `autokmc/cache.py` instead of writing arbitrary `G.graph[...]` state; it installs `G.graph["autokmc"]` and graph aliases (`sites`, `unique_sites`, `site_positions`, `adsorbate_sites`).
- Use “adsorbate-site” names (`AdsorbateSite`, `find_adsorbate_sites`, `optimise_adsorbate_site_positions`, `cache.adsorbate_sites`); the old `MultiSite`/`multisites` aliases have been removed from the package API.
- Tune package-wide numeric defaults in `autokmc/constants.py` (`NL_MULT_DEFAULT`, `CO_FACTOR`, `N_SHELLS_DEFAULT`, `BOND_TOLERANCE`, etc.) rather than scattering magic numbers.
- Public result objects such as `SurfaceClassification` in `autokmc/results.py` are dataclasses but remain iterable for old tuple-unpacking call sites; preserve that compatibility when changing APIs.
- Use `get_logger(__name__)` and `verbose_scope` from `autokmc/logging_utils.py` for new logging; some older structure code still prints under `verbose=True`.

## Developer workflows

- Install locally with optional backends as needed: `python -m pip install -e '.[dev]'`, or add extras like `'.[all,dev]'` for WulffPack/pymatgen/RDKit.
- Run tests with `python -m pytest`; `pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "-ra"`.
- Lint/type-check with `python -m ruff check .` and `python -m mypy autokmc`; Ruff targets Python 3.10 and ignores long lines plus domain-style short names (`E741`).
- There are no existing `README.md`, Copilot, Cursor, Claude, Windsurf, or Cline instruction files in this repo.

## Integration points and caveats

- Required runtime dependencies are `numpy`, `scipy`, `networkx`, and `ase`; optional features require `wulffpack` (`build_nanoparticle`), `pymatgen` (`build_surface`), and `rdkit` (`build_reactant`).
- Keep graph/site algorithms MIC-aware: several helpers use `G.graph["cell"]` and inferred `G.graph["pbc"]`, and nanoparticle hull equations may be cached via `atoms.info["_hull_equations"]`.
- `autokmc/dev/` contains notebooks/scripts and model files for exploration; package discovery excludes `autokmc.dev*`.
- Current `tests/test_reactions.py` is ahead of the source tree and skips at module level until the pending `autokmc.reactions` API and `TS_OFFSET_EV` constant exist. Treat this as a repo caveat before assuming reaction coverage is active.
