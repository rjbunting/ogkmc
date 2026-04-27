# autokmc

A Python library for setting up surface kinetic Monte Carlo (KMC) inputs
from atomic structures.

`autokmc` builds bulk metals, nanoparticles and orthogonalised surface
slabs (FCC / BCC / HCP, pure or alloy), classifies surface atoms,
constructs an atom-connectivity graph, parses SMILES into 3-D reactants,
and enumerates the geometrically feasible adsorbate sites for one- and
multi-atom adsorbates.  All steps consume / produce ASE `Atoms` and
`networkx.Graph` objects so the pipeline plugs into any ASE-compatible
calculator (EMT, NequIP, MACE, …).

There is no app or CLI: the package is consumed from notebooks / scripts
in `autokmc/dev/`.

## Install

```bash
pip install -e .
# optional ML calculator support (NequIP / Torch):
pip install -e ".[ml]"
# optional dev tools (matplotlib / jupyter):
pip install -e ".[dev]"
```

## Pipeline at a glance

```python
from ase.calculators.emt import EMT
from autokmc import (
    build_surface,
    find_surface_atoms,
    build_graph,
    build_reactant,
    find_anchor_sites,
    find_adsorbate_sites,
    optimise_adsorbate_site_positions,
)

# 1. Geometry
slab = build_surface(
    composition="Cu", crystal_structure="fcc", miller_index=(1, 1, 1),
    calculator=EMT(),
)

# 2. Surface classification → atoms.arrays["surface"]
find_surface_atoms(slab, tag_atoms=True)

# 3. Connectivity graph
G = build_graph(slab)

# 4. Reactant (SMILES → optimised gas-phase geometry + graph)
co = build_reactant("[C-]#[O+]", calculator=EMT())

# 5. Single-element anchor sites (top, bridge, hollow, ...)
find_anchor_sites(G, "C")
find_anchor_sites(G, "O")

# 6. Multi-atom adsorbate placements
sites = find_adsorbate_sites(G, co)

# 7. Rigid-body refinement of every iso-class representative
optimise_adsorbate_site_positions(G, co.smiles, co)
```

The full pipeline, design rationale and load-bearing invariants are
documented in **AGENTS.md** — read that for anything beyond the quick
start, especially when contributing new pipeline stages.

## Module layout

| Module | Role |
|--------|------|
| `structure.py` | Bulk / nanoparticle / slab builders + relaxation. |
| `surface.py` | Surface-atom classification (ray-casting / convex hull). |
| `graph.py` | ASE `Atoms` → `networkx.Graph` connectivity graph. |
| `reactants.py` | SMILES → 3-D `Reactant` with anchor-atom metadata. |
| `find_anchors.py` | Per-element top / bridge / hollow / … site enumeration. |
| `find_adsorbate_sites.py` | Multi-atom adsorbate placement and rigid-body refinement. |
| `constants.py` | Single source of truth for every tunable default. |
| `results.py` | Lightweight result dataclasses. |
| `logging_utils.py` | Package-wide logger helper. |

## Documentation

- **AGENTS.md** — pipeline ordering, conventions (MIC handling, invisible
  graph nodes, cache invalidation), open issues.
- **REVIEW.md** — most recent code review.
- **autokmc/todo.MD** — open work items.

## Status

Active research code.  No tests, no CI yet.  See `todo.MD` and `AGENTS.md`
for known limitations (oxide adsorbates, weakly adsorbing molecules,
rigid-molecule assumption).

## License

MIT — see `pyproject.toml`.

