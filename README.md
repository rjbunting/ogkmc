# autokmc2

A Python library for setting up surface kinetic Monte Carlo (KMC) inputs
from atomic structures.

`autokmc2` builds bulk metals, nanoparticles and orthogonalised surface
slabs (FCC / BCC / HCP, pure or alloy), classifies surface atoms,
constructs an atom-connectivity graph, parses SMILES into 3-D reactants,
and enumerates the geometrically feasible adsorbate sites for one- and
multi-atom adsorbates.  All steps consume / produce ASE `Atoms` and
`networkx.Graph` objects so the pipeline plugs into any ASE-compatible
calculator (EMT, NequIP, MACE, …).

The package is consumed through explicit domain imports and includes an
`autokmc2` CLI entry point for config-driven runs.

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
from autokmc2.structure import build_surface
from autokmc2.structure import find_surface_atoms
from autokmc2.core.graph import build_graph
from autokmc2.species.reactant import build_reactant
from autokmc2.sites.anchors import find_anchor_sites
from autokmc2.sites.adsorbate import (
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
documented in `structure.MD` and inline module docstrings.

## Module layout

| Module | Role |
|--------|------|
| `autokmc2.structure.*` | Structure builders, slab/nanoparticle construction, optimization, and surface classification. |
| `autokmc2.core.*` | Atom graph construction, graph-state accessors, constants, and shared result models. |
| `autokmc2.species.*` | SMILES/reactant handling and molecular bond-changing chemistry. |
| `autokmc2.sites.*` | Anchor, adsorbate, diffusion, and bond-site enumeration plus stability checks. |
| `autokmc2.reactions.*` | Adsorption, diffusion, and bond reaction models/rate construction. |
| `autokmc2.kmc.*` | KMC engine and on-the-fly site expansion. |
| `autokmc2.io.*` | Config loading, persistence, trajectories and summaries. |
| `autokmc2.thermo.*` | Vibrational free-energy helpers. |
| `autokmc2.cli.*` | CLI entry point and config-driven pipeline. |

## Documentation

- **structure.MD** — restructuring rationale, target boundaries, and migration notes.
- **TODO.md** — open work items.

## Status

Active research code.  A lightweight pytest suite covers config, CLI parsing,
persistence, summaries, and trajectories.  See `TODO.md` for known limitations
(oxide adsorbates, weakly adsorbing molecules, rigid-molecule assumption).

## License

MIT — see `pyproject.toml`.
