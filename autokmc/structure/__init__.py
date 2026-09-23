"""Structure builders and optimizers.

The implementation is split into focused modules:

- :mod:`autokmc.structure.builders` for shared composition/lattice helpers
- :mod:`autokmc.structure.optimization` for relaxation helpers
- :mod:`autokmc.structure.nanoparticle` for Wulff nanoparticle construction
- :mod:`autokmc.structure.slab` for surface slab construction
- :mod:`autokmc.structure.surface` for slab/nanoparticle surface classification
"""

from __future__ import annotations

from autokmc.structure.types import Composition, LatticeParams
from autokmc.structure.builders import (
    _apply_composition,
    _ase_reference_lp,
    _build_primitive_cell,
    _build_surface_parent_cell,
    _extract_lp,
    _fmt_lp,
    _normalise_lp,
    _parse_composition,
    _primary_element,
    _print_divider,
    _print_header,
    _validate_crystal_structure,
)
from autokmc.structure.optimization import (
    StructureOptimisationError,
    _resolve_lattice_params,
    optimise_bulk,
    optimise_structure,
)
from autokmc.structure.nanoparticle import (
    build_nanoparticle,
    calculate_surface_energies,
    normalise_surface_energies,
)
from autokmc.structure.loading import (
    StructureInputError,
    load_structure_file,
    resolve_frozen_indices,
    resolve_structure_path,
)
from autokmc.structure.slab import (
    _align_slab_normal,
    _get_bottom_layer_indices,
    _orthogonalise_slab,
    build_surface,
)
from autokmc.structure.surface import (
    align_periodic_slab_frame,
    find_surface_atoms,
    find_surface_atoms_convexhull,
    find_surface_atoms_raycasting,
    has_pbc_connectivity,
    tag_surface_atoms,
)

__all__ = [
    "Composition",
    "LatticeParams",
    "build_nanoparticle",
    "calculate_surface_energies",
    "normalise_surface_energies",
    "build_surface",
    "StructureInputError",
    "StructureOptimisationError",
    "load_structure_file",
    "resolve_frozen_indices",
    "resolve_structure_path",
    "optimise_bulk",
    "optimise_structure",
    "align_periodic_slab_frame",
    "find_surface_atoms",
    "find_surface_atoms_convexhull",
    "find_surface_atoms_raycasting",
    "has_pbc_connectivity",
    "tag_surface_atoms",
    "_resolve_lattice_params",
    "_parse_composition",
    "_primary_element",
    "_apply_composition",
    "_validate_crystal_structure",
    "_ase_reference_lp",
    "_normalise_lp",
    "_build_primitive_cell",
    "_build_surface_parent_cell",
    "_extract_lp",
    "_fmt_lp",
    "_print_header",
    "_print_divider",
    "_align_slab_normal",
    "_orthogonalise_slab",
    "_get_bottom_layer_indices",
]
