"""Lightweight result containers shared across the package.

Currently:

* :class:`SurfaceClassification` — return type of
  :func:`ogkmc.structure.find_surface_atoms`.  Behaves like a regular
  dataclass (``.mask``, ``.indices``, ``.method``, ``.hull``,
  ``.diagnostics``) and is iterable in a tuple-like order for convenient
  unpacking::

      mask, indices, method = find_surface_atoms(slab)         # raycasting
      mask, indices, hull, method = find_surface_atoms(np_)    # convexhull
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class SurfaceClassification:
    """Surface-atom classification result.

    Attributes
    ----------
    mask : np.ndarray, shape (N,), dtype bool
        ``True`` where the atom is on the surface.
    indices : np.ndarray, shape (k,), dtype int
        Indices of surface atoms (``np.where(mask)[0]``).
    method : str
        Algorithm used: ``"raycasting"`` (slabs) or ``"convexhull"``
        (nanoparticles).
    hull : Any, optional
        :class:`scipy.spatial.ConvexHull` instance — only set on the
        nanoparticle path.  ``None`` for slabs.
    diagnostics : dict, optional
        Per-atom intermediates from the convex-hull path when
        ``return_diagnostics=True`` was requested.  ``None`` otherwise.

    Iteration order
    ---------------
    Raycasting (no hull):       ``(mask, indices, method)``
    Convex-hull (with hull):    ``(mask, indices, hull, method)``

    This keeps tuple unpacking convenient for callers that do not need the
    dataclass attributes directly.
    """

    mask: np.ndarray
    indices: np.ndarray
    method: str
    hull: Any = None
    diagnostics: Any = None

    # ------------------------------------------------------------------
    # Iteration / unpacking compatibility
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        if self.hull is not None:
            yield self.mask
            yield self.indices
            yield self.hull
            yield self.method
        else:
            yield self.mask
            yield self.indices
            yield self.method

    def __len__(self) -> int:
        return 4 if self.hull is not None else 3

