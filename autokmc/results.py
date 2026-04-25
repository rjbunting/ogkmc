"""
autokmc.results
===============
Public dataclasses returned from the analysis stages.

These replace the previous variable-length tuple returns (e.g. the
``surface_mask, surface_indices, [hull, [diagnostics,]] method`` tuple
from :func:`autokmc.surface.find_surface_atoms`) so call-sites do not
have to count tuple positions.

The dataclasses are *iterable* — i.e. ``mask, indices, method = result``
still works — so existing call-sites do not need to change.  New code
should prefer attribute access (``result.method``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np


@dataclass
class SurfaceClassification:
    """Result of :func:`autokmc.surface.find_surface_atoms`.

    Attributes
    ----------
    mask : np.ndarray[bool], shape (N,)
        ``True`` for atoms classified as surface.
    indices : np.ndarray[int]
        ``np.where(mask)[0]`` — the surface atom indices.
    method : str
        ``"raycasting"`` (periodic slab) or ``"convexhull"`` (nanoparticle).
    hull : Any | None
        :class:`scipy.spatial.ConvexHull` for the nanoparticle path,
        otherwise ``None``.
    diagnostics : dict | None
        Per-atom signed distances etc. for the nanoparticle path when
        ``return_diagnostics=True`` was passed; otherwise ``None``.
    """
    mask        : np.ndarray
    indices     : np.ndarray
    method      : str
    hull        : Optional[Any]   = None
    diagnostics : Optional[dict]  = None

    # Backwards-compatible iteration: lets callers keep doing
    #   mask, indices, method = find_surface_atoms(...)
    #   mask, indices, hull, method = find_surface_atoms(...)            (NP, no diag)
    #   mask, indices, hull, diag, method = find_surface_atoms(...)      (NP, with diag)
    # by emitting the same tuple shape the legacy implementation did.
    def __iter__(self) -> Iterator[Any]:
        if self.method == "raycasting":
            yield self.mask
            yield self.indices
            yield self.method
            return
        # convexhull
        yield self.mask
        yield self.indices
        if self.hull is not None:
            yield self.hull
        if self.diagnostics is not None:
            yield self.diagnostics
        yield self.method

