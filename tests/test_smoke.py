"""Smoke tests for the autokmc pipeline.

These are deliberately tiny — they guard against the most obvious
regressions (imports break, build_graph misses edge attributes, the
SiteCache aliasing breaks the legacy keys, the new SurfaceClassification
is iterable in legacy tuple shape) without depending on heavy optional
backends like wulffpack / pymatgen / rdkit.

Run with ``pytest`` from the repo root.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_cu111_slab():
    """4×4×3 Cu(111) slab via ASE — no pymatgen/wulffpack required."""
    pytest.importorskip("ase")
    from ase.build import fcc111
    slab = fcc111("Cu", size=(4, 4, 3), vacuum=10.0, periodic=True)
    return slab


def test_constants_importable():
    from autokmc import constants
    assert constants.NL_MULT_DEFAULT == 1.0
    assert constants.CO_FACTOR == pytest.approx(0.95)
    assert constants.OPT_FACTOR == pytest.approx(0.85)


def test_logger_root_exists():
    from autokmc.logging_utils import get_logger
    log = get_logger(__name__)
    assert log.name.startswith("autokmc.")


def test_build_graph_has_edge_distance_and_offset():
    from autokmc.graph import build_graph
    from autokmc.surface import find_surface_atoms

    slab = _make_cu111_slab()
    find_surface_atoms(slab, tag_atoms=True)
    G = build_graph(slab)

    # Edge attrs (point 7.4): every edge carries distance + offset.
    for _u, _v, data in G.edges(data=True):
        assert "distance" in data
        assert "offset"   in data
        assert isinstance(data["offset"], tuple)
        assert len(data["offset"]) == 3
        assert data["distance"] > 0.0
        break

    # PBC is now an ndarray (point 4.2).
    assert isinstance(G.graph["pbc"], np.ndarray)
    assert G.graph["pbc"].dtype == bool
    assert G.graph["pbc"].any()  # slab is periodic in xy


def test_cache_aliases_legacy_keys():
    from autokmc.cache import SiteCache, get_cache
    from autokmc.graph import build_graph
    from autokmc.surface import find_surface_atoms

    slab = _make_cu111_slab()
    find_surface_atoms(slab, tag_atoms=True)
    G = build_graph(slab)

    cache = get_cache(G)
    assert isinstance(cache, SiteCache)
    # The legacy keys are the *same object* as the typed cache attrs.
    assert G.graph["sites"] is cache.sites
    assert G.graph["unique_sites"] is cache.unique_sites
    assert G.graph["site_positions"] is cache.site_positions


def test_surface_classification_iterable_legacy_shape():
    """The new SurfaceClassification dataclass must still unpack to the
    legacy variable-length tuple so notebook code keeps working."""
    from autokmc.surface import find_surface_atoms

    slab = _make_cu111_slab()
    result = find_surface_atoms(slab, tag_atoms=True)

    # Legacy raycasting tuple shape: (mask, indices, method).
    mask, indices, method = result
    assert method == "raycasting"
    assert mask.dtype == bool
    assert indices.dtype.kind in "iu"
    assert mask.sum() > 0


def test_find_sites_for_element_caches_via_typed_cache():
    pytest.importorskip("scipy")
    from autokmc.default_sites import find_sites_for_element
    from autokmc.cache import get_cache
    from autokmc.graph import build_graph
    from autokmc.surface import find_surface_atoms

    slab = _make_cu111_slab()
    find_surface_atoms(slab, tag_atoms=True)
    G = build_graph(slab)

    sites = find_sites_for_element(G, "O")
    cache = get_cache(G)
    assert cache.sites["O"] is sites
    assert cache.k_max["O"] >= 1
    # Every Cu(111) slab admits top, bridge, hollow → at least k = 1, 2, 3.
    assert set(sites.keys()) >= {1, 2, 3}

