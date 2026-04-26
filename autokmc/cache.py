"""
autokmc.cache
=============
Typed wrapper around the formerly-loose ``G.graph[...]`` dict that the
site / multisite pipeline used to scribble its caches into.

Goals
-----
* Single attribute (``G.graph["autokmc"]``) carrying every cached stage.
* Typed access (auto-completion in editors; mypy-checked).
* Cheap to introspect (``cache.summary()``) and reset (``cache.invalidate("O")``).
* Backwards compatible — the old top-level keys
  (``G.graph["sites"]``, ``["unique_sites"]``, ``["site_positions"]``,
  ``["multisites"]``, ``["k_max"]``) are still kept in sync via property
  setters so legacy code that pokes them directly continues to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import networkx as nx


# ---------------------------------------------------------------------------
# Cache container
# ---------------------------------------------------------------------------

@dataclass
class SiteCache:
    """Per-graph cache of every stage in the site-discovery pipeline.

    Lives at ``G.graph["autokmc"]``; obtain via :func:`get_cache`.

    The legacy top-level keys (``sites``, ``unique_sites``, ...) inside
    ``G.graph`` are kept in sync automatically (see :func:`get_cache`),
    so existing code that reads ``G.graph["sites"]["O"]`` still works.
    """
    # element -> int
    k_max          : Dict[str, int]                                            = field(default_factory=dict)
    # element -> {k: [frozenset[node_id]]}
    sites          : Dict[str, Dict[int, List[Any]]]                           = field(default_factory=dict)
    # element -> {n_shells: {k: [IsoClass]}}
    unique_sites   : Dict[str, Dict[int, Dict[int, List[Any]]]]                = field(default_factory=dict)
    # element -> {k: [np.ndarray]}
    site_positions : Dict[str, Dict[int, List[Any]]]                           = field(default_factory=dict)
    # smiles -> [AdsorbateSite]
    adsorbate_sites: Dict[str, List[Any]]                                      = field(default_factory=dict)
    # element -> {k: [anchor_node_id]}; one entry per raw site clique, in the
    # *same order* as ``sites[element][k]``.  Anchor nodes are materialised on
    # the graph (type="anchor") by ``find_sites_for_element`` so that every
    # site is addressable by a stable graph node id.
    anchor_nodes   : Dict[str, Dict[int, List[int]]]                           = field(default_factory=dict)
    # cached convex hull (nanoparticles only) and surface APSP
    hull           : Any                                                       = None
    surface_apsp   : Any                                                       = None  # nx-style dict-of-dicts

    # Backward-compatible alias for the pre-rename ``multisites`` field.
    # Reads/writes proxy to :attr:`adsorbate_sites` so callers using either
    # name see the same dict object — important because legacy code stamps
    # entries onto ``cache.multisites[smiles]`` directly.
    @property
    def multisites(self) -> Dict[str, List[Any]]:
        return self.adsorbate_sites

    @multisites.setter
    def multisites(self, value: Dict[str, List[Any]]) -> None:
        self.adsorbate_sites = value

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def invalidate(self, element: str | None = None,
                   smiles: str | None = None) -> None:
        """Drop cached results for one element / SMILES (or everything)."""
        if element is None and smiles is None:
            self.k_max.clear()
            self.sites.clear()
            self.unique_sites.clear()
            self.site_positions.clear()
            self.adsorbate_sites.clear()
            self.anchor_nodes.clear()
            self.hull = None
            self.surface_apsp = None
            return
        if element is not None:
            self.k_max.pop(element, None)
            self.sites.pop(element, None)
            self.unique_sites.pop(element, None)
            self.site_positions.pop(element, None)
            self.anchor_nodes.pop(element, None)
        if smiles is not None:
            self.adsorbate_sites.pop(smiles, None)

    def summary(self) -> str:
        """Short human-readable description of what has been cached."""
        lines = ["SiteCache:"]
        if self.sites:
            for el, by_k in self.sites.items():
                total = sum(len(v) for v in by_k.values())
                lines.append(f"  sites[{el!r}]          : {total} cliques "
                             f"(k_max={self.k_max.get(el, '?')})")
        if self.unique_sites:
            for el, by_n in self.unique_sites.items():
                lines.append(f"  unique_sites[{el!r}]   : depths={sorted(by_n)}")
        if self.site_positions:
            for el in self.site_positions:
                lines.append(f"  site_positions[{el!r}] : present")
        if self.adsorbate_sites:
            for sm, ms in self.adsorbate_sites.items():
                lines.append(f"  adsorbate_sites[{sm!r}] : {len(ms)} iso-classes")
        if self.hull is not None:
            lines.append("  hull                  : cached")
        if self.surface_apsp is not None:
            lines.append("  surface_apsp          : cached")
        if len(lines) == 1:
            lines.append("  (empty)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accessor that also keeps the legacy top-level keys in sync
# ---------------------------------------------------------------------------

_LEGACY_KEYS = ("k_max", "sites", "unique_sites", "site_positions",
                "adsorbate_sites", "multisites")


def get_cache(G: nx.Graph) -> SiteCache:
    """Return the :class:`SiteCache` attached to *G*, creating it if absent.

    Also installs a one-time alias so that legacy code reading
    ``G.graph["sites"]`` (etc.) sees the same dict object backing the
    typed attribute.  This lets existing notebooks continue to work
    unchanged while new code uses the typed interface.
    """
    cache = G.graph.get("autokmc")
    if isinstance(cache, SiteCache):
        return cache

    cache = SiteCache()
    G.graph["autokmc"] = cache
    # Re-bind any pre-existing legacy keys into the typed cache so that
    # code that mixed-and-matched access patterns sees a consistent state.
    for key in _LEGACY_KEYS:
        if key in G.graph:
            existing = G.graph[key]
            if isinstance(existing, dict):
                getattr(cache, key).update(existing)
    # Now alias: the legacy keys *are* the typed dicts (same object).
    for key in _LEGACY_KEYS:
        G.graph[key] = getattr(cache, key)
    return cache

