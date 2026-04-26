"""Removed legacy module.

The old ``autokmc.site_graph`` API no longer re-exports anything.  Use
``autokmc.sites`` for the current adsorption-site API.
"""

from __future__ import annotations

raise ImportError("autokmc.site_graph was removed; import from autokmc.sites instead.")
