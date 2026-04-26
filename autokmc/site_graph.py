"""
autokmc.site_graph
==================
**Deprecated.**  The legacy "anchor + co-face" multi-atom enumerator
that lived in this module has been superseded by
:mod:`autokmc.find_multisite` (the orbit-canonical, surface-connectivity
guarded enumerator), which is in turn re-exported through the unified
:mod:`autokmc.sites` façade.

Nothing in the package depends on this module any more.  It is kept as
a thin deprecation stub for one release so that pinned ``from
autokmc.site_graph import …`` lines fail with a clear pointer rather
than ``ModuleNotFoundError``.

For new code, import from :mod:`autokmc.sites`::

    from autokmc.sites import find_adsorbate_sites, AdsorbateSite
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "autokmc.site_graph is deprecated; import from autokmc.sites instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the unified site API so the few external scripts that did
# ``from autokmc.site_graph import find_multisites`` still resolve.
from autokmc.sites import *  # noqa: E402, F401, F403
