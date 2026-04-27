"""
autokmc.logging_utils
=====================
Thin wrapper around :mod:`logging` so the package gets a single, named
logger hierarchy with a sensible default handler.

Use::

    from autokmc.logging_utils import get_logger
    _log = get_logger(__name__)
    _log.debug("something happened: %r", value)

The root ``autokmc`` logger is configured once with a NullHandler so the
library never prints unsolicited messages.  Applications that want output
should call :func:`logging.basicConfig` themselves or attach their own
handler to ``logging.getLogger("autokmc")``.
"""

from __future__ import annotations

import logging

# Configure the root package logger exactly once.
_ROOT = logging.getLogger("autokmc")
if not _ROOT.handlers:
    _ROOT.addHandler(logging.NullHandler())
# Don't set a level here — let the application decide.


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``autokmc`` root.

    Names that don't already start with ``"autokmc"`` are re-rooted under
    it so that ``logging.getLogger("autokmc").setLevel(...)`` controls
    every module in the package.
    """
    if name == "autokmc" or name.startswith("autokmc."):
        return logging.getLogger(name)
    return logging.getLogger(f"autokmc.{name}")

