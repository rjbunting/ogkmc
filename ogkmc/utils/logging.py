"""Logging helpers for the :mod:`ogkmc` package.

Thin wrapper around :mod:`logging` so the package gets a single, named logger
hierarchy with a sensible default handler.

Use::

    from ogkmc.utils.logging import get_logger
    _log = get_logger(__name__)
    _log.debug("something happened: %r", value)

The root ``ogkmc`` logger is configured once with a NullHandler so the
library never prints unsolicited messages.  Applications that want output
should call :func:`logging.basicConfig` themselves or attach their own
handler to ``logging.getLogger("ogkmc")``.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "ogkmc"

# Configure the root package logger exactly once.
_ROOT = logging.getLogger(_ROOT_NAME)
if not _ROOT.handlers:
    _ROOT.addHandler(logging.NullHandler())
# Don't set a level here — let the application decide.


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``ogkmc`` root.

    Names that don't already start with ``"ogkmc"`` are re-rooted under
    it so that ``logging.getLogger("ogkmc").setLevel(...)`` controls
    every module in the package.
    """
    if not name:
        return logging.getLogger(_ROOT_NAME)
    if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
