"""
autokmc.logging_utils
=====================
Stdlib-``logging``-based replacement for the previous ``print(...)`` /
``verbose=True`` pattern.

Use :func:`get_logger` everywhere inside the package so that notebook
users can globally toggle output with one line::

    import logging
    logging.basicConfig(level=logging.INFO)        # show INFO + WARNING
    logging.getLogger("autokmc").setLevel(logging.DEBUG)   # be loud

The ``verbose=`` keyword arguments on public functions are kept for
backwards compatibility — when ``True`` they bump the per-call effective
level to ``DEBUG`` for the duration of the call.

ASCII headers / dividers (the 58-char ``=`` rule used historically in
``structure.py``) live here so every module emits them identically.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

# Single root logger for the whole package; child loggers are obtained
# via ``get_logger(__name__)`` in each module.
_ROOT_NAME = "autokmc"

# A NullHandler is attached so that "no logging configured" silently
# discards messages instead of printing the default warning.
_root = logging.getLogger(_ROOT_NAME)
if not _root.handlers:
    _root.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package root.

    Pass ``__name__`` from inside a module — e.g. ``get_logger(__name__)``
    yields ``autokmc.default_sites`` — so users can filter by module.
    """
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(_ROOT_NAME + "." + name.lstrip("."))


@contextmanager
def verbose_scope(logger: logging.Logger, enabled: bool) -> Iterator[None]:
    """Context manager that temporarily sets *logger* to DEBUG.

    Used to honour legacy ``verbose=True`` keyword arguments without
    permanently changing the logger configuration::

        log = get_logger(__name__)
        with verbose_scope(log, verbose):
            log.debug("...")
    """
    if not enabled:
        yield
        return
    old = logger.level
    logger.setLevel(logging.DEBUG)
    # Also ensure a StreamHandler is attached somewhere up the chain so
    # that DEBUG messages are actually emitted in interactive / notebook
    # sessions where the user has not configured logging at all.
    has_real_handler = False
    n: logging.Logger | None = logger
    while n is not None:
        for h in n.handlers:
            if not isinstance(h, logging.NullHandler):
                has_real_handler = True
                break
        if has_real_handler or not n.propagate:
            break
        n = n.parent
    temp_handler: logging.Handler | None = None
    if not has_real_handler:
        temp_handler = logging.StreamHandler()
        temp_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(temp_handler)
    try:
        yield
    finally:
        logger.setLevel(old)
        if temp_handler is not None:
            logger.removeHandler(temp_handler)


# ---------------------------------------------------------------------------
# Pretty headers (kept identical to the old ``_print_header`` style so that
# notebook output is unchanged)
# ---------------------------------------------------------------------------

_RULE_WIDTH = 58


def header(logger: logging.Logger, title: str, *, width: int = _RULE_WIDTH,
           level: int = logging.INFO) -> None:
    """Emit a 3-line ``===`` header at *level*."""
    rule = "=" * width
    logger.log(level, rule)
    logger.log(level, "  %s", title)
    logger.log(level, rule)


def divider(logger: logging.Logger, *, width: int = _RULE_WIDTH,
            level: int = logging.INFO) -> None:
    """Emit a single ``===`` rule at *level*."""
    logger.log(level, "=" * width)

