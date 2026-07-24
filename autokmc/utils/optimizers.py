"""Names and validation helpers for configurable ASE optimizers."""

from __future__ import annotations

DEFAULT_OPTIMIZER = "lbfgs"
DEFAULT_NEB_OPTIMIZER = "bfgs"

REGULAR_OPTIMIZERS = frozenset({"lbfgs", "bfgs", "fire", "mdmin"})
NEB_OPTIMIZERS = frozenset({"bfgs", "fire", "mdmin"})


def normalize_optimizer_name(
    value: str,
    *,
    allowed: frozenset[str] = REGULAR_OPTIMIZERS,
    setting: str = "optimizer",
) -> str:
    """Return a canonical optimizer name or raise a useful error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{setting} must be a non-empty string, got {value!r}")
    name = value.strip().lower()
    if name not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"{setting} must be one of {choices}; got {value!r}"
        )
    return name


__all__ = [
    "DEFAULT_NEB_OPTIMIZER",
    "DEFAULT_OPTIMIZER",
    "NEB_OPTIMIZERS",
    "REGULAR_OPTIMIZERS",
    "normalize_optimizer_name",
]
