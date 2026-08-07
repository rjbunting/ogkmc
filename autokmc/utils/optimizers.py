"""Names and validation helpers for configurable ASE optimizers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from ase.optimize import BFGS, FIRE, LBFGS, MDMin

DEFAULT_OPTIMIZER = "lbfgs"
DEFAULT_NEB_OPTIMIZER = "bfgs"

REGULAR_OPTIMIZERS = frozenset({"lbfgs", "bfgs", "fire", "mdmin"})
NEB_OPTIMIZERS = frozenset({"bfgs", "fire", "mdmin"})

_OPTIMIZER_CLASSES = {
    "lbfgs": LBFGS,
    "bfgs": BFGS,
    "fire": FIRE,
    "mdmin": MDMin,
}
_MANAGED_OPTIMIZER_KWARGS = frozenset({"atoms", "logfile"})


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


def normalize_optimizer_kwargs(
    name: str,
    value: Mapping[str, Any] | None,
    *,
    allowed: frozenset[str] = REGULAR_OPTIMIZERS,
    setting: str = "optimizer_kwargs",
) -> dict[str, Any]:
    """Validate constructor keywords against the installed ASE optimizer.

    AutoKMC owns the optimized object and logfile, while every other keyword
    accepted by the selected ASE optimizer is forwarded unchanged.  Inspecting
    ASE at runtime keeps this interface aligned with the installed version.
    """
    optimizer_name = normalize_optimizer_name(
        name,
        allowed=allowed,
        setting=setting.removesuffix("_kwargs"),
    )
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{setting} must be a mapping, got {value!r}")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"{setting} keys must be non-empty strings")

    signature = inspect.signature(_OPTIMIZER_CLASSES[optimizer_name].__init__)
    accepted = {
        parameter_name
        for parameter_name, parameter in signature.parameters.items()
        if parameter_name != "self"
        and parameter.kind
        in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }
    accepts_arbitrary_keywords = any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    managed = sorted(set(value) & _MANAGED_OPTIMIZER_KWARGS)
    if managed:
        raise ValueError(
            f"{setting} cannot override AutoKMC-managed argument(s): "
            f"{', '.join(managed)}"
        )
    unknown = [] if accepts_arbitrary_keywords else sorted(set(value) - accepted)
    if unknown:
        choices = ", ".join(sorted(accepted - _MANAGED_OPTIMIZER_KWARGS))
        raise ValueError(
            f"{setting} has unsupported {optimizer_name} argument(s): "
            f"{', '.join(unknown)}; accepted arguments: {choices}"
        )
    return dict(value)


__all__ = [
    "DEFAULT_NEB_OPTIMIZER",
    "DEFAULT_OPTIMIZER",
    "NEB_OPTIMIZERS",
    "REGULAR_OPTIMIZERS",
    "normalize_optimizer_name",
    "normalize_optimizer_kwargs",
]
