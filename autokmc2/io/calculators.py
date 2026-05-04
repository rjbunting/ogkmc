"""Dynamic ASE calculator construction helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalculatorCfg:
	"""Pluggable ASE calculator description.

	Two equivalent forms are supported:

	* ``import_path`` + ``kwargs`` — dotted path to the calculator class.
	* ``factory`` + ``factory_kwargs`` — dotted path to a callable that
	  returns a calculator instance.

	``factory`` takes precedence when both are supplied.
	"""
	import_path:     str | None = None
	kwargs:          dict       = field(default_factory=dict)
	factory:         str | None = None
	factory_kwargs:  dict       = field(default_factory=dict)


class CalculatorConfigError(ValueError):
	"""Raised when a dynamic calculator path cannot be resolved."""


def _resolve(dotted: str):
	"""Resolve a dotted path like ``"pkg.mod.Class.method"`` to an object."""
	if ":" in dotted:
		mod_name, rest = dotted.split(":", 1)
		mod = importlib.import_module(mod_name)
		obj = mod
		for part in rest.split("."):
			try:
				obj = getattr(obj, part)
			except AttributeError as exc:
				raise CalculatorConfigError(
					f"cannot resolve {dotted!r}: {mod_name!r} has no {part!r}"
				) from exc
		return obj

	parts = dotted.split(".")
	if len(parts) < 2:
		raise CalculatorConfigError(
			f"calculator path {dotted!r} must be dotted (e.g. pkg.mod.Class)"
		)

	last_err: Exception | None = None
	for split in range(len(parts) - 1, 0, -1):
		mod_name = ".".join(parts[:split])
		attr_chain = parts[split:]
		try:
			mod = importlib.import_module(mod_name)
		except ImportError as exc:
			last_err = exc
			continue
		obj = mod
		for attr in attr_chain:
			try:
				obj = getattr(obj, attr)
			except AttributeError as exc:
				raise CalculatorConfigError(
					f"cannot resolve {dotted!r}: {obj!r} has no attribute {attr!r}"
				) from exc
		return obj

	raise CalculatorConfigError(
		f"cannot import any module prefix of {dotted!r}. Last import error: {last_err}"
	)


def build_calculator(cfg: CalculatorCfg):
	"""Instantiate the ASE-compatible calculator described by *cfg*.

	Returns ``None`` when neither *import_path* nor *factory* is set.
	"""
	if cfg.factory:
		fn = _resolve(cfg.factory)
		return fn(**(cfg.factory_kwargs or {}))
	if cfg.import_path:
		cls = _resolve(cfg.import_path)
		return cls(**(cfg.kwargs or {}))
	return None


def calculator_meta(cfg: CalculatorCfg) -> dict[str, Any]:
	"""Echo the calculator description into a JSON-friendly metadata dict."""
	return {
		"import_path":    cfg.import_path,
		"kwargs":         dict(cfg.kwargs or {}),
		"factory":        cfg.factory,
		"factory_kwargs": dict(cfg.factory_kwargs or {}),
	}


__all__ = [
	"CalculatorCfg",
	"CalculatorConfigError",
	"_resolve",
	"build_calculator",
	"calculator_meta",
]
