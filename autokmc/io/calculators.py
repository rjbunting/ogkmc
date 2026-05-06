"""Dynamic ASE calculator construction helpers."""

from __future__ import annotations

import copy
import importlib
import queue
from contextlib import contextmanager
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
	copies:          int        = 1
	gpu_devices:     list[str] | tuple[str, ...] | None = None
	gpu_device_arg:  str       = "device"
	max_workers:     int | None = None


class CalculatorConfigError(ValueError):
	"""Raised when a dynamic calculator path cannot be resolved."""


class CalculatorPool:
	"""Small thread-safe pool of independent ASE calculator instances."""

	def __init__(self, calculators: list[Any], *, max_workers: int | None = None):
		if not calculators:
			raise CalculatorConfigError("CalculatorPool requires at least one calculator")
		self.calculators = list(calculators)
		self.max_workers = int(max_workers or len(self.calculators))
		self._queue: queue.Queue[Any] = queue.Queue()
		for calc in self.calculators:
			self._queue.put(calc)

	@property
	def primary(self):
		return self.calculators[0]

	def __len__(self) -> int:
		return len(self.calculators)

	@contextmanager
	def acquire(self):
		calc = self._queue.get()
		try:
			yield calc
		finally:
			self._queue.put(calc)


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
	def _build_one(extra_kwargs: dict | None = None):
		if cfg.factory:
			fn = _resolve(cfg.factory)
			kwargs = dict(cfg.factory_kwargs or {})
			kwargs.update(extra_kwargs or {})
			return fn(**kwargs)
		if cfg.import_path:
			cls = _resolve(cfg.import_path)
			kwargs = dict(cfg.kwargs or {})
			kwargs.update(extra_kwargs or {})
			return cls(**kwargs)
		return None

	devices = list(cfg.gpu_devices or [])
	n_copies = max(int(cfg.copies or 1), len(devices) or 1)
	if n_copies <= 1:
		extra = (
			{str(cfg.gpu_device_arg): devices[0]}
			if devices else None
		)
		return _build_one(extra)
	calculators = []
	for idx in range(n_copies):
		extra = (
			{str(cfg.gpu_device_arg): devices[idx]}
			if idx < len(devices) else None
		)
		if idx == 0 or devices:
			calc = _build_one(extra)
		else:
			calc = copy.deepcopy(calculators[0])
		if calc is not None:
			calculators.append(calc)
	if calculators:
		return CalculatorPool(calculators, max_workers=cfg.max_workers)
	return None


def primary_calculator(calculator):
	"""Return a concrete ASE calculator from a calculator or CalculatorPool."""
	if isinstance(calculator, CalculatorPool):
		return calculator.primary
	return calculator


def calculator_meta(cfg: CalculatorCfg) -> dict[str, Any]:
	"""Echo the calculator description into a JSON-friendly metadata dict."""
	return {
		"import_path":    cfg.import_path,
		"kwargs":         dict(cfg.kwargs or {}),
		"factory":        cfg.factory,
		"factory_kwargs": dict(cfg.factory_kwargs or {}),
		"copies":         int(cfg.copies or 1),
		"gpu_devices":    list(cfg.gpu_devices or []),
		"gpu_device_arg": cfg.gpu_device_arg,
		"max_workers":    cfg.max_workers,
	}


__all__ = [
	"CalculatorCfg",
	"CalculatorPool",
	"CalculatorConfigError",
	"_resolve",
	"build_calculator",
	"primary_calculator",
	"calculator_meta",
]
