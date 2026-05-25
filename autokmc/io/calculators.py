"""Dynamic ASE calculator construction helpers."""

from __future__ import annotations

import copy
import importlib
import queue
import threading
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

	``copies`` controls how many independent calculator instances are built
	up front.  The science code acquires these instances from a
	CalculatorPool instead of deep-copying live calculator objects.
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
		self._batch_lock = threading.Lock()
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

	@contextmanager
	def acquire_many(self, count: int, *, purpose: str | None = None):
		"""Reserve multiple independent calculators for one operation."""
		n = int(count)
		if n < 0:
			raise CalculatorConfigError("cannot acquire a negative number of calculators")
		if n == 0:
			yield []
			return
		if n > len(self.calculators):
			label = f" for {purpose}" if purpose else ""
			raise CalculatorConfigError(
				f"{n} independent calculator(s){label} requested, but "
				f"the pool has only {len(self.calculators)}. Increase "
				"`calculator.copies` or reduce the number of simultaneous "
				"images/calculations."
			)
		acquired: list[Any] = []
		try:
			with self._batch_lock:
				for _ in range(n):
					acquired.append(self._queue.get())
			yield acquired
		finally:
			for calc in acquired:
				self._queue.put(calc)


@contextmanager
def acquire_calculator(calculator, *, purpose: str | None = None):
	"""Yield one concrete calculator from either a pool or a legacy instance."""
	if isinstance(calculator, CalculatorPool):
		with calculator.acquire() as calc:
			yield calc
	else:
		yield calculator


@contextmanager
def acquire_calculators(calculator, count: int, *, purpose: str | None = None):
	"""Yield *count* independent calculators without relying on deepcopy."""
	n = int(count)
	if n <= 0:
		yield []
		return
	if isinstance(calculator, CalculatorPool):
		with calculator.acquire_many(n, purpose=purpose) as calcs:
			yield calcs
		return
	if n == 1:
		yield [calculator]
		return
	label = f" for {purpose}" if purpose else ""
	raise CalculatorConfigError(
		f"{n} independent calculator(s){label} requested, but a single "
		"calculator instance was supplied. Pass a CalculatorPool or configure "
		"`calculator.copies` with enough instances."
	)


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


def _looks_like_factory_spec(value: Any) -> bool:
	return isinstance(value, dict) and (
		"factory" in value or "import_path" in value
	)


def _set_dotted(mapping: dict, dotted: str, value: Any) -> None:
	parts = str(dotted).split(".")
	target = mapping
	for part in parts[:-1]:
		child = target.get(part)
		if child is None:
			child = {}
			target[part] = child
		if not isinstance(child, dict):
			raise CalculatorConfigError(
				f"cannot set calculator argument {dotted!r}: "
				f"{part!r} is not a mapping"
			)
		target = child
	target[parts[-1]] = value


def _build_factory_spec(spec: dict):
	"""Instantiate a nested generic factory/import spec from config data."""
	if spec.get("factory"):
		callable_obj = _resolve(str(spec["factory"]))
		raw_kwargs = dict(spec.get("factory_kwargs") or {})
	elif spec.get("import_path"):
		callable_obj = _resolve(str(spec["import_path"]))
		raw_kwargs = dict(spec.get("kwargs") or {})
	else:
		raise CalculatorConfigError(
			"nested calculator spec must include 'factory' or 'import_path'"
		)
	args = [_resolve_config_value(v) for v in spec.get("args", ())]
	kwargs = {k: _resolve_config_value(v) for k, v in raw_kwargs.items()}
	return callable_obj(*args, **kwargs)


def _resolve_config_value(value: Any):
	if _looks_like_factory_spec(value):
		return _build_factory_spec(value)
	if isinstance(value, dict):
		return {k: _resolve_config_value(v) for k, v in value.items()}
	if isinstance(value, list):
		return [_resolve_config_value(v) for v in value]
	if isinstance(value, tuple):
		return tuple(_resolve_config_value(v) for v in value)
	return value


def build_calculator(cfg: CalculatorCfg):
	"""Build a CalculatorPool for the calculator described by *cfg*.

	Returns ``None`` when neither *import_path* nor *factory* is set.
	"""
	def _build_one(extra_kwargs: dict | None = None):
		if cfg.factory:
			fn = _resolve(cfg.factory)
			kwargs = copy.deepcopy(cfg.factory_kwargs or {})
			for key, value in (extra_kwargs or {}).items():
				_set_dotted(kwargs, key, value)
			kwargs = _resolve_config_value(kwargs)
			return fn(**kwargs)
		if cfg.import_path:
			cls = _resolve(cfg.import_path)
			kwargs = copy.deepcopy(cfg.kwargs or {})
			for key, value in (extra_kwargs or {}).items():
				_set_dotted(kwargs, key, value)
			kwargs = _resolve_config_value(kwargs)
			return cls(**kwargs)
		return None

	calculators = []
	devices = list(cfg.gpu_devices or [])
	n_copies = max(int(cfg.copies or 1), len(devices) or 1)
	for idx in range(n_copies):
		extra = (
			{str(cfg.gpu_device_arg): devices[idx]}
			if idx < len(devices) else None
		)
		calc = _build_one(extra)
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
	"acquire_calculator",
	"acquire_calculators",
	"build_calculator",
	"primary_calculator",
	"calculator_meta",
]
