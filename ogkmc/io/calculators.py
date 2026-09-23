"""Dynamic ASE calculator construction helpers."""

from __future__ import annotations

import copy
import importlib
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


_CALCULATOR_IDENTITY_ATTRIBUTE = "_ogkmc_calculator_config_identity"
_CALCULATOR_SCIENTIFIC_IDENTITY_ATTRIBUTE = (
	"_ogkmc_calculator_scientific_identity"
)
_CALCULATOR_BATCH_DEPTH: ContextVar[int] = ContextVar(
	"ogkmc_calculator_batch_depth",
	default=0,
)


@contextmanager
def calculator_batch_context():
	"""Mark a worker that is already one member of a calculator-wide batch.

	Expensive kernels use this signal to avoid launching another pool-sized
	layer of image/displacement workers.  The context propagates explicitly to
	the shared executor, so independent sites remain parallel while each site
	uses one calculator at a time.
	"""
	token = _CALCULATOR_BATCH_DEPTH.set(_CALCULATOR_BATCH_DEPTH.get() + 1)
	try:
		yield
	finally:
		_CALCULATOR_BATCH_DEPTH.reset(token)


def calculator_batch_active() -> bool:
	"""Return whether execution is already inside calculator-wide parallelism."""
	return _CALCULATOR_BATCH_DEPTH.get() > 0


@dataclass
class CalculatorCfg:
	"""Pluggable ASE calculator description.

	Exactly one of two construction forms is required:

	* ``import_path`` + ``kwargs`` — dotted path to the calculator class.
	* ``factory`` + ``factory_kwargs`` — dotted path to a callable that
	  returns a calculator instance.

	``import_path`` and ``factory`` are mutually exclusive.

	``copies`` controls how many independent calculator instances are built
	up front.  The science code acquires these instances from a
	CalculatorPool instead of deep-copying live calculator objects.  A complete
	NEB holds one instance; additional copies serve other independent work.
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
	"""Thread-safe pool with a pool-wide limit on active calculator leases."""

	def __init__(self, calculators: list[Any], *, max_workers: int | None = None):
		if not calculators:
			raise CalculatorConfigError("CalculatorPool requires at least one calculator")
		seen_calculators: dict[int, int] = {}
		for index, calculator in enumerate(calculators):
			identity = id(calculator)
			if identity in seen_calculators:
				raise CalculatorConfigError(
					"CalculatorPool requires independent calculator instances; "
					f"entries {seen_calculators[identity]} and {index} refer to "
					"the same object"
				)
			seen_calculators[identity] = index
		self.calculators = list(calculators)
		requested_workers = (
			len(self.calculators)
			if max_workers is None
			else int(max_workers)
		)
		if requested_workers <= 0:
			raise CalculatorConfigError("CalculatorPool max_workers must be positive")
		self.max_workers = min(
			len(self.calculators),
			requested_workers,
		)
		self._queue: queue.Queue[Any] = queue.Queue()
		self._permit_condition = threading.Condition()
		self._available_permits = self.max_workers
		self._thread_permits = threading.local()
		self._executor_lock = threading.Lock()
		self._executor: ThreadPoolExecutor | None = None
		self._closed = False
		for calc in self.calculators:
			self._queue.put(calc)

	@property
	def primary(self):
		return self.calculators[0]

	def __len__(self) -> int:
		return len(self.calculators)

	@property
	def executor(self) -> ThreadPoolExecutor:
		"""Return the pool's lazily created, run-lifetime worker executor."""
		with self._executor_lock:
			if self._closed:
				raise RuntimeError("CalculatorPool has been shut down")
			if self._executor is None:
				self._executor = ThreadPoolExecutor(
					max_workers=self.max_workers,
					thread_name_prefix="ogkmc-calculator",
				)
			return self._executor

	def submit(self, function, /, *args, **kwargs) -> Future:
		"""Schedule work on the shared executor bounded by ``max_workers``."""
		return self.executor.submit(function, *args, **kwargs)

	def gather(self, futures: list[Future]) -> list[Any]:
		"""Wait for a submitted batch, then return results in submission order.

		Waiting for the whole batch before propagating its first exception
		prevents background workers from continuing to mutate site state after
		the scientific caller has already unwound.
		"""
		wait(futures)
		return [future.result() for future in futures]

	def shutdown(
		self,
		*,
		wait: bool = True,
		cancel_futures: bool = False,
	) -> None:
		"""Stop the shared executor; calculator instances remain inspectable."""
		with self._executor_lock:
			self._closed = True
			executor = self._executor
			self._executor = None
		if executor is not None:
			executor.shutdown(
				wait=wait,
				cancel_futures=cancel_futures,
			)

	def _reserve_permits(
		self,
		count: int,
		*,
		purpose: str | None = None,
	) -> None:
		"""Atomically reserve calculator-task capacity.

		A thread that already owns permits must never wait for more: another
		nested owner could be doing the same, leaving every participant blocked
		while collectively holding all capacity.  Immediate nested reservations
		remain supported when enough capacity is free.
		"""
		label = f" for {purpose}" if purpose else ""
		if count > self.max_workers:
			raise CalculatorConfigError(
				f"{count} simultaneous calculator(s){label} requested, but "
				f"the pool allows only {self.max_workers} worker(s). Increase "
				"`calculator.max_workers` or reduce the simultaneous request."
			)
		with self._permit_condition:
			held = int(getattr(self._thread_permits, "count", 0))
			if held and self._available_permits < count:
				raise CalculatorConfigError(
					f"nested request for {count} calculator(s){label} would "
					f"block while this thread already holds {held}. Release "
					"the current calculator lease before requesting more, or "
					"increase `calculator.max_workers`."
				)
			while self._available_permits < count:
				self._permit_condition.wait()
			self._available_permits -= count
			self._thread_permits.count = held + count

	def _release_permits(self, count: int) -> None:
		with self._permit_condition:
			held = int(getattr(self._thread_permits, "count", 0))
			if held < count:
				raise RuntimeError(
					"CalculatorPool permit accounting became inconsistent"
				)
			self._thread_permits.count = held - count
			self._available_permits += count
			self._permit_condition.notify_all()

	def _take_calculator(self) -> Any:
		try:
			return self._queue.get_nowait()
		except queue.Empty as exc:  # pragma: no cover - internal invariant
			raise RuntimeError(
				"CalculatorPool permit and calculator inventories diverged"
			) from exc

	@contextmanager
	def acquire(self, *, purpose: str | None = None):
		self._reserve_permits(1, purpose=purpose)
		try:
			calc = self._take_calculator()
		except BaseException:
			self._release_permits(1)
			raise
		try:
			yield calc
		finally:
			self._queue.put(calc)
			self._release_permits(1)

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
		self._reserve_permits(n, purpose=purpose)
		acquired: list[Any] = []
		try:
			for _ in range(n):
				acquired.append(self._take_calculator())
			yield acquired
		finally:
			for calc in acquired:
				self._queue.put(calc)
			self._release_permits(n)


@contextmanager
def acquire_calculator(calculator, *, purpose: str | None = None):
	"""Yield one concrete calculator from either a pool or a legacy instance."""
	if isinstance(calculator, CalculatorPool):
		with calculator.acquire(purpose=purpose) as calc:
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


def _configuration_identity(cfg: CalculatorCfg) -> dict[str, Any] | None:
	"""Return the scientific constructor/factory declaration for *cfg*."""
	if cfg.factory:
		return {
			"construction": "factory",
			"factory": str(cfg.factory),
			"factory_kwargs": copy.deepcopy(dict(cfg.factory_kwargs or {})),
		}
	if cfg.import_path:
		return {
			"construction": "import",
			"import_path": str(cfg.import_path),
			"kwargs": copy.deepcopy(dict(cfg.kwargs or {})),
		}
	return None


def _stamp_configuration_identity(calculator: Any, identity: dict[str, Any]) -> None:
	"""Attach the unresolved scientific config when the object permits it."""
	try:
		setattr(calculator, _CALCULATOR_IDENTITY_ATTRIBUTE, identity)
	except (AttributeError, TypeError):
		# CalculatorPool is always stamped below, so callers using the standard
		# construction path retain the declaration even for slot-only backends.
		pass


def configured_calculator_identity(calculator: Any) -> dict[str, Any] | None:
	"""Return the config declaration stamped by :func:`build_calculator`."""
	value = getattr(calculator, _CALCULATOR_IDENTITY_ATTRIBUTE, None)
	if isinstance(value, dict):
		return value
	return None


def cached_calculator_scientific_identity(
	calculator: Any,
) -> dict[str, Any] | None:
	"""Return the content-verified identity snapshot attached to *calculator*.

	The expensive snapshot is produced lazily by
	:func:`ogkmc.io.calculation_cache.calculator_identity`, so calculator
	construction remains cheap when the persistent calculation cache is
	disabled.
	"""
	value = getattr(calculator, _CALCULATOR_SCIENTIFIC_IDENTITY_ATTRIBUTE, None)
	if isinstance(value, dict):
		return value
	if isinstance(calculator, CalculatorPool):
		value = getattr(
			calculator.primary,
			_CALCULATOR_SCIENTIFIC_IDENTITY_ATTRIBUTE,
			None,
		)
		if isinstance(value, dict):
			return value
	return None


def stamp_calculator_scientific_identity(
	calculator: Any,
	identity: dict[str, Any],
) -> None:
	"""Attach one immutable scientific-identity snapshot to a calculator pool."""
	targets = (
		[calculator, *calculator.calculators]
		if isinstance(calculator, CalculatorPool)
		else [calculator]
	)
	for target in targets:
		try:
			setattr(
				target,
				_CALCULATOR_SCIENTIFIC_IDENTITY_ATTRIBUTE,
				identity,
			)
		except (AttributeError, TypeError):
			# Most ASE calculators permit arbitrary attributes.  Slot-only
			# third-party calculators simply fall back to recomputation.
			continue


def invalidate_calculator_identity(calculator: Any) -> None:
	"""Discard a cached scientific identity after intentional model mutation.

	A loaded calculator normally represents immutable model weights.  Call this
	helper before reusing an instance whose parameters or backing artifacts were
	changed deliberately.
	"""
	targets = (
		[calculator, *calculator.calculators]
		if isinstance(calculator, CalculatorPool)
		else [calculator]
	)
	for target in targets:
		try:
			delattr(target, _CALCULATOR_SCIENTIFIC_IDENTITY_ATTRIBUTE)
		except (AttributeError, TypeError):
			continue


def build_calculator(cfg: CalculatorCfg) -> CalculatorPool:
	"""Build a calculator pool from an explicit class or factory declaration."""
	if not cfg.import_path and not cfg.factory:
		raise CalculatorConfigError(
			"calculator must configure exactly one of "
			"calculator.import_path or calculator.factory"
		)
	configuration_identity = _configuration_identity(cfg)

	def _build_one(extra_kwargs: dict | None = None):
		if cfg.factory:
			fn = _resolve(cfg.factory)
			kwargs = copy.deepcopy(cfg.factory_kwargs or {})
			for key, value in (extra_kwargs or {}).items():
				_set_dotted(kwargs, key, value)
			kwargs = _resolve_config_value(kwargs)
			calculator = fn(**kwargs)
			if configuration_identity is not None:
				_stamp_configuration_identity(calculator, configuration_identity)
			return calculator
		if cfg.import_path:
			cls = _resolve(cfg.import_path)
			kwargs = copy.deepcopy(cfg.kwargs or {})
			for key, value in (extra_kwargs or {}).items():
				_set_dotted(kwargs, key, value)
			kwargs = _resolve_config_value(kwargs)
			calculator = cls(**kwargs)
			if configuration_identity is not None:
				_stamp_configuration_identity(calculator, configuration_identity)
			return calculator
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
	if not calculators:
		raise CalculatorConfigError(
			"calculator configuration did not construct a calculator"
		)
	pool = CalculatorPool(calculators, max_workers=cfg.max_workers)
	if configuration_identity is not None:
		_stamp_configuration_identity(pool, configuration_identity)
	return pool


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
	"calculator_batch_active",
	"calculator_batch_context",
	"cached_calculator_scientific_identity",
	"configured_calculator_identity",
	"invalidate_calculator_identity",
	"primary_calculator",
	"stamp_calculator_scientific_identity",
	"calculator_meta",
]
