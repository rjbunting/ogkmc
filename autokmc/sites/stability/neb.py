"""Shared ASE NEB construction and execution infrastructure.

Diffusion and bond reactions deliberately retain their own endpoint builders,
error hierarchies, and transition-state topology checks.  This module owns the
mechanically identical part of both workflows: ASE compatibility imports,
band construction/interpolation, optimisation, transition-image selection,
and calculator cleanup.

Legacy channel-specific names remain available lazily so importing this module
does not create a diffusion <-> bond import cycle.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from autokmc.io.calculators import (
    CalculatorPool,
    acquire_calculator,
    calculator_batch_active,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.telemetry import instrument

if TYPE_CHECKING:
    from autokmc.sites.stability.bond import (
        BondNEBNotConvergedError,
        BondTransitionStateInvalidError,
        _check_bond_ts_validity,
    )
    from autokmc.sites.stability.diffusion import (
        NEBNotConvergedError,
        TransitionStateInvalidError,
        _check_ts_validity,
    )

try:  # pragma: no cover - depends on installed ASE version
    from ase.mep import NEB
except ImportError:  # pragma: no cover
    from ase.neb import NEB

try:  # pragma: no cover - depends on installed ASE version
    from ase.mep import idpp_interpolate as _idpp_interpolate
except ImportError:  # pragma: no cover
    try:
        from ase.neb import idpp_interpolate as _idpp_interpolate
    except ImportError:
        _idpp_interpolate = None

_log = get_logger(__name__)


@dataclass(frozen=True)
class NEBRunResult:
    """Calculator-detached result of one converged NEB optimisation."""

    atoms_ts: Atoms
    energy_ts: float
    transition_index: int
    n_interior: int
    optimizer_steps: int
    path_energies: list[float] | None = None
    path_images: list[Atoms] | None = None


class _NEBPoolScheduler:
    """Bound concurrent image evaluations by a CalculatorPool's worker limit."""

    def __init__(self, pool: CalculatorPool):
        self.pool = pool
        self.max_workers = max(
            1,
            min(len(pool), int(getattr(pool, "max_workers", len(pool)) or len(pool))),
        )
        self._semaphore = threading.BoundedSemaphore(self.max_workers)

    @contextmanager
    def acquire(self):
        with self._semaphore:
            with self.pool.acquire() as calculator:
                yield calculator


class _PooledNEBCalculator:
    """Per-image ASE calculator facade backed by a shared calculator pool.

    ASE's thread-parallel NEB requires a distinct calculator object on every
    image.  The facades are distinct, while each force evaluation leases one
    real calculator.  Energy is captured during the force call so the
    immediately following energy request does not trigger a second model
    inference after the lease has been returned.
    """

    def __init__(self, scheduler: _NEBPoolScheduler):
        self._scheduler = scheduler
        self._positions: np.ndarray | None = None
        self._cell: np.ndarray | None = None
        self._numbers: np.ndarray | None = None
        self._energy: float | None = None

    def _matches(self, atoms: Atoms) -> bool:
        return bool(
            self._energy is not None
            and self._positions is not None
            and self._cell is not None
            and self._numbers is not None
            and np.array_equal(self._positions, np.asarray(atoms.positions))
            and np.array_equal(self._cell, np.asarray(atoms.cell.array))
            and np.array_equal(self._numbers, np.asarray(atoms.numbers))
        )

    def _remember(self, atoms: Atoms, energy: float) -> None:
        self._positions = np.asarray(atoms.positions, dtype=float).copy()
        self._cell = np.asarray(atoms.cell.array, dtype=float).copy()
        self._numbers = np.asarray(atoms.numbers, dtype=int).copy()
        self._energy = float(energy)

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        with self._scheduler.acquire() as calculator:
            forces = np.asarray(calculator.get_forces(atoms), dtype=float).copy()
            energy = float(calculator.get_potential_energy(atoms))
        self._remember(atoms, energy)
        return forces

    def get_potential_energy(
        self,
        atoms: Atoms,
        force_consistent: bool = False,
    ) -> float:
        if self._matches(atoms):
            return float(self._energy)
        with self._scheduler.acquire() as calculator:
            energy = float(
                calculator.get_potential_energy(
                    atoms,
                    force_consistent=force_consistent,
                )
            )
        self._remember(atoms, energy)
        return energy


def neb_optimizer_logfile(verbose: bool) -> str:
    """Return the ASE optimizer logfile target used by both channels."""
    return "-" if verbose else os.devnull


def make_neb_band(
    atoms_initial: Atoms,
    atoms_final: Atoms,
    *,
    n_images: int,
    interpolation: str,
    spring_k: float,
    climb: bool,
    calculator,
    frozen_indices: list[int] | None,
) -> tuple[Any, list[Atoms]]:
    """Build an ASE NEB band with one shared, non-deepcopyable calculator.

    ``n_images`` counts interior images.  IDPP interpolation falls back to a
    linear path for the same short-band/pathological-geometry cases handled by
    the former channel-local implementations.
    """
    images: list[Atoms] = [atoms_initial.copy()]
    for _ in range(int(n_images)):
        images.append(atoms_initial.copy())
    images.append(atoms_final.copy())

    if frozen_indices:
        for image in images:
            image.set_constraint(FixAtoms(indices=list(frozen_indices)))

    parallel_images = (
        isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and int(getattr(calculator, "max_workers", len(calculator)) or len(calculator)) > 1
        and not calculator_batch_active()
    )
    if parallel_images:
        scheduler = _NEBPoolScheduler(calculator)
        for image in images:
            image.calc = _PooledNEBCalculator(scheduler)
    else:
        for image in images:
            image.calc = calculator

    neb = NEB(
        images,
        k=float(spring_k),
        climb=bool(climb),
        method="improvedtangent",
        parallel=parallel_images,
        allow_shared_calculator=not parallel_images,
    )

    if interpolation == "idpp" and _idpp_interpolate is not None:
        try:
            _idpp_interpolate(neb, mic=True)
        except Exception as exc:
            _log.warning(
                "IDPP interpolation failed (%s: %s); falling back to linear.",
                type(exc).__name__,
                exc,
            )
            neb.interpolate("linear", mic=True)
    else:
        neb.interpolate("linear", mic=True)

    return neb, images


@instrument("neb")
def run_neb(
    atoms_initial: Atoms,
    atoms_final: Atoms,
    *,
    calculator,
    purpose: str,
    n_images: int,
    interpolation: str,
    spring_k: float,
    climb: bool,
    frozen_indices: list[int] | None,
    fmax: float,
    max_steps: int,
    verbose: bool,
    not_converged_error: type[Exception],
    persist_path: bool = False,
    band_factory=None,
    logfile_factory=None,
) -> NEBRunResult:
    """Optimise one NEB band and return a calculator-detached result.

    The caller supplies its channel-specific non-convergence exception class;
    scientific transition-state validation remains in the caller after this
    mechanical optimisation step.
    """
    build_band = band_factory or make_neb_band
    select_logfile = logfile_factory or neb_optimizer_logfile
    use_parallel_pool = (
        isinstance(calculator, CalculatorPool)
        and len(calculator) > 1
        and int(getattr(calculator, "max_workers", len(calculator)) or len(calculator)) > 1
        and not calculator_batch_active()
    )

    @contextmanager
    def _band_calculator():
        if use_parallel_pool:
            yield calculator
            return
        with acquire_calculator(calculator, purpose=purpose) as concrete:
            yield concrete

    with _band_calculator() as neb_calculator:
        neb, images = build_band(
            atoms_initial,
            atoms_final,
            n_images=int(n_images),
            interpolation=str(interpolation),
            spring_k=float(spring_k),
            climb=bool(climb),
            calculator=neb_calculator,
            frozen_indices=frozen_indices,
        )
        try:
            optimizer = BFGS(neb, logfile=select_logfile(verbose))
            optimizer.run(fmax=float(fmax), steps=int(max_steps))

            if not optimizer.converged():
                raise not_converged_error(
                    f"CI-NEB did not converge: fmax={fmax} eV/Å not reached in "
                    f"{max_steps} steps."
                )

            energies = [float(image.get_potential_energy()) for image in images]
            interior = energies[1:-1]
            if not interior:
                raise not_converged_error(
                    "NEB band has no interior images (n_images=0); "
                    "cannot identify a TS."
                )

            transition_index = 1 + int(np.argmax(interior))
            atoms_ts = images[transition_index].copy()
            atoms_ts.calc = None

            path_energies = list(energies) if persist_path else None
            path_images = None
            if persist_path:
                path_images = []
                for image in images:
                    snapshot = image.copy()
                    snapshot.calc = None
                    path_images.append(snapshot)

            result = NEBRunResult(
                atoms_ts=atoms_ts,
                energy_ts=float(energies[transition_index]),
                transition_index=transition_index,
                n_interior=len(interior),
                optimizer_steps=int(optimizer.nsteps),
                path_energies=path_energies,
                path_images=path_images,
            )
        finally:
            for image in images:
                image.calc = None
    return result


# Compatibility aliases retained for callers of the previous private facade.
_make_neb_band = make_neb_band
_make_bond_neb_band = make_neb_band
_neb_optimizer_logfile = neb_optimizer_logfile

_LEGACY_CHANNEL_EXPORTS = {
    "NEBNotConvergedError": (
        "autokmc.sites.stability.diffusion",
        "NEBNotConvergedError",
    ),
    "TransitionStateInvalidError": (
        "autokmc.sites.stability.diffusion",
        "TransitionStateInvalidError",
    ),
    "_check_ts_validity": (
        "autokmc.sites.stability.diffusion",
        "_check_ts_validity",
    ),
    "BondNEBNotConvergedError": (
        "autokmc.sites.stability.bond",
        "BondNEBNotConvergedError",
    ),
    "BondTransitionStateInvalidError": (
        "autokmc.sites.stability.bond",
        "BondTransitionStateInvalidError",
    ),
    "_check_bond_ts_validity": (
        "autokmc.sites.stability.bond",
        "_check_bond_ts_validity",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily forward legacy channel-specific exports without import cycles."""
    try:
        module_name, attribute_name = _LEGACY_CHANNEL_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "NEBRunResult",
    "make_neb_band",
    "neb_optimizer_logfile",
    "run_neb",
    # Legacy facade exports.
    "NEBNotConvergedError",
    "TransitionStateInvalidError",
    "_make_neb_band",
    "_check_ts_validity",
    "BondNEBNotConvergedError",
    "BondTransitionStateInvalidError",
    "_make_bond_neb_band",
    "_check_bond_ts_validity",
]
