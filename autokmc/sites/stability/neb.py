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
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.optimize import BFGS, FIRE, MDMin

from autokmc.io.calculators import (
    CalculatorConfigError,
    CalculatorPool,
    acquire_calculator,
    calculator_batch_active,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import (
    DEFAULT_NEB_OPTIMIZER,
    NEB_OPTIMIZERS,
    normalize_optimizer_name,
)
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
    """Calculator-detached result of a converged NEB optimisation."""

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
        self._forces: np.ndarray | None = None

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

    def _remember(
        self,
        atoms: Atoms,
        energy: float,
        forces: np.ndarray | None = None,
    ) -> None:
        self._positions = np.asarray(atoms.positions, dtype=float).copy()
        self._cell = np.asarray(atoms.cell.array, dtype=float).copy()
        self._numbers = np.asarray(atoms.numbers, dtype=int).copy()
        self._energy = float(energy)
        self._forces = None if forces is None else np.asarray(forces, dtype=float).copy()

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        if self._matches(atoms) and self._forces is not None:
            return self._forces.copy()
        with self._scheduler.acquire() as calculator:
            forces = np.asarray(calculator.get_forces(atoms), dtype=float).copy()
            energy = float(calculator.get_potential_energy(atoms))
        self._remember(atoms, energy, forces)
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


class _ExceptionSafeParallelNEB(NEB):
    """NEB with concurrent image evaluation and synchronous error propagation.

    ASE's local ``parallel=True`` backend uses raw ``threading.Thread`` objects.
    Exceptions raised by those threads do not reach ``get_forces()``, which can
    leave uninitialised force/energy entries and let an optimiser continue with
    a corrupt band.  This facade pre-evaluates every interior image with futures,
    observes all results, and then asks ASE to assemble the band serially from
    the calculator caches.
    """

    def __init__(self, *args, image_max_workers: int, **kwargs):
        self._image_max_workers = max(1, int(image_max_workers))
        super().__init__(*args, parallel=False, **kwargs)
        # Preserve the public indication that image evaluations are concurrent.
        # ``get_forces`` temporarily disables ASE's unsafe raw-thread branch.
        self.parallel = True

    def _prefetch_interior_forces(self) -> None:
        interior_images = self.images[1:-1]
        if not interior_images:
            return

        with ThreadPoolExecutor(
            max_workers=min(self._image_max_workers, len(interior_images)),
            thread_name_prefix="autokmc-neb-image",
        ) as executor:
            futures = [
                executor.submit(image.get_forces)
                for image in interior_images
            ]
            wait(futures)

            first_failure: tuple[BaseException, Any] | None = None
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if first_failure is None:
                        first_failure = (exc, exc.__traceback__)

        if first_failure is not None:
            error, traceback = first_failure
            raise error.with_traceback(traceback)

    def get_forces(self):
        self._prefetch_interior_forces()
        self.parallel = False
        try:
            return super().get_forces()
        finally:
            self.parallel = True


def neb_optimizer_logfile(verbose: bool) -> str:
    """Return the ASE optimizer logfile target used by both channels."""
    return "-" if verbose else os.devnull


def project_neb_path(
    source_path: Sequence[Atoms],
    atoms_initial: Atoms,
    atoms_final: Atoms,
    *,
    n_slab: int,
    n_lateral: int,
) -> list[Atoms] | None:
    """Project a bare optimized path into a lateral endpoint pair.

    ``source_path`` must use the bare layout ``[slab | reacting]`` while the
    target endpoints use ``[slab | lateral | reacting]``.  The target path is
    first interpolated linearly with minimum-image displacements.  For each
    interior image, the source path's minimum-image displacement away from its
    own linear path is then transferred to the matching slab and reacting
    atoms.  Lateral atoms therefore remain on the target linear path. The
    source and target must represent the same concrete reaction member; a
    caller must not transfer unrotated Cartesian residuals between different
    symmetry-equivalent members. The returned endpoint images are exact copies
    of ``atoms_initial`` and ``atoms_final``.

    ``None`` is a deliberate fallback signal.  It is returned for any layout,
    chemistry, cell, periodicity, image-count, or finite-coordinate mismatch
    rather than constructing a potentially invalid NEB band.
    """
    try:
        source_images = list(source_path)
    except TypeError:
        return None

    try:
        n_slab = int(n_slab)
        n_lateral = int(n_lateral)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        len(source_images) < 3
        or n_slab < 0
        or n_lateral < 0
        or len(atoms_initial) != len(atoms_final)
    ):
        return None

    n_reacting = len(atoms_initial) - n_slab - n_lateral
    if n_reacting <= 0:
        return None
    n_source = n_slab + n_reacting
    if any(
        not isinstance(image, Atoms) or len(image) != n_source
        for image in source_images
    ):
        return None

    target_cell = np.asarray(atoms_initial.cell.array, dtype=float)
    target_pbc = np.asarray(atoms_initial.pbc, dtype=bool)
    if (
        not np.isfinite(target_cell).all()
        or not np.array_equal(target_pbc, np.asarray(atoms_final.pbc, dtype=bool))
        or not np.allclose(
            target_cell,
            np.asarray(atoms_final.cell.array, dtype=float),
            rtol=0.0,
            atol=1.0e-8,
        )
    ):
        return None

    source_cell = np.asarray(source_images[0].cell.array, dtype=float)
    source_pbc = np.asarray(source_images[0].pbc, dtype=bool)
    if (
        not np.isfinite(source_cell).all()
        or not np.array_equal(source_pbc, target_pbc)
        or not np.allclose(
            source_cell,
            target_cell,
            rtol=0.0,
            atol=1.0e-8,
        )
    ):
        return None
    for image in source_images[1:]:
        if (
            not np.array_equal(np.asarray(image.pbc, dtype=bool), source_pbc)
            or not np.allclose(
                np.asarray(image.cell.array, dtype=float),
                source_cell,
                rtol=0.0,
                atol=1.0e-8,
            )
        ):
            return None

    target_numbers_initial = np.asarray(atoms_initial.numbers, dtype=int)
    target_numbers_final = np.asarray(atoms_final.numbers, dtype=int)
    source_numbers = np.asarray(source_images[0].numbers, dtype=int)
    target_reacting_indices = np.arange(
        n_slab + n_lateral,
        n_slab + n_lateral + n_reacting,
        dtype=int,
    )
    target_common_indices = np.concatenate(
        (np.arange(n_slab, dtype=int), target_reacting_indices)
    )
    source_common_indices = np.arange(n_source, dtype=int)
    if (
        not np.array_equal(target_numbers_initial, target_numbers_final)
        or not np.array_equal(
            source_numbers,
            target_numbers_initial[
                np.concatenate(
                    (np.arange(n_slab, dtype=int), target_reacting_indices)
                )
            ],
        )
        or any(
            not np.array_equal(np.asarray(image.numbers, dtype=int), source_numbers)
            for image in source_images[1:]
        )
    ):
        return None

    all_positions = (
        [np.asarray(image.positions, dtype=float) for image in source_images]
        + [
            np.asarray(atoms_initial.positions, dtype=float),
            np.asarray(atoms_final.positions, dtype=float),
        ]
    )
    if not all(np.isfinite(positions).all() for positions in all_positions):
        return None

    def _minimum_image(vectors: np.ndarray) -> np.ndarray:
        if not np.any(target_pbc):
            return np.asarray(vectors, dtype=float).copy()
        mic_vectors, _ = find_mic(vectors, cell=target_cell, pbc=target_pbc)
        return np.asarray(mic_vectors, dtype=float)

    source_initial_positions = all_positions[0]
    source_delta = _minimum_image(
        all_positions[len(source_images) - 1] - source_initial_positions
    )
    target_initial_positions = all_positions[-2]
    target_delta = _minimum_image(all_positions[-1] - target_initial_positions)

    projected = [atoms_initial.copy()]
    denominator = float(len(source_images) - 1)
    for image_index, source_image in enumerate(source_images[1:-1], start=1):
        fraction = float(image_index) / denominator
        target_positions = target_initial_positions + fraction * target_delta
        source_linear = source_initial_positions + fraction * source_delta
        source_residual = _minimum_image(
            np.asarray(source_image.positions, dtype=float) - source_linear
        )
        target_positions[target_common_indices] += source_residual[
            source_common_indices
        ]

        image = atoms_initial.copy()
        image.set_positions(target_positions, apply_constraint=False)
        image.calc = None
        projected.append(image)

    projected.append(atoms_final.copy())
    for image in projected:
        image.calc = None
    return projected


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
    initial_path: Sequence[Atoms] | None = None,
) -> tuple[Any, list[Atoms]]:
    """Build an ASE NEB band with one shared, non-deepcopyable calculator.

    ``n_images`` counts interior images.  A compatible ``initial_path`` is
    copied directly into the band; otherwise IDPP interpolation falls back to
    a linear path for the same short-band/pathological-geometry cases handled
    by the former channel-local implementations.
    """
    images: list[Atoms] | None = None
    if initial_path is not None:
        try:
            candidates = list(initial_path)
        except TypeError:
            candidates = []
        expected_count = int(n_images) + 2
        target_numbers = np.asarray(atoms_initial.numbers, dtype=int)
        target_cell = np.asarray(atoms_initial.cell.array, dtype=float)
        target_pbc = np.asarray(atoms_initial.pbc, dtype=bool)
        compatible = (
            len(candidates) == expected_count
            and len(atoms_initial) == len(atoms_final)
            and np.array_equal(
                target_numbers,
                np.asarray(atoms_final.numbers, dtype=int),
            )
            and np.array_equal(
                target_pbc,
                np.asarray(atoms_final.pbc, dtype=bool),
            )
            and np.allclose(
                target_cell,
                np.asarray(atoms_final.cell.array, dtype=float),
                rtol=0.0,
                atol=1.0e-8,
            )
            and all(
                isinstance(image, Atoms)
                and len(image) == len(atoms_initial)
                and np.array_equal(
                    np.asarray(image.numbers, dtype=int),
                    target_numbers,
                )
                and np.array_equal(
                    np.asarray(image.pbc, dtype=bool),
                    target_pbc,
                )
                and np.allclose(
                    np.asarray(image.cell.array, dtype=float),
                    target_cell,
                    rtol=0.0,
                    atol=1.0e-8,
                )
                and np.isfinite(
                    np.asarray(image.positions, dtype=float)
                ).all()
                for image in candidates
            )
        )
        if compatible:
            images = [atoms_initial.copy()]
            images.extend(image.copy() for image in candidates[1:-1])
            images.append(atoms_final.copy())
        else:
            _log.warning(
                "Bare NEB warm-start path is incompatible with the requested "
                "band; falling back to %s interpolation.",
                interpolation,
            )

    seeded = images is not None
    if images is None:
        images = [atoms_initial.copy()]
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

    neb_kwargs = {
        "k": float(spring_k),
        "climb": bool(climb),
        "method": "improvedtangent",
        "allow_shared_calculator": not parallel_images,
    }
    if parallel_images:
        neb = _ExceptionSafeParallelNEB(
            images,
            image_max_workers=scheduler.max_workers,
            **neb_kwargs,
        )
    else:
        neb = NEB(images, parallel=False, **neb_kwargs)

    if not seeded and interpolation == "idpp" and _idpp_interpolate is not None:
        # ASE's public ``NEB.interpolate(method="idpp")`` first builds a
        # linear path before invoking the low-level IDPP optimiser.  Calling
        # ``idpp_interpolate`` directly on the initial-state copies above
        # leaves coincident interior images, so the improved tangent has zero
        # norm and IDPP can spend all of its steps propagating NaNs.
        neb.interpolate("linear", mic=True)
        linear_positions = [
            np.asarray(image.positions, dtype=float).copy()
            for image in images
        ]
        real_calculators = [image.calc for image in images]
        try:
            with warnings.catch_warnings():
                # Numerical warnings during IDPP otherwise do not stop ASE's
                # optimiser, which can continue through every requested step
                # with a non-finite band.
                warnings.simplefilter("error", RuntimeWarning)
                _idpp_interpolate(
                    neb,
                    traj=None,
                    log=None,
                    mic=True,
                )
            if not all(
                np.isfinite(np.asarray(image.positions, dtype=float)).all()
                for image in images
            ):
                raise FloatingPointError(
                    "IDPP interpolation returned non-finite positions"
                )
        except Exception as exc:
            for image, positions in zip(images, linear_positions):
                image.set_positions(positions, apply_constraint=False)
            _log.warning(
                "IDPP interpolation failed (%s: %s); falling back to linear.",
                type(exc).__name__,
                exc,
            )
        finally:
            # ASE restores these after a successful IDPP run, but not if its
            # optimiser raises before reaching the restoration loop.
            for image, calculator_for_image in zip(images, real_calculators):
                image.calc = calculator_for_image
    elif not seeded:
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
    optimizer: str = DEFAULT_NEB_OPTIMIZER,
    persist_path: bool = False,
    capture_path: bool = False,
    initial_path: Sequence[Atoms] | None = None,
    initial_path_callback: Callable[[list[Atoms]], None] | None = None,
    failure_path_callback: Callable[[list[Atoms]], None] | None = None,
    band_factory=None,
    logfile_factory=None,
) -> NEBRunResult:
    """Optimise one NEB band and return a calculator-detached result.

    The caller supplies its channel-specific non-convergence exception class;
    scientific transition-state validation remains in the caller after this
    mechanical optimisation step.  When ``climb`` is requested, the ordinary
    NEB is converged first and the same band is then converged again after
    enabling its climbing image.  ``max_steps`` applies independently to each
    stage, while ``optimizer_steps`` reports their combined step count.  If an
    optimizer, calculator, or convergence check raises after band construction,
    *failure_path_callback* receives a calculator-detached snapshot of the
    last-known full band before the exception is re-raised.
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
        band_kwargs = {
            "n_images": int(n_images),
            "interpolation": str(interpolation),
            "spring_k": float(spring_k),
            "climb": False,
            "calculator": neb_calculator,
            "frozen_indices": frozen_indices,
        }
        if initial_path is not None:
            band_kwargs["initial_path"] = initial_path
        try:
            neb, images = build_band(
                atoms_initial,
                atoms_final,
                **band_kwargs,
            )
        except CalculatorConfigError:
            raise
        except Exception as exc:
            if isinstance(exc, not_converged_error):
                raise
            raise not_converged_error(
                "NEB band construction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            if initial_path_callback is not None:
                initial_snapshot_path = []
                for image in images:
                    snapshot = image.copy()
                    snapshot.calc = None
                    initial_snapshot_path.append(snapshot)
                initial_path_callback(initial_snapshot_path)

            optimizer_steps = 0

            def _optimise_stage(*, stage: str) -> None:
                nonlocal optimizer_steps
                optimizer_name = normalize_optimizer_name(
                    optimizer,
                    allowed=NEB_OPTIMIZERS,
                    setting="neb_optimizer",
                )
                optimizer_cls = {
                    "bfgs": BFGS,
                    "fire": FIRE,
                    "mdmin": MDMin,
                }[optimizer_name]
                stage_optimizer = optimizer_cls(
                    neb,
                    logfile=select_logfile(verbose),
                )
                stage_optimizer.run(
                    fmax=float(fmax),
                    steps=int(max_steps),
                )
                optimizer_steps += int(stage_optimizer.nsteps)
                if not stage_optimizer.converged():
                    raise not_converged_error(
                        f"{stage} did not converge: fmax={fmax} eV/Å not "
                        f"reached in {max_steps} steps."
                    )

            _optimise_stage(
                stage="NEB pre-climb relaxation" if climb else "NEB"
            )
            if climb:
                neb.climb = True
                _optimise_stage(stage="CI-NEB")

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

            retain_path = bool(persist_path or capture_path)
            path_energies = list(energies) if retain_path else None
            path_images = None
            if retain_path:
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
                optimizer_steps=optimizer_steps,
                path_energies=path_energies,
                path_images=path_images,
            )
        except Exception as exc:
            if failure_path_callback is not None:
                failed_path = []
                for image in images:
                    snapshot = image.copy()
                    snapshot.calc = None
                    failed_path.append(snapshot)
                try:
                    failure_path_callback(failed_path)
                except Exception as callback_exc:
                    _log.warning(
                        "Could not retain failed NEB path for diagnostics: %s",
                        callback_exc,
                    )
            if isinstance(exc, (CalculatorConfigError, not_converged_error)):
                raise
            raise not_converged_error(
                "NEB optimization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
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
    "project_neb_path",
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
