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
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.optimize import BFGS, FIRE, MDMin

from autokmc.core.constants import (
    NEB_BAND_EVAL as DEFAULT_NEB_BAND_EVAL,
    NEB_BAND_EVALS,
)
from autokmc.io.calculators import (
    CalculatorConfigError,
    CalculatorPool,
    acquire_calculator,
    primary_calculator,
)
from autokmc.sites.stability.band_eval import (
    BandEvaluator,
    BandImageCalculator,
    resolve_band_evaluator,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import (
    DEFAULT_NEB_OPTIMIZER,
    NEB_OPTIMIZERS,
    normalize_optimizer_name,
    normalize_optimizer_kwargs,
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

#: Set once the first time a batchable calculator is seen under the ``images``
#: default, so the "you could turn on batching" hint is logged a single time
#: per process rather than on every barrier.
_batched_hint_emitted = False


def _maybe_hint_batched_available(calculator: Any) -> None:
    """Log a one-time hint if this calculator could use batched NEB.

    Fires only under the ``images`` default: many runs use a FAIR-Chem/UMA
    calculator that supports whole-band batching but never flip the flag
    because the default is silent.  Never raises: a hint must not affect a run.
    """
    global _batched_hint_emitted
    if _batched_hint_emitted:
        return
    try:
        probe = primary_calculator(calculator)
        if resolve_band_evaluator(probe) is not None:
            _batched_hint_emitted = True
            _log.info(
                "Calculator %s supports batched NEB evaluation. Set "
                "optimization.neb_band_eval: batched for roughly 10x faster "
                "barriers with identical results.",
                type(probe).__name__,
            )
    except Exception:  # pragma: no cover - a hint must never break a run
        pass


def normalize_band_eval(value: str) -> str:
    """Return a canonical band-eval mode or raise a useful error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"neb_band_eval must be a non-empty string, got {value!r}"
        )
    name = value.strip().lower()
    if name not in NEB_BAND_EVALS:
        choices = ", ".join(sorted(NEB_BAND_EVALS))
        raise ValueError(
            f"neb_band_eval must be one of {choices}; got {value!r}"
        )
    return name


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


class _BatchedBandNEB(NEB):
    """NEB whose images are evaluated in one batched model call per step.

    The band's tangent/spring/climbing math and the optimizer are untouched
    ASE code: before ASE assembles the band, every image is evaluated by a
    single ``BandEvaluator.evaluate_band`` call and the results are primed
    into per-image cache facades, so ASE's per-image queries are served from
    the batch.  Endpoints never move, so they join the batch only on the
    first evaluation.  Constraint handling is identical to the serial path —
    the batch carries raw model forces and ASE applies ``FixAtoms``
    projections per image.
    """

    def __init__(self, *args, band_evaluator: BandEvaluator, **kwargs):
        super().__init__(*args, parallel=False, **kwargs)
        self._band_evaluator = band_evaluator
        self._endpoints_evaluated = False

    def _prefetch_band(self) -> None:
        if self._endpoints_evaluated:
            targets = list(self.images[1:-1])
        else:
            targets = list(self.images)
        if not targets:
            return
        if any(
            not isinstance(image.calc, BandImageCalculator)
            for image in targets
        ):
            # Foreign per-image calculators are attached (e.g. ASE's IDPP
            # interpolation temporarily swaps them in) — evaluate normally.
            return
        # ASE optimizers issue several ``get_forces`` per step and lean on
        # calculator caching to make the repeats free; batch only the images
        # whose geometry actually changed since their cached evaluation.
        targets = [
            image
            for image in targets
            if not image.calc.has_result_for(image)
        ]
        if not targets:
            self._endpoints_evaluated = True
            return
        results = self._band_evaluator.evaluate_band(targets)
        for image, (energy, forces) in zip(targets, results):
            image.calc.store(image, energy, forces)
        self._endpoints_evaluated = True

    def get_forces(self):
        self._prefetch_band()
        return super().get_forces()


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
    band_eval: str = DEFAULT_NEB_BAND_EVAL,
) -> tuple[Any, list[Atoms]]:
    """Build an ASE NEB band with one shared concrete calculator.

    ``n_images`` counts interior images.  A compatible ``initial_path`` is
    copied directly into the band; otherwise IDPP interpolation falls back to
    a linear path for the same short-band/pathological-geometry cases handled
    by the former channel-local implementations.  Calculator pools must be
    leased by :func:`run_neb` before this function is called so the same
    concrete calculator remains assigned for the band's entire lifetime.
    """
    if isinstance(calculator, CalculatorPool):
        raise CalculatorConfigError(
            "make_neb_band requires one concrete calculator; call run_neb() "
            "with the CalculatorPool so it can hold one lease for the full NEB"
        )
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

    band_evaluator: BandEvaluator | None = None
    if normalize_band_eval(band_eval) == "batched":
        band_evaluator = resolve_band_evaluator(calculator)
        if band_evaluator is None:
            _log.warning(
                "Calculator %s does not support batched band "
                "evaluation; falling back to per-image NEB evaluation.",
                type(calculator).__name__,
            )

    if band_evaluator is not None:
        for image in images:
            image.calc = BandImageCalculator(calculator)
    else:
        for image in images:
            image.calc = calculator

    neb_kwargs = {
        "k": float(spring_k),
        "climb": bool(climb),
        "method": "improvedtangent",
        "allow_shared_calculator": band_evaluator is None,
    }
    if band_evaluator is not None:
        neb = _BatchedBandNEB(
            images,
            band_evaluator=band_evaluator,
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
    optimizer_kwargs: Mapping[str, Any] | None = None,
    climb_optimizer: str | None = None,
    climb_optimizer_kwargs: Mapping[str, Any] | None = None,
    start_climbing: bool = False,
    persist_path: bool = False,
    capture_path: bool = False,
    initial_path: Sequence[Atoms] | None = None,
    initial_path_callback: Callable[[list[Atoms]], None] | None = None,
    failure_path_callback: Callable[[list[Atoms]], None] | None = None,
    band_factory=None,
    logfile_factory=None,
    band_eval: str = DEFAULT_NEB_BAND_EVAL,
) -> NEBRunResult:
    """Optimise one NEB band and return a calculator-detached result.

    The caller supplies its channel-specific non-convergence exception class;
    scientific transition-state validation remains in the caller after this
    mechanical optimisation step.  When ``climb`` is requested, the ordinary
    NEB is converged first and the same band is then converged again after
    enabling its climbing image with a fresh optimizer instance. ``max_steps``
    applies independently to each stage, while ``optimizer_steps`` reports
    their combined step count. ``start_climbing`` skips the ordinary stage for
    a caller-supplied CI restart band. If an
    optimizer, calculator, or convergence check raises after band construction,
    *failure_path_callback* receives a calculator-detached snapshot of the
    last-known full band before the exception is re-raised.  A pool supplies
    exactly one concrete calculator, and that lease is held for the complete
    NEB lifecycle.  Other pool calculators remain available for independent
    NEBs, never for other images in this band.
    """
    build_band = band_factory or make_neb_band
    select_logfile = logfile_factory or neb_optimizer_logfile
    band_eval_mode = normalize_band_eval(band_eval)
    if band_eval_mode == "images":
        _maybe_hint_batched_available(calculator)
    with acquire_calculator(calculator, purpose=purpose) as neb_calculator:
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
        if band_factory is None:
            band_kwargs["band_eval"] = band_eval_mode
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

            def _optimise_stage(
                *,
                stage: str,
                selected_optimizer: str,
                selected_optimizer_kwargs: Mapping[str, Any] | None,
                target_fmax: float,
            ) -> None:
                nonlocal optimizer_steps
                climbing_stage = stage.startswith("CI-NEB")
                optimizer_name = normalize_optimizer_name(
                    selected_optimizer,
                    allowed=NEB_OPTIMIZERS,
                    setting=(
                        "neb_climb_optimizer"
                        if climbing_stage
                        else "neb_optimizer"
                    ),
                )
                optimizer_cls = {
                    "bfgs": BFGS,
                    "fire": FIRE,
                    "mdmin": MDMin,
                }[optimizer_name]
                constructor_kwargs = normalize_optimizer_kwargs(
                    optimizer_name,
                    selected_optimizer_kwargs,
                    allowed=NEB_OPTIMIZERS,
                    setting=(
                        "neb_climb_optimizer_kwargs"
                        if climbing_stage
                        else "neb_optimizer_kwargs"
                    ),
                )
                if (
                    climbing_stage
                    and optimizer_name == "fire"
                    and constructor_kwargs.get("downhill_check", False)
                ):
                    constructor_kwargs["downhill_check"] = False
                    _log.warning(
                        "FIRE downhill_check is incompatible with CI-NEB "
                        "because the climbing image is intentionally driven "
                        "uphill; disabling it for the climbing stage."
                    )
                stage_optimizer = optimizer_cls(
                    neb,
                    logfile=select_logfile(verbose),
                    **constructor_kwargs,
                )
                stage_optimizer.run(
                    fmax=float(target_fmax),
                    steps=int(max_steps),
                )
                optimizer_steps += int(stage_optimizer.nsteps)
                if not stage_optimizer.converged():
                    raise not_converged_error(
                        f"{stage} did not converge: fmax={target_fmax} eV/Å not "
                        f"reached in {max_steps} steps."
                    )

            if not (climb and start_climbing):
                _optimise_stage(
                    stage="NEB pre-climb relaxation" if climb else "NEB",
                    selected_optimizer=optimizer,
                    selected_optimizer_kwargs=optimizer_kwargs,
                    target_fmax=float(fmax),
                )
            if climb:
                neb.climb = True
                _optimise_stage(
                    stage="CI-NEB",
                    selected_optimizer=climb_optimizer or optimizer,
                    selected_optimizer_kwargs=(
                        optimizer_kwargs
                        if climb_optimizer_kwargs is None
                        else climb_optimizer_kwargs
                    ),
                    target_fmax=float(fmax),
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
    "DEFAULT_NEB_BAND_EVAL",
    "NEB_BAND_EVALS",
    "NEBRunResult",
    "make_neb_band",
    "neb_optimizer_logfile",
    "normalize_band_eval",
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
