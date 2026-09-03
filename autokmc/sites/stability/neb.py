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
    NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    NEB_INTERMEDIATE_MAX_REFINEMENTS,
    NEB_INTERMEDIATE_MINIMUM_PROMINENCE,
    NEB_INTERMEDIATE_REFINEMENT_POLICY,
    NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER,
    NEB_METHOD as DEFAULT_NEB_METHOD,
    NEB_METHODS,
)
from autokmc.io.calculators import (
    CalculatorConfigError,
    CalculatorPool,
    acquire_calculator,
    primary_calculator,
)
from autokmc.io.atoms import copy_atoms_with_results
from autokmc.sites.stability.band_eval import (
    BandEvaluator,
    BandImageCalculator,
    resolve_band_evaluator,
)
from autokmc.utils.logging import get_logger
from autokmc.utils.optimizers import (
    DEFAULT_NEB_OPTIMIZER,
    DEFAULT_OPTIMIZER,
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
        raise ValueError(f"neb_band_eval must be a non-empty string, got {value!r}")
    name = value.strip().lower()
    if name not in NEB_BAND_EVALS:
        choices = ", ".join(sorted(NEB_BAND_EVALS))
        raise ValueError(f"neb_band_eval must be one of {choices}; got {value!r}")
    return name


def normalize_neb_method(value: str) -> str:
    """Return a canonical ASE NEB method or raise a useful error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"neb_method must be a non-empty string, got {value!r}")
    name = value.strip().lower()
    if name not in NEB_METHODS:
        choices = ", ".join(sorted(NEB_METHODS))
        raise ValueError(f"neb_method must be one of {choices}; got {value!r}")
    return name


@dataclass(frozen=True)
class NEBRunResult:
    """Live-calculator-free result of a converged NEB optimization."""

    atoms_ts: Atoms
    energy_ts: float
    transition_index: int
    n_interior: int
    optimizer_steps: int
    path_energies: list[float] | None = None
    path_images: list[Atoms] | None = None
    climb_performed: bool = False
    intermediate_refinement_performed: bool = False
    intermediate_refinement_count: int = 0
    intermediate_max_refinements: int = NEB_INTERMEDIATE_MAX_REFINEMENTS
    intermediate_stagnation_steps: int | None = None
    intermediate_trigger: str | None = None
    intermediate_source_stage: str | None = None
    intermediate_checkpoint_fmax: float | None = None
    intermediate_checkpoint_optimizer_steps: int | None = None
    intermediate_peak_index: int | None = None
    intermediate_left_index: int | None = None
    intermediate_right_index: int | None = None
    intermediate_stalled_steps: int = 0
    intermediate_profile_energies: list[float] | None = None
    refinement_initial_atoms: Atoms | None = None
    refinement_final_atoms: Atoms | None = None
    refinement_initial_energy: float | None = None
    refinement_final_energy: float | None = None
    refinement_max_endpoint_displacement: float | None = None
    refinement_target_image_spacing: float | None = None
    refinement_estimated_image_spacing: float | None = None
    refinement_image_count_limited_by: str | None = None


@dataclass(frozen=True)
class NEBImageSelection:
    """Resolved interior-image count and endpoint-spacing diagnostics."""

    n_images: int
    n_frames: int
    max_endpoint_displacement: float
    target_spacing: float | None
    estimated_linear_spacing: float
    limited_by: str


@dataclass(frozen=True)
class _NEBBandGap:
    """Largest MIC-aware corresponding-atom gap in an NEB band."""

    distance: float
    left_image: int
    atom_index: int


class _NEBBandSpacingViolation(RuntimeError):
    """Internal signal used to restart an optimizer from a valid band."""

    def __init__(self, gap: _NEBBandGap, limit: float) -> None:
        self.gap = gap
        self.limit = float(limit)
        super().__init__(
            "adjacent NEB images separated by "
            f"{gap.distance:.6f} Å at image pair "
            f"{gap.left_image}-{gap.left_image + 1}, atom {gap.atom_index}; "
            f"limit={self.limit:.6f} Å"
        )


class _NEBIntermediateRefinement(RuntimeError):
    """Internal signal carrying a valid band and selected peak bracket."""

    def __init__(
        self,
        *,
        images: Sequence[Atoms],
        energies: Sequence[float],
        peak_index: int,
        left_index: int,
        right_index: int,
        trigger: str,
        source_stage: str,
        checkpoint_fmax: float | None = None,
        checkpoint_optimizer_steps: int | None = None,
    ) -> None:
        self.energies = [float(value) for value in energies]
        self.images = [
            copy_atoms_with_results(image, energy=self.energies[index])
            for index, image in enumerate(images)
        ]
        self.peak_index = int(peak_index)
        self.left_index = int(left_index)
        self.right_index = int(right_index)
        self.trigger = trigger
        self.source_stage = source_stage
        self.checkpoint_fmax = checkpoint_fmax
        self.checkpoint_optimizer_steps = checkpoint_optimizer_steps
        self.optimizer_steps = 0
        super().__init__(
            f"{source_stage} found an intermediate minimum after {trigger}; "
            f"highest image {self.peak_index} is bracketed by states "
            f"{self.left_index} and {self.right_index}"
        )


def _highest_peak_minimum_bracket(
    energies: Sequence[float],
    *,
    minimum_prominence: float,
) -> tuple[int, int, int] | None:
    """Return the highest interior peak and nearest bracketing minima."""
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        return None
    prominence = float(minimum_prominence)
    if not np.isfinite(prominence) or prominence < 0.0:
        raise ValueError("minimum_prominence must be finite and non-negative")
    peak_index = 1 + int(np.argmax(values[1:-1]))
    minima = [0]
    minima.extend(
        index
        for index in range(1, len(values) - 1)
        if (
            values[index] <= values[index - 1] - prominence
            and values[index] <= values[index + 1] - prominence
        )
    )
    minima.append(len(values) - 1)
    left_index = max(index for index in minima if index < peak_index)
    right_index = min(index for index in minima if index > peak_index)
    if left_index == 0 and right_index == len(values) - 1:
        return None
    return peak_index, left_index, right_index


def _maximum_adjacent_image_displacement(
    images: Sequence[Atoms],
    *,
    frozen_indices: Sequence[int] | None,
) -> _NEBBandGap:
    """Return the largest unfrozen-atom displacement between band images."""
    if len(images) < 2:
        return _NEBBandGap(0.0, -1, -1)

    n_atoms = len(images[0])
    mobile = np.ones(n_atoms, dtype=bool)
    if frozen_indices:
        frozen = np.asarray(list(frozen_indices), dtype=int)
        if np.any((frozen < 0) | (frozen >= n_atoms)):
            raise ValueError("frozen atom index is outside the NEB image")
        mobile[frozen] = False
    mobile_indices = np.flatnonzero(mobile)
    if not len(mobile_indices):
        return _NEBBandGap(0.0, -1, -1)

    maximum = _NEBBandGap(0.0, 0, int(mobile_indices[0]))
    for left_index, (left, right) in enumerate(zip(images[:-1], images[1:])):
        delta = np.asarray(right.positions - left.positions, dtype=float)
        if np.asarray(left.pbc, dtype=bool).any():
            delta, _ = find_mic(
                delta,
                np.asarray(left.cell.array, dtype=float),
                pbc=np.asarray(left.pbc, dtype=bool),
            )
        distances = np.linalg.norm(delta[mobile_indices], axis=1)
        local_offset = int(np.argmax(distances))
        local_distance = float(distances[local_offset])
        if local_distance > maximum.distance:
            maximum = _NEBBandGap(
                local_distance,
                left_index,
                int(mobile_indices[local_offset]),
            )
    return maximum


def resolve_neb_image_count(
    atoms_initial: Atoms,
    atoms_final: Atoms,
    *,
    fixed_n_images: int,
    image_spacing: float | None,
    min_images: int,
    max_images: int,
) -> NEBImageSelection:
    """Select a MIC-aware NEB image count from endpoint atom displacement.

    ``n_images`` in ASE/AutoKMC means *interior* images, so a band has
    ``n_images + 2`` frames and ``n_images + 1`` adjacent intervals.  In
    dynamic mode the smallest count is chosen for which the largest
    corresponding-atom displacement in a linear MIC interpolation is no
    greater than ``image_spacing``, subject to the configured bounds.
    """
    fixed = int(fixed_n_images)
    lower = int(min_images)
    upper = int(max_images)
    if fixed < 1:
        raise ValueError("fixed_n_images must be >= 1")
    if lower < 1:
        raise ValueError("min_images must be >= 1")
    if upper < lower:
        raise ValueError("max_images must be >= min_images")
    if len(atoms_initial) != len(atoms_final):
        raise ValueError("NEB endpoints must have the same atom count")
    if not np.array_equal(atoms_initial.numbers, atoms_final.numbers):
        raise ValueError("NEB endpoints must have the same atom ordering")
    if not np.array_equal(atoms_initial.pbc, atoms_final.pbc):
        raise ValueError("NEB endpoints must have the same PBC")
    if not np.allclose(
        atoms_initial.cell.array,
        atoms_final.cell.array,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("NEB endpoints must have the same cell")

    delta = np.asarray(
        atoms_final.positions - atoms_initial.positions,
        dtype=float,
    )
    if not np.isfinite(delta).all():
        raise ValueError("NEB endpoint displacement contains non-finite values")
    pbc = np.asarray(atoms_initial.pbc, dtype=bool)
    if pbc.any():
        mic_delta, _ = find_mic(
            delta,
            np.asarray(atoms_initial.cell.array, dtype=float),
            pbc=pbc,
        )
    else:
        mic_delta = delta
    displacement = np.linalg.norm(np.asarray(mic_delta, dtype=float), axis=1)
    maximum = float(displacement.max()) if len(displacement) else 0.0

    if image_spacing is None:
        resolved = fixed
        target = None
        limited_by = "fixed"
    else:
        target = float(image_spacing)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("image_spacing must be finite and positive")
        required_intervals = max(2, int(np.ceil(maximum / target)))
        required_images = required_intervals - 1
        resolved = min(upper, max(lower, required_images))
        if resolved < required_images:
            limited_by = "maximum"
        elif resolved > required_images:
            limited_by = "minimum"
        else:
            limited_by = "distance"

    return NEBImageSelection(
        n_images=int(resolved),
        n_frames=int(resolved) + 2,
        max_endpoint_displacement=maximum,
        target_spacing=target,
        estimated_linear_spacing=(maximum / float(int(resolved) + 1)),
        limited_by=limited_by,
    )


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
        if any(not isinstance(image.calc, BandImageCalculator) for image in targets):
            # Foreign per-image calculators are attached (e.g. ASE's IDPP
            # interpolation temporarily swaps them in) — evaluate normally.
            return
        # ASE optimizers issue several ``get_forces`` per step and lean on
        # calculator caching to make the repeats free; batch only the images
        # whose geometry actually changed since their cached evaluation.
        targets = [image for image in targets if not image.calc.has_result_for(image)]
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
    if any(not isinstance(image, Atoms) or len(image) != n_source for image in source_images):
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
        if not np.array_equal(np.asarray(image.pbc, dtype=bool), source_pbc) or not np.allclose(
            np.asarray(image.cell.array, dtype=float),
            source_cell,
            rtol=0.0,
            atol=1.0e-8,
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
    target_common_indices = np.concatenate((np.arange(n_slab, dtype=int), target_reacting_indices))
    source_common_indices = np.arange(n_source, dtype=int)
    if (
        not np.array_equal(target_numbers_initial, target_numbers_final)
        or not np.array_equal(
            source_numbers,
            target_numbers_initial[
                np.concatenate((np.arange(n_slab, dtype=int), target_reacting_indices))
            ],
        )
        or any(
            not np.array_equal(np.asarray(image.numbers, dtype=int), source_numbers)
            for image in source_images[1:]
        )
    ):
        return None

    all_positions = [np.asarray(image.positions, dtype=float) for image in source_images] + [
        np.asarray(atoms_initial.positions, dtype=float),
        np.asarray(atoms_final.positions, dtype=float),
    ]
    if not all(np.isfinite(positions).all() for positions in all_positions):
        return None

    def _minimum_image(vectors: np.ndarray) -> np.ndarray:
        if not np.any(target_pbc):
            return np.asarray(vectors, dtype=float).copy()
        mic_vectors, _ = find_mic(vectors, cell=target_cell, pbc=target_pbc)
        return np.asarray(mic_vectors, dtype=float)

    source_initial_positions = all_positions[0]
    source_delta = _minimum_image(all_positions[len(source_images) - 1] - source_initial_positions)
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
        target_positions[target_common_indices] += source_residual[source_common_indices]

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
    neb_method: str = DEFAULT_NEB_METHOD,
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
                and np.isfinite(np.asarray(image.positions, dtype=float)).all()
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
        "method": normalize_neb_method(neb_method),
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
        linear_positions = [np.asarray(image.positions, dtype=float).copy() for image in images]
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
                np.isfinite(np.asarray(image.positions, dtype=float)).all() for image in images
            ):
                raise FloatingPointError("IDPP interpolation returned non-finite positions")
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
    neb_method: str = DEFAULT_NEB_METHOD,
    start_climbing: bool = False,
    persist_path: bool = False,
    capture_path: bool = False,
    initial_path: Sequence[Atoms] | None = None,
    initial_path_callback: Callable[[list[Atoms]], None] | None = None,
    failure_path_callback: Callable[[list[Atoms]], None] | None = None,
    band_factory=None,
    logfile_factory=None,
    band_eval: str = DEFAULT_NEB_BAND_EVAL,
    image_spacing: float | None = None,
    geometry_guard_multiplier: float = (NEB_MAX_ADJACENT_IMAGE_SPACING_MULTIPLIER),
    intermediate_stagnation_steps: int | None = None,
    intermediate_max_refinements: int = NEB_INTERMEDIATE_MAX_REFINEMENTS,
    intermediate_energy_tolerance: float = NEB_INTERMEDIATE_ENERGY_TOLERANCE,
    intermediate_minimum_prominence: float = NEB_INTERMEDIATE_MINIMUM_PROMINENCE,
    intermediate_optimizer: str = DEFAULT_OPTIMIZER,
    intermediate_optimizer_kwargs: Mapping[str, Any] | None = None,
    intermediate_relaxer: Callable[[Atoms, str], tuple[Atoms, float]] | None = None,
    intermediate_min_images: int | None = None,
    intermediate_max_images: int | None = None,
    intermediate_refinement_callback: (
        Callable[[Atoms, Atoms, dict[str, Any]], None] | None
    ) = None,
) -> NEBRunResult:
    """Optimize one NEB band and return a live-calculator-free result.

    The caller supplies its channel-specific non-convergence exception class;
    scientific transition-state validation remains in the caller after this
    mechanical optimisation step.  When ``climb`` is requested, the ordinary
    NEB is converged first and the same band is then converged again after
    enabling its climbing image with a fresh optimizer instance. ``max_steps``
    applies independently to each stage, while ``optimizer_steps`` reports
    their combined step count. ``start_climbing`` skips the ordinary stage for
    a caller-supplied CI restart band.

    Ordinary FIRE may begin with ``downhill_check=True`` as a preconditioner.
    If five rollback halvings collapse its timestep, AutoKMC disables the
    check, restores the stage's initial ``dt``, and continues without creating
    another optimizer stage. CI-FIRE disables downhill checking immediately.

    If an optimizer, calculator, or convergence check raises after band
    construction, *failure_path_callback* receives a snapshot of the last-known
    full band before the exception is re-raised. Live calculators are removed,
    while valid cached energy and force results are retained as safe
    single-point data. A pool supplies exactly one concrete calculator, and
    that lease is held for the complete NEB lifecycle. Other pool calculators
    remain available for independent NEBs, never for other images in this band.

    When ``image_spacing`` is configured, no unfrozen atom may move more than
    ``geometry_guard_multiplier`` times that distance between adjacent images
    (using the minimum-image convention). The default multiplier is three.
    The lowest-force geometrically valid band in the current stage is
    checkpointed. If a later step crosses the limit, that checkpoint is
    restored. With intermediate refinement enabled, its electronic-energy
    profile is inspected immediately, before checking the remaining step
    budget. Bracketing minima trigger the same single-segment refinement as
    energy stagnation, including on rollback during CI-NEB. Without a usable
    bracket (or after the refinement limit is reached), a fresh optimizer
    resumes the restored band. FIRE halves ``dt`` and ``dtmax``; other
    supported optimizers reduce their available displacement control. These
    same-band restarts share the original stage step budget.

    During an ordinary stage, ``intermediate_stagnation_steps`` monitors the
    lowest interior-image electronic energy. If it does not decrease by
    ``intermediate_energy_tolerance`` for that many optimizer steps, the band
    is inspected for local minima. When the highest-energy image is bracketed
    by at least one interior minimum, only those two bracketing states are
    relaxed and a fresh standard NEB is run between them. This intentionally
    faster approximation does not refine the other portions of the original
    path. The returned transition energy remains on the original calculator
    energy reference. Up to ``intermediate_max_refinements`` replacements are
    allowed per call, and each replacement band is eligible for the same
    stagnation and rollback inspection until that limit is reached. A
    replacement band always begins with ordinary NEB, even if a CI-only restart
    triggered it. ``intermediate_refinement_callback`` captures each optimized
    pair and its trigger/checkpoint provenance before constructing the next band.
    Independently of the stagnation clock, every successfully converged ordinary
    or CI stage receives one final profile inspection. A usable bracket restarts
    the shortened segment before the workflow advances or returns its result.

    """
    build_band = band_factory or make_neb_band
    select_logfile = logfile_factory or neb_optimizer_logfile
    active_endpoint_results = (
        copy_atoms_with_results(atoms_initial),
        copy_atoms_with_results(atoms_final),
    )
    band_eval_mode = normalize_band_eval(band_eval)
    method_name = normalize_neb_method(neb_method)
    spacing_limit = None
    if image_spacing is not None:
        resolved_spacing = float(image_spacing)
        if not np.isfinite(resolved_spacing) or resolved_spacing <= 0.0:
            raise ValueError("image_spacing must be finite and positive")
        resolved_guard_multiplier = float(geometry_guard_multiplier)
        if not np.isfinite(resolved_guard_multiplier) or resolved_guard_multiplier <= 0.0:
            raise ValueError("geometry_guard_multiplier must be finite and positive")
        spacing_limit = resolved_guard_multiplier * resolved_spacing
    resolved_stagnation_steps = None
    if intermediate_stagnation_steps is not None:
        if type(intermediate_stagnation_steps) is not int:
            raise ValueError("intermediate_stagnation_steps must be an integer or None")
        if intermediate_stagnation_steps < 1:
            raise ValueError("intermediate_stagnation_steps must be >= 1")
        resolved_stagnation_steps = int(intermediate_stagnation_steps)
    if type(intermediate_max_refinements) is not int:
        raise ValueError("intermediate_max_refinements must be an integer")
    if intermediate_max_refinements < 1:
        raise ValueError("intermediate_max_refinements must be >= 1")
    resolved_max_refinements = int(intermediate_max_refinements)
    resolved_energy_tolerance = float(intermediate_energy_tolerance)
    if not np.isfinite(resolved_energy_tolerance) or resolved_energy_tolerance < 0.0:
        raise ValueError("intermediate_energy_tolerance must be finite and non-negative")
    resolved_minimum_prominence = float(intermediate_minimum_prominence)
    if not np.isfinite(resolved_minimum_prominence) or resolved_minimum_prominence < 0.0:
        raise ValueError("intermediate_minimum_prominence must be finite and non-negative")
    resolved_intermediate_min_images = (
        1 if intermediate_min_images is None else intermediate_min_images
    )
    resolved_intermediate_max_images = (
        max(1, int(n_images))
        if intermediate_max_images is None
        else intermediate_max_images
    )
    if type(resolved_intermediate_min_images) is not int:
        raise ValueError("intermediate_min_images must be an integer or None")
    if type(resolved_intermediate_max_images) is not int:
        raise ValueError("intermediate_max_images must be an integer or None")
    if resolved_intermediate_min_images < 1:
        raise ValueError("intermediate_min_images must be >= 1")
    if resolved_intermediate_max_images < resolved_intermediate_min_images:
        raise ValueError(
            "intermediate_max_images must be >= intermediate_min_images"
        )
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
        # Preserve compatibility with legacy custom band factories when the
        # default is requested. A non-default method must be understood by the
        # factory or construction fails explicitly rather than silently using
        # different NEB physics.
        if band_factory is None or method_name != DEFAULT_NEB_METHOD:
            band_kwargs["neb_method"] = method_name
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
                f"NEB band construction failed: {type(exc).__name__}: {exc}"
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
            intermediate_refinement_performed = False
            intermediate_refinement_count = 0
            intermediate_trigger = None
            intermediate_source_stage = None
            intermediate_checkpoint_fmax = None
            intermediate_checkpoint_optimizer_steps = None
            intermediate_peak_index = None
            intermediate_left_index = None
            intermediate_right_index = None
            intermediate_stalled_steps = 0
            intermediate_profile_energies = None
            refinement_initial_atoms = None
            refinement_final_atoms = None
            refinement_initial_energy = None
            refinement_final_energy = None
            refinement_max_endpoint_displacement = None
            refinement_target_image_spacing = None
            refinement_estimated_image_spacing = None
            refinement_image_count_limited_by = None

            def _current_neb_fmax() -> float:
                forces = np.asarray(neb.get_forces(), dtype=float)
                force_norms = np.linalg.norm(
                    forces.reshape((-1, 3)),
                    axis=1,
                )
                return float(force_norms.max()) if len(force_norms) else 0.0

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
                    setting=("neb_climb_optimizer" if climbing_stage else "neb_optimizer"),
                )
                optimizer_cls = {
                    "bfgs": BFGS,
                    "fire": FIRE,
                    "mdmin": MDMin,
                }[optimizer_name]
                stage_constructor_kwargs = normalize_optimizer_kwargs(
                    optimizer_name,
                    selected_optimizer_kwargs,
                    allowed=NEB_OPTIMIZERS,
                    setting=(
                        "neb_climb_optimizer_kwargs" if climbing_stage else "neb_optimizer_kwargs"
                    ),
                )
                if (
                    climbing_stage
                    and optimizer_name == "fire"
                    and stage_constructor_kwargs.get("downhill_check", False)
                ):
                    stage_constructor_kwargs["downhill_check"] = False
                    _log.warning(
                        "FIRE downhill_check is incompatible with CI-NEB "
                        "because the climbing image is intentionally driven "
                        "uphill; disabling it for %s.",
                        stage,
                    )

                best_fmax = float("inf")
                best_positions: list[np.ndarray] | None = None
                best_optimizer_steps: int | None = None
                best_interior_energy = float("inf")
                steps_without_lower_interior_energy = 0
                remaining_steps = int(max_steps)

                while remaining_steps > 0:
                    constructor_kwargs = dict(stage_constructor_kwargs)
                    fire_recovery_state = None
                    if (
                        not climbing_stage
                        and optimizer_name == "fire"
                        and constructor_kwargs.get("downhill_check", False)
                    ):
                        user_reset_callback = constructor_kwargs.get("position_reset_callback")
                        fire_recovery_state = {
                            "optimizer": None,
                            "initial_dt": None,
                            "switched": False,
                        }

                        def recover_stalled_fire(
                            optimizable,
                            last_positions,
                            energy,
                            last_energy,
                        ) -> None:
                            if user_reset_callback is not None:
                                user_reset_callback(
                                    optimizable,
                                    last_positions,
                                    energy,
                                    last_energy,
                                )
                            state = fire_recovery_state
                            stage_fire = state["optimizer"]
                            initial_dt = state["initial_dt"]
                            if state["switched"] or stage_fire is None or initial_dt is None:
                                return
                            # FIRE applies fdec after this callback. Trigger
                            # after five rollback halvings and compensate for
                            # the final pending halving so the next trial uses
                            # the initial dt.
                            threshold = initial_dt * stage_fire.fdec**4
                            if stage_fire.dt > threshold:
                                return
                            state["switched"] = True
                            stage_fire.downhill_check = False
                            stage_fire.dt = initial_dt / stage_fire.fdec
                            _log.warning(
                                "FIRE downhill_check stalled %s after repeated "
                                "energy rollbacks; disabling it and restoring "
                                "dt=%g.",
                                stage,
                                initial_dt,
                            )

                        constructor_kwargs["position_reset_callback"] = recover_stalled_fire

                    stage_optimizer = optimizer_cls(
                        neb,
                        logfile=select_logfile(verbose),
                        **constructor_kwargs,
                    )
                    if fire_recovery_state is not None:
                        fire_recovery_state["optimizer"] = stage_optimizer
                        fire_recovery_state["initial_dt"] = float(stage_optimizer.dt)

                    initial_controls = {
                        name: float(getattr(stage_optimizer, name))
                        for name in ("dt", "dtmax", "maxstep")
                        if hasattr(stage_optimizer, name)
                    }

                    can_refine_intermediate = bool(
                        resolved_stagnation_steps is not None
                        and intermediate_refinement_count < resolved_max_refinements
                    )
                    monitor_intermediate = can_refine_intermediate and not climbing_stage
                    if spacing_limit is not None or monitor_intermediate:

                        def monitor_band() -> None:
                            nonlocal best_fmax, best_positions
                            nonlocal best_optimizer_steps
                            nonlocal best_interior_energy
                            nonlocal steps_without_lower_interior_energy
                            if spacing_limit is not None:
                                gap = _maximum_adjacent_image_displacement(
                                    images,
                                    frozen_indices=frozen_indices,
                                )
                                if gap.distance > spacing_limit * (1.0 + 1.0e-12):
                                    raise _NEBBandSpacingViolation(
                                        gap,
                                        spacing_limit,
                                    )
                                current_fmax = _current_neb_fmax()
                                if (
                                    np.isfinite(current_fmax)
                                    and current_fmax < best_fmax
                                ):
                                    best_fmax = current_fmax
                                    best_optimizer_steps = (
                                        optimizer_steps + int(stage_optimizer.nsteps)
                                    )
                                    best_positions = [
                                        np.asarray(image.positions, dtype=float).copy()
                                        for image in images
                                    ]
                            if not monitor_intermediate:
                                return
                            energies = [
                                float(image.get_potential_energy())
                                for image in images
                            ]
                            interior = energies[1:-1]
                            if not interior or not np.isfinite(interior).all():
                                return
                            current_minimum = float(min(interior))
                            if (
                                current_minimum
                                < best_interior_energy - resolved_energy_tolerance
                            ):
                                best_interior_energy = current_minimum
                                steps_without_lower_interior_energy = 0
                                return
                            steps_without_lower_interior_energy += 1
                            assert resolved_stagnation_steps is not None
                            if (
                                steps_without_lower_interior_energy
                                < resolved_stagnation_steps
                            ):
                                return
                            steps_without_lower_interior_energy = 0
                            bracket = _highest_peak_minimum_bracket(
                                energies,
                                minimum_prominence=resolved_minimum_prominence,
                            )
                            if bracket is None:
                                return
                            peak_index, left_index, right_index = bracket
                            raise _NEBIntermediateRefinement(
                                images=images,
                                energies=energies,
                                peak_index=peak_index,
                                left_index=left_index,
                                right_index=right_index,
                                trigger="energy_stagnation",
                                source_stage=stage,
                            )
                        stage_optimizer.attach(
                            monitor_band,
                            interval=1,
                        )

                    try:
                        stage_optimizer.run(
                            fmax=float(target_fmax),
                            steps=remaining_steps,
                        )
                    except _NEBIntermediateRefinement as exc:
                        steps_used = int(stage_optimizer.nsteps)
                        optimizer_steps += steps_used
                        remaining_steps -= steps_used
                        exc.optimizer_steps = optimizer_steps
                        raise
                    except _NEBBandSpacingViolation as exc:
                        steps_used = int(stage_optimizer.nsteps)
                        optimizer_steps += steps_used
                        remaining_steps -= steps_used
                        if best_positions is None:
                            raise not_converged_error(
                                f"{stage} has no geometrically valid band to restore: {exc}"
                            ) from exc
                        for image, positions in zip(images, best_positions):
                            image.set_positions(
                                positions,
                                apply_constraint=False,
                            )
                        if can_refine_intermediate:
                            # Inspect only the restored, lowest-force valid
                            # band, never the rejected over-stretched frame.
                            # This trigger is independent of the stagnation
                            # clock and is checked even on the last stage step.
                            energies = [
                                float(image.get_potential_energy())
                                for image in images
                            ]
                            bracket = _highest_peak_minimum_bracket(
                                energies,
                                minimum_prominence=resolved_minimum_prominence,
                            )
                            if bracket is not None:
                                peak_index, left_index, right_index = bracket
                                _log.warning(
                                    "%s exceeded the adjacent-image limit: %s. "
                                    "Restored the lowest-force valid band "
                                    "(fmax=%.6f eV/Å); its energy profile contains "
                                    "minima bracketing the highest peak.",
                                    stage,
                                    exc,
                                    best_fmax,
                                )
                                refinement = _NEBIntermediateRefinement(
                                    images=images,
                                    energies=energies,
                                    peak_index=peak_index,
                                    left_index=left_index,
                                    right_index=right_index,
                                    trigger="geometry_rollback",
                                    source_stage=stage,
                                    checkpoint_fmax=best_fmax,
                                    checkpoint_optimizer_steps=best_optimizer_steps,
                                )
                                # The failed attempt was counted above; a
                                # signal raised in this except block bypasses
                                # the sibling refinement-exception handler.
                                refinement.optimizer_steps = optimizer_steps
                                raise refinement from exc
                        if remaining_steps <= 0:
                            raise not_converged_error(
                                f"{stage} exceeded its adjacent-image spacing "
                                f"limit and exhausted {max_steps} steps; the "
                                "lowest-force valid band was restored."
                            ) from exc

                        reduced_controls: dict[str, float] = {}
                        for control in ("dt", "dtmax"):
                            if control in initial_controls:
                                reduced_controls[control] = 0.5 * initial_controls[control]
                        if not reduced_controls and "maxstep" in initial_controls:
                            reduced_controls["maxstep"] = 0.5 * initial_controls["maxstep"]
                        stage_constructor_kwargs.update(reduced_controls)
                        controls = ", ".join(
                            f"{name}={value:g}" for name, value in reduced_controls.items()
                        )
                        _log.warning(
                            "%s exceeded the adjacent-image limit: %s. "
                            "Restored the lowest-force valid band "
                            "(fmax=%.6f eV/Å) and restarting %s with %s; "
                            "%d steps remain.",
                            stage,
                            exc,
                            best_fmax,
                            optimizer_name.upper(),
                            controls,
                            remaining_steps,
                        )
                        continue

                    steps_used = int(stage_optimizer.nsteps)
                    optimizer_steps += steps_used
                    remaining_steps -= steps_used
                    if not stage_optimizer.converged():
                        raise not_converged_error(
                            f"{stage} did not converge: fmax={target_fmax} "
                            f"eV/Å not reached in {max_steps} steps."
                        )
                    if can_refine_intermediate:
                        # A band can converge before the stagnation clock fires,
                        # and CI can reshape a profile that passed the ordinary
                        # stage. Inspect every converged stage once before either
                        # advancing to CI or returning the final result.
                        energies = [
                            float(image.get_potential_energy())
                            for image in images
                        ]
                        bracket = _highest_peak_minimum_bracket(
                            energies,
                            minimum_prominence=resolved_minimum_prominence,
                        )
                        if bracket is not None:
                            peak_index, left_index, right_index = bracket
                            refinement = _NEBIntermediateRefinement(
                                images=images,
                                energies=energies,
                                peak_index=peak_index,
                                left_index=left_index,
                                right_index=right_index,
                                trigger="converged_profile",
                                source_stage=stage,
                            )
                            refinement.optimizer_steps = optimizer_steps
                            raise refinement
                    return

            ordinary_stage = "NEB pre-climb relaxation" if climb else "NEB"
            run_climbing_restart = bool(climb and start_climbing)
            while True:
                climb_performed = False
                try:
                    if not run_climbing_restart:
                        _optimise_stage(
                            stage=ordinary_stage,
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
                        climb_performed = True
                    break
                except _NEBIntermediateRefinement as refinement:
                    intermediate_refinement_performed = True
                    intermediate_refinement_count += 1
                    intermediate_trigger = refinement.trigger
                    intermediate_source_stage = refinement.source_stage
                    intermediate_checkpoint_fmax = refinement.checkpoint_fmax
                    intermediate_checkpoint_optimizer_steps = (
                        refinement.checkpoint_optimizer_steps
                    )
                    intermediate_peak_index = refinement.peak_index
                    intermediate_left_index = refinement.left_index
                    intermediate_right_index = refinement.right_index
                    intermediate_stalled_steps = refinement.optimizer_steps
                    intermediate_profile_energies = list(refinement.energies)
                    source_images = refinement.images
                    source_energies = refinement.energies

                    def relax_refinement_state(
                        index: int,
                        label: str,
                    ) -> tuple[Atoms, float]:
                        candidate = source_images[index].copy()
                        candidate.calc = None
                        if index in {0, len(source_images) - 1}:
                            return (
                                copy_atoms_with_results(
                                    source_images[index],
                                    energy=float(source_energies[index]),
                                ),
                                float(source_energies[index]),
                            )
                        if intermediate_relaxer is not None:
                            optimized, energy = intermediate_relaxer(candidate, label)
                        else:
                            from autokmc.structure import optimise_structure

                            optimized = optimise_structure(
                                candidate,
                                calculator=neb_calculator,
                                fmax=float(fmax),
                                steps=int(max_steps),
                                optimizer=intermediate_optimizer,
                                optimizer_kwargs=intermediate_optimizer_kwargs,
                                verbose=verbose,
                            )
                            energy = float(optimized.get_potential_energy())
                        if not isinstance(optimized, Atoms):
                            raise TypeError(
                                "intermediate relaxation must return ase.Atoms"
                            )
                        resolved_energy = float(energy)
                        if not np.isfinite(resolved_energy):
                            raise ValueError(
                                "intermediate relaxation returned a non-finite energy"
                            )
                        detached = copy_atoms_with_results(
                            optimized,
                            energy=resolved_energy,
                        )
                        return detached, resolved_energy

                    refinement_initial_atoms, refinement_initial_energy = (
                        relax_refinement_state(
                            refinement.left_index,
                            "highest-peak segment initial state",
                        )
                    )
                    refinement_final_atoms, refinement_final_energy = (
                        relax_refinement_state(
                            refinement.right_index,
                            "highest-peak segment final state",
                        )
                    )
                    segment_selection = resolve_neb_image_count(
                        refinement_initial_atoms,
                        refinement_final_atoms,
                        fixed_n_images=max(1, int(n_images)),
                        image_spacing=image_spacing,
                        min_images=resolved_intermediate_min_images,
                        max_images=resolved_intermediate_max_images,
                    )
                    if segment_selection.max_endpoint_displacement <= 1.0e-6:
                        raise not_converged_error(
                            "optimized intermediate states collapsed onto the same geometry"
                        )
                    refinement_max_endpoint_displacement = (
                        segment_selection.max_endpoint_displacement
                    )
                    refinement_target_image_spacing = segment_selection.target_spacing
                    refinement_estimated_image_spacing = (
                        segment_selection.estimated_linear_spacing
                    )
                    refinement_image_count_limited_by = segment_selection.limited_by
                    refinement_metadata = {
                        "performed": True,
                        "policy": NEB_INTERMEDIATE_REFINEMENT_POLICY,
                        "refinement_index": intermediate_refinement_count,
                        "max_refinements": resolved_max_refinements,
                        "trigger": refinement.trigger,
                        "source_stage": refinement.source_stage,
                        "checkpoint_fmax_ev_per_ang": refinement.checkpoint_fmax,
                        "checkpoint_optimizer_steps": refinement.checkpoint_optimizer_steps,
                        "stagnation_steps": resolved_stagnation_steps,
                        "optimizer_steps_at_detection": refinement.optimizer_steps,
                        "peak_image_index": refinement.peak_index,
                        "left_state_image_index": refinement.left_index,
                        "right_state_image_index": refinement.right_index,
                        "stalled_profile_energies_ev": list(refinement.energies),
                        "refinement_initial_energy_ev": refinement_initial_energy,
                        "refinement_final_energy_ev": refinement_final_energy,
                        "replacement_interior_images": segment_selection.n_images,
                        "replacement_max_endpoint_displacement_ang": (
                            segment_selection.max_endpoint_displacement
                        ),
                        "replacement_target_image_spacing_ang": (
                            segment_selection.target_spacing
                        ),
                        "replacement_estimated_image_spacing_ang": (
                            segment_selection.estimated_linear_spacing
                        ),
                        "replacement_image_count_limited_by": (
                            segment_selection.limited_by
                        ),
                        "other_segments_refined": False,
                    }
                    if intermediate_refinement_callback is not None:
                        intermediate_refinement_callback(
                            refinement_initial_atoms.copy(),
                            refinement_final_atoms.copy(),
                            refinement_metadata,
                        )
                    if refinement.trigger == "geometry_rollback":
                        trigger_description = (
                            f"{refinement.source_stage} restored its lowest-force "
                            "valid band after a geometry rollback."
                        )
                    elif refinement.trigger == "converged_profile":
                        trigger_description = (
                            f"{refinement.source_stage} converged, but its final "
                            "energy profile still contains an intermediate minimum."
                        )
                    else:
                        trigger_description = (
                            f"{refinement.source_stage} found no lower interior-image "
                            f"energy for {resolved_stagnation_steps} steps."
                        )
                    _log.warning(
                        "%s Refining only the highest-energy segment %d-%d around "
                        "image %d; other portions of the original path are not "
                        "reoptimized.",
                        trigger_description,
                        refinement.left_index,
                        refinement.right_index,
                        refinement.peak_index,
                    )
                    for image in images:
                        image.calc = None
                    segment_band_kwargs = dict(band_kwargs)
                    segment_band_kwargs.pop("initial_path", None)
                    segment_band_kwargs["n_images"] = segment_selection.n_images
                    try:
                        active_endpoint_results = (
                            copy_atoms_with_results(refinement_initial_atoms),
                            copy_atoms_with_results(refinement_final_atoms),
                        )
                        neb, images = build_band(
                            refinement_initial_atoms,
                            refinement_final_atoms,
                            **segment_band_kwargs,
                        )
                    except CalculatorConfigError:
                        raise
                    except Exception as exc:
                        if isinstance(exc, not_converged_error):
                            raise
                        raise not_converged_error(
                            "highest-energy segment NEB construction failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if initial_path_callback is not None:
                        segment_initial_path = []
                        for image in images:
                            snapshot = image.copy()
                            snapshot.calc = None
                            segment_initial_path.append(snapshot)
                        initial_path_callback(segment_initial_path)
                    # A replacement band always uses the standard ordinary
                    # then optional CI workflow, even after a CI-only restart.
                    run_climbing_restart = False

            final_result_snapshots = [
                copy_atoms_with_results(image) for image in images
            ]
            final_forces = [
                (
                    snapshot.calc.results.get("forces")
                    if snapshot.calc is not None
                    else None
                )
                for snapshot in final_result_snapshots
            ]
            for index, endpoint in (
                (0, active_endpoint_results[0]),
                (len(images) - 1, active_endpoint_results[1]),
            ):
                if final_forces[index] is None and endpoint.calc is not None:
                    final_forces[index] = endpoint.calc.results.get("forces")
            energies = [float(image.get_potential_energy()) for image in images]
            interior = energies[1:-1]
            if not interior:
                raise not_converged_error(
                    "NEB band has no interior images (n_images=0); cannot identify a TS."
                )

            transition_index = 1 + int(np.argmax(interior))
            atoms_ts = copy_atoms_with_results(
                images[transition_index],
                energy=energies[transition_index],
                forces=final_forces[transition_index],
            )

            retain_path = bool(persist_path or capture_path)
            path_energies = list(energies) if retain_path else None
            path_images = None
            if retain_path:
                path_images = [
                    copy_atoms_with_results(
                        image,
                        energy=energies[index],
                        forces=final_forces[index],
                    )
                    for index, image in enumerate(images)
                ]

            result = NEBRunResult(
                atoms_ts=atoms_ts,
                energy_ts=float(energies[transition_index]),
                transition_index=transition_index,
                n_interior=len(interior),
                optimizer_steps=optimizer_steps,
                climb_performed=climb_performed,
                intermediate_refinement_performed=(
                    intermediate_refinement_performed
                ),
                intermediate_refinement_count=intermediate_refinement_count,
                intermediate_max_refinements=resolved_max_refinements,
                intermediate_stagnation_steps=resolved_stagnation_steps,
                intermediate_trigger=intermediate_trigger,
                intermediate_source_stage=intermediate_source_stage,
                intermediate_checkpoint_fmax=intermediate_checkpoint_fmax,
                intermediate_checkpoint_optimizer_steps=(
                    intermediate_checkpoint_optimizer_steps
                ),
                intermediate_peak_index=intermediate_peak_index,
                intermediate_left_index=intermediate_left_index,
                intermediate_right_index=intermediate_right_index,
                intermediate_stalled_steps=intermediate_stalled_steps,
                intermediate_profile_energies=intermediate_profile_energies,
                refinement_initial_atoms=refinement_initial_atoms,
                refinement_final_atoms=refinement_final_atoms,
                refinement_initial_energy=refinement_initial_energy,
                refinement_final_energy=refinement_final_energy,
                refinement_max_endpoint_displacement=(
                    refinement_max_endpoint_displacement
                ),
                refinement_target_image_spacing=refinement_target_image_spacing,
                refinement_estimated_image_spacing=(
                    refinement_estimated_image_spacing
                ),
                refinement_image_count_limited_by=(
                    refinement_image_count_limited_by
                ),
                path_energies=path_energies,
                path_images=path_images,
            )
        except Exception as exc:
            if failure_path_callback is not None:
                failed_path = [
                    copy_atoms_with_results(image) for image in images
                ]
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
                f"NEB optimization failed: {type(exc).__name__}: {exc}"
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
    "DEFAULT_NEB_METHOD",
    "NEB_BAND_EVALS",
    "NEB_METHODS",
    "NEBRunResult",
    "NEBImageSelection",
    "make_neb_band",
    "neb_optimizer_logfile",
    "normalize_band_eval",
    "normalize_neb_method",
    "project_neb_path",
    "resolve_neb_image_count",
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
