"""Whole-band batched evaluation for NEB images.

A band evaluator computes raw model energies and forces for every image of a
NEB band in one call, so a GPU MLIP sees a single stacked forward pass per
optimizer step instead of ~n_images sequential single-structure calls.  The
NEB physics is unchanged: results are cached per image and ASE's own NEB
tangent/spring/climbing math consumes them exactly as it would consume a
per-image calculator's output.  Constraints (``FixAtoms``) are likewise
untouched — evaluators return *raw* model forces, and ASE applies constraint
projections when each image's ``get_forces()`` is called.

Two evaluator sources are supported:

* Any calculator exposing a callable ``evaluate_band(images)`` returning one
  ``(energy, forces)`` pair per image (the explicit opt-in protocol).
* FAIR-Chem's ``FAIRChemCalculator`` (the production UMA path), batched
  through its underlying predict unit.

``resolve_band_evaluator`` returns ``None`` for anything else, and callers
fall back to the standard per-image evaluation path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from ase import Atoms

from autokmc.utils.logging import get_logger

_log = get_logger(__name__)

#: One image's raw model output: potential energy (eV), forces (n_atoms, 3).
BandImageResult = tuple[float, np.ndarray]


class BandEvaluationError(RuntimeError):
    """A batched band evaluation failed or produced inconsistent output."""


@runtime_checkable
class BandEvaluator(Protocol):
    """Evaluate raw energies/forces for every image of a band in one call."""

    def evaluate_band(
        self, images: Sequence[Atoms]
    ) -> list[BandImageResult]:  # pragma: no cover - protocol
        ...


def _validated_results(
    images: Sequence[Atoms],
    results: Sequence[Any],
    *,
    source: str,
) -> list[BandImageResult]:
    """Normalise and shape-check evaluator output against the band."""
    try:
        pairs = list(results)
    except TypeError as exc:
        raise BandEvaluationError(
            f"{source} returned a non-iterable band result"
        ) from exc
    if len(pairs) != len(images):
        raise BandEvaluationError(
            f"{source} returned {len(pairs)} results for "
            f"{len(images)} images"
        )
    validated: list[BandImageResult] = []
    for image, pair in zip(images, pairs):
        try:
            energy_raw, forces_raw = pair
        except (TypeError, ValueError) as exc:
            raise BandEvaluationError(
                f"{source} results must be (energy, forces) pairs"
            ) from exc
        energy = float(energy_raw)
        forces = np.asarray(forces_raw, dtype=float)
        if forces.shape != (len(image), 3):
            raise BandEvaluationError(
                f"{source} returned forces of shape {forces.shape} for an "
                f"image of {len(image)} atoms"
            )
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise BandEvaluationError(
                f"{source} returned non-finite energies or forces"
            )
        validated.append((energy, forces.copy()))
    return validated


class CallableBandEvaluator:
    """Adapter over any calculator exposing ``evaluate_band(images)``."""

    def __init__(self, calculator: Any):
        self._calculator = calculator

    def evaluate_band(self, images: Sequence[Atoms]) -> list[BandImageResult]:
        results = self._calculator.evaluate_band(images)
        return _validated_results(
            images,
            results,
            source=type(self._calculator).__name__ + ".evaluate_band",
        )


class FairChemBandEvaluator:
    """Batch a band through a FAIR-Chem predict unit in one forward pass.

    Mirrors the single-image path exactly: each image is validated with the
    predictor for the calculator's task, then converted with the calculator's
    own atoms-to-graph converter (``calc.a2g``).  The resulting graphs are
    collated into one batch and the calculator's predictor runs a single
    forward.  Reusing the validation and ``a2g`` surfaces preserves
    task-specific defaults (including charge/spin) and graph construction.

    Built defensively against fairchem-core API drift: construction raises
    ``BandEvaluationError`` when the calculator does not expose the expected
    predictor/converter surface, and ``resolve_band_evaluator`` treats that as
    "unsupported" rather than a hard failure.
    """

    def __init__(self, calculator: Any):
        predictor = getattr(calculator, "predictor", None)
        if predictor is None or not callable(getattr(predictor, "predict", None)):
            raise BandEvaluationError(
                "calculator does not expose a predictor with a predict() "
                "method"
            )
        converter = getattr(calculator, "a2g", None)
        if not callable(converter):
            raise BandEvaluationError(
                "calculator does not expose a callable a2g atoms->graph "
                "converter"
            )
        validator = getattr(predictor, "validate_atoms_data", None)
        if not callable(validator):
            raise BandEvaluationError(
                "calculator predictor does not expose validate_atoms_data()"
            )
        task_name = getattr(calculator, "task_name", None)
        if not isinstance(task_name, str) or not task_name:
            raise BandEvaluationError(
                "calculator does not expose a non-empty task_name"
            )
        try:
            from fairchem.core.datasets.atomic_data import (  # type: ignore
                atomicdata_list_to_batch,
            )
        except ImportError as exc:
            raise BandEvaluationError(
                f"fairchem batch collate helper unavailable: {exc}"
            ) from exc
        self._calculator = calculator
        self._predictor = predictor
        self._converter = converter
        self._validator = validator
        self._task_name = task_name
        self._collate = atomicdata_list_to_batch
        self._device = getattr(predictor, "device", None)

    def evaluate_band(self, images: Sequence[Atoms]) -> list[BandImageResult]:
        try:
            data_list = []
            for image in images:
                self._validator(image, self._task_name)
                data_list.append(self._converter(image))
            batch = self._collate(data_list)
            if self._device is not None and hasattr(batch, "to"):
                batch = batch.to(self._device)
            predictions = self._predictor.predict(batch)
        except Exception as exc:
            raise BandEvaluationError(
                f"batched fairchem prediction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        energies = predictions.get("energy")
        forces = predictions.get("forces")
        if energies is None or forces is None:
            raise BandEvaluationError(
                "fairchem prediction is missing energy or forces"
            )
        energy_values = np.asarray(
            energies.detach().cpu().numpy()
            if hasattr(energies, "detach")
            else energies,
            dtype=float,
        ).reshape(-1)
        force_values = np.asarray(
            forces.detach().cpu().numpy()
            if hasattr(forces, "detach")
            else forces,
            dtype=float,
        )
        counts = [len(image) for image in images]
        if energy_values.shape[0] != len(images) or force_values.shape[0] != sum(
            counts
        ):
            raise BandEvaluationError(
                "fairchem prediction shapes do not match the band: "
                f"{energy_values.shape[0]} energies for {len(images)} images, "
                f"{force_values.shape[0]} force rows for {sum(counts)} atoms"
            )
        results: list[BandImageResult] = []
        offset = 0
        for count, energy in zip(counts, energy_values):
            results.append(
                (float(energy), force_values[offset:offset + count])
            )
            offset += count
        return _validated_results(
            images,
            results,
            source="FairChemBandEvaluator",
        )


def resolve_band_evaluator(calculator: Any) -> BandEvaluator | None:
    """Return a band evaluator for ``calculator``, or ``None`` if unsupported.

    Resolution never raises: an unsupported or misbehaving calculator logs a
    warning once per call site and the caller falls back to per-image
    evaluation.
    """
    if calculator is None:
        return None
    if callable(getattr(calculator, "evaluate_band", None)):
        return CallableBandEvaluator(calculator)
    if hasattr(calculator, "predictor") and hasattr(calculator, "a2g"):
        try:
            return FairChemBandEvaluator(calculator)
        except BandEvaluationError as exc:
            _log.warning(
                "Calculator %s looks like a FAIR-Chem calculator but cannot "
                "be batched (%s); falling back to per-image NEB evaluation.",
                type(calculator).__name__,
                exc,
            )
            return None
    return None


class BandImageCalculator:
    """Per-image calculator facade primed by a batched band evaluation.

    Mirrors the pooled-NEB facade contract: distinct object per image,
    ``get_forces``/``get_potential_energy`` served from the primed cache when
    the queried geometry matches the last batched evaluation.  A cache miss
    (which only happens if ASE queries a geometry outside the batched
    prefetch) falls back to the single concrete calculator so correctness
    never depends on prefetch completeness.
    """

    def __init__(self, fallback_calculator: Any):
        self._fallback = fallback_calculator
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

    def has_result_for(self, atoms: Atoms) -> bool:
        """Whether energy *and* forces are cached for this exact geometry."""
        return self._matches(atoms) and self._forces is not None

    def store(self, atoms: Atoms, energy: float, forces: np.ndarray) -> None:
        self._positions = np.asarray(atoms.positions, dtype=float).copy()
        self._cell = np.asarray(atoms.cell.array, dtype=float).copy()
        self._numbers = np.asarray(atoms.numbers, dtype=int).copy()
        self._energy = float(energy)
        self._forces = np.asarray(forces, dtype=float).copy()

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        if self._matches(atoms) and self._forces is not None:
            return self._forces.copy()
        forces = np.asarray(
            self._fallback.get_forces(atoms), dtype=float
        ).copy()
        energy = float(self._fallback.get_potential_energy(atoms))
        self.store(atoms, energy, forces)
        return forces

    def get_potential_energy(
        self,
        atoms: Atoms,
        force_consistent: bool = False,
    ) -> float:
        if self._matches(atoms):
            return float(self._energy)  # type: ignore[arg-type]
        energy = float(
            self._fallback.get_potential_energy(
                atoms,
                force_consistent=force_consistent,
            )
        )
        self._positions = np.asarray(atoms.positions, dtype=float).copy()
        self._cell = np.asarray(atoms.cell.array, dtype=float).copy()
        self._numbers = np.asarray(atoms.numbers, dtype=int).copy()
        self._energy = energy
        self._forces = None
        return energy


__all__ = [
    "BandEvaluationError",
    "BandEvaluator",
    "BandImageCalculator",
    "BandImageResult",
    "CallableBandEvaluator",
    "FairChemBandEvaluator",
    "resolve_band_evaluator",
]
