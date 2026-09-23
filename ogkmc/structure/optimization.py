"""Bulk and structure optimisation helpers."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS, MDMin

try:
    from ase.filters import ExpCellFilter
except ImportError:  # pragma: no cover
    from ase.constraints import ExpCellFilter  # type: ignore[no-redef]

from ogkmc.structure.builders import (
    _ase_reference_lp,
    _build_primitive_cell,
    _extract_lp,
    _normalise_lp,
    _print_divider,
    _print_header,
    _validate_crystal_structure,
)
from ogkmc.core.pbc import set_full_pbc_if_cell
from ogkmc.io.calculators import CalculatorConfigError, acquire_calculator
from ogkmc.io.atoms import copy_atoms_with_results
from ogkmc.structure.types import LatticeParams
from ogkmc.utils.optimizers import (
    DEFAULT_OPTIMIZER,
    REGULAR_OPTIMIZERS,
    normalize_optimizer_name,
    normalize_optimizer_kwargs,
)
from ogkmc.utils.telemetry import instrument


class StructureOptimisationError(RuntimeError):
    """Structure relaxation failed while retaining its last geometry.

    ``atoms`` is detached from the live model after the last completed
    optimizer update. Cached energy and forces, when valid, are retained in a
    safe ASE single-point calculator so callers can persist the failed geometry
    without rerunning the model.
    """

    def __init__(
        self,
        message: str,
        atoms: Atoms,
        *,
        converged: bool | None,
        steps: int,
    ) -> None:
        super().__init__(message)
        self.atoms = copy_atoms_with_results(atoms)
        self.converged = converged
        self.steps = int(steps)


def _optimizer_class(name: str):
    canonical = normalize_optimizer_name(
        name,
        allowed=REGULAR_OPTIMIZERS,
        setting="optimizer",
    )
    return {
        "lbfgs": LBFGS,
        "bfgs": BFGS,
        "fire": FIRE,
        "mdmin": MDMin,
    }[canonical]


@instrument("optimization.bulk")
def optimise_bulk(
    symbol: str,
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    calculator=None,
    fmax: float = 0.01,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> Tuple[Atoms, Dict[str, float]]:
    """Relax a bulk unit cell and return the optimised Atoms and lattice params."""
    if calculator is None:
        raise CalculatorConfigError(
            "optimise_bulk requires an explicit calculator"
        )

    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if lattice_constant is None:
        lp_in = _ase_reference_lp(symbol, crystal_structure)
    else:
        lp_in = _normalise_lp(lattice_constant, crystal_structure)

    bulk_atoms = _build_primitive_cell(symbol, crystal_structure, lp_in)
    set_full_pbc_if_cell(bulk_atoms)

    with acquire_calculator(calculator, purpose="bulk lattice relaxation") as calc:
        bulk_atoms.calc = calc

        ecf = ExpCellFilter(bulk_atoms)
        optimizer_cls = _optimizer_class(optimizer)
        constructor_kwargs = normalize_optimizer_kwargs(
            optimizer,
            optimizer_kwargs,
            allowed=REGULAR_OPTIMIZERS,
            setting="optimizer_kwargs",
        )
        opt = optimizer_cls(  # type: ignore[arg-type]
            ecf,
            logfile=os.devnull,
            **constructor_kwargs,
        )
        opt.run(fmax=fmax)

        if not opt.converged():
            raise RuntimeError(
                f"bulk lattice relaxation did not converge at fmax={fmax} eV/Å"
            )

        lp_out = _extract_lp(bulk_atoms, crystal_structure)

        if verbose:
            e = bulk_atoms.get_potential_energy()
            _print_header(f"Bulk {symbol} ({crystal_structure.upper()}) optimised")
            print(f"  Converged : {opt.converged()}  |  steps : {opt.get_number_of_steps()}")
            print(f"  E/atom    : {e / len(bulk_atoms):.5f} eV")
            for k, v in lp_out.items():
                print(f"  Opt {k}     : {v:.4f} Å")
            _print_divider()
        bulk_atoms.calc = None

    return bulk_atoms, lp_out


@instrument("optimization.structure")
def optimise_structure(
    atoms: Atoms,
    calculator=None,
    fmax: float = 0.05,
    steps: int = 1000,
    logfile: Optional[str] = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> Atoms:
    """Relax an ASE Atoms object with the selected ASE optimizer."""
    result = atoms.copy()
    set_full_pbc_if_cell(result)

    if calculator is not None:
        result.calc = calculator
    else:
        existing = atoms.calc
        if existing is not None:
            try:
                result.calc = copy.deepcopy(existing)
            except Exception as deepcopy_exc:
                try:
                    result.calc = copy.copy(existing)
                except Exception:
                    raise RuntimeError(
                        f"optimise_structure: could not copy calculator "
                        f"'{existing.__class__.__name__}'. deepcopy error: "
                        f"{deepcopy_exc}. Pass the calculator explicitly via "
                        "the 'calculator' argument."
                    ) from deepcopy_exc
        else:
            raise CalculatorConfigError(
                "optimise_structure requires an explicit calculator or an "
                "input Atoms object with an attached calculator"
            )

    log = logfile if logfile is not None else os.devnull
    optimizer_cls = _optimizer_class(optimizer)
    constructor_kwargs = normalize_optimizer_kwargs(
        optimizer,
        optimizer_kwargs,
        allowed=REGULAR_OPTIMIZERS,
        setting="optimizer_kwargs",
    )
    opt = optimizer_cls(result, logfile=log, **constructor_kwargs)
    try:
        opt.run(fmax=fmax, steps=steps)
    except CalculatorConfigError:
        raise
    except Exception as exc:
        completed_steps = int(opt.get_number_of_steps())
        raise StructureOptimisationError(
            "optimise_structure failed after "
            f"{completed_steps} steps: {type(exc).__name__}: {exc}",
            result,
            converged=None,
            steps=completed_steps,
        ) from exc

    if verbose:
        e = result.get_potential_energy()
        print(
            f"  Optimisation : converged={opt.converged()} "
            f"steps={opt.get_number_of_steps()}  "
            f"E={e:.4f} eV  E/atom={e/len(result):.4f} eV/atom"
        )

    if not opt.converged():
        completed_steps = int(opt.get_number_of_steps())
        raise StructureOptimisationError(
            f"optimise_structure did not converge within {steps} steps "
            f"(fmax={fmax} eV/Å)",
            result,
            converged=False,
            steps=completed_steps,
        )

    return result


def _resolve_lattice_params(
    symbol: str,
    crystal_structure: str,
    lattice_constant: LatticeParams,
    calculator,
    fmax: float = 0.01,
    optimizer: str = DEFAULT_OPTIMIZER,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Return a lattice-parameter dict, running bulk relaxation if needed."""
    if lattice_constant is not None:
        return _normalise_lp(lattice_constant, crystal_structure)
    _, lp = optimise_bulk(
        symbol,
        crystal_structure=crystal_structure,
        lattice_constant=None,
        calculator=calculator,
        fmax=fmax,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
    )
    return lp


__all__ = ["optimise_bulk", "optimise_structure", "_resolve_lattice_params"]
