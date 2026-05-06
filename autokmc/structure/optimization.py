"""Bulk and structure optimisation helpers."""

from __future__ import annotations

import copy
import os
import warnings
from typing import Dict, Optional, Tuple

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

try:
    from ase.filters import ExpCellFilter
except ImportError:  # pragma: no cover
    from ase.constraints import ExpCellFilter  # type: ignore[no-redef]

from autokmc.structure.builders import (
    _ase_reference_lp,
    _build_primitive_cell,
    _extract_lp,
    _normalise_lp,
    _print_divider,
    _print_header,
    _validate_crystal_structure,
)
from autokmc.io.calculators import acquire_calculator
from autokmc.structure.types import LatticeParams


def optimise_bulk(
    symbol: str,
    crystal_structure: str = "fcc",
    lattice_constant: LatticeParams = None,
    calculator=None,
    fmax: float = 0.01,
    verbose: bool = True,
) -> Tuple[Atoms, Dict[str, float]]:
    """Relax a bulk unit cell and return the optimised Atoms and lattice params."""
    if calculator is None:
        calculator = EMT()

    crystal_structure = crystal_structure.lower()
    _validate_crystal_structure(crystal_structure)

    if lattice_constant is None:
        lp_in = _ase_reference_lp(symbol, crystal_structure)
    else:
        lp_in = _normalise_lp(lattice_constant, crystal_structure)

    bulk_atoms = _build_primitive_cell(symbol, crystal_structure, lp_in)

    with acquire_calculator(calculator, purpose="bulk lattice relaxation") as calc:
        bulk_atoms.calc = calc

        ecf = ExpCellFilter(bulk_atoms)
        opt = LBFGS(ecf, logfile=os.devnull)  # type: ignore[arg-type]
        opt.run(fmax=fmax)

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


def optimise_structure(
    atoms: Atoms,
    calculator=None,
    fmax: float = 0.05,
    steps: int = 1000,
    logfile: Optional[str] = None,
    verbose: bool = True,
) -> Atoms:
    """Relax an ASE Atoms object with LBFGS and return an optimised copy."""
    result = atoms.copy()

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
            result.calc = EMT()

    log = logfile if logfile is not None else os.devnull
    opt = LBFGS(result, logfile=log)
    opt.run(fmax=fmax, steps=steps)

    if verbose:
        e = result.get_potential_energy()
        print(
            f"  Optimisation : converged={opt.converged()} "
            f"steps={opt.get_number_of_steps()}  "
            f"E={e:.4f} eV  E/atom={e/len(result):.4f} eV/atom"
        )

    if not opt.converged():
        warnings.warn(
            f"optimise_structure did not converge within {steps} steps "
            f"(fmax={fmax} eV/Å). The returned structure may not be at a "
            "local minimum.",
            RuntimeWarning,
            stacklevel=2,
        )

    return result


def _resolve_lattice_params(
    symbol: str,
    crystal_structure: str,
    lattice_constant: LatticeParams,
    calculator,
    fmax: float = 0.01,
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
        verbose=verbose,
    )
    return lp


__all__ = ["optimise_bulk", "optimise_structure", "_resolve_lattice_params"]
