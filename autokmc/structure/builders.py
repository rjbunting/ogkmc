"""Shared composition, lattice, and formatting helpers for structure builders."""

from __future__ import annotations

import math
from typing import Dict, Union

import numpy as np
from ase import Atoms
from ase.build import bulk

from autokmc.core.constants import RANDOM_SEED
from autokmc.structure.types import Composition


def _parse_composition(composition: Composition) -> Dict[str, float]:
    """Return a normalised ``{symbol: fraction}`` composition dict."""
    if isinstance(composition, str):
        return {composition: 1.0}

    comp = dict(composition)
    total = sum(comp.values())
    if total <= 0:
        raise ValueError(f"Composition values must be positive, got: {comp}")
    for sym, val in comp.items():
        if val < 0:
            raise ValueError(
                f"Composition fraction for '{sym}' is negative ({val}). "
                "All fractions must be non-negative."
            )
    return {sym: val / total for sym, val in comp.items()}


def _primary_element(composition: Dict[str, float]) -> str:
    """Return the highest-fraction element used as the parent lattice."""
    return max(composition, key=composition.__getitem__)


def _apply_composition(
    atoms: Atoms,
    composition: Dict[str, float],
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Atoms:
    """Randomly substitute atoms to match target composition fractions."""
    result = atoms.copy()
    n_total = len(result)
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_total)
    syms = np.array(result.get_chemical_symbols())

    primary = _primary_element(composition)
    cursor = 0
    for sym, frac in composition.items():
        if sym == primary:
            continue
        n_sub = int(round(n_total * frac))
        syms[indices[cursor:cursor + n_sub]] = sym
        cursor += n_sub

    result.set_chemical_symbols(syms.tolist())

    if verbose:
        unique, counts = np.unique(syms, return_counts=True)
        print("  Alloy composition applied:")
        for s, c in zip(unique, counts):
            print(f"    {s}: {c} atoms  ({c/n_total*100:.1f} %)")

    return result


_VALID_STRUCTURES = {"fcc", "bcc", "hcp"}
_HCP_IDEAL_CA = math.sqrt(8.0 / 3.0)


def _validate_crystal_structure(cs: str) -> None:
    if cs not in _VALID_STRUCTURES:
        raise ValueError(
            f"crystal_structure must be one of {sorted(_VALID_STRUCTURES)}, "
            f"got '{cs}'."
        )


def _ase_reference_lp(symbol: str, crystal_structure: str) -> Dict[str, float]:
    """Return ASE reference lattice parameters for *symbol* as a dict."""
    from ase.data import atomic_numbers, reference_states

    Z = atomic_numbers[symbol]
    ref = reference_states[Z] or {}
    a = float(ref.get("a", 3.5))
    if crystal_structure == "hcp":
        ca = float(ref.get("c/a", _HCP_IDEAL_CA))
        return {"a": a, "c": a * ca}
    return {"a": a}


def _normalise_lp(
    lattice_constant: Union[float, Dict[str, float]],
    crystal_structure: str,
) -> Dict[str, float]:
    """Coerce lattice constants to ``{"a": ..., ["c": ...]}``."""
    if isinstance(lattice_constant, (int, float)):
        a = float(lattice_constant)
        if crystal_structure == "hcp":
            return {"a": a, "c": a * _HCP_IDEAL_CA}
        return {"a": a}
    return dict(lattice_constant)


def _build_primitive_cell(symbol: str, crystal_structure: str, lp: Dict[str, float]) -> Atoms:
    """Build a bulk unit cell from lattice parameters."""
    a = lp["a"]
    if crystal_structure == "hcp":
        c = lp.get("c", a * _HCP_IDEAL_CA)
        return bulk(symbol, crystalstructure="hcp", a=a, c=c)
    return bulk(symbol, crystalstructure=crystal_structure, a=a, cubic=True)


def _build_surface_parent_cell(symbol: str, crystal_structure: str, lp: Dict[str, float]) -> Atoms:
    """Build the parent bulk cell used by pymatgen SlabGenerator."""
    a = lp["a"]
    if crystal_structure == "hcp":
        c = lp.get("c", a * _HCP_IDEAL_CA)
        return bulk(symbol, crystalstructure="hcp", a=a, c=c)
    return bulk(symbol, crystalstructure=crystal_structure, a=a, cubic=True)


def _extract_lp(atoms: Atoms, crystal_structure: str) -> Dict[str, float]:
    """Extract lattice parameters from an optimised bulk Atoms object."""
    cell = atoms.get_cell()
    if crystal_structure in ("fcc", "bcc"):
        return {"a": float(np.linalg.norm(cell[0]))}
    return {
        "a": float(np.linalg.norm(cell[0])),
        "c": float(np.linalg.norm(cell[2])),
    }


def _fmt_lp(lp: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f} Å" for k, v in lp.items())


def _print_header(title: str, width: int = 58) -> None:
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_divider(width: int = 58) -> None:
    print("=" * width)


__all__ = [
    "_parse_composition",
    "_primary_element",
    "_apply_composition",
    "_VALID_STRUCTURES",
    "_HCP_IDEAL_CA",
    "_validate_crystal_structure",
    "_ase_reference_lp",
    "_normalise_lp",
    "_build_primitive_cell",
    "_build_surface_parent_cell",
    "_extract_lp",
    "_fmt_lp",
    "_print_header",
    "_print_divider",
]
