"""SMILES normalization and filesystem-label helpers."""

from __future__ import annotations

import re
from typing import Any

from autokmc.utils.rdkit_logging import silence_rdkit_warnings


def canonical_smiles(smiles: str | None) -> str:
    """Return RDKit canonical SMILES when RDKit can parse the input."""
    if smiles is None:
        return ""
    try:
        silence_rdkit_warnings()
        from rdkit import Chem

        return Chem.CanonSmiles(str(smiles))
    except Exception:
        return str(smiles)


def canonical_atom_inventory_smiles(
    smiles: str | None, *, add_hydrogens: bool = False,
) -> str:
    """Canonicalize without erasing explicitly represented H atoms.

    RDKit's default SMILES parser removes bonded ``[H]`` nodes and may later
    serialize that hydrogen as an implicit valence.  That normalization is
    chemically equivalent for ordinary molecular identity, but it is not
    equivalent for AutoKMC when ``add_hydrogens=False``: explicit atoms are
    part of the simulated atom inventory while implicit hydrogens are not.

    This bond-network canonicalizer therefore keeps explicit H nodes and
    expands bracket H counts such as ``[OH]`` into those nodes before writing
    canonical SMILES.  Hydrogens that were merely implicit remain implicit
    unless ``add_hydrogens=True`` requests their materialization, as when
    recording the atom inventory of a configured feed molecule.
    """
    if smiles is None:
        return ""
    try:
        silence_rdkit_warnings()
        from rdkit import Chem

        parser: Any = Chem.SmilesParserParams()
        parser.removeHs = False
        molecule = Chem.MolFromSmiles(str(smiles), parser)
        if molecule is None:
            return str(smiles)
        explicit_hydrogen_atoms = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if (
                atom.GetNumExplicitHs() > 0
                or (add_hydrogens and atom.GetNumImplicitHs() > 0)
            )
        ]
        if explicit_hydrogen_atoms:
            molecule = Chem.AddHs(
                molecule,
                onlyOnAtoms=explicit_hydrogen_atoms,
                explicitOnly=not add_hydrogens,
            )
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        return str(smiles)


def reactant_atom_inventory_smiles(reactant) -> str:
    """Return the explicit H inventory used to build a reactant.

    Old checkpoints and manually constructed Reactants may lack the recorded
    inventory.  Recover their H policy by comparing both allowed SMILES
    interpretations with the actual atoms; never silently discard H atoms.
    """
    stored = getattr(reactant, "atom_inventory_smiles", "")
    if stored:
        return str(stored)
    smiles = str(reactant.smiles)
    atoms = getattr(reactant, "atoms", None)
    if atoms is None:
        return canonical_atom_inventory_smiles(smiles)

    from collections import Counter
    from rdkit import Chem

    expected = Counter(int(number) for number in atoms.numbers)
    parser: Any = Chem.SmilesParserParams()
    parser.removeHs = False
    for add_hydrogens in (False, True):
        candidate = canonical_atom_inventory_smiles(
            smiles, add_hydrogens=add_hydrogens,
        )
        molecule = Chem.MolFromSmiles(candidate, parser)
        if molecule is not None and Counter(
            atom.GetAtomicNum() for atom in molecule.GetAtoms()
        ) == expected:
            return candidate
    raise ValueError(
        f"reactant {smiles!r} has an atom inventory inconsistent with its SMILES"
    )


def smiles_to_dirname(label: str) -> str:
    """Convert a SMILES or reaction-process label to a filesystem-safe name."""
    text = str(label).replace("↔", "~").replace("[", "(").replace("]", ")")
    text = re.sub(r'[\\/:*?"<>|]', "_", text).strip(". ")
    return text[:64] or "unknown"


__all__ = [
    "canonical_atom_inventory_smiles",
    "canonical_smiles",
    "reactant_atom_inventory_smiles",
    "smiles_to_dirname",
]
