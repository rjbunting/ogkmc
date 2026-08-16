"""SMILES normalization and filesystem-label helpers."""

from __future__ import annotations

import re

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


def canonical_atom_inventory_smiles(smiles: str | None) -> str:
    """Canonicalize without erasing explicitly represented H atoms.

    RDKit's default SMILES parser removes bonded ``[H]`` nodes and may later
    serialize that hydrogen as an implicit valence.  That normalization is
    chemically equivalent for ordinary molecular identity, but it is not
    equivalent for AutoKMC when ``add_hydrogens=False``: explicit atoms are
    part of the simulated atom inventory while implicit hydrogens are not.

    This bond-network canonicalizer therefore keeps explicit H nodes and
    expands bracket H counts such as ``[OH]`` into those nodes before writing
    canonical SMILES.  Hydrogens that were merely implicit remain implicit.
    """
    if smiles is None:
        return ""
    try:
        silence_rdkit_warnings()
        from rdkit import Chem

        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        molecule = Chem.MolFromSmiles(str(smiles), parser)
        if molecule is None:
            return str(smiles)
        explicit_hydrogen_atoms = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetNumExplicitHs() > 0
        ]
        if explicit_hydrogen_atoms:
            molecule = Chem.AddHs(
                molecule,
                onlyOnAtoms=explicit_hydrogen_atoms,
            )
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        return str(smiles)


def smiles_to_dirname(label: str) -> str:
    """Convert a SMILES or reaction-process label to a filesystem-safe name."""
    text = str(label).replace("↔", "~").replace("[", "(").replace("]", ")")
    text = re.sub(r'[\\/:*?"<>|]', "_", text).strip(". ")
    return text[:64] or "unknown"


__all__ = [
    "canonical_atom_inventory_smiles",
    "canonical_smiles",
    "smiles_to_dirname",
]
