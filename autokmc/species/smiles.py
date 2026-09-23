"""SMILES normalization and filesystem-label helpers."""

from __future__ import annotations

import re
import hashlib
from collections import Counter
from typing import Any

from autokmc.utils.rdkit_logging import silence_rdkit_warnings


SMILES_IDENTITY_VERSION = "2"


class SmilesError(ValueError):
    """An invalid or unsupported molecular definition, before geometry work."""


def molecule_from_smiles(
    smiles: str,
    *,
    add_hydrogens: bool | None = None,
    allow_dummies: bool = False,
):
    """Parse one molecule without losing explicit atoms or indexed ordering.

    ``None`` retains implicit H notation for a species label. A boolean
    materializes the requested atom inventory and then disables implicit H.
    Atom maps are bookkeeping, while isotopes and stereochemistry are identity.
    CO's charge-separated spelling is normalized to AutoKMC's neutral CO
    convention; this is a deliberate CO alias, not general charge removal.
    """
    from rdkit import Chem

    silence_rdkit_warnings()
    if not isinstance(smiles, str) or not smiles.strip():
        raise SmilesError("SMILES must be a non-empty string")
    parser: Any = Chem.SmilesParserParams()
    parser.removeHs = False
    parser.parseName = False
    parser.allowCXSMILES = False
    mol = Chem.MolFromSmiles(smiles.strip(), parser)
    if mol is None:
        raise SmilesError(f"RDKit could not parse SMILES: {smiles!r}")
    if not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise SmilesError(f"SMILES must describe one connected molecule: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            if not allow_dummies:
                raise SmilesError(f"Dummy attachment atoms are not species: {smiles!r}")
            if atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetAtomicNum() == 0:
                raise SmilesError(f"Each dummy must have one real-atom neighbour: {smiles!r}")

    if mol.GetNumAtoms() == 2 and mol.GetNumBonds() == 1:
        atoms = {atom.GetAtomicNum(): atom for atom in mol.GetAtoms()}
        bond = mol.GetBondWithIdx(0)
        if (
            set(atoms) == {6, 8}
            and atoms[6].GetFormalCharge() == -1
            and atoms[8].GetFormalCharge() == 1
            and bond.GetBondType() == Chem.BondType.TRIPLE
            and all(atom.GetTotalNumHs() == 0 for atom in mol.GetAtoms())
        ):
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
                atom.SetNumRadicalElectrons(0)
                atom.SetNoImplicit(True)
            bond.SetBondType(Chem.BondType.DOUBLE)
            Chem.SanitizeMol(mol)

    hydrogen_hosts = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetNumExplicitHs() > 0
        or (add_hydrogens is True and atom.GetNumImplicitHs() > 0)
    ]
    if hydrogen_hosts:
        mol = Chem.AddHs(
            mol, onlyOnAtoms=hydrogen_hosts, explicitOnly=add_hydrogens is not True,
        )
    if add_hydrogens is not None:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                atom.SetNoImplicit(True)
        Chem.SanitizeMol(mol)
    return mol


def require_charge_free(molecule) -> None:
    """Enforce the supported domain of the radical-based bond enumerator."""
    if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()):
        raise SmilesError(
            "Bond chemistry requires charge-free atoms after CO normalization; "
            "other formally charged species are not supported by this enumerator"
        )


def canonical_smiles(smiles: str | None) -> str:
    """Canonical species label, retaining explicit H and implicit-H notation.

    Non-molecular diagnostic labels are returned unchanged. Molecular input
    boundaries must use ``molecule_from_smiles`` so invalid inputs fail there.
    """
    if smiles is None:
        return ""
    try:
        from rdkit import Chem

        return Chem.MolToSmiles(molecule_from_smiles(smiles), canonical=True)
    except (ValueError, ImportError):
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

    Bracket H counts are expanded into nodes. Implicit H is materialized only
    when requested; otherwise its valence becomes radical valence. Thus the
    result describes exactly the atoms that will be simulated.
    """
    if smiles is None:
        return ""
    try:
        from rdkit import Chem

        molecule = molecule_from_smiles(smiles, add_hydrogens=add_hydrogens)
        return Chem.MolToSmiles(molecule, canonical=True)
    except (ValueError, ImportError):
        return str(smiles)


def reactant_atom_inventory_smiles(reactant) -> str:
    """Return the explicit H inventory used to build a reactant.

    Old checkpoints and manually constructed Reactants may lack the recorded
    inventory.  Recover their H policy by comparing both allowed SMILES
    interpretations with the actual atoms; never silently discard H atoms.
    """
    stored = getattr(reactant, "atom_inventory_smiles", "")
    smiles = str(reactant.smiles)
    atoms = getattr(reactant, "atoms", None)
    if atoms is None:
        return canonical_atom_inventory_smiles(stored or smiles)

    expected = _ase_inventory(atoms)
    candidates = (
        [canonical_atom_inventory_smiles(stored)] if stored else
        [canonical_atom_inventory_smiles(smiles, add_hydrogens=add_hydrogens)
         for add_hydrogens in (False, True)]
    )
    for candidate in candidates:
        molecule = molecule_from_smiles(candidate, add_hydrogens=False)
        if _molecule_inventory(molecule) == expected:
            if stored and candidate not in {
                canonical_atom_inventory_smiles(smiles, add_hydrogens=False),
                canonical_atom_inventory_smiles(smiles, add_hydrogens=True),
            }:
                raise SmilesError(
                    f"Reactant {smiles!r} disagrees with its stored chemical identity"
                )
            return candidate
    raise ValueError(
        f"reactant {smiles!r} has an atom inventory inconsistent with its SMILES"
    )


def _ase_inventory(atoms):
    return Counter(
        (int(number), round(float(mass), 6))
        for number, mass in zip(atoms.numbers, atoms.get_masses())
    )


def _molecule_inventory(molecule):
    from ase.data import atomic_masses

    return Counter(
        (atom.GetAtomicNum(), round(float(
            atom.GetMass() if atom.GetIsotope() else atomic_masses[atom.GetAtomicNum()]
        ), 6))
        for atom in molecule.GetAtoms()
    )


def molecule_from_reactant(reactant):
    """Recover chemical bond orders without inferring them from coordinates."""
    expected = reactant_atom_inventory_smiles(reactant)
    from rdkit import Chem

    for add_hydrogens in (False, True):
        molecule = molecule_from_smiles(reactant.smiles, add_hydrogens=add_hydrogens)
        if Chem.MolToSmiles(molecule, canonical=True) == expected:
            from ase.data import atomic_masses

            indexed = [
                (atom.GetAtomicNum(), round(float(
                    atom.GetMass() if atom.GetIsotope() else atomic_masses[atom.GetAtomicNum()]
                ), 6)) for atom in molecule.GetAtoms()
            ]
            actual = [
                (int(number), round(float(mass), 6))
                for number, mass in zip(reactant.atoms.numbers, reactant.atoms.get_masses())
            ]
            if indexed != actual:
                raise SmilesError(f"Reactant {reactant.smiles!r} has inconsistent indexed atoms")
            return molecule
    raise SmilesError(f"Reactant {reactant.smiles!r} disagrees with its stored atom inventory")


def smiles_to_dirname(label: str) -> str:
    """Readable label plus a digest, safe against lossy filesystem spelling.

    Slash/backslash stereochemistry, case-insensitive filesystems, bracket
    replacement, and truncation must not make distinct species share a folder.
    """
    raw = str(label)
    text = raw.replace("↔", "~").replace("[", "(").replace("]", ")")
    text = re.sub(r'[\\/:*?"<>|]', "_", text).strip(". ")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{text[:47] or 'unknown'}-{digest}"


__all__ = [
    "canonical_atom_inventory_smiles",
    "canonical_smiles",
    "molecule_from_smiles",
    "molecule_from_reactant",
    "require_charge_free",
    "SmilesError",
    "SMILES_IDENTITY_VERSION",
    "reactant_atom_inventory_smiles",
    "smiles_to_dirname",
]
