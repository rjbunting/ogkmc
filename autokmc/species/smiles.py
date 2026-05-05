"""SMILES normalization and filesystem-label helpers."""

from __future__ import annotations

import re


def canonical_smiles(smiles: str | None) -> str:
    """Return RDKit canonical SMILES when RDKit can parse the input."""
    if smiles is None:
        return ""
    try:
        from rdkit import Chem

        return Chem.CanonSmiles(str(smiles))
    except Exception:
        return str(smiles)


def smiles_to_dirname(label: str) -> str:
    """Convert a SMILES or reaction-process label to a filesystem-safe name."""
    text = str(label).replace("↔", "~").replace("[", "(").replace("]", ")")
    text = re.sub(r'[\\/:*?"<>|]', "_", text).strip(". ")
    return text[:64] or "unknown"


__all__ = ["canonical_smiles", "smiles_to_dirname"]
