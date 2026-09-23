"""Canonical labels and paths shared by reaction persistence and summaries."""

from __future__ import annotations

from ogkmc.core.constants import (
    BOND_FOLDER_FMT,
    DIFFUSION_FOLDER_FMT,
    REACTIONS_DIR,
)
from ogkmc.species.smiles import smiles_to_dirname


KIND_SUBDIR: dict[str, str] = {
    "adsorption": "adsorption",
    "desorption": "adsorption",
    "diffusion": "diffusion",
    "bond": "bond",
}


def kind_subdir(kind: str) -> str:
    return KIND_SUBDIR.get(str(kind), str(kind))


def reaction_folder_name(iso_class: int, lateral_class: int) -> str:
    return f"iso{int(iso_class)}_lat{int(lateral_class)}"


def diffusion_folder_name(iso_class: int, lateral_class: int) -> str:
    return DIFFUSION_FOLDER_FMT.format(iso=int(iso_class), lat=int(lateral_class))


def bond_folder_name(iso_class: int, lateral_class: int) -> str:
    return BOND_FOLDER_FMT.format(iso=int(iso_class), lat=int(lateral_class))


def kind_folder_name(subdir: str, iso_class: int, lateral_class: int) -> str:
    if subdir == "diffusion":
        return diffusion_folder_name(iso_class, lateral_class)
    if subdir == "bond":
        return bond_folder_name(iso_class, lateral_class)
    return reaction_folder_name(iso_class, lateral_class)


def reaction_smiles(reaction) -> str:
    site = reaction.site
    smiles = getattr(site, "reactant", None)
    if smiles:
        return str(smiles)
    template = getattr(site, "template", None)
    if template is not None:
        return f"{template.smiles_a}+{template.smiles_b}↔{template.smiles_c}"
    return ""


def reaction_relative_dir(
    kind: str,
    iso_class: int,
    lateral_class: int,
    smiles: str = "",
) -> str:
    subdir = kind_subdir(kind)
    species = smiles_to_dirname(smiles) if smiles else "unknown"
    folder = kind_folder_name(subdir, iso_class, lateral_class)
    return f"{REACTIONS_DIR}/{subdir}/{species}/{folder}"


__all__ = [
    "KIND_SUBDIR",
    "bond_folder_name",
    "diffusion_folder_name",
    "kind_folder_name",
    "kind_subdir",
    "reaction_folder_name",
    "reaction_relative_dir",
    "reaction_smiles",
]
