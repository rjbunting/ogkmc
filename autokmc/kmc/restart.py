"""Checkpoint and result helpers shared by KMC sessions and outputs."""

from __future__ import annotations

import random
from typing import Iterable

import networkx as nx
import numpy as np

from autokmc.core.graph_state import get_bond_registry
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.species.reactant import Reactant
from autokmc.species.smiles import reactant_atom_inventory_smiles


def capture_rng_state(rng) -> dict | None:
    """Return a serialisable state payload for a supported random generator."""
    if isinstance(rng, np.random.Generator):
        return {"kind": "numpy", "state": rng.bit_generator.state}
    if isinstance(rng, random.Random):
        return {"kind": "python", "state": rng.getstate()}
    return None


def restore_rng_state(rng, payload: dict | None):
    """Restore *payload* into a compatible generator, creating one if needed."""
    if payload is None:
        return rng
    kind = payload.get("kind")
    if kind == "numpy":
        state = payload["state"]
        generator_name = state.get("bit_generator")
        generator_type = getattr(np.random, str(generator_name), None)
        if (
            not isinstance(generator_type, type)
            or not issubclass(generator_type, np.random.BitGenerator)
        ):
            raise ValueError(f"unsupported NumPy bit generator: {generator_name!r}")
        numpy_generator = (
            rng
            if isinstance(rng, np.random.Generator)
            and type(rng.bit_generator) is generator_type
            else np.random.Generator(generator_type())
        )
        numpy_generator.bit_generator.state = state
        return numpy_generator
    if kind == "python":
        python_generator = rng if isinstance(rng, random.Random) else random.Random()
        python_generator.setstate(payload["state"])
        return python_generator
    raise ValueError(f"unsupported checkpoint RNG kind: {kind!r}")


def normalise_rng(rng, rng_state: dict | None = None):
    """Normalise seeds and optional checkpoint state to a random generator."""
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    return restore_rng_state(rng, rng_state)


def reactants_for_checkpoint(reactants, graph: nx.Graph) -> list:
    """Return initial and dynamically discovered species for restart."""
    if isinstance(reactants, dict):
        initial = list(reactants.values())
    elif isinstance(reactants, Iterable) and not isinstance(reactants, Reactant):
        initial = list(reactants)
    else:
        initial = [reactants]

    registry_species = get_bond_registry(graph).get("species", {}) or {}
    candidates = initial + list(registry_species.values())
    result: list = []
    seen_species: set[str] = set()
    seen_other: set[int] = set()
    for item in candidates:
        if item is None:
            continue
        if isinstance(item, Reactant):
            key = reactant_atom_inventory_smiles(item)
            if key in seen_species:
                continue
            seen_species.add(key)
        else:
            identity = id(item)
            if identity in seen_other:
                continue
            seen_other.add(identity)
        result.append(item)
    return result


def final_occupancy_by_species(
    adsorbate_sites: list[AdsorbateSite],
) -> dict[str, int]:
    """Return final occupancy keyed by species plus iso-class."""
    out: dict[str, int] = {}
    for site in adsorbate_sites:
        smiles = str(getattr(site, "reactant", "unknown") or "unknown")
        iso = int(getattr(site, "iso_class"))
        key = f"{smiles}:iso{iso}"
        out[key] = out.get(key, 0) + int(getattr(site, "_n_occupied", 0))
    return out


__all__ = [
    "capture_rng_state",
    "final_occupancy_by_species",
    "normalise_rng",
    "reactants_for_checkpoint",
    "restore_rng_state",
]
