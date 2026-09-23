"""Lightweight accessors for raw ``networkx.Graph.graph`` state.

These helpers centralize common graph metadata keys without hiding the raw
``G.graph[...]`` dictionaries used during development. They are intentionally
small: callers can migrate one repeated key at a time while notebooks and
exploratory workflows can still inspect the underlying dictionaries directly.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from ogkmc.core.pbc import full_pbc_for_cell

CELL = "cell"
PBC = "pbc"
HULL_EQUATIONS = "hull_equations"
ANCHOR_SITES = "anchor_sites"
RAW_CLIQUES = "raw_cliques"
ADSORBATE_SITES = "adsorbate_sites"
DIFFUSION_SITES = "diffusion_sites"
BOND_REACTION_SITES = "bond_reaction_sites"
CLIQUE_TO_MEMBERS = "clique_to_members"
SURFACE_NODE_TO_MEMBERS = "surface_node_to_members"
DIFFUSION_CLIQUE_TO_MEMBERS = "diffusion_clique_to_members"
DIFFUSION_SURFACE_NODE_TO_MEMBERS = "diffusion_surface_node_to_members"
BOND_CLIQUE_TO_MEMBERS = "bond_clique_to_members"
BOND_SURFACE_NODE_TO_MEMBERS = "bond_surface_node_to_members"
OCCUPIED_BY_CLIQUE = "occupied_by_clique"
N_OCCUPIED = "n_occupied"
BOND_REGISTRY = "bond_registry"
SURFACE_APSP = "surface_apsp"
SURFACE_SHELLS_CACHE = "_surface_shells_cache"
CLIQUE_POSITION_INDEX_CACHE = "_clique_position_index_cache"
SURFACE_ATOMS_ARRAY_CACHE = "_surface_atoms_array_cache"
RUN_ID = "run_id"
FROZEN_INDICES = "frozen_indices"


def get_cell(G: nx.Graph) -> np.ndarray:
    """Return graph cell metadata as a float ``(3, 3)`` array."""
    return np.array(G.graph[CELL], dtype=float)


def get_pbc(G: nx.Graph) -> np.ndarray:
    """Return graph PBC metadata as a bool length-3 array."""
    if PBC in G.graph:
        return np.asarray(G.graph[PBC], dtype=bool)
    if CELL in G.graph:
        return full_pbc_for_cell(G.graph[CELL])
    return np.zeros(3, dtype=bool)


def get_run_id(G: nx.Graph) -> str | None:
    value = G.graph.get(RUN_ID)
    return None if value is None else str(value)


def set_run_id(G: nx.Graph, value: str) -> None:
    G.graph[RUN_ID] = str(value)


def get_frozen_indices(G: nx.Graph) -> list[int]:
    return [int(index) for index in G.graph.get(FROZEN_INDICES, []) or []]


def set_frozen_indices(G: nx.Graph, values: Any | None) -> None:
    G.graph[FROZEN_INDICES] = [int(index) for index in values or []]


def get_anchor_sites(G: nx.Graph, element: str | None = None) -> Any:
    sites = G.graph.get(ANCHOR_SITES, {})
    return sites if element is None else sites.get(element, [])


def get_raw_cliques(G: nx.Graph, element: str | None = None) -> Any:
    raw = G.graph.get(RAW_CLIQUES, {})
    return raw if element is None else raw.get(element, {})


def get_adsorbate_sites(G: nx.Graph, smiles: str | None = None) -> Any:
    sites = G.graph.get(ADSORBATE_SITES, {})
    return sites if smiles is None else sites.get(smiles, [])


def get_diffusion_sites(G: nx.Graph, smiles: str | None = None) -> Any:
    sites = G.graph.get(DIFFUSION_SITES, {})
    return sites if smiles is None else sites.get(smiles, [])


def set_diffusion_sites(G: nx.Graph, sites: Any) -> None:
    G.graph[DIFFUSION_SITES] = sites


def get_bond_reaction_sites(G: nx.Graph) -> list:
    return list(G.graph.get(BOND_REACTION_SITES, []) or [])


def set_bond_reaction_sites(G: nx.Graph, sites: Any) -> None:
    G.graph[BOND_REACTION_SITES] = list(sites or [])


def get_clique_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(CLIQUE_TO_MEMBERS, {})
    return G.graph.get(CLIQUE_TO_MEMBERS, {})


def get_surface_node_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(SURFACE_NODE_TO_MEMBERS, {})
    return G.graph.get(SURFACE_NODE_TO_MEMBERS, {})


def get_diffusion_clique_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(DIFFUSION_CLIQUE_TO_MEMBERS, {})
    return G.graph.get(DIFFUSION_CLIQUE_TO_MEMBERS, {})


def get_diffusion_surface_node_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(DIFFUSION_SURFACE_NODE_TO_MEMBERS, {})
    return G.graph.get(DIFFUSION_SURFACE_NODE_TO_MEMBERS, {})


def get_bond_clique_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(BOND_CLIQUE_TO_MEMBERS, {})
    return G.graph.get(BOND_CLIQUE_TO_MEMBERS, {})


def get_bond_surface_node_to_members(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(BOND_SURFACE_NODE_TO_MEMBERS, {})
    return G.graph.get(BOND_SURFACE_NODE_TO_MEMBERS, {})


def get_occupied_by_clique(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(OCCUPIED_BY_CLIQUE, {})
    return G.graph.get(OCCUPIED_BY_CLIQUE, {})


def get_n_occupied(G: nx.Graph) -> int:
    return int(G.graph.get(N_OCCUPIED, 0))


def set_n_occupied(G: nx.Graph, value: int) -> None:
    G.graph[N_OCCUPIED] = max(0, int(value))


def get_bond_registry(G: nx.Graph, *, create: bool = False) -> dict:
    if create:
        return G.graph.setdefault(BOND_REGISTRY, {})
    return G.graph.get(BOND_REGISTRY, {})


def set_bond_registry(G: nx.Graph, registry: dict) -> None:
    G.graph[BOND_REGISTRY] = registry


def invalidate_surface_shell_cache(G: nx.Graph) -> None:
    G.graph.pop(SURFACE_SHELLS_CACHE, None)


def invalidate_surface_apsp(G: nx.Graph) -> None:
    G.graph.pop(SURFACE_APSP, None)


def invalidate_position_caches(G: nx.Graph) -> None:
    G.graph.pop(CLIQUE_POSITION_INDEX_CACHE, None)
    G.graph.pop(SURFACE_ATOMS_ARRAY_CACHE, None)


def validate_runtime_indexes(G: nx.Graph) -> None:
    """Raise when occupancy indexes disagree with the live graph.

    This intentionally validates only inexpensive, representation-independent
    invariants.  It is suitable for tests, checkpoint boundaries, and debug
    builds without imposing a new wrapper around NetworkX.
    """
    occupied_by_clique = get_occupied_by_clique(G)
    for clique, node_ids in occupied_by_clique.items():
        for node_id in node_ids:
            if node_id not in G:
                raise ValueError(
                    f"occupied_by_clique references missing node {node_id!r}"
                )
            node = G.nodes[node_id]
            if not node.get("occupied", False):
                raise ValueError(
                    f"occupied_by_clique references unoccupied node {node_id!r}"
                )
            if node.get("clique") != clique:
                raise ValueError(
                    f"occupied node {node_id!r} is indexed under the wrong clique"
                )

    occupied_placements = {
        (
            node.get("reactant"),
            node.get("site_iso_class"),
            node.get("site_member_index"),
        )
        for _, node in G.nodes(data=True)
        if node.get("type") == "adsorbate" and node.get("occupied", False)
    }
    if get_n_occupied(G) != len(occupied_placements):
        raise ValueError(
            "n_occupied does not match the number of occupied placements: "
            f"{get_n_occupied(G)} != {len(occupied_placements)}"
        )


__all__ = [
    "get_cell",
    "get_pbc",
    "get_run_id",
    "set_run_id",
    "get_frozen_indices",
    "set_frozen_indices",
    "get_anchor_sites",
    "get_raw_cliques",
    "get_adsorbate_sites",
    "get_diffusion_sites",
    "set_diffusion_sites",
    "get_bond_reaction_sites",
    "set_bond_reaction_sites",
    "get_clique_to_members",
    "get_surface_node_to_members",
    "get_diffusion_clique_to_members",
    "get_diffusion_surface_node_to_members",
    "get_bond_clique_to_members",
    "get_bond_surface_node_to_members",
    "get_occupied_by_clique",
    "get_n_occupied",
    "set_n_occupied",
    "get_bond_registry",
    "set_bond_registry",
    "invalidate_surface_shell_cache",
    "invalidate_surface_apsp",
    "invalidate_position_caches",
    "validate_runtime_indexes",
]
