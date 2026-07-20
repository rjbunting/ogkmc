"""Canonical state transitions for persisted KMC events.

This module records *what changed* without tracking products or mechanisms in
the live KMC state.  The explicit inputs and outputs are sufficient for an
offline analyzer to back-propagate from a product desorption through the fired
surface reactions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from autokmc.species.smiles import canonical_smiles


def _node_ids(site: Any, member_index: int) -> list[int]:
    try:
        return sorted(int(node) for node in site.member_node_ids[int(member_index)])
    except (AttributeError, IndexError, TypeError, ValueError):
        return []


def _surface_cliques(site: Any, member_index: int) -> list[list[int]]:
    cached = getattr(site, "_member_cliques", None)
    if cached is not None:
        try:
            cliques = cached[int(member_index)]
            return sorted(
                (sorted(int(node) for node in clique) for clique in cliques),
                key=lambda clique: tuple(clique),
            )
        except (IndexError, TypeError, ValueError):
            pass
    try:
        member = site.members[int(member_index)]
    except (AttributeError, IndexError, TypeError):
        return []
    out = []
    for clique in member:
        if clique is None or not isinstance(clique, (set, frozenset, tuple, list)):
            continue
        try:
            out.append(sorted(int(node) for node in clique))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda clique: tuple(clique))


def surface_state(site: Any, member_index: int, *, species: str | None = None) -> dict[str, Any]:
    """Describe one concrete occupied adsorbate placement."""
    canonical = canonical_smiles(species or getattr(site, "reactant", ""))
    nodes = _node_ids(site, member_index)
    identity = json.dumps(
        {"species": canonical, "node_ids": nodes},
        sort_keys=True,
        separators=(",", ":"),
    )
    placement_id = "placement-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return {
        "phase": "surface",
        "species": canonical,
        "placement_id": placement_id,
        "site_iso_class": int(getattr(site, "iso_class", -1)),
        "member_index": int(member_index),
        "node_ids": nodes,
        "surface_cliques": _surface_cliques(site, member_index),
    }


def gas_state(species: str) -> dict[str, Any]:
    return {"phase": "gas", "species": canonical_smiles(species)}


def _diffusion_endpoints(reaction: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    site = reaction.site
    index = int(reaction.member_index)
    try:
        site_a, member_a, site_b, member_b = site.members[index]
        state_a = surface_state(site_a, member_a, species=site.reactant)
        state_b = surface_state(site_b, member_b, species=site.reactant)
        return state_a, state_b
    except (AttributeError, IndexError, TypeError, ValueError):
        # Compatibility for lightweight external/test reaction objects.
        raw = getattr(site, "member_node_ids", [])
        pair = raw[index] if index < len(raw) else raw
        if (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(item, (list, tuple, set, frozenset)) for item in pair)
        ):
            pseudo_a = type("Placement", (), {
                "reactant": site.reactant,
                "iso_class": getattr(site, "iso_class", -1),
                "member_node_ids": [list(pair[0])],
            })()
            pseudo_b = type("Placement", (), {
                "reactant": site.reactant,
                "iso_class": getattr(site, "iso_class", -1),
                "member_node_ids": [list(pair[1])],
            })()
        else:
            endpoint_nodes = list(raw) if isinstance(raw, (list, tuple)) else []
            nodes_a = endpoint_nodes[0] if endpoint_nodes else []
            nodes_b = endpoint_nodes[1] if len(endpoint_nodes) > 1 else []
            pseudo_a = type("Placement", (), {
                "reactant": site.reactant,
                "iso_class": getattr(site, "iso_class", -1),
                "member_node_ids": [list(nodes_a)],
            })()
            pseudo_b = type("Placement", (), {
                "reactant": site.reactant,
                "iso_class": getattr(site, "iso_class", -1),
                "member_node_ids": [list(nodes_b)],
            })()
        return surface_state(pseudo_a, 0), surface_state(pseudo_b, 0)


def _bond_endpoints(
    reaction: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    site = reaction.site
    template = site.template
    index = int(reaction.member_index)
    try:
        site_a, member_a, site_b, member_b, site_c, member_c = site.members[index]
        state_a = surface_state(site_a, member_a, species=template.smiles_a)
        state_b = surface_state(site_b, member_b, species=template.smiles_b)
        state_c = (
            None
            if site_c is None
            else surface_state(site_c, member_c, species=template.smiles_c)
        )
        return state_a, state_b, state_c
    except (AttributeError, IndexError, TypeError, ValueError):
        raw = getattr(site, "member_node_ids", [])
        triple = raw[index] if index < len(raw) else raw
        if not (
            isinstance(triple, (list, tuple))
            and len(triple) == 3
            and all(isinstance(item, (list, tuple, set, frozenset)) for item in triple)
        ):
            triple = ([index * 3], [index * 3 + 1], [index * 3 + 2])
        states = []
        for species, nodes in zip(
            (template.smiles_a, template.smiles_b, template.smiles_c), triple
        ):
            pseudo = type("Placement", (), {
                "reactant": species,
                "iso_class": getattr(site, "iso_class", -1),
                "member_node_ids": [list(nodes)],
            })()
            states.append(surface_state(pseudo, 0, species=species))
        return states[0], states[1], states[2]


def reaction_transition(reaction: Any) -> dict[str, Any]:
    """Return canonical ``inputs``/``outputs`` for a fired reaction."""
    kind = str(getattr(reaction, "kind", ""))
    direction = getattr(reaction, "direction", None)

    if kind in {"adsorption", "desorption"}:
        species = canonical_smiles(getattr(reaction.site, "reactant", ""))
        surface = surface_state(reaction.site, reaction.member_index, species=species)
        gas = gas_state(species)
        inputs, outputs = (([gas], [surface]) if kind == "adsorption" else ([surface], [gas]))
        return {
            "inputs": inputs,
            "outputs": outputs,
            "template": {"species": species},
            "gas_product": False,
        }

    if kind == "diffusion":
        state_a, state_b = _diffusion_endpoints(reaction)
        if direction == "a_to_b":
            inputs, outputs = [state_a], [state_b]
        elif direction == "b_to_a":
            inputs, outputs = [state_b], [state_a]
        else:
            raise ValueError(f"unknown diffusion direction {direction!r}")
        return {
            "inputs": inputs,
            "outputs": outputs,
            "template": {"species": canonical_smiles(reaction.site.reactant)},
            "gas_product": False,
        }

    if kind == "bond":
        template = reaction.site.template
        smiles_a = canonical_smiles(template.smiles_a)
        smiles_b = canonical_smiles(template.smiles_b)
        smiles_c = canonical_smiles(template.smiles_c)
        state_a, state_b, state_c = _bond_endpoints(reaction)
        gas_product = bool(getattr(reaction.site, "gas_product", False))
        c_state = gas_state(smiles_c) if gas_product else state_c
        if c_state is None:
            raise ValueError("surface bond product is missing its C placement")
        if direction == "couple":
            inputs, outputs = [state_a, state_b], [c_state]
        elif direction == "dissoc":
            inputs, outputs = [c_state], [state_a, state_b]
        else:
            raise ValueError(f"unknown bond direction {direction!r}")
        return {
            "inputs": inputs,
            "outputs": outputs,
            "template": {
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "smiles_c": smiles_c,
                "bond_type": getattr(template, "bond_type", None),
            },
            "gas_product": gas_product,
        }

    raise ValueError(f"unsupported reaction kind {kind!r}")


def occupied_surface_states(graph: Any, sites: Iterable[Any]) -> list[dict[str, Any]]:
    """Snapshot occupied placements for the run manifest."""
    states: list[dict[str, Any]] = []
    seen: set[str] = set()
    for site in sites:
        for member_index, node_ids in enumerate(getattr(site, "member_node_ids", ())):
            if not any(
                node in graph and graph.nodes[node].get("occupied", False)
                for node in node_ids
            ):
                continue
            state = surface_state(site, member_index)
            if state["placement_id"] not in seen:
                seen.add(state["placement_id"])
                states.append(state)
    states.sort(key=lambda state: state["placement_id"])
    return states


__all__ = [
    "gas_state",
    "occupied_surface_states",
    "reaction_transition",
    "surface_state",
]
