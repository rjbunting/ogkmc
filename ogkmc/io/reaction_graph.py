"""Portable graph representation used by the reaction-result database.

The in-memory lateral graphs contain run-local identifiers such as
``iso_class`` and node ids.  Those values are useful while one KMC run is
active, but they are not stable database keys.  This module reduces a lateral
graph to chemistry/topology labels that can be compared across runs, creates a
cheap Weisfeiler-Lehman prefilter, and always confirms a hit with full graph
isomorphism.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx

from ogkmc.core.atom_metadata import atom_metadata_key


REACTION_GRAPH_SCHEMA = "ogkmc-reaction-graph-v1"


def _node_label(data: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(data.get("type", "unknown")),
            str(data.get("element", "X")),
            str(data.get("reactant", "")),
            str(data.get("reactant_index", "")),
            str(data.get("endpoint_role", "")),
            str(data.get("atom_metadata_key", atom_metadata_key(data))),
        )
    )


def _edge_relation(graph: nx.Graph, u: Any, v: Any, data: Mapping[str, Any]) -> str:
    if data.get("relation"):
        return str(data["relation"])
    if data.get("anchor_bond"):
        return "anchor_bond"
    if data.get("intra_adsorbate"):
        return "intra_adsorbate"
    u_type = graph.nodes[u].get("type")
    v_type = graph.nodes[v].get("type")
    if u_type == "surface" and v_type == "surface":
        return "surface_bond"
    return "bond"


def normalise_reaction_graph(
    graph: nx.Graph,
    *,
    endpoint_node_ids: Iterable[Any] | None = None,
    endpoint_role: str = "site",
) -> nx.Graph:
    """Return the portable, labelled form of a lateral reaction graph.

    Node ids and ``iso_class`` are intentionally discarded.  The latter is a
    run-local enumeration and would otherwise prevent a chemically identical
    graph from matching in another run.  ``endpoint_node_ids`` is used for
    adsorption graphs, whose in-memory form predates explicit endpoint labels.
    Diffusion and bond graphs already carry their endpoint roles.
    """
    if graph is None:
        raise ValueError("a reaction graph is required")

    endpoint_ids = set(endpoint_node_ids or ())
    out = nx.Graph()
    for node, raw in graph.nodes(data=True):
        node_type = str(raw.get("type", "unknown"))
        attrs: dict[str, Any] = {
            "type": node_type,
            "element": str(raw.get("element", "X")),
            "atom_metadata_key": raw.get("atom_metadata_key", atom_metadata_key(raw)),
        }
        if node_type == "adsorbate":
            attrs["reactant"] = str(raw.get("reactant", ""))
            if raw.get("reactant_index") is not None:
                attrs["reactant_index"] = int(raw["reactant_index"])
            role = endpoint_role if node in endpoint_ids else raw.get("endpoint_role")
            if role not in (None, ""):
                attrs["endpoint_role"] = str(role)
        attrs["label"] = _node_label(attrs)
        out.add_node(str(node), **attrs)

    for u, v, raw in graph.edges(data=True):
        out.add_edge(str(u), str(v), relation=_edge_relation(graph, u, v, raw))

    out.graph.update(
        {
            "schema": REACTION_GRAPH_SCHEMA,
            "n_shells": int(graph.graph.get("n_shells", 0) or 0),
        }
    )
    return out


def reaction_graph_payload(graph: nx.Graph) -> dict[str, Any]:
    """Serialize a normalized graph without depending on node ids for meaning."""
    graph = normalise_reaction_graph(graph)
    nodes = [
        {"id": str(node), **{k: v for k, v in data.items() if k != "label"}}
        for node, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    ]
    edges = [
        {"source": str(u), "target": str(v), "relation": data["relation"]}
        for u, v, data in graph.edges(data=True)
    ]
    edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["relation"]))
    return {
        "schema": REACTION_GRAPH_SCHEMA,
        "directed": False,
        "multigraph": False,
        "n_shells": int(graph.graph.get("n_shells", 0) or 0),
        "nodes": nodes,
        "edges": edges,
    }


def reaction_graph_from_payload(payload: Mapping[str, Any]) -> nx.Graph:
    """Deserialize and validate an OGKMC reaction graph asset."""
    if payload.get("schema") != REACTION_GRAPH_SCHEMA:
        raise ValueError(f"unsupported reaction graph schema: {payload.get('schema')!r}")
    if payload.get("directed") is not False or payload.get("multigraph") is not False:
        raise ValueError("reaction graphs must be simple undirected graphs")

    graph = nx.Graph()
    for item in payload.get("nodes", []):
        attrs = {key: value for key, value in item.items() if key != "id"}
        attrs["label"] = _node_label(attrs)
        graph.add_node(str(item["id"]), **attrs)
    for item in payload.get("edges", []):
        graph.add_edge(
            str(item["source"]),
            str(item["target"]),
            relation=str(item.get("relation", "bond")),
        )
    graph.graph.update(
        {
            "schema": REACTION_GRAPH_SCHEMA,
            "n_shells": int(payload.get("n_shells", 0) or 0),
        }
    )
    return graph


def reaction_graph_hash(graph: nx.Graph) -> str:
    """Return a cheap isomorphism-invariant lookup hash.

    Hash equality is only a prefilter.  :func:`reaction_graphs_isomorphic` is
    the authoritative database-match test.
    """
    graph = normalise_reaction_graph(graph)
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="label",
        edge_attr="relation",
        iterations=5,
        digest_size=32,
    )


def reaction_graphs_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    """Confirm that two portable reaction graphs have identical labels/topology."""
    left = normalise_reaction_graph(left)
    right = normalise_reaction_graph(right)
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a.get("label") == b.get("label"),
        edge_match=lambda a, b: a.get("relation") == b.get("relation"),
    )


__all__ = [
    "REACTION_GRAPH_SCHEMA",
    "normalise_reaction_graph",
    "reaction_graph_payload",
    "reaction_graph_from_payload",
    "reaction_graph_hash",
    "reaction_graphs_isomorphic",
]
