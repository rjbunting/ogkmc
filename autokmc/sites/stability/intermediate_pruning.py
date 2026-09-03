"""Conservative classification of NEB intermediates as composite KMC events.

An optimized local minimum along a direct NEB is only actionable when it can
be identified with a materialized adsorption placement and the corresponding
shorter reaction is already present in the enumerated network.  This module
performs that exact, topology-aware match.  It deliberately returns ``None``
for missing or ambiguous matches: uncertainty must never delete a KMC event.
"""

from __future__ import annotations

from hashlib import sha256
import itertools
import json
from typing import Any, Iterable

import networkx as nx
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from autokmc.sites.identity import member_identifier


DIRECT_EVENT_ELEMENTARY = "elementary"
DIRECT_EVENT_COMPOSITE = "composite"
INTERMEDIATE_PRUNING_POLICY = "materialized_component_exact_v1"


class CompositeDirectEventDetected(RuntimeError):
    """Raised internally to stop an NEB once a composite event is proven."""

    def __init__(self, reason: str, certificate: dict[str, Any]):
        super().__init__(reason)
        self.certificate = certificate


def direct_event_is_admissible(lateral_class: Any) -> bool:
    """Return whether a lateral result may be installed in the KMC index."""
    return (
        getattr(lateral_class, "direct_event_status", None)
        != DIRECT_EVENT_COMPOSITE
    )


def mark_composite_direct_event(
    lateral_class: Any,
    certificate: dict[str, Any],
) -> None:
    """Latch a proven composite classification without changing failure state."""
    lateral_class.direct_event_status = DIRECT_EVENT_COMPOSITE
    lateral_class.direct_event_certificate = dict(certificate)
    lateral_class.direct_event_reason = str(certificate["reason"])
    lateral_class.stable = None


def _identifier_payload(identifier: tuple[str, str]) -> list[str]:
    return [str(identifier[0]), str(identifier[1])]


def _placement_key(site: Any, member_index: int) -> tuple[str, str]:
    return member_identifier(site, int(member_index))


def _ordered_nodes(G: nx.Graph, node_ids: Iterable[int]) -> list[int]:
    return sorted(
        (int(node_id) for node_id in node_ids if int(node_id) in G),
        key=lambda node_id: (
            int(G.nodes[node_id].get("reactant_index", node_id)),
            int(node_id),
        ),
    )


def _expected_graph(G: nx.Graph, node_ids: Iterable[int]) -> nx.Graph:
    """Return the element/topology/surface-coordination graph of a placement."""
    ordered = _ordered_nodes(G, node_ids)
    out = nx.Graph()
    for local_index, node_id in enumerate(ordered):
        out.add_node(
            local_index,
            element=str(G.nodes[node_id].get("element", "")),
            clique=frozenset(
                int(value)
                for value in (G.nodes[node_id].get("clique") or ())
            ),
        )
    for left, left_id in enumerate(ordered):
        for right in range(left + 1, len(ordered)):
            right_id = ordered[right]
            siblings = G.nodes[left_id].get("siblings") or ()
            if G.has_edge(left_id, right_id) or right_id in siblings:
                out.add_edge(left, right)
    return out


def _observed_graph(
    G: nx.Graph,
    atoms: Atoms,
    *,
    n_slab: int,
    n_lateral: int,
    n_reacting: int,
    nl_mult: float,
) -> nx.Graph | None:
    """Extract the reacting block using the same neighbor rule as stability."""
    start = int(n_slab) + int(n_lateral)
    stop = start + int(n_reacting)
    if start < 0 or stop > len(atoms) or n_reacting < 1:
        return None

    slab_nodes = sorted(
        (
            int(node_id)
            for node_id, data in G.nodes(data=True)
            if data.get("type") in ("bulk", "surface")
        ),
        key=lambda node_id: G.nodes[node_id].get("index", node_id),
    )
    if len(slab_nodes) != int(n_slab):
        return None

    cutoffs = natural_cutoffs(atoms, mult=float(nl_mult))
    neighbors = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(atoms)

    out = nx.Graph()
    symbols = atoms.get_chemical_symbols()
    for local_index, ase_index in enumerate(range(start, stop)):
        bonded = {int(value) for value in neighbors.get_neighbors(ase_index)[0]}
        clique = frozenset(
            slab_nodes[slab_index]
            for slab_index in bonded
            if 0 <= slab_index < int(n_slab)
            and G.nodes[slab_nodes[slab_index]].get("type") == "surface"
        )
        out.add_node(
            local_index,
            element=str(symbols[ase_index]),
            clique=clique,
        )
    for left in range(int(n_reacting)):
        left_ase = start + left
        bonded = {int(value) for value in neighbors.get_neighbors(left_ase)[0]}
        for right in range(left + 1, int(n_reacting)):
            if start + right in bonded:
                out.add_edge(left, right)
    return out


def _same_endpoint(observed: nx.Graph, expected: nx.Graph) -> bool:
    if observed.number_of_nodes() != expected.number_of_nodes():
        return False

    def node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            left.get("element") == right.get("element")
            and left.get("clique") == right.get("clique")
        )

    return nx.is_isomorphic(observed, expected, node_match=node_match)


def _interior_refinements(
    refinement_initial: Atoms,
    refinement_final: Atoms,
    metadata: dict[str, Any],
) -> list[tuple[str, int, Atoms]]:
    energies = list(metadata.get("stalled_profile_energies_ev") or [])
    if len(energies) < 3:
        return []
    candidates = (
        (
            "left",
            metadata.get("left_state_image_index"),
            refinement_initial,
        ),
        (
            "right",
            metadata.get("right_state_image_index"),
            refinement_final,
        ),
    )
    out: list[tuple[str, int, Atoms]] = []
    for side, raw_index, atoms in candidates:
        if raw_index is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < index < len(energies) - 1:
            out.append((side, index, atoms))
    return out


def _adsorbate_sites_for(G: nx.Graph, smiles: str) -> list[Any]:
    sites_by_smiles = G.graph.get("adsorbate_sites", {}) or {}
    if not isinstance(sites_by_smiles, dict):
        return []
    return list(sites_by_smiles.get(smiles, ()) or ())


def classify_diffusion_intermediate(
    G: nx.Graph,
    diffusion_site: Any,
    member_index: int,
    refinement_initial: Atoms,
    refinement_final: Atoms,
    metadata: dict[str, Any],
    *,
    n_slab: int,
    n_lateral: int,
    n_reacting: int,
    nl_mult: float,
) -> dict[str, Any] | None:
    """Prove that a direct hop contains a registered diffusion placement."""
    try:
        site_a, member_a, site_b, member_b = diffusion_site.members[
            int(member_index)
        ]
    except (IndexError, TypeError, ValueError):
        return None
    original = {
        _placement_key(site_a, member_a),
        _placement_key(site_b, member_b),
    }
    component_sites = list(
        (G.graph.get("diffusion_sites", {}) or {}).get(
            diffusion_site.reactant,
            (),
        )
    )

    for side, image_index, atoms in _interior_refinements(
        refinement_initial,
        refinement_final,
        metadata,
    ):
        observed = _observed_graph(
            G,
            atoms,
            n_slab=n_slab,
            n_lateral=n_lateral,
            n_reacting=n_reacting,
            nl_mult=nl_mult,
        )
        if observed is None:
            continue
        matches: dict[tuple[str, str], tuple[Any, int]] = {}
        for candidate_site in _adsorbate_sites_for(
            G,
            diffusion_site.reactant,
        ):
            for candidate_member, node_ids in enumerate(
                candidate_site.member_node_ids
            ):
                key = _placement_key(candidate_site, candidate_member)
                if key in original:
                    continue
                if _same_endpoint(observed, _expected_graph(G, node_ids)):
                    matches[key] = (candidate_site, candidate_member)
        # A geometric minimum must identify one physical placement uniquely.
        if len(matches) != 1:
            continue
        intermediate_key = next(iter(matches))
        components: list[list[str]] = []
        connected_original: set[tuple[str, str]] = set()
        for candidate_diffusion in component_sites:
            for candidate_member, member in enumerate(candidate_diffusion.members):
                candidate_pair = {
                    _placement_key(member[0], member[1]),
                    _placement_key(member[2], member[3]),
                }
                if intermediate_key in candidate_pair and candidate_pair & original:
                    connected_original.update(candidate_pair & original)
                    components.append(
                        _identifier_payload(
                            member_identifier(
                                candidate_diffusion,
                                candidate_member,
                            )
                        )
                    )
        # Removing A->B is safe only when the KMC network already contains
        # both replacement hops A->I and I->B.
        if connected_original != original:
            continue
        return {
            "policy": INTERMEDIATE_PRUNING_POLICY,
            "classification": DIRECT_EVENT_COMPOSITE,
            "channel": "diffusion",
            "reason": "registered_intermediate_diffusion_placement",
            "original_member_id": _identifier_payload(
                member_identifier(diffusion_site, int(member_index))
            ),
            "intermediate_placement_id": _identifier_payload(intermediate_key),
            "component_member_ids": sorted(components),
            "refinement_side": side,
            "intermediate_image_index": int(image_index),
            "source_stage": metadata.get("source_stage"),
            "trigger": metadata.get("trigger"),
        }
    return None


def _template_key(template: Any) -> tuple[Any, ...]:
    return (
        getattr(template, "smiles_a", None),
        getattr(template, "smiles_b", None),
        getattr(template, "smiles_c", None),
        getattr(template, "bond_type", None),
    )


def intermediate_pruning_network_signature(
    G: nx.Graph,
    channel: str,
    site: Any,
) -> str:
    """Hash the materialized members relevant to one pruning decision."""
    identifiers: list[list[str]] = []
    diffusion_store = G.graph.get("diffusion_sites", {}) or {}
    if channel == "diffusion":
        smiles = str(site.reactant)
        for adsorbate_site in _adsorbate_sites_for(G, smiles):
            identifiers.extend(
                _identifier_payload(
                    member_identifier(adsorbate_site, member_index)
                )
                for member_index in range(len(adsorbate_site.member_node_ids))
            )
        if isinstance(diffusion_store, dict):
            for diffusion_site in diffusion_store.get(smiles, ()) or ():
                identifiers.extend(
                    _identifier_payload(
                        member_identifier(diffusion_site, member_index)
                    )
                    for member_index in range(len(diffusion_site.member_node_ids))
                )
    elif channel == "bond":
        template = _template_key(site.template)
        gas_product = bool(getattr(site, "gas_product", False))
        for bond_site in G.graph.get("bond_reaction_sites", []) or []:
            if (
                _template_key(bond_site.template) == template
                and bool(getattr(bond_site, "gas_product", False)) == gas_product
            ):
                identifiers.extend(
                    _identifier_payload(
                        member_identifier(bond_site, member_index)
                    )
                    for member_index in range(len(bond_site.member_node_ids))
                )
        if isinstance(diffusion_store, dict):
            for smiles in {str(value) for value in template[:3] if value}:
                for diffusion_site in diffusion_store.get(smiles, ()) or ():
                    identifiers.extend(
                        _identifier_payload(
                            member_identifier(diffusion_site, member_index)
                        )
                        for member_index in range(
                            len(diffusion_site.member_node_ids)
                        )
                    )
    else:
        raise ValueError(f"unsupported pruning channel {channel!r}")
    encoded = json.dumps(
        sorted(identifiers),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _ab_key(member: tuple[Any, ...]) -> frozenset[tuple[str, str]]:
    return frozenset(
        (
            _placement_key(member[0], member[1]),
            _placement_key(member[2], member[3]),
        )
    )


def _c_key(member: tuple[Any, ...]) -> tuple[str, str] | None:
    if member[4] is None or int(member[5]) < 0:
        return None
    return _placement_key(member[4], member[5])


def _diffusion_component_members(
    G: nx.Graph,
    left: tuple[str, str],
    right: tuple[str, str],
    smiles: str,
) -> list[list[str]]:
    target = {left, right}
    found: list[list[str]] = []
    sites_by_smiles = G.graph.get("diffusion_sites", {}) or {}
    if not isinstance(sites_by_smiles, dict):
        return found
    for diffusion_site in sites_by_smiles.get(smiles, ()) or ():
        for member_index, member in enumerate(diffusion_site.members):
            pair = {
                _placement_key(member[0], member[1]),
                _placement_key(member[2], member[3]),
            }
            if pair == target:
                found.append(
                    _identifier_payload(
                        member_identifier(diffusion_site, member_index)
                    )
                )
    return sorted(found)


def _diffusion_links_for_changed_placements(
    G: nx.Graph,
    original: list[tuple[Any, int]],
    alternative: list[tuple[Any, int]],
) -> list[list[str]] | None:
    original_by_key = {
        _placement_key(site, member_index): (site, int(member_index))
        for site, member_index in original
    }
    alternative_by_key = {
        _placement_key(site, member_index): (site, int(member_index))
        for site, member_index in alternative
    }
    shared = set(original_by_key) & set(alternative_by_key)
    old = [value for key, value in original_by_key.items() if key not in shared]
    new = [value for key, value in alternative_by_key.items() if key not in shared]
    if not old or len(old) != len(new):
        return None

    for old_order in itertools.permutations(old):
        component_ids: list[list[str]] = []
        valid = True
        for (new_site, new_member), (old_site, old_member) in zip(
            new,
            old_order,
        ):
            if str(new_site.reactant) != str(old_site.reactant):
                valid = False
                break
            links = _diffusion_component_members(
                G,
                _placement_key(old_site, old_member),
                _placement_key(new_site, new_member),
                str(new_site.reactant),
            )
            if not links:
                valid = False
                break
            component_ids.extend(links)
        if valid:
            return sorted(component_ids)
    return None


def classify_bond_intermediate(
    G: nx.Graph,
    bond_site: Any,
    member_index: int,
    refinement_initial: Atoms,
    refinement_final: Atoms,
    metadata: dict[str, Any],
    *,
    n_slab: int,
    n_lateral: int,
    n_reacting: int,
    nl_mult: float,
) -> dict[str, Any] | None:
    """Prove diffusion-before/after-barrier through an alternative site pair."""
    try:
        original_member = bond_site.members[int(member_index)]
    except (IndexError, TypeError, ValueError):
        return None
    original_ab = _ab_key(original_member)
    original_c = _c_key(original_member)
    original_id = member_identifier(bond_site, int(member_index))
    candidates = [
        site
        for site in (G.graph.get("bond_reaction_sites", []) or [])
        if _template_key(site.template) == _template_key(bond_site.template)
        and bool(getattr(site, "gas_product", False))
        == bool(getattr(bond_site, "gas_product", False))
    ]

    for side, image_index, atoms in _interior_refinements(
        refinement_initial,
        refinement_final,
        metadata,
    ):
        observed = _observed_graph(
            G,
            atoms,
            n_slab=n_slab,
            n_lateral=n_lateral,
            n_reacting=n_reacting,
            nl_mult=nl_mult,
        )
        if observed is None:
            continue
        routes: dict[tuple[Any, ...], list[list[str]]] = {}
        route_details: dict[tuple[Any, ...], dict[str, Any]] = {}
        for candidate_site in candidates:
            for candidate_index, member in enumerate(candidate_site.members):
                candidate_id = member_identifier(candidate_site, candidate_index)
                if candidate_id == original_id:
                    continue
                candidate_ab = _ab_key(member)
                candidate_c = _c_key(member)
                a_nodes = member[0].member_node_ids[int(member[1])]
                b_nodes = member[2].member_node_ids[int(member[3])]
                ab_matches = _same_endpoint(
                    observed,
                    _expected_graph(G, [*a_nodes, *b_nodes]),
                )
                c_matches = False
                if member[4] is not None and int(member[5]) >= 0:
                    c_nodes = member[4].member_node_ids[int(member[5])]
                    c_matches = _same_endpoint(
                        observed,
                        _expected_graph(G, c_nodes),
                    )

                changed_side: str | None = None
                diffusion_components: list[list[str]] | None = None
                if (
                    ab_matches
                    and candidate_ab != original_ab
                    and candidate_c == original_c
                ):
                    changed_side = "ab"
                    diffusion_components = _diffusion_links_for_changed_placements(
                        G,
                        [
                            (original_member[0], int(original_member[1])),
                            (original_member[2], int(original_member[3])),
                        ],
                        [
                            (member[0], int(member[1])),
                            (member[2], int(member[3])),
                        ],
                    )
                elif (
                    c_matches
                    and candidate_c != original_c
                    and candidate_ab == original_ab
                ):
                    changed_side = "c"
                    if original_member[4] is not None and member[4] is not None:
                        diffusion_components = (
                            _diffusion_links_for_changed_placements(
                                G,
                                [(original_member[4], int(original_member[5]))],
                                [(member[4], int(member[5]))],
                            )
                        )
                if changed_side is None or not diffusion_components:
                    continue
                route_key = (
                    tuple(sorted(candidate_ab)),
                    candidate_c,
                    changed_side,
                )
                routes.setdefault(route_key, []).append(
                    _identifier_payload(candidate_id)
                )
                route_details[route_key] = {
                    "changed_endpoint": changed_side,
                    "diffusion_component_member_ids": diffusion_components,
                    "bond_component_member_ids": [
                        _identifier_payload(candidate_id)
                    ],
                    "alternative_ab_placement_ids": [
                        _identifier_payload(value)
                        for value in sorted(candidate_ab)
                    ],
                    "alternative_c_placement_id": (
                        None
                        if candidate_c is None
                        else _identifier_payload(candidate_c)
                    ),
                }
        # Multiple physical alternative routes are ambiguous even if each
        # independently resembles a known reaction member.
        if len(routes) != 1:
            continue
        route_key = next(iter(routes))
        bond_components = sorted(routes[route_key])
        diffusion_components = route_details[route_key][
            "diffusion_component_member_ids"
        ]
        route_details[route_key]["bond_component_member_ids"] = bond_components
        return {
            "policy": INTERMEDIATE_PRUNING_POLICY,
            "classification": DIRECT_EVENT_COMPOSITE,
            "channel": "bond",
            "reason": "registered_alternative_bond_endpoint",
            "original_member_id": _identifier_payload(original_id),
            "component_member_ids": sorted(
                [*bond_components, *diffusion_components]
            ),
            **route_details[route_key],
            "refinement_side": side,
            "intermediate_image_index": int(image_index),
            "source_stage": metadata.get("source_stage"),
            "trigger": metadata.get("trigger"),
        }
    return None


def retain_refinement_and_maybe_suppress(
    lateral_class: Any,
    refinement_initial: Atoms,
    refinement_final: Atoms,
    metadata: dict[str, Any],
    certificate: dict[str, Any] | None,
) -> None:
    """Persist every check and stop immediately on a proven composite event."""
    lateral_class.atoms_neb_refinement_initial = refinement_initial
    lateral_class.atoms_neb_refinement_final = refinement_final
    lateral_class.neb_intermediate_refinement = dict(metadata)
    history = getattr(
        lateral_class,
        "neb_intermediate_refinement_history",
        None,
    )
    if not isinstance(history, list):
        history = []
        lateral_class.neb_intermediate_refinement_history = history
    history.append(dict(metadata))
    if certificate is None:
        return

    mark_composite_direct_event(lateral_class, certificate)
    raise CompositeDirectEventDetected(
        "NEB contains a registered intermediate; the direct event is "
        "composite and has been suppressed from KMC",
        dict(certificate),
    )


__all__ = [
    "CompositeDirectEventDetected",
    "DIRECT_EVENT_COMPOSITE",
    "DIRECT_EVENT_ELEMENTARY",
    "INTERMEDIATE_PRUNING_POLICY",
    "classify_bond_intermediate",
    "classify_diffusion_intermediate",
    "direct_event_is_admissible",
    "intermediate_pruning_network_signature",
    "mark_composite_direct_event",
    "retain_refinement_and_maybe_suppress",
]
