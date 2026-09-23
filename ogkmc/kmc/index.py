"""Reaction leaf indexing for incremental KMC rate updates."""

from __future__ import annotations

from typing import Any, TypeAlias

from ogkmc.kmc.sampling import _RateSegmentTree
from ogkmc.reactions.adsorption import AdsorptionReaction
from ogkmc.reactions.bond import BondReaction
from ogkmc.reactions.diffusion import DiffusionReaction
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.bond import BondReactionSite
from ogkmc.sites.diffusion import DiffusionSite
from ogkmc.sites.identity import (
    SiteId,
    SiteMemberId,
    member_identifier,
    site_identifier,
)
from ogkmc.sites.stability.intermediate_pruning import (
    direct_event_is_admissible,
)

IndexedReaction: TypeAlias = AdsorptionReaction | DiffusionReaction | BondReaction


def _admissible_reaction(reaction: IndexedReaction | None) -> IndexedReaction | None:
    """Defense in depth: composite direct events always install as zero rate."""
    if reaction is None:
        return None
    if getattr(reaction, "kind", None) not in {"diffusion", "bond"}:
        return reaction
    if not direct_event_is_admissible(getattr(reaction, "lateral_class", None)):
        return None
    return reaction


class _ReactionIndex:
    """Flat ``(site, member)`` to leaf-id map plus a segment-tree of rates."""

    __slots__ = (
        "base",
        "n_total",
        "tree",
        "reactions",
        "site_order",
        "diffusion_site_order",
        "bond_site_order",
        "_adsorbate_ids",
        "_diffusion_ids",
        "_bond_ids",
        "_site_leaves",
        "_member_leaves",
    )

    def __init__(
        self,
        sites: list[AdsorbateSite],
        diffusion_sites: list[DiffusionSite] | None = None,
        bond_sites: list[BondReactionSite] | None = None,
    ):
        self.site_order: list[AdsorbateSite] = []
        self.diffusion_site_order: list[DiffusionSite] = []
        self.bond_site_order: list[BondReactionSite] = []
        self.base: dict[SiteId, int] = {}
        self._adsorbate_ids: set[SiteId] = set()
        self._diffusion_ids: set[SiteId] = set()
        self._bond_ids: set[SiteId] = set()
        self._site_leaves: dict[SiteId, list[int]] = {}
        self._member_leaves: dict[SiteMemberId, int] = {}
        self.n_total = 0
        self.tree = _RateSegmentTree(0)
        self.reactions: list[IndexedReaction | None] = []
        self.extend_sites(
            adsorbate_sites=sites,
            diffusion_sites=diffusion_sites,
            bond_sites=bond_sites,
        )

    def extend_sites(
        self,
        *,
        adsorbate_sites: list[AdsorbateSite] | None = None,
        diffusion_sites: list[DiffusionSite] | None = None,
        bond_sites: list[BondReactionSite] | None = None,
    ) -> int:
        """Append new site/member leaves without reinstalling existing rates.

        Returns the number of leaves added.  Duplicate stable IDs are rejected
        so a dynamically reconstructed object cannot silently shadow an active
        scientific site.
        """
        groups = (
            (
                list(adsorbate_sites or []),
                self.site_order,
                self._adsorbate_ids,
                "adsorbate",
            ),
            (
                list(diffusion_sites or []),
                self.diffusion_site_order,
                self._diffusion_ids,
                "diffusion",
            ),
            (
                list(bond_sites or []),
                self.bond_site_order,
                self._bond_ids,
                "bond",
            ),
        )
        pending: list[
            tuple[Any, list[Any], set[SiteId], SiteId, list[SiteMemberId]]
        ] = []
        pending_site_ids: set[SiteId] = set()
        pending_member_ids: set[SiteMemberId] = set()
        for sites, order, identifiers, label in groups:
            for site in sites:
                identifier = site_identifier(site)
                if identifier in self.base or identifier in pending_site_ids:
                    raise ValueError(
                        f"duplicate {label} site identifier {identifier!r}"
                    )
                pending_site_ids.add(identifier)
                member_ids = [
                    member_identifier(site, member_index)
                    for member_index in range(len(site.member_node_ids))
                ]
                duplicate_members: set[SiteMemberId] = set()
                for member_id in member_ids:
                    if (
                        member_id in self._member_leaves
                        or member_id in pending_member_ids
                    ):
                        duplicate_members.add(member_id)
                    pending_member_ids.add(member_id)
                if duplicate_members:
                    raise ValueError(
                        f"duplicate concrete member identifier(s) for "
                        f"{label} site {identifier!r}"
                    )
                pending.append(
                    (site, order, identifiers, identifier, member_ids)
                )

        old_total = self.n_total
        offset = old_total
        for site, order, identifiers, identifier, member_ids in pending:
            self.base[identifier] = offset
            identifiers.add(identifier)
            order.append(site)
            leaves = list(range(offset, offset + len(member_ids)))
            self._site_leaves[identifier] = leaves
            self._member_leaves.update(zip(member_ids, leaves))
            offset += len(member_ids)

        self.n_total = offset
        self.tree.grow(offset)
        self.reactions.extend([None] * (offset - old_total))
        return offset - old_total

    def contains(self, site) -> bool:
        """Return whether the stable scientific site is already indexed."""
        return site_identifier(site) in self.base

    def leaf_id(self, site, m_idx: int) -> int:
        return self._member_leaves[member_identifier(site, m_idx)]

    def install(
        self,
        rxn: IndexedReaction | None,
        site,
        m_idx: int,
    ) -> None:
        """Replace one member leaf without touching sibling members."""
        rxn = _admissible_reaction(rxn)
        i = self.leaf_id(site, m_idx)
        self.reactions[i] = rxn
        updates = {
            i: rxn.rate if (rxn is not None and rxn.rate > 0.0) else 0.0
        }
        if any(
            not direct_event_is_admissible(lateral_class)
            for lateral_class in (getattr(site, "lateral_classes", None) or [])
        ):
            # A lateral class can be shared by symmetry-equivalent members.
            # Once it becomes composite, remove every already-installed
            # sibling reaction backed by that same suppressed class.
            for leaf in self._site_leaves[site_identifier(site)]:
                existing = self.reactions[leaf]
                if _admissible_reaction(existing) is None:
                    self.reactions[leaf] = None
                    updates[leaf] = 0.0
        if len(updates) == 1:
            leaf, rate = next(iter(updates.items()))
            self.tree.update(leaf, rate)
        else:
            self.tree.update_many(updates.items())

    def install_site(self, site, reactions: list) -> None:
        """Refresh every leaf for *site* from a freshly computed reaction list."""
        identifier = site_identifier(site)
        updates: dict[int, float] = {}
        for leaf in self._site_leaves[identifier]:
            self.reactions[leaf] = None
            updates[leaf] = 0.0
        for reaction in reactions:
            reaction = _admissible_reaction(reaction)
            if reaction is None:
                continue
            leaf = self.leaf_id(reaction.site, reaction.member_index)
            self.reactions[leaf] = reaction
            updates[leaf] = reaction.rate if reaction.rate > 0.0 else 0.0
        if len(updates) == self.n_total:
            self.tree.build(updates[index] for index in range(self.n_total))
        else:
            self.tree.update_many(updates.items())

    def install_sites(self, sites: list[Any] | tuple[Any, ...]) -> None:
        """Refresh several complete sites with one batched tree reduction."""
        updates: dict[int, float] = {}
        for site in sites:
            identifier = site_identifier(site)
            for leaf in self._site_leaves[identifier]:
                self.reactions[leaf] = None
                updates[leaf] = 0.0
            for reaction in (
                getattr(site, "applicable_reactions", None) or []
            ):
                reaction = _admissible_reaction(reaction)
                if reaction is None:
                    continue
                leaf = self.leaf_id(
                    reaction.site,
                    reaction.member_index,
                )
                self.reactions[leaf] = reaction
                updates[leaf] = (
                    reaction.rate if reaction.rate > 0.0 else 0.0
                )
        if len(updates) == self.n_total:
            self.tree.build(updates[index] for index in range(self.n_total))
        else:
            self.tree.update_many(updates.items())

    def total_rate(self) -> float:
        return self.tree.total

    def sample(self, u: float) -> IndexedReaction | None:
        i = self.tree.sample(u)
        if i < 0:
            return None
        return self.reactions[i]


__all__ = ["_ReactionIndex", "IndexedReaction"]
