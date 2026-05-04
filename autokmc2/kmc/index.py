"""Reaction leaf indexing for incremental KMC rate updates."""

from __future__ import annotations

from autokmc2.kmc.sampling import _RateSegmentTree
from autokmc2.reactions.adsorption import AdsorptionReaction
from autokmc2.reactions.bond import BondReaction
from autokmc2.reactions.diffusion import DiffusionReaction
from autokmc2.sites.adsorbate import AdsorbateSite
from autokmc2.sites.bond import BondReactionSite
from autokmc2.sites.diffusion import DiffusionSite

IndexedReaction = AdsorptionReaction | DiffusionReaction | BondReaction


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
    )

    def __init__(
        self,
        sites: list[AdsorbateSite],
        diffusion_sites: list[DiffusionSite] | None = None,
        bond_sites: list[BondReactionSite] | None = None,
    ):
        self.site_order = list(sites)
        self.diffusion_site_order = list(diffusion_sites or [])
        self.bond_site_order = list(bond_sites or [])
        self.base: dict[int, int] = {}
        self._adsorbate_ids: set[int] = set()
        self._diffusion_ids: set[int] = set()
        self._bond_ids: set[int] = set()

        offset = 0
        for site in self.site_order:
            self.base[id(site)] = offset
            self._adsorbate_ids.add(id(site))
            offset += len(site.member_node_ids)
        for site in self.diffusion_site_order:
            self.base[id(site)] = offset
            self._diffusion_ids.add(id(site))
            offset += len(site.member_node_ids)
        for site in self.bond_site_order:
            self.base[id(site)] = offset
            self._bond_ids.add(id(site))
            offset += len(site.member_node_ids)

        self.n_total = offset
        self.tree = _RateSegmentTree(self.n_total)
        self.reactions: list[IndexedReaction | None] = [None] * self.n_total

    def leaf_id(self, site, m_idx: int) -> int:
        return self.base[id(site)] + int(m_idx)

    def install(self, rxn, site, m_idx: int) -> None:
        i = self.leaf_id(site, m_idx)
        self.reactions[i] = rxn
        self.tree.update(i, rxn.rate if (rxn is not None and rxn.rate > 0.0) else 0.0)

    def install_site(self, site, reactions: list) -> None:
        """Refresh every leaf for *site* from a freshly computed reaction list."""
        base = self.base[id(site)]
        for offset in range(len(site.member_node_ids)):
            self.reactions[base + offset] = None
            self.tree.update(base + offset, 0.0)
        for reaction in reactions:
            self.install(reaction, reaction.site, reaction.member_index)

    def total_rate(self) -> float:
        return self.tree.total

    def sample(self, u: float) -> IndexedReaction | None:
        i = self.tree.sample(u)
        if i < 0:
            return None
        return self.reactions[i]


__all__ = ["_ReactionIndex", "IndexedReaction"]
