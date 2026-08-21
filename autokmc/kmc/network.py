"""Dynamic reaction-network expansion after bond events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autokmc.core.graph_state import get_bond_registry, get_diffusion_sites
from autokmc.kmc.models import (
    KMCChannels,
    KMCFunctions,
    KMCRuntime,
    KMCSettings,
    KMCSystem,
    KMCThermochemistry,
)
from autokmc.reactions.bond import compute_all_bond_reactions
from autokmc.reactions.diffusion import compute_all_diffusions
from autokmc.sites.adsorbate import AdsorbateSite
from autokmc.sites.diffusion import DiffusionSite
from autokmc.sites.identity import SiteId, site_identifier
from autokmc.utils.telemetry import increment, instrument, set_gauge


@dataclass
class ExpansionChanges:
    """New objects whose output records must be materialised this step."""

    reactions: list = field(default_factory=list)
    invalid_diffusion_sites: list[DiffusionSite] = field(default_factory=list)


class DynamicNetworkExpander:
    """Activate species and reaction channels created by one bond event."""

    def __init__(
        self,
        system: KMCSystem,
        settings: KMCSettings,
        channels: KMCChannels,
        thermochemistry: KMCThermochemistry,
        functions: KMCFunctions,
        runtime: KMCRuntime,
    ) -> None:
        self.system = system
        self.settings = settings
        self.channels = channels
        self.thermochemistry = thermochemistry
        self.functions = functions
        self.runtime = runtime

    @instrument("kmc.expansion")
    def expand(self, reaction) -> ExpansionChanges:
        """Expand all channels affected by a just-executed bond reaction."""
        changes = ExpansionChanges()
        self._announce_event(reaction)

        registry_before = get_bond_registry(self.system.graph)
        produced_species = self._produced_species(reaction)
        expanded_species = registry_before.get("expanded_species", set())
        if (
            getattr(reaction, "kind", None) == "bond"
            and produced_species
            and produced_species.issubset(expanded_species)
        ):
            increment("kmc.expansion.already_expanded")
            return changes

        active_adsorbate_ids = self.runtime.reaction_index._adsorbate_ids
        active_diffusion_ids = self.runtime.reaction_index._diffusion_ids
        registered_adsorbate_ids = {
            site_identifier(site)
            for site_list in registry_before.get("adsorbate_sites", {}).values()
            for site in site_list
        }

        growth_kwargs = self.channels.bond_growth_options.to_kwargs()
        growth_kwargs["verbose"] = self.settings.verbose
        if growth_kwargs.get("frozen_indices") is None:
            growth_kwargs["frozen_indices"] = self.settings.frozen_indices
        growth_kwargs.setdefault(
            "free_energy_options",
            self.thermochemistry.free_energy_options,
        )
        growth_kwargs.setdefault(
            "free_energy_temperature_k",
            (
                float(self.thermochemistry.free_energy_temperature_k)
                if self.thermochemistry.free_energy_temperature_k is not None
                else float(self.settings.temperature)
            ),
        )
        growth_kwargs.setdefault("vib_cache_root", self.thermochemistry.vib_cache_root)
        new_bond_sites = self.functions.expand_bond_network(
            self.system.graph,
            reaction,
            calculator=self.system.calculator,
            **growth_kwargs,
        )

        new_adsorbate_sites = self._new_adsorbate_sites(
            active_adsorbate_ids,
            registered_adsorbate_ids,
            produced_species,
        )
        if new_adsorbate_sites:
            self._compute_new_adsorption(new_adsorbate_sites)
            self.system.adsorbate_sites.extend(new_adsorbate_sites)
            changes.reactions.extend(
                candidate
                for site in new_adsorbate_sites
                for candidate in (getattr(site, "applicable_reactions", None) or [])
            )

        new_diffusion_sites = self._new_diffusion_sites(active_diffusion_ids)
        if new_diffusion_sites:
            self._compute_new_diffusion(new_diffusion_sites)
            self.channels.diffusion_sites.extend(new_diffusion_sites)
            changes.reactions.extend(
                candidate
                for site in new_diffusion_sites
                for candidate in (getattr(site, "applicable_reactions", None) or [])
            )
            changes.invalid_diffusion_sites.extend(new_diffusion_sites)

        unseen_bond_sites = []
        active_bond_ids = self.runtime.reaction_index._bond_ids
        new_bond_ids: set[SiteId] = set()
        for site in new_bond_sites:
            identifier = site_identifier(site)
            if (
                identifier not in active_bond_ids
                and identifier not in new_bond_ids
            ):
                new_bond_ids.add(identifier)
                unseen_bond_sites.append(site)
        new_bond_sites = unseen_bond_sites
        self.channels.bond_sites.extend(new_bond_sites)
        self._announce_new_bond_sites(new_bond_sites)

        if new_bond_sites or new_adsorbate_sites or new_diffusion_sites:
            self._extend_index(
                new_adsorbate_sites,
                new_diffusion_sites,
                new_bond_sites,
                changes,
            )
        return changes

    def _announce_event(self, reaction) -> None:
        if not self.settings.verbose:
            return
        template = getattr(getattr(reaction, "site", None), "template", None)
        direction = getattr(reaction, "direction", "")
        if template is None:
            return
        if direction == "couple":
            print(
                "[KMC] Bond coupling fired: "
                f"{template.smiles_a} + {template.smiles_b} → {template.smiles_c}"
                f"  (iso={reaction.site.iso_class} m={reaction.member_index})"
            )
            print(
                f"[KMC]  Formed {template.smiles_c!r}; checking for "
                "new adsorbate, diffusion, and bond reaction series."
            )
        else:
            print(
                "[KMC] Bond dissociation fired: "
                f"{template.smiles_c} → {template.smiles_a} + {template.smiles_b}"
                f"  (iso={reaction.site.iso_class} m={reaction.member_index})"
            )
            print(
                f"[KMC]  Formed fragments {template.smiles_a!r} and "
                f"{template.smiles_b!r}; checking for new reaction series."
            )

    @staticmethod
    def _produced_species(reaction) -> set[str]:
        template = getattr(getattr(reaction, "site", None), "template", None)
        direction = getattr(reaction, "direction", None)
        if template is not None and direction == "couple":
            produced = {getattr(template, "smiles_c", "")}
        elif template is not None and direction == "dissoc":
            produced = {
                getattr(template, "smiles_a", ""),
                getattr(template, "smiles_b", ""),
            }
        else:
            produced = set()
        produced.discard("")
        return produced

    def _new_adsorbate_sites(
        self,
        active_ids: set[SiteId],
        registered_before: set[SiteId],
        produced_species: set[str],
    ) -> list[AdsorbateSite]:
        registry = get_bond_registry(self.system.graph)
        new_sites: list[AdsorbateSite] = []
        new_ids: set[SiteId] = set()
        for smiles, sites in registry.get("adsorbate_sites", {}).items():
            for site in sites:
                identifier = site_identifier(site)
                if identifier in active_ids or identifier in new_ids:
                    continue
                if identifier not in registered_before or smiles in produced_species:
                    new_sites.append(site)
                    new_ids.add(identifier)

        if new_sites and self.settings.verbose:
            species = sorted({site.reactant for site in new_sites})
            print(
                f"[KMC] New adsorbate iso-class(es) discovered: "
                f"{len(new_sites)} class(es) across species {species}"
            )
            for smiles in species:
                count = sum(site.reactant == smiles for site in new_sites)
                members = sum(
                    len(site.member_node_ids)
                    for site in new_sites
                    if site.reactant == smiles
                )
                print(
                    f"[KMC]    {smiles!r:>12}  "
                    f"{count} iso-class(es)  │  {members} member(s)"
                )
        return new_sites

    def _extend_gas_lookups(self, sites: list[AdsorbateSite]) -> None:
        registry_species = get_bond_registry(self.system.graph).get(
            "species",
            {},
        )
        for site in sites:
            reactant = registry_species.get(site.reactant) if site.reactant else None
            if reactant is None:
                continue
            smiles = getattr(reactant, "smiles", site.reactant)
            if smiles not in self.runtime.gas_energies:
                energy = getattr(reactant, "energy", float("nan"))
                if not np.isnan(energy):
                    self.runtime.gas_energies[smiles] = float(energy)
            if smiles not in self.runtime.partial_pressures:
                self.runtime.partial_pressures[smiles] = float(
                    getattr(reactant, "partial_pressure_bar", 0.0)
                )
            if smiles not in self.runtime.gas_free_energies:
                free_energy = getattr(reactant, "gibbs_energy", float("nan"))
                if not np.isnan(free_energy):
                    self.runtime.gas_free_energies[smiles] = float(free_energy)

    def _compute_new_adsorption(self, sites: list[AdsorbateSite]) -> None:
        self._extend_gas_lookups(sites)
        if self.settings.verbose:
            members = sum(len(site.member_node_ids) for site in sites)
            print(
                "[KMC]    Running adsorption/desorption stability "
                f"sweep for {members} new placement(s)."
            )
        self.functions.compute_adsorption(
            self.system.graph,
            sites,
            self.system.calculator,
            self.runtime.gas_energies,
            temperature=self.settings.temperature,
            transmission_coefficient=self.settings.transmission_coefficient,
            frozen_indices=self.settings.frozen_indices,
            fmax=self.channels.adsorption_options.fmax,
            max_steps=self.channels.adsorption_options.max_steps,
            optimizer=self.settings.optimizer,
            optimizer_kwargs=self.settings.optimizer_kwargs,
            verbose=self.settings.verbose,
            lateral_interactions=self.settings.lateral_interactions,
            lateral_shells=self.settings.lateral_shells,
            gas_g=self.runtime.gas_free_energies,
            partial_pressures=self.runtime.partial_pressures,
            free_energy_options=self.thermochemistry.free_energy_options,
            vib_cache_root=self.thermochemistry.vib_cache_root,
            calculation_cache_root=self.thermochemistry.calculation_cache_root,
            calculation_cache_lookup_enabled=bool(
                self.thermochemistry.calculation_cache_lookup_enabled
            ),
        )
        if self.settings.verbose:
            count = sum(
                len(getattr(site, "applicable_reactions", None) or [])
                for site in sites
            )
            print(
                f"[KMC]    → {count} applicable adsorption/desorption "
                "reaction(s) for new species"
            )

    def _new_diffusion_sites(self, active_ids: set[SiteId]) -> list[DiffusionSite]:
        graph_sites = get_diffusion_sites(self.system.graph)
        new_sites: list[DiffusionSite] = []
        new_ids: set[SiteId] = set()
        for site_list in (
            graph_sites.values() if isinstance(graph_sites, dict) else []
        ):
            if not isinstance(site_list, list):
                continue
            for site in site_list:
                identifier = site_identifier(site)
                if identifier in active_ids or identifier in new_ids:
                    continue
                new_ids.add(identifier)
                new_sites.append(site)
        if new_sites and self.settings.verbose:
            print(
                f"[KMC] New diffusion iso-class(es) discovered: "
                f"{len(new_sites)} site-pair(s)"
            )
        return new_sites

    def _compute_new_diffusion(self, sites: list[DiffusionSite]) -> None:
        if self.settings.verbose:
            members = sum(len(site.member_node_ids) for site in sites)
            print(
                "[KMC]    Running diffusion applicability and NEB "
                f"sweep for {members} new hop member(s)."
            )
        compute_all_diffusions(
            self.system.graph,
            sites,
            self.system.calculator,
            temperature=self.settings.temperature,
            transmission_coefficient=self.settings.transmission_coefficient,
            frozen_indices=self.settings.frozen_indices,
            verbose=self.settings.verbose,
            lateral_interactions=self.settings.lateral_interactions,
            lateral_shells=self.settings.lateral_shells,
            free_energy_options=self.thermochemistry.free_energy_options,
            vib_cache_root=self.thermochemistry.vib_cache_root,
            calculation_cache_root=self.thermochemistry.calculation_cache_root,
            calculation_cache_lookup_enabled=bool(
                self.thermochemistry.calculation_cache_lookup_enabled
            ),
            **self.channels.diffusion_options.to_kwargs(),
        )
        if self.settings.verbose:
            count = sum(
                len(getattr(site, "applicable_reactions", None) or [])
                for site in sites
            )
            print(
                f"[KMC]    → {count} applicable diffusion reaction(s) "
                "for new species"
            )

    def _announce_new_bond_sites(self, sites: list) -> None:
        if not (self.settings.verbose and sites):
            return
        print(
            f"[KMC] New bond-reaction iso-class(es) enumerated: "
            f"{len(sites)} iso-class(es)  "
            f"(total bond iso-classes now: {len(self.channels.bond_sites)})"
        )
        for site in sites:
            template = site.template
            print(
                f"[KMC]    bond_iso {site.iso_class:>3}  "
                f"{template.smiles_a!r}+{template.smiles_b!r}⇌{template.smiles_c!r}"
                f"  source={template.source}"
                f"  members={len(site.member_node_ids)}"
            )

    def _extend_index(
        self,
        new_adsorbate_sites: list[AdsorbateSite],
        new_diffusion_sites: list[DiffusionSite],
        new_bond_sites: list,
        changes: ExpansionChanges,
    ) -> None:
        if new_bond_sites:
            if self.settings.verbose:
                members = sum(len(site.member_node_ids) for site in new_bond_sites)
                print(
                    "[KMC]    Running bond reaction stability/NEB "
                    f"sweep for {members} new member(s)."
                )
            compute_all_bond_reactions(
                self.system.graph,
                new_bond_sites,
                self.system.calculator,
                temperature=self.settings.temperature,
                transmission_coefficient=self.settings.transmission_coefficient,
                frozen_indices=self.settings.frozen_indices,
                verbose=self.settings.verbose,
                lateral_interactions=self.settings.lateral_interactions,
                lateral_shells=self.settings.lateral_shells,
                calculation_cache_root=self.thermochemistry.calculation_cache_root,
                calculation_cache_lookup_enabled=bool(
                    self.thermochemistry.calculation_cache_lookup_enabled
                ),
                free_energy_options=self.thermochemistry.free_energy_options,
                vib_cache_root=self.thermochemistry.vib_cache_root,
                **self.channels.bond_options.to_kwargs(),
            )
            changes.reactions.extend(
                reaction
                for site in new_bond_sites
                for reaction in (getattr(site, "applicable_reactions", None) or [])
            )
        index = self.runtime.reaction_index
        leaves_added = index.extend_sites(
            adsorbate_sites=new_adsorbate_sites,
            diffusion_sites=new_diffusion_sites,
            bond_sites=new_bond_sites,
        )
        new_sites = (
            *new_adsorbate_sites,
            *new_diffusion_sites,
            *new_bond_sites,
        )
        index.install_sites(new_sites)
        increment("kmc.index.expansions")
        increment("kmc.index.leaves_added", leaves_added)
        set_gauge("kmc.index.leaves", index.n_total)

        if self.settings.verbose:
            n_active = sum(reaction is not None for reaction in index.reactions)
            print(
                "[KMC]  Segment-tree extended"
                f"  │  {index.n_total} leaves"
                f"  │  {n_active} active reaction(s)"
                f"  │  Q = {index.total_rate():.3e} Hz"
                "  │  channels: "
                f"ads={len(self.system.adsorbate_sites)}"
                f" diff={len(self.channels.diffusion_sites)}"
                f" bond={len(self.channels.bond_sites)}"
            )


__all__ = ["DynamicNetworkExpander", "ExpansionChanges"]
