"""Initial species-network construction shared by configured workflows."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import networkx as nx

from autokmc.core.graph_state import set_bond_reaction_sites
from autokmc.species.smiles import (
    canonical_atom_inventory_smiles,
    reactant_atom_inventory_smiles,
)
from autokmc.workflow.models import PreparedNetwork, RunIdentity, ThermoRuntime
from autokmc.workflow.stages import (
    configured_adsorbate_site_kwargs,
    summarize_sites,
)

if TYPE_CHECKING:
    from autokmc.io.config import RunConfig


def derive_configured_bond_templates(reactant_configs, reactants, bond_cfg):
    """Derive templates while preserving each feed reactant's H policy."""
    from autokmc.reactions.templates import derive_bond_templates

    reactants = list(reactants)
    reactant_smiles = [reactant.smiles for reactant in reactants]
    inventories = {
        reactant.smiles: reactant_atom_inventory_smiles(reactant)
        for reactant in reactants
    }
    templates = []
    if bond_cfg.include_dissociation:
        for reactant_cfg, reactant in zip(reactant_configs, reactants):
            templates.extend(
                derive_bond_templates(
                    [reactant.smiles],
                    include_dissociation=True,
                    include_coupling=False,
                    bond_types=tuple(bond_cfg.bond_types),
                    include_ring_bonds=bond_cfg.include_ring_bonds,
                    add_hydrogens=reactant_cfg.add_hydrogens,
                    atom_inventory_smiles=inventories,
                )
            )
    if bond_cfg.include_coupling:
        templates.extend(
            derive_bond_templates(
                reactant_smiles,
                include_dissociation=False,
                include_coupling=True,
                include_homo_coupling=bond_cfg.include_homo_coupling,
                atom_inventory_smiles=inventories,
            )
        )

    unique = []
    seen: set[tuple[str, str, str]] = set()
    for template in templates:
        key = (template.smiles_a, template.smiles_b, template.smiles_c)
        if key not in seen:
            seen.add(key)
            unique.append(template)
    return unique


@dataclass
class SpeciesNetworkBuilder:
    """Build and register the initial adsorption/diffusion/bond network.

    Discovery and graph registration are kept in one service so the initial
    workflow and future runtime-expansion callers share the same ordering
    rules instead of reproducing them in the CLI.
    """

    cfg: RunConfig
    identity: RunIdentity
    graph: nx.Graph
    calculator_resource: Any
    frozen_indices: list[int] | None
    thermo_runtime: ThermoRuntime
    template_builder: Callable[..., list] | None = None
    verbose: bool = False

    def prepare(self, reactants: list, feed_sites: list) -> PreparedNetwork:
        if self.identity.resume_state is not None:
            state = self.identity.resume_state
            restored_sites = list(state.adsorbate_sites)
            return PreparedNetwork(
                reactants=list(state.reactants),
                adsorbate_sites=restored_sites,
                initial_adsorbate_sites=list(restored_sites),
                diffusion_sites=list(state.diffusion_sites),
                bond_sites=list(state.bond_sites),
            )

        all_reactants = list(reactants)
        all_sites = list(feed_sites)
        initial_sites = list(feed_sites)
        bond_sites = self._discover_bonds(all_reactants, all_sites)
        # Bond discovery materializes fragment/product placements as well as
        # feed placements. They must all have diffusion before KMC starts.
        diffusion_sites = self._discover_diffusion(all_sites)
        return PreparedNetwork(
            reactants=all_reactants,
            adsorbate_sites=all_sites,
            initial_adsorbate_sites=initial_sites,
            diffusion_sites=diffusion_sites,
            bond_sites=bond_sites,
        )

    def _discover_diffusion(self, sites: list) -> list:
        from autokmc.sites.diffusion import find_diffusion_sites

        settings = self.cfg.diffusion
        if not settings.enabled:
            return []
        if self.verbose:
            print("[autokmc]   diffusion enabled: enumerating hop permutations")
        by_smiles = find_diffusion_sites(
            self.graph,
            sites,
            max_hops=settings.max_hops,
            n_shells_pair=settings.n_shells_pair,
            prune_by_adsorption_pair=settings.prune_by_adsorption_pair,
            verbose=self.verbose,
        )
        result = [site for group in by_smiles.values() for site in group]
        if self.verbose:
            print(
                f"[autokmc] Diffusion enabled: {len(result)} DiffusionSite "
                f"iso-class(es) across {len(by_smiles)} SMILES."
            )
        return result

    def _discover_bonds(self, reactants: list, sites: list) -> list:
        from autokmc.kmc.expansion import initialise_bond_registry
        from autokmc.sites.bond import (
            _prune_one_per_adsorption_triple,
            find_bond_sites,
            prune_unstable_bond_sites,
            rebuild_bond_reverse_indexes,
        )

        settings = self.cfg.bond
        if not settings.enabled:
            return []

        feed_smiles = [reactant.smiles for reactant in reactants]
        template_builder = self.template_builder or derive_configured_bond_templates
        templates = template_builder(
            self.cfg.reactants,
            reactants,
            settings,
        )
        if self.verbose:
            print(
                f"[autokmc] Bond reactions enabled: derived {len(templates)} "
                f"template(s) from {len(feed_smiles)} reactant SMILES."
            )

        reactant_by_smiles = {
            canonical_atom_inventory_smiles(reactant.smiles): reactant
            for reactant in reactants
        }
        sites_by_smiles: dict[str, list] = {}
        for site in sites:
            sites_by_smiles.setdefault(
                canonical_atom_inventory_smiles(site.reactant), []
            ).append(site)

        self._build_leaf_species(
            templates,
            reactants,
            sites,
            reactant_by_smiles,
            sites_by_smiles,
        )

        bond_sites: list = []
        if templates:
            # Scientific ordering is intentional: all calculator stability
            # checks run before representatives compete in the triple prune.
            bond_sites = find_bond_sites(
                self.graph,
                sites,
                templates,
                max_hops=settings.bond_max_hops,
                deduplicate_iso=settings.deduplicate_iso,
                n_shells_pair=settings.pair_n_shells,
                prune_by_triple=False,
                gas_species=reactant_by_smiles,
                gas_lift_height=settings.gas_lift_height,
                verbose=self.verbose,
            )

            if (
                settings.prune_with_calculator
                and self.calculator_resource is not None
                and bond_sites
            ):
                species_by_smiles = {
                    canonical_atom_inventory_smiles(reactant.smiles): reactant
                    for reactant in reactants
                }
                bond_sites = prune_unstable_bond_sites(
                    self.graph,
                    bond_sites,
                    species_by_smiles,
                    self.calculator_resource,
                    frozen_indices=self.frozen_indices,
                    fmax=settings.prune_fmax,
                    max_steps=settings.prune_max_steps,
                    nl_mult=self.cfg.constants.neighbor_list_multiplier,
                    optimizer=self.cfg.optimization.optimizer,
                    optimizer_kwargs=self.cfg.optimization.optimizer_kwargs,
                    verbose=self.verbose,
                )

            if settings.prune_by_triple and bond_sites:
                bond_sites = _prune_one_per_adsorption_triple(
                    bond_sites,
                    verbose=self.verbose,
                    prefix=" (post-stability)",
                )
                for new_index, site in enumerate(bond_sites):
                    site.iso_class = new_index
                set_bond_reaction_sites(self.graph, bond_sites)
                rebuild_bond_reverse_indexes(self.graph, bond_sites)

        initialise_bond_registry(
            self.graph,
            reactants=list(reactant_by_smiles.values()),
            adsorbate_sites=sites_by_smiles,
            templates=templates,
            bond_sites=bond_sites,
            expanded_smiles=feed_smiles,
        )
        if self.verbose:
            print(
                f"[autokmc] Bond reactions: {len(bond_sites)} "
                "BondReactionSite iso-class(es) enumerated; registry seeded "
                f"with {len(reactant_by_smiles)} species."
            )
        return bond_sites

    def _build_leaf_species(
        self,
        templates: list,
        reactants: list,
        sites: list,
        reactant_by_smiles: dict,
        sites_by_smiles: dict[str, list],
    ) -> None:
        from autokmc.sites.adsorbate import find_adsorbate_sites
        from autokmc.species.reactant import build_reactant

        leaves: list[str] = []
        for template in templates:
            for smiles in (
                template.smiles_a,
                template.smiles_b,
                template.smiles_c,
            ):
                canonical = canonical_atom_inventory_smiles(smiles)
                if (
                    canonical
                    and canonical not in reactant_by_smiles
                    and canonical not in leaves
                ):
                    leaves.append(canonical)

        if leaves and not self.cfg.bond.auto_build_leaf_species:
            raise ValueError(
                "bond.auto_build_leaf_species is False but the derived "
                f"templates reference {len(leaves)} species not in `reactants`: "
                f"{sorted(leaves)}. Add them to `reactants` or set "
                "`bond.auto_build_leaf_species: true`."
            )

        site_kwargs = configured_adsorbate_site_kwargs(self.cfg)
        for canonical in leaves:
            if self.verbose:
                print(
                    f"[autokmc]   leaf species: {canonical!r} — building "
                    "Reactant + adsorbate sites…"
                )
            leaf = build_reactant(
                canonical,
                add_hydrogens=False,
                calculator=self.calculator_resource,
                nl_mult=self.cfg.constants.neighbor_list_multiplier,
                random_seed=self.cfg.kmc.random_seed,
                free_energy_options=(
                    self.thermo_runtime.options
                    if self.cfg.free_energy.enabled
                    else None
                ),
                free_energy_temperature_k=self.cfg.kmc.temperature_k,
                partial_pressure_bar=0.0,
                vib_cache_root=self.thermo_runtime.vibration_cache_root,
                optimizer=self.cfg.optimization.optimizer,
                optimizer_kwargs=self.cfg.optimization.optimizer_kwargs,
            )
            reactants.append(leaf)
            reactant_by_smiles[canonical] = leaf
            leaf_sites = find_adsorbate_sites(
                self.graph,
                leaf,
                calculator=self.calculator_resource,
                frozen_indices=self.frozen_indices,
                diagnostics_dir=str(self.identity.output_dir / "diagnostics"),
                verbose=self.verbose,
                **site_kwargs,
            )
            sites.extend(leaf_sites)
            sites_by_smiles.setdefault(canonical, []).extend(leaf_sites)
            if self.verbose:
                n_iso, n_members = summarize_sites(leaf_sites)
                print(
                    f"[autokmc]   leaf species {canonical!r}: {n_iso} stable "
                    f"adsorbate iso-class(es), {n_members} member placement(s)"
                )


__all__ = ["SpeciesNetworkBuilder", "derive_configured_bond_templates"]
