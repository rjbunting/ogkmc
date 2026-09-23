"""End-to-end characterization tests for refactor-sensitive orchestration seams."""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from ogkmc.analysis.products import analyze_run
from ogkmc.cli.pipeline import run_from_config
from ogkmc.io.calculators import CalculatorCfg
from ogkmc.io.config import (
    AdsorptionCfg,
    ConstantsCfg,
    FreeEnergyCfg,
    KMCCfg,
    OutputCfg,
    ReactantCfg,
    RunConfig,
    StructureCfg,
)
from ogkmc.io.persistence import ReactionWriter
from ogkmc.io.reaction_index import (
    load_reaction_index,
    resolve_event_definition,
)
from ogkmc.io.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    finish_run_manifest,
    start_run_manifest,
)
from ogkmc.io.schemas import EVENT_SCHEMA_VERSION
from ogkmc.kmc.engine import run_kmc_steps
from ogkmc.reactions.adsorption import AdsorptionReaction
from ogkmc.sites.adsorbate import AdsorbateSite, AdsorbateSiteLateral
from ogkmc.species.reactant import Reactant


def _single_site_system(*, occupied: bool = False):
    """Return the smallest graph/site pair accepted by the production KMC loop."""
    clique = frozenset({1})
    graph = nx.Graph()
    graph.add_node(
        1,
        type="surface",
        element="Pt",
        index=0,
        position=np.array([0.0, 0.0, 0.0]),
    )
    graph.add_node(
        10,
        type="adsorbate",
        element="O",
        index=1,
        position=np.array([0.0, 0.0, 1.5]),
        reactant="[O]",
        occupied=occupied,
        clique=clique,
    )

    lateral = AdsorbateSiteLateral(
        lateral_class=0,
        members=[0],
        energy_occupied=-1.2,
        energy_unoccupied=-1.0,
        stable=True,
    )
    site = AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[clique],
        positions=np.array([[0.0, 0.0, 1.5]]),
        iso_class=0,
        members=[[clique]],
        member_node_ids=[[10]],
        lateral_classes=[lateral],
    )
    site._member_cliques = [(clique,)]
    site._member_lc = {0: lateral}
    site._n_occupied = int(occupied)

    graph.graph.update(
        {
            "adsorbate_sites": {"[O]": [site]},
            "clique_to_members": {clique: [(site, 0)]},
            "surface_node_to_members": {1: [(site, 0)]},
            "occupied_by_clique": {clique: ({10} if occupied else set())},
            "n_occupied": int(occupied),
        }
    )
    return graph, site


def _install_deterministic_adsorption_kernel(monkeypatch):
    """Replace expensive discovery while retaining the real KMC loop/index/mutation."""
    import ogkmc.kmc.engine as engine

    def current_reaction(graph, site):
        occupied = bool(graph.nodes[site.member_node_ids[0][0]]["occupied"])
        kind = "desorption" if occupied else "adsorption"
        return AdsorptionReaction(
            kind=kind,
            site=site,
            member_index=0,
            lateral_class=site.lateral_classes[0],
            delta_e=0.2 if occupied else -0.2,
            barrier=0.05,
            rate=4.0,
        )

    def compute_all(graph, sites, *_args, **_kwargs):
        reactions = []
        for site in sites:
            reaction = current_reaction(graph, site)
            site.applicable_reactions = [reaction]
            reactions.append(reaction)
        return reactions

    def recompute(graph, sites, *_args, rxn_index=None, **_kwargs):
        reactions = []
        for site in sites:
            reaction = current_reaction(graph, site)
            site.applicable_reactions = [reaction]
            reactions.append(reaction)
            if rxn_index is not None:
                rxn_index.install_site(site, [reaction])
        return reactions, []

    monkeypatch.setattr(engine, "compute_all_reactions", compute_all)
    monkeypatch.setattr(engine, "_recompute_affected_sites", recompute)


def _run_deterministic_kmc(graph, site, *, n_steps, **kwargs):
    return run_kmc_steps(
        graph,
        [site],
        calculator=None,
        reactants={"[O]": 0.0},
        temperature=500.0,
        n_steps=n_steps,
        rng=2026,
        log_every=0,
        verbose=False,
        lateral_interactions=False,
        **kwargs,
    )


def test_run_kmc_steps_fixed_seed_characterizes_event_sequence(monkeypatch):
    _install_deterministic_adsorption_kernel(monkeypatch)
    graph, site = _single_site_system()

    result = _run_deterministic_kmc(graph, site, n_steps=6)

    assert result["steps_executed"] == 6
    assert [row[0] for row in result["history"]] == [1, 2, 3, 4, 5, 6]
    assert [row[2] for row in result["history"]] == [
        "adsorption",
        "desorption",
        "adsorption",
        "desorption",
        "adsorption",
        "desorption",
    ]
    assert result["time"] == pytest.approx(1.174298161340483)
    assert result["reaction_counts"] == {
        "adsorption": 3,
        "desorption": 3,
        "diffusion": 0,
        "bond": 0,
        "bond_couple": 0,
        "bond_dissoc": 0,
    }
    assert result["final_occupancy"] == {"[O]:iso0": 0}
    assert graph.nodes[10]["occupied"] is False


class _CheckpointCapture:
    def __init__(self):
        self.last = None

    def maybe_write(self, **payload):
        self.last = payload
        return None


def test_run_kmc_steps_resume_matches_uninterrupted_rng_and_result(monkeypatch):
    _install_deterministic_adsorption_kernel(monkeypatch)
    full_graph, full_site = _single_site_system()
    full = _run_deterministic_kmc(full_graph, full_site, n_steps=6)

    split_graph, split_site = _single_site_system()
    checkpoint = _CheckpointCapture()
    first = _run_deterministic_kmc(
        split_graph,
        split_site,
        n_steps=3,
        checkpoint_writer=checkpoint,
    )
    assert checkpoint.last is not None

    resumed = _run_deterministic_kmc(
        split_graph,
        split_site,
        n_steps=3,
        initial_step=3,
        initial_time_s=first["time"],
        initial_history=first["history"],
        initial_reaction_counts=first["reaction_counts"],
        initial_rng_state=checkpoint.last["rng_state"],
    )

    assert resumed["steps_executed"] == 3
    assert resumed["time"] == full["time"]
    assert resumed["history"] == full["history"]
    assert resumed["reaction_counts"] == full["reaction_counts"]
    assert resumed["final_occupancy"] == full["final_occupancy"]
    assert split_graph.nodes[10]["occupied"] == full_graph.nodes[10]["occupied"]


def test_fresh_run_from_config_orchestrates_all_nonoptional_stages(
    tmp_path,
    monkeypatch,
):
    import ogkmc.core.graph as graph_module
    import ogkmc.cli.pipeline as pipeline_module
    import ogkmc.kmc.engine as engine_module
    import ogkmc.sites.adsorbate as adsorbate_module
    import ogkmc.species.reactant as reactant_module
    import ogkmc.structure as structure_module

    calls = []
    built_graph = nx.Graph()
    built_site = None

    def build_surface(**kwargs):
        calls.append("build_surface")
        assert kwargs["composition"] == "Pt"
        assert kwargs["composition_seed"] == 19
        assert kwargs["surface_radius_factor"] == pytest.approx(1.13)
        assert kwargs["raycast_coverage_threshold"] == pytest.approx(0.61)
        assert kwargs["raycast_disc_samples"] == 12
        atoms = Atoms(
            "Pt",
            positions=[[0.0, 0.0, 0.0]],
            cell=[5.0, 5.0, 10.0],
            pbc=[True, True, False],
        )
        atoms.info["frozen_indices"] = []
        return atoms

    def find_surface_atoms(atoms, **kwargs):
        calls.append("find_surface_atoms")
        assert kwargs == {
            "nl_mult": pytest.approx(0.77),
            "surf_radius_factor": pytest.approx(1.13),
            "coverage_threshold": pytest.approx(0.61),
            "n_disc_sample": 12,
            "which": "both",
            "hull_tol_factor": pytest.approx(0.44),
            "tag_atoms": True,
        }
        atoms.arrays["surface"] = np.ones(len(atoms), dtype=np.int8)
        return SimpleNamespace(indices=[0], method="characterization")

    def build_graph(atoms, *, nl_mult):
        calls.append("build_graph")
        assert nl_mult == pytest.approx(0.77)
        built_graph.add_node(
            0,
            type="surface",
            element="Pt",
            index=0,
            position=np.array(atoms.positions[0]),
        )
        return built_graph

    def build_reactant(smiles, **kwargs):
        calls.append("build_reactant")
        assert kwargs["relax"] is False
        assert kwargs["fmax"] == pytest.approx(0.012)
        assert kwargs["steps"] == 77
        assert kwargs["nl_mult"] == pytest.approx(0.77)
        assert kwargs["random_seed"] == 19
        return Reactant(
            smiles=smiles,
            atoms=Atoms("O", positions=[[0.0, 0.0, 0.0]]),
            graph=nx.Graph(),
            energy=-1.0,
            partial_pressure_bar=kwargs["partial_pressure_bar"],
        )

    def find_adsorbate_sites(graph, reactant, **kwargs):
        nonlocal built_site
        calls.append("find_adsorbate_sites")
        assert kwargs["prune_stable_only"] is False
        assert kwargs["prune_fmax"] == pytest.approx(0.023)
        assert kwargs["prune_max_steps"] == 88
        assert kwargs["anchor_k_max"] == 4
        assert {
            key: kwargs[key]
            for key in (
                "bond_tolerance",
                "n_shells_anchor",
                "n_shells_pair",
                "co_factor",
                "opt_factor",
                "repulsion_weight",
                "repulsion_cutoff",
                "contact_factor",
                "standoff_factor",
                "n_restarts",
                "nn_distance",
                "max_pair_shells",
                "hull_tolerance",
                "kabsch_max_mappings",
                "nl_mult",
            )
        } == {
            "bond_tolerance": pytest.approx(0.25),
            "n_shells_anchor": 2,
            "n_shells_pair": 3,
            "co_factor": pytest.approx(0.83),
            "opt_factor": pytest.approx(0.81),
            "repulsion_weight": pytest.approx(0.31),
            "repulsion_cutoff": pytest.approx(7.5),
            "contact_factor": pytest.approx(1.11),
            "standoff_factor": pytest.approx(0.15),
            "n_restarts": 4,
            "nn_distance": pytest.approx(2.7),
            "max_pair_shells": 6,
            "hull_tolerance": pytest.approx(-0.15),
            "kabsch_max_mappings": 321,
            "nl_mult": pytest.approx(0.77),
        }
        clique = frozenset({0})
        graph.add_node(
            10,
            type="adsorbate",
            element="O",
            reactant=reactant.smiles,
            occupied=False,
            clique=clique,
        )
        built_site = AdsorbateSite(
            reactant=reactant.smiles,
            n_atoms=1,
            atom_cliques=[clique],
            positions=np.array([[0.0, 0.0, 1.5]]),
            iso_class=0,
            members=[[clique]],
            member_node_ids=[[10]],
        )
        return [built_site]

    def run_steps(graph, sites, calculator, reactants, **kwargs):
        calls.append("run_kmc_steps")
        assert graph is built_graph
        assert sites == [built_site]
        assert len(reactants) == 1
        assert kwargs["n_steps"] == 2
        assert kwargs["fmax"] == pytest.approx(0.034)
        assert kwargs["max_steps"] == 99
        assert kwargs["diffusion_sites"] == []
        assert kwargs["bond_sites"] is None
        assert kwargs["calculation_cache_root"] is None
        assert kwargs["lateral_shells"] == 2
        assert "progress" not in kwargs
        assert kwargs["reaction_writer"].run_id == graph.graph["run_id"]
        return {
            "time": 0.25,
            "steps_executed": 2,
            "history": [],
            "reaction_counts": {"adsorption": 1, "desorption": 1},
            "final_occupancy": {"[O]:iso0": 0},
            "performance": {
                "counters": {"kmc.session.runs": 1},
                "timings_s": {"kmc.session.seconds": 0.25},
                "gauges": {"kmc.last_step": 2.0},
            },
        }

    monkeypatch.setattr(structure_module, "build_surface", build_surface)
    monkeypatch.setattr(structure_module, "find_surface_atoms", find_surface_atoms)
    monkeypatch.setattr(graph_module, "build_graph", build_graph)
    monkeypatch.setattr(
        pipeline_module,
        "_resolved_partial_pressure_bar",
        lambda *_args: 0.25,
    )
    monkeypatch.setattr(reactant_module, "build_reactant", build_reactant)
    monkeypatch.setattr(adsorbate_module, "find_adsorbate_sites", find_adsorbate_sites)
    monkeypatch.setattr(engine_module, "run_kmc_steps", run_steps)

    cfg = RunConfig(
        output=OutputCfg(
            dir=str(tmp_path / "fresh-run"),
            trajectory_dump_every=0,
            calculation_cache_enabled=False,
            log_level="WARNING",
        ),
        constants=ConstantsCfg(
            neighbor_list_multiplier=0.77,
            co_bond_factor=0.83,
            anchor_bond_factor=0.81,
            anchor_repulsion_weight=0.31,
            site_repulsion_cutoff=7.5,
            adsorbate_contact_factor=1.11,
            adsorbate_standoff_factor=0.15,
            adsorbate_rotational_restarts=4,
            typical_neighbor_distance=2.7,
            adsorbate_bond_tolerance=0.25,
            anchor_hull_tolerance=-0.15,
            raycast_coverage_threshold=0.61,
            raycast_disc_samples=12,
            kabsch_max_mappings=321,
            lateral_shells=2,
        ),
        structure=StructureCfg(
            kind="surface",
            composition="Pt",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            lattice_constant=None,
            min_slab_size=5.0,
            min_vacuum_size=5.0,
            goal_x=5.0,
            goal_y=5.0,
            n_freeze_layers=0,
            surface_side="both",
            surface_radius_factor=1.13,
            nanoparticle_hull_tolerance_factor=0.44,
            fmax=0.05,
            max_steps=5,
            n_atoms=None,
            surface_energies=None,
            surface_energy_facets=((1, 1, 1),),
            surface_energy_layers=1,
            surface_energy_vacuum=5.0,
            surface_energy_fmax=None,
            surface_energy_max_steps=None,
            extra_kwargs={},
        ),
        reactants=[
            ReactantCfg(
                smiles="[O]",
                add_hydrogens=False,
                relax_in_gas=False,
                fmax=0.012,
                max_steps=77,
                partial_pressure_bar=None,
            )
        ],
        calculator=CalculatorCfg(import_path="ase.calculators.emt.EMT"),
        adsorption=AdsorptionCfg(
            prune_stable_only=False,
            prune_fmax=0.023,
            prune_max_steps=88,
            endpoint_fmax=0.034,
            endpoint_max_steps=99,
            n_shells_anchor=2,
            pair_n_shells=3,
            max_pair_shells=6,
        ),
        kmc=KMCCfg(n_steps=2, random_seed=19, log_every=0),
        free_energy=FreeEnergyCfg(enabled=False),
    )

    result = run_from_config(cfg)

    assert calls == [
        "build_surface",
        "find_surface_atoms",
        "build_graph",
        "build_reactant",
        "find_adsorbate_sites",
        "run_kmc_steps",
    ]
    assert result["steps_executed"] == 2
    assert result["outputs"]["calculation_cache"] is None
    assert result["outputs"]["trajectory"] is None
    assert result["outputs"]["n_unique_reactions"] == 0

    manifest = json.loads((tmp_path / "fresh-run" / "run_manifest.json").read_text())
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert manifest["result"]["status"] == "complete"
    assert manifest["result"]["termination_reason"] == (
        "requested_steps_completed"
    )
    assert manifest["result"]["termination_stage"] == "stage_7_kmc"
    assert manifest["result"]["final_step"] == 2
    assert manifest["result"]["final_time_s"] == 0.25
    assert manifest["result"]["simulated_time_s"] == 0.25
    assert manifest["result"]["steps_executed"] == 2
    assert manifest["result"]["wall_time_s"] >= 0.0
    assert manifest["lifecycle"]["status"] == "complete"
    assert manifest["segments"][0]["ended_utc"]
    assert manifest["outputs"] == result["outputs"]
    assert manifest["artifacts"]["events"]["status"] == "complete"
    assert manifest["artifacts"]["events"]["sha256"]
    assert manifest["artifacts"]["run_manifest"]["status"] == "partial"
    assert manifest["artifacts"]["isaac_records"]["status"] == "disabled"
    assert manifest["run_id"] == built_graph.graph["run_id"]
    assert manifest["feed_reactants"] == [
        {
            "input_smiles": "[O]",
            "partial_pressure_bar": 0.25,
            "species": "[O]",
        }
    ]
    persisted_summary = json.loads(
        (tmp_path / "fresh-run" / "summary.json").read_text()
    )
    assert persisted_summary["run"]["performance"] == result["performance"]
    raw_performance = json.loads(
        (
            tmp_path / "fresh-run" / "diagnostics" / "performance.json"
        ).read_text()
    )
    assert raw_performance["summary"] == result["performance"]
    assert (tmp_path / "fresh-run" / "events.jsonl").is_file()


def _placement(species: str, node_id: int, surface_id: int):
    clique = frozenset({surface_id})
    return SimpleNamespace(
        reactant=species,
        iso_class=0,
        member_node_ids=[[node_id]],
        members=[[clique]],
        _member_cliques=[(clique,)],
    )


def _reaction(kind, site, lateral, *, direction=None, delta_e=0.0, barrier=0.1):
    return SimpleNamespace(
        kind=kind,
        direction=direction,
        site=site,
        member_index=0,
        lateral_class=lateral,
        delta_e=delta_e,
        barrier=barrier,
        rate=10.0,
    )


def test_reaction_writer_events_are_consumed_by_offline_analysis(tmp_path):
    run_id = "characterization-contract-run"
    graph = nx.Graph()
    for node in range(5):
        graph.add_node(node, type="surface", element="Pt")

    start_run_manifest(
        tmp_path / "run_manifest.json",
        graph=graph,
        adsorbate_sites=[],
        feed_reactants=[
            {"smiles": "[C]=O", "partial_pressure_bar": 1.0},
            {"smiles": "O=O", "partial_pressure_bar": 0.2},
        ],
        temperature_k=500.0,
        random_seed=7,
        structure_kind="surface",
        composition="Pt",
        config_path=None,
        run_id=run_id,
    )

    co = _placement("[C]=O", 10, 0)
    oxygen_molecule = _placement("O=O", 20, 1)
    oxygen_a = _placement("[O]", 30, 2)
    oxygen_b = _placement("[O]", 31, 3)
    oxygen_after_hop = _placement("[O]", 32, 4)
    carbon_dioxide = _placement("O=C=O", 40, 2)

    ads_lateral = SimpleNamespace(
        lateral_class=0,
        energy_occupied=-1.0,
        energy_unoccupied=0.0,
    )
    diffusion_lateral = SimpleNamespace(
        lateral_class=0,
        energy_a=-1.0,
        energy_b=-1.0,
        energy_ts=-0.8,
    )
    dissociation_template = SimpleNamespace(
        smiles_a="[O]",
        smiles_b="[O]",
        smiles_c="O=O",
        bond_type="DOUBLE",
    )
    formation_template = SimpleNamespace(
        smiles_a="[C]=O",
        smiles_b="[O]",
        smiles_c="O=C=O",
        bond_type="DOUBLE",
    )
    bond_lateral = SimpleNamespace(
        lateral_class=0,
        energy_ab=-2.0,
        energy_c=-3.0,
        energy_ts=-1.5,
    )

    dissociation_site = SimpleNamespace(
        iso_class=0,
        template=dissociation_template,
        gas_product=False,
        members=[(oxygen_a, 0, oxygen_b, 0, oxygen_molecule, 0)],
        member_node_ids=[[[30], [31], [20]]],
    )
    diffusion_site = SimpleNamespace(
        iso_class=0,
        reactant="[O]",
        members=[(oxygen_a, 0, oxygen_after_hop, 0)],
        member_node_ids=[[[30], [32]]],
    )
    formation_site = SimpleNamespace(
        iso_class=0,
        template=formation_template,
        gas_product=False,
        members=[(co, 0, oxygen_after_hop, 0, carbon_dioxide, 0)],
        member_node_ids=[[[10], [32], [40]]],
    )

    events = [
        _reaction("adsorption", co, ads_lateral, delta_e=-1.0),
        _reaction("adsorption", oxygen_molecule, ads_lateral, delta_e=-1.0),
        _reaction(
            "bond",
            dissociation_site,
            bond_lateral,
            direction="dissoc",
            delta_e=1.0,
            barrier=1.5,
        ),
        _reaction(
            "diffusion",
            diffusion_site,
            diffusion_lateral,
            direction="a_to_b",
            barrier=0.2,
        ),
        _reaction(
            "bond",
            formation_site,
            bond_lateral,
            direction="couple",
            delta_e=-1.0,
            barrier=0.5,
        ),
        _reaction("desorption", carbon_dioxide, ads_lateral, delta_e=1.0),
    ]

    writer = ReactionWriter(tmp_path, run_id=run_id)
    for step, event in enumerate(events, start=1):
        writer.record(
            step=step,
            time_s=float(step),
            tau_s=1.0,
            reaction=event,
        )
    writer.close()
    finish_run_manifest(
        tmp_path / "run_manifest.json",
        final_step=6,
        final_time_s=10.0,
        steps_executed=6,
    )

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(row["schema_version"] == EVENT_SCHEMA_VERSION for row in rows)
    assert all(row["run_id"] == run_id for row in rows)
    assert all(row["event_id"].startswith("event-") for row in rows)
    assert all(row["reaction_id"].startswith("reaction-") for row in rows)
    assert all(
        {
            "description",
            "reaction_dir",
            "template",
            "gas_product",
        }.isdisjoint(row)
        for row in rows
    )
    assert [row["kind"] for row in rows] == [
        "adsorption",
        "adsorption",
        "bond",
        "diffusion",
        "bond",
        "desorption",
    ]

    result = analyze_run(tmp_path, n_blocks=2)

    assert result["products"] == [
        pytest.approx(
            {
                "product": "O=C=O",
                "count": 1,
                "start_time_s": 0.0,
                "end_time_s": 10.0,
                "duration_s": 10.0,
                "rate_hz": 0.1,
                "rate_ci95_low_hz": 0.002531780798428988,
                "rate_ci95_high_hz": 0.5571643390938898,
                "tof_per_surface_atom_s-1": 0.02,
            }
        )
    ]
    product_event = json.loads((tmp_path / "analysis" / "product_events.jsonl").read_text())
    assert product_event["step"] == rows[-1]["step"]
    assert product_event["time_s"] == rows[-1]["time_s"]
    assert product_event["source_event_id"] == rows[-1]["event_id"]
    definitions = load_reaction_index(tmp_path / "reactions" / "index.jsonl")
    resolved_last = resolve_event_definition(
        rows[-1],
        definitions,
        require_definition=True,
    )
    assert resolved_last["reaction_dir"].startswith("reactions/adsorption/")
    assert product_event["mechanism"][-1] == "O=C=O* → O=C=O(g)"
    assert any("O=O* → [O]* + [O]*" in step for step in product_event["mechanism"])
    assert any("[C]=O* + [O]* → O=C=O*" in step for step in product_event["mechanism"])
