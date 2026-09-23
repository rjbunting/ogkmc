"""Regression tests for stable identities and the typed KMC boundary."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import ogkmc.kmc.initialization as initialization_module
import ogkmc.kmc.session as session_module
from ogkmc.kmc.engine import run_kmc
from ogkmc.kmc.index import _ReactionIndex
from ogkmc.kmc.models import (
    KMCChannels,
    KMCFunctions,
    KMCObservers,
    KMCResumeState,
    KMCRunRequest,
    KMCRunResult,
    KMCSettings,
    KMCSystem,
)
from ogkmc.kmc.outputs import KMCOutputManager
from ogkmc.sites.adsorbate import AdsorbateSite
from ogkmc.sites.identity import (
    member_identifier,
    member_signature,
    site_identifier,
)
from ogkmc.workflow.models import KMCResumeState as WorkflowResumeState


def _site(*, iso_class: int, member_nodes: list[list[int]]) -> AdsorbateSite:
    clique = frozenset({iso_class})
    return AdsorbateSite(
        reactant="[O]",
        n_atoms=1,
        atom_cliques=[clique],
        positions=np.array([[0.0, 0.0, 1.0]]),
        iso_class=iso_class,
        members=[[clique] for _ in member_nodes],
        member_node_ids=member_nodes,
    )


def test_site_and_member_identifiers_survive_reconstruction_and_copy():
    site = _site(iso_class=3, member_nodes=[[10], [11]])
    reconstructed = _site(iso_class=3, member_nodes=[[10], [11]])

    identifier = site_identifier(site)

    assert identifier == site.site_id
    assert site_identifier(reconstructed) == identifier
    assert site_identifier(deepcopy(site)) == identifier
    assert member_identifier(reconstructed, 1) == (
        identifier,
        member_signature(reconstructed, 1),
    )


def test_member_identifiers_are_materialised_once_per_site(monkeypatch):
    import ogkmc.sites.identity as identity_module

    site = _site(iso_class=3, member_nodes=[[10], [11]])
    expected = member_identifier(site, 1)

    def unexpected_rehash(*_args, **_kwargs):
        raise AssertionError("cached member identifier was rehashed")

    monkeypatch.setattr(
        identity_module,
        "_build_member_signature",
        unexpected_rehash,
    )

    assert member_identifier(site, 1) == expected
    assert member_signature(site, 0).startswith("member:")


def test_reaction_index_extends_without_reinstalling_existing_reactions():
    first = _site(iso_class=0, member_nodes=[[10]])
    index = _ReactionIndex([first])
    existing = SimpleNamespace(
        rate=4.25,
        site=first,
        member_index=0,
    )
    index.install(existing, first, 0)  # type: ignore[arg-type]

    added = _site(iso_class=1, member_nodes=[[20], [21], [22], [23]])
    leaves_added = index.extend_sites(adsorbate_sites=[added])

    assert leaves_added == 4
    assert index.n_total == 5
    assert index.reactions[0] is existing
    assert index.total_rate() == pytest.approx(4.25)
    assert index.leaf_id(added, 3) == 4


def test_reaction_index_rejects_duplicate_stable_site_identity():
    original = _site(iso_class=0, member_nodes=[[10]])
    equivalent_reconstruction = _site(iso_class=0, member_nodes=[[10]])
    index = _ReactionIndex([original])

    with pytest.raises(ValueError, match="duplicate adsorbate site identifier"):
        index.extend_sites(adsorbate_sites=[equivalent_reconstruction])


def test_typed_request_returns_typed_result_with_run_telemetry():
    functions = KMCFunctions(
        compute_adsorption=lambda *_args, **_kwargs: [],
        recompute_affected=lambda *_args, **_kwargs: ([], []),
        expand_bond_network=lambda *_args, **_kwargs: [],
    )
    request = KMCRunRequest(
        system=KMCSystem(nx.Graph(), [], None, {}),
        settings=KMCSettings(temperature=500.0, n_steps=0, verbose=False),
        rng=7,
    )

    result = run_kmc(request, functions=functions)

    assert isinstance(result, KMCRunResult)
    assert result.steps_executed == 0
    assert result.performance["counters"]["kmc.session.runs"] == 1
    assert result.performance["timings_s"]["kmc.session.seconds"] >= 0.0
    assert result.to_legacy_dict()["time"] == result.time_s


def test_initialisation_failure_flushes_reactions_and_numerical_bond_diagnostics(
    monkeypatch,
):
    completed_reaction = SimpleNamespace(kind="adsorption")
    adsorption_site = _site(iso_class=0, member_nodes=[[10]])
    adsorption_site.applicable_reactions = [completed_reaction]
    unresolved_lateral = SimpleNamespace(
        lateral_class=2,
        members=[0],
        stable=None,
        last_failure_reason=(
            "BondNEBNotConvergedError: forced numerical failure"
        ),
    )
    bond_site = SimpleNamespace(
        lateral_classes=[unresolved_lateral],
        applicable_reactions=[],
    )

    calls = []

    class Writer:
        def ensure_reaction(self, reaction, *, step):
            calls.append(("reaction", reaction, step))

        def write_invalid_bond(self, site, lateral, *, step):
            calls.append(("invalid_bond", site, lateral, step))

        def sync_for_checkpoint(self):
            calls.append(("sync",))

    def fail_initialisation(*_args, **_kwargs):
        raise RuntimeError("initial NEB sweep failed")

    monkeypatch.setattr(
        session_module,
        "initialise_runtime",
        fail_initialisation,
    )
    request = KMCRunRequest(
        system=KMCSystem(
            nx.Graph(),
            [adsorption_site],
            None,
            {"[O]": 0.0},
        ),
        settings=KMCSettings(
            temperature=500.0,
            n_steps=0,
            verbose=False,
        ),
        channels=KMCChannels(bond_sites=[bond_site]),
        observers=KMCObservers(reaction_writer=Writer()),  # type: ignore[arg-type]
    )
    functions = KMCFunctions(
        compute_adsorption=lambda *_args, **_kwargs: [],
        recompute_affected=lambda *_args, **_kwargs: ([], []),
        expand_bond_network=lambda *_args, **_kwargs: [],
    )

    with pytest.raises(RuntimeError, match="initial NEB sweep failed"):
        run_kmc(request, functions=functions)

    assert calls == [
        ("reaction", completed_reaction, 0),
        ("invalid_bond", bond_site, unresolved_lateral, 0),
        ("sync",),
    ]


def test_unresolved_transition_omission_does_not_block_kmc_step(monkeypatch):
    adsorption_site = _site(iso_class=0, member_nodes=[[10]])
    adsorption_reaction = SimpleNamespace(
        kind="adsorption",
        site=adsorption_site,
        member_index=0,
        lateral_class=SimpleNamespace(lateral_class=0),
        delta_e=-0.1,
        barrier=0.2,
        rate=1.0,
    )
    unresolved_lateral = SimpleNamespace(
        lateral_class=1,
        members=[0],
        stable=None,
        last_failure_reason=(
            "BondNEBNotConvergedError: forced numerical failure"
        ),
    )
    bond_site = SimpleNamespace(
        iso_class=1,
        template=SimpleNamespace(
            smiles_a="[H]",
            smiles_b="[H]",
            smiles_c="[H][H]",
            bond_type="single",
            source="test",
            is_symmetric=True,
        ),
        member_node_ids=[([20], [21], [22])],
        lateral_classes=[unresolved_lateral],
        applicable_reactions=[],
        site_id="",
    )

    def compute_adsorption(*_args, **_kwargs):
        adsorption_site.applicable_reactions = [adsorption_reaction]
        return [adsorption_reaction]

    def omit_unresolved_bond(_graph, sites, *_args, **_kwargs):
        assert sites == [bond_site]
        bond_site.applicable_reactions = []
        return []

    monkeypatch.setattr(
        initialization_module,
        "compute_all_bond_reactions",
        omit_unresolved_bond,
    )
    monkeypatch.setattr(
        session_module,
        "execute_reaction",
        lambda *_args, **_kwargs: set(),
    )
    monkeypatch.setattr(
        KMCOutputManager,
        "capture_transition",
        lambda *_args, **_kwargs: None,
    )

    request = KMCRunRequest(
        system=KMCSystem(
            nx.Graph(),
            [adsorption_site],
            None,
            {"[O]": 0.0},
        ),
        settings=KMCSettings(
            temperature=500.0,
            n_steps=1,
            verbose=False,
        ),
        channels=KMCChannels(bond_sites=[bond_site]),
        rng=13,
    )
    functions = KMCFunctions(
        compute_adsorption=compute_adsorption,
        recompute_affected=lambda *_args, **_kwargs: ([], []),
        expand_bond_network=lambda *_args, **_kwargs: [],
    )

    result = run_kmc(request, functions=functions)

    assert result.steps_executed == 1
    assert result.termination_status == "complete"
    assert result.termination_reason == "requested_steps_completed"


def test_event_is_not_published_when_recomputation_fails(monkeypatch):
    site = _site(iso_class=0, member_nodes=[[10]])
    reaction = SimpleNamespace(
        kind="adsorption",
        site=site,
        member_index=0,
        lateral_class=SimpleNamespace(lateral_class=0),
        delta_e=-0.1,
        barrier=0.2,
        rate=1.0,
    )

    def compute(*_args, **_kwargs):
        site.applicable_reactions = [reaction]
        return [reaction]

    def fail_recompute(*_args, **_kwargs):
        raise RuntimeError("recompute failed")

    class Writer:
        def __init__(self):
            self.records = []

        def ensure_reaction(self, *_args, **_kwargs):
            return None

        def record(self, **payload):
            self.records.append(payload)

        def write_invalid_diffusion(self, *_args, **_kwargs):
            return None

    writer = Writer()
    request = KMCRunRequest(
        system=KMCSystem(nx.Graph(), [site], None, {"[O]": 0.0}),
        settings=KMCSettings(temperature=500.0, n_steps=1, verbose=False),
        observers=KMCObservers(reaction_writer=writer),  # type: ignore[arg-type]
        rng=13,
    )
    functions = KMCFunctions(
        compute_adsorption=compute,
        recompute_affected=fail_recompute,
        expand_bond_network=lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        session_module,
        "execute_reaction",
        lambda *_args, **_kwargs: set(),
    )
    monkeypatch.setattr(
        KMCOutputManager,
        "capture_transition",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="recompute failed"):
        run_kmc(request, functions=functions)

    assert writer.records == []


def test_workflow_uses_the_canonical_resume_model():
    assert WorkflowResumeState is KMCResumeState
