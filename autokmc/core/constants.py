"""Centralized tunable defaults.

This module remains the compatibility home for defaults that are imported
across several domains. Over time, domain-specific groups can move to their
own packages once the import churn is worth it.

Centralized tunable defaults are the **single source of truth** for every
magic number that controls site discovery, neighbour-list cutoffs,
co-bonding cutoffs and isomorphism heuristics.

Every other module imports its defaults from here::

    from autokmc.core.constants import CO_FACTOR, OPT_FACTOR, ...

Override globally by mutating these module attributes *before* importing
the consumer modules, or per-call via the explicit keyword arguments on
``find_anchor_sites``, ``find_adsorbate_sites``, etc.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

#: Single random seed used by **every** stochastic step in the package
#: (alloy substitution in :func:`autokmc.structure._apply_composition`,
#: RDKit ETKDGv3 conformer embedding in
#: :func:`autokmc.species.reactant._smiles_to_atoms`, …).  Centralized so a single
#: change makes every workflow reproducibly different.
RANDOM_SEED: int = 69

# ---------------------------------------------------------------------------
# Neighbour-list / co-bonding
# ---------------------------------------------------------------------------

#: Multiplier for ``ase.neighborlist.natural_cutoffs``.  Used by
#: :func:`autokmc.core.graph.build_graph`,
#: :func:`autokmc.structure.find_surface_atoms`
#: and :func:`autokmc.species.reactant.build_reactant`.  A single value is used
#: across the package so that "is this bond a cross-image bond?" gives the
#: same answer everywhere.
NL_MULT_DEFAULT: float = 0.90

#: Per-atom skin (Å) passed explicitly to ASE ``NeighborList``.  Keeping this
#: value centralized prevents direct distance checks from drifting if ASE
#: changes its constructor default.
NEIGHBORLIST_SKIN: float = 0.30

#: Co-bonding cutoff scale used by
#: :func:`autokmc.sites.anchors._build_co_bond_graph`: two surface atoms can
#: simultaneously bind a single adsorbate when their separation is at most
#: ``CO_FACTOR * (2*r_cov_ads + r_cov_i + r_cov_j)``.
CO_FACTOR: float = 0.90

# ---------------------------------------------------------------------------
# Geometric site / multisite optimisation
# ---------------------------------------------------------------------------

#: Ideal-bond-length scale for calculator-free site geometry optimisation
#: in :func:`autokmc.sites.anchors._optimise_position`:
#: ``d_ideal = OPT_FACTOR * (r_cov_ads + r_cov_i)``.
OPT_FACTOR: float = 0.85

#: Weight of the ``1/r²`` non-bonded soft repulsion term used by
#: :func:`autokmc.sites.anchors._optimise_position`.
REPULSION_WEIGHT: float = 0.2

#: Steric-contact scale used by
#: :func:`autokmc.sites.adsorbate.optimise_adsorbate_site_positions` —
#: ``R_min = CONTACT_FACTOR * (r_cov_a + r_cov_s)`` for every (adsorbate,
#: surface) pair.  Matches the ``OPT_FACTOR`` semantics above.
CONTACT_FACTOR: float = 1.05

#: Standoff-bond scale used by
#: :func:`autokmc.sites.adsorbate.optimise_adsorbate_site_positions`.
#: When ``STANDOFF_FACTOR > 0``, bonded anchors are restrained toward
#: ``clique_centroid + STANDOFF_FACTOR*(r_cov_a + <r_cov_s>)*n_hat``.
#: When ``0.0`` (default), the anchor-node position computed by
#: :func:`autokmc.sites.anchors._optimise_position` (already placed at
#: ``OPT_FACTOR*(r_cov_a+r_cov_s)`` from its clique) is used directly as the
#: restraint target — no additional lift.  This ensures the adsorbate anchor
#: sits exactly at the anchor-site position for both slabs and nanoparticles.
STANDOFF_FACTOR: float = 0.0

#: Default number of rigid-body rotational restarts about the local
#: outward surface normal used by
#: :func:`autokmc.sites.adsorbate.optimise_adsorbate_site_positions`.
#: Combats local minima for asymmetric adsorbates on bridge/hollow sites.
#: ``1`` disables multi-start.
N_ADSORBATE_RESTARTS: int = 6

#: Alias kept for symmetry with the per-module name.
N_RESTARTS: int = N_ADSORBATE_RESTARTS


#: Default radius (Å) of the spatial cutoff used to filter the non-bonded
#: surface atoms that contribute to the repulsion sum in
#: :func:`autokmc.sites.anchors._optimise_position`.  Only atoms
#: within this distance of the clique centroid are considered, since the
#: ``1/r²`` term dies off rapidly.  Set to ``None`` to disable spatial
#: filtering and fall back to the slower all-atoms behaviour.
SITE_REPULSION_CUTOFF: float = 10.0

#: Alias for :data:`SITE_REPULSION_CUTOFF` matching the per-module name.
REPULSION_CUTOFF: float = SITE_REPULSION_CUTOFF

# ---------------------------------------------------------------------------
# Isomorphism / iso-class deduplication
# ---------------------------------------------------------------------------

#: Default number of ego-graph shells used by
#: :func:`autokmc.sites.anchors._reduce_by_isomorphism`.  ``1``
#: distinguishes fcc vs hcp hollows on Cu(111); ``0`` collapses them.
N_SHELLS_DEFAULT: int = 1

#: Alias matching the per-module name in :mod:`autokmc.sites.anchors`.
N_SHELLS: int = N_SHELLS_DEFAULT

# ---------------------------------------------------------------------------
# Adsorbate-site enumeration
# ---------------------------------------------------------------------------

#: Typical metal nearest-neighbour distance (Å) used by
#: :func:`autokmc.sites.adsorbate._suggested_n_shells` to pick a default
#: ``n_shells_anchor`` from the molecular reach.
NN_DISTANCE: float = 2.5

#: Hard cap on the per-placement adaptive ego depth used when
#: ``require_surface_connected`` is on in
#: :func:`autokmc.sites.adsorbate.find_adsorbate_sites`.
MAX_PAIR_SHELLS: int = 10

#: Default tolerance (Å) when matching surface anchor-pair distances to
#: intramolecular distances in :func:`autokmc.sites.adsorbate.find_adsorbate_sites`.
BOND_TOLERANCE: float = 0.4

# ---------------------------------------------------------------------------
# Stability pruning
# ---------------------------------------------------------------------------

#: Force convergence threshold (eV/Å) for the ML-potential relaxation used by
#: :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites` to
#: decide whether a representative placement is stable.
PRUNE_FMAX: float = 0.05

#: Maximum LBFGS steps for the ML-potential relaxation used by
#: :func:`autokmc.sites.adsorbate.prune_unstable_adsorbate_sites`.
PRUNE_MAX_STEPS: int = 500

# ---------------------------------------------------------------------------
# Convex-hull / surface classification
# ---------------------------------------------------------------------------

#: Tolerance (Å) below which a clique centroid is considered to lie
#: "inside" the nanoparticle convex hull and discarded as a wrap-around
#: spurious site in :func:`autokmc.sites.anchors._enumerate_cliques`.
HULL_TOL: float = -0.2

# ---------------------------------------------------------------------------
# Surface ray-casting (slabs)
# ---------------------------------------------------------------------------

#: Default per-disc coverage fraction required to mark an atom as
#: "exposed" in :func:`autokmc.structure.find_surface_atoms_raycasting`.
#: Used by both the dispatcher (:func:`autokmc.structure.find_surface_atoms`)
#: and the reverse-ray-casting layer-freezing helper
#: (:func:`autokmc.structure._get_bottom_layer_indices`) so the same
#: threshold is applied everywhere.
RAYCAST_COVERAGE_THRESHOLD: float = 0.7

#: Default ray-disc resolution (per axis) in
#: :func:`autokmc.structure.find_surface_atoms_raycasting`.  ``n_disc_sample=10``
#: yields ``≈ π·10²/4 ≈ 78`` rays per atom.
RAYCAST_N_DISC_SAMPLE: int = 10

# ---------------------------------------------------------------------------
# Kabsch ego-alignment
# ---------------------------------------------------------------------------

#: Hard cap on the number of automorphisms enumerated by
#: :func:`autokmc.sites.anchors._kabsch_align_ego` when searching for the
#: lowest-RMSD mapping between a representative and a member ego-graph.
#: 6969 is plenty for any chemically meaningful symmetry group while
#: still bounding pathological complete-graph blow-ups.
KABSCH_MAX_MAPPINGS: int = 6969

# ---------------------------------------------------------------------------
# Persistence / CLI / output (consumed by autokmc.io + autokmc.cli)
# ---------------------------------------------------------------------------

#: Schema version stamped onto every reaction event and summary JSON document
#: written by :mod:`autokmc.io.persistence`.  Bump on any breaking schema change.
PERSISTENCE_SCHEMA_VERSION: str = "2"

#: Default config-file schema version expected by :mod:`autokmc.io.config`.
CONFIG_SCHEMA_VERSION: str = "1"

#: Default cadence for :class:`autokmc.io.trajectory.TrajectoryWriter` — write
#: an ASE ``.traj`` frame every N executed KMC steps.  ``0`` disables.
TRAJ_DUMP_EVERY: int = 10

#: Default BFS depth used by
#: :func:`autokmc.sites.stability.adsorption.check_adsorbate_site_lateral` when
#: building the *lateral ego-graph* around a member's bonded surface clique.
#:
#: ``0`` (default) — only adsorbates that bond to an **exact member** of the
#: site's bonded surface clique are counted as lateral neighbours.  This is
#: the "sharing surface atoms" criterion: two adsorbates interact laterally
#: only when they compete for the same surface atom(s).  Sets the number of
#: distinct lateral classes to a minimum and avoids picking up far-away
#: adsorbates.
#:
#: Raise to ``1`` to also include adsorbates on first-nearest-neighbour Cu
#: atoms (one surface hop from the bonded clique), or ``2`` for second-NN,
#: etc.  The :mod:`autokmc.kmc.engine` incremental trigger radius is
#: automatically derived from this value.
LATERAL_SHELLS_DEFAULT: int = 0

#: Default basenames written by :func:`autokmc.cli.pipeline.run_from_config`.
REACTIONS_FILENAME:   str = "events.jsonl"
SUMMARY_FILENAME:     str = "summary.json"
RUN_MANIFEST_FILENAME: str = "run_manifest.json"
TRAJECTORY_FILENAME:  str = "kmc.extxyz"
CALCULATION_CACHE_DIR:       str = "calculation_cache"
ISAAC_EXPORT_FILENAME:       str = "isaac_records.json"
REACTIONS_DIR:        str = "reactions"
DEFAULT_OUTPUT_DIR:   str = "autokmc_run"

#: Backwards-compat alias — older code referred to per-event sidecars under
#: ``frames/``; the new layout writes per-lateral-class folders under
#: ``reactions/`` instead, but the constant is kept so external callers do
#: not break.
ATOMS_SIDECAR_DIR:    str = REACTIONS_DIR

#: Template used to build the human-readable ``description`` field on a
#: persisted reaction event.  See :class:`autokmc.io.records.ReactionRecord`.
REACTION_DESCRIPTION_FMT: str = (
    "{kind} of {smiles} at iso={iso} m={member} lat={lateral} "
    "(ΔE={delta_e:+.4f} eV, Ea={barrier:.4f} eV, k={rate:.3e} Hz)"
)

# ---------------------------------------------------------------------------
# Diffusion / NEB
# ---------------------------------------------------------------------------

#: Maximum surface-graph hop distance allowed between the bonded surface
#: cliques of the two endpoints of a diffusion pair (consumed by
#: :func:`autokmc.sites.diffusion.find_diffusion_sites`).  ``0`` requires
#: clique overlap; ``1`` (default) means "share a surface atom OR are bonded
#: surface-to-surface neighbours"; larger values allow longer hops.
DIFFUSION_MAX_HOPS: int = 0

#: Number of *intermediate* NEB images (excluding the two endpoints) used by
#: :func:`autokmc.sites.stability.diffusion.check_diffusion_stability`.
NEB_N_IMAGES: int = 10

#: Force convergence threshold (eV/Å) for the NEB band relaxation in
#: :func:`autokmc.sites.stability.diffusion.check_diffusion_stability`.
NEB_FMAX: float = 0.01

#: Maximum optimiser steps for the NEB band relaxation.
NEB_MAX_STEPS: int = 200

#: After ordinary NEB convergence, refine the highest-energy image with
#: climbing-image NEB (CI-NEB) so it converges onto the saddle point.
NEB_CLIMB: bool = True

#: NEB spring constant (eV / Å²).
NEB_SPRING_K: float = 5.0

#: NEB initial-band interpolation method: ``"idpp"`` (image-dependent pair
#: potential, ASE default for chemistry) or ``"linear"``.
NEB_INTERPOLATION: str = "linear"

#: ASE NEB force/tangent formulation. ``"improvedtangent"`` is the default;
#: the other values map directly to ASE's ``NEB(method=...)`` choices.
NEB_METHODS: frozenset[str] = frozenset(
    {"aseneb", "eb", "improvedtangent", "spline", "string"}
)
NEB_METHOD: str = "improvedtangent"

#: How NEB band images are evaluated each optimizer step: ``"images"``
#: (per-image calculator calls — serial with a shared calculator, or pooled
#: threads on a multi-worker CalculatorPool) or ``"batched"`` (the whole band
#: in one stacked model forward, when the calculator supports it; falls back
#: to ``"images"`` otherwise).  Physics is identical either way — only the
#: force-evaluation access pattern changes.
NEB_BAND_EVALS: frozenset[str] = frozenset({"images", "batched"})
NEB_BAND_EVAL: str = "images"

#: When ``True`` (default), :func:`autokmc.sites.diffusion.find_diffusion_sites`
#: keeps only **one** :class:`~autokmc.sites.diffusion.DiffusionSite` per
#: unordered pair of adsorption iso-classes — the one whose ego-graph has the
#: fewest nodes + edges (i.e. the most direct / geometrically closest hop path).
#: Set to ``False`` to retain all crystallographically-distinct hop directions.
DIFFUSION_PRUNE_BY_ADS_PAIR: bool = True

#: Folder-name format for diffusion reaction folders persisted by
#: :class:`autokmc.io.persistence.ReactionWriter`.  See
#: :func:`autokmc.io.persistence._diffusion_folder_name`.
DIFFUSION_FOLDER_FMT: str = "diff_iso{iso}_lat{lat}"

#: Description template for a diffusion event line.
DIFFUSION_DESCRIPTION_FMT: str = (
    "diffusion of {smiles} at diff_iso={iso} m={member} lat={lateral} "
    "dir={direction} (ΔE={delta_e:+.4f} eV, Ea={barrier:.4f} eV, "
    "k={rate:.3e} Hz)"
)


# ---------------------------------------------------------------------------
# Bond reactions (A + B  ⇌  C  on the surface)
# ---------------------------------------------------------------------------

#: Maximum surface-graph hop distance allowed (a) between A's and B's bonded
#: cliques and (b) between C's bonded clique and the union (A ∪ B).
#: ``0`` (default) means "share at least one surface atom" — the strictest
#: locality constraint.  Consumed by
#: :func:`autokmc.sites.bond.find_bond_sites`.
BOND_MAX_HOPS: int = 0

#: When ``True`` (default), :func:`autokmc.sites.bond.find_bond_sites`
#: keeps only **one** :class:`~autokmc.sites.bond.BondReactionSite` per
#: unordered triple of adsorption iso-classes ``(frozenset({iso_a, iso_b}),
#: iso_c)`` — the one whose triple ego-graph has the fewest nodes + edges
#: (most direct / geometrically closest reaction).  Mirrors
#: :data:`DIFFUSION_PRUNE_BY_ADS_PAIR`.
BOND_PRUNE_BY_TRIPLE: bool = True

#: When ``True`` (default), :func:`autokmc.sites.bond.prune_unstable_bond_sites`
#: is invoked by the CLI to drop bond-reaction iso-classes whose ``A + B``
#: endpoint state is not bond-connectivity-stable under a calculator
#: relaxation — i.e. the reaction is not physically viable.
BOND_PRUNE_WITH_CALCULATOR: bool = True

#: BFS depth used to build the triple ego-graph in
#: :func:`autokmc.sites.bond.find_bond_sites` for triple iso-class
#: deduplication / pruning.
BOND_PAIR_N_SHELLS: int = N_SHELLS_DEFAULT

#: Bond-reaction NEB interpolation default.  Bond-forming/breaking paths are
#: more sensitive to the initial band than diffusion hops, so use IDPP by
#: default while leaving the generic/diffusion default unchanged.
BOND_NEB_INTERPOLATION: str = "idpp"

#: Atom-correspondence strategy for bond-reaction NEB endpoints.
#: ``"auto"`` tries a small candidate set and keeps the lowest-displacement
#: path; ``"hungarian"`` solves the global same-element assignment problem;
#: ``"greedy"`` preserves the pre-existing nearest-neighbour behaviour; and
#: ``"reactant_index"`` keeps C's reactant atom order when chemically valid.
BOND_ATOM_MATCHING: str = "auto"

#: Maximum number of same-element assignment candidates considered by
#: ``BOND_ATOM_MATCHING == "auto"`` before the best pre-NEB path is chosen.
BOND_MATCHING_TRIALS: int = 8

#: Default lift height (Å) used when the C endpoint of ``A + B ⇌ C`` is a
#: gas-phase product rather than a materialised surface placement.
BOND_GAS_LIFT_HEIGHT: float = 6.0

#: Prepare gas-product bond NEBs from a relaxed, intact molecular precursor
#: adsorbed above the reacting surface site.  The gas-phase asymptote remains
#: the thermodynamic C-state reference used by KMC rates.
BOND_GAS_PRECURSOR_RELAX: bool = True

#: Initial minimum molecule-to-slab distance (Å) for the fixed-environment
#: gas-precursor relaxation.
BOND_GAS_PRECURSOR_DISTANCE: float = 1.8

#: Folder-name format for bond reaction folders persisted by
#: :class:`autokmc.io.persistence.ReactionWriter` (when enabled).
BOND_FOLDER_FMT: str = "bond_iso{iso}_lat{lat}"

#: Description template for a bond reaction event line.
BOND_DESCRIPTION_FMT: str = (
    "bond reaction {smiles_a}+{smiles_b}↔{smiles_c} at bond_iso={iso} "
    "m={member} lat={lateral} dir={direction} "
    "(ΔE={delta_e:+.4f} eV, Ea={barrier:.4f} eV, k={rate:.3e} Hz)"
)
