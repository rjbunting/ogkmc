"""
autokmc.constants
=================
Centralised tunable defaults — the **single source of truth** for every
magic number that controls site discovery, neighbour-list cutoffs,
co-bonding cutoffs and isomorphism heuristics.

Every other module imports its defaults from here::

    from autokmc.constants import CO_FACTOR, OPT_FACTOR, ...

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
#: :func:`autokmc.reactants._smiles_to_atoms`, …).  Centralised so a single
#: change makes every workflow reproducibly different.
RANDOM_SEED: int = 69

# ---------------------------------------------------------------------------
# Neighbour-list / co-bonding
# ---------------------------------------------------------------------------

#: Multiplier for ``ase.neighborlist.natural_cutoffs``.  Used by
#: :func:`autokmc.graph.build_graph`, :func:`autokmc.surface.find_surface_atoms`
#: and :func:`autokmc.reactants.build_reactant`.  A single value is used
#: across the package so that "is this bond a cross-image bond?" gives the
#: same answer everywhere.
NL_MULT_DEFAULT: float = 1.00

#: Co-bonding cutoff scale used by
#: :func:`autokmc.find_anchors._build_co_bond_graph`: two surface atoms can
#: simultaneously bind a single adsorbate when their separation is at most
#: ``CO_FACTOR * (2*r_cov_ads + r_cov_i + r_cov_j)``.
CO_FACTOR: float = 0.90

# ---------------------------------------------------------------------------
# Geometric site / multisite optimisation
# ---------------------------------------------------------------------------

#: Ideal-bond-length scale for calculator-free site geometry optimisation
#: in :func:`autokmc.find_anchors._optimise_position`:
#: ``d_ideal = OPT_FACTOR * (r_cov_ads + r_cov_i)``.
OPT_FACTOR: float = 0.85

#: Weight of the ``1/r²`` non-bonded soft repulsion term used by
#: :func:`autokmc.find_anchors._optimise_position`.
REPULSION_WEIGHT: float = 0.1

#: Steric-contact scale used by
#: :func:`autokmc.find_adsorbate_sites.optimise_adsorbate_site_positions` —
#: ``R_min = CONTACT_FACTOR * (r_cov_a + r_cov_s)`` for every (adsorbate,
#: surface) pair.  Matches the ``OPT_FACTOR`` semantics above.
CONTACT_FACTOR: float = 0.9

#: Standoff-bond scale used by
#: :func:`autokmc.find_adsorbate_sites.optimise_adsorbate_site_positions`.
#: When ``STANDOFF_FACTOR > 0``, bonded anchors are restrained toward
#: ``clique_centroid + STANDOFF_FACTOR*(r_cov_a + <r_cov_s>)*n_hat``.
#: When ``0.0`` (default), the anchor-node position computed by
#: :func:`autokmc.find_anchors._optimise_position` (already placed at
#: ``OPT_FACTOR*(r_cov_a+r_cov_s)`` from its clique) is used directly as the
#: restraint target — no additional lift.  This ensures the adsorbate anchor
#: sits exactly at the anchor-site position for both slabs and nanoparticles.
STANDOFF_FACTOR: float = 0.0

#: Default number of rigid-body rotational restarts about the local
#: outward surface normal used by
#: :func:`autokmc.find_adsorbate_sites.optimise_adsorbate_site_positions`.
#: Combats local minima for asymmetric adsorbates on bridge/hollow sites.
#: ``1`` disables multi-start.
N_ADSORBATE_RESTARTS: int = 6

#: Alias kept for symmetry with the per-module name.
N_RESTARTS: int = N_ADSORBATE_RESTARTS


#: Default radius (Å) of the spatial cutoff used to filter the non-bonded
#: surface atoms that contribute to the repulsion sum in
#: :func:`autokmc.find_anchors._optimise_position`.  Only atoms
#: within this distance of the clique centroid are considered, since the
#: ``1/r²`` term dies off rapidly.  Set to ``None`` to disable spatial
#: filtering and fall back to the slower all-atoms behaviour.
SITE_REPULSION_CUTOFF: float = 6.0

#: Alias for :data:`SITE_REPULSION_CUTOFF` matching the per-module name.
REPULSION_CUTOFF: float = SITE_REPULSION_CUTOFF

# ---------------------------------------------------------------------------
# Isomorphism / iso-class deduplication
# ---------------------------------------------------------------------------

#: Default number of ego-graph shells used by
#: :func:`autokmc.find_anchors._reduce_by_isomorphism`.  ``1``
#: distinguishes fcc vs hcp hollows on Cu(111); ``0`` collapses them.
N_SHELLS_DEFAULT: int = 1

#: Alias matching the per-module name in :mod:`autokmc.find_anchors`.
N_SHELLS: int = N_SHELLS_DEFAULT

# ---------------------------------------------------------------------------
# Adsorbate-site enumeration
# ---------------------------------------------------------------------------

#: Typical metal nearest-neighbour distance (Å) used by
#: :func:`autokmc.find_adsorbate_sites._suggested_n_shells` to pick a default
#: ``n_shells_anchor`` from the molecular reach.
NN_DISTANCE: float = 2.5

#: Hard cap on the per-placement adaptive ego depth used when
#: ``require_surface_connected`` is on in
#: :func:`autokmc.find_adsorbate_sites.find_adsorbate_sites`.
MAX_PAIR_SHELLS: int = 10

#: Default tolerance (Å) when matching surface anchor-pair distances to
#: intramolecular distances in :func:`autokmc.find_adsorbate_sites.find_adsorbate_sites`.
BOND_TOLERANCE: float = 0.4

# ---------------------------------------------------------------------------
# Stability pruning
# ---------------------------------------------------------------------------

#: Force convergence threshold (eV/Å) for the ML-potential relaxation used by
#: :func:`autokmc.find_adsorbate_sites.prune_unstable_adsorbate_sites` to
#: decide whether a representative placement is stable.
PRUNE_FMAX: float = 0.05

#: Maximum LBFGS steps for the ML-potential relaxation used by
#: :func:`autokmc.find_adsorbate_sites.prune_unstable_adsorbate_sites`.
PRUNE_MAX_STEPS: int = 500

# ---------------------------------------------------------------------------
# Convex-hull / surface classification
# ---------------------------------------------------------------------------

#: Tolerance (Å) below which a clique centroid is considered to lie
#: "inside" the nanoparticle convex hull and discarded as a wrap-around
#: spurious site in :func:`autokmc.find_anchors._enumerate_cliques`.
HULL_TOL: float = -0.2

# ---------------------------------------------------------------------------
# Surface ray-casting (slabs)
# ---------------------------------------------------------------------------

#: Default per-disc coverage fraction required to mark an atom as
#: "exposed" in :func:`autokmc.surface.find_surface_atoms_raycasting`.
#: Used by both the dispatcher (:func:`autokmc.surface.find_surface_atoms`)
#: and the reverse-ray-casting layer-freezing helper
#: (:func:`autokmc.structure._get_bottom_layer_indices`) so the same
#: threshold is applied everywhere.
RAYCAST_COVERAGE_THRESHOLD: float = 0.7

#: Default ray-disc resolution (per axis) in
#: :func:`autokmc.surface.find_surface_atoms_raycasting`.  ``n_disc_sample=10``
#: yields ``≈ π·10²/4 ≈ 78`` rays per atom.
RAYCAST_N_DISC_SAMPLE: int = 10

# ---------------------------------------------------------------------------
# Kabsch ego-alignment
# ---------------------------------------------------------------------------

#: Hard cap on the number of automorphisms enumerated by
#: :func:`autokmc.find_anchors._kabsch_align_ego` when searching for the
#: lowest-RMSD mapping between a representative and a member ego-graph.
#: 6969 is plenty for any chemically meaningful symmetry group while
#: still bounding pathological complete-graph blow-ups.
KABSCH_MAX_MAPPINGS: int = 6969

# ---------------------------------------------------------------------------
# Persistence / CLI / output (consumed by autokmc.persistence + autokmc.cli)
# ---------------------------------------------------------------------------

#: Schema version stamped onto every reaction event and summary JSON document
#: written by :mod:`autokmc.persistence`.  Bump on any breaking schema change.
PERSISTENCE_SCHEMA_VERSION: str = "1"

#: Default config-file schema version expected by :mod:`autokmc.config`.
CONFIG_SCHEMA_VERSION: str = "1"

#: Default cadence for :class:`autokmc.persistence.TrajectoryWriter` — write
#: an ASE ``.traj`` frame every N executed KMC steps.  ``0`` disables.
TRAJ_DUMP_EVERY: int = 10

#: Default BFS depth used by
#: :func:`autokmc.check_adsorbate_sites.check_adsorbate_site_lateral` when
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
#: etc.  The :mod:`autokmc.kmc_simulation` incremental trigger radius is
#: automatically derived from this value.
LATERAL_SHELLS_DEFAULT: int = 0

#: Default basenames written by :func:`autokmc.cli.run_from_config`.
REACTIONS_FILENAME:   str = "events.jsonl"
SUMMARY_FILENAME:     str = "summary.json"
TRAJECTORY_FILENAME:  str = "kmc.extxyz"
REACTIONS_DIR:        str = "reactions"
DEFAULT_OUTPUT_DIR:   str = "autokmc_run"

#: Backwards-compat alias — older code referred to per-event sidecars under
#: ``frames/``; the new layout writes per-lateral-class folders under
#: ``reactions/`` instead, but the constant is kept so external callers do
#: not break.
ATOMS_SIDECAR_DIR:    str = REACTIONS_DIR

#: Template used to build the human-readable ``description`` field on a
#: persisted reaction event.  See :class:`autokmc.persistence.ReactionRecord`.
REACTION_DESCRIPTION_FMT: str = (
    "{kind} of {smiles} at iso={iso} m={member} lat={lateral} "
    "(ΔE={delta_e:+.4f} eV, Ea={barrier:.4f} eV, k={rate:.3e} Hz)"
)
