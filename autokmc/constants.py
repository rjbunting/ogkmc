"""
autokmc.constants
=================
Centralised tunable defaults.

Every magic number that controls site discovery, neighbour-list cutoffs,
co-bonding cutoffs and isomorphism heuristics is collected here so that
users can override them in one place, and so that readers do not have to
hunt for them across the codebase.

Override globally by mutating these module attributes *before* calling
into :mod:`autokmc`, or per-call via the explicit keyword arguments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Neighbour-list / co-bonding
# ---------------------------------------------------------------------------

#: Multiplier for ``ase.neighborlist.natural_cutoffs``.  Used by
#: :func:`autokmc.graph.build_graph`, :func:`autokmc.surface.find_surface_atoms`
#: and :func:`autokmc.reactants.build_reactant`.  A single value is used
#: across the package so that "is this bond a cross-image bond?" gives the
#: same answer everywhere.
NL_MULT_DEFAULT: float = 1.0

#: Co-bonding cutoff scale used by
#: :func:`autokmc.default_sites._build_co_bond_graph`: two surface atoms can
#: simultaneously bind a single adsorbate when their separation is at most
#: ``CO_FACTOR * (2*r_cov_ads + r_cov_i + r_cov_j)``.
CO_FACTOR: float = 0.90

# ---------------------------------------------------------------------------
# Geometric site / multisite optimisation
# ---------------------------------------------------------------------------

#: Ideal-bond-length scale for calculator-free site geometry optimisation
#: in :func:`autokmc.default_sites._optimize_site_position`:
#: ``d_ideal = OPT_FACTOR * (r_cov_ads + r_cov_i)``.
OPT_FACTOR: float = 0.85

#: Weight of the ``1/r²`` non-bonded soft repulsion term used by
#: :func:`autokmc.default_sites._optimize_site_position`.
REPULSION_WEIGHT: float = 0.1

#: Steric-contact scale used by
#: :func:`autokmc.find_multisite.optimise_adsorbate_site_positions` —
#: ``R_min = CONTACT_FACTOR * (r_cov_a + r_cov_s)`` for every (adsorbate,
#: surface) pair.  Matches the ``OPT_FACTOR`` semantics above.
CONTACT_FACTOR: float = 0.9

#: Standoff-bond scale used by
#: :func:`autokmc.find_multisite.optimise_adsorbate_site_positions`: each
#: bonded anchor is restrained toward a target ``standoff`` Å above the
#: surface clique centroid along the local outward normal, where
#: ``standoff = STANDOFF_FACTOR * (r_cov_a + <r_cov_s>)``.  Prevents the
#: rigid molecule from being pulled into the surface plane.
STANDOFF_FACTOR: float = 0.85

#: Default number of rigid-body rotational restarts about the local
#: outward surface normal used by
#: :func:`autokmc.sites.optimise_adsorbate_site_positions`.  Combats
#: local minima for asymmetric adsorbates on bridge/hollow sites.  ``1``
#: disables multi-start.
N_ADSORBATE_RESTARTS: int = 6


#: Default radius (Å) of the spatial cutoff used to filter the non-bonded
#: surface atoms that contribute to the repulsion sum in
#: :func:`autokmc.default_sites._optimize_site_position`.  Only atoms
#: within this distance of the clique centroid are considered, since the
#: ``1/r²`` term dies off rapidly.  Set to ``None`` to disable spatial
#: filtering and fall back to the slower all-atoms behaviour.
SITE_REPULSION_CUTOFF: float = 6.0

# ---------------------------------------------------------------------------
# Isomorphism / iso-class deduplication
# ---------------------------------------------------------------------------

#: Default number of ego-graph shells used by
#: :func:`autokmc.default_sites.reduce_sites_by_isomorphism`.  ``1``
#: distinguishes fcc vs hcp hollows on Cu(111); ``0`` collapses them.
N_SHELLS_DEFAULT: int = 1

# ---------------------------------------------------------------------------
# Adsorbate-site enumeration
# ---------------------------------------------------------------------------

#: Typical metal nearest-neighbour distance (Å) used by
#: :func:`autokmc.find_multisite._suggested_n_shells` to pick a default
#: ``n_shells_anchor`` from the molecular reach.
NN_DISTANCE: float = 2.5

#: Hard cap on the per-placement adaptive ego depth used when
#: ``require_surface_connected`` is on in
#: :func:`autokmc.find_multisite.find_adsorbate_sites`.
MAX_PAIR_SHELLS: int = 10

#: Default tolerance (Å) when matching surface anchor-pair distances to
#: intramolecular distances in :func:`autokmc.find_multisite.find_adsorbate_sites`.
BOND_TOLERANCE: float = 0.4

# ---------------------------------------------------------------------------
# Convex-hull / surface classification
# ---------------------------------------------------------------------------

#: Tolerance (Å) below which a clique centroid is considered to lie
#: "inside" the nanoparticle convex hull and discarded as a wrap-around
#: spurious site in :func:`autokmc.default_sites.find_sites_for_element`.
HULL_TOL: float = -0.2

