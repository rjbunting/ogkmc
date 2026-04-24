"""
autokmc.adsorbate
=================
Optimise single-atom adsorbates at every unique adsorption site using an
ML force-field (or any ASE calculator), and verify that the adsorbate
connectivity is preserved after relaxation.

Workflow
--------
**Stage 1 – Geometric site positions (no calculator)**

Unique adsorption sites are found purely from graph geometry using
:mod:`autokmc.default_sites`:

1. :func:`~autokmc.default_sites.find_sites_for_element` enumerates every
   k-clique of the adsorbate co-bonding graph.
2. :func:`~autokmc.default_sites.reduce_sites_by_isomorphism` groups cliques
   into iso-classes by n-shell ego-graph isomorphism.
3. :func:`~autokmc.default_sites.optimise_site_positions` places the
   adsorbate at the ideal bond-length distance from each clique atom using a
   pure-geometry (scipy) minimisation — **no calculator is involved**.

**Stage 2 – Energy minimisation, connectivity check, and adsorption energy**

For each unique :class:`~autokmc.default_sites.IsoClass` the adsorbate is
then relaxed with an ASE calculator:

1. The geometrically pre-optimised position (``IsoClass.position``) is used
   as the starting point.
2. A single adsorbate atom is appended to a copy of the bare host structure.
3. The combined structure is relaxed with LBFGS.  Existing ASE constraints
   on *atoms* are preserved.
4. After relaxation the *actual* bonded surface atoms are identified
   (:func:`find_actual_clique`) and compared to the original clique.
   A :data:`ConnectivityStatus` flag records whether bonds were lost or
   gained (:func:`check_connectivity`).
5. If connectivity changed, :func:`find_matching_isoclass` checks whether the
   new bond topology is isomorphic to any *other* known iso-class — a sanity
   check that the adsorbate has simply migrated to another well-defined site.
6. The adsorption energy is computed as::

       E_ads = E_total − (E_gas + E_surface)

   where *E_gas* is the energy of the isolated adsorbate atom and *E_surface*
   is the energy of the bare slab.  Both reference energies are either
   supplied by the caller or computed automatically.

**Stage 3 – Populating the graph for KMC**

Every clique instance (not just iso-class representatives) is registered in
``G.graph['adsorption_sites'][element][n_shells]`` as an
:class:`AdsorptionSite` object.  Each entry carries the adsorption energy,
a reference to the iso-class optimisation result, an ``occupied`` flag
that can be toggled during kinetic Monte Carlo, and a **subgraph**.

The subgraph is a full copy of the clean-surface graph ``G`` extended with
one adsorbate node (key = ``len(slab)``, ``type="adsorbate"``) connected to
the clique atoms by edges.  The original ``G`` — including
``G.graph['sites']``, ``G.graph['unique_sites']``, ``G.graph['site_positions']``
— is **never modified**, so the default-site data is always preserved.

Public API
----------
* :func:`optimise_unique_sites`      -- main entry point (Stage 2/3 bootstrap)
* :func:`calculate_gas_phase_energy` -- compute isolated-atom reference energy
* :class:`SiteOptResult`             -- per-iso-class result record
* :class:`AdsorptionSite`            -- per-clique-instance record for KMC
* :class:`ConnectivityStatus`        -- connectivity check outcome enum
* :func:`place_adsorbate`            -- low-level: append adsorbate to Atoms copy
* :func:`find_actual_clique`         -- low-level: bonds after relaxation

On-the-fly KMC API (see ``dev/PLAN_adsorption_sites.md``)
---------------------------------------------------------
* :func:`compute_neighbour_sites`    -- shared-surface-atom neighbour map
* :func:`discover_context_site`      -- relax (or cache-hit) one site under a
                                        given occupancy context
* :func:`register_adsorption`        -- KMC event hook: an atom adsorbed at a site
* :func:`register_desorption`        -- KMC event hook: an atom desorbed from a site
* :func:`update_reactive_flags`      -- recompute ``reactive`` for a clique and its
                                        neighbours after an event
* :func:`check_connectivity`         -- low-level: verify bond topology
* :func:`find_matching_isoclass`     -- low-level: isomorphism match for migrated sites
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, cast

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.optimize import LBFGS
import networkx as nx
from networkx.algorithms import isomorphism

from autokmc.default_sites import IsoClass, _build_clique_ego, _iso_prefilter_key


# ---------------------------------------------------------------------------
# Connectivity status
# ---------------------------------------------------------------------------

class ConnectivityStatus(str, Enum):
    """Outcome of the post-optimisation connectivity check.

    Attributes
    ----------
    OK
        The adsorbate is still bonded to exactly the original clique atoms.
    LOST_BOND
        At least one original clique atom is no longer within bond distance.
    NEW_BOND
        The adsorbate has formed a bond to a surface atom outside the clique.
    MIGRATED
        Both a bond was lost *and* a new bond was gained (site migration).
    UNKNOWN
        The check could not be performed (e.g. no position data).
    """
    OK        = "ok"
    LOST_BOND = "lost_bond"
    NEW_BOND  = "new_bond"
    MIGRATED  = "migrated"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class SiteOptResult:
    """Full result for one unique iso-class after calculator relaxation.

    Attributes
    ----------
    iso_class : IsoClass
        The iso-class this result belongs to.
    atoms_initial : Atoms
        Slab copy with adsorbate inserted at the geometric starting position.
    atoms_final : Atoms
        Relaxed structure (slab + adsorbate).
    energy : float
        Total energy of *atoms_final* in eV.  ``nan`` if the calculator
        did not converge.
    adsorption_energy : float
        Adsorption energy in eV: ``E_total − (E_gas + E_surface)``.
        ``nan`` if either reference energy is unavailable.
    converged : bool
        Whether the LBFGS optimiser reached *fmax* within *steps*.
    n_steps : int
        Number of LBFGS steps taken.
    connectivity : ConnectivityStatus
        Outcome of the post-relaxation connectivity check.
    displacement : float
        Euclidean distance (Å) between the initial and final adsorbate
        positions (minimum-image for periodic structures).
    ads_index : int
        Index of the adsorbate atom in *atoms_final* (always ``len(slab)``).
    actual_clique : frozenset or None
        Surface-atom indices that the adsorbate is *actually* bonded to
        after relaxation (may differ from ``iso_class.representative`` when
        ``connectivity != OK``).
    matched_iso_class : IsoClass or None
        When ``connectivity != OK``, the iso-class whose ego-graph is
        isomorphic to the ego-graph built around ``actual_clique``.
        ``None`` if connectivity is OK or no match was found (novel site).
    opt_log : str
        Path to the LBFGS log file for this site (empty string if silenced).
    """
    iso_class              : IsoClass
    atoms_initial          : Atoms
    atoms_final            : Atoms
    energy                 : float
    adsorption_energy      : float
    converged              : bool
    n_steps                : int
    connectivity           : ConnectivityStatus
    displacement           : float
    ads_index              : int
    actual_clique              : Optional[frozenset]  = field(default=None)
    matched_iso_class          : Optional[IsoClass]   = field(default=None)
    surface_connectivity_changed : bool               = field(default=False)
    opt_log                    : str                  = field(default="", repr=False)


# ---------------------------------------------------------------------------
# KMC adsorption-site record
# ---------------------------------------------------------------------------

@dataclass
class AdsorptionSite:
    """One concrete adsorption site instance for use in kinetic Monte Carlo.

    Unlike :class:`SiteOptResult` (which covers only one representative per
    iso-class), an :class:`AdsorptionSite` exists for **every** clique on the
    surface.  Sites that belong to the same iso-class share the same
    :attr:`result` and therefore the same :attr:`adsorption_energy`.

    All instances are stored in::

        G.graph['adsorption_sites'][element][n_shells]  # dict[frozenset, AdsorptionSite]

    Parameters / Attributes
    -----------------------
    clique : frozenset[int]
        Global surface-atom indices that define this site.
    iso_class : IsoClass
        The isomorphism class this site belongs to.
    result : SiteOptResult
        Optimisation result for the representative of this iso-class.
    adsorption_energy : float
        Adsorption energy in eV (``E_total − (E_gas + E_surface)``).
        Negative means exothermic / stable adsorption.
    occupied : bool
        KMC occupancy flag.  ``False`` = vacant site (adsorption is
        possible); ``True`` = occupied.
    ads_position : np.ndarray, shape (3,)
        Cartesian coordinates (Å) of the adsorbate atom at this site.
        For the iso-class **representative** this is the fully **relaxed**
        position taken from ``result.atoms_final``.  For all other members
        of the same iso-class this is the **geometrically optimised**
        position from ``G.graph['site_positions']`` — the best available
        without running additional calculations.
    subgraph : nx.Graph
        A copy of the clean-surface graph ``G`` with one additional node
        representing the adsorbed atom.  The node key is
        ``G.number_of_nodes()`` (i.e. ``len(slab)``).  It carries the same
        node attributes as every other node (``element``, ``position``,
        ``type="adsorbate"``, ``covalent_radius``, ``index``).  Edges
        connect it to every atom in :attr:`clique`.

        The ``position`` attribute of the adsorbate node matches
        :attr:`ads_position` — relaxed for the representative, geometric
        for other members.  The original ``G`` and all default-site data
        inside it are **never modified**.
    """
    clique             : frozenset
    iso_class          : IsoClass
    result             : SiteOptResult
    adsorption_energy  : float
    occupied           : bool                 = field(default=False)
    ads_position       : Optional[np.ndarray] = field(default=None, repr=False)
    subgraph           : Optional[nx.Graph]   = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # On-the-fly KMC state (populated by optimise_unique_sites Stage 3
    # and updated by register_adsorption / register_desorption /
    # update_reactive_flags).  See dev/PLAN_adsorption_sites.md.
    # ------------------------------------------------------------------
    stable             : bool                 = field(default=True)
    """``True`` iff the iso-class representative relaxed with
    ``ConnectivityStatus.OK`` on the clean surface.  All clique members of
    the same iso-class share this flag."""

    reactive           : bool                 = field(default=True)
    """``True`` iff this site is stable, vacant, *and* a relaxation result
    is cached for the current neighbour-occupancy context.  Updated by
    :func:`update_reactive_flags`."""

    migrated_to        : Optional[frozenset]  = field(default=None)
    """For unstable sites only: surface-atom indices the adsorbate
    actually bonded to after relaxation (from
    ``SiteOptResult.actual_clique``).  ``None`` for stable sites."""

    neighbour_sites    : list                 = field(default_factory=list, repr=False)
    """Cliques sharing at least one surface atom with this one (computed
    by :func:`compute_neighbour_sites`).  List of ``frozenset[int]``."""

    context_results    : dict                 = field(default_factory=dict, repr=False)
    """Per-site cache: ``{frozenset(occupied_neighbour_cliques) ->
    SiteOptResult}``.  Empty key = clean-surface bootstrap result."""

    current_result     : Optional[SiteOptResult] = field(default=None, repr=False)
    """Most recently looked-up :class:`SiteOptResult` for this site under
    the current surface state.  KMC reads ``current_result.adsorption_energy``
    to compute rates."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def place_adsorbate(
    atoms: Atoms,
    position: np.ndarray,
    element: str,
) -> Atoms:
    """Return a copy of *atoms* with a single adsorbate atom appended.

    The adsorbate is placed at *position* (Cartesian, Å).  All existing
    constraints on *atoms* are deep-copied so that they are preserved during
    subsequent relaxation.

    Parameters
    ----------
    atoms : Atoms
        Host structure (slab or nanoparticle).
    position : np.ndarray, shape (3,)
        Cartesian coordinates for the adsorbate.
    element : str
        Chemical symbol of the adsorbate atom.

    Returns
    -------
    Atoms
        New Atoms object with ``len(atoms) + 1`` entries.  The adsorbate
        is the last atom (index ``len(atoms)``).
    """
    result = atoms.copy()
    result += Atoms(element, positions=[position])
    return result


def _mic_dist_vec(
    ads_pos: np.ndarray,
    ref_pos: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray | None,
    pbc: np.ndarray,
) -> float:
    """Minimum-image distance between *ads_pos* and *ref_pos*."""
    dv = ads_pos - ref_pos
    if cell_inv is not None and pbc.any():
        frac = dv @ cell_inv
        for ax in range(3):
            if pbc[ax]:
                frac[ax] -= np.round(frac[ax])
        dv = frac @ cell
    return float(np.linalg.norm(dv))


def find_actual_clique(
    atoms_final: Atoms,
    ads_index: int,
    surface_graph: nx.Graph,
    r_cov_ads: float,
    bond_factor: float = 1.10,
) -> frozenset:
    """Return the set of surface atoms actually bonded to the adsorbate.

    Scans every surface-type node in *surface_graph* and applies the same
    bond criterion used during site enumeration::

        distance(ads, i) <= bond_factor * (r_cov_ads + r_cov_i)

    Distances use the minimum-image convention for periodic structures.

    Parameters
    ----------
    atoms_final : Atoms
        Relaxed structure containing both slab and adsorbate atoms.
    ads_index : int
        Index of the adsorbate atom in *atoms_final*.
    surface_graph : nx.Graph
        The original atom graph (node ``"covalent_radius"`` and graph-level
        ``"cell"`` / ``"pbc"`` are used).
    r_cov_ads : float
        Covalent radius of the adsorbate (Å).
    bond_factor : float
        Multiplier for the bond-length cutoff.  Default 1.10.

    Returns
    -------
    frozenset[int]
        Global atom indices (matching node IDs in *surface_graph*) of all
        surface atoms within bond distance of the adsorbate.
    """
    pos_all = atoms_final.get_positions()
    ads_pos = pos_all[ads_index]

    cell    = np.array(surface_graph.graph["cell"], dtype=float)
    pbc     = np.asarray(surface_graph.graph.get("pbc", [True, True, False]), dtype=bool)
    cell_inv = None
    if pbc.any():
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            pass

    bonded: list[int] = []
    for n, d in surface_graph.nodes(data=True):
        if d["type"] != "surface":
            continue
        cutoff = bond_factor * (r_cov_ads + float(d["covalent_radius"]))
        dist = _mic_dist_vec(ads_pos, pos_all[n], cell, cell_inv, pbc)
        if dist <= cutoff:
            bonded.append(cast(int, n))

    return frozenset(bonded)


def check_connectivity(
    atoms_final: Atoms,
    ads_index: int,
    clique: frozenset,
    r_cov_ads: float,
    surface_graph: nx.Graph,
    bond_factor: float = 1.10,
) -> ConnectivityStatus:
    """Check whether the adsorbate connectivity is preserved after relaxation.

    The adsorbate (atom *ads_index* in *atoms_final*) is compared against the
    original *clique* (bonded surface-atom indices) and all other surface atoms.

    A bond is considered formed when::

        distance(ads, i) <= bond_factor * (r_cov_ads + r_cov_i)

    Distances use the minimum-image convention for periodic structures.

    Parameters
    ----------
    atoms_final : Atoms
        Relaxed structure containing both slab atoms and the adsorbate.
    ads_index : int
        Index of the adsorbate atom in *atoms_final*.
    clique : frozenset[int]
        Surface-atom indices that the adsorbate was supposed to bond to.
    r_cov_ads : float
        Covalent radius of the adsorbate (Å).
    surface_graph : nx.Graph
        The original atom graph (used for covalent radii and cell/pbc).
    bond_factor : float
        Multiplier for the bond-length cutoff.  Default 1.10.

    Returns
    -------
    ConnectivityStatus
    """
    actual = find_actual_clique(
        atoms_final, ads_index, surface_graph, r_cov_ads, bond_factor
    )

    lost_bonds = [n for n in clique   if n not in actual]
    new_bonds  = [n for n in actual   if n not in clique]

    if lost_bonds and new_bonds:
        return ConnectivityStatus.MIGRATED
    if lost_bonds:
        return ConnectivityStatus.LOST_BOND
    if new_bonds:
        return ConnectivityStatus.NEW_BOND
    return ConnectivityStatus.OK


def find_matching_isoclass(
    G: nx.Graph,
    actual_clique: frozenset,
    element: str,
    n_shells: int = 1,
) -> Optional[IsoClass]:
    """Return the IsoClass whose topology matches *actual_clique*, or ``None``.

    After relaxation the adsorbate may have migrated to a different site.
    This function checks whether the new bond topology (described by
    *actual_clique*) is graph-isomorphic (with element labels matched) to
    any known iso-class stored in ``G.graph['unique_sites'][element][n_shells]``.

    The comparison uses the same n-shell ego-subgraph expansion as
    :func:`~autokmc.default_sites.reduce_sites_by_isomorphism`.

    Parameters
    ----------
    G : nx.Graph
        Atom graph with ``G.graph['unique_sites'][element][n_shells]``
        already populated.
    actual_clique : frozenset[int]
        Surface-atom indices that the adsorbate is *actually* bonded to
        (e.g. from :func:`find_actual_clique`).
    element : str
        Chemical symbol of the adsorbate (used to look up iso-classes).
    n_shells : int
        Shell depth used when ``reduce_sites_by_isomorphism`` was called.
        Default 2.

    Returns
    -------
    IsoClass or None
        The first iso-class whose ego-graph is isomorphic to the ego-graph
        of *actual_clique*, or ``None`` if no match is found (the adsorbate
        has moved to a genuinely novel site topology).
    """
    unique_by_k: dict = (
        G.graph.get("unique_sites", {})
         .get(element, {})
         .get(n_shells, {})
    )

    k = len(actual_clique)
    if k not in unique_by_k or not actual_clique:
        return None

    actual_ego = _build_clique_ego(G, actual_clique, n_shells)
    actual_key = _iso_prefilter_key(actual_ego)
    node_match  = isomorphism.categorical_node_match("element", "X")

    for iso in unique_by_k[k]:
        if iso.ego_graph is None:
            continue
        if _iso_prefilter_key(iso.ego_graph) != actual_key:
            continue
        gm = isomorphism.GraphMatcher(actual_ego, iso.ego_graph,
                                      node_match=node_match)
        if gm.is_isomorphic():
            return iso

    return None


def _ads_displacement(
    atoms_initial: Atoms,
    atoms_final: Atoms,
    ads_index: int,
    surface_graph: nx.Graph,
) -> float:
    """Return the MIC displacement of the adsorbate between initial and final."""
    p0 = atoms_initial.get_positions()[ads_index]
    p1 = atoms_final.get_positions()[ads_index]
    dv = p1 - p0

    cell = np.array(surface_graph.graph["cell"], dtype=float)
    pbc  = np.asarray(surface_graph.graph.get("pbc", [True, True, False]), dtype=bool)
    if pbc.any():
        try:
            cell_inv = np.linalg.inv(cell)
            frac = dv @ cell_inv
            for ax in range(3):
                if pbc[ax]:
                    frac[ax] -= np.round(frac[ax])
            dv = frac @ cell
        except np.linalg.LinAlgError:
            pass
    return float(np.linalg.norm(dv))


# ---------------------------------------------------------------------------
# Adsorbate subgraph builder
# ---------------------------------------------------------------------------

def _build_adsorbate_subgraph(
    G: nx.Graph,
    clique: frozenset,
    element: str,
    position: np.ndarray,
    r_cov_ads: float,
) -> nx.Graph:
    """Return a copy of *G* extended with one adsorbate node.

    The new node key is ``G.number_of_nodes()`` (consistent with the atom
    index used in the ASE :class:`~ase.Atoms` object, where the adsorbate is
    appended at ``len(slab)``).  Edges are added from the adsorbate node to
    every atom in *clique*.

    The original graph *G* — and all site data stored inside it — is **not
    modified**.

    Parameters
    ----------
    G : nx.Graph
        Clean-surface atom graph.
    clique : frozenset[int]
        Surface-atom indices the adsorbate bonds to.
    element : str
        Chemical symbol of the adsorbate.
    position : np.ndarray, shape (3,)
        Cartesian coordinates for the adsorbate node (Å).
    r_cov_ads : float
        Covalent radius of the adsorbate (Å).

    Returns
    -------
    nx.Graph
        New graph with all nodes/edges of *G* plus one adsorbate node.
    """
    H = G.copy()
    ads_node = G.number_of_nodes()   # next available integer index
    H.add_node(
        ads_node,
        element         = element,
        position        = position.copy(),
        index           = ads_node,
        type            = "adsorbate",
        covalent_radius = r_cov_ads,
    )
    for surf_atom in clique:
        H.add_edge(ads_node, surf_atom)
    return H


# ---------------------------------------------------------------------------
# Gas-phase reference energy
# ---------------------------------------------------------------------------

def calculate_gas_phase_energy(
    element: str,
    calculator,
    *,
    box_size: float = 15.0,
    fmax: float = 0.01,
    steps: int = 200,
    verbose: bool = False,
) -> float:
    """Compute the total energy of a single isolated adsorbate atom.

    Places one atom of *element* in a cubic vacuum box of side *box_size* Å,
    attaches *calculator*, and relaxes with LBFGS (single atoms typically
    converge in 0 steps as there are no forces, but this is included for
    consistency with multi-atom molecules).

    The result is the gas-phase reference energy :math:`E_\\text{gas}` used
    when computing the adsorption energy::

        E_ads = E_total − (E_gas + E_surface)

    Parameters
    ----------
    element : str
        Chemical symbol of the adsorbate atom, e.g. ``"O"``.
    calculator
        Any ASE-compatible calculator.  A **copy** is used internally so the
        original object is not modified.
    box_size : float
        Side length of the cubic vacuum cell in Å.  Default 15.0.
    fmax : float
        Force convergence criterion for the single-atom relaxation.  Default 0.01.
    steps : int
        Maximum LBFGS steps.  Default 200.
    verbose : bool
        Print the computed gas-phase energy.

    Returns
    -------
    float
        Total energy of the isolated atom in eV.

    Raises
    ------
    KeyError
        If *element* is not recognised by ASE.
    """
    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")

    gas = Atoms(element, positions=[[box_size / 2, box_size / 2, box_size / 2]])
    gas.set_cell([box_size, box_size, box_size])
    gas.set_pbc(False)
    gas.calc = calculator

    opt = LBFGS(gas, logfile=os.devnull)
    try:
        opt.run(fmax=fmax, steps=steps)
        e_gas = float(gas.get_potential_energy())
    except Exception as exc:
        warnings.warn(
            f"Gas-phase relaxation failed for '{element}': {exc}",
            RuntimeWarning, stacklevel=2,
        )
        e_gas = float("nan")

    if verbose:
        print(f"calculate_gas_phase_energy: '{element}'  E_gas = {e_gas:.6f} eV")

    return e_gas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Surface connectivity check
# ---------------------------------------------------------------------------

def check_surface_connectivity(
    atoms_final: Atoms,
    surface_graph: nx.Graph,
    nl_mult: float = 1.0,
) -> bool:
    """Return ``True`` if surface-surface bond topology changed after relaxation.

    Rebuilds the surface bond adjacency from the *relaxed* atom positions using
    the same ``natural_cutoffs`` + ``NeighborList`` criterion as
    :func:`~autokmc.graph.build_graph`, then compares the surface-to-surface
    edge set against the original edges stored in *surface_graph*.

    Only pairs where **both** atoms are surface nodes are considered; the
    adsorbate atom (appended at ``len(slab)``) is ignored.

    Parameters
    ----------
    atoms_final : Atoms
        Relaxed structure (slab + adsorbate).  The slab atoms occupy the same
        indices as the nodes of *surface_graph*; the last atom is the adsorbate.
    surface_graph : nx.Graph
        Original atom graph (before adsorbate insertion).
    nl_mult : float
        ``natural_cutoffs`` multiplier.  Must match the value used when the
        original graph was built (default ``1.0``).

    Returns
    -------
    bool
        ``True`` if any surface–surface bond was formed or broken.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs as _natural_cutoffs

    # Surface node set from the original graph
    surface_nodes: set[int] = {
        int(n) for n, d in surface_graph.nodes(data=True) if d["type"] == "surface"  # type: ignore[arg-type]
    }

    # Original surface–surface edges (frozensets for set comparison)
    orig_edges: set[frozenset] = {
        frozenset((u, v))
        for u, v in surface_graph.edges()
        if u in surface_nodes and v in surface_nodes
    }

    # Rebuild neighbor list from relaxed positions
    # atoms_final includes the adsorbate at the last index — we only care
    # about the slab portion (indices 0 … len(slab)-1 = len(surface_graph)-1)
    n_slab = surface_graph.number_of_nodes()
    slab_only = atoms_final[:n_slab]  # slicing returns a view / copy without adsorbate

    cutoffs = _natural_cutoffs(slab_only, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(slab_only)

    new_edges: set[frozenset] = set()
    for i in surface_nodes:
        neighbours, _ = nl.get_neighbors(i)
        for j in map(int, neighbours):
            if j in surface_nodes and j > i:
                new_edges.add(frozenset((i, j)))

    return orig_edges != new_edges


# ---------------------------------------------------------------------------
# Main optimisation driver
# ---------------------------------------------------------------------------

def optimise_unique_sites(
    G: nx.Graph,
    element: str,
    atoms: Atoms,
    calculator,
    *,
    n_shells: int = 1,
    bond_factor: float = 1.10,
    fmax: float = 0.05,
    steps: int = 500,
    e_surface: float | None = None,
    e_gas: float | None = None,
    gas_box_size: float = 15.0,
    logfile: str | None = None,
    verbose: bool = True,
) -> list[SiteOptResult]:
    """Optimise every unique adsorption site with a calculator and check topology.

    **Stage 1 (geometric positions) must already be complete.**  Call
    :func:`~autokmc.default_sites.find_sites_for_element`,
    :func:`~autokmc.default_sites.reduce_sites_by_isomorphism`, and
    :func:`~autokmc.default_sites.optimise_site_positions` before calling
    this function.  The geometric positions in ``IsoClass.position`` serve as
    the starting point — **no geometry fitting with a calculator is performed
    at that stage**.

    **Stage 2** (this function) for each iso-class:

    1. Place the adsorbate at the geometric starting position.
    2. Relax the full structure (slab + adsorbate) with *calculator* + LBFGS.
    3. Identify the actual bonded surface atoms after relaxation
       (:func:`find_actual_clique`).
    4. Compare to the original clique (:func:`check_connectivity`).
    5. If connectivity changed, check whether the new bond topology is
       isomorphic to any known iso-class (:func:`find_matching_isoclass`).
       A match indicates the adsorbate has migrated to a known site type;
       no match suggests a genuinely novel topology.
    6. Compute the adsorption energy::

           E_ads = E_total − (E_gas + E_surface)

       *E_surface* and *E_gas* can be supplied explicitly; if either is
       ``None`` it is computed automatically (*E_surface* from *atoms* +
       *calculator*, *E_gas* from a single atom in a :data:`gas_box_size` Å
       cubic box).

    **Stage 3** – All clique *instances* (not only iso-class representatives)
    are registered in ``G.graph['adsorption_sites'][element][n_shells]`` as
    :class:`AdsorptionSite` objects.  Sites in the same iso-class share the
    same :attr:`AdsorptionSite.result` and :attr:`AdsorptionSite.adsorption_energy`.
    The ``occupied`` flag of every site is initialised to ``False`` (vacant),
    ready for kinetic Monte Carlo.

    Each :class:`AdsorptionSite` also carries a :attr:`~AdsorptionSite.subgraph`
    — a copy of the clean-surface graph ``G`` extended with the adsorbate node
    (key ``len(slab)``, ``type="adsorbate"``) bonded to its clique atoms.  The
    original ``G`` (including ``G.graph['sites']``, ``G.graph['unique_sites']``,
    ``G.graph['site_positions']``) is **never modified**; default-site data is
    always preserved alongside the adsorption-site data.

    Parameters
    ----------
    G : nx.Graph
        Atom graph.  ``G.graph['sites']``, ``G.graph['unique_sites']``, and
        ``G.graph['site_positions']`` must all have been populated for
        *element*.
    element : str
        Chemical symbol of the adsorbate, e.g. ``"C"``.
    atoms : Atoms
        The bare host structure **without** any adsorbate.
    calculator
        Any ASE-compatible calculator.
    n_shells : int
        Shell depth used when ``reduce_sites_by_isomorphism`` was called.
        Default 1.
    bond_factor : float
        Multiplier for the bond-length cutoff used in post-optimisation
        connectivity checks.  Default 1.10.
    fmax : float
        Force convergence criterion in eV/Å.  Default 0.05.
    steps : int
        Maximum LBFGS steps per site.  Default 500.
    e_surface : float or None
        Total energy of the bare slab in eV.  If ``None``, it is computed
        automatically using *atoms* and *calculator*.
    e_gas : float or None
        Gas-phase energy of a single isolated adsorbate atom in eV.  If
        ``None``, it is computed automatically via
        :func:`calculate_gas_phase_energy`.
    gas_box_size : float
        Side length of the vacuum box used when computing *e_gas*
        automatically.  Default 15.0 Å.
    logfile : str or None
        Path template for per-site LBFGS log files.  ``"{k}"`` and
        ``"{cls}"`` are replaced with coordination number and class index.
        ``None`` silences all output.
    verbose : bool
        Print a one-line summary per site.

    Returns
    -------
    results : list[SiteOptResult]
        One entry per unique iso-class.  Also stored in
        ``G.graph['site_opt'][element][n_shells]``.

        All clique instances are additionally stored in
        ``G.graph['adsorption_sites'][element][n_shells]`` as
        :class:`AdsorptionSite` objects.

    Raises
    ------
    KeyError
        If the required graph data has not been populated.
    ValueError
        If ``IsoClass.position`` is ``None`` for any iso-class.
    """
    # ── Validate prerequisites ────────────────────────────────────────────
    for key, desc in [
        ("sites",          "find_sites_for_element"),
        ("unique_sites",   "reduce_sites_by_isomorphism"),
        ("site_positions", "optimise_site_positions"),
    ]:
        if key not in G.graph or element not in G.graph[key]:
            raise KeyError(
                f"G.graph['{key}']['{element}'] not found. "
                f"Call {desc}(G, '{element}') first."
            )

    if element not in ASE_ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element '{element}'.")
    r_cov_ads = float(ASE_COVALENT_RADII[ASE_ATOMIC_NUMBERS[element]])

    unique_by_k: dict = G.graph["unique_sites"][element].get(n_shells, {})
    if not unique_by_k:
        raise KeyError(
            f"No unique sites for n_shells={n_shells}. "
            "Call reduce_sites_by_isomorphism with the correct n_shells first."
        )

    # ── Reference energies ───────────────────────────────────────────────
    if e_surface is None:
        if verbose:
            print("optimise_unique_sites: computing clean-surface energy …")
        slab_ref = atoms.copy()
        slab_ref.calc = calculator
        try:
            e_surface = float(slab_ref.get_potential_energy())
            if verbose:
                print(f"  E_surface = {e_surface:.6f} eV")
        except Exception as exc:
            warnings.warn(
                f"Could not compute surface energy: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            e_surface = float("nan")

    if e_gas is None:
        if verbose:
            print(f"optimise_unique_sites: computing gas-phase energy for '{element}' …")
        e_gas = calculate_gas_phase_energy(
            element, calculator,
            box_size=gas_box_size,
            verbose=verbose,
        )

    e_ref = float(e_gas) + float(e_surface)  # type: ignore[arg-type]  # both resolved above

    _LABELS = {1: "top", 2: "bridge", 3: "hollow"}
    results: list[SiteOptResult] = []

    if verbose:
        total = sum(len(v) for v in unique_by_k.values())
        print(f"optimise_unique_sites: '{element}'  n_shells={n_shells}  "
              f"bond_factor={bond_factor}  fmax={fmax}  ({total} unique sites)")
        print(f"  E_surface={e_surface:.4f} eV  E_gas={e_gas:.4f} eV  "
              f"E_ref={e_ref:.4f} eV")
        print(f"  {'k':>3}  {'type':8s}  {'cls':>4}  "
              f"{'conv':>5}  {'steps':>5}  {'E (eV)':>10}  "
              f"{'E_ads (eV)':>11}  {'disp (Å)':>9}  "
              f"{'connect':>10}  {'surf_chg':>8}  {'match':>12}")
        print("  " + "-" * 100)

    for k, iso_classes in sorted(unique_by_k.items()):
        label = _LABELS.get(k, f"{k}-fold")
        for iso in iso_classes:
            if iso.position is None:
                raise ValueError(
                    f"IsoClass(k={k}, cls={iso.iso_class}) has no position. "
                    "Call optimise_site_positions first."
                )

            ads_index  = len(atoms)
            trial      = place_adsorbate(atoms, iso.position, element)
            trial_init = trial.copy()
            trial.calc = calculator

            log = os.devnull
            if logfile is not None:
                log = logfile.format(k=k, cls=iso.iso_class)

            opt = LBFGS(trial, logfile=log)
            try:
                opt.run(fmax=fmax, steps=steps)
                converged = opt.converged()
                n_steps   = opt.get_number_of_steps()
                energy    = float(trial.get_potential_energy())
            except Exception as exc:
                warnings.warn(
                    f"LBFGS failed for k={k} cls={iso.iso_class}: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                converged, n_steps, energy = False, 0, float("nan")

            # ── Adsorption energy ─────────────────────────────────────────
            adsorption_energy = energy - e_ref  # nan propagates if any ref is nan

            # ── Stage 2: connectivity analysis ───────────────────────
            actual_clique = find_actual_clique(
                trial, ads_index, G, r_cov_ads, bond_factor=bond_factor
            )
            connectivity = check_connectivity(
                trial, ads_index, iso.representative,
                r_cov_ads, G, bond_factor=bond_factor,
            )
            disp = _ads_displacement(trial_init, trial, ads_index, G)
            surface_changed = check_surface_connectivity(trial, G)

            # ── Stage 2b: isomorphism check if connectivity changed ───────
            matched: Optional[IsoClass] = None
            if connectivity != ConnectivityStatus.OK:
                matched = find_matching_isoclass(
                    G, actual_clique, element, n_shells
                )

            result = SiteOptResult(
                iso_class                    = iso,
                atoms_initial                = trial_init,
                atoms_final                  = trial,
                energy                       = energy,
                adsorption_energy            = adsorption_energy,
                converged                    = converged,
                n_steps                      = n_steps,
                connectivity                 = connectivity,
                displacement                 = disp,
                ads_index                    = ads_index,
                actual_clique                = actual_clique,
                matched_iso_class            = matched,
                surface_connectivity_changed = surface_changed,
                opt_log                      = log if logfile else "",
            )
            results.append(result)

            if verbose:
                if matched is not None:
                    match_str = f"k={matched.k}/cls={matched.iso_class}"
                elif connectivity != ConnectivityStatus.OK:
                    match_str = "novel"
                else:
                    match_str = "-"
                ads_str = f"{adsorption_energy:+.4f}" if not np.isnan(adsorption_energy) else "     nan"
                print(f"  {k:>3}  {label:8s}  {iso.iso_class:>4}  "
                      f"{'✓' if converged else '✗':>5}  {n_steps:>5}  "
                      f"{energy:>10.4f}  {ads_str:>11}  {disp:>9.3f}  "
                      f"{connectivity.value:>10}  {'✓' if surface_changed else '-':>8}  "
                      f"{match_str:>12}")

    if verbose:
        n_ok      = sum(1 for r in results if r.connectivity == ConnectivityStatus.OK)
        n_matched = sum(1 for r in results
                        if r.connectivity != ConnectivityStatus.OK
                        and r.matched_iso_class is not None)
        n_novel   = sum(1 for r in results
                        if r.connectivity != ConnectivityStatus.OK
                        and r.matched_iso_class is None)
        n_surf_chg = sum(1 for r in results if r.surface_connectivity_changed)
        print("  " + "-" * 100)
        print(f"  Total: {len(results)}  connectivity OK: {n_ok}  "
              f"migrated→known: {n_matched}  novel: {n_novel}  "
              f"surface_changed: {n_surf_chg}")

    G.graph.setdefault("site_opt", {}).setdefault(element, {})[n_shells] = results

    # ── Stage 3: populate adsorption_sites for KMC ───────────────────────
    # Build a lookup: iso → result
    iso_result_map: dict[int, SiteOptResult] = {
        id(res.iso_class): res for res in results
    }

    # Per-clique geometric positions (ordered the same as G.graph['sites'])
    sites_by_k: dict     = G.graph["sites"][element]
    site_pos_by_k: dict  = G.graph["site_positions"][element]

    ads_sites: dict[frozenset, AdsorptionSite] = {}
    for k, iso_classes in sorted(unique_by_k.items()):
        k_cliques   = sites_by_k.get(k, [])
        k_positions = site_pos_by_k.get(k, [])
        for iso in iso_classes:
            res = iso_result_map.get(id(iso))
            if res is None:
                continue
            for clique in iso.members:
                # Adsorbate position for this specific clique:
                #   - representative  → relaxed position from atoms_final
                #   - other members   → geometric position from site_positions
                if clique == iso.representative:
                    ads_pos: np.ndarray = res.atoms_final.get_positions()[res.ads_index].copy()
                else:
                    try:
                        idx     = k_cliques.index(clique)
                        ads_pos = np.asarray(k_positions[idx]).copy()
                    except (ValueError, IndexError):
                        # iso.position is guaranteed non-None here (checked above)
                        ads_pos = np.asarray(iso.position).copy()

                H = _build_adsorbate_subgraph(G, clique, element, ads_pos, r_cov_ads)

                stable = (res.connectivity == ConnectivityStatus.OK)
                migrated_to = (
                    res.actual_clique
                    if (not stable and res.actual_clique)
                    else None
                )

                site = AdsorptionSite(
                    clique            = clique,
                    iso_class         = iso,
                    result            = res,
                    adsorption_energy = res.adsorption_energy,
                    occupied          = False,
                    ads_position      = ads_pos,
                    subgraph          = H,
                    stable            = stable,
                    # Reactive only on a clean surface AND stable; flips off
                    # as soon as a neighbour adsorbs (see update_reactive_flags).
                    reactive          = stable,
                    migrated_to       = migrated_to,
                    current_result    = res,
                )
                # Seed the context cache with the clean-surface (no occupied
                # neighbours) result.  Discovery loops can then short-circuit
                # back to this entry on full desorption.
                site.context_results[frozenset()] = res
                ads_sites[clique] = site

    (G.graph
       .setdefault("adsorption_sites", {})
       .setdefault(element, {})[n_shells]) = ads_sites

    # ── Stage 3b: populate per-site neighbour lists ──────────────────────
    compute_neighbour_sites(G, element, n_shells)

    if verbose:
        n_vacant = sum(1 for s in ads_sites.values() if not s.occupied)
        n_stable = sum(1 for s in ads_sites.values() if s.stable)
        n_unstab = len(ads_sites) - n_stable
        print(f"\n  adsorption_sites registered: {len(ads_sites)} "
              f"({n_vacant} vacant, {n_stable} stable, {n_unstab} unstable) "
              f"→ G.graph['adsorption_sites']['{element}'][{n_shells}]")
        print(f"  Each AdsorptionSite.subgraph = copy(G) + adsorbate node "
              f"(node {G.number_of_nodes()}) bonded to its clique.")
        print(f"  G.graph['sites'] / 'unique_sites' / 'site_positions' unchanged.")

    return results


# ===========================================================================
# Multi-atom adsorbate placement and discovery
#
# See dev/PLAN_multiatom_adsorbates.md for the full design.  This block
# parallels the single-atom API above but is deliberately separate: it
# never mutates G.graph["adsorption_sites"] / ["site_opt"] /
# ["context_cache"] *except* for the documented occupancy-blocking
# cross-write performed by register_adsorption_multi (PLAN §13).
# ===========================================================================

from itertools import product as _iproduct  # noqa: E402

# Forward reference – Reactant is only needed for typing.
from autokmc.reactants import Reactant  # noqa: E402


# ---------------------------------------------------------------------------
# Result + configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConfigOptResult:
    """Optimisation result for one multi-atom :class:`AdsorptionConfiguration`.

    Mirrors :class:`SiteOptResult` but per-anchor for connectivity bookkeeping.
    """
    config_key            : frozenset
    smiles                : str
    atoms_initial         : Atoms
    atoms_final           : Atoms
    energy                : float
    adsorption_energy     : float
    converged             : bool
    n_steps               : int
    connectivity          : dict        # {anchor_atom_idx -> ConnectivityStatus}
    actual_clique         : dict        # {anchor_atom_idx -> frozenset}
    matched_iso_class     : Optional[object] = field(default=None)
    intramolecular_intact : bool        = field(default=True)
    displacement          : float       = field(default=0.0)
    ads_indices           : list        = field(default_factory=list)
    opt_log               : str         = field(default="", repr=False)


@dataclass
class AdsorptionConfiguration:
    """One concrete multi-atom adsorption configuration on the surface.

    Parallels :class:`AdsorptionSite` but covers a *molecular* reactant:
    every anchor of the reactant is mapped onto a slab clique.
    """
    reactant_smiles    : str
    reactant           : Reactant
    anchor_clique_map  : dict                         # {anchor_idx -> frozenset(clique)}
    subgraph           : nx.Graph
    iso_class          : object                       # opaque hash repr (set by dedup)
    ads_positions      : dict                         # {reactant_atom_idx -> np.ndarray(3,)}
    adsorption_energy  : float                        = field(default=float("nan"))
    energy             : float                        = field(default=float("nan"))
    occupied           : bool                         = field(default=False)
    stable             : bool                         = field(default=True)
    reactive           : bool                         = field(default=True)
    migrated_to        : dict                         = field(default_factory=dict)
    neighbour_sites    : list                         = field(default_factory=list, repr=False)
    context_results    : dict                         = field(default_factory=dict, repr=False)
    current_result     : Optional[ConfigOptResult]    = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _config_key(anchor_clique_map: dict) -> frozenset:
    """Canonical hashable key for an :class:`AdsorptionConfiguration`."""
    return frozenset(frozenset(c) for c in anchor_clique_map.values())


def _surface_normal(G: nx.Graph, clique: frozenset) -> np.ndarray:
    """Return a unit vector pointing *away* from the slab/NP at *clique*.

    Strategy
    --------
    * If the cell is diagonal and PBC has at least one False axis, the
      non-periodic axis is taken as the surface normal direction.
    * Otherwise (nanoparticle / non-orthogonal cell), use the vector from
      the slab centroid (mean of all surface-typed nodes) to the clique
      centroid.
    """
    cell = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)

    pos = {n: np.asarray(d["position"], dtype=float)
           for n, d in G.nodes(data=True)}
    clique_pos = np.array([pos[a] for a in clique])
    centroid = clique_pos.mean(axis=0)

    # Diagonal-ish slab → use the non-periodic axis as the normal.
    is_diag = np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-6)
    if is_diag and not pbc.all():
        ax = int(np.argmin(pbc.astype(int)))    # first False axis
        # Decide sign: outward = away from the slab COM along that axis.
        slab_com_ax = np.mean([pos[n][ax] for n, d in G.nodes(data=True)
                               if d.get("type") == "surface"])
        sign = 1.0 if centroid[ax] >= slab_com_ax else -1.0
        n = np.zeros(3); n[ax] = sign
        return n

    # Fallback: clique centroid - structure centroid.
    surf_pos = np.array([pos[n] for n, d in G.nodes(data=True)
                         if d.get("type") == "surface"])
    if len(surf_pos) == 0:
        surf_pos = np.array(list(pos.values()))
    com = surf_pos.mean(axis=0)
    v = centroid - com
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / nrm


def _rigid_transform_for_anchor(
    reactant: Reactant,
    anchor_idx: int,
    target_pos: np.ndarray,
    surface_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, t)`` aligning *reactant* such that *anchor_idx* lands on
    *target_pos* and the molecule's "anchor outward axis" lines up with
    *surface_normal*.

    The molecular outward axis is the unit vector from the molecular
    centroid to *anchor_idx* (so the rest of the molecule points away from
    the surface).  Single-atom reactants get the identity rotation.
    """
    pts = reactant.atoms.get_positions()
    n_atoms = len(pts)
    if n_atoms == 1:
        return np.eye(3), target_pos - pts[anchor_idx]

    centroid = pts.mean(axis=0)
    axis = pts[anchor_idx] - centroid
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:
        # Anchor at centroid → fall back to identity rotation.
        R = np.eye(3)
    else:
        a = axis / nrm
        n = surface_normal / max(np.linalg.norm(surface_normal), 1e-12)
        R = _rotation_between(a, n)
    t = target_pos - R @ pts[anchor_idx]
    return R, t


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector *a* onto unit vector *b*."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180° rotation about any axis perpendicular to a.
        # Pick the smallest component of a as the seed for orthogonality.
        seed = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, seed)
        axis /= np.linalg.norm(axis)
        K = np.array([[    0, -axis[2],  axis[1]],
                      [ axis[2],      0, -axis[0]],
                      [-axis[1],  axis[0],     0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[    0, -v[2],  v[1]],
                  [ v[2],      0, -v[0]],
                  [-v[1],  v[0],     0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _apply_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply ``R @ p + t`` row-wise to *pts*."""
    return pts @ R.T + t


def _mic_distance(p: np.ndarray, q: np.ndarray, cell: np.ndarray,
                  cell_inv: np.ndarray | None, pbc: np.ndarray) -> float:
    return _mic_dist_vec(np.asarray(p), np.asarray(q), cell, cell_inv, pbc)


# ---------------------------------------------------------------------------
# Combined-subgraph builder + iso-dedup
# ---------------------------------------------------------------------------

def _build_config_subgraph(
    G: nx.Graph,
    reactant: Reactant,
    anchor_clique_map: dict,
    ads_positions: dict,
) -> nx.Graph:
    """Return ``G.copy()`` extended with every reactant atom + intramolecular
    edges + anchor↔clique bonds.  Reactant atoms occupy keys
    ``len(slab) + i`` for ``i`` in reactant atom index.
    """
    H = G.copy()
    n_slab = G.number_of_nodes()
    R = reactant.graph
    for ratom, attrs in R.nodes(data=True):
        node_id = n_slab + int(ratom)
        new_attrs = dict(attrs)
        if int(ratom) in ads_positions:
            new_attrs["position"] = np.asarray(ads_positions[int(ratom)]).copy()
        new_attrs["type"] = "adsorbate"
        new_attrs["index"] = node_id
        H.add_node(node_id, **new_attrs)
    for u, v in R.edges():
        H.add_edge(n_slab + int(u), n_slab + int(v))
    for anchor, clique in anchor_clique_map.items():
        for surf in clique:
            H.add_edge(n_slab + int(anchor), int(surf))
    return H


def _config_iso_ego(
    G: nx.Graph,
    reactant: Reactant,
    anchor_clique_map: dict,
) -> nx.Graph:
    """One-shell ego subgraph used to deduplicate configurations."""
    n_slab = G.number_of_nodes()
    # Build an augmented graph: G ∪ reactant nodes ∪ intramolecular edges
    # ∪ anchor-to-clique-member edges.
    H = G.copy()
    for ratom, attrs in reactant.graph.nodes(data=True):
        node_id = n_slab + int(ratom)
        new_attrs = dict(attrs)
        new_attrs["type"] = "adsorbate"
        new_attrs["index"] = node_id
        H.add_node(node_id, **new_attrs)
    for u, v in reactant.graph.edges():
        H.add_edge(n_slab + int(u), n_slab + int(v))
    for anchor, clique in anchor_clique_map.items():
        for surf in clique:
            H.add_edge(n_slab + int(anchor), int(surf))

    seed: set = set()
    for clique in anchor_clique_map.values():
        seed |= set(clique)
    for ratom in reactant.graph.nodes:
        seed.add(n_slab + int(ratom))

    visited = set(seed)
    next_shell: set = set()
    for n in seed:
        if n in H:
            next_shell.update(H.neighbors(n))
    visited |= next_shell

    # Surface labels feed the categorical match below; copy them onto the
    # ego nodes so the prefilter / isomorphism check sees them.
    ego = H.subgraph(visited).copy()
    surf_arr = G.graph.get("_surface_label_cache")
    return ego


def _config_iso_key(ego: nx.Graph) -> tuple:
    return _iso_prefilter_key(ego)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_configurations(
    G: nx.Graph,
    reactant: Reactant,
    *,
    n_shells: int = 1,
    max_anchors: int = 4,
    k_candidates: int = 8,
    anchor_distance_tol: float = 0.5,
    clash_factor: float = 0.75,
    bond_factor: float = 1.10,
    verbose: bool = False,
) -> list[AdsorptionConfiguration]:
    """Enumerate all geometrically-feasible multi-anchor configurations
    of *reactant* on the surface graph *G*.

    See ``dev/PLAN_multiatom_adsorbates.md`` §4–§6 / §9 for the
    algorithm and parameter semantics.

    Complexity
    ----------
    Worst case ``O(n_anchors! · k_candidates ** n_anchors)``.  Anchor
    orbit deduplication divides by ``|aut(reactant.graph)|``; the
    clash + anchor-distance filters dominate in practice.

    Raises
    ------
    NotImplementedError
        If ``n_shells != 1`` (per plan v1).
    ValueError
        If the reactant has more than ``max_anchors`` anchors.
    """
    if n_shells != 1:
        raise NotImplementedError(
            "Multi-atom adsorbate placement only supports n_shells=1 in v1."
        )

    anchors = list(reactant.anchor_atoms)
    if len(anchors) > max_anchors:
        raise ValueError(
            f"Reactant {reactant.smiles!r} has {len(anchors)} anchors "
            f"(> max_anchors={max_anchors}). v1 caps multi-anchor "
            "scaling — raise max_anchors explicitly to override."
        )

    # Canonical anchor ordering: orbit id ascending, then atom index.
    orbit_of = reactant.anchor_orbit
    anchors.sort(key=lambda a: (orbit_of.get(a, -1), a))

    cell = np.array(G.graph.get("cell", np.eye(3)), dtype=float)
    pbc  = np.asarray(G.graph.get("pbc", [True, True, False]), dtype=bool)
    cell_inv = None
    if pbc.any():
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            cell_inv = None

    # Per-element flat list of (k, clique, position) for fast lookup.
    sites = G.graph.get("sites", {})
    site_pos = G.graph.get("site_positions", {})

    def _flat_sites(elem: str) -> list[tuple[int, frozenset, np.ndarray]]:
        out: list[tuple[int, frozenset, np.ndarray]] = []
        e_sites = sites.get(elem, {})
        e_pos = site_pos.get(elem, {})
        for k, cliques in e_sites.items():
            positions = e_pos.get(k, [])
            for i, c in enumerate(cliques):
                p = (np.asarray(positions[i]) if i < len(positions)
                     else np.mean([G.nodes[a]["position"] for a in c], axis=0))
                out.append((k, c, p))
        return out

    elements = [reactant.graph.nodes[a]["element"] for a in anchors]
    flat_per_anchor: dict[int, list[tuple[int, frozenset, np.ndarray]]] = {
        a: _flat_sites(e) for a, e in zip(anchors, elements)
    }
    if any(not v for v in flat_per_anchor.values()):
        return []

    # Reactant intramolecular pair distances (used by the early reject).
    rpts = reactant.atoms.get_positions()
    intra_d = {(i, j): float(np.linalg.norm(rpts[i] - rpts[j]))
               for i in anchors for j in anchors if i < j}

    intramol_pairs = {frozenset((u, v)) for u, v in reactant.graph.edges()}
    r_cov_slab = {n: float(d["covalent_radius"])
                  for n, d in G.nodes(data=True)}
    r_cov_react = {a: float(reactant.graph.nodes[a]["covalent_radius"])
                   for a in reactant.graph.nodes}

    slab_positions = {n: np.asarray(d["position"], dtype=float)
                      for n, d in G.nodes(data=True)
                      if d.get("type") in ("surface", "bulk")}

    # ---- Enumerate ------------------------------------------------------
    seen_iso: dict[tuple, AdsorptionConfiguration] = {}
    seen_keys: set = set()
    configs: list[AdsorptionConfiguration] = []

    first = anchors[0]
    rest  = anchors[1:]

    for _k0, clique0, pos0 in flat_per_anchor[first]:
        normal = _surface_normal(G, clique0)
        R, t = _rigid_transform_for_anchor(reactant, first, pos0, normal)

        # Predict positions of all reactant atoms after the rigid transform.
        all_pos = _apply_transform(rpts, R, t)

        # For every remaining anchor, gather candidates within tolerance
        # of its predicted Cartesian position.
        candidate_lists = []
        feasible = True
        for a in rest:
            pred = all_pos[a]
            cands = []
            for ks, cl, cp in flat_per_anchor[a]:
                if cl == clique0:
                    continue   # disallow identical clique (PLAN §4d note)
                d = _mic_distance(pred, cp, cell, cell_inv, pbc)
                if d <= anchor_distance_tol:
                    cands.append((d, cl, cp))
            cands.sort(key=lambda x: x[0])
            cands = cands[:k_candidates]
            if not cands:
                feasible = False
                break
            candidate_lists.append(cands)
        if not feasible:
            continue

        if not rest:
            # Single-anchor degenerate case
            assignments = [tuple()]
        else:
            assignments = list(_iproduct(*candidate_lists))

        for asgn in assignments:
            anchor_clique_map: dict = {first: clique0}
            for a, (_d, cl, _cp) in zip(rest, asgn):
                anchor_clique_map[a] = cl
            # Reject duplicate cliques across anchors.
            cliques_used = list(anchor_clique_map.values())
            if len(set(cliques_used)) != len(cliques_used):
                continue
            # Anchor-anchor MIC-distance prefilter against intramolecular dist.
            ok = True
            cliques_pos = {}
            for a, cl in anchor_clique_map.items():
                if a == first:
                    cliques_pos[a] = pos0
                else:
                    # find position from flat list
                    for ks, c2, cp2 in flat_per_anchor[a]:
                        if c2 == cl:
                            cliques_pos[a] = cp2
                            break
            for i in range(len(anchors)):
                for j in range(i + 1, len(anchors)):
                    ai, aj = anchors[i], anchors[j]
                    d = _mic_distance(cliques_pos[ai], cliques_pos[aj],
                                      cell, cell_inv, pbc)
                    if abs(d - intra_d[(min(ai, aj), max(ai, aj))]) > anchor_distance_tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Clash filter: reactant atoms vs slab atoms.
            ads_positions = {int(i): all_pos[int(i)].copy()
                             for i in range(len(rpts))}
            clash = False
            for ri, rp in ads_positions.items():
                rc = r_cov_react[ri]
                for sn, sp in slab_positions.items():
                    if frozenset((sn, ri)) in intramol_pairs:
                        continue
                    cutoff = clash_factor * (rc + r_cov_slab[sn])
                    # Allow legitimate anchor-clique bond contacts.
                    if (ri in anchor_clique_map
                            and sn in anchor_clique_map[ri]):
                        continue
                    d = _mic_distance(rp, sp, cell, cell_inv, pbc)
                    if d < cutoff:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                continue

            ckey = _config_key(anchor_clique_map)
            if ckey in seen_keys:
                continue
            seen_keys.add(ckey)

            subg = _build_config_subgraph(G, reactant, anchor_clique_map, ads_positions)
            ego  = _config_iso_ego(G, reactant, anchor_clique_map)
            iso_key = _config_iso_key(ego)

            cfg = AdsorptionConfiguration(
                reactant_smiles    = reactant.smiles,
                reactant           = reactant,
                anchor_clique_map  = dict(anchor_clique_map),
                subgraph           = subg,
                iso_class          = iso_key,
                ads_positions      = ads_positions,
            )
            configs.append(cfg)

            # Iso-dedup: keep first representative per (prefilter, iso) key.
            existing = seen_iso.get(iso_key)
            if existing is None:
                seen_iso[iso_key] = cfg
            else:
                node_match = isomorphism.categorical_node_match("element", "X")
                gm = isomorphism.GraphMatcher(
                    ego, _config_iso_ego(G, reactant, existing.anchor_clique_map),
                    node_match=node_match,
                )
                if gm.is_isomorphic():
                    cfg.iso_class = existing.iso_class

    if verbose:
        n_unique = len({id(seen_iso[k]) for k in seen_iso})
        print(f"enumerate_configurations: smiles={reactant.smiles!r}  "
              f"anchors={anchors}  total={len(configs)}  "
              f"unique_iso_classes={n_unique}")

    return configs


# ---------------------------------------------------------------------------
# Optimisation pipeline (Stage 2/3 for multi-atom)
# ---------------------------------------------------------------------------

def _intramolecular_intact(
    atoms_final: Atoms,
    reactant: Reactant,
    n_slab: int,
    bond_factor: float = 1.10,
) -> bool:
    """Check that every intramolecular bond in *reactant.graph* is still
    present in *atoms_final* (and no new internal bonds appeared)."""
    pts = atoms_final.get_positions()
    R = reactant.graph
    rad = {a: float(R.nodes[a]["covalent_radius"]) for a in R.nodes}
    n_react = R.number_of_nodes()
    expected = {frozenset((u, v)) for u, v in R.edges()}
    actual: set = set()
    for i in range(n_react):
        pi = pts[n_slab + i]
        for j in range(i + 1, n_react):
            pj = pts[n_slab + j]
            cutoff = bond_factor * (rad[i] + rad[j])
            if float(np.linalg.norm(pi - pj)) <= cutoff:
                actual.add(frozenset((i, j)))
    return expected == actual


def optimise_unique_configurations(
    G: nx.Graph,
    reactant: Reactant,
    atoms: Atoms,
    calculator,
    *,
    n_shells: int = 1,
    bond_factor: float = 1.10,
    fmax: float = 0.05,
    steps: int = 500,
    e_surface: float | None = None,
    e_reactant: float | None = None,
    max_anchors: int = 4,
    k_candidates: int = 8,
    anchor_distance_tol: float = 0.5,
    clash_factor: float = 0.75,
    logfile: str | None = None,
    verbose: bool = True,
) -> list[ConfigOptResult]:
    """Multi-atom analogue of :func:`optimise_unique_sites`.

    Pipeline
    --------
    1. Enumerate all configurations (:func:`enumerate_configurations`).
    2. For each iso-class representative: assemble trial Atoms, relax with
       LBFGS, check per-anchor connectivity + intramolecular bond
       preservation, build a :class:`ConfigOptResult`.
    3. Stage-3 expansion: every configuration sharing the iso-class
       inherits the representative's result; populated into
       ``G.graph['adsorption_configs'][smiles][n_shells]``.
    4. Compute neighbour configurations for the discovery loop.

    Reference energies: ``E_ads = E_total − (E_reactant + E_surface)`` where
    ``E_reactant = reactant.energy`` (relaxed gas-phase) by default.
    """
    if n_shells != 1:
        raise NotImplementedError(
            "Multi-atom adsorbate optimisation only supports n_shells=1 in v1."
        )

    smiles = reactant.smiles

    # ── References ────────────────────────────────────────────────────────
    if e_surface is None:
        slab_ref = atoms.copy()
        slab_ref.calc = calculator
        try:
            e_surface = float(slab_ref.get_potential_energy())
        except Exception as exc:
            warnings.warn(
                f"Could not compute surface energy: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            e_surface = float("nan")

    if e_reactant is None:
        e_reactant = float(reactant.energy)

    e_ref = float(e_reactant) + float(e_surface)

    # ── Enumerate candidates ─────────────────────────────────────────────
    configs = enumerate_configurations(
        G, reactant,
        n_shells=n_shells,
        max_anchors=max_anchors,
        k_candidates=k_candidates,
        anchor_distance_tol=anchor_distance_tol,
        clash_factor=clash_factor,
        bond_factor=bond_factor,
        verbose=verbose,
    )

    # Group by iso_class (the prefilter/iso key already deduplicated).
    by_iso: dict = {}
    for cfg in configs:
        by_iso.setdefault(cfg.iso_class, []).append(cfg)

    n_slab = G.number_of_nodes()
    results: list[ConfigOptResult] = []

    if verbose:
        print(f"optimise_unique_configurations: smiles={smiles!r}  "
              f"n_configs={len(configs)}  unique_iso={len(by_iso)}  "
              f"E_surface={e_surface:.4f}  E_reactant={e_reactant:.4f}")

    # ── Relax one representative per iso-class ───────────────────────────
    iso_to_result: dict = {}
    for iso_key, cfgs in by_iso.items():
        rep = cfgs[0]

        # Build trial Atoms = host + reactant atoms at rigid-aligned positions.
        trial = atoms.copy()
        ads_indices = list(range(n_slab, n_slab + len(reactant.atoms)))
        order = sorted(rep.ads_positions.keys())
        symbols = [reactant.atoms[i].symbol for i in order]
        positions = np.array([rep.ads_positions[i] for i in order])
        trial += Atoms(symbols=symbols, positions=positions)
        trial_init = trial.copy()
        trial.calc = calculator

        log = os.devnull if logfile is None else logfile.format(
            smiles=smiles, iso=hash(iso_key) & 0xFFFFFFFF
        )

        opt = LBFGS(trial, logfile=log)
        try:
            opt.run(fmax=fmax, steps=steps)
            converged = opt.converged()
            n_steps = opt.get_number_of_steps()
            energy = float(trial.get_potential_energy())
        except Exception as exc:
            warnings.warn(
                f"LBFGS failed for config (iso={iso_key}): {exc}",
                RuntimeWarning, stacklevel=2,
            )
            converged, n_steps, energy = False, 0, float("nan")

        ads_energy = energy - e_ref

        # Per-anchor connectivity.
        per_conn: dict = {}
        per_actual: dict = {}
        all_ok = True
        for anchor, planned in rep.anchor_clique_map.items():
            ads_node_idx = n_slab + int(anchor)
            r_cov_a = float(reactant.graph.nodes[anchor]["covalent_radius"])
            actual = find_actual_clique(
                trial, ads_node_idx, G, r_cov_a, bond_factor=bond_factor,
            )
            per_actual[anchor] = actual
            status = check_connectivity(
                trial, ads_node_idx, planned, r_cov_a, G,
                bond_factor=bond_factor,
            )
            per_conn[anchor] = status
            if status != ConnectivityStatus.OK:
                all_ok = False

        intact = _intramolecular_intact(trial, reactant, n_slab,
                                        bond_factor=bond_factor)

        # Displacement of the reactant centroid (MIC-aware via anchor 0).
        first = sorted(rep.anchor_clique_map.keys())[0]
        disp = _ads_displacement(
            trial_init, trial, n_slab + int(first), G,
        )

        result = ConfigOptResult(
            config_key            = _config_key(rep.anchor_clique_map),
            smiles                = smiles,
            atoms_initial         = trial_init,
            atoms_final           = trial,
            energy                = energy,
            adsorption_energy     = ads_energy,
            converged             = converged,
            n_steps               = n_steps,
            connectivity          = per_conn,
            actual_clique         = per_actual,
            matched_iso_class     = None,
            intramolecular_intact = intact,
            displacement          = disp,
            ads_indices           = ads_indices,
            opt_log               = log if logfile else "",
        )
        results.append(result)
        iso_to_result[iso_key] = result

        stable = all_ok and intact
        # Propagate to every config in the iso-class.
        for cfg in cfgs:
            cfg.energy = energy
            cfg.adsorption_energy = ads_energy
            cfg.stable = stable
            cfg.reactive = stable
            cfg.current_result = result
            cfg.context_results[frozenset()] = result
            if not stable:
                cfg.migrated_to = dict(per_actual)

        if verbose:
            ads_str = (f"{ads_energy:+.4f}" if not np.isnan(ads_energy)
                       else "     nan")
            print(f"  iso={hash(iso_key) & 0xFFFF:04x}  members={len(cfgs)}  "
                  f"conv={'✓' if converged else '✗'}  steps={n_steps:>4}  "
                  f"E={energy:>10.4f}  E_ads={ads_str}  "
                  f"intact={intact}  stable={stable}")

    # ── Register configurations on the graph ─────────────────────────────
    cfg_dict: dict = {}
    for cfg in configs:
        cfg_dict[_config_key(cfg.anchor_clique_map)] = cfg
    (G.graph
       .setdefault("adsorption_configs", {})
       .setdefault(smiles, {})[n_shells]) = cfg_dict
    (G.graph
       .setdefault("unique_configs", {})
       .setdefault(smiles, {})[n_shells]) = [
        by_iso[k][0] for k in by_iso
    ]
    (G.graph
       .setdefault("config_opt", {})
       .setdefault(smiles, {})[n_shells]) = results

    compute_neighbour_configurations(G, smiles, n_shells=n_shells)

    if verbose:
        print(f"  adsorption_configs registered: {len(cfg_dict)} → "
              f"G.graph['adsorption_configs'][{smiles!r}][{n_shells}]")

    return results


# ---------------------------------------------------------------------------
# Neighbour configurations
# ---------------------------------------------------------------------------

def compute_neighbour_configurations(
    G: nx.Graph,
    smiles: str,
    n_shells: int = 1,
) -> dict:
    """Populate :attr:`AdsorptionConfiguration.neighbour_sites`.

    Two configurations are neighbours iff the union of their anchor cliques
    share at least one surface atom (consistent with single-atom
    :func:`compute_neighbour_sites`).
    """
    cfgs: dict = (
        G.graph.get("adsorption_configs", {}).get(smiles, {}).get(n_shells, {})
    )
    if not cfgs:
        raise KeyError(
            f"No adsorption_configs for smiles={smiles!r} at n_shells={n_shells}. "
            "Call optimise_unique_configurations first."
        )

    # atom -> [config_key]
    atom_to_cfg: dict = {}
    for ckey, cfg in cfgs.items():
        for clique in cfg.anchor_clique_map.values():
            for a in clique:
                atom_to_cfg.setdefault(int(a), []).append(ckey)

    neighbours: dict = {}
    for ckey, cfg in cfgs.items():
        seen: set = set()
        for clique in cfg.anchor_clique_map.values():
            for a in clique:
                for other in atom_to_cfg.get(int(a), []):
                    if other != ckey:
                        seen.add(other)
        cfg.neighbour_sites = list(seen)
        neighbours[ckey] = list(seen)
    return neighbours


# ---------------------------------------------------------------------------
# Discovery / KMC event hooks (multi-atom)
# ---------------------------------------------------------------------------

def _build_trial_atoms_multi(
    host_atoms: Atoms,
    occupied_configs: list,
    target_cfg: AdsorptionConfiguration,
) -> tuple[Atoms, list[int]]:
    """Assemble host + every occupied reactant + target reactant.

    Returns ``(trial, target_ads_indices)`` — the indices of the target
    reactant's atoms inside *trial*.
    """
    trial = host_atoms.copy()
    for occ in occupied_configs:
        result = occ.current_result
        if result is not None and result.atoms_final is not None:
            n_slab_now = len(trial)
            # Pull reactant atoms out of the cached final structure.
            r = occ.reactant
            n_react = len(r.atoms)
            ads_part = result.atoms_final[len(result.atoms_final) - n_react:]
            trial += ads_part
        else:
            # Fallback: rigid-aligned ads_positions
            r = occ.reactant
            order = sorted(occ.ads_positions.keys())
            symbols = [r.atoms[i].symbol for i in order]
            positions = np.array([occ.ads_positions[i] for i in order])
            trial += Atoms(symbols=symbols, positions=positions)

    start = len(trial)
    r = target_cfg.reactant
    order = sorted(target_cfg.ads_positions.keys())
    symbols = [r.atoms[i].symbol for i in order]
    positions = np.array([target_cfg.ads_positions[i] for i in order])
    trial += Atoms(symbols=symbols, positions=positions)
    target_indices = list(range(start, start + len(order)))
    return trial, target_indices


def discover_context_configuration(
    G: nx.Graph,
    smiles: str,
    config_key: frozenset,
    host_atoms: Atoms,
    calculator,
    *,
    n_shells: int = 1,
    bond_factor: float = 1.10,
    fmax: float = 0.05,
    steps: int = 500,
    logfile: str | None = None,
    verbose: bool = False,
) -> ConfigOptResult:
    """Multi-atom analogue of :func:`discover_context_site`.

    Two-tier cache lookup (per-config exact-occupancy → global
    iso-deduplicated → calculator miss).  Stores the result on both the
    target config's ``context_results`` and the global
    ``G.graph['context_cache_multi']`` cache.
    """
    cfgs: dict = (
        G.graph["adsorption_configs"][smiles][n_shells]
    )
    cfg = cfgs[config_key]

    occupied_neighbours = [
        cfgs[k] for k in cfg.neighbour_sites
        if k in cfgs and cfgs[k].occupied
    ]
    occ_key = frozenset(_config_key(o.anchor_clique_map)
                        for o in occupied_neighbours)

    cached = cfg.context_results.get(occ_key)
    if cached is not None:
        cfg.current_result = cached
        return cached

    # Global iso-dedup cache: keyed by (cfg.iso_class, tuple-of-occupied-iso).
    occ_iso_key = (cfg.iso_class,
                   tuple(sorted(hash(o.iso_class) for o in occupied_neighbours)))
    global_cache = (
        G.graph
         .setdefault("context_cache_multi", {})
         .setdefault(smiles, {})
         .setdefault(n_shells, {})
    )
    cached = global_cache.get(occ_iso_key)
    if cached is not None:
        cfg.context_results[occ_key] = cached
        cfg.current_result = cached
        return cached

    # ── Miss: full relaxation ────────────────────────────────────────────
    bootstrap = cfg.context_results.get(frozenset())
    if bootstrap is None:
        raise RuntimeError(
            "Cannot infer reference energies: no clean-surface bootstrap "
            "result is cached for this config."
        )
    e_ref = float(bootstrap.energy) - float(bootstrap.adsorption_energy)

    trial, target_indices = _build_trial_atoms_multi(
        host_atoms, occupied_neighbours, cfg,
    )
    trial_init = trial.copy()
    trial.calc = calculator

    log = os.devnull if logfile is None else logfile
    opt = LBFGS(trial, logfile=log)
    try:
        opt.run(fmax=fmax, steps=steps)
        converged = opt.converged()
        n_steps = opt.get_number_of_steps()
        energy = float(trial.get_potential_energy())
    except Exception as exc:
        warnings.warn(
            f"discover_context_configuration LBFGS failed: {exc}",
            RuntimeWarning, stacklevel=2,
        )
        converged, n_steps, energy = False, 0, float("nan")

    ads_energy = energy - e_ref

    # Per-anchor connectivity for the *target* configuration.
    n_slab_in_trial = target_indices[0]
    per_conn: dict = {}
    per_actual: dict = {}
    all_ok = True
    for anchor, planned in cfg.anchor_clique_map.items():
        idx = n_slab_in_trial + int(anchor)
        r_cov_a = float(cfg.reactant.graph.nodes[anchor]["covalent_radius"])
        actual = find_actual_clique(
            trial, idx, G, r_cov_a, bond_factor=bond_factor,
        )
        per_actual[anchor] = actual
        status = check_connectivity(
            trial, idx, planned, r_cov_a, G,
            bond_factor=bond_factor,
        )
        per_conn[anchor] = status
        if status != ConnectivityStatus.OK:
            all_ok = False

    intact = _intramolecular_intact(
        trial, cfg.reactant, n_slab_in_trial, bond_factor=bond_factor,
    )

    disp = _ads_displacement(
        trial_init, trial, n_slab_in_trial, G,
    )

    result = ConfigOptResult(
        config_key            = config_key,
        smiles                = smiles,
        atoms_initial         = trial_init,
        atoms_final           = trial,
        energy                = energy,
        adsorption_energy     = ads_energy,
        converged             = converged,
        n_steps               = n_steps,
        connectivity          = per_conn,
        actual_clique         = per_actual,
        matched_iso_class     = None,
        intramolecular_intact = intact,
        displacement          = disp,
        ads_indices           = target_indices,
        opt_log               = log if logfile else "",
    )

    cfg.context_results[occ_key] = result
    cfg.current_result = result
    global_cache[occ_iso_key] = result

    if verbose:
        print(f"discover_context_configuration: smiles={smiles!r} "
              f"|N_occ|={len(occupied_neighbours)}  "
              f"E_ads={ads_energy:+.4f}  "
              f"stable={all_ok and intact}  steps={n_steps}")

    return result


def update_reactive_flags_multi(
    G: nx.Graph,
    smiles: str,
    n_shells: int,
    changed_key: frozenset,
) -> set:
    """Recompute ``reactive`` for *changed_key* and its neighbour configs."""
    cfgs: dict = G.graph["adsorption_configs"][smiles][n_shells]
    affected: set = {changed_key}
    if changed_key in cfgs:
        affected.update(cfgs[changed_key].neighbour_sites)

    flipped: set = set()
    for ck in affected:
        cfg = cfgs.get(ck)
        if cfg is None:
            continue
        occ_key = frozenset(
            k for k in cfg.neighbour_sites
            if k in cfgs and cfgs[k].occupied
        )
        cached = cfg.context_results.get(occ_key)
        new_reactive = (
            cfg.stable and not cfg.occupied and cached is not None
        )
        if cached is not None:
            cfg.current_result = cached
        if new_reactive != cfg.reactive:
            cfg.reactive = new_reactive
            flipped.add(ck)
    return flipped


def register_adsorption_multi(
    G: nx.Graph,
    smiles: str,
    config_key: frozenset,
    host_atoms: Atoms,
    calculator,
    *,
    n_shells: int = 1,
    discover: bool = True,
    **discover_kwargs,
) -> set:
    """KMC event hook: a multi-atom reactant adsorbed at *config_key*.

    Side-effects (PLAN §13):

    * Marks every clique covered by an anchor of *config_key* as
      ``occupied=True`` in the **single-atom**
      ``G.graph['adsorption_sites'][element][1]`` dict so single-atom
      discovery treats them as blockers.
    * Marks the configuration ``occupied=True``, ``reactive=False``.
    * If *discover*: re-runs :func:`discover_context_configuration` for
      every stable, vacant neighbour configuration.
    * Returns the set of config_keys whose ``reactive`` flag flipped.
    """
    cfgs: dict = G.graph["adsorption_configs"][smiles][n_shells]
    cfg = cfgs[config_key]
    if cfg.occupied:
        warnings.warn(
            f"register_adsorption_multi: config {config_key} already occupied.",
            RuntimeWarning, stacklevel=2,
        )
    cfg.occupied = True
    cfg.reactive = False

    # Cross-write to single-atom adsorption_sites (PLAN §13).
    single = G.graph.get("adsorption_sites", {})
    for anchor, clique in cfg.anchor_clique_map.items():
        elem = cfg.reactant.graph.nodes[anchor]["element"]
        site_dict = single.get(elem, {}).get(n_shells, {})
        s = site_dict.get(clique)
        if s is not None:
            s.occupied = True

    if discover:
        for nbr_key in cfg.neighbour_sites:
            nbr = cfgs.get(nbr_key)
            if nbr is None or nbr.occupied or not nbr.stable:
                continue
            discover_context_configuration(
                G, smiles, nbr_key,
                host_atoms, calculator,
                n_shells=n_shells,
                **discover_kwargs,
            )

    return update_reactive_flags_multi(G, smiles, n_shells, config_key)


def register_desorption_multi(
    G: nx.Graph,
    smiles: str,
    config_key: frozenset,
    *,
    n_shells: int = 1,
) -> set:
    """KMC event hook: a multi-atom reactant desorbed from *config_key*.

    Inverse of :func:`register_adsorption_multi`.
    """
    cfgs: dict = G.graph["adsorption_configs"][smiles][n_shells]
    cfg = cfgs[config_key]
    if not cfg.occupied:
        warnings.warn(
            f"register_desorption_multi: config {config_key} was not occupied.",
            RuntimeWarning, stacklevel=2,
        )
    cfg.occupied = False

    single = G.graph.get("adsorption_sites", {})
    for anchor, clique in cfg.anchor_clique_map.items():
        elem = cfg.reactant.graph.nodes[anchor]["element"]
        site_dict = single.get(elem, {}).get(n_shells, {})
        s = site_dict.get(clique)
        if s is not None:
            s.occupied = False

    return update_reactive_flags_multi(G, smiles, n_shells, config_key)
