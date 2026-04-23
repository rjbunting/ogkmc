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
* :func:`optimise_unique_sites`      -- main entry point
* :func:`calculate_gas_phase_energy` -- compute isolated-atom reference energy
* :class:`SiteOptResult`             -- per-iso-class result record
* :class:`AdsorptionSite`            -- per-clique-instance record for KMC
* :class:`ConnectivityStatus`        -- connectivity check outcome enum
* :func:`place_adsorbate`            -- low-level: append adsorbate to Atoms copy
* :func:`find_actual_clique`         -- low-level: bonds after relaxation
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

                ads_sites[clique] = AdsorptionSite(
                    clique            = clique,
                    iso_class         = iso,
                    result            = res,
                    adsorption_energy = res.adsorption_energy,
                    occupied          = False,
                    ads_position      = ads_pos,
                    subgraph          = H,
                )

    (G.graph
       .setdefault("adsorption_sites", {})
       .setdefault(element, {})[n_shells]) = ads_sites

    if verbose:
        n_vacant = sum(1 for s in ads_sites.values() if not s.occupied)
        print(f"\n  adsorption_sites registered: {len(ads_sites)} "
              f"({n_vacant} vacant) → G.graph['adsorption_sites']['{element}'][{n_shells}]")
        print(f"  Each AdsorptionSite.subgraph = copy(G) + adsorbate node "
              f"(node {G.number_of_nodes()}) bonded to its clique.")
        print(f"  G.graph['sites'] / 'unique_sites' / 'site_positions' unchanged.")

    return results

