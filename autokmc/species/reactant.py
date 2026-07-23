"""
autokmc.species.reactant
=================
Convert a SMILES string into an optimised 3-D :class:`~ase.Atoms` object,
its atom-connectivity graph, and the metadata needed to place it onto a
surface as a multi-atom adsorbate.

Pipeline
--------
1. Parse SMILES with RDKit and embed a 3-D conformer (ETKDGv3 + MMFF94).
2. Optionally refine with an ASE calculator via L-BFGS and stamp the
   relaxed total energy onto the :class:`Reactant`.
3. Tag every atom as ``surface = 2`` (molecules will adsorb onto a
   surface; they are neither bulk nor surface themselves).
4. Build a :class:`networkx.Graph` via :func:`~autokmc.core.graph.build_graph`.
5. Compute intramolecular automorphism orbits (per element) — the
   "unique nodes" used for de-duplicating equivalent anchor permutations
   in :mod:`autokmc.sites.adsorbate`.
6. Run a convex-hull pass to identify which atoms are *exposed* and
   therefore eligible to bond to a surface (the **anchor atoms**).

See ``dev/PLAN_multiatom_adsorbates.md`` for the full design.

Typical usage
-------------
::

    from ase.calculators.emt import EMT
    from autokmc.species.reactant import build_reactant

    co = build_reactant("[C-]#[O+]", calculator=EMT())
    print(co.atoms.get_chemical_formula())   # CO
    print(co.energy)                         # eV  (nan without calculator)
    print(co.unique_nodes)                   # {'C': [[0]], 'O': [[1]]}
    print(co.anchor_atoms)                   # [0, 1]   (diatomic → both)

Dependencies
------------
* RDKit  (``conda install -c conda-forge rdkit`` or ``pip install rdkit``)
* ASE
* networkx
* scipy  (for :class:`scipy.spatial.ConvexHull`)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

from ase import Atoms
from ase.optimize import LBFGS

from autokmc.core.graph import build_graph
from autokmc.core.constants import NL_MULT_DEFAULT, RANDOM_SEED
from autokmc.io.calculators import acquire_calculator
from autokmc.species.smiles import smiles_to_dirname
from autokmc.utils.logging import get_logger
from autokmc.utils.rdkit_logging import silence_rdkit_warnings

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reactant:
    """A molecule represented as an optimised :class:`~ase.Atoms` object,
    its atom-connectivity graph, and the metadata needed to place it onto
    a surface as a multi-atom adsorbate.

    Attributes
    ----------
    smiles : str
        SMILES string the structure was built from.
    atoms : Atoms
        3-D geometry (optionally calculator-relaxed).  Every atom has
        ``atoms.arrays["surface"] = 2`` (adsorbate) because molecules
        are neither bulk nor surface — they will adsorb onto the surface.
    graph : nx.Graph
        Connectivity graph built by :func:`~autokmc.core.graph.build_graph`.
        Node attributes: ``element``, ``position``, ``index``,
        ``type`` (always ``"adsorbate"``), ``covalent_radius``.
    energy : float
        Total energy of the relaxed gas-phase reactant in eV.  ``nan`` if
        no calculator was supplied to :func:`build_reactant` (or the
        single-point evaluation failed).  Used as the gas-phase reference
        ``E_gas`` in :func:`autokmc.sites.adsorbate.optimise_unique_configurations`.
    unique_nodes : dict[str, list[list[int]]]
        Orbits of the intramolecular automorphism group, keyed by
        element.  Two atom indices in the same orbit are equivalent under
        the molecule's own graph symmetry (with element labels matched)
        and therefore produce identical configurations when used as
        anchors.  Computed by :func:`find_unique_atoms`.
    anchor_atoms : list[int]
        Atom indices that are exposed on the convex hull of the relaxed
        geometry and therefore eligible to bond to the surface.  For
        diatomics and other degenerate-hull cases (``QHullError``) every
        atom is returned.  Computed by :func:`find_anchor_atoms`.
    anchor_orbit : dict[int, int]
        Anchor index → orbit id.  Anchors sharing an id belong to the
        same intramolecular orbit and produce equivalent placements when
        the configuration enumerator deduplicates by anchor permutation.
    """
    smiles       : str
    atoms        : Atoms
    graph        : nx.Graph
    energy       : float                       = field(default=float("nan"))
    unique_nodes : dict                        = field(default_factory=dict)
    anchor_atoms : list                        = field(default_factory=list)
    anchor_orbit : dict                        = field(default_factory=dict)
    # ── Free-energy / vibrational fields (populated by autokmc.thermo.free_energy) ──
    #: Gibbs free-energy correction relative to ``energy`` (eV) at the
    #: simulation T / p.  ``nan`` when free-energy mode is disabled.
    g_correction : float                       = field(default=float("nan"))
    #: Absolute gas-phase Gibbs free energy (eV) — equal to ``energy +
    #: g_correction``.  Cached so the rate code does not have to redo the
    #: addition.
    gibbs_energy : float                       = field(default=float("nan"))
    #: Zero-point energy (eV).
    zpe          : float                       = field(default=float("nan"))
    #: Standard-state entropy (eV/K).
    entropy      : float                       = field(default=float("nan"))
    #: Real vibrational mode energies (eV).
    frequencies_ev : list                      = field(default_factory=list)
    #: Imaginary / spurious low-mode energies (eV) — kept for audit.
    imaginary_ev   : list                      = field(default_factory=list)
    #: Partial pressure (bar) of this reactant — multiplies the
    #: adsorption rate in :func:`autokmc.reactions.adsorption._energetics_cached`
    #: so the persisted ΔG / barrier remain at the 1-bar reference.
    partial_pressure_bar : float               = 1.0
    #: Free-floating dict for thermochemistry metadata, including geometry,
    #: rotational-symmetry inference, spin, temperature, and pressure.
    thermo_meta  : dict                        = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smiles_to_atoms(smiles: str, *, add_hydrogens: bool = True) -> Atoms:
    """Convert a SMILES string to a 3-D :class:`~ase.Atoms` object.

    Uses RDKit ETKDGv3 for conformer embedding followed by MMFF94 force-field
    minimisation to give a reasonable starting geometry.

    Parameters
    ----------
    smiles : str
    add_hydrogens : bool
        Whether to add explicit hydrogens.  Default ``True``.  Hydrogens
        are only added to atoms whose SMILES specification implies them
        (i.e. those with a non-zero implicit-H count); explicit-H atoms
        and atoms with closed valences (``[O]``, ``[Au]``, …) are left
        untouched.  Pass ``add_hydrogens=False`` to skip this entirely.

    Returns
    -------
    Atoms
        Non-periodic structure with no calculator attached.
    """
    silence_rdkit_warnings()
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise ImportError(
            "RDKit is required for SMILES parsing.  "
            "Install it with:  conda install -c conda-forge rdkit"
        ) from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    if add_hydrogens:
        # Only add Hs to atoms that *want* them — i.e. those with a
        # non-zero implicit-H count given the SMILES.  This matches the
        # SMILES author's intent: e.g. "[O]" stays as a bare O, but "O"
        # gets two Hs (water).  Build a per-atom mask before calling
        # AddHs so atoms with closed/forced valences are preserved.
        only_atoms = [a.GetIdx() for a in mol.GetAtoms()
                      if a.GetNumImplicitHs() > 0 or a.GetNumExplicitHs() > 0]
        if only_atoms:
            mol = Chem.AddHs(mol, onlyOnAtoms=only_atoms)
    else:
        # Even when add_hydrogens=False, always materialise H atoms that
        # are **explicitly** specified in the bracket SMILES notation
        # (GetNumExplicitHs() > 0).  These are genuinely part of the
        # molecular formula — for example, the product of coupling [O]+[H]
        # is written by RDKit as "[OH]" where the H is an explicit H count
        # on O, not an implicit valence-fill H.  Skipping AddHs for these
        # atoms would give a 1-atom (bare O) Reactant for a 2-atom (O-H)
        # molecule, causing |A|+|B| ≠ |C| mismatches in bond NEB checks.
        # Atoms with only implicit H (e.g. unreacted [O] radical) have
        # GetNumExplicitHs()==0 and are left untouched.
        only_explicit = [a.GetIdx() for a in mol.GetAtoms()
                         if a.GetNumExplicitHs() > 0]
        if only_explicit:
            mol = Chem.AddHs(mol, onlyOnAtoms=only_explicit)

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        fallback = AllChem.EmbedParameters()
        fallback.randomSeed = RANDOM_SEED
        fallback.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, fallback) == -1:
            raise ValueError(
                f"RDKit could not embed a 3D conformer for SMILES: {smiles!r}. "
                "Both ETKDGv3 and the random fallback embedder failed. "
                "Try a different SMILES representation or simplify the molecule."
            )

    # Quick MMFF94 pre-relaxation in RDKit before handing to ASE
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    conf      = mol.GetConformer()
    positions = conf.GetPositions()                        # (N, 3)  Å
    numbers   = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
    atoms.center(vacuum=6.0)   # add vacuum so periodic-code tools don't complain
    return atoms


def _optimise(atoms: Atoms, calculator, *, fmax: float = 0.05,
               steps: int = 500, logfile: str = "/dev/null") -> None:
    """Relax *atoms* in-place with *calculator* using L-BFGS.

    Parameters
    ----------
    atoms : Atoms
        Modified in-place.
    calculator
        Any ASE-compatible calculator.
    fmax : float
        Convergence threshold (eV/Å).  Default 0.05.
    steps : int
        Maximum optimisation steps.  Default 500.
    logfile : str
        Path for the LBFGS log.  Default ``"/dev/null"`` (silent).
    """
    atoms.calc = calculator
    opt = LBFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    if not opt.converged():
        raise RuntimeError(
            f"gas-phase optimization did not converge within {steps} steps "
            f"at fmax={fmax} eV/Å"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API — analysis helpers
# ---------------------------------------------------------------------------

def find_unique_atoms(reactant: "Reactant") -> dict[str, list[list[int]]]:
    """Return intramolecular automorphism orbits of *reactant.graph*, by element.

    Two atoms are in the same orbit if there exists a graph automorphism
    (with ``categorical_node_match("element", "X")``) that maps one to
    the other.  The grouping is then partitioned by chemical element.

    Algorithm
    ---------
    Use :func:`networkx.algorithms.isomorphism.GraphMatcher.isomorphisms_iter`
    to enumerate every automorphism of the reactant graph onto itself.
    For each atom *i*, collect every image *σ(i)* across all
    automorphisms — this is the orbit of *i*.  Identical orbits are
    deduplicated.

    For molecules with no non-trivial automorphisms (e.g. CO, H₂O), every
    atom is its own singleton orbit.

    Parameters
    ----------
    reactant : Reactant

    Returns
    -------
    dict[str, list[list[int]]]
        ``{element: [orbit_1, orbit_2, ...]}``, where each orbit is a
        sorted list of atom indices.

    Examples
    --------
    >>> r = build_reactant("O")            # H2O (no calculator needed)
    >>> find_unique_atoms(r)
    {'O': [[0]], 'H': [[1, 2]]}
    """
    G = reactant.graph
    node_match = isomorphism.categorical_node_match("element", "X")
    gm = isomorphism.GraphMatcher(G, G, node_match=node_match)

    # orbit[i] = set of nodes that node i is mapped to under any automorphism
    orbits: dict[int, set[int]] = {n: {n} for n in G.nodes}
    for mapping in gm.isomorphisms_iter():
        for src, dst in mapping.items():
            orbits[src].add(dst)

    seen: set[frozenset] = set()
    by_element: dict[str, list[list[int]]] = {}
    for n in sorted(G.nodes):
        orb = frozenset(orbits[n])
        if orb in seen:
            continue
        seen.add(orb)
        elem = G.nodes[n]["element"]
        by_element.setdefault(elem, []).append(sorted(int(x) for x in orb))

    for elem in by_element:
        by_element[elem].sort(key=lambda lst: lst[0])

    return by_element


def find_anchor_atoms(
    reactant: "Reactant",
    *,
    hull_tol: float = 0.1,
) -> list[int]:
    """Return the atom indices of *reactant* exposed on its convex hull.

    Anchor atoms are the candidates eligible to bond to a surface in
    :func:`autokmc.sites.adsorbate.enumerate_configurations`.  An atom is
    considered exposed if **either**:

    * it is a vertex of the convex hull of the relaxed Cartesian
      coordinates, **or**
    * its atom centre lies within *hull_tol* Å of any hull facet — i.e. it
      is almost coplanar with the molecular exterior but was not selected
      as a strict hull vertex.

    If any non-hydrogen atom is exposed, hydrogen atoms are dropped from the
    anchor set.  This keeps hydrocarbon fragments such as CH3 carbon-bound
    while still allowing H-bound activation geometries for closed-shell CH4,
    where the central carbon is buried and only hydrogens are exposed.

    Degenerate cases (single atom, diatomic, planar molecule with fewer
    than four atoms) raise ``QhullError`` from
    :class:`scipy.spatial.ConvexHull`.  In that case **every atom is
    treated as geometrically exposed** before the non-H preference above is
    applied.

    Parameters
    ----------
    reactant : Reactant
    hull_tol : float
        Centre-to-facet tolerance in Å for deciding whether a non-vertex
        atom is close enough to the molecular exterior to count as exposed.
        Default 0.1.

    Returns
    -------
    list[int]
        Sorted unique anchor indices (atom indices in
        ``reactant.atoms``).
    """
    from scipy.spatial import ConvexHull, QhullError

    def _prefer_non_hydrogen(exposed_atoms: set[int]) -> list[int]:
        symbols = reactant.atoms.get_chemical_symbols()
        heavy = sorted(
            int(i) for i in exposed_atoms
            if symbols[int(i)] != "H"
        )
        return heavy if heavy else sorted(int(i) for i in exposed_atoms)

    pts = reactant.atoms.get_positions()
    n_atoms = len(pts)
    if n_atoms < 4:
        return _prefer_non_hydrogen(set(range(n_atoms)))

    try:
        hull = ConvexHull(pts)
    except QhullError:
        # Coplanar / collinear → every atom is treated as an anchor.
        return _prefer_non_hydrogen(set(range(n_atoms)))

    exposed: set[int] = set(int(v) for v in hull.vertices)

    eqs     = hull.equations                 # (n_facets, 4) – a x + b y + c z + d = 0
    normals = eqs[:, :3]
    offsets = eqs[:, 3]
    for i in range(n_atoms):
        if i in exposed:
            continue
        # signed distance is negative inside hull, zero on facet, positive outside.
        signed = normals @ pts[i] + offsets
        if np.any(signed >= -float(hull_tol)):
            exposed.add(i)

    return _prefer_non_hydrogen(exposed)


# ---------------------------------------------------------------------------
# Public API — main entry point
# ---------------------------------------------------------------------------

def build_reactant(
    smiles: str,
    *,
    calculator=None,
    add_hydrogens: bool = True,
    fmax: float = 0.05,
    steps: int = 500,
    nl_mult: float = NL_MULT_DEFAULT,
    hull_tol: float = 0.1,
    free_energy_options=None,
    free_energy_temperature_k: float | None = None,
    partial_pressure_bar: float = 1.0,
    symmetry_number: int | None = None,
    spin: float | None = None,
    geometry: str | None = None,
    vib_cache_root: str | None = None,
    relax: bool = True,
) -> Reactant:
    """Build a :class:`Reactant` from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule, e.g. ``"[C-]#[O+]"`` for CO.
    calculator : ASE calculator or None
        If provided, the geometry is refined with L-BFGS and a
        single-point energy is stored on :attr:`Reactant.energy`.  Any
        ASE-compatible calculator works (EMT, XTB, MACE, …).  If
        ``None``, the MMFF94-pre-relaxed RDKit geometry is used as-is and
        :attr:`Reactant.energy` remains ``nan``.
    add_hydrogens : bool
        Add explicit hydrogens to the SMILES before embedding.  Default ``True``.
    fmax : float
        Force convergence threshold for ASE optimisation (eV/Å).  Default 0.05.
    steps : int
        Maximum ASE optimisation steps.  Default 500.
    nl_mult : float
        Neighbour-list multiplier passed to :func:`~autokmc.core.graph.build_graph`.
        Default ``NL_MULT_DEFAULT`` (currently 0.90).
    hull_tol : float
        Tolerance passed to :func:`find_anchor_atoms`.  Default 0.1 Å.

    Returns
    -------
    Reactant
        Dataclass with all fields populated (``smiles``, ``atoms``, ``graph``,
        ``energy``, ``unique_nodes``, ``anchor_atoms``, ``anchor_orbit``).

    Examples
    --------
    >>> from autokmc.species.reactant import build_reactant
    >>> r = build_reactant("O")          # water, no calculator
    >>> r.atoms.get_chemical_formula()
    'H2O'
    >>> r.graph.number_of_nodes()
    3
    >>> all(d["type"] == "adsorbate" for _, d in r.graph.nodes(data=True))
    True
    >>> r.unique_nodes["H"]              # the two H atoms are equivalent
    [[1, 2]]
    """
    # 1. SMILES → 3-D geometry
    atoms = _smiles_to_atoms(smiles, add_hydrogens=add_hydrogens)

    # 2. Optional ASE relaxation + energy
    energy = float("nan")
    if calculator is not None:
        with acquire_calculator(calculator, purpose="gas-phase reactant relaxation") as calc:
            if relax:
                _optimise(atoms, calc, fmax=fmax, steps=steps)
            atoms.calc = calc
            try:
                energy = float(atoms.get_potential_energy())
            except Exception as exc:
                raise RuntimeError(
                    f"build_reactant({smiles!r}) could not evaluate its gas-phase energy"
                ) from exc
            if not np.isfinite(energy):
                raise ValueError(
                    f"build_reactant({smiles!r}) returned non-finite gas energy {energy!r}"
                )
        atoms.calc = None

    # 3. Tag every atom as adsorbate (molecules have no bulk interior and are
    #    not part of the surface — they will adsorb onto it).
    atoms.arrays["surface"] = np.full(len(atoms), 2, dtype=np.int8)

    # 4. Build graph
    graph = build_graph(atoms, nl_mult=nl_mult)

    reactant = Reactant(smiles=smiles, atoms=atoms, graph=graph, energy=energy)

    # 5. Intramolecular orbits
    reactant.unique_nodes = find_unique_atoms(reactant)

    # 6. Convex-hull-exposed anchor atoms + orbit map
    reactant.anchor_atoms = find_anchor_atoms(reactant, hull_tol=hull_tol)

    orbit_id_of: dict[int, int] = {}
    next_id = 0
    for _elem, orbits in reactant.unique_nodes.items():
        for orb in orbits:
            for atom in orb:
                orbit_id_of[atom] = next_id
            next_id += 1
    reactant.anchor_orbit = {a: orbit_id_of[a] for a in reactant.anchor_atoms}

    # 7. Optional gas-phase thermochemistry — IdealGasThermo via ASE Vibrations.
    reactant.partial_pressure_bar = float(partial_pressure_bar)
    if (
        free_energy_options is not None
        and getattr(free_energy_options, "enabled", False)
        and calculator is not None
        and free_energy_temperature_k is not None
        and not (isinstance(energy, float) and np.isnan(energy))
    ):
        from autokmc.thermo.free_energy import compute_gas_thermo
        from pathlib import Path as _Path

        cache_dir = (
            str(_Path(vib_cache_root) / f"gas_{smiles_to_dirname(smiles)}")
            if vib_cache_root is not None else None
        )
        thermo = compute_gas_thermo(
            atoms,
            energy_ev       = float(energy),
            temperature_k   = float(free_energy_temperature_k),
            pressure_bar    = float(partial_pressure_bar),
            calculator      = calculator,
            options         = free_energy_options,
            symmetry_number = symmetry_number,
            spin            = spin,
            geometry        = geometry,
            cache_dir       = cache_dir,
        )
        reactant.g_correction   = float(thermo["g_corr_ev"])
        reactant.gibbs_energy   = float(thermo["g_total_ev"])
        reactant.zpe            = float(thermo["zpe_ev"])
        reactant.entropy        = float(thermo["entropy_ev_per_k"])
        reactant.frequencies_ev = list(thermo["frequencies_ev"])
        reactant.imaginary_ev   = list(thermo["imaginary_ev"])
        reactant.thermo_meta = {
            "geometry":               thermo.get("geometry"),
            "symmetry_number":        thermo.get("symmetry_number"),
            "symmetry_number_source": thermo.get("symmetry_number_source"),
            "point_group":            thermo.get("point_group"),
            "symmetry_tolerance":     thermo.get("symmetry_tolerance"),
            "spin":                   thermo.get("spin"),
            "temperature_k":          thermo.get("temperature_k"),
            "pressure_bar":           thermo.get("pressure_bar"),
        }

    return reactant
