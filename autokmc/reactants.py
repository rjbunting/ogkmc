"""
autokmc.reactants
=================
Convert a SMILES string into an optimised 3-D :class:`~ase.Atoms` object and
its atom-connectivity graph.

Pipeline
--------
1. Parse SMILES with RDKit and embed a 3-D conformer (ETKDGv3 + MMFF94).
2. Optionally refine with an ASE calculator via L-BFGS.
3. Tag every atom as ``surface = 1`` (molecules have no bulk interior).
4. Build a :class:`networkx.Graph` via :func:`~autokmc.graph.build_graph`.

The result is a :class:`Reactant` dataclass that bundles the atoms object and
its graph together.

Typical usage
-------------
::

    from ase.calculators.emt import EMT
    from autokmc.reactants import build_reactant

    co = build_reactant("[C-]#[O+]", calculator=EMT())
    print(co.atoms.get_chemical_formula())   # CO
    print(co.graph.number_of_nodes())        # 2

Dependencies
------------
* RDKit  (``conda install -c conda-forge rdkit`` or ``pip install rdkit``)
* ASE
* networkx
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from ase import Atoms
from ase.optimize import LBFGS

from autokmc.graph import build_graph


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reactant:
    """A molecule represented as an optimised :class:`~ase.Atoms` object
    and its atom-connectivity graph.

    Attributes
    ----------
    smiles : str
        SMILES string the structure was built from.
    atoms : Atoms
        3-D geometry (optionally calculator-relaxed).  Every atom has
        ``atoms.arrays["surface"] = 2`` (adsorbate) because molecules
        are neither bulk nor surface — they will adsorb onto the surface.
    graph : nx.Graph
        Connectivity graph built by :func:`~autokmc.graph.build_graph`.
        Node attributes: ``element``, ``position``, ``index``,
        ``type`` (always ``"adsorbate"``), ``covalent_radius``.
    """
    smiles : str
    atoms  : Atoms
    graph  : nx.Graph


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
        Whether to add explicit hydrogens.  Default ``True``.

    Returns
    -------
    Atoms
        Non-periodic structure with no calculator attached.
    """
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
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fall back to random embedding
        AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reactant(
    smiles: str,
    *,
    calculator=None,
    add_hydrogens: bool = True,
    fmax: float = 0.05,
    steps: int = 500,
    nl_mult: float = 1.1,
) -> Reactant:
    """Build a :class:`Reactant` from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule, e.g. ``"[C-]#[O+]"`` for CO.
    calculator : ASE calculator or None
        If provided, the geometry is refined with L-BFGS before the graph
        is built.  Any ASE-compatible calculator works (EMT, XTB, MACE, …).
        If ``None``, the MMFF94-pre-relaxed RDKit geometry is used as-is.
    add_hydrogens : bool
        Add explicit hydrogens to the SMILES before embedding.  Default ``True``.
    fmax : float
        Force convergence threshold for ASE optimisation (eV/Å).  Default 0.05.
    steps : int
        Maximum ASE optimisation steps.  Default 500.
    nl_mult : float
        Neighbour-list multiplier passed to :func:`~autokmc.graph.build_graph`.
        Default 1.1.

    Returns
    -------
    Reactant
        Dataclass with ``.smiles``, ``.atoms``, and ``.graph``.

    Examples
    --------
    >>> from autokmc.reactants import build_reactant
    >>> r = build_reactant("O")          # water, no calculator
    >>> r.atoms.get_chemical_formula()
    'H2O'
    >>> r.graph.number_of_nodes()
    3
    >>> all(d["type"] == "adsorbate" for _, d in r.graph.nodes(data=True))
    True
    """
    # 1. SMILES → 3-D geometry
    atoms = _smiles_to_atoms(smiles, add_hydrogens=add_hydrogens)

    # 2. Optional ASE relaxation
    if calculator is not None:
        _optimise(atoms, calculator, fmax=fmax, steps=steps)

    # 3. Tag every atom as adsorbate (molecules have no bulk interior and are
    #    not part of the surface — they will adsorb onto it).
    atoms.arrays["surface"] = np.full(len(atoms), 2, dtype=np.int8)

    # 4. Build graph
    graph = build_graph(atoms, nl_mult=nl_mult)

    return Reactant(smiles=smiles, atoms=atoms, graph=graph)


