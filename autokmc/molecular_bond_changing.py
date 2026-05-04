"""
autokmc.molecular_bond_changing
===============================
Two complementary tools for enumerating bond-breaking and bond-forming
reactions of molecular species:

* :func:`get_all_fragments` — given a molecule (SMILES or ASE ``Atoms``),
  return every pair of fragments from single-bond cleavage.
* :func:`combine_fragments` — given two fragments, return every species that
  can be formed by joining them with a new bond.

.. important::
    All SMILES passed to these functions must be **charge-free**.  Formal
    charges confuse radical-electron counting and prevent correct valence
    detection.  Use bracket notation with explicit radical electrons instead
    of dative-bond shorthands:

    * CO: use ``[C]=O`` (C has 2 radical electrons) — **not** ``[C-]#[O+]``
    * OH: ``[OH]`` (O has 1 radical electron)

Typical usage — fragmentation
------------------------------
::

    from autokmc.molecular_bond_changing import get_all_fragments

    pairs = get_all_fragments("[C]=O")          # CO — breaks the double bond
    for p in pairs:
        print(p.bond_type, p.smiles_a, p.smiles_b)

    # Water ��� break every O-H bond
    pairs = get_all_fragments("O", bond_types=("SINGLE",))

    # Build Reactant objects for each fragment (requires RDKit)
    pairs = get_all_fragments("CC", as_reactants=True)
    for p in pairs:
        print(p.reactant_a.atoms.get_chemical_formula())

Typical usage — combination
-----------------------------
::

    from autokmc.molecular_bond_changing import combine_fragments

    # Reconnect fragments that already carry * attachment points
    products = combine_fragments("[CH3]*", "[OH]*")
    for s in products:
        print(s.smiles)           # 'CO'  (methanol)

    # Enumerate every possible bond between two bare radicals
    products = combine_fragments("[CH3]", "[OH]")

    # CO + OH  (CO represented as neutral radical [C]=O)
    products = combine_fragments("[C]=O", "[OH]")
    for s in products:
        print(s.smiles)           # HOCO/carboxyl radical

Notes
-----
* :func:`get_all_fragments` uses RDKit :func:`~rdkit.Chem.FragmentOnBonds` to
  cleave bonds.  Each fragment carries a ``*`` (dummy) atom at the cleavage
  site so it is clear which valence was broken.
* :func:`combine_fragments` detects ``*`` atoms in its input SMILES and treats
  them as **directed** attachment points.  When no ``*`` atoms are present it
  falls back to **undirected** enumeration — every atom pair whose available
  valence permits a new bond is tried.
* Pass ``strip_dummies=True`` to :func:`get_all_fragments` to remove the ``*``
  markers before storing SMILES (fragments become open-valence radicals).
* If *smiles_or_atoms* is an ASE :class:`~ase.Atoms` object, connectivity is
  inferred from the covalent-radius neighbour list via
  :func:`autokmc.graph.build_graph`; bond orders default to ``SINGLE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import NL_MULT_DEFAULT, RANDOM_SEED
from .logging_utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FragmentPair:
    """Two molecular fragments produced by cleaving one bond.

    Attributes
    ----------
    bond_idx : int
        Bond index in the **original** RDKit molecule (before cleavage).
    bond_type : str
        RDKit bond type string: ``"SINGLE"``, ``"DOUBLE"``, ``"TRIPLE"``, or
        ``"AROMATIC"``.
    atom_idx_a : int
        Atom index (in the original molecule) of bond end A.
    atom_idx_b : int
        Atom index (in the original molecule) of bond end B.
    element_a : str
        Element symbol of bond end A (e.g. ``"C"``).
    element_b : str
        Element symbol of bond end B (e.g. ``"O"``).
    smiles_a : str
        Canonical SMILES of fragment A.  Contains a ``*`` atom at the
        cleavage site unless *strip_dummies* was ``True``.
    smiles_b : str
        Canonical SMILES of fragment B.
    reactant_a : Reactant or None
        :class:`autokmc.reactants.Reactant` for fragment A; populated only
        when :func:`get_all_fragments` is called with ``as_reactants=True``.
    reactant_b : Reactant or None
        :class:`autokmc.reactants.Reactant` for fragment B.
    """

    bond_idx   : int
    bond_type  : str
    atom_idx_a : int
    atom_idx_b : int
    element_a  : str
    element_b  : str
    smiles_a   : str
    smiles_b   : str
    reactant_a : object = field(default=None, repr=False)
    reactant_b : object = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"FragmentPair(bond={self.atom_idx_a}({self.element_a})"
            f"-{self.atom_idx_b}({self.element_b}) [{self.bond_type}], "
            f"A={self.smiles_a!r}, B={self.smiles_b!r})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rdkit_mol_from_smiles(smiles: str, *, add_hydrogens: bool):
    """Return an RDKit ``Mol`` from *smiles*, optionally with explicit H."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required.  Install with: conda install -c conda-forge rdkit"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    if add_hydrogens:
        only_atoms = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetNumImplicitHs() > 0 or a.GetNumExplicitHs() > 0
        ]
        if only_atoms:
            mol = Chem.AddHs(mol, onlyOnAtoms=only_atoms)

    # Embed + quick MMFF pre-relax so positions are meaningful.
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    return mol


def _rdkit_mol_from_atoms(atoms):
    """Build an RDKit ``Mol`` from an ASE :class:`~ase.Atoms` object.

    Bond orders from the covalent-radius neighbour list are set to ``SINGLE``
    (ASE / autokmc graphs do not track bond orders).  Positions are taken
    directly from *atoms*.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required.  Install with: conda install -c conda-forge rdkit"
        ) from exc

    from .graph import build_graph
    import numpy as _np

    # Ensure surface tags exist (build_graph requires them).
    if "surface" not in atoms.arrays:
        atoms = atoms.copy()
        atoms.arrays["surface"] = _np.full(len(atoms), 2, dtype=_np.int8)

    G = build_graph(atoms, nl_mult=NL_MULT_DEFAULT)

    # Map graph node-id → ASE atom index (skip bookkeeping anchor nodes).
    node_to_idx: dict[int, int] = {
        n: d["index"]
        for n, d in G.nodes(data=True)
        if d.get("type") in ("bulk", "surface", "adsorbate")
    }

    symbols = atoms.get_chemical_symbols()
    rwmol = Chem.RWMol()

    for sym in symbols:
        atom = Chem.Atom(sym)
        atom.SetNoImplicit(True)
        rwmol.AddAtom(atom)

    added: set[frozenset] = set()
    for u, v in G.edges():
        iu = node_to_idx.get(u)
        iv = node_to_idx.get(v)
        if iu is None or iv is None:
            continue
        key = frozenset((iu, iv))
        if key in added:
            continue
        added.add(key)
        rwmol.AddBond(iu, iv, Chem.BondType.SINGLE)

    # Attach 3-D positions via a conformer.
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(atoms.get_positions()):
        conf.SetAtomPosition(i, pos.tolist())
    rwmol.AddConformer(conf, assignId=True)

    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
    except Exception as exc:
        _log.warning("_rdkit_mol_from_atoms: sanitization warning: %s", exc)
        mol = rwmol.GetMol()

    return mol


def _set_no_implicit(rw) -> None:
    """Set ``NoImplicit = True`` on every heavy atom in *rw*.

    This prevents RDKit's sanitiser from silently filling unsatisfied valence
    with implicit H.  Instead, any remaining open valence is expressed as
    radical electrons, which correctly models surface fragments whose broken
    bonds will later connect to the metal — not to hydrogen.

    Dummy atoms (``*``, atomic number 0) are left unchanged; they carry no
    valence of their own.
    """
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetNoImplicit(True)


def _strip_dummy_atoms(mol) -> str:
    """Return canonical SMILES with all dummy (``*``) atoms removed.

    After removal the open valence at the cleavage site is represented as
    radical electrons on the heavy atom neighbour — **not** as implicit H.
    ``SetNoImplicit(True)`` is applied **only to atoms directly bonded to a
    dummy** before the dummy is removed, so that only the broken-bond site
    becomes a radical.  Atoms elsewhere in the fragment (e.g. an OH that was
    not involved in the cleavage) are left untouched and retain their H atoms.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    rw = RWMol(mol)
    # Mark NoImplicit only on the heavy-atom neighbours of each dummy —
    # those are the atoms that will lose a bond to * and must not silently
    # gain an implicit H in its place.
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            for nbr in atom.GetNeighbors():
                nbr.SetNoImplicit(True)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return Chem.MolToSmiles(rw.GetMol())


# ---------------------------------------------------------------------------
# Public helper — SMILES → ASE Atoms
# ---------------------------------------------------------------------------

def fragment_smiles_to_atoms(smiles: str, *, add_hydrogens: bool = True):
    """Convert a fragment SMILES string to an ASE :class:`~ase.Atoms` object.

    Dummy (``*``) attachment-point atoms are stripped before embedding so
    the resulting structure contains only real elements.  A 3-D conformer is
    generated with RDKit ETKDGv3 and pre-relaxed with MMFF94.

    Parameters
    ----------
    smiles : str
        Fragment SMILES, optionally containing ``*`` attachment-point markers
        (as produced by :func:`get_all_fragments` with
        ``strip_dummies=False``).
    add_hydrogens : bool
        Add explicit H to atoms with implicit H in the cleaned SMILES before
        embedding.  Default ``True``.

    Returns
    -------
    ase.Atoms
        Non-periodic 3-D structure with a 6 Å vacuum padding.

    Raises
    ------
    ImportError
        If RDKit or ASE are not installed.
    ValueError
        If the cleaned SMILES cannot be parsed or embedded by RDKit.
    """
    from .reactants import _smiles_to_atoms

    # Strip * atoms to get a clean SMILES for embedding.
    clean = _strip_dummy_atoms_from_smiles(smiles)
    if not clean:
        raise ValueError(
            f"fragment_smiles_to_atoms: SMILES {smiles!r} reduces to "
            "an empty molecule after stripping dummy atoms."
        )
    return _smiles_to_atoms(clean, add_hydrogens=add_hydrogens)


def _strip_dummy_atoms_from_smiles(smiles: str) -> str:
    """Return *smiles* with all ``*`` dummy atoms removed (string-level helper).

    Used by :func:`fragment_smiles_to_atoms` to clean attachment-point SMILES
    before passing to RDKit's 3-D embedder.

    ``SetNoImplicit(True)`` is applied **only to atoms directly bonded to a
    dummy** before that dummy is removed, so that the broken-bond site does not
    silently gain an implicit H.  Atoms that are not bonded to any ``*`` — such
    as an ``[OH]`` group elsewhere in the molecule — are left untouched and
    retain their hydrogen atoms.  When the SMILES contains no dummy atoms at
    all (e.g. a product SMILES from :func:`combine_fragments`), no atoms are
    modified and the function is essentially a round-trip canonicaliser.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except ImportError:
        import re
        return re.sub(r'\[\*\]|\*', '', smiles).strip('- ')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    rw = RWMol(mol)

    dummies = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,
    )

    if dummies:
        # Set NoImplicit ONLY on atoms directly bonded to a dummy — those are
        # the atoms that will lose a bond when the dummy is removed and must
        # not fill the resulting open valence with an implicit H.  All other
        # atoms (e.g. an OH not involved in the cleavage) keep their Hs.
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                for nbr in atom.GetNeighbors():
                    nbr.SetNoImplicit(True)
        for idx in dummies:
            rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return Chem.MolToSmiles(rw.GetMol())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_fragments(
    smiles_or_atoms,
    *,
    add_hydrogens: bool = True,
    bond_types: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE"),
    include_ring_bonds: bool = False,
    strip_dummies: bool = False,
    deduplicate: bool = True,
    as_reactants: bool = False,
    calculator=None,
    nl_mult: float = NL_MULT_DEFAULT,
) -> list[FragmentPair]:
    """Return all fragment pairs from single-bond cleavage of a molecule.

    Parameters
    ----------
    smiles_or_atoms : str or ase.Atoms
        Molecule to fragment.  A SMILES string is parsed with RDKit; an ASE
        :class:`~ase.Atoms` object is converted via the covalent-radius
        neighbour list (bond orders will all be ``SINGLE``).
    add_hydrogens : bool
        Add explicit H atoms to the SMILES before fragmenting.  Ignored when
        *smiles_or_atoms* is an ``Atoms`` object.  Default ``True``.
    bond_types : tuple[str, ...]
        Bond orders to cleave.  Each entry must be one of ``"SINGLE"``,
        ``"DOUBLE"``, ``"TRIPLE"``, ``"AROMATIC"``.
        Default ``("SINGLE", "DOUBLE", "TRIPLE")``.
    include_ring_bonds : bool
        If ``False`` (default), ring bonds are skipped because cleaving them
        yields a single ring-opened fragment rather than two separate molecules.
    strip_dummies : bool
        If ``True``, remove the ``*`` dummy atoms from fragment SMILES so the
        broken valence appears as an open radical.  Default ``False``.
    deduplicate : bool
        Collapse symmetry-equivalent bonds that produce the same two fragment
        SMILES into a single entry.  Default ``True``.
    as_reactants : bool
        If ``True``, build an :class:`autokmc.reactants.Reactant` for each
        fragment and store it in :attr:`FragmentPair.reactant_a` /
        :attr:`FragmentPair.reactant_b`.
    calculator
        ASE calculator forwarded to
        :func:`autokmc.reactants.build_reactant` when ``as_reactants=True``.
    nl_mult : float
        Neighbour-list multiplier forwarded to
        :func:`autokmc.reactants.build_reactant`.

    Returns
    -------
    list[FragmentPair]
        One :class:`FragmentPair` per broken bond (or unique fragment pair).
        Bonds that do not produce exactly two disconnected fragments are
        silently skipped.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If *smiles_or_atoms* is an unparseable SMILES string, or if any entry
        in *bond_types* is not a recognised bond-type name.

    Examples
    --------
    >>> from autokmc.molecular_bond_changing import get_all_fragments
    >>> pairs = get_all_fragments("O")          # H₂O — two equivalent O-H bonds
    >>> len(pairs)                              # deduplicated → 1
    1
    >>> pairs[0].element_a, pairs[0].element_b
    ('O', 'H')

    >>> # CO as neutral radical: [C]=O  (C carries 2 radical electrons)
    >>> pairs = get_all_fragments("[C]=O", bond_types=("DOUBLE",), strip_dummies=True)
    >>> pairs[0].element_a, pairs[0].element_b
    ('C', 'O')
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required.  Install with: conda install -c conda-forge rdkit"
        ) from exc

    # ── Build the RDKit molecule ─────────────────────────────────────────────
    try:
        from ase import Atoms as _ASEAtoms
        _ase_available = True
    except ImportError:
        _ASEAtoms = None
        _ase_available = False

    if isinstance(smiles_or_atoms, str):
        input_label = smiles_or_atoms
        mol = _rdkit_mol_from_smiles(smiles_or_atoms, add_hydrogens=add_hydrogens)
    elif _ase_available and isinstance(smiles_or_atoms, _ASEAtoms):
        input_label = smiles_or_atoms.get_chemical_formula()
        mol = _rdkit_mol_from_atoms(smiles_or_atoms)
    else:
        raise TypeError(
            f"smiles_or_atoms must be a str (SMILES) or an ase.Atoms object, "
            f"got {type(smiles_or_atoms).__name__!r}."
        )

    # Map bond-type name → RDKit BondType enum value.
    _bt_map = {
        "SINGLE":   Chem.BondType.SINGLE,
        "DOUBLE":   Chem.BondType.DOUBLE,
        "TRIPLE":   Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    for bt in bond_types:
        if bt not in _bt_map:
            raise ValueError(
                f"Unknown bond_type {bt!r}; must be one of {sorted(_bt_map)}"
            )
    wanted_types = {_bt_map[bt] for bt in bond_types}

    _bt_name_map = {v: k for k, v in _bt_map.items()}

    # ── Iterate over bonds ───────────────────────────────────────────────────
    results: list[FragmentPair] = []
    seen_smiles_pairs: set[frozenset] = set()

    for bond in mol.GetBonds():
        # Bond-type filter.
        if bond.GetBondType() not in wanted_types:
            continue
        # Ring-bond filter.
        if not include_ring_bonds and bond.IsInRing():
            continue

        bond_idx = bond.GetIdx()
        atom_a   = bond.GetBeginAtom()
        atom_b   = bond.GetEndAtom()
        idx_a    = atom_a.GetIdx()
        idx_b    = atom_b.GetIdx()
        elem_a   = atom_a.GetSymbol()
        elem_b   = atom_b.GetSymbol()
        bt_name  = _bt_name_map.get(bond.GetBondType(), str(bond.GetBondType()))

        # ── Fragment the molecule at this bond ────────────────────────────
        try:
            frag_mol = Chem.FragmentOnBonds(
                mol,
                [bond_idx],
                addDummies=True,
                dummyLabels=[(0, 0)],
            )
        except Exception as exc:
            _log.debug(
                "get_all_fragments: FragmentOnBonds failed for bond %d (%s-%s): %s",
                bond_idx, elem_a, elem_b, exc,
            )
            continue

        try:
            frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        except Exception as exc:
            _log.debug(
                "get_all_fragments: GetMolFrags failed for bond %d: %s",
                bond_idx, exc,
            )
            continue

        if len(frags) != 2:
            # Ring bond (only opened, not split) or already disconnected mol.
            _log.debug(
                "get_all_fragments: bond %d (%s-%s) gave %d fragment(s), skipping.",
                bond_idx, elem_a, elem_b, len(frags),
            )
            continue

        frag_a, frag_b = frags

        # For each fragment, set NoImplicit only on the atom directly bonded
        # to the * dummy — that atom is the one that might otherwise fill its
        # open valence with implicit H instead of a radical electron.
        def _mark_dummy_nbrs(frag):
            rw = Chem.RWMol(frag)
            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    for nbr in atom.GetNeighbors():
                        nbr.SetNoImplicit(True)
            return rw.GetMol()

        frag_a = _mark_dummy_nbrs(frag_a)
        frag_b = _mark_dummy_nbrs(frag_b)

        # ── Convert to SMILES ─────────────────────────────────────────────
        if strip_dummies:
            smi_a = _strip_dummy_atoms(frag_a)
            smi_b = _strip_dummy_atoms(frag_b)
        else:
            smi_a = Chem.MolToSmiles(frag_a)
            smi_b = Chem.MolToSmiles(frag_b)

        # ── Optional deduplication by {smi_a, smi_b} ─────────────────────
        if deduplicate:
            key = frozenset({smi_a, smi_b})
            if key in seen_smiles_pairs:
                continue
            seen_smiles_pairs.add(key)

        results.append(FragmentPair(
            bond_idx   = bond_idx,
            bond_type  = bt_name,
            atom_idx_a = idx_a,
            atom_idx_b = idx_b,
            element_a  = elem_a,
            element_b  = elem_b,
            smiles_a   = smi_a,
            smiles_b   = smi_b,
        ))

    _log.debug(
        "get_all_fragments(%r): %d unique fragment pair(s) found",
        input_label, len(results),
    )

    # ── Optionally build Reactant objects for each fragment ──────────────────
    if as_reactants:
        from .reactants import build_reactant

        def _clean_smiles_for_reactant(smi: str) -> str | None:
            """Remove dummy (*) atoms; return None if result is empty."""
            from rdkit import Chem as _C
            from rdkit.Chem import RWMol as _RW
            m = _C.MolFromSmiles(smi)
            if m is None:
                return None
            rw = _RW(m)
            dummies = sorted(
                [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0],
                reverse=True,
            )
            for idx in dummies:
                rw.RemoveAtom(idx)
            try:
                _C.SanitizeMol(rw)
            except Exception:
                pass
            clean = _C.MolToSmiles(rw.GetMol())
            return clean if clean else None

        for pair in results:
            for attr, smi_attr in (("reactant_a", "smiles_a"),
                                   ("reactant_b", "smiles_b")):
                raw_smi = getattr(pair, smi_attr)
                clean   = _clean_smiles_for_reactant(raw_smi)
                if not clean:
                    _log.warning(
                        "get_all_fragments: could not clean SMILES %r for "
                        "Reactant construction; skipping.",
                        raw_smi,
                    )
                    continue
                try:
                    reactant = build_reactant(
                        clean,
                        calculator    = calculator,
                        add_hydrogens = False,
                        nl_mult       = nl_mult,
                    )
                    object.__setattr__(pair, attr, reactant)
                except Exception as exc:
                    _log.warning(
                        "get_all_fragments: build_reactant(%r) failed: %s",
                        clean, exc,
                    )

    return results


# ===========================================================================
# Reverse operation: combine two fragments into all possible bonded species
# ===========================================================================

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class CombinedSpecies:
    """One species formed by joining two fragments with a single new bond.

    Attributes
    ----------
    smiles : str
        Canonical SMILES of the combined molecule.
    bond_type : str
        Bond order used to join the fragments: ``"SINGLE"``, ``"DOUBLE"``,
        ``"TRIPLE"``, or ``"AROMATIC"``.
    atom_idx_a : int
        Index of the bonding atom in **fragment A** (before combination; in
        the fragment's own atom numbering).
    atom_idx_b : int
        Index of the bonding atom in **fragment B** (before combination).
    element_a : str
        Element symbol of the bonding atom from fragment A.
    element_b : str
        Element symbol of the bonding atom from fragment B.
    reactant : Reactant or None
        :class:`autokmc.reactants.Reactant` for the combined species;
        populated only when :func:`combine_fragments` is called with
        ``as_reactants=True``.
    """

    smiles     : str
    bond_type  : str
    atom_idx_a : int
    atom_idx_b : int
    element_a  : str
    element_b  : str
    reactant   : object = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"CombinedSpecies({self.smiles!r}, "
            f"{self.element_a}[{self.atom_idx_a}]-{self.element_b}[{self.atom_idx_b}] "
            f"[{self.bond_type}])"
        )


# ---------------------------------------------------------------------------
# Internal helpers for combine_fragments
# ---------------------------------------------------------------------------

def _dummy_indices(mol) -> list[int]:
    """Return atom indices of all ``*`` (dummy, atomic number 0) atoms."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _available_valence_indices(mol) -> list[int]:
    """Return indices of heavy atoms with genuine open valence (radical electrons).

    Only atoms with ``GetNumRadicalElectrons() > 0`` are returned — i.e.
    atoms that carry an explicit unpaired electron from a broken bond (e.g.
    ``[CH3]``, ``[OH]``, ``[C]=O`` with a radical C).  Atoms that are merely
    saturated with implicit H (e.g. fully-bonded CH4) are **not** included;
    a new bond at such an atom would displace an H, which is a separate
    reaction (H-abstraction) and not what this function enumerates.

    The bond-order for the new connection should match the number of radical
    electrons available: 1 radical → SINGLE, 2 radicals → DOUBLE, 3 → TRIPLE.
    """
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue  # skip dummy atoms
        if atom.GetNumRadicalElectrons() > 0:
            out.append(atom.GetIdx())
    return out


def _fix_overvalent_atoms(rw, pt=None) -> None:
    """Reduce bond orders to resolve over-valency introduced by a new bond.

    After :func:`_join_mols` adds a new single bond between two fragments,
    one of the atoms may now exceed its default valence (e.g. the O in
    ``[C]=O`` gains a third bond when O–OH is formed).  This function scans
    every non-H atom and, if its bond-order sum exceeds its default valence,
    reduces the highest-order *existing* bond (TRIPLE→DOUBLE or
    DOUBLE→SINGLE) to make room.  The process repeats until no over-valent
    atom remains.

    Note: radical electron counts are **not** explicitly updated here; they
    are recomputed correctly when the caller subsequently calls
    ``Chem.SanitizeMol`` with ``SANITIZE_FINDRADICALS``.
    """
    from rdkit import Chem

    if pt is None:
        pt = Chem.GetPeriodicTable()

    _reduce = {
        Chem.BondType.TRIPLE: Chem.BondType.DOUBLE,
        Chem.BondType.DOUBLE: Chem.BondType.SINGLE,
    }

    changed = True
    max_iters = 20
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue  # skip H and dummy (*)
            dv = pt.GetDefaultValence(atom.GetAtomicNum())
            if dv <= 0:
                continue
            bo = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())))
            if bo <= dv:
                continue  # valence OK
            # Reduce the highest-order bond of this over-valent atom.
            for bond in sorted(atom.GetBonds(),
                               key=lambda b: b.GetBondTypeAsDouble(), reverse=True):
                new_type = _reduce.get(bond.GetBondType())
                if new_type is None:
                    continue  # already SINGLE — cannot reduce further
                bond.SetBondType(new_type)
                changed = True
                break


def _join_mols(mol_a, mol_b, idx_a: int, idx_b: int, rdkit_bond_type) -> str | None:
    """Join *mol_a* and *mol_b* by forming a bond between atom *idx_a* in A
    and atom *idx_b* in B.

    If either atom is a dummy (``*``), the dummy is **removed** after the bond
    is formed and replaced by a direct bond to the dummy's neighbour.  This
    handles the ``FragmentOnBonds``-style attachment-point SMILES.

    Returns the canonical SMILES of the combined molecule, or ``None`` if
    RDKit sanitisation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    # Determine whether the target atoms are dummies.  If so, the real
    # bonding partner is the dummy's single neighbour.
    def _resolve(mol, idx):
        """(real_idx, is_dummy)"""
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 0:
            nbrs = list(atom.GetNeighbors())
            if not nbrs:
                return None, True   # isolated dummy — nothing to connect to
            return nbrs[0].GetIdx(), True
        return idx, False

    real_a, is_dummy_a = _resolve(mol_a, idx_a)
    real_b, is_dummy_b = _resolve(mol_b, idx_b)
    if real_a is None or real_b is None:
        return None

    combined = Chem.CombineMols(mol_a, mol_b)
    rw = RWMol(combined)
    offset = mol_a.GetNumAtoms()

    try:
        rw.AddBond(real_a, offset + real_b, rdkit_bond_type)
    except Exception:
        return None

    # Remove dummy atoms (highest index first to keep numbering stable).
    dummies_to_remove: list[int] = []
    if is_dummy_a:
        dummies_to_remove.append(idx_a)
    if is_dummy_b:
        dummies_to_remove.append(offset + idx_b)
    for d in sorted(dummies_to_remove, reverse=True):
        rw.RemoveAtom(d)

    # Prevent RDKit from filling the newly formed bond's residual open valence
    # with implicit H — that valence belongs to the metal surface, not H.
    _set_no_implicit(rw)

    # If the new bond made any atom over-valent (e.g. O in C=O gaining a
    # third bond), reduce existing bond orders to resolve it.
    _fix_overvalent_atoms(rw)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _parse_fragment_smiles(smiles_or_obj, add_hydrogens: bool = False):
    """Return an RDKit ``Mol`` from *smiles_or_obj*.

    Accepts:
    * ``str``             — treated as SMILES (no embedding needed here).
    * ``FragmentPair``    — raises ``TypeError`` (pass ``.smiles_a`` directly).
    * ``Reactant``        — uses ``reactant.atoms`` via ``_rdkit_mol_from_atoms``.
    * ``ase.Atoms``       — via ``_rdkit_mol_from_atoms``.

    ``SetNoImplicit(True)`` is applied to all heavy atoms so that downstream
    operations never silently introduce implicit H.
    """
    from rdkit import Chem
    from rdkit.Chem import RWMol

    if isinstance(smiles_or_obj, str):
        mol = Chem.MolFromSmiles(smiles_or_obj)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles_or_obj!r}")
        if add_hydrogens:
            only = [a.GetIdx() for a in mol.GetAtoms()
                    if a.GetNumImplicitHs() > 0 or a.GetNumExplicitHs() > 0]
            if only:
                mol = Chem.AddHs(mol, onlyOnAtoms=only)
        rw = RWMol(mol)
        _set_no_implicit(rw)
        return rw.GetMol()

    # Reactant dataclass has an .atoms attribute (ASE Atoms).
    if hasattr(smiles_or_obj, "atoms"):
        return _rdkit_mol_from_atoms(smiles_or_obj.atoms)

    try:
        from ase import Atoms as _ASEAtoms
        if isinstance(smiles_or_obj, _ASEAtoms):
            return _rdkit_mol_from_atoms(smiles_or_obj)
    except ImportError:
        pass

    raise TypeError(
        f"combine_fragments: expected str, Reactant, or ase.Atoms, "
        f"got {type(smiles_or_obj).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Public API — combine_fragments
# ---------------------------------------------------------------------------

def combine_fragments(
    fragment_a,
    fragment_b,
    *,
    deduplicate: bool = True,
    as_reactants: bool = False,
    calculator=None,
    nl_mult: float = NL_MULT_DEFAULT,
) -> list[CombinedSpecies]:
    """Return all species formed by joining *fragment_a* and *fragment_b*
    with a new **single** bond.

    A single bond is always formed at each valid atom pair.  The actual
    equilibrium bond order (single / double / triple) is a property of the
    electronic structure and is best determined by a subsequent geometry
    optimisation rather than hard-coded at the enumeration stage — this
    function's job is only to identify *which atoms can connect* and to
    return a chemically valid SMILES seed for each unique connectivity.

    The function operates in two modes depending on whether the input SMILES
    contain ``*`` (dummy) attachment-point atoms:

    **Directed mode** (one or both fragments contain ``*``)
        ``*`` atoms are treated as explicit attachment points — as produced by
        :func:`get_all_fragments` with ``strip_dummies=False`` (the default).
        Every pairing of a ``*`` in fragment A with a ``*`` in fragment B is
        tried; the two dummies are removed and replaced by a single bond
        between their respective heavy-atom neighbours.

    **Undirected mode** (neither fragment contains ``*``)
        Every pair (atom_i from A, atom_j from B) whose atoms both carry at
        least one radical electron is tried.  A radical electron signals a
        genuine open valence from a previous bond-breaking event (not merely
        an implicit H that could be displaced).

    Parameters
    ----------
    fragment_a, fragment_b : str, Reactant, or ase.Atoms
        The two fragments to join.  SMILES strings are parsed directly;
        :class:`autokmc.reactants.Reactant` objects and ASE
        :class:`~ase.Atoms` objects are converted via
        :func:`_rdkit_mol_from_atoms`.
    deduplicate : bool
        Collapse entries that produce the same canonical SMILES into a single
        :class:`CombinedSpecies`.  Default ``True``.
    as_reactants : bool
        If ``True``, build an :class:`autokmc.reactants.Reactant` for each
        combined species and store it in :attr:`CombinedSpecies.reactant`.
    calculator
        ASE calculator forwarded to :func:`autokmc.reactants.build_reactant`
        when ``as_reactants=True``.
    nl_mult : float
        Neighbour-list multiplier forwarded to
        :func:`autokmc.reactants.build_reactant`.

    Returns
    -------
    list[CombinedSpecies]
        One entry per unique single-bond connectivity (when *deduplicate* is
        ``True``).  The ``bond_type`` field is always ``"SINGLE"``.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If either fragment SMILES cannot be parsed.

    Examples
    --------
    **Reconnect attachment-point fragments (directed mode)**::

        >>> from autokmc.molecular_bond_changing import combine_fragments
        >>> products = combine_fragments("[CH3]*", "*[OH]")
        >>> [s.smiles for s in products]
        ['CO']                  # methanol

    **Bare radical fragments (undirected mode)**::

        >>> products = combine_fragments("[CH3]", "[OH]")
        >>> [s.smiles for s in products]
        ['CO']                  # C radical + O radical → methanol

    **Round-trip: break CO double bond, recombine**::

        >>> from autokmc.molecular_bond_changing import get_all_fragments, combine_fragments
        >>> pairs = get_all_fragments("[C]=O", bond_types=("DOUBLE",))
        >>> p = pairs[0]
        >>> products = combine_fragments(p.smiles_a, p.smiles_b)
        >>> [s.smiles for s in products]    # single bond seed
        ['[C][O]']
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required.  Install with: conda install -c conda-forge rdkit"
        ) from exc

    mol_a = _parse_fragment_smiles(fragment_a)
    mol_b = _parse_fragment_smiles(fragment_b)

    dummies_a = _dummy_indices(mol_a)
    dummies_b = _dummy_indices(mol_b)

    # ── Choose mode ──────────────────────────────────────────────────────────
    directed = bool(dummies_a or dummies_b)

    if directed:
        # Directed mode: pair up attachment points.
        # If only one side has dummies, treat every heavy atom with radical
        # electrons on the other side as a potential recipient.
        if dummies_a and dummies_b:
            pairs_to_try = [(ia, ib) for ia in dummies_a for ib in dummies_b]
        elif dummies_a:
            avail_b = _available_valence_indices(mol_b) or list(range(mol_b.GetNumAtoms()))
            pairs_to_try = [(ia, ib) for ia in dummies_a for ib in avail_b]
        else:
            avail_a = _available_valence_indices(mol_a) or list(range(mol_a.GetNumAtoms()))
            pairs_to_try = [(ia, ib) for ia in avail_a for ib in dummies_b]
    else:
        # Undirected mode: for each radical atom in fragment A, try every
        # non-dummy atom in fragment B, and vice versa.  Using "at least one
        # radical" (rather than "both radicals") allows a radical atom to
        # attack a saturated site — the bond-order reduction in _join_mols
        # resolves any resulting over-valency.
        # NOTE: filter is > 0 (exclude dummy * atoms only), NOT > 1 — using
        # > 1 incorrectly excludes hydrogen (GetAtomicNum() == 1), which
        # prevents homo-coupling of radical H atoms to form H₂ ([H][H]).
        avail_a = _available_valence_indices(mol_a)
        avail_b = _available_valence_indices(mol_b)
        heavy_a = [a.GetIdx() for a in mol_a.GetAtoms() if a.GetAtomicNum() > 0]
        heavy_b = [a.GetIdx() for a in mol_b.GetAtoms() if a.GetAtomicNum() > 0]

        seen_pairs: set[tuple[int, int]] = set()
        pairs_to_try = []
        for ia in avail_a:
            for ib in heavy_b:
                if (ia, ib) not in seen_pairs:
                    seen_pairs.add((ia, ib))
                    pairs_to_try.append((ia, ib))
        for ib in avail_b:
            for ia in heavy_a:
                if (ia, ib) not in seen_pairs:
                    seen_pairs.add((ia, ib))
                    pairs_to_try.append((ia, ib))

    # ── Always form a single bond ─────────────────────────────────────────────
    # The equilibrium bond order (single/double/triple) is a property of the
    # electronic structure — determining it is the job of the downstream
    # calculator, not the enumerator.
    rdkit_single = Chem.BondType.SINGLE

    def _real_atom(mol, idx):
        """Resolve dummy → its heavy-atom neighbour."""
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() == 0:
            nbrs = list(a.GetNeighbors())
            return nbrs[0] if nbrs else a
        return a

    results: list[CombinedSpecies] = []
    seen_smiles: set[str] = set()

    for idx_a, idx_b in pairs_to_try:
        real_atom_a = _real_atom(mol_a, idx_a)
        real_atom_b = _real_atom(mol_b, idx_b)
        elem_a      = real_atom_a.GetSymbol()
        elem_b      = real_atom_b.GetSymbol()
        real_idx_a  = real_atom_a.GetIdx()
        real_idx_b  = real_atom_b.GetIdx()

        smi = _join_mols(mol_a, mol_b, idx_a, idx_b, rdkit_single)
        if smi is None:
            continue

        if deduplicate:
            if smi in seen_smiles:
                continue
            seen_smiles.add(smi)

        results.append(CombinedSpecies(
            smiles     = smi,
            bond_type  = "SINGLE",
            atom_idx_a = real_idx_a,
            atom_idx_b = real_idx_b,
            element_a  = elem_a,
            element_b  = elem_b,
        ))

    _log.debug(
        "combine_fragments: %d unique single-bond species found",
        len(results),
    )

    # ── Optionally build Reactant objects ────────────────────────────────────
    if as_reactants:
        from .reactants import build_reactant

        for species in results:
            try:
                reactant = build_reactant(
                    species.smiles,
                    calculator    = calculator,
                    add_hydrogens = False,
                    nl_mult       = nl_mult,
                )
                object.__setattr__(species, "reactant", reactant)
            except Exception as exc:
                _log.warning(
                    "combine_fragments: build_reactant(%r) failed: %s",
                    species.smiles, exc,
                )

    return results


__all__ = [
    "FragmentPair",
    "CombinedSpecies",
    "get_all_fragments",
    "combine_fragments",
    "fragment_smiles_to_atoms",
]

