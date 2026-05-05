# %% [markdown]
# # AutoKMC — Bond reactions on Cu(111) for [OH]
#
# A step-by-step debug script for inspecting the **bond reaction** pipeline.
# Every stage is a separate ``# %%`` cell so it can be converted to a Jupyter
# notebook with jupytext or run section-by-section in an IDE.
#
# Reactions enumerated for the SMILES ``[OH]`` are:
#
# 1. **Dissociation**  ``OH → O + H`` — derived from
#    :func:`autokmc.species.bond_chemistry.get_all_fragments`.
# 2. **Coupling**      ``OH + OH → H2O2`` — derived from
#    :func:`autokmc.species.bond_chemistry.combine_fragments`.
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Build Cu(111) slab |
# | 2 | Find surface atoms + build graph |
# | 3 | Build Reactants for [OH], [O], [H], OO (H2O2) |
# | 4 | Find adsorbate sites for each species |
# | 5 | **derive_bond_templates** — enumerate (A, B, C) patterns |
# | 6 | **find_bond_reactions** — enumerate triples on the graph |
# | 7 | Pretty-print the BondReactionSites |
# | 8 | Sanity-check applicability for a synthetic occupancy pattern |

# %% ── 0. Imports ────────────────────────────────────��────────────────────────
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from nequip.ase import NequIPCalculator

from autokmc.structure import build_surface, find_surface_atoms
from autokmc.core.graph import build_graph
from autokmc.species.reactant import build_reactant
from autokmc.sites.adsorbate import find_adsorbate_sites
from autokmc.sites.bond import (
    derive_dissociation_templates,
    derive_coupling_templates,
    derive_bond_templates,
    find_bond_sites,
)
from autokmc.reactions.bond import is_bond_applicable
from autokmc.kmc.expansion import (
    initialise_bond_registry,
    bond_species_known,
    expand_bond_sites_for_new_species,
)

# %% ── 1. Configuration ──────────────────────────────────────────────────────
SMALL_SLAB = True

FMAX_ADS      : float = 0.05      # adsorbate pruning convergence (eV/Å)
MAX_OPT_STEPS : int   = 500
RANDOM_SEED   : int   = 69
BOND_MAX_HOPS : int   = 0         # share at least one surface atom

_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path(""))
    / _MODEL_FILE
)

_BAR = "─" * 64
def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")

print(f"Device : {_DEVICE}   Model : {_MODEL_FILE}")
print(f"Small slab : {SMALL_SLAB}   bond max_hops : {BOND_MAX_HOPS}")

# %% ── 2. Calculator ─────────────────────────────────────────────────────────
_section("Calculator")

def make_calc():
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )

calc = make_calc()
print("Calculator ready.")

# %% ── 3. Build Cu(111) slab ─────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

_goal = 12.0 if SMALL_SLAB else 20.0

atoms = build_surface(
    composition       = "Cu",
    crystal_structure = "fcc",
    miller_index      = (1, 1, 1),
    lattice_constant  = 3.615,
    min_slab_size     = 8.0,
    min_vacuum_size   = 12.0,
    goal_x            = _goal,
    goal_y            = _goal,
    n_freeze_layers   = 2,
    calculator        = calc,
    verbose           = True,
)
print(f"Slab: {len(atoms)} atoms   cell diag: {np.diag(atoms.get_cell()).round(3)} Å")

frozen_indices: list[int] | None = list(atoms.info.get("frozen_indices", []) or [])
if not frozen_indices:
    frozen_indices = None

# %% ── 4. Surface atoms + graph ───────────────────────────────────────────────
_section("STAGE 2 — Surface atoms + graph")

find_surface_atoms(atoms, tag_atoms=True)
G = build_graph(atoms)
print(f"Graph: {G.number_of_nodes()} nodes   {G.number_of_edges()} edges")
print(f"PBC: {G.graph['pbc']}")

# %% ── 5. Reactants for every species in the bond reactions ──────────────────
_section("STAGE 3 — Reactants  ([OH], [O], [H], OO)")

# All four species we expect to see in the templates derived from [OH]:
#   OH  →  O + H        (dissociation)
#   OH + OH  →  HOOH    (coupling)
SPECIES_SMILES = ["[OH]", "[O]", "[H]", "O=O"]

reactants = {}
for smi in SPECIES_SMILES:
    r = build_reactant(
        smi,
        add_hydrogens = False,
        calculator    = calc,
    )
    reactants[smi] = r
    print(f"  {smi:>6s}  →  formula={r.atoms.get_chemical_formula()}  "
          f"E_gas={r.energy:+.4f} eV  anchors={r.anchor_atoms}")

# %% ── 6. Adsorbate sites for every species ──────────────────────────────────
_section("STAGE 4 — Adsorbate sites for every species")

all_ads_sites = []
for smi, r in reactants.items():
    print(f"\n  ── find_adsorbate_sites for {smi!r} ──")
    sites = find_adsorbate_sites(
        G, r,
        prune_stable_only = True,
        calculator        = calc,
        frozen_indices    = frozen_indices,
        prune_fmax        = FMAX_ADS,
        prune_max_steps   = MAX_OPT_STEPS,
        verbose           = True,
    )
    n_members = sum(len(s.members) for s in sites)
    print(f"  → {len(sites)} stable iso-class(es), {n_members} total members")
    all_ads_sites.extend(sites)

print(f"\nTotal adsorbate iso-classes across all species: {len(all_ads_sites)}")

# %% ── 7. Derive bond-reaction templates from [OH] ───────────────────────────
_section("STAGE 5 — derive bond reaction templates from [OH]")

dissoc_templates = derive_dissociation_templates(
    "[OH]",
    bond_types = ("SINGLE",),   # OH only has a single bond
)
print(f"Dissociation templates ({len(dissoc_templates)}):")
for t in dissoc_templates:
    print(f"  {t.smiles_a!r} + {t.smiles_b!r}  ⇌  {t.smiles_c!r}   "
          f"({t.bond_type}, source={t.source})")

coupling_templates = derive_coupling_templates(
    "[OH]",
    include_homo = True,
)
print(f"\nCoupling templates ({len(coupling_templates)}):")
for t in coupling_templates:
    print(f"  {t.smiles_a!r} + {t.smiles_b!r}  ⇌  {t.smiles_c!r}   "
          f"({t.bond_type}, source={t.source})")

templates = derive_bond_templates(
    "[OH]",
    bond_types = ("SINGLE",),
)
print(f"\nUnion (deduplicated): {len(templates)} template(s).")

# %% ── 8. Enumerate bond reactions on the graph ──────────────────────────────
_section("STAGE 6 — find_bond_sites  (max_hops=0 → share a surface atom)")

bond_sites = find_bond_sites(
    G, all_ads_sites, templates,
    max_hops        = BOND_MAX_HOPS,
    deduplicate_iso = True,
    verbose         = True,
)

print(f"\nTotal bond reaction iso-classes: {len(bond_sites)}")
for brs in bond_sites:
    sa, ma, sb, mb, sc, mc = brs.members[0]
    print(f"  bond_iso {brs.iso_class:2d}  "
          f"template={brs.template.smiles_a!r}+{brs.template.smiles_b!r}"
          f"⇌{brs.template.smiles_c!r}  "
          f"members={len(brs.members):<3d}  "
          f"rep: A=(ads_iso={sa.iso_class}, m={ma})  "
          f"B=(ads_iso={sb.iso_class}, m={mb})  "
          f"C=(ads_iso={sc.iso_class}, m={mc})")

if not bond_sites:
    raise RuntimeError(
        "No bond reaction sites found.  Check that adsorbate sites exist "
        "for all four species ([OH], [O], [H], OO) on this slab."
    )

# %% ── 9. Sanity-check applicability for a synthetic occupancy ───────────────
_section("STAGE 7 — Synthetic applicability test")

# Pick the first member of the first bond reaction iso-class.
TARGET = bond_sites[0]
m_idx  = 0
sa, ma, sb, mb, sc, mc = TARGET.members[m_idx]
a_nids, b_nids, c_nids = TARGET.member_node_ids[m_idx]

print(f"Test triple: bond_iso={TARGET.iso_class}, m={m_idx}")
print(f"  A nodes: {list(a_nids)}")
print(f"  B nodes: {list(b_nids)}")
print(f"  C nodes: {list(c_nids)}")

def _set_occupancy(nids, value):
    for nid in nids:
        if nid in G:
            G.nodes[nid]["occupied"] = value

# All empty → not applicable.
_set_occupancy(a_nids, False); _set_occupancy(b_nids, False); _set_occupancy(c_nids, False)
print(f"\nAll empty                : {is_bond_applicable(G, TARGET, m_idx)}")

# A + B occupied, C empty → couple.
_set_occupancy(a_nids, True); _set_occupancy(b_nids, True); _set_occupancy(c_nids, False)
print(f"A+B occupied, C empty    : {is_bond_applicable(G, TARGET, m_idx)}")

# A occupied only → not applicable (B missing).
_set_occupancy(a_nids, True); _set_occupancy(b_nids, False); _set_occupancy(c_nids, False)
print(f"A occupied only          : {is_bond_applicable(G, TARGET, m_idx)}")

# C occupied, A+B empty → dissoc.
_set_occupancy(a_nids, False); _set_occupancy(b_nids, False); _set_occupancy(c_nids, True)
print(f"C occupied, A+B empty    : {is_bond_applicable(G, TARGET, m_idx)}")

# All occupied → not applicable (overlapping states).
_set_occupancy(a_nids, True); _set_occupancy(b_nids, True); _set_occupancy(c_nids, True)
print(f"All occupied             : {is_bond_applicable(G, TARGET, m_idx)}")

# Reset
_set_occupancy(a_nids, False); _set_occupancy(b_nids, False); _set_occupancy(c_nids, False)

# %% ── 10. On-the-fly growth: register the initial state, then expand ────────
_section("STAGE 8 — Initialise bond registry + on-the-fly expansion")

# 10a. Bootstrap the registry with everything we have built so far.  This is
# the "I am about to enter the KMC loop, here is my starting state" snapshot.
initialise_bond_registry(
    G,
    reactants       = reactants,            # dict[smi, Reactant]
    adsorbate_sites = all_ads_sites,        # flat iterable
    templates       = templates,            # initial template list
    bond_sites      = bond_sites,           # initial enumeration
)
print(f"Initial registry:")
print(f"  species              : {sorted(reactants.keys())}")
print(f"  templates            : {len(templates)}")
print(f"  bond reaction sites  : {len(bond_sites)}")

# 10b. Suppose KMC has just fired   OH + OH → H2O2   (template smi_c='OO').
#      Pretend H2O2 is a *new* species — drop it from the registry to
#      simulate the on-the-fly trigger.  In real use, the KMC loop calls
#      expand_bond_sites_after_event(G, fired_reaction, calculator=calc)
#      and only species genuinely missing from the registry get expanded.
NEW_SMILES = "OO"   # H2O2 (canonical SMILES from RDKit)

# Forcibly clear it from the registry to demonstrate the expansion path.
G.graph["bond_registry"]["species"].pop(NEW_SMILES, None)
G.graph["bond_registry"]["adsorbate_sites"].pop(NEW_SMILES, None)
print(f"\nSimulating discovery of {NEW_SMILES!r}: known? "
      f"{bond_species_known(G, NEW_SMILES)}")

new_brs = expand_bond_sites_for_new_species(
    G, NEW_SMILES,
    calculator      = calc,
    frozen_indices  = frozen_indices,
    bond_max_hops   = BOND_MAX_HOPS,
    prune_fmax      = FMAX_ADS,
    prune_max_steps = MAX_OPT_STEPS,
    add_hydrogens   = False,
    verbose         = True,
)

print(f"\nAfter expansion: known? {bond_species_known(G, NEW_SMILES)}")
print(f"  new BondReactionSites : {len(new_brs)}")
print(f"  total BondReactionSites: {len(G.graph['bond_reaction_sites'])}")
print(f"  registry templates    : {len(G.graph['bond_registry']['templates'])}")
print(f"  registry species      : "
      f"{sorted(G.graph['bond_registry']['species'].keys())}")

print("\nDone.")

