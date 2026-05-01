# %% [markdown]
# # AutoKMC — Bond-Reaction-Site Pipeline (Debug) — Atomic O on Cu(111)
#
# Mirrors `get_diffusion_sites_pruned.py` but continues into the bond-reaction
# enumeration step so you can inspect exactly which (A + B ⇌ C) iso-classes
# are produced for **atomic oxygen** ``[O]`` on Cu(111) and why.
#
# For ``[O]`` the possible surface reactions are:
#
# | Channel | Reaction |
# |---|---|
# | Homo-coupling    | O + O → O₂  (``O=O``) |
# | (No dissociation) | ``[O]`` is a single atom — no intramolecular bonds to break |
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Build Cu(111) FCC slab |
# | 2 | Find surface atoms |
# | 3 | Build connectivity graph |
# | 4 | Build ``[O]`` reactant (gas-phase relaxation) |
# | 5 | `find_adsorbate_sites` for ``[O]`` — ML-pruned iso-classes |
# | 6 | `derive_bond_templates("[O]")` — enumerate (A, B, C) patterns |
# | 7 | Build reactants + adsorbate sites for every leaf species (e.g. O₂) |
# | 8 | `find_bond_sites` — enumerate triple iso-classes |
# | 8b | `prune_unstable_bond_sites` — drop triples whose A+B state is bond-changing-unstable |
# | 8c | `_prune_one_per_adsorption_triple` — keep smallest-ego iso per (isoA,isoB,isoC) tuple |
# | 9 | Summary table: all BondReactionSite iso-classes |
# | 10 | 3-D overview: A, B, C placements for all bond_isos |
# | 11 | Per-bond-iso 3-D: A, B, C with representative highlighted |

# %% ── 0. Imports & helpers ──────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc.structure            import build_surface
from autokmc.surface              import find_surface_atoms
from autokmc.graph                import build_graph
from autokmc.reactants            import build_reactant
from autokmc.find_adsorbate_sites import find_adsorbate_sites
from autokmc.find_bond_sites      import (
    derive_bond_templates,
    find_bond_sites,
    prune_unstable_bond_sites,
    _prune_one_per_adsorption_triple,
    _canon_smiles,
)

_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path("."))
    / _MODEL_FILE
)
print(f"Device : {_DEVICE}")
print(f"Model  : {_MODEL_FILE}")


def make_calc():
    """Return a fresh NequIPCalculator instance."""
    return NequIPCalculator.from_compiled_model(
        compile_path=str(_MODEL_PATH),
        device=_DEVICE,
    )


calc = make_calc()
print(f"Calculator : {calc.__class__.__name__}")

# ── Visual palette ────────────────────────────────────────────────────────────
_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_SYM  = {"O": "cross", "C": "diamond"}
_ISO_COLS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BCF60C",
    "#FABEBE", "#008080",
]
# Distinct palette for bond iso-classes
_BOND_COLS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#C77DFF", "#FF9F1C", "#2EC4B6", "#E71D36",
    "#B5838D", "#6D6875",
]
# Endpoint colours (A, B, C)
_EP_COLS = {"a": "#FF4500", "b": "#1E90FF", "c": "#32CD32"}
_EP_SYMS = {"a": "circle", "b": "diamond", "c": "square"}


def _ads_col(iso_class: int) -> str:
    return _ISO_COLS[iso_class % len(_ISO_COLS)]


def _bond_col(bond_iso: int) -> str:
    return _BOND_COLS[bond_iso % len(_BOND_COLS)]


def _section(title: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def _atom_scatter3d(pos, colors, sizes, names, opacity=1.0, symbol="circle",
                    line_color="black", line_width=0.5, name="", showlegend=True):
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=colors, symbol=symbol, opacity=opacity,
                    line=dict(color=line_color, width=line_width)),
        text=names,
        hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
        name=name, showlegend=showlegend,
    )


def _bond_lines3d(p_from, p_to, color="rgba(160,160,160,0.4)", width=1,
                  name="", showlegend=False):
    n = len(p_from)
    x = np.empty(n * 3); y = np.empty(n * 3); z = np.empty(n * 3)
    x[0::3] = p_from[:, 0]; y[0::3] = p_from[:, 1]; z[0::3] = p_from[:, 2]
    x[1::3] = p_to[:, 0];   y[1::3] = p_to[:, 1];   z[1::3] = p_to[:, 2]
    x[2::3] = np.nan;        y[2::3] = np.nan;        z[2::3] = np.nan
    return go.Scatter3d(x=x, y=y, z=z, mode="lines",
                        line=dict(color=color, width=width),
                        hoverinfo="skip", name=name, showlegend=showlegend)


def _layout3d(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13)),
        showlegend=True,
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=70, b=0),
        template="plotly_white",
    )


def _add_slab_backdrop(fig, pos, bulk_mask, surf_pos):
    if bulk_mask.any():
        fig.add_trace(_atom_scatter3d(
            pos[bulk_mask], colors=_BULK_COL,
            sizes=3, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
            opacity=0.2, name="bulk", showlegend=True,
        ))
    fig.add_trace(_atom_scatter3d(
        surf_pos, colors=_SURF_COL,
        sizes=6, names=["Cu surf"] * len(surf_pos),
        opacity=0.55, name="surface Cu", showlegend=True,
    ))


def _nid_positions(G, nids):
    """Return an (N, 3) array of positions for a list of graph node ids."""
    pts = [G.nodes[nid]["position"]
           for nid in nids if nid in G and "position" in G.nodes[nid]]
    return np.array(pts, dtype=float) if pts else np.empty((0, 3))


def _nid_mean_pos(G, nids):
    """Mean Cartesian position of a list of node ids, or zeros if empty."""
    pts = _nid_positions(G, nids)
    return pts.mean(axis=0) if len(pts) else np.zeros(3)


def _dedup_symmetric_member_indices(brs, G) -> list[int]:
    """Return de-duplicated member indices for symmetric (A==B) reactions.

    For homo-coupling reactions where ``smiles_a == smiles_b``, every unique
    pair of placements (pa, pb) appears twice in ``brs.member_node_ids`` —
    once as (A=pa, B=pb) and once as (A=pb, B=pa).  This function keeps only
    one representative per unordered {pa, pb} centroid pair by hashing rounded
    centroid coordinates into a frozenset.

    For hetero-coupling (smiles_a != smiles_b) all indices are returned as-is.
    """
    if _canon_smiles(brs.template.smiles_a) != _canon_smiles(brs.template.smiles_b):
        return list(range(len(brs.member_node_ids)))

    seen: set[frozenset] = set()
    kept: list[int] = []
    for m_idx in range(len(brs.member_node_ids)):
        a_nids, b_nids, _ = brs.member_node_ids[m_idx]
        pa = _nid_mean_pos(G, a_nids)
        pb = _nid_mean_pos(G, b_nids)
        key = frozenset({tuple(pa.round(3)), tuple(pb.round(3))})
        if key not in seen:
            seen.add(key)
            kept.append(m_idx)
    return kept


# %% ── 1. Build Cu(111) slab ──────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition="Cu",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    lattice_constant=3.615,
    min_slab_size=10.0,
    min_vacuum_size=12.0,
    goal_x=15.0,
    goal_y=15.0,
    n_freeze_layers=2,
    calculator=calc,
    verbose=True,
)

frozen_indices = list(atoms.info.get("frozen_indices", []))
pos  = atoms.get_positions()
syms = np.array(atoms.get_chemical_symbols())

print(f"\nSlab built  : {len(atoms)} atoms")
print(f"Cell diag   : {np.diag(atoms.get_cell()).round(3)} Å")
print(f"Frozen atoms: {len(frozen_indices)}")

# %% ── 2. Surface atoms ───────────────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms")

surf_result = find_surface_atoms(atoms, tag_atoms=True)
surf_mask   = surf_result.mask
surf_pos    = pos[surf_mask]
bulk_mask   = ~surf_mask

print(f"Method        : {surf_result.method}")
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")

# %% ── 3. Build graph ─────────────────────────────────────────────────────────
_section("STAGE 3 — Build connectivity graph")

G = build_graph(atoms)

print(f"Nodes : {G.number_of_nodes()}  Edges : {G.number_of_edges()}")
print(f"PBC   : {G.graph['pbc'].tolist()}")

# %% ── 4. Build the primary adsorbate: atomic O ───────────────────────────────
_section("STAGE 4 — Build [O] reactant  (gas-phase relaxation)")

# [O] = charge-free atomic oxygen radical (required by bond-changing module).
# Do NOT use [O-] or similar charged forms.
o_reactant = build_reactant("[O]", add_hydrogens=False, calculator=calc)

print(f"Formula      : {o_reactant.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {o_reactant.anchor_atoms}")
print(f"Gas-phase E  : {o_reactant.energy:.4f} eV")

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites for [O] WITH Stability Pruning
#
# Identical to `get_adsorbate_sites_pruned.py` but for atomic oxygen.
# A single-atom adsorbate has only one anchor atom, so all stable sites are
# top, bridge, or hollow positions on the Cu(111) surface.

# %% ── 5. Find adsorbate sites for [O] ───────────────────────────────────────
_section("STAGE 5 — find_adsorbate_sites for [O]  [prune_stable_only=True]")

o_sites = find_adsorbate_sites(
    G, o_reactant,
    prune_stable_only = True,
    calculator        = calc,
    frozen_indices    = frozen_indices,
    prune_fmax        = 0.05,
    prune_max_steps   = 500,
    verbose           = True,
)

print(f"\n[O]: {len(o_sites)} stable iso-class(es) after pruning")
for site in o_sites:
    clq = site.atom_cliques[0]
    kstr = f"k={len(clq)}" if clq is not None else "floating"
    print(f"  ads_iso {site.iso_class:2d}  {kstr:<8s}  members={len(site.members):<4d}  "
          f"rep_pos={np.round(site.positions[0], 3)}")

# %% [markdown]
# ---
# ## Stage 6 — Derive Bond Reaction Templates from [O]
#
# `derive_bond_templates("[O]")` calls both
# `derive_dissociation_templates` and `derive_coupling_templates`:
#
# * **Dissociation**: ``[O]`` is a single atom — no bonds to break → zero templates.
# * **Coupling**: ``[O] + [O] → O=O`` (molecular oxygen / peroxide skeleton).
#
# The templates encode which triples ``(smiles_a, smiles_b, smiles_c)`` the
# enumerator needs to find adsorbate sites for.

# %% ── 6. Derive bond templates ───────────────────────────────────────────────
_section("STAGE 6 — derive_bond_templates('[O]')")

templates = derive_bond_templates(
    "[O]",
    bond_types            = ("SINGLE", "DOUBLE"),
    include_dissociation  = True,
    include_coupling      = True,
    include_homo_coupling = True,
    include_ring_bonds    = False,
)

if templates:
    print(f"Templates derived ({len(templates)}):")
    for t in templates:
        print(f"  {t.smiles_a!r:>8} + {t.smiles_b!r:<8}  ⇌  {t.smiles_c!r:<12}  "
              f"bond={t.bond_type}  source={t.source}")
else:
    print("  (no templates derived — [O] cannot be fragmented further)")
    print("  Coupling templates require a neighbour species; add one if needed.")

# Collect all unique leaf SMILES referenced by the templates that are NOT [O].
leaf_smiles: list[str] = []
for t in templates:
    for smi in (_canon_smiles(t.smiles_a),
                _canon_smiles(t.smiles_b),
                _canon_smiles(t.smiles_c)):
        if smi and smi != _canon_smiles("[O]") and smi not in leaf_smiles:
            leaf_smiles.append(smi)

print(f"\nLeaf species referenced by templates: {leaf_smiles or '(none)'}")

# %% [markdown]
# ---
# ## Stage 7 — Build Reactants & Adsorbate Sites for Leaf Species
#
# For every SMILES referenced by the templates that is *not* the primary
# adsorbate ``[O]``, build a :class:`Reactant` and find its stable adsorbate
# sites.  These sites are needed by `find_bond_sites` to build the triple.

# %% ── 7. Leaf species: reactants + adsorbate sites ───────────────────────────
_section("STAGE 7 — Leaf-species reactants + adsorbate sites")

# Seed with the primary adsorbate; extend with leaf species.
all_reactants: dict[str, object] = {_canon_smiles("[O]"): o_reactant}
all_ads_sites: list = list(o_sites)       # start with O sites already found

for leaf_smi in leaf_smiles:
    print(f"\n  ── {leaf_smi!r} ──────────────────────────────────────")
    try:
        r_leaf = build_reactant(
            leaf_smi,
            add_hydrogens = False,
            calculator    = calc,
        )
    except Exception as exc:
        print(f"  ⚠  build_reactant({leaf_smi!r}) failed: {exc}")
        continue

    all_reactants[leaf_smi] = r_leaf
    print(f"  Formula      : {r_leaf.atoms.get_chemical_formula()}")
    print(f"  Anchor atoms : {r_leaf.anchor_atoms}")
    print(f"  Gas-phase E  : {r_leaf.energy:.4f} eV")

    try:
        leaf_sites = find_adsorbate_sites(
            G, r_leaf,
            prune_stable_only = True,
            calculator        = calc,
            frozen_indices    = frozen_indices,
            prune_fmax        = 0.05,
            prune_max_steps   = 500,
            verbose           = True,
        )
    except Exception as exc:
        print(f"  ⚠  find_adsorbate_sites({leaf_smi!r}) failed: {exc}")
        leaf_sites = []

    n_mem = sum(len(s.members) for s in leaf_sites)
    print(f"  Stable iso-classes: {len(leaf_sites)}   total members: {n_mem}")
    all_ads_sites.extend(leaf_sites)

print(f"\nTotal adsorbate iso-classes (all species) : {len(all_ads_sites)}")
print(f"Species with sites : {sorted(all_reactants.keys())}")

# %% [markdown]
# ---
# ## Stage 8 — Enumerate Bond Reaction Iso-Classes
#
# `find_bond_sites` takes the full flat list of adsorbate sites (across all
# species), the templates, and the graph, and returns iso-classes of triples
# ``(placement_A, placement_B, placement_C)`` satisfying the locality
# constraints.
#
# Key parameters:
#
# | parameter         | default | meaning |
# |---|---|---|
# | `max_hops`        | `0`     | "share ≥1 surface atom" between A↔B and (A∪B)↔C |
# | `deduplicate_iso` | `True`  | collapse geometrically equivalent triples |
# | `prune_by_triple` | `True`  | keep only smallest-ego iso per (isoA,isoB,isoC) triple |

# %% ── 8. Find bond reaction sites ────────────────────────────────────────────
_section("STAGE 8 — find_bond_sites  [max_hops=0]")

if not templates:
    print("  ⚠  No templates �� skipping find_bond_sites.")
    bond_sites: list = []
else:
    bond_sites = find_bond_sites(
        G, all_ads_sites, templates,
        max_hops        = 0,
        deduplicate_iso = True,
        prune_by_triple = True,
        verbose         = True,
    )

print(f"\nBond reaction iso-classes: {len(bond_sites)}")
for brs in bond_sites:
    sa, ma, sb, mb, sc, mc = brs.members[0]
    eg = brs.ego_graph
    print(f"  bond_iso {brs.iso_class:2d}  "
          f"{brs.template.smiles_a!r}+{brs.template.smiles_b!r}"
          f"⇌{brs.template.smiles_c!r}  "
          f"source={brs.template.source:<12s}  "
          f"members={len(brs.members):<4d}  "
          f"ego(n={eg.number_of_nodes() if eg else '—'},"
          f"e={eg.number_of_edges() if eg else '—'})")

if not bond_sites:
    print("\n  ⚠  No bond-reaction iso-classes found!")
    print("  Possible causes:")
    print("    • no templates (single-atom adsorbate, no dissociation possible)")
    print("    • leaf species adsorbate sites are empty (pruning removed everything)")
    print("    • max_hops=0 too strict — try max_hops=1")

# %% [markdown]
# ---
# ## Stage 8b — Prune Unstable Bond Reaction Sites
#
# `prune_unstable_bond_sites` mirrors `prune_unstable_adsorbate_sites` for the
# bond-reaction case.  For every iso-class it:
#
# 1. Builds the **pre-reaction** slab state (A and B placed at the representative
#    member's sites, C absent).
# 2. Runs a calculator relaxation.
# 3. Compares the relaxed adsorbate connectivity against the *intended* edge set
#    (intramolecular bonds + anchor bonds for A and B).
# 4. **Prunes** the iso-class if A and B spontaneously formed C, either fragment
#    dissociated, or any anchor bond was gained/lost.
#
# This is the bond-channel equivalent of the diffusion channel having no
# explicit pruning step (diffusion endpoints are already individually pruned by
# Stage 5).  The CLI runs this by default when ``bond.prune_with_calculator: true``.

# %% ── 8b. Prune unstable bond reaction sites ─────────────────────────────────
_section("STAGE 8b — prune_unstable_bond_sites  [ML stability check]")

if bond_sites:
    n_before = len(bond_sites)
    bond_sites = prune_unstable_bond_sites(
        G, bond_sites,
        species_by_smiles = all_reactants,   # dict[canonical_smi, Reactant]
        calculator        = calc,
        frozen_indices    = frozen_indices,
        fmax              = 0.05,
        max_steps         = 500,
        verbose           = True,
    )
    print(f"\nAfter pruning: {len(bond_sites)} / {n_before} iso-class(es) survived")
else:
    print("  (no bond sites to prune)")

# %% [markdown]
# ---
# ## Stage 8c — Prune to One Iso-Class per Adsorption Triple
#
# `_prune_one_per_adsorption_triple` is run a **second time** after the
# stability check, mirroring exactly what the CLI does
# (``cli.py`` → ``b.prune_by_triple`` post-stability block).
#
# The first call (inside ``find_bond_sites(..., prune_by_triple=True)``) removed
# duplicates from the raw enumeration.  After stability pruning removed some
# iso-classes entirely, a different iso-class for the same
# ``(frozenset({isoA, isoB}), isoC)`` triple may now be the smallest remaining
# one — so the deduplication is re-run on the survivors.
#
# After selection the ``iso_class`` indices and the graph reverse-indices
# (``G.graph["bond_clique_to_members"]``, ``G.graph["bond_surface_node_to_members"]``,
# ``G.graph["bond_reaction_sites"]``) are all renumbered / rebuilt to stay
# consistent.

# %% ── 8c. Post-stability triple deduplication ────────────────────────────────
_section("STAGE 8c — _prune_one_per_adsorption_triple  (post-stability)")

if bond_sites:
    n_before_c = len(bond_sites)
    bond_sites = _prune_one_per_adsorption_triple(
        bond_sites,
        verbose = True,
        prefix  = " (post-stability)",
    )
    print(f"\nAfter post-stability triple dedup: "
          f"{len(bond_sites)} / {n_before_c} iso-class(es) survived")

    # Renumber iso_class and rebuild graph reverse-indices — mirrors cli.py exactly.
    G.graph["bond_clique_to_members"]        = {}
    G.graph["bond_surface_node_to_members"]  = {}
    rebuilt_idx  = G.graph["bond_clique_to_members"]
    rebuilt_surf = G.graph["bond_surface_node_to_members"]
    for new_idx, brs in enumerate(bond_sites):
        brs.iso_class = new_idx
        for m_idx, (cliques_a, cliques_b, cliques_c) in enumerate(brs._member_cliques):
            for clq in (*cliques_a, *cliques_b, *cliques_c):
                rebuilt_idx.setdefault(clq, []).append((brs, m_idx))
                for surf_id in clq:
                    rebuilt_surf.setdefault(int(surf_id), []).append((brs, m_idx))
    G.graph["bond_reaction_sites"] = bond_sites
    print(f"  iso_class indices renumbered 0 – {len(bond_sites) - 1}")
else:
    print("  (no bond sites remain)")

# %% ── 9. Summary table ───────────────────────────────────────────────────────
_section("STAGE 9 — Bond reaction iso-class summary table")

if bond_sites:
    print(f"\n{'bond_iso':>8}  {'smi_A':>8}  {'smi_B':>8}  {'smi_C':>10}  "
          f"{'source':>12}  {'members':>7}  {'ego_n':>5}  {'ego_e':>5}  n_shells")
    print("─" * 88)
    for brs in bond_sites:
        sa, ma, sb, mb, sc, mc = brs.members[0]
        eg = brs.ego_graph
        n_n = eg.number_of_nodes() if eg else -1
        n_e = eg.number_of_edges() if eg else -1
        print(f"  {brs.iso_class:>6}  "
              f"{brs.template.smiles_a!r:>8}  "
              f"{brs.template.smiles_b!r:>8}  "
              f"{brs.template.smiles_c!r:>10}  "
              f"{brs.template.source:>12}  "
              f"{len(brs.members):>7}  "
              f"{n_n:>5}  {n_e:>5}  "
              f"{brs.n_shells_pair_settled}")
else:
    print("  (no bond reaction iso-classes)")

# %% [markdown]
# ---
# ## Stage 10 — 3-D Overview: Bond-Site Triple Centroids Across the Surface
#
# **One point per member per endpoint**, coloured by bond_iso.
# Using per-member *centroids* (mean position of all atoms in that endpoint)
# avoids the O₂ / multi-atom bias: O=O has 2 node IDs per member, so a
# per-atom plot would show twice as many C points as A or B points, making C
# appear "spread across the surface" while A/B look sparse.  Centroid ensures
# every member contributes exactly one point per endpoint.
#
# Lines connect A–B (solid, per bond_iso colour) and mid(A,B)–C (dashed).
# The slab backbone is shown at very low opacity for spatial reference only.

# %% ── 10. 3-D overview ───────────────────────────────────────────────────────
_section("STAGE 10 — 3-D overview: bond-site triple centroids (one point / member)")

fig_all = go.Figure()

# Very transparent surface backdrop — reference only, not the focus.
fig_all.add_trace(_atom_scatter3d(
    surf_pos, colors=_SURF_COL,
    sizes=4, names=["Cu surf"] * len(surf_pos),
    opacity=0.10, name="surface Cu", showlegend=True,
))

for brs in bond_sites:
    col = _bond_col(brs.iso_class)
    r_int = int(col[1:3], 16)
    g_int = int(col[3:5], 16)
    b_int = int(col[5:7], 16)

    # De-duplicate symmetric (A==B) members before plotting.
    plot_indices = _dedup_symmetric_member_indices(brs, G)
    n_plot = len(plot_indices)
    is_sym = n_plot < len(brs.member_node_ids)

    # Per-member centroid positions — always one point per member per endpoint.
    pa_list, pb_list, pc_list = [], [], []
    ab_from, ab_to             = [], []
    abc_from, abc_to           = [], []

    for m_idx in plot_indices:
        a_nids, b_nids, c_nids = brs.member_node_ids[m_idx]
        pa = _nid_mean_pos(G, a_nids)
        pb = _nid_mean_pos(G, b_nids)
        pc = _nid_mean_pos(G, c_nids)
        pa_list.append(pa)
        pb_list.append(pb)
        pc_list.append(pc)
        pab = (pa + pb) / 2.0
        # Skip lines that wrap across the periodic boundary (cosmetic guard).
        if np.linalg.norm(pb - pa) < 8.0:
            ab_from.append(pa); ab_to.append(pb)
        if np.linalg.norm(pc - pab) < 8.0:
            abc_from.append(pab); abc_to.append(pc)

    pa_arr = np.array(pa_list)
    pb_arr = np.array(pb_list)
    pc_arr = np.array(pc_list)

    # A–B lines
    if ab_from:
        fig_all.add_trace(_bond_lines3d(
            np.array(ab_from), np.array(ab_to),
            color=f"rgba({r_int},{g_int},{b_int},0.45)", width=2, showlegend=False,
        ))
    # mid(A,B)–C lines
    if abc_from:
        fig_all.add_trace(_bond_lines3d(
            np.array(abc_from), np.array(abc_to),
            color=f"rgba({r_int},{g_int},{b_int},0.25)", width=1, showlegend=False,
        ))

    # Endpoint markers — one per member, all coloured by bond_iso.
    for ep_label, arr, sym in (
        ("A", pa_arr, "circle"),
        ("B", pb_arr, "diamond"),
        ("C", pc_arr, "square"),
    ):
        if not len(arr):
            continue
        sym_note = " (sym-dedup)" if is_sym else ""
        fig_all.add_trace(_atom_scatter3d(
            arr, colors=col, sizes=11,
            names=[f"bond_iso {brs.iso_class} {ep_label} m={i}"
                   for i in range(len(arr))],
            opacity=0.90, symbol=sym,
            line_color="black", line_width=0.7,
            name=f"bond_iso {brs.iso_class} {ep_label} ({n_plot}m{sym_note})",
            showlegend=True,
        ))

fig_all.update_layout(**_layout3d(
    f"Stage 10 — bond-site triple centroids  "
    f"({len(bond_sites)} bond_iso, colour=bond_iso, ● A  ◆ B  ■ C)"
))
fig_all.show()

# %% [markdown]
# ---
# ## Stage 11 — Per-Bond-Iso 3-D Plots
#
# One figure per bond-reaction iso-class.  **All members** are shown with the
# same representative colours (A=orange-red circle, B=blue diamond,
# C=lime-green square) — one point per member per endpoint using the centroid
# of all atoms in that placement.  This prevents unlabelled duplicate points
# from multi-atom endpoints (e.g. both O atoms of O=O) and ensures the
# complete propagation of A, B, C across the surface is visible.
#
# Connection lines run between centroids:
#   A centroid ↔ B centroid  (orange-red, A–B lines)
#   mid(A,B)   ↔ C centroid  (lime-green, bond lines)

# %% ── 11. Per-bond-iso 3-D plots ─────────────────────────────────────────────
_section("STAGE 11 — Per-bond-iso 3-D: A / B / C centroid propagation across surface")

for brs in bond_sites:
    eg  = brs.ego_graph
    n_mem = len(brs.member_node_ids)

    # De-duplicate symmetric (A==B) members before plotting.
    plot_indices = _dedup_symmetric_member_indices(brs, G)
    n_plot = len(plot_indices)
    is_sym = n_plot < n_mem

    # Accumulate per-endpoint centroids across de-duplicated members.
    pa_list, pb_list, pc_list = [], [], []
    ab_from, ab_to            = [], []
    abc_from, abc_to          = [], []

    for m_idx in plot_indices:
        a_nids, b_nids, c_nids = brs.member_node_ids[m_idx]
        pa  = _nid_mean_pos(G, a_nids)
        pb  = _nid_mean_pos(G, b_nids)
        pc  = _nid_mean_pos(G, c_nids)
        pa_list.append(pa); pb_list.append(pb); pc_list.append(pc)
        pab = (pa + pb) / 2.0
        if np.linalg.norm(pb - pa) < 8.0:
            ab_from.append(pa);  ab_to.append(pb)
        if np.linalg.norm(pc - pab) < 8.0:
            abc_from.append(pab); abc_to.append(pc)

    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    # A–B connection lines.
    if ab_from:
        fig.add_trace(_bond_lines3d(
            np.array(ab_from), np.array(ab_to),
            color="rgba(255,69,0,0.55)", width=2, showlegend=False,
        ))
    # mid(A,B)–C connection lines.
    if abc_from:
        fig.add_trace(_bond_lines3d(
            np.array(abc_from), np.array(abc_to),
            color="rgba(50,205,50,0.55)", width=2, showlegend=False,
        ))

    # One labelled trace per endpoint type — de-duplicated members, uniform rep colours.
    sym_note = f" (sym-dedup, {n_plot}/{n_mem})" if is_sym else ""
    for ep_label, pts_list, sym in (
        ("A", pa_list, "circle"),
        ("B", pb_list, "diamond"),
        ("C", pc_list, "square"),
    ):
        arr = np.array(pts_list)
        if not len(arr):
            continue
        ep_col = _EP_COLS[ep_label.lower()]
        fig.add_trace(_atom_scatter3d(
            arr, colors=ep_col, sizes=12,
            names=[f"bond_iso {brs.iso_class} {ep_label} m={i}"
                   for i in range(len(arr))],
            opacity=0.90, symbol=sym,
            line_color="black", line_width=0.8,
            name=f"● {ep_label}  ({n_plot} member{'s' if n_plot != 1 else ''}{sym_note})",
            showlegend=True,
        ))

    fig.update_layout(**_layout3d(
        f"bond_iso {brs.iso_class}  |  "
        f"{brs.template.smiles_a!r}+{brs.template.smiles_b!r}"
        f"⇌{brs.template.smiles_c!r}  |  "
        f"source={brs.template.source}  |  "
        f"{n_plot}/{n_mem} member(s){' (sym-dedup)' if is_sym else ''}  "
        f"ego(n={eg.number_of_nodes() if eg else '-'},"
        f"e={eg.number_of_edges() if eg else '-'})  |  "
        f"● A  ◆ B  ■ C  (centroid per member)"
    ))
    fig.show()

# %% ── Final summary ───────────────────────────────────────────────────────────
_section("FINAL — Pipeline summary")

print(f"Primary adsorbate  : [O]   iso-classes = {len(o_sites)}")
print(f"  {'ads_iso':>7}  {'site_k':>6}  {'members':>7}")
for site in o_sites:
    clq  = site.atom_cliques[0]
    kstr = f"k={len(clq)}" if clq is not None else "float"
    print(f"  {site.iso_class:>7}  {kstr:>6}  {len(site.members):>7}")

if leaf_smiles:
    print(f"\nLeaf species adsorbate iso-classes:")
    for smi in leaf_smiles:
        leaf_s = [s for s in all_ads_sites if _canon_smiles(s.reactant) == smi]
        print(f"  {smi!r:>10}  iso-classes={len(leaf_s)}")

print(f"\nBond templates          : {len(templates)}")
for t in templates:
    print(f"  {t.smiles_a!r}+{t.smiles_b!r} ⇌ {t.smiles_c!r}  "
          f"({t.bond_type}, {t.source})")

print(f"\nBond reaction iso-classes : {len(bond_sites)}")
print(f"  {'bond_iso':>8}  {'A':>8}  {'B':>8}  {'C':>10}  {'members':>7}")
for brs in bond_sites:
    print(f"  {brs.iso_class:>8}  "
          f"{brs.template.smiles_a!r:>8}  "
          f"{brs.template.smiles_b!r:>8}  "
          f"{brs.template.smiles_c!r:>10}  "
          f"{len(brs.members):>7}")

print("\nDone.  Inspect figures above to check the enumerated bond reactions.")

