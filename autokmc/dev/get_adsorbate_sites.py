# %% [markdown]
# # AutoKMC2 — Adsorbate‑Site Pipeline Demo
#
# Walks through every stage of the multi‑atom adsorbate workflow for a **CO**
# molecule on a Cu(111) FCC slab, showing each step visually:
#
# 1. **Build** a Cu(111) FCC surface slab
# 2. **Find** surface atoms via ray‑casting
# 3. **Build** the atom‑connectivity graph
# 4. **Build** a CO reactant from SMILES — gas‑phase geometry, anchors & orbits
# 5. **Find adsorbate sites** — enumerate placements, reduce to iso‑classes,
#    materialise nodes on the graph
# 6. **Inspect** the representative positions for each iso‑class
# 7. **Optimise** iso‑class representative positions (rigid‑body L‑BFGS‑B)
# 8. **Propagate** the representative to every member node and show the
#    final graph

# %% ── 0. Imports & helpers ──────────────────────────────────────────────────
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
from nequip.ase import NequIPCalculator

# ── Make the workspace root importable so `autokmc` resolves ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from autokmc.structure import build_surface                          # noqa: E402
from autokmc.surface   import find_surface_atoms                     # noqa: E402
from autokmc.graph     import build_graph                            # noqa: E402
from autokmc.reactants import build_reactant                         # noqa: E402
from autokmc.find_adsorbate_sites import (                           # noqa: E402
    find_adsorbate_sites,
    optimise_adsorbate_site_positions,
)

# ── Notebook‑friendly matplotlib (magic is a no‑op when run as plain .py) ──
try:
    get_ipython().run_line_magic("matplotlib", "inline")   # type: ignore[name-defined]
except NameError:
    pass
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ── Visual palette ──────────────────────────────────────────────────────────
_SURF_COL   = "#4A90D9"       # surface atoms
_BULK_COL   = "#cccccc"       # bulk atoms
_ADS_COL    = {"C": "#2ECC71", "O": "#E74C3C"}   # adsorbate atom colours
_ADS_MARKER = {"C": "D",       "O": "^"}          # adsorbate markers
_ISO_CMAP   = plt.get_cmap("tab10")               # one colour per iso‑class

_ELEM_COL = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}


def _ecolor(sym: str) -> str:
    return _ELEM_COL.get(sym, "#888888")


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# %% [markdown]
# ---
# ## Stage 1 — Build a Cu(111) FCC Slab
#
# `build_surface` generates a pymatgen slab, orthogonalises it, tiles to
# ~10 × 10 Å and relaxes with the Allegro NequIP calculator.
# A fixed `lattice_constant` skips the bulk‑relaxation step.

# %% ── 1. Build structure ────────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str((Path(__file__).resolve().parent if "__file__" in globals() else Path(".")) / _MODEL_FILE)
print(f"Using device : {_DEVICE}")
print(f"Model file   : {_MODEL_FILE}")


def make_calc():
    """Return a fresh NequIPCalculator instance loaded from the model."""
    return NequIPCalculator.from_compiled_model(
        compile_path=str(_MODEL_PATH),
        device=_DEVICE,
    )


calc = make_calc()
print(f"Calculator : {calc.__class__.__name__}")
print(f"Model      : {_MODEL_PATH}")

atoms = build_surface(
    composition="Cu",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    lattice_constant=3.615,   # Å — fixed; skips bulk relax
    min_slab_size=7.0,
    min_vacuum_size=12.0,
    goal_x=10.0,
    goal_y=10.0,
    n_freeze_layers=2,
    calculator=calc,
    verbose=True,
)

print(f"\nSlab built  : {len(atoms)} atoms")
print(f"Cell diag   : {np.diag(atoms.get_cell()).round(3)} Å")

# ── Visualise ────────────────────────────────────────────────────────────────
pos  = atoms.get_positions()
syms = np.array(atoms.get_chemical_symbols())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Stage 1 — Built Structure", fontweight="bold")

ax = axes[0]
ax.scatter(pos[:, 0], pos[:, 2], c=[_ecolor(s) for s in syms],
           s=60, edgecolors="k", linewidths=0.4)
ax.set_xlabel("x (Å)"); ax.set_ylabel("z (Å)")
ax.set_title("Side view (x–z)")
ax.legend(handles=[mpatches.Patch(color=_ecolor(s), label=s)
                   for s in np.unique(syms)], fontsize=8)

ax = axes[1]
sc = ax.scatter(pos[:, 0], pos[:, 1], c=pos[:, 2],
                cmap="viridis", s=60, edgecolors="k", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="z (Å)")
ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title("Top view (x–y), coloured by layer height")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 2 — Find Surface Atoms
#
# `find_surface_atoms` auto‑detects the geometry (slab → ray‑casting) and
# tags atoms in `atoms.arrays["surface"]` for `build_graph`.

# %% ── 2. Find surface atoms ─────────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms")

surf_result = find_surface_atoms(atoms, tag_atoms=True)
surf_mask   = surf_result.mask

print(f"Method        : {surf_result.method}")
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")

bulk_mask = ~surf_mask
surf_pos  = pos[surf_mask]

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Stage 2 — Surface Atom Identification", fontweight="bold")
ax.scatter(pos[bulk_mask, 0], pos[bulk_mask, 1],
           c=_BULK_COL, s=45, edgecolors="k", linewidths=0.3,
           label="bulk", zorder=2)
ax.scatter(surf_pos[:, 0], surf_pos[:, 1],
           c=_SURF_COL, s=100, edgecolors="k", linewidths=0.5,
           label="surface", zorder=3)
ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Top‑down  —  {surf_mask.sum()} surface  /  {bulk_mask.sum()} bulk")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 3 — Build the Connectivity Graph
#
# `build_graph` converts the tagged `Atoms` into a NetworkX graph.  Every
# node is one atom; every edge is a covalent bond.

# %% ── 3. Build graph ────────────────────────────────────────────────────────
_section("STAGE 3 — Build connectivity graph")

G = build_graph(atoms)

n_surf_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "surface")
n_bulk_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "bulk")
print(f"Nodes : {G.number_of_nodes()}  ({n_surf_g} surface, {n_bulk_g} bulk)")
print(f"Edges : {G.number_of_edges()}")
print(f"PBC   : {G.graph['pbc'].tolist()}")

surf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]
Gsub       = G.subgraph(surf_nodes)
pos_dict   = {n: (G.nodes[n]["position"][0], G.nodes[n]["position"][1])
              for n in Gsub.nodes}

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Stage 3 — Surface Connectivity Subgraph", fontweight="bold")
nx.draw_networkx_edges(Gsub, pos_dict, ax=ax, alpha=0.35,
                       edge_color="#888888", width=0.8)
nx.draw_networkx_nodes(Gsub, pos_dict, ax=ax, node_size=80,
                       node_color=_SURF_COL, edgecolors="k", linewidths=0.5)
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Surface subgraph — {len(surf_nodes)} nodes, "
             f"{Gsub.number_of_edges()} edges")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 4 — Build the CO Reactant
#
# `build_reactant` parses the SMILES string with RDKit, embeds a 3‑D
# conformer (ETKDGv3 + MMFF94), tags every atom as `adsorbate`, builds its
# connectivity graph, computes intramolecular automorphism orbits and marks
# the convex‑hull‑exposed **anchor atoms** eligible to bond to the surface.

# %% ── 4. Build CO reactant ──────────────────────────────────────────────────
_section("STAGE 4 — Build CO reactant")

# CO: carbon bonded to oxygen via triple bond.
# [C-]#[O+] is the standard Lewis‑structure SMILES for carbon monoxide.
co = build_reactant("[C-]#[O+]", add_hydrogens=False)

print(f"Formula       : {co.atoms.get_chemical_formula()}")
print(f"n_atoms       : {len(co.atoms)}")
print(f"Bond length   : {co.atoms.get_all_distances()[0, 1]:.3f} Å")
print(f"Anchor atoms  : {co.anchor_atoms}  "
      f"(elements: {[co.atoms.get_chemical_symbols()[i] for i in co.anchor_atoms]})")
print(f"Unique nodes  : {co.unique_nodes}")
print(f"Anchor orbits : {co.anchor_orbit}")
print(f"Gas energy    : {co.energy:.4f} eV"
      if not np.isnan(co.energy) else "Gas energy    : (no calculator — NaN)")

# ── Visualise gas‑phase CO geometry ─────────────────────────────────────────
react_pos  = co.atoms.get_positions()
react_syms = co.atoms.get_chemical_symbols()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Stage 4 — CO Reactant (gas phase)", fontweight="bold")

# 3‑D scatter as two orthogonal projections
for ax, (xi, yi, xlabel, ylabel, title) in zip(axes, [
    (0, 2, "x (Å)", "z (Å)", "x–z view"),
    (0, 1, "x (Å)", "y (Å)", "x–y view"),
]):
    for i, (sym, p) in enumerate(zip(react_syms, react_pos)):
        col = _ADS_COL.get(sym, "#888888")
        mrk = _ADS_MARKER.get(sym, "o")
        ax.scatter(p[xi], p[yi], c=col, s=300, marker=mrk,
                   edgecolors="k", linewidths=0.8, zorder=3, label=sym)
        ax.annotate(f" {sym}{i}", (p[xi], p[yi]), fontsize=9)
    # Bond line
    ax.plot([react_pos[0, xi], react_pos[1, xi]],
            [react_pos[0, yi], react_pos[1, yi]],
            color="#555555", lw=2, zorder=2)
    # Highlight anchor atoms with a circle
    for i in co.anchor_atoms:
        ax.scatter(react_pos[i, xi], react_pos[i, yi],
                   s=600, facecolors="none", edgecolors="gold",
                   linewidths=2.0, zorder=4, label="anchor" if i == co.anchor_atoms[0] else "")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [mpatches.Patch(color=_ADS_COL[s], label=s) for s in react_syms]
    handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                            markeredgecolor="gold", markersize=10,
                            markeredgewidth=2, label="anchor")]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites
#
# `find_adsorbate_sites` runs the full pipeline:
#
# 1. Lazily compute single‑atom anchor sites for every anchor element
# 2. Enumerate all orbit‑canonicalised anchor subsets
# 3. Backtracking chain placement (intramolecular distance filter)
# 4. Surface‑connectivity guard (APSP BFS)
# 5. Reduce to iso‑classes via ego‑subgraph isomorphism
# 6. Materialise one `type="adsorbate"` node per reactant atom per member on *G*

# %% ── 5. Find adsorbate sites ───────────────────────────────────────────────
_section("STAGE 5 — Find adsorbate sites for CO")

sites_CO = find_adsorbate_sites(G, co, verbose=True)

print(f"\nCO: {len(sites_CO)} unique iso‑class(es)")
for site in sites_CO:
    bonded = [(i, c) for i, c in enumerate(site.atom_cliques) if c is not None]
    bonded_str = "  ".join(
        f"{react_syms[i]}→k={len(c)}" for i, c in bonded
    ) if bonded else "all‑floating"
    print(f"  iso {site.iso_class:2d}  {bonded_str:<30s}"
          f"  members={len(site.members):<4d}"
          f"  rep_pos[C]={np.round(site.positions[0], 2)}")

# ── Visualise all member placements top‑down, coloured by iso‑class ─────────
ads_nodes = [(n, d) for n, d in G.nodes(data=True)
             if d.get("type") == "adsorbate" and d.get("reactant") == co.smiles]

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Stage 5 — All CO Placements (top‑down by iso‑class)",
             fontweight="bold")

# Surface backdrop
ax.scatter(surf_pos[:, 0], surf_pos[:, 1],
           c=_SURF_COL, s=55, edgecolors="k", linewidths=0.4,
           zorder=1, label="surface Cu")

# Draw anchor‑bond edges
for n, d in ads_nodes:
    if not d.get("is_bonded"):
        continue
    p_ads = d["position"]
    for nb in G.neighbors(n):
        if G.nodes[nb].get("type") != "surface":
            continue
        p_s = G.nodes[nb]["position"]
        ax.plot([p_ads[0], p_s[0]], [p_ads[1], p_s[1]],
                color="#aaaaaa", lw=0.5, alpha=0.4, zorder=2)

# Draw adsorbate atoms and intra‑molecular bonds
for n, d in ads_nodes:
    sym      = d["element"]
    p        = d["position"]
    iso_cls  = d["iso_class"]
    col      = _ISO_CMAP(iso_cls % 10)
    filled   = d.get("is_bonded", False)
    ax.scatter(p[0], p[1],
               c=[col], s=90,
               marker=_ADS_MARKER.get(sym, "o"),
               edgecolors="k" if filled else col,
               linewidths=0.8 if filled else 1.5,
               facecolors=[col] if filled else "none",
               zorder=4)

ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"{len(sites_CO)} iso‑classes  ·  "
             f"{sum(len(s.members) for s in sites_CO)} total placements")

iso_patches = [
    mpatches.Patch(color=_ISO_CMAP(i % 10), label=f"iso {i}")
    for i in range(len(sites_CO))
]
type_handles = [
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#555",
               markeredgecolor="k", markersize=8, label="C (bonded, filled)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#555",
               markeredgecolor="k", markersize=8, label="O (bonded, filled)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#555", markersize=8, markeredgewidth=1.5,
               label="floating (open)"),
]
ax.legend(handles=iso_patches + type_handles, fontsize=7,
          loc="upper right", ncol=2, framealpha=0.9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 6 — Representative Positions per Iso‑Class
#
# Each `AdsorbateSite.positions` (shape `(n_atoms, 3)`) holds the Cartesian
# positions for the **representative** member of that iso‑class.  The atoms
# are shown on top of the surface layer, with the iso‑class ego‑subgraph
# drawn for context.

# %% ── 6. Representative positions ───────────────────────────────────────────
_section("STAGE 6 — Representative positions per iso‑class")

n_iso  = len(sites_CO)
ncols  = min(n_iso, 4)
nrows  = (n_iso + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4 * ncols, 3.8 * nrows),
                         squeeze=False)
fig.suptitle("Stage 6 — Representative Positions per Iso‑Class",
             fontweight="bold")

for ax_flat, site in zip(axes.ravel(), sites_CO):
    col = _ISO_CMAP(site.iso_class % 10)

    # Surface backdrop
    ax_flat.scatter(surf_pos[:, 0], surf_pos[:, 1],
                    c="#e0e0e0", s=35, edgecolors="#aaaaaa",
                    linewidths=0.3, zorder=1)

    # Ego‑subgraph surface atoms (border highlight)
    if site.ego_graph is not None:
        ego_surf = [
            G.nodes[n]["position"] for n in site.ego_graph.nodes
            if n in G and G.nodes[n].get("type") == "surface"
        ]
        if ego_surf:
            ego_arr = np.array(ego_surf)
            ax_flat.scatter(ego_arr[:, 0], ego_arr[:, 1],
                            c="#f5c518", s=60, edgecolors="k",
                            linewidths=0.5, zorder=2, label="ego surf")

    # Representative adsorbate atoms
    rep_pos  = np.asarray(site.positions, dtype=float)
    rep_syms = [co.atoms.get_chemical_symbols()[i] for i in range(site.n_atoms)]
    bonded   = [c is not None for c in site.atom_cliques]

    # Intra‑molecular bond line
    if site.n_atoms >= 2:
        ax_flat.plot([rep_pos[0, 0], rep_pos[1, 0]],
                     [rep_pos[0, 1], rep_pos[1, 1]],
                     color=col, lw=2.0, zorder=4, alpha=0.8)

    for i, (sym, p, is_b) in enumerate(zip(rep_syms, rep_pos, bonded)):
        mrk = _ADS_MARKER.get(sym, "o")
        fc  = col if is_b else "none"
        ec  = "k" if is_b else col
        ax_flat.scatter(p[0], p[1], c=[col] if is_b else [[0, 0, 0, 0]],
                        s=220, marker=mrk,
                        facecolors=fc, edgecolors=ec,
                        linewidths=1.5, zorder=5)
        ax_flat.annotate(f" {sym}", (p[0], p[1]), fontsize=7, va="center")

    n_bonded  = sum(1 for c in site.atom_cliques if c is not None)
    ks        = [len(c) for c in site.atom_cliques if c is not None]
    ks_str    = "+".join(str(k) for k in ks) if ks else "none"
    ax_flat.set_aspect("equal")
    ax_flat.set_title(
        f"iso {site.iso_class}  ·  {n_bonded}/{site.n_atoms} bonded  "
        f"(k={ks_str})\n{len(site.members)} members",
        fontsize=8,
    )
    ax_flat.set_xlabel("x (Å)", fontsize=7)
    ax_flat.set_ylabel("y (Å)", fontsize=7)
    ax_flat.tick_params(labelsize=7)

for ax_flat in axes.ravel()[n_iso:]:
    ax_flat.set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 7 — Optimise Representative Positions
#
# `optimise_adsorbate_site_positions` treats each placement as 6 rigid‑body
# DOF (3 translation + 3 axis‑angle rotation) and minimises a
# calculator‑free objective:
#
# ```
# E = restraint × Σ_{bonded i} ‖p_i − p*_i‖²
#   + repulsion × Σ_{i,s}       max(0, R_min − d_is)²
# ```
#
# After refining the representative, Kabsch ego‑alignment propagates the new
# geometry to every other member.

# %% ── 7. Optimise positions ──────────────────────────────────────────────────
_section("STAGE 7 — Optimise representative positions")

# Store pre‑optimisation representative positions for comparison.
pre_opt = {site.iso_class: np.asarray(site.positions, dtype=float).copy()
           for site in sites_CO}

sites_CO = optimise_adsorbate_site_positions(
    G, co.smiles, co,
    restraint_weight=10.0,
    repulsion_weight=1.0,
    n_restarts=6,
    try_flip=True,
    verbose=True,
)

# ── Visualise before / after per iso‑class ───────────────────────────────────
ncols = min(n_iso, 4)
nrows = (n_iso + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4 * ncols, 3.8 * nrows),
                         squeeze=False)
fig.suptitle("Stage 7 — Before (open ◌) vs After (filled ●) Optimisation",
             fontweight="bold")

for ax_flat, site in zip(axes.ravel(), sites_CO):
    col     = _ISO_CMAP(site.iso_class % 10)
    pre     = pre_opt[site.iso_class]
    post    = np.asarray(site.positions, dtype=float)
    rep_sym = [co.atoms.get_chemical_symbols()[i] for i in range(site.n_atoms)]

    # Surface backdrop
    ax_flat.scatter(surf_pos[:, 0], surf_pos[:, 1],
                    c="#e0e0e0", s=35, edgecolors="#aaaaaa",
                    linewidths=0.3, zorder=1)

    # Before — open markers with dashed bond
    if site.n_atoms >= 2:
        ax_flat.plot([pre[0, 0], pre[1, 0]], [pre[0, 1], pre[1, 1]],
                     color=col, lw=1.5, ls="--", alpha=0.5, zorder=3)
    for i, sym in enumerate(rep_sym):
        mrk = _ADS_MARKER.get(sym, "o")
        ax_flat.scatter(pre[i, 0], pre[i, 1],
                        facecolors="none", edgecolors=col,
                        s=140, marker=mrk, linewidths=1.5,
                        zorder=4, alpha=0.7)

    # After — filled markers with solid bond
    if site.n_atoms >= 2:
        ax_flat.plot([post[0, 0], post[1, 0]], [post[0, 1], post[1, 1]],
                     color=col, lw=2.0, zorder=5)
    for i, sym in enumerate(rep_sym):
        mrk = _ADS_MARKER.get(sym, "o")
        ax_flat.scatter(post[i, 0], post[i, 1],
                        c=[col], s=200, marker=mrk,
                        edgecolors="k", linewidths=0.8, zorder=6)
        ax_flat.annotate(f" {sym}", (post[i, 0], post[i, 1]),
                         fontsize=7, va="center")

    rms = float(np.sqrt(np.mean(np.sum((post - pre) ** 2, axis=1))))
    ax_flat.set_aspect("equal")
    ax_flat.set_title(f"iso {site.iso_class}  ·  ΔRMS {rms:.3f} Å", fontsize=8)
    ax_flat.set_xlabel("x (Å)", fontsize=7)
    ax_flat.set_ylabel("y (Å)", fontsize=7)
    ax_flat.tick_params(labelsize=7)

for ax_flat in axes.ravel()[n_iso:]:
    ax_flat.set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 8 — Propagation: Representative → All Members
#
# After optimisation, `member_node_ids[0]` holds the representative's nodes
# (already updated on *G*).  Every subsequent member was Kabsch‑propagated
# from the representative's ego‑alignment.  Here we read the positions back
# directly from *G* and show:
#
# * ★ gold star — optimised representative position
# * ○ coloured circles — all other member positions from *G*
# * thin lines — representative → each member

# %% ── 8a. Propagation table ──────────────────────────────────────────────────
_section("STAGE 8 — Propagation: representative → all members")

print(f"\n{'iso':>4}  {'n_atoms':>7}  {'n_members':>9}  "
      f"{'rep_pos[C]':>30}  first_other_member_pos[C]")
print("─" * 90)
for site in sites_CO:
    rep_p  = site.positions[0]          # C atom in representative
    if len(site.member_node_ids) > 1:
        mem1_p = G.nodes[site.member_node_ids[1][0]]["position"]   # C of member 1
    else:
        mem1_p = np.full(3, np.nan)
    print(f"  {site.iso_class:>2}   {site.n_atoms:>7}  {len(site.members):>9}  "
          f"  {np.round(rep_p, 3)}  {np.round(mem1_p, 3)}")

# %% ── 8b. Per‑iso‑class propagation plots ────────────────────────────────────
ncols = min(n_iso, 4)
nrows = (n_iso + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4 * ncols, 3.8 * nrows),
                         squeeze=False)
fig.suptitle("Stage 8 — Propagation:  representative ★  vs  all members ○",
             fontweight="bold")

for ax_flat, site in zip(axes.ravel(), sites_CO):
    col     = _ISO_CMAP(site.iso_class % 10)
    rep_sym = co.atoms.get_chemical_symbols()

    ax_flat.scatter(surf_pos[:, 0], surf_pos[:, 1],
                    c="#e0e0e0", s=30, edgecolors="#bbbbbb",
                    linewidths=0.3, zorder=1)

    # All member placements from G
    for m_idx, node_ids in enumerate(site.member_node_ids):
        m_pos = np.array([G.nodes[nid]["position"] for nid in node_ids])
        for i, (sym, p) in enumerate(zip(rep_sym, m_pos)):
            mrk = _ADS_MARKER.get(sym, "o")
            fc  = col if m_idx == 0 else "none"
            ax_flat.scatter(p[0], p[1],
                            facecolors=fc, edgecolors=col,
                            s=80, marker=mrk, linewidths=1.0, zorder=3)
        if len(m_pos) >= 2:
            ax_flat.plot([m_pos[0, 0], m_pos[1, 0]],
                         [m_pos[0, 1], m_pos[1, 1]],
                         color=col, lw=0.9, alpha=0.5, zorder=2)

    # Representative — large gold star per atom
    rep_p = np.asarray(site.positions, dtype=float)
    for i, sym in enumerate(rep_sym):
        ax_flat.scatter(rep_p[i, 0], rep_p[i, 1],
                        c="gold", s=320, marker="*",
                        edgecolors="k", linewidths=0.8, zorder=5)

    # Lines: representative C atom → member C atoms
    rep_c = rep_p[0]
    for node_ids in site.member_node_ids[1:]:
        mem_c = G.nodes[node_ids[0]]["position"]
        ax_flat.plot([rep_c[0], mem_c[0]], [rep_c[1], mem_c[1]],
                     color="gray", lw=0.6, alpha=0.5, zorder=2)

    ax_flat.set_aspect("equal")
    ax_flat.set_title(
        f"iso {site.iso_class}  ·  {len(site.members)} members\n"
        "filled = rep  ·  open = other  ·  ★ = optimised rep",
        fontsize=7.5,
    )
    ax_flat.set_xlabel("x (Å)", fontsize=7)
    ax_flat.set_ylabel("y (Å)", fontsize=7)
    ax_flat.tick_params(labelsize=7)

for ax_flat in axes.ravel()[n_iso:]:
    ax_flat.set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Final Summary — Full Graph with All CO Adsorbate Nodes
#
# `G` now contains `type="adsorbate"` nodes for every CO placement.  The
# final plot overlays all of them on the surface subgraph, using:
#
# * **◆ diamond** → C atom    (coloured by iso‑class)
# * **▲ triangle** → O atom   (coloured by iso‑class)
# * filled = bonded to surface,  open = floating

# %% ── Final: full graph with adsorbate nodes ─────────────────────────────────
_section("FINAL — Full graph with all CO adsorbate nodes")

ads_nodes_final = [(n, d) for n, d in G.nodes(data=True)
                   if d.get("type") == "adsorbate"
                   and d.get("reactant") == co.smiles]

total_ads = len(ads_nodes_final)
n_bonded  = sum(1 for _, d in ads_nodes_final if d.get("is_bonded"))
print(f"Adsorbate nodes in G : {total_ads}  ({n_bonded} bonded, "
      f"{total_ads - n_bonded} floating)")
print(f"Iso‑classes          : {len(sites_CO)}")
print(f"Total placements     : {sum(len(s.members) for s in sites_CO)}")

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Final — Surface Graph + All CO Adsorbate Nodes", fontweight="bold")

# Surface atoms
ax.scatter(surf_pos[:, 0], surf_pos[:, 1],
           c=_SURF_COL, s=55, edgecolors="k", linewidths=0.4,
           zorder=2, label="surface Cu")

# Anchor‑bond edges
for n, d in ads_nodes_final:
    if not d.get("is_bonded"):
        continue
    ap = d["position"]
    for nb in G.neighbors(n):
        if G.nodes[nb].get("type") != "surface":
            continue
        sp = G.nodes[nb]["position"]
        ax.plot([ap[0], sp[0]], [ap[1], sp[1]],
                color="#999999", lw=0.5, alpha=0.4, zorder=1)

# Intra‑molecular bonds
drawn_pairs: set = set()
for n, d in ads_nodes_final:
    for nb in G.neighbors(n):
        if not G.edges[n, nb].get("intra_adsorbate"):
            continue
        key = (min(n, nb), max(n, nb))
        if key in drawn_pairs:
            continue
        drawn_pairs.add(key)
        pa = d["position"]
        pb = G.nodes[nb]["position"]
        iso_cls = d["iso_class"]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color=_ISO_CMAP(iso_cls % 10), lw=1.2, alpha=0.7, zorder=3)

# Adsorbate atoms
for n, d in ads_nodes_final:
    sym     = d["element"]
    p       = d["position"]
    iso_cls = d["iso_class"]
    col     = _ISO_CMAP(iso_cls % 10)
    mrk     = _ADS_MARKER.get(sym, "o")
    filled  = d.get("is_bonded", False)
    ax.scatter(p[0], p[1],
               facecolors=[col] if filled else "none",
               edgecolors=[col] if not filled else "k",
               s=100, marker=mrk,
               linewidths=1.5 if not filled else 0.8,
               zorder=4)

# Legend
iso_patches = [
    mpatches.Patch(color=_ISO_CMAP(i % 10), label=f"iso {i}")
    for i in range(len(sites_CO))
]
type_handles = [
    mpatches.Patch(color=_SURF_COL, label="surface Cu"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#27AE60",
               markeredgecolor="k", markersize=9, label="C (bonded)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#E74C3C",
               markeredgecolor="k", markersize=9, label="O (bonded)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#555", markersize=9, markeredgewidth=1.5,
               label="floating"),
]
ax.legend(handles=type_handles + iso_patches, fontsize=7,
          loc="upper right", ncol=2, framealpha=0.9)

ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Surface + {total_ads} CO adsorbate nodes across "
             f"{len(sites_CO)} iso‑classes")
plt.tight_layout()
plt.show()

print("\nPipeline complete.")

