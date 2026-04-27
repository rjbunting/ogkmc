# %% [markdown]
# # AutoKMC2 — Anchor‑Site Pipeline Demo
#
# Walks through every stage of the anchor‑site workflow, showing each step
# visually:
#
# 1. **Build** a Cu(111) FCC surface slab
# 2. **Find** surface atoms via ray‑casting
# 3. **Build** the atom‑connectivity graph
# 4. **Find anchor sites** for C and O (enumerate → iso‑class → optimise)
# 5. **Inspect** the optimised representative position for each unique site
# 6. **Propagate** the representative position to every matching anchor node
#    on the graph, and visualise the full result

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

# # ── Make the workspace root importable so `autokmc` resolves ──
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

from autokmc.structure import build_surface          # noqa: E402
from autokmc.surface   import find_surface_atoms      # noqa: E402
from autokmc.graph     import build_graph             # noqa: E402
from autokmc.find_anchors import find_anchor_sites    # noqa: E402

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

# ── Notebook‑friendly matplotlib (magic is a no‑op when run as plain .py) ──
try:
    get_ipython().run_line_magic("matplotlib", "inline")   # type: ignore[name-defined]
except NameError:
    pass
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ── Visual palette ──────────────────────────────────────────────────────────
_ELEM_COL    = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}
_TYPE_COL    = {"bulk": "#cccccc", "surface": "#4A90D9"}
_K_COL       = {1: "#27AE60", 2: "#E67E22", 3: "#8E44AD", 4: "#E74C3C"}
_K_LABEL     = {1: "top (k=1)", 2: "bridge (k=2)", 3: "hollow (k=3)", 4: "4‑fold (k=4)"}
_ELEM_MARKER = {"C": "D", "O": "^"}


def _ecolor(sym: str) -> str:
    return _ELEM_COL.get(sym, "#888888")


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# %% [markdown]
# ---
# ## Stage 1 — Build a Cu(111) FCC Slab
#
# `build_surface` from **autokmc** generates a pymatgen slab, orthogonalises
# it, tiles to ~10 × 10 Å, and relaxes with EMT.  Passing a fixed
# `lattice_constant` skips the bulk‑relaxation step.

# %% ── 1. Build structure ────────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition="Cu",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    lattice_constant=3.615,   # Å — EMT‑optimised; skips bulk relax
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
cols = [_ecolor(s) for s in syms]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Stage 1 — Built Structure", fontweight="bold")

# Side view (x–z)
ax = axes[0]
ax.scatter(pos[:, 0], pos[:, 2], c=cols, s=60,
           edgecolors="k", linewidths=0.4, zorder=2)
ax.set_xlabel("x (Å)"); ax.set_ylabel("z (Å)")
ax.set_title("Side view (x–z)")
patches = [mpatches.Patch(color=_ecolor(s), label=s)
           for s in np.unique(syms)]
ax.legend(handles=patches, fontsize=8)

# Top view (x–y) coloured by z‑height to reveal layers
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
# `find_surface_atoms` auto‑detects the geometry via PBC‑connectivity, then
# runs **ray‑casting** for a slab.  `tag_atoms=True` writes the result into
# `atoms.arrays["surface"]` (required by `build_graph`).

# %% ── 2. Find surface atoms ─────────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms")

surf_result = find_surface_atoms(atoms, tag_atoms=True)
surf_mask   = surf_result.mask
surf_idx    = surf_result.indices

print(f"Method        : {surf_result.method}")
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")
print(f"Bulk atoms    : {(~surf_mask).sum()}")

# ── Visualise ────────────────────────────────────────────────────────────────
bulk_mask = ~surf_mask

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Stage 2 — Surface Atom Identification", fontweight="bold")

ax.scatter(pos[bulk_mask, 0], pos[bulk_mask, 1],
           c=_TYPE_COL["bulk"], s=45, edgecolors="k",
           linewidths=0.3, label="bulk", zorder=2)
ax.scatter(pos[surf_mask, 0], pos[surf_mask, 1],
           c=_TYPE_COL["surface"], s=100, edgecolors="k",
           linewidths=0.5, label="surface", zorder=3)

ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Top‑down view  —  {surf_mask.sum()} surface  /  "
             f"{bulk_mask.sum()} bulk")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 3 — Build the Connectivity Graph
#
# `build_graph` converts the tagged `Atoms` into a **NetworkX** graph where
# every node is one atom and every edge is a covalent bond (determined by
# ASE natural cut‑offs).  Graph‑level keys `"cell"` and `"pbc"` are set
# automatically.

# %% ── 3. Build graph ────────────────────────────────────────────────────────
_section("STAGE 3 — Build connectivity graph")

G = build_graph(atoms)

n_surf_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "surface")
n_bulk_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "bulk")
print(f"Nodes : {G.number_of_nodes()}  ({n_surf_g} surface, {n_bulk_g} bulk)")
print(f"Edges : {G.number_of_edges()}")
print(f"PBC   : {G.graph['pbc'].tolist()}")

# ── Visualise — surface subgraph in x–y ─────────────────────────────────────
surf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]
Gsub = G.subgraph(surf_nodes)
pos_dict = {n: (G.nodes[n]["position"][0], G.nodes[n]["position"][1])
            for n in Gsub.nodes}

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Stage 3 — Surface Connectivity Subgraph", fontweight="bold")

nx.draw_networkx_edges(Gsub, pos_dict, ax=ax, alpha=0.35,
                       edge_color="#888888", width=0.8)
nx.draw_networkx_nodes(Gsub, pos_dict, ax=ax, node_size=80,
                       node_color=_TYPE_COL["surface"],
                       edgecolors="k", linewidths=0.5)

ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Surface subgraph — {len(surf_nodes)} nodes, "
             f"{Gsub.number_of_edges()} edges")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 4 — Find Anchor Sites for C and O
#
# `find_anchor_sites` runs the full pipeline internally:
#
# 1. Build the **co‑bonding graph** for the adsorbate element
# 2. Enumerate all **cliques** (k=1 … k_max)
# 3. Reduce to **iso‑classes** by ego‑subgraph isomorphism
# 4. **Optimise** the representative position (L‑BFGS‑B, no calculator)
# 5. **Kabsch‑propagate** to every member clique
# 6. **Materialise** anchor nodes on `G` (`type="anchor"`)

# %% ── 4. Find anchor sites ──────────────────────────────────────────────────
_section("STAGE 4 — Find anchor sites for C and O")

sites_C = find_anchor_sites(G, "C", verbose=True)
sites_O = find_anchor_sites(G, "O", verbose=True)

for elem, sites in [("C", sites_C), ("O", sites_O)]:
    print(f"\n{elem}: {len(sites)} unique iso‑class(es)")
    for iso in sites:
        lbl = _K_LABEL.get(iso.k, f"k={iso.k}")
        print(f"  iso_class={iso.iso_class}  {lbl:<18s}"
              f"  members={len(iso.members):<4d}"
              f"  rep_pos={np.round(iso.position, 3)}")

# %% [markdown]
# ---
# ## Stage 5 — Optimised Representative Positions
#
# Each `AnchorSite.position` is the geometrically optimised position for the
# *representative* clique of that iso‑class.  Sites are coloured by **k**
# and shown on top of the surface atom layer.

# %% ── 5. Plot representative positions ──────────────────────────────────────
_section("STAGE 5 — Optimised representative positions")

surf_pos = pos[surf_mask]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Stage 5 — Optimised Representative Positions", fontweight="bold")

for ax, (elem, sites) in zip(axes, [("C", sites_C), ("O", sites_O)]):
    # Surface atom backdrop
    ax.scatter(surf_pos[:, 0], surf_pos[:, 1],
               c="#dddddd", s=50, edgecolors="#aaaaaa",
               linewidths=0.3, zorder=1, label="surface")

    # Unique representative per iso‑class
    plotted_labels: set = set()
    for iso in sites:
        p   = iso.position
        col = _K_COL.get(iso.k, "#333333")
        lbl = _K_LABEL.get(iso.k, f"k={iso.k}")
        ax.scatter(p[0], p[1], c=col, s=220,
                   marker=_ELEM_MARKER.get(elem, "*"),
                   edgecolors="k", linewidths=0.7, zorder=4,
                   label=lbl if lbl not in plotted_labels else "")
        plotted_labels.add(lbl)
        ax.annotate(f"cls {iso.iso_class}", (p[0], p[1]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_title(f"{elem} — {len(sites)} unique anchor site(s)")
    ax.legend(fontsize=7, loc="upper right")

axes[0].set_ylabel("y (Å)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Stage 6 — Propagate Representative → All Matching Anchor Nodes
#
# `find_anchor_sites` already propagated the representative's position to
# every member via Kabsch ego‑alignment and wrote it into the graph.  Here
# we **explicitly read those positions back** from `G` and confirm that
# every `iso.node_ids[i]` carries a position consistent with the iso‑class.
#
# The plots show:
# * ★ gold star — optimised representative position (`iso.position`)
# * ○ coloured circle — each member anchor node position from `G`
# * thin grey lines — representative → member

# %% ── 6a. Propagation table ─────────────────────────────────────────────────
_section("STAGE 6 — Propagation table")

for elem, sites in [("C", sites_C), ("O", sites_O)]:
    print(f"\nElement: {elem}")
    print(f"  {'iso':>4}  {'k':>3}  {'members':>7}  "
          f"{'rep_pos':>32}  first_member_pos")
    print("  " + "─" * 85)
    for iso in sites:
        rep_p = iso.position
        mem_p = (G.nodes[iso.node_ids[0]]["position"]
                 if iso.node_ids else np.full(3, np.nan))
        print(f"  {iso.iso_class:>4}  {iso.k:>3}  {len(iso.members):>7}"
              f"  {np.round(rep_p, 3)}  {np.round(mem_p, 3)}")

# %% ── 6b. Per‑iso‑class propagation plots ───────────────────────────────────
for elem, sites in [("C", sites_C), ("O", sites_O)]:
    ncols = min(len(sites), 4)
    nrows = (len(sites) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.8 * nrows),
                             squeeze=False)
    fig.suptitle(
        f"Stage 6 — {elem} propagation:  representative ★  vs  all members ○",
        fontweight="bold",
    )

    for ax_flat, iso in zip(axes.ravel(), sites):
        col = _K_COL.get(iso.k, "#333333")

        # Surface backdrop
        ax_flat.scatter(surf_pos[:, 0], surf_pos[:, 1],
                        c="#e8e8e8", s=30, edgecolors="#bbbbbb",
                        linewidths=0.3, zorder=1)

        # Member anchor nodes (from G)
        for nid in iso.node_ids:
            mp = G.nodes[nid]["position"]
            ax_flat.scatter(mp[0], mp[1], c=col, s=80, marker="o",
                            edgecolors="k", linewidths=0.5, zorder=3)

        # Representative — gold star
        rp = iso.position
        ax_flat.scatter(rp[0], rp[1], c="gold", s=360,
                        marker="*", edgecolors="k",
                        linewidths=0.8, zorder=5)

        # Lines: representative → each member
        for nid in iso.node_ids:
            mp = G.nodes[nid]["position"]
            ax_flat.plot([rp[0], mp[0]], [rp[1], mp[1]],
                         color="gray", lw=0.6, alpha=0.55, zorder=2)

        lbl = _K_LABEL.get(iso.k, f"k={iso.k}")
        ax_flat.set_aspect("equal")
        ax_flat.set_title(f"{elem}  {lbl}  cls {iso.iso_class}\n"
                          f"{len(iso.members)} members", fontsize=8)
        ax_flat.set_xlabel("x (Å)", fontsize=7)
        ax_flat.set_ylabel("y (Å)", fontsize=7)
        ax_flat.tick_params(labelsize=7)

    # Hide spare axes
    for ax_flat in axes.ravel()[len(sites):]:
        ax_flat.set_visible(False)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## Final Summary — Full Graph with All Anchor Nodes
#
# `G` now contains anchor nodes for both C and O.  Below we overlay every
# anchor node on the surface subgraph, colouring by element and k.

# %% ── Final: full graph with anchor nodes ───────────────────────────────────
_section("FINAL — Surface graph + all C / O anchor nodes")

anchor_C = [n for n, d in G.nodes(data=True)
            if d["type"] == "anchor" and d["element"] == "C"]
anchor_O = [n for n, d in G.nodes(data=True)
            if d["type"] == "anchor" and d["element"] == "O"]

print(f"Anchor nodes in G :  C → {len(anchor_C)},  O → {len(anchor_O)}")

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Final — Surface Graph + All Anchor Nodes (C & O)",
             fontweight="bold")

# Surface atoms
ax.scatter(surf_pos[:, 0], surf_pos[:, 1],
           c=_TYPE_COL["surface"], s=55, edgecolors="k",
           linewidths=0.4, zorder=2)

# Anchor‑bond edges
for nid in anchor_C + anchor_O:
    ap = G.nodes[nid]["position"]
    for nb in G.neighbors(nid):
        if G.nodes[nb]["type"] != "surface":
            continue
        sp = G.nodes[nb]["position"]
        ax.plot([ap[0], sp[0]], [ap[1], sp[1]],
                color="#999999", lw=0.5, alpha=0.35, zorder=1)

# C anchor nodes (diamonds, coloured by k)
if anchor_C:
    cp  = np.array([G.nodes[n]["position"] for n in anchor_C])
    ck  = [G.nodes[n]["k"] for n in anchor_C]
    ax.scatter(cp[:, 0], cp[:, 1],
               c=[_K_COL.get(k, "#333") for k in ck],
               s=110, marker="D", edgecolors="darkgreen",
               linewidths=0.8, zorder=4)

# O anchor nodes (triangles, coloured by k)
if anchor_O:
    op  = np.array([G.nodes[n]["position"] for n in anchor_O])
    ok  = [G.nodes[n]["k"] for n in anchor_O]
    ax.scatter(op[:, 0], op[:, 1],
               c=[_K_COL.get(k, "#333") for k in ok],
               s=110, marker="^", edgecolors="darkred",
               linewidths=0.8, zorder=4)

# Legend
type_handles = [
    mpatches.Patch(color=_TYPE_COL["surface"], label="surface atom"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#27AE60",
               markeredgecolor="darkgreen", markersize=9, label="C anchor"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#27AE60",
               markeredgecolor="darkred", markersize=9, label="O anchor"),
]
k_handles = [mpatches.Patch(color=col, label=_K_LABEL[k])
             for k, col in _K_COL.items() if k <= 4]
ax.legend(handles=type_handles + k_handles, fontsize=7,
          loc="upper right", framealpha=0.9)

ax.set_aspect("equal")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
ax.set_title(f"Surface + C ({len(anchor_C)}) + O ({len(anchor_O)}) anchor nodes")
plt.tight_layout()
plt.show()

print("\nPipeline complete.")

