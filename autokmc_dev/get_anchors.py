# %% [markdown]
# # AutoKMC2 — Anchor‑Site Pipeline Demo
#
# Walks through every stage of the anchor‑site workflow, showing each step
# visually in **3‑D (Plotly)**:
#
# 1. **Build** a Cu(111) FCC surface slab
# 2. **Find** surface atoms via ray‑casting
# 3. **Build** the atom‑connectivity graph
# 4. **Find anchor sites** for C and O (enumerate → iso‑class → optimise)
# 5. **Inspect** the optimised representative position for each unique site
# 6. **Propagate** the representative position to every matching anchor node
#    on the graph, and visualise the full result

# %% ── 0. Imports & helpers ──────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc.structure import build_surface
from autokmc.surface   import find_surface_atoms
from autokmc.graph     import build_graph
from autokmc.find_anchors import find_anchor_sites

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str((Path(__file__).resolve().parent if "__file__" in globals() else Path("")) / _MODEL_FILE)
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

# ── Visual palette ──────────────────��───────────────────────────────────────
_ELEM_COL    = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}
_BULK_COL    = "#aaaaaa"
_SURF_COL    = "#4A90D9"
_K_COL       = {1: "#27AE60", 2: "#E67E22", 3: "#8E44AD", 4: "#E74C3C"}
_K_LABEL     = {1: "top (k=1)", 2: "bridge (k=2)", 3: "hollow (k=3)", 4: "4‑fold (k=4)"}
_ELEM_ANCHOR_COL = {"C": "#2ECC71", "O": "#E74C3C"}
_ELEM_ANCHOR_SYM = {"C": "diamond", "O": "cross"}


def _ecolor(sym: str) -> str:
    return _ELEM_COL.get(sym, "#888888")


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _atom_scatter3d(pos, colors, sizes, names, opacity=1.0, symbol="circle",
                    line_color="black", line_width=0.5,
                    name="", showlegend=True):
    """Return a Scatter3d trace for a set of atoms."""
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbol,
            opacity=opacity,
            line=dict(color=line_color, width=line_width),
        ),
        text=names,
        hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
        name=name,
        showlegend=showlegend,
    )


def _bond_lines3d(p_from, p_to, color="rgba(160,160,160,0.4)", width=1,
                  name="", showlegend=False):
    """Return a Scatter3d trace drawing lines between pairs of points.

    *p_from* and *p_to* are (N, 3) arrays.  Plotly draws lines via NaN
    separators so we interleave the pairs with NaN rows.
    """
    n = len(p_from)
    x = np.empty(n * 3); y = np.empty(n * 3); z = np.empty(n * 3)
    x[0::3] = p_from[:, 0]; y[0::3] = p_from[:, 1]; z[0::3] = p_from[:, 2]
    x[1::3] = p_to[:, 0];   y[1::3] = p_to[:, 1];   z[1::3] = p_to[:, 2]
    x[2::3] = np.nan;        y[2::3] = np.nan;        z[2::3] = np.nan
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
    )


def _layout(title, scene_equal=True):
    """Common 3‑D layout."""
    layout = dict(
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        scene=dict(
            xaxis_title="x (Å)",
            yaxis_title="y (Å)",
            zaxis_title="z (Å)",
            aspectmode="data" if scene_equal else "auto",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_white",
    )
    return layout


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
    min_slab_size=10.0,
    min_vacuum_size=12.0,
    goal_x=20.0,
    goal_y=20.0,
    n_freeze_layers=2,
    calculator=calc,
    verbose=True,
)

print(f"\nSlab built  : {len(atoms)} atoms")
print(f"Cell diag   : {np.diag(atoms.get_cell()).round(3)} Å")

pos  = atoms.get_positions()
syms = np.array(atoms.get_chemical_symbols())

fig = go.Figure()
for sym in np.unique(syms):
    mask = syms == sym
    fig.add_trace(_atom_scatter3d(
        pos[mask],
        colors=[_ecolor(sym)] * int(mask.sum()),
        sizes=6,
        names=[f"{sym} ({i})" for i in np.where(mask)[0]],
        opacity=0.85,
        name=sym,
        showlegend=True,
    ))

fig.update_layout(**_layout("Stage 1 — Built Structure"))
fig.show()

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
bulk_mask   = ~surf_mask

print(f"Method        : {surf_result.method}")
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")
print(f"Bulk atoms    : {bulk_mask.sum()}")

fig = go.Figure()

# Bulk atoms (small, grey, semi-transparent)
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.35, name="bulk", showlegend=True,
    ))

# Surface atoms (larger, blue)
if surf_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[surf_mask], colors=_SURF_COL,
        sizes=8, names=[f"surface {i}" for i in surf_idx],
        opacity=0.95, name="surface", showlegend=True,
    ))

fig.update_layout(**_layout("Stage 2 — Surface Atom Identification"))
fig.show()

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

# Collect surface-subgraph bond endpoints
surf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]
Gsub = G.subgraph(surf_nodes)
bond_from, bond_to = [], []
for u, v in Gsub.edges():
    bond_from.append(G.nodes[u]["position"])
    bond_to.append(G.nodes[v]["position"])

surf_pos_all = np.array([G.nodes[n]["position"] for n in surf_nodes])

fig = go.Figure()

# Bulk atom backdrop (all layers visible in 3-D)
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=3, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.2, name="bulk", showlegend=True,
    ))

# Bond lines (surface subgraph only)
if bond_from:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from), np.array(bond_to),
        color="rgba(140,140,140,0.4)", width=2,
        name="bond", showlegend=False,
    ))

# Surface atoms
fig.add_trace(_atom_scatter3d(
    surf_pos_all, colors=_SURF_COL,
    sizes=7, names=[f"surf {n}" for n in surf_nodes],
    name="surface", showlegend=True,
))

fig.update_layout(**_layout(
    f"Stage 3 — Surface Connectivity Subgraph "
    f"({len(surf_nodes)} nodes, {Gsub.number_of_edges()} edges)"
))
fig.show()

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
# ## Stage 5 — Optimised Representative Positions (3‑D)
#
# Each `AnchorSite.position` is the geometrically optimised position for the
# *representative* clique.  Sites are labelled by iso‑class index and coloured
# by **k** (coordination number).

# %% ── 5. Plot representative positions ──────────────────────────────────────
_section("STAGE 5 — Optimised representative positions")

surf_pos = pos[surf_mask]

for elem, sites in [("C", sites_C), ("O", sites_O)]:
    fig = go.Figure()

    # Full slab backdrop: bulk atoms (small, grey, transparent) + surface layer
    if bulk_mask.any():
        fig.add_trace(_atom_scatter3d(
            pos[bulk_mask], colors=_BULK_COL,
            sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
            opacity=0.2, name="bulk", showlegend=True,
        ))
    fig.add_trace(_atom_scatter3d(
        surf_pos, colors=_SURF_COL,
        sizes=6, names=["Cu surf"] * len(surf_pos),
        opacity=0.6, name="surface Cu", showlegend=True,
    ))

    # One scatter trace per k value so the legend labels are clean
    by_k: dict[int, list] = {}
    for iso in sites:
        by_k.setdefault(iso.k, []).append(iso)

    for k, iso_list in sorted(by_k.items()):
        rep_pos_k = np.array([iso.position for iso in iso_list])
        labels    = [f"{elem} cls {iso.iso_class} k={iso.k}" for iso in iso_list]
        fig.add_trace(_atom_scatter3d(
            rep_pos_k,
            colors=_K_COL.get(k, "#333333"),
            sizes=12,
            names=labels,
            symbol=_ELEM_ANCHOR_SYM.get(elem, "circle"),
            line_color="black",
            line_width=1,
            name=_K_LABEL.get(k, f"k={k}"),
            showlegend=True,
        ))

    fig.update_layout(**_layout(
        f"Stage 5 — {elem} Representative Anchor Positions "
        f"({len(sites)} iso‑classes)"
    ))
    fig.show()

# %% [markdown]
# ---
# ## Stage 6 — Propagate Representative → All Matching Anchor Nodes (3‑D)
#
# `find_anchor_sites` already Kabsch-propagated the representative to every
# member and stored the positions on the graph.  Here we read them back and
# show:
#
# * ★ gold star — optimised representative (`iso.position`)
# * coloured dots — each member anchor node position from `G`
# * thin grey lines — representative → member

# %% ── 6a. Propagation table ─────────────────────────────────────────────────
_section("STAGE 6 — Propagation table")

for elem, sites in [("C", sites_C), ("O", sites_O)]:
    print(f"\nElement: {elem}")
    print(f"  {'iso':>4}  {'k':>3}  {'members':>7}  "
          f"{'rep_pos':>32}  first_member_pos")
    print("  " + "─" * 90)
    for iso in sites:
        rep_p = iso.position
        mem_p = (G.nodes[iso.node_ids[0]]["position"]
                 if iso.node_ids else np.full(3, np.nan))
        print(f"  {iso.iso_class:>4}  {iso.k:>3}  {len(iso.members):>7}"
              f"  {np.round(rep_p, 3)}  {np.round(mem_p, 3)}")

# %% ── 6b. Per‑element propagation 3‑D plot ───────────────────────────────────
for elem, sites in [("C", sites_C), ("O", sites_O)]:
    fig = go.Figure()

    # Full slab backdrop
    if bulk_mask.any():
        fig.add_trace(_atom_scatter3d(
            pos[bulk_mask], colors=_BULK_COL,
            sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
            opacity=0.2, name="bulk", showlegend=True,
        ))
    fig.add_trace(_atom_scatter3d(
        surf_pos, colors=_SURF_COL,
        sizes=6, names=["Cu surf"] * len(surf_pos),
        opacity=0.5, name="surface Cu", showlegend=True,
    ))

    for iso in sites:
        col   = _K_COL.get(iso.k, "#333333")
        rep_p = np.asarray(iso.position, dtype=float)

        # Member positions from G
        mem_positions = np.array(
            [G.nodes[nid]["position"] for nid in iso.node_ids]
        ) if iso.node_ids else np.empty((0, 3))

        # Bond lines: rep → each member
        if len(mem_positions):
            rep_broadcast = np.tile(rep_p, (len(mem_positions), 1))
            fig.add_trace(_bond_lines3d(
                rep_broadcast, mem_positions,
                color="rgba(180,180,180,0.5)", width=1,
                showlegend=False,
            ))

            # Member dots
            fig.add_trace(_atom_scatter3d(
                mem_positions, colors=col,
                sizes=6,
                names=[f"{elem} cls{iso.iso_class} k={iso.k} mem {i}"
                       for i in range(len(mem_positions))],
                opacity=0.85,
                symbol="circle",
                line_color="black", line_width=0.5,
                name=f"{_K_LABEL.get(iso.k, f'k={iso.k}')} cls{iso.iso_class}",
                showlegend=True,
            ))

        # Representative — gold diamond
        fig.add_trace(go.Scatter3d(
            x=[rep_p[0]], y=[rep_p[1]], z=[rep_p[2]],
            mode="markers",
            marker=dict(size=14, color="gold", symbol="diamond",
                        line=dict(color="black", width=1)),
            text=[f"{elem} cls{iso.iso_class} rep"],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            name=f"rep cls{iso.iso_class}",
            showlegend=True,
        ))

    fig.update_layout(**_layout(
        f"Stage 6 — {elem} Propagation: representative ★ vs all members ●"
    ))
    fig.show()

# %% [markdown]
# ---
# ## Final Summary — Full Graph with All C & O Anchor Nodes (3‑D)

# %% ── Final: full graph with anchor nodes ───────────────────────────────────
_section("FINAL — Surface graph + all C / O anchor nodes")

anchor_C = [(n, G.nodes[n]) for n, d in G.nodes(data=True)
            if d["type"] == "anchor" and d["element"] == "C"]
anchor_O = [(n, G.nodes[n]) for n, d in G.nodes(data=True)
            if d["type"] == "anchor" and d["element"] == "O"]

print(f"Anchor nodes in G :  C → {len(anchor_C)},  O → {len(anchor_O)}")

fig = go.Figure()

# Full slab backdrop
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.2, name="bulk", showlegend=True,
    ))

# Surface atoms
fig.add_trace(_atom_scatter3d(
    surf_pos, colors=_SURF_COL,
    sizes=6, names=["Cu"] * len(surf_pos), opacity=0.65,
    name="surface Cu", showlegend=True,
))

# Anchor-bond edges for both elements
all_anchor_ids = [n for n, _ in anchor_C + anchor_O]
bond_from_a, bond_to_a = [], []
for nid in all_anchor_ids:
    ap = G.nodes[nid]["position"]
    for nb in G.neighbors(nid):
        if G.nodes[nb]["type"] != "surface":
            continue
        bond_from_a.append(ap)
        bond_to_a.append(G.nodes[nb]["position"])

if bond_from_a:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from_a), np.array(bond_to_a),
        color="rgba(160,160,160,0.3)", width=1,
        showlegend=False,
    ))

for elem, anchor_list, sym in [("C", anchor_C, "diamond"), ("O", anchor_O, "cross")]:
    if not anchor_list:
        continue
    by_k: dict[int, list[np.ndarray]] = {}
    for _nid, d in anchor_list:
        by_k.setdefault(d["k"], []).append(np.asarray(d["position"]))
    for k, pts in sorted(by_k.items()):
        pts_arr = np.array(pts)
        fig.add_trace(_atom_scatter3d(
            pts_arr,
            colors=_K_COL.get(k, "#333"),
            sizes=9,
            names=[f"{elem} k={k}"] * len(pts_arr),
            symbol=sym,
            line_color="black", line_width=0.8,
            name=f"{elem} {_K_LABEL.get(k, f'k={k}')}",
            showlegend=True,
        ))

fig.update_layout(**_layout(
    f"Final — Surface + C ({len(anchor_C)}) + O ({len(anchor_O)}) anchor nodes"
))
fig.show()

print("\nPipeline complete.")


