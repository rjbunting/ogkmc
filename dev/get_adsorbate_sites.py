# %% [markdown]
# # AutoKMC2 — Adsorbate‑Site Pipeline Demo
#
# Walks through every stage of the multi‑atom adsorbate workflow for a **CO**
# molecule on a Cu(111) FCC slab, showing each step visually in **3‑D (Plotly)**:
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
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc.structure import build_surface, find_surface_atoms
from autokmc.core.graph import build_graph
from autokmc.species.reactant import build_reactant
from autokmc.sites.adsorbate import (
    find_adsorbate_sites,
    optimise_adsorbate_site_positions,
)

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

# ── Visual palette ───────────────────────────────────────────────────────────
_ELEM_COL  = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}
_BULK_COL  = "#aaaaaa"
_SURF_COL  = "#4A90D9"
_ADS_COL   = {"C": "#2ECC71", "O": "#E74C3C"}
_ADS_SYM   = {"C": "diamond", "O": "cross"}
# Ten distinct colours for iso‑classes (wraps with % 10)
_ISO_COLS  = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BCF60C",
    "#FABEBE", "#008080",
]


def _iso_col(iso_class: int) -> str:
    return _ISO_COLS[iso_class % len(_ISO_COLS)]


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
    """Return a Scatter3d line trace for bond pairs (NaN-separated)."""
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


def _layout(title: str, scene_equal: bool = True) -> dict:
    """Common 3‑D layout dict."""
    return dict(
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


def _add_slab_backdrop(fig: go.Figure, pos, bulk_mask, surf_pos) -> None:
    """Add bulk (transparent) + surface atom backdrop traces to *fig*."""
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


# %% [markdown]
# ---
# ## Stage 1 — Build a Cu(111) FCC Slab

# %% ── 1. Build structure ────────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition="Cu",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    lattice_constant=3.615,
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
        name=sym, showlegend=True,
    ))
fig.update_layout(**_layout("Stage 1 — Built Structure"))
fig.show()

# %% [markdown]
# ---
# ## Stage 2 — Find Surface Atoms

# %% ── 2. Find surface atoms ─────────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms")

surf_result = find_surface_atoms(atoms, tag_atoms=True)
surf_mask   = surf_result.mask
surf_idx    = surf_result.indices
bulk_mask   = ~surf_mask
surf_pos    = pos[surf_mask]

print(f"Method        : {surf_result.method}")
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")
print(f"Bulk atoms    : {bulk_mask.sum()}")

fig = go.Figure()
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.35, name="bulk", showlegend=True,
    ))
if surf_mask.any():
    fig.add_trace(_atom_scatter3d(
        surf_pos, colors=_SURF_COL,
        sizes=8, names=[f"surface {i}" for i in surf_idx],
        opacity=0.95, name="surface", showlegend=True,
    ))
fig.update_layout(**_layout("Stage 2 — Surface Atom Identification"))
fig.show()

# %% [markdown]
# ---
# ## Stage 3 — Build the Connectivity Graph

# %% ── 3. Build graph ────────────────────────────────────────────────────────
_section("STAGE 3 — Build connectivity graph")

G = build_graph(atoms)

n_surf_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "surface")
n_bulk_g = sum(1 for _, d in G.nodes(data=True) if d["type"] == "bulk")
print(f"Nodes : {G.number_of_nodes()}  ({n_surf_g} surface, {n_bulk_g} bulk)")
print(f"Edges : {G.number_of_edges()}")
print(f"PBC   : {G.graph['pbc'].tolist()}")

surf_nodes   = [n for n, d in G.nodes(data=True) if d["type"] == "surface"]
Gsub         = G.subgraph(surf_nodes)
surf_pos_all = np.array([G.nodes[n]["position"] for n in surf_nodes])
bond_from, bond_to = [], []
for u, v in Gsub.edges():
    bond_from.append(G.nodes[u]["position"])
    bond_to.append(G.nodes[v]["position"])

fig = go.Figure()
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=3, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.2, name="bulk", showlegend=True,
    ))
if bond_from:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from), np.array(bond_to),
        color="rgba(140,140,140,0.4)", width=2,
        name="bond", showlegend=False,
    ))
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
# ## Stage 4 — Build the CO Reactant
#
# `build_reactant` parses the SMILES string with RDKit, embeds a 3‑D
# conformer (ETKDGv3 + MMFF94), computes automorphism orbits and marks the
# convex‑hull‑exposed **anchor atoms** eligible to bond to the surface.

# %% ── 4. Build CO reactant ──────────────────────────────────────────────────
_section("STAGE 4 — Build CO reactant")

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

react_pos  = co.atoms.get_positions()
react_syms = co.atoms.get_chemical_symbols()

# 3‑D plot of the gas-phase molecule
fig = go.Figure()
for i, (sym, p) in enumerate(zip(react_syms, react_pos)):
    col = _ADS_COL.get(sym, "#888888")
    sym3d = _ADS_SYM.get(sym, "circle")
    is_anchor = (i in co.anchor_atoms)
    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode="markers",
        marker=dict(
            size=18 if is_anchor else 14,
            color=col,
            symbol=sym3d,
            line=dict(color="gold" if is_anchor else "black",
                      width=3 if is_anchor else 1),
        ),
        text=[f"{sym}{i}{'  ★anchor' if is_anchor else ''}"],
        hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
        name=f"{sym}{i}{'  (anchor)' if is_anchor else ''}",
        showlegend=True,
    ))
# Bond line
if len(react_pos) >= 2:
    fig.add_trace(_bond_lines3d(
        react_pos[:1], react_pos[1:2],
        color="#555555", width=6, name="C≡O bond", showlegend=True,
    ))
fig.update_layout(**_layout("Stage 4 — CO Reactant (gas phase)"))
fig.show()

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites
#
# `find_adsorbate_sites` runs the full backtracking + iso‑class pipeline:
# anchor placement → surface connectivity guard → ego isomorphism → materialise.

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

# All member placements — one trace per iso‑class per element
ads_nodes = [(n, d) for n, d in G.nodes(data=True)
             if d.get("type") == "adsorbate" and d.get("reactant") == co.smiles]

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

# Anchor-bond edges (surface ↔ bonded adsorbate atom)
bond_from_a, bond_to_a = [], []
for n, d in ads_nodes:
    if not d.get("is_bonded"):
        continue
    for nb in G.neighbors(n):
        if G.nodes[nb].get("type") != "surface":
            continue
        bond_from_a.append(d["position"])
        bond_to_a.append(G.nodes[nb]["position"])
if bond_from_a:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from_a), np.array(bond_to_a),
        color="rgba(160,160,160,0.3)", width=1,
        showlegend=False,
    ))

# Intra-molecular bonds
drawn_pairs: set = set()
for n, d in ads_nodes:
    for nb in G.neighbors(n):
        if not G.edges[n, nb].get("intra_adsorbate"):
            continue
        key = (min(n, nb), max(n, nb))
        if key in drawn_pairs:
            continue
        drawn_pairs.add(key)
        iso_cls = d["iso_class"]
        pa = np.asarray(d["position"])
        pb = np.asarray(G.nodes[nb]["position"])
        fig.add_trace(_bond_lines3d(
            pa[np.newaxis], pb[np.newaxis],
            color=_iso_col(iso_cls), width=3,
            showlegend=False,
        ))

# Adsorbate atoms, one trace per (iso_class, element, bonded?)
grp: dict[tuple, list] = {}
for n, d in ads_nodes:
    key = (d["iso_class"], d["element"], bool(d.get("is_bonded")))
    grp.setdefault(key, []).append(d["position"])

seen_legend: set = set()
for (iso_cls, sym, bonded), pts in sorted(grp.items()):
    pts_arr = np.array(pts)
    sym3d   = _ADS_SYM.get(sym, "circle")
    col     = _iso_col(iso_cls)
    legend_key = (iso_cls, sym)
    lbl = f"iso {iso_cls}  {sym} ({'bonded' if bonded else 'float'})"
    fig.add_trace(_atom_scatter3d(
        pts_arr,
        colors=col,
        sizes=8 if bonded else 5,
        names=[f"iso{iso_cls} {sym}" ] * len(pts_arr),
        opacity=0.9 if bonded else 0.5,
        symbol=sym3d,
        line_color="black" if bonded else col,
        line_width=0.8 if bonded else 0.0,
        name=lbl,
        showlegend=(legend_key not in seen_legend),
    ))
    seen_legend.add(legend_key)

fig.update_layout(**_layout(
    f"Stage 5 — All CO Placements  ({len(sites_CO)} iso‑classes, "
    f"{sum(len(s.members) for s in sites_CO)} total)"
))
fig.show()

# %% [markdown]
# ---
# ## Stage 6 — Representative Positions per Iso‑Class (3‑D)
#
# Each `AdsorbateSite.positions` (shape `(n_atoms, 3)`) holds the Cartesian
# positions for the **representative** member.  One figure per iso‑class.

# %% ── 6. Representative positions ───────────────────────────────────────────
_section("STAGE 6 — Representative positions per iso‑class")

for site in sites_CO:
    col      = _iso_col(site.iso_class)
    rep_pos  = np.asarray(site.positions, dtype=float)
    rep_syms = [react_syms[i] for i in range(site.n_atoms)]
    bonded   = [c is not None for c in site.atom_cliques]
    ks_str   = "+".join(str(len(c)) for c in site.atom_cliques if c is not None) or "none"

    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    # Ego‑subgraph surface atoms highlighted in yellow
    if site.ego_graph is not None:
        ego_surf_pos = np.array([
            G.nodes[n]["position"] for n in site.ego_graph.nodes
            if n in G and G.nodes[n].get("type") == "surface"
        ])
        if len(ego_surf_pos):
            fig.add_trace(_atom_scatter3d(
                ego_surf_pos, colors="#F5C518",
                sizes=9, names=["ego surf"] * len(ego_surf_pos),
                opacity=0.9, name="ego surf", showlegend=True,
            ))

    # Intra-molecular bond
    if site.n_atoms >= 2:
        fig.add_trace(_bond_lines3d(
            rep_pos[:1], rep_pos[1:2],
            color=col, width=5, name="C≡O", showlegend=True,
        ))

    # Adsorbate atoms
    for i, (sym, p, is_b) in enumerate(zip(rep_syms, rep_pos, bonded)):
        sym3d = _ADS_SYM.get(sym, "circle")
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers",
            marker=dict(
                size=14, color=col, symbol=sym3d,
                opacity=1.0,
                line=dict(color="black" if is_b else col, width=2 if is_b else 0),
            ),
            text=[f"{sym}{'  (bonded)' if is_b else '  (float)'}"],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            name=f"{sym} {'bonded' if is_b else 'float'}",
            showlegend=True,
        ))

    n_bonded = sum(bonded)
    fig.update_layout(**_layout(
        f"Stage 6 — iso {site.iso_class}  |  "
        f"{n_bonded}/{site.n_atoms} bonded (k={ks_str})  |  "
        f"{len(site.members)} members"
    ))
    fig.show()

# %% [markdown]
# ---
# ## Stage 7 — Optimise Representative Positions
#
# `optimise_adsorbate_site_positions` runs 6‑DOF rigid‑body L‑BFGS‑B
# (restraint + steric repulsion), then Kabsch‑propagates to every member.

# %% ── 7. Optimise positions ──────────────────────────────────────────────────
_section("STAGE 7 — Optimise representative positions")

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

# Before / after comparison, one figure per iso‑class
for site in sites_CO:
    col     = _iso_col(site.iso_class)
    pre     = pre_opt[site.iso_class]
    post    = np.asarray(site.positions, dtype=float)
    rep_sym = [react_syms[i] for i in range(site.n_atoms)]
    rms     = float(np.sqrt(np.mean(np.sum((post - pre) ** 2, axis=1))))

    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    # Before — open markers, dashed bond
    if site.n_atoms >= 2:
        fig.add_trace(_bond_lines3d(
            pre[:1], pre[1:2],
            color=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.4)",
            width=3, name="before bond", showlegend=True,
        ))
    for i, sym in enumerate(rep_sym):
        sym3d = _ADS_SYM.get(sym, "circle")
        fig.add_trace(go.Scatter3d(
            x=[pre[i, 0]], y=[pre[i, 1]], z=[pre[i, 2]],
            mode="markers",
            marker=dict(size=10, color="white", symbol=sym3d,
                        line=dict(color=col, width=2)),
            text=[f"{sym} before"],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            name=f"{sym} before",
            showlegend=True,
        ))

    # After — filled markers, solid bond
    if site.n_atoms >= 2:
        fig.add_trace(_bond_lines3d(
            post[:1], post[1:2],
            color=col, width=5, name="after bond", showlegend=True,
        ))
    for i, sym in enumerate(rep_sym):
        sym3d = _ADS_SYM.get(sym, "circle")
        fig.add_trace(go.Scatter3d(
            x=[post[i, 0]], y=[post[i, 1]], z=[post[i, 2]],
            mode="markers",
            marker=dict(size=14, color=col, symbol=sym3d,
                        line=dict(color="black", width=1)),
            text=[f"{sym} after"],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            name=f"{sym} after",
            showlegend=True,
        ))

    fig.update_layout(**_layout(
        f"Stage 7 — iso {site.iso_class}  |  ΔRMS = {rms:.3f} Å"
    ))
    fig.show()

# %% [markdown]
# ---
# ## Stage 8 — Propagation: Representative → All Members (3‑D)
#
# * ★ gold diamond — optimised representative
# * coloured circles — all member positions from *G*
# * thin grey lines — representative C → member C

# %% ── 8a. Propagation table ──────────────────────────────────────────────────
_section("STAGE 8 — Propagation: representative → all members")

print(f"\n{'iso':>4}  {'n_atoms':>7}  {'n_members':>9}  "
      f"{'rep_pos[C]':>30}  first_other_member_pos[C]")
print("─" * 90)
for site in sites_CO:
    rep_p  = site.positions[0]
    mem1_p = (G.nodes[site.member_node_ids[1][0]]["position"]
              if len(site.member_node_ids) > 1 else np.full(3, np.nan))
    print(f"  {site.iso_class:>2}   {site.n_atoms:>7}  {len(site.members):>9}  "
          f"  {np.round(rep_p, 3)}  {np.round(mem1_p, 3)}")

# %% ── 8b. Per‑iso‑class propagation 3‑D plots ───────────────────────────────
for site in sites_CO:
    col     = _iso_col(site.iso_class)
    rep_sym = list(react_syms)

    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    # All member placements from G
    for m_idx, node_ids in enumerate(site.member_node_ids):
        m_pos = np.array([G.nodes[nid]["position"] for nid in node_ids])
        is_rep = (m_idx == 0)

        # Intra-molecular bond
        if len(m_pos) >= 2:
            fig.add_trace(_bond_lines3d(
                m_pos[:1], m_pos[1:2],
                color=col if is_rep else f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.35)",
                width=4 if is_rep else 1,
                showlegend=False,
            ))

        for i, (sym, p) in enumerate(zip(rep_sym, m_pos)):
            sym3d = _ADS_SYM.get(sym, "circle")
            if is_rep:
                # rep: gold diamond
                fig.add_trace(go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2]],
                    mode="markers",
                    marker=dict(size=16, color="gold", symbol="diamond",
                                line=dict(color="black", width=1)),
                    text=[f"rep {sym} iso{site.iso_class}"],
                    hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
                    name=f"rep {sym}" if i == 0 else f"rep {sym}",
                    showlegend=(i == 0),
                ))
            else:
                fig.add_trace(_atom_scatter3d(
                    p[np.newaxis],
                    colors=col, sizes=6,
                    names=[f"mem{m_idx} {sym}"],
                    opacity=0.7,
                    symbol=sym3d,
                    line_color="black", line_width=0.5,
                    name=f"members {sym}" if (m_idx == 1 and i == 0) else f"members {sym}",
                    showlegend=(m_idx == 1 and i == 0),
                ))

    # Lines: rep C → every member C
    rep_c = np.asarray(site.positions[0])
    mem_c_list = [G.nodes[nids[0]]["position"]
                  for nids in site.member_node_ids[1:] if nids]
    if mem_c_list:
        rep_broadcast = np.tile(rep_c, (len(mem_c_list), 1))
        fig.add_trace(_bond_lines3d(
            rep_broadcast, np.array(mem_c_list),
            color="rgba(180,180,180,0.5)", width=1,
            showlegend=False,
        ))

    fig.update_layout(**_layout(
        f"Stage 8 — iso {site.iso_class}  |  "
        f"{len(site.members)} members  ★ = rep"
    ))
    fig.show()

# %% [markdown]
# ---
# ## Final Summary — Full Graph with All CO Adsorbate Nodes (3‑D)

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

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

# Anchor‑bond edges
bond_from_f, bond_to_f = [], []
for n, d in ads_nodes_final:
    if not d.get("is_bonded"):
        continue
    for nb in G.neighbors(n):
        if G.nodes[nb].get("type") != "surface":
            continue
        bond_from_f.append(d["position"])
        bond_to_f.append(G.nodes[nb]["position"])
if bond_from_f:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from_f), np.array(bond_to_f),
        color="rgba(150,150,150,0.3)", width=1, showlegend=False,
    ))

# Intra‑molecular bonds
drawn_pairs = set()
for n, d in ads_nodes_final:
    for nb in G.neighbors(n):
        if not G.edges[n, nb].get("intra_adsorbate"):
            continue
        key = (min(n, nb), max(n, nb))
        if key in drawn_pairs:
            continue
        drawn_pairs.add(key)
        iso_cls = d["iso_class"]
        pa = np.asarray(d["position"])
        pb = np.asarray(G.nodes[nb]["position"])
        fig.add_trace(_bond_lines3d(
            pa[np.newaxis], pb[np.newaxis],
            color=_iso_col(iso_cls), width=3, showlegend=False,
        ))

# Adsorbate atoms — one trace per (iso_class, element, bonded?)
grp_f: dict[tuple, list] = {}
for n, d in ads_nodes_final:
    key = (d["iso_class"], d["element"], bool(d.get("is_bonded")))
    grp_f.setdefault(key, []).append(d["position"])

seen_f: set = set()
for (iso_cls, sym, bonded), pts in sorted(grp_f.items()):
    pts_arr = np.array(pts)
    sym3d   = _ADS_SYM.get(sym, "circle")
    col     = _iso_col(iso_cls)
    lk      = (iso_cls, sym)
    lbl     = f"iso {iso_cls}  {sym} ({'bonded' if bonded else 'float'})"
    fig.add_trace(_atom_scatter3d(
        pts_arr,
        colors=col,
        sizes=9 if bonded else 5,
        names=[f"iso{iso_cls} {sym}"] * len(pts_arr),
        opacity=0.95 if bonded else 0.45,
        symbol=sym3d,
        line_color="black" if bonded else col,
        line_width=0.8 if bonded else 0.0,
        name=lbl,
        showlegend=(lk not in seen_f),
    ))
    seen_f.add(lk)

fig.update_layout(**_layout(
    f"Final — Surface + {total_ads} CO adsorbate nodes  "
    f"({len(sites_CO)} iso‑classes)"
))
fig.show()

print("\nPipeline complete.")

