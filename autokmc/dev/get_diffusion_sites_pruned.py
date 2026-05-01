# %% [markdown]
# # AutoKMC — Diffusion-Site Pipeline (Debug)
#
# Mirrors `get_adsorbate_sites_pruned.py` but continues into the diffusion
# enumeration step so you can inspect exactly which hop-pair iso-classes are
# produced and why.
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Build Cu(111) FCC slab |
# | 2 | Find surface atoms |
# | 3 | Build connectivity graph |
# | 4 | Build CO reactant (gas-phase relaxation) |
# | 5 | `find_adsorbate_sites(..., prune_stable_only=True)` — ML-pruned iso-classes |
# | 6 | `find_diffusion_sites` — enumerate hop-pair iso-classes |
# | 7 | Summary table: all DiffusionSite iso-classes |
# | 8 | 3-D overview: both endpoints of every diff_iso on the slab |
# | 9 | Ego-graph subplots (2-D, one panel per diff_iso) |
# | 10 | Per-diff-iso 3-D: endpoint A vs endpoint B placements |

# %% ── 0. Imports & helpers ──────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nequip.ase import NequIPCalculator

from autokmc.structure          import build_surface
from autokmc.surface            import find_surface_atoms
from autokmc.graph              import build_graph
from autokmc.reactants          import build_reactant
from autokmc.find_adsorbate_sites import find_adsorbate_sites
from autokmc.find_diffusion_sites import find_diffusion_sites

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

# ── Visual palette (matches get_adsorbate_sites_pruned.py) ───────────────────
_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_COL  = {"C": "#2ECC71", "O": "#E74C3C"}
_ADS_SYM  = {"C": "diamond", "O": "cross"}
_ISO_COLS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BCF60C",
    "#FABEBE", "#008080",
]
# Extra palette for diffusion iso-classes (contrasts clearly with ads palette)
_DIFF_COLS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#C77DFF", "#FF9F1C", "#2EC4B6", "#E71D36",
    "#011627", "#FDFFFC",
]


def _ads_col(iso_class: int) -> str:
    return _ISO_COLS[iso_class % len(_ISO_COLS)]


def _diff_col(diff_iso: int) -> str:
    return _DIFF_COLS[diff_iso % len(_DIFF_COLS)]


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
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0),
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

# %% ── 4. Build CO reactant ───────────────────────────────────────────────────
_section("STAGE 4 — Build CO reactant  (gas-phase relaxation for Egas reference)")

co = build_reactant("[C]=O", add_hydrogens=False, calculator=calc)
react_syms = co.atoms.get_chemical_symbols()

print(f"Formula      : {co.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {co.anchor_atoms}  "
      f"(elements: {[react_syms[i] for i in co.anchor_atoms]})")
print(f"Gas-phase E  : {co.energy:.4f} eV")

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites WITH Stability Pruning
#
# This is identical to `get_adsorbate_sites_pruned.py`.  The output is a list
# of stable :class:`~autokmc.find_adsorbate_sites.AdsorbateSite` iso-classes
# whose representative positions have been ML-refined.  These are the inputs
# to `find_diffusion_sites`.

# %% ── 5. Find adsorbate sites (with pruning) ─────────────────────────────────
_section("STAGE 5 — Find adsorbate sites [prune_stable_only=True]")

adsorbate_sites = find_adsorbate_sites(
    G, co,
    prune_stable_only = True,
    calculator        = calc,
    frozen_indices    = frozen_indices,
    prune_fmax        = 0.05,
    prune_max_steps   = 500,
    verbose           = True,
)

print(f"\nCO: {len(adsorbate_sites)} stable iso-class(es) after pruning")
for site in adsorbate_sites:
    bonded = [(i, c) for i, c in enumerate(site.atom_cliques) if c is not None]
    bstr   = "  ".join(f"{react_syms[i]}→k={len(c)}" for i, c in bonded) \
             if bonded else "all-floating"
    print(f"  ads_iso {site.iso_class:2d}  {bstr:<30s}"
          f"  members={len(site.members):<4d}")

# %% [markdown]
# ---
# ## Stage 6 — Enumerate Diffusion (Hop-Pair) Iso-Classes
#
# `find_diffusion_sites` pairs up adsorbate placements that:
# 1. Share the same SMILES.
# 2. Have bonded surface cliques within ``max_hops`` hops of each other in the
#    surface graph (``max_hops=0`` means "share at least one surface atom").
#
# Pairs are deduplicated into iso-classes by graph-isomorphism of the union
# ego-graph (the surface-only ``n_shells_pair``-shell BFS around the union of
# both endpoints' bonded cliques with the two placements included as labelled
# occupied leaves).

# %% ── 6. Find diffusion sites ────────────────────────────────────────────────
_section("STAGE 6 — find_diffusion_sites  [max_hops=0, n_shells_pair=1]")

diff_by_smiles = find_diffusion_sites(
    G, adsorbate_sites,
    max_hops                 = 0,    # 0 = "share at least one surface atom"
    n_shells_pair            = 1,    # ego-graph depth for iso-class matching
    prune_by_adsorption_pair = True, # keep only smallest-ego site per ads-iso pair
    verbose                  = True,
)

diff_sites_flat = [ds for dss in diff_by_smiles.values() for ds in dss]
total_diff_members = sum(len(ds.members) for ds in diff_sites_flat)

print(f"\nDiffusion iso-classes  : {len(diff_sites_flat)}")
print(f"Total hop-pair members : {total_diff_members}")

if not diff_sites_flat:
    print("\n  ⚠  No diffusion iso-classes found!")
    print("  Possible causes:")
    print("    • max_hops=0 but no pair shares a surface atom — try max_hops=1")
    print("    • only 1 adsorbate iso-class (need ≥2 nearby placements to hop between)")
    print("    • slab too small so every placement is the same iso-class")

# %% ── 7. Summary table ───────────────────────────────────────────────────────
_section("STAGE 7 — Diffusion iso-class summary table")

print(f"\n{'diff_iso':>8}  {'ads_isoA':>8}  {'ads_isoB':>8}  "
      f"{'members':>7}  {'ego_nodes':>9}  {'ego_edges':>9}  n_shells")
print("─" * 72)
for ds in diff_sites_flat:
    site_a0, m_a0, site_b0, m_b0 = ds.members[0]
    g = ds.ego_graph
    n_nodes = g.number_of_nodes() if g is not None else -1
    n_edges = g.number_of_edges() if g is not None else -1
    print(f"  {ds.iso_class:>6}  {site_a0.iso_class:>8}  {site_b0.iso_class:>8}  "
          f"{len(ds.members):>7}  {n_nodes:>9}  {n_edges:>9}  "
          f"{ds.n_shells_pair_settled}")

# %% [markdown]
# ---
# ## Stage 8 — 3-D Overview: All Hop-Pair Endpoint Positions
#
# Shows the full slab backdrop with every materialised endpoint A (circle)
# and endpoint B (diamond) for every diffusion iso-class.  Each diff_iso gets
# its own colour.  The endpoint positions are taken directly from the live
# adsorbate nodes on G (i.e. the ML-refined positions set by Stage 5).

# %% ── 8. 3-D overview ─────────────────────────────────────────────────────────
_section("STAGE 8 — 3-D overview: all endpoint positions")

fig_all = go.Figure()
_add_slab_backdrop(fig_all, pos, bulk_mask, surf_pos)

legend_added: set = set()

for ds in diff_sites_flat:
    col = _diff_col(ds.iso_class)
    a_nids_list, b_nids_list = [], []

    for m_idx, (site_a, _, site_b, _) in enumerate(ds.members):
        a_nids, b_nids = ds.member_node_ids[m_idx]
        a_nids_list.extend(a_nids)
        b_nids_list.extend(b_nids)

    def _pos_from_nids(nids):
        pts = []
        for nid in nids:
            if nid in G:
                p = G.nodes[nid].get("position")
                if p is not None:
                    pts.append(np.asarray(p, dtype=float))
        return pts

    pts_a = _pos_from_nids(a_nids_list)
    pts_b = _pos_from_nids(b_nids_list)

    # Endpoint A — circles
    if pts_a:
        lg_a = f"diff_iso {ds.iso_class} A"
        fig_all.add_trace(_atom_scatter3d(
            np.array(pts_a), colors=col, sizes=10,
            names=[f"diff_iso {ds.iso_class} A"] * len(pts_a),
            opacity=0.85, symbol="circle",
            line_color="black", line_width=0.6,
            name=lg_a, showlegend=(lg_a not in legend_added),
        ))
        legend_added.add(lg_a)

    # Endpoint B — diamonds
    if pts_b:
        lg_b = f"diff_iso {ds.iso_class} B"
        fig_all.add_trace(_atom_scatter3d(
            np.array(pts_b), colors=col, sizes=10,
            names=[f"diff_iso {ds.iso_class} B"] * len(pts_b),
            opacity=0.85, symbol="diamond",
            line_color="white", line_width=0.6,
            name=lg_b, showlegend=(lg_b not in legend_added),
        ))
        legend_added.add(lg_b)

fig_all.update_layout(**_layout3d(
    f"Stage 8 — all hop-pair endpoints  "
    f"({len(diff_sites_flat)} diff_iso, {total_diff_members} members)  "
    f"● = A, ◆ = B"
))
fig_all.show()

# %% [markdown]
# ---
# ## Stage 9 — Ego-Graph Subplots (2-D)
#
# One panel per diffusion iso-class.  Shows the surface atoms (grey circles),
# endpoint adsorbate nodes (coloured diamonds), and the three edge flavours:
#
# | Line style     | Meaning |
# |---|---|
# | solid grey     | surface–surface bond |
# | solid blue     | anchor bond (adsorbate–surface) |
# | dotted green   | intra-adsorbate bond |
#
# The two endpoint placements are labelled ``endpoint`` in `endpoint_role` to
# distinguish them from bystander adsorbate neighbours.

# %% ── 9. Ego-graph subplots ──────────────────────────────────────────────────
_section("STAGE 9 — Ego-graph subplots  (2-D, one panel per diff_iso)")

_n    = len(diff_sites_flat)
_cols = min(3, max(1, _n))
_rows = (_n + _cols - 1) // _cols

fig_ego = make_subplots(
    rows=_rows, cols=_cols,
    subplot_titles=[
        (f"diff_iso {ds.iso_class}  "
         f"(ads {ds.members[0][0].iso_class}↔{ds.members[0][2].iso_class}, "
         f"mem={len(ds.members)})")
        for ds in diff_sites_flat
    ],
    horizontal_spacing=0.04, vertical_spacing=0.10,
)

_edge_styles = {
    "surface":          dict(color="#888888", width=1.0, dash="solid"),
    "anchor_bond":      dict(color="#1F77B4", width=2.0, dash="solid"),
    "intra_adsorbate":  dict(color="#2CA02C", width=2.0, dash="dot"),
}


def _node_xy(g, nid):
    p = g.nodes[nid].get("position")
    if p is None:
        return 0.0, 0.0
    return float(p[0]), float(p[1])


for _idx, ds in enumerate(diff_sites_flat):
    g = ds.ego_graph
    _r = _idx // _cols + 1
    _c = _idx %  _cols + 1
    _first = (_idx == 0)  # legend only on first panel

    if g is None:
        continue

    seg_x = {k: [] for k in _edge_styles}
    seg_y = {k: [] for k in _edge_styles}

    for u, v, ed in g.edges(data=True):
        kind = (
            "anchor_bond"     if ed.get("anchor_bond")     else
            "intra_adsorbate" if ed.get("intra_adsorbate") else
            "surface"
        )
        x0, y0 = _node_xy(g, u)
        x1, y1 = _node_xy(g, v)
        # Skip wrap-around MIC edges (cosmetic only)
        if abs(x1 - x0) > 8.0 or abs(y1 - y0) > 8.0:
            continue
        seg_x[kind] += [x0, x1, None]
        seg_y[kind] += [y0, y1, None]

    for kind, style in _edge_styles.items():
        if seg_x[kind]:
            fig_ego.add_trace(go.Scatter(
                x=seg_x[kind], y=seg_y[kind], mode="lines",
                line=style, hoverinfo="skip",
                showlegend=_first, name=kind,
            ), row=_r, col=_c)

    surf_x, surf_y, surf_txt = [], [], []
    ep_x,   ep_y,   ep_txt   = [], [], []
    ep_col_list = []

    for n, d in g.nodes(data=True):
        x, y = _node_xy(g, n)
        if d.get("type") == "surface":
            surf_x.append(x); surf_y.append(y)
            surf_txt.append(f"surf #{n} {d.get('element','?')}")
        else:
            ep_x.append(x); ep_y.append(y)
            ep_txt.append(
                f"{d.get('element','?')} #{n}  "
                f"iso={d.get('iso_class','?')}  "
                f"role={d.get('endpoint_role','—')}  "
                f"occ={d.get('occupied', False)}"
            )
            ep_col_list.append(
                "#E74C3C" if d.get("element") == "C" else "#F1C40F"
            )

    fig_ego.add_trace(go.Scatter(
        x=surf_x, y=surf_y, mode="markers",
        marker=dict(size=10, color="#BDC3C7", line=dict(width=0.5, color="#555")),
        text=surf_txt, hoverinfo="text",
        showlegend=_first, name="surface",
    ), row=_r, col=_c)

    if ep_x:
        fig_ego.add_trace(go.Scatter(
            x=ep_x, y=ep_y, mode="markers",
            marker=dict(size=14, color=ep_col_list, symbol="diamond",
                        line=dict(width=1, color="black")),
            text=ep_txt, hoverinfo="text",
            showlegend=_first, name="endpoint",
        ), row=_r, col=_c)

    fig_ego.update_xaxes(
        scaleanchor=f"y{_idx + 1 if _idx else ''}",
        scaleratio=1.0, row=_r, col=_c, showgrid=False, zeroline=False, visible=False,
    )
    fig_ego.update_yaxes(showgrid=False, zeroline=False, visible=False, row=_r, col=_c)

fig_ego.update_layout(
    title=f"Stage 9 — Diffusion ego-graphs ({_n} diff iso-class(es))",
    height=320 * max(1, _rows), width=420 * _cols,
    template="plotly_white",
    margin=dict(l=20, r=20, t=70, b=20),
)
fig_ego.show()

# %% [markdown]
# ---
# ## Stage 10 — Per-Diff-Iso 3-D Plots
#
# One figure per diffusion iso-class.  The **representative** hop pair is
# highlighted (gold star for endpoint A, silver star for endpoint B); all
# other member pairs are shown more faintly.  Lines connect the A and B
# positions within each member pair so you can see the hop direction.

# %% ── 10. Per-diff-iso 3-D plots ─────────────────────────────────────────────
_section("STAGE 10 — Per-diff-iso 3-D: endpoint A ↔ endpoint B for all members")

for ds in diff_sites_flat:
    col = _diff_col(ds.iso_class)

    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    for m_idx, (site_a, ma, site_b, mb) in enumerate(ds.members):
        a_nids, b_nids = ds.member_node_ids[m_idx]
        is_rep = (m_idx == 0)

        def _nid_pos(nids):
            """Mean Cartesian position of a list of graph node ids."""
            pts = [G.nodes[nid]["position"] for nid in nids if nid in G]
            return np.mean(pts, axis=0) if pts else np.zeros(3)

        pa = _nid_pos(a_nids)
        pb = _nid_pos(b_nids)

        hop_alpha = 0.9 if is_rep else 0.35
        hop_width = 4   if is_rep else 1

        # Hop arrow A → B (NaN-broken line segment)
        fig.add_trace(_bond_lines3d(
            pa[np.newaxis], pb[np.newaxis],
            color=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},{hop_alpha})",
            width=hop_width, showlegend=False,
        ))

        # Full adsorbate node cloud for A
        pts_a = np.array([G.nodes[nid]["position"] for nid in a_nids if nid in G])
        pts_b = np.array([G.nodes[nid]["position"] for nid in b_nids if nid in G])
        elem_a = [G.nodes[nid].get("element", "?") for nid in a_nids if nid in G]
        elem_b = [G.nodes[nid].get("element", "?") for nid in b_nids if nid in G]

        if len(pts_a):
            fig.add_trace(_atom_scatter3d(
                pts_a,
                colors="#FFD700" if is_rep else col,
                sizes=14 if is_rep else 7,
                names=[f"A m={m_idx} ({e})" for e in elem_a],
                opacity=0.95 if is_rep else 0.45,
                symbol="circle",
                line_color="black" if is_rep else col,
                line_width=1.5 if is_rep else 0.0,
                name="endpoint A (rep)" if is_rep else "endpoint A",
                showlegend=(m_idx == 0),
            ))

        if len(pts_b):
            fig.add_trace(_atom_scatter3d(
                pts_b,
                colors="#C0C0C0" if is_rep else col,
                sizes=14 if is_rep else 7,
                names=[f"B m={m_idx} ({e})" for e in elem_b],
                opacity=0.95 if is_rep else 0.45,
                symbol="diamond",
                line_color="black" if is_rep else col,
                line_width=1.5 if is_rep else 0.0,
                name="endpoint B (rep)" if is_rep else "endpoint B",
                showlegend=(m_idx == 0),
            ))

    site_a0, ma0, site_b0, mb0 = ds.members[0]
    g = ds.ego_graph
    fig.update_layout(**_layout3d(
        f"diff_iso {ds.iso_class}  |  "
        f"ads_iso {site_a0.iso_class} ↔ {site_b0.iso_class}  |  "
        f"{len(ds.members)} member(s)  "
        f"ego(n={g.number_of_nodes() if g else '—'}, "
        f"e={g.number_of_edges() if g else '—'})  |  "
        f"★gold = A rep, ★silver = B rep"
    ))
    fig.show()

# %% ── Final summary ───────────────────────────────────────────────────────────
_section("FINAL — Pipeline summary")

print(f"Adsorbate iso-classes   : {len(adsorbate_sites)}")
print(f"  {'iso':>4}  {'members':>7}")
for site in adsorbate_sites:
    print(f"  {site.iso_class:>4}  {len(site.members):>7}")

print()
print(f"Diffusion iso-classes   : {len(diff_sites_flat)}")
print(f"  {'diff_iso':>8}  {'ads A':>5}  {'ads B':>5}  {'members':>7}")
for ds in diff_sites_flat:
    a0, _, b0, _ = ds.members[0]
    print(f"  {ds.iso_class:>8}  {a0.iso_class:>5}  {b0.iso_class:>5}  "
          f"{len(ds.members):>7}")

print("\nDone.  Inspect figures above to check the enumerated hop types.")

