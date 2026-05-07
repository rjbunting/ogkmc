# %% [markdown]
# # AutoKMC — Adsorbate-Site Pipeline with ML Stability Pruning
#
# Demonstrates the **full pruned pipeline** for CO on Cu(111).
# When `find_adsorbate_sites` is called with `prune_stable_only=True` and a
# `calculator`, it executes the following sub-steps internally — in order:
#
# | Sub-step | What happens |
# |---|---|
# | **A** | Enumerate all raw placements → reduce to iso-classes → materialise nodes on G |
# | **B-1** | Calc-free rigid-body L-BFGS-B optimisation of every representative; Kabsch-propagates to all members |
# | **B-2** | Per iso-class: ML-relax, check bond topology; **prune** unstable iso-classes; **update** `ms.positions` from the ML-relaxed geometry; **propagate** to all members |
#
# Pipeline steps
# --------------
# 1. Build Cu(111) slab
# 2. Find surface atoms
# 3. Build connectivity graph
# 4. Build CO reactant
# 5. `find_adsorbate_sites(..., calculator=calc)` — all sub-steps above run inside
# 6. Visualise final stable sites (positions are already ML-refined)
#
# You can also call `prune_unstable_adsorbate_sites` as a standalone step
# *after* `find_adsorbate_sites(..., prune_stable_only=False)` if you want
# more control (shown in the "standalone pruning" section at the bottom).

# %% ── 0. Imports & helpers ──────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc.structure import build_surface, find_surface_atoms
from autokmc.core.graph import build_graph
from autokmc.species.reactant import build_reactant
from autokmc.sites.adsorbate import find_adsorbate_sites

_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path(""))
    / _MODEL_FILE
)
print(f"Device : {_DEVICE}")
print(f"Model  : {_MODEL_FILE}")


def make_calc():
    """Return a fresh NequIPCalculator instance loaded from the model."""
    return NequIPCalculator.from_compiled_model(
        compile_path=str(_MODEL_PATH),
        device=_DEVICE,
    )


calc = make_calc()
print(f"Calculator : {calc.__class__.__name__}")

# ── Visual palette ────────────────────────────────────────────────────────────
_ELEM_COL = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}
_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_COL  = {"C": "#2ECC71", "O": "#E74C3C"}
_ADS_SYM  = {"C": "diamond", "O": "cross"}
_ISO_COLS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BCF60C",
    "#FABEBE", "#008080",
]


def _iso_col(iso_class: int) -> str:
    return _ISO_COLS[iso_class % len(_ISO_COLS)]


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _atom_scatter3d(pos, colors, sizes, names, opacity=1.0, symbol="circle",
                    line_color="black", line_width=0.5,
                    name="", showlegend=True):
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


def _layout(title: str) -> dict:
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


def _site_graph_nodes(G, site):
    rows = []
    for m_idx, node_ids in enumerate(site.member_node_ids):
        for atom_i, nid in enumerate(node_ids):
            if nid not in G:
                continue
            d = G.nodes[nid]
            if d.get("type") == "adsorbate":
                rows.append((m_idx, atom_i, nid, d))
    return rows


def _add_site_member_traces(fig, G, site, visible=True):
    rows = _site_graph_nodes(G, site)
    if not rows:
        return

    col = _iso_col(site.iso_class)
    node_set = {nid for _, _, nid, _ in rows}
    legend_group = f"iso {site.iso_class}"

    # Anchor-bond edges for this iso-class only.
    bond_from, bond_to = [], []
    for _, _, nid, d in rows:
        if not d.get("is_bonded"):
            continue
        for nb in G.neighbors(nid):
            if G.nodes[nb].get("type") != "surface":
                continue
            bond_from.append(d["position"])
            bond_to.append(G.nodes[nb]["position"])
    if bond_from:
        tr = _bond_lines3d(
            np.array(bond_from), np.array(bond_to),
            color="rgba(90,90,90,0.45)", width=2,
            name=f"iso {site.iso_class} anchor bonds",
        )
        tr.visible = visible
        tr.legendgroup = legend_group
        fig.add_trace(tr)

    # Intra-molecular bonds for this iso-class only.
    drawn_pairs: set = set()
    mol_from, mol_to = [], []
    for _, _, nid, d in rows:
        for nb in G.neighbors(nid):
            if nb not in node_set:
                continue
            if not G.edges[nid, nb].get("intra_adsorbate"):
                continue
            key = (min(nid, nb), max(nid, nb))
            if key in drawn_pairs:
                continue
            drawn_pairs.add(key)
            mol_from.append(d["position"])
            mol_to.append(G.nodes[nb]["position"])
    if mol_from:
        tr = _bond_lines3d(
            np.array(mol_from), np.array(mol_to),
            color=col, width=4,
            name=f"iso {site.iso_class} C-O",
        )
        tr.visible = visible
        tr.legendgroup = legend_group
        fig.add_trace(tr)

    grp: dict = {}
    for _m_idx, _atom_i, _nid, d in rows:
        key = (d["element"], bool(d.get("is_bonded")))
        grp.setdefault(key, []).append(d["position"])

    for (sym, bonded), pts in sorted(grp.items()):
        tr = _atom_scatter3d(
            np.array(pts),
            colors=col,
            sizes=10 if bonded else 6,
            names=[f"iso{site.iso_class} {sym}"] * len(pts),
            opacity=0.98 if bonded else 0.65,
            symbol=_ADS_SYM.get(sym, "circle"),
            line_color="black" if bonded else col,
            line_width=1.0 if bonded else 0.0,
            name=f"iso {site.iso_class} {sym} "
                 f"({'bonded' if bonded else 'float'})",
            showlegend=True,
        )
        tr.visible = visible
        tr.legendgroup = legend_group
        fig.add_trace(tr)


# %% ── 1. Build Cu(111) slab ──────────────────────────────────────────────────
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

print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")

# %% ── 3. Build graph ─────────────────────────────────────────────────────────
_section("STAGE 3 — Build connectivity graph")

G = build_graph(atoms)

print(f"Nodes : {G.number_of_nodes()}  Edges : {G.number_of_edges()}")
print(f"PBC   : {G.graph['pbc'].tolist()}")

# %% ── 4. Build CO reactant ───────────────────────────────────────────────────
_section("STAGE 4 — Build CO reactant")

co = build_reactant("[C-]#[O+]", add_hydrogens=False)
react_syms = co.atoms.get_chemical_symbols()

print(f"Formula      : {co.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {co.anchor_atoms}  "
      f"(elements: {[react_syms[i] for i in co.anchor_atoms]})")

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites WITH Stability Pruning
#
# Key parameters on `find_adsorbate_sites`:
#
# | parameter          | default | meaning |
# |--------------------|---------|---------|
# | `prune_stable_only`| `True`  | enable the full B-1 → B-2 pruning pipeline |
# | `calculator`       | `None`  | ML potential: required to activate pruning |
# | `frozen_indices`   | `None`  | frozen slab atoms during ML relaxation |
# | `prune_fmax`       | `0.05`  | force convergence threshold (eV/Å) |
# | `prune_max_steps`  | `200`   | max LBFGS steps |
#
# With `prune_stable_only=True` and a valid calculator the call automatically:
# 1. Materialises all enumerated placements onto G.
# 2. (**B-1**) Runs calc-free rigid-body optimisation and propagates to members.
# 3. (**B-2**) ML-relaxes each representative, prunes unstable iso-classes,
#    updates `ms.positions` from the relaxed Atoms, and Kabsch-propagates to
#    all members.
#
# After the call, all node positions on G are **ML-refined**.
# There is no need to call `optimise_adsorbate_site_positions` separately.

# %% ── 5. Find adsorbate sites (with pruning) ─────────────────────────────────
_section("STAGE 5 — Find adsorbate sites for CO  [pruning ON]")

sites_CO = find_adsorbate_sites(
    G, co,
    prune_stable_only = True,        # ← new: prune unstable iso-classes
    calculator        = calc,        # ← required for pruning
    frozen_indices    = frozen_indices,
    prune_fmax        = 0.05,
    prune_max_steps   = 200,
    verbose           = True,
)

print(f"\nCO: {len(sites_CO)} stable unique iso-class(es) after pruning")
for site in sites_CO:
    bonded = [(i, c) for i, c in enumerate(site.atom_cliques) if c is not None]
    bonded_str = "  ".join(
        f"{react_syms[i]}→k={len(c)}" for i, c in bonded
    ) if bonded else "all-floating"
    print(f"  iso {site.iso_class:2d}  {bonded_str:<30s}"
          f"  members={len(site.members):<4d}"
          f"  rep_pos[C]={np.round(site.positions[0], 2)}")

# Plot: stable ISO-classes, representative placement only
fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

for site in sites_CO:
    col     = _iso_col(site.iso_class)
    rep_pos = np.asarray(site.positions, dtype=float)

    if site.n_atoms >= 2:
        fig.add_trace(_bond_lines3d(
            rep_pos[:1], rep_pos[1:2],
            color=col, width=5, showlegend=False,
        ))
    for i, sym in enumerate(react_syms):
        is_b  = site.atom_cliques[i] is not None
        sym3d = _ADS_SYM.get(sym, "circle")
        fig.add_trace(go.Scatter3d(
            x=[rep_pos[i, 0]], y=[rep_pos[i, 1]], z=[rep_pos[i, 2]],
            mode="markers",
            marker=dict(size=12, color=col, symbol=sym3d, opacity=0.95,
                        line=dict(color="black" if is_b else col,
                                  width=2 if is_b else 0)),
            text=[f"iso {site.iso_class} {sym}"],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            name=f"iso {site.iso_class} {sym}",
            showlegend=(i == 0),
        ))

fig.update_layout(**_layout(
    f"Stage 5 — {len(sites_CO)} stable CO iso-classes (representative positions)"
))
fig.show()

# %% [markdown]
# ---
# ## Stage 6 — Visualise stable sites
#
# All positions on G are already ML-refined.  No further optimisation step
# is needed.

# %% ── 6. Final visualisation ─────────────────────────────────────────────────
_section("STAGE 6 / FINAL — Stable CO adsorbate sites (all members, ML-refined)")

ads_nodes = [(n, d) for n, d in G.nodes(data=True)
             if d.get("type") == "adsorbate" and d.get("reactant") == co.smiles]

total_ads = len(ads_nodes)
n_bonded  = sum(1 for _, d in ads_nodes if d.get("is_bonded"))
print(f"Adsorbate nodes in G : {total_ads}  ({n_bonded} bonded)")
print(f"Stable iso-classes   : {len(sites_CO)}")
print(f"Total placements     : {sum(len(s.members) for s in sites_CO)}")

print("\nPer-iso graph position audit:")
for site in sites_CO:
    rows = _site_graph_nodes(G, site)
    expected = len(site.member_node_ids) * site.n_atoms
    present = len(rows)
    optimised = sum(1 for row in rows if row[3].get("optimised"))
    missing_members = [
        m_idx
        for m_idx, node_ids in enumerate(site.member_node_ids)
        if any(nid not in G for nid in node_ids)
    ]
    print(
        f"  iso {site.iso_class:2d}: members={len(site.member_node_ids):4d}  "
        f"nodes={present:4d}/{expected:<4d}  "
        f"optimised={optimised:4d}/{present:<4d}  "
        f"missing_members={len(missing_members)}"
    )

# The combined all-iso view below is dense: several stable iso-classes can sit
# close enough that Plotly depth sorting or later traces visually cover earlier
# ones.  This dropdown renders one iso-class at a time from the same graph-node
# positions used by KMC.
if sites_CO:
    first_iso = sites_CO[0].iso_class
    fig_iso = go.Figure()
    _add_slab_backdrop(fig_iso, pos, bulk_mask, surf_pos)
    backdrop_n = len(fig_iso.data)
    iso_trace_ids: dict[int, list[int]] = {}

    for site in sites_CO:
        before = len(fig_iso.data)
        _add_site_member_traces(
            fig_iso, G, site,
            visible=(site.iso_class == first_iso),
        )
        iso_trace_ids[site.iso_class] = list(range(before, len(fig_iso.data)))

    buttons = []
    for site in sites_CO:
        iso_cls = int(site.iso_class)
        visible = [
            (i < backdrop_n) or (i in iso_trace_ids[iso_cls])
            for i in range(len(fig_iso.data))
        ]
        buttons.append(dict(
            label=f"iso {iso_cls}",
            method="update",
            args=[
                {"visible": visible},
                {"title": dict(
                    text=(f"Stage 6 — isolated iso {iso_cls} "
                          f"({len(site.member_node_ids)} members)"),
                    font=dict(size=14),
                )},
            ],
        ))

    fig_iso.update_layout(**_layout(
        f"Stage 6 — isolated iso {first_iso} "
        f"({len(sites_CO[0].member_node_ids)} members)"
    ))
    fig_iso.update_layout(
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        )]
    )
    fig_iso.show()

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

# Anchor-bond edges
bond_from, bond_to = [], []
for n, d in ads_nodes:
    if not d.get("is_bonded"):
        continue
    for nb in G.neighbors(n):
        if G.nodes[nb].get("type") != "surface":
            continue
        bond_from.append(d["position"])
        bond_to.append(G.nodes[nb]["position"])
if bond_from:
    fig.add_trace(_bond_lines3d(
        np.array(bond_from), np.array(bond_to),
        color="rgba(160,160,160,0.3)", width=1, showlegend=False,
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
        pa = np.asarray(d["position"])
        pb = np.asarray(G.nodes[nb]["position"])
        fig.add_trace(_bond_lines3d(
            pa[np.newaxis], pb[np.newaxis],
            color=_iso_col(d["iso_class"]), width=3, showlegend=False,
        ))

# Adsorbate atoms — one trace per (iso_class, element)
grp: dict = {}
for n, d in ads_nodes:
    key = (d["iso_class"], d["element"], bool(d.get("is_bonded")))
    grp.setdefault(key, []).append(d["position"])

seen: set = set()
for (iso_cls, sym, bonded), pts in sorted(grp.items()):
    lk  = (iso_cls, sym)
    fig.add_trace(_atom_scatter3d(
        np.array(pts),
        colors=_iso_col(iso_cls),
        sizes=9 if bonded else 5,
        names=[f"iso{iso_cls} {sym}"] * len(pts),
        opacity=0.95 if bonded else 0.4,
        symbol=_ADS_SYM.get(sym, "circle"),
        line_color="black" if bonded else _iso_col(iso_cls),
        line_width=0.8 if bonded else 0.0,
        name=f"iso {iso_cls}  {sym} ({'bonded' if bonded else 'float'})",
        showlegend=(lk not in seen),
    ))
    seen.add(lk)

fig.update_layout(**_layout(
    f"Final — {total_ads} CO adsorbate nodes  ({len(sites_CO)} stable iso-classes)"
))
fig.show()

# %% [markdown]
# ---
# ## Standalone pruning after the fact
#
# If you already ran `find_adsorbate_sites` without a calculator, you can
# prune afterwards.  `prune_unstable_adsorbate_sites` does the full B-2
# pipeline: ML-relax, check connectivity, prune unstable, update
# `ms.positions` from the relaxed Atoms, and Kabsch-propagate to all members.
#
# ```python
# from autokmc.sites.adsorbate import (
#     find_adsorbate_sites,
#     optimise_adsorbate_site_positions,
#     prune_unstable_adsorbate_sites,
# )
#
# # Step A: enumerate + materialise (no calculator needed)
# sites_CO_all = find_adsorbate_sites(G, co, prune_stable_only=False, verbose=True)
#
# # Step B-1: calc-free geometry optimisation (optional but recommended before B-2)
# optimise_adsorbate_site_positions(G, co.smiles, co, verbose=True)
#
# # Step B-2: ML stability check → prune → update positions → propagate
# sites_CO_stable = prune_unstable_adsorbate_sites(
#     G, sites_CO_all, co, calc,
#     frozen_indices = frozen_indices,
#     fmax           = 0.05,
#     max_steps      = 200,
#     verbose        = True,
# )
# # G.graph["adsorbate_sites"][co.smiles] is now the stable list.
# # ms.positions and all member node positions on G are ML-refined.
# ```

print("\nPipeline complete.")
