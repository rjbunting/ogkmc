# %% [markdown]
# # AutoKMC — CH4 Activation on Cu(111) with ML Stability Pruning
#
# Demonstrates the full pruned pipeline for methane activation on Cu(111):
#
#     CH4(g) ⇌ CH3* + H*
#
# This follows the same style as `get_adsorbate_sites_pruned.py`, but continues
# from stable adsorbate-site enumeration into the gas-product bond-reaction
# enumerator.  CH4 is deliberately kept as a gas-phase species: the surface
# placements are only CH3* and H*, and the bond-reaction template is supplied as
# `CH3* + H* ⇌ CH4(g)`.
#
# Pipeline steps
# --------------
# 1. Build Cu(111) slab
# 2. Find surface atoms
# 3. Build connectivity graph
# 4. Build CH4(g), CH3, and H reactants
# 5. `find_adsorbate_sites(..., calculator=calc)` for CH3 and H
# 6. Visualise final stable CH3*/H* sites
# 7. Build explicit `CH3 + H ⇌ CH4(g)` bond template
# 8. `find_bond_sites(..., gas_species={CH4})`
# 9. `prune_unstable_bond_sites(...)` for A+B endpoint stability
# 10. Visualise the surviving CH4-activation reaction sites

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
from autokmc.sites.bond import (
    BondReactionTemplate,
    find_bond_sites,
    prune_unstable_bond_sites,
)

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
_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_COL  = {"C": "#2ECC71", "H": "#F4D03F"}
_ADS_SYM  = {"C": "diamond", "H": "circle"}
_ISO_COLS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BCF60C",
    "#FABEBE", "#008080",
]
_EP_COLS = {"A": "#FF4500", "B": "#1E90FF", "C": "#32CD32"}


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


def _node_positions(G, nids):
    pts = [
        np.asarray(G.nodes[n]["position"], dtype=float)
        for n in nids if n in G and "position" in G.nodes[n]
    ]
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3))


def _node_centroid(G, nids):
    pts = _node_positions(G, nids)
    return pts.mean(axis=0) if len(pts) else np.zeros(3)


def _plot_reactant_site_reps(fig, sites, reactant, *, title_prefix: str):
    react_syms = reactant.atoms.get_chemical_symbols()
    for site in sites:
        col = _iso_col(site.iso_class)
        rep_pos = np.asarray(site.positions, dtype=float)
        for u, v in reactant.graph.edges():
            fig.add_trace(_bond_lines3d(
                rep_pos[[int(u)]], rep_pos[[int(v)]],
                color=col, width=4, showlegend=False,
            ))
        for i, sym in enumerate(react_syms):
            is_bonded = site.atom_cliques[i] is not None
            fig.add_trace(go.Scatter3d(
                x=[rep_pos[i, 0]], y=[rep_pos[i, 1]], z=[rep_pos[i, 2]],
                mode="markers",
                marker=dict(
                    size=13 if is_bonded else 9,
                    color=col,
                    symbol=_ADS_SYM.get(sym, "circle"),
                    opacity=0.95 if is_bonded else 0.55,
                    line=dict(color="black" if is_bonded else col,
                              width=2 if is_bonded else 0),
                ),
                text=[f"{title_prefix} iso {site.iso_class} atom {i} {sym}"],
                hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
                name=f"{title_prefix} iso {site.iso_class}",
                showlegend=(i == 0),
            ))


def _plot_bond_reaction_site(
    G,
    brs,
    *,
    pos,
    bulk_mask,
    surf_pos,
    title: str,
    max_members: int | None = None,
):
    """Show one BondReactionSite in its own 3-D figure."""
    fig = go.Figure()
    _add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

    members = list(enumerate(brs.member_node_ids))
    if max_members is not None:
        members = members[: int(max_members)]

    col = _iso_col(brs.iso_class)
    for m_idx, (a_nids, b_nids, _c_nids) in members:
        a_pos = _node_positions(G, a_nids)
        b_pos = _node_positions(G, b_nids)
        if not len(a_pos) or not len(b_pos):
            continue

        a_cent = a_pos.mean(axis=0)
        b_cent = b_pos.mean(axis=0)
        mid = (a_cent + b_cent) / 2.0
        c_cent = mid + np.array([0.0, 0.0, 6.0])

        fig.add_trace(_bond_lines3d(
            a_cent[np.newaxis], b_cent[np.newaxis],
            color=col, width=5, showlegend=False,
        ))
        fig.add_trace(_bond_lines3d(
            mid[np.newaxis], c_cent[np.newaxis],
            color="rgba(80,80,80,0.35)", width=2, showlegend=False,
        ))

        show = (m_idx == members[0][0])
        fig.add_trace(_atom_scatter3d(
            a_pos, colors=_EP_COLS["A"], sizes=9,
            names=[f"bond_iso {brs.iso_class} m{m_idx} CH3"] * len(a_pos),
            opacity=0.95, symbol="diamond", name="CH3*",
            showlegend=show,
        ))
        fig.add_trace(_atom_scatter3d(
            b_pos, colors=_EP_COLS["B"], sizes=9,
            names=[f"bond_iso {brs.iso_class} m{m_idx} H"] * len(b_pos),
            opacity=0.95, symbol="circle", name="H*",
            showlegend=show,
        ))
        fig.add_trace(_atom_scatter3d(
            c_cent[np.newaxis], colors=_EP_COLS["C"], sizes=7,
            names=[f"bond_iso {brs.iso_class} m{m_idx} CH4(g) lifted endpoint"],
            opacity=0.45, symbol="square", name="CH4(g) NEB endpoint",
            showlegend=show,
        ))

    fig.update_layout(**_layout(title))
    fig.show()
    return fig


# %% ── 1. Build Cu(111) slab ──────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition="Au",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    lattice_constant=4.015,
    min_slab_size=8.0,
    min_vacuum_size=15.0,
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

# %% ── 4. Build reactants ─────────────────────────────────────────────────────
_section("STAGE 4 — Build CH4(g), CH3, and H reactants")

ch4 = build_reactant("[CH4]", add_hydrogens=False, calculator=calc,
                     partial_pressure_bar=1.0)
ch3 = build_reactant("[CH3]", add_hydrogens=False, calculator=calc)
h_atom = build_reactant("[H]", add_hydrogens=False, calculator=calc)

reactants = {
    ch3.smiles: ch3,
    h_atom.smiles: h_atom,
    ch4.smiles: ch4,
}

for r in (ch4, ch3, h_atom):
    r_syms = r.atoms.get_chemical_symbols()
    print(f"SMILES       : {r.smiles}")
    print(f"Formula      : {r.atoms.get_chemical_formula()}")
    print(f"Gas energy   : {r.energy:+.6f} eV")
    print(f"Pressure     : {getattr(r, 'partial_pressure_bar', 1.0):.3g} bar")
    print(f"Anchor atoms : {r.anchor_atoms}  "
          f"(elements: {[r_syms[i] for i in r.anchor_atoms]})")
    print()

# %% [markdown]
# ---
# ## Stage 5 — Find Adsorbate Sites WITH Stability Pruning
#
# `CH4` is not enumerated as an adsorbate here.  The surface species are
# `CH3*` and `H*`; `CH4` enters later through `gas_species` in the bond
# reaction enumerator.
#
# With `prune_stable_only=True` and a valid calculator each call automatically:
# 1. Materialises all enumerated placements onto G.
# 2. Runs calc-free rigid-body optimisation and propagates to members.
# 3. ML-relaxes each representative, prunes unstable iso-classes, updates
#    `ms.positions`, and Kabsch-propagates to all members.

# %% ── 5. Find adsorbate sites (with pruning) ─────────────────────────────────
_section("STAGE 5 — Find adsorbate sites for CH3 and H  [pruning ON]")

sites_by_smiles = {}
all_ads_sites = []

for reactant in (ch3, h_atom):
    sites = find_adsorbate_sites(
        G, reactant,
        prune_stable_only = True,
        calculator        = calc,
        frozen_indices    = frozen_indices,
        prune_fmax        = 0.05,
        prune_max_steps   = 200,
        verbose           = True,
    )
    sites_by_smiles[reactant.smiles] = sites
    all_ads_sites.extend(sites)

    react_syms = reactant.atoms.get_chemical_symbols()
    print(f"\n{reactant.smiles}: {len(sites)} stable unique iso-class(es) after pruning")
    for site in sites:
        bonded = [(i, c) for i, c in enumerate(site.atom_cliques) if c is not None]
        bonded_str = "  ".join(
            f"{react_syms[i]}→k={len(c)}" for i, c in bonded
        ) if bonded else "all-floating"
        print(f"  iso {site.iso_class:2d}  {bonded_str:<30s}"
              f"  members={len(site.members):<4d}"
              f"  rep_pos[0]={np.round(site.positions[0], 2)}")

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)
_plot_reactant_site_reps(fig, sites_by_smiles[ch3.smiles], ch3, title_prefix="CH3")
_plot_reactant_site_reps(fig, sites_by_smiles[h_atom.smiles], h_atom, title_prefix="H")
fig.update_layout(**_layout(
    "Stage 5 — stable CH3 and H iso-classes (representative positions)"
))
fig.show()

# %% [markdown]
# ---
# ## Stage 6 — Visualise stable adsorbate sites
#
# All positions on G are already ML-refined.  No further optimisation step is
# needed before bond-reaction enumeration.

# %% ── 6. Final adsorbate-site visualisation ──────────────────────────────────
_section("STAGE 6 — Stable CH3*/H* adsorbate sites (all members, ML-refined)")

ads_nodes = [
    (n, d) for n, d in G.nodes(data=True)
    if d.get("type") == "adsorbate" and d.get("reactant") in {ch3.smiles, h_atom.smiles}
]

total_ads = len(ads_nodes)
n_bonded  = sum(1 for _, d in ads_nodes if d.get("is_bonded"))
print(f"Adsorbate nodes in G : {total_ads}  ({n_bonded} bonded)")
print(f"Stable iso-classes   : {len(all_ads_sites)}")
print(f"Total placements     : {sum(len(s.members) for s in all_ads_sites)}")

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

# Anchor-bond edges.
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

# Intra-molecular bonds.
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

# Adsorbate atoms, one trace per (reactant, iso_class, element).
grp: dict = {}
for n, d in ads_nodes:
    key = (d["reactant"], d["iso_class"], d["element"], bool(d.get("is_bonded")))
    grp.setdefault(key, []).append(d["position"])

seen: set = set()
for (smi, iso_cls, sym, bonded), pts in sorted(grp.items()):
    lk = (smi, iso_cls, sym)
    fig.add_trace(_atom_scatter3d(
        np.array(pts),
        colors=_iso_col(iso_cls),
        sizes=9 if bonded else 5,
        names=[f"{smi} iso{iso_cls} {sym}"] * len(pts),
        opacity=0.95 if bonded else 0.4,
        symbol=_ADS_SYM.get(sym, "circle"),
        line_color="black" if bonded else _iso_col(iso_cls),
        line_width=0.8 if bonded else 0.0,
        name=f"{smi} iso {iso_cls} {sym} ({'bonded' if bonded else 'float'})",
        showlegend=(lk not in seen),
    ))
    seen.add(lk)

fig.update_layout(**_layout(
    f"Final adsorbates — {total_ads} CH3/H nodes  ({len(all_ads_sites)} stable iso-classes)"
))
fig.show()

# %% ── 7. Explicit CH4 activation template ───────────────────────────────────
_section("STAGE 7 — Build explicit CH4 activation template")

template = BondReactionTemplate(
    smiles_a  = ch3.smiles,
    smiles_b  = h_atom.smiles,
    smiles_c  = ch4.smiles,
    bond_type = "SINGLE",
    element_a = "C",
    element_b = "H",
    source    = "manual_ch4_activation",
)
templates = [template]
gas_species = {ch4.smiles: ch4}

print(f"Template : {template}")
print("CH4 is passed through gas_species; no CH4 adsorbate site is required.")

# %% ── 8. Find bond reaction sites ────────────────────────────────────────────
_section("STAGE 8 — find_bond_sites for CH3* + H* ⇌ CH4(g)")

bond_sites = find_bond_sites(
    G,
    all_ads_sites,
    templates,
    max_hops        = 0,
    deduplicate_iso = True,
    prune_by_triple = False,
    gas_species     = gas_species,
    allow_gas_products = True,
    gas_lift_height = 6.0,
    verbose         = True,
)

print(f"\nBond reaction iso-classes before A+B pruning: {len(bond_sites)}")
for brs in bond_sites:
    sa, ma, sb, mb, _sc, _mc = brs.members[0]
    print(f"  bond_iso {brs.iso_class:2d}  "
          f"gas_product={brs.gas_product}  members={len(brs.members):<4d}  "
          f"A={sa.reactant} iso={sa.iso_class} m={ma}  "
          f"B={sb.reactant} iso={sb.iso_class} m={mb}  "
          f"C=gas({brs.template.smiles_c})")

if not bond_sites:
    raise RuntimeError(
        "No CH4 activation bond-reaction iso-classes found.  Check that CH3* "
        "and H* sites survived pruning, or increase max_hops."
    )

# %% ── 8b. Pre-pruning bond-site plots ────────────────────────────────────────
_section("STAGE 8b — Pre-pruning CH4 activation bond sites")

raw_bond_sites = list(bond_sites)
print(
    f"Showing {len(raw_bond_sites)} pre-pruning BondReactionSite iso-class(es). "
    "Each figure contains all members in that iso-class."
)

for brs in raw_bond_sites:
    _plot_bond_reaction_site(
        G,
        brs,
        pos=pos,
        bulk_mask=bulk_mask,
        surf_pos=surf_pos,
        title=(
            f"Pre-pruning bond_iso {brs.iso_class}: "
            f"{brs.template.smiles_a} + {brs.template.smiles_b} ⇌ "
            f"{brs.template.smiles_c}  ({len(brs.members)} member(s))"
        ),
    )

# %% ── 9. Prune unstable A+B endpoints ───────────────────────────────────────
_section("STAGE 9 — prune_unstable_bond_sites for A+B endpoint")

bond_sites = prune_unstable_bond_sites(
    G,
    bond_sites,
    reactants,
    calc,
    frozen_indices = frozen_indices,
    fmax           = 0.05,
    max_steps      = 200,
    verbose        = True,
)

print(f"\nBond reaction iso-classes after A+B pruning: {len(bond_sites)}")
for brs in bond_sites:
    print(f"  bond_iso {brs.iso_class:2d}  members={len(brs.members):<4d}  "
          f"template={brs.template.smiles_a}+{brs.template.smiles_b}"
          f"⇌{brs.template.smiles_c}  gas_product={brs.gas_product}")

if not bond_sites:
    raise RuntimeError(
        "A+B pruning removed every CH4 activation site.  Inspect the verbose "
        "endpoint output above for CH3/H connectivity changes."
    )

# %% ── 10. Final reaction-site visualisation ─────────────────────────────────
_section("STAGE 10 / FINAL — CH4 activation reaction sites")

fig = go.Figure()
_add_slab_backdrop(fig, pos, bulk_mask, surf_pos)

for brs in bond_sites:
    col = _iso_col(brs.iso_class)
    for m_idx, (a_nids, b_nids, _c_nids) in enumerate(brs.member_node_ids):
        a_pos = _node_positions(G, a_nids)
        b_pos = _node_positions(G, b_nids)
        if not len(a_pos) or not len(b_pos):
            continue
        a_cent = a_pos.mean(axis=0)
        b_cent = b_pos.mean(axis=0)
        c_cent = (a_cent + b_cent) / 2.0 + np.array([0.0, 0.0, 6.0])

        fig.add_trace(_bond_lines3d(
            a_cent[np.newaxis], b_cent[np.newaxis],
            color=col, width=4, showlegend=False,
        ))
        fig.add_trace(_bond_lines3d(
            ((a_cent + b_cent) / 2.0)[np.newaxis], c_cent[np.newaxis],
            color="rgba(80,80,80,0.35)", width=2, showlegend=False,
        ))

        show = (m_idx == 0)
        fig.add_trace(_atom_scatter3d(
            a_pos, colors=_EP_COLS["A"], sizes=8,
            names=[f"bond_iso {brs.iso_class} m{m_idx} CH3"] * len(a_pos),
            opacity=0.9, symbol="diamond", name=f"bond {brs.iso_class} CH3*",
            showlegend=show,
        ))
        fig.add_trace(_atom_scatter3d(
            b_pos, colors=_EP_COLS["B"], sizes=8,
            names=[f"bond_iso {brs.iso_class} m{m_idx} H"] * len(b_pos),
            opacity=0.9, symbol="circle", name=f"bond {brs.iso_class} H*",
            showlegend=show,
        ))
        fig.add_trace(_atom_scatter3d(
            c_cent[np.newaxis], colors=_EP_COLS["C"], sizes=7,
            names=[f"bond_iso {brs.iso_class} m{m_idx} CH4(g) lifted endpoint"],
            opacity=0.45, symbol="square", name=f"bond {brs.iso_class} CH4(g)",
            showlegend=show,
        ))

fig.update_layout(**_layout(
    f"Final — {len(bond_sites)} CH4 activation bond iso-class(es)"
))
fig.show()

# %% [markdown]
# ---
# ## Optional NEB after the fact
#
# To debug the actual gas-product endpoint relaxation and NEB for one surviving
# lateral class, run:
#
# ```python
# from autokmc.sites.stability.bond import (
#     check_bond_site_lateral,
#     check_bond_site_stability,
# )
#
# brs = bond_sites[0]
# member_index = 0
# lc = check_bond_site_lateral(G, brs, member_index, n_shells=1)
# E_ab, E_c, E_ts = check_bond_site_stability(
#     G, brs, member_index, lc, calc,
#     frozen_indices=frozen_indices,
#     fmax=0.05,
#     max_steps=500,
#     n_images=8,
#     climb=True,
#     spring_k=0.1,
#     interpolation="linear",
#     persist_neb_path=True,
#     verbose=True,
# )
# ```
#
# In that path, the CH4 geometry used for the NEB endpoint is lifted 6 Å above
# the CH3/H centroid.  The energy for the gas endpoint is still the relaxed
# empty surface plus the gas-phase CH4 energy.

print("\nPipeline complete.")
