# %% [markdown]
# # AutoKMC — CO Adsorption / Desorption KMC on a Cu Nanoparticle
#
# End-to-end **KMC** example for CO on a Wulff-constructed Cu nanoparticle,
# mirroring `get_co_adsorption.py` but using :func:`~autokmc.build_nanoparticle`
# in place of :func:`~autokmc.build_surface`.
#
# Key differences from the slab version
# --------------------------------------
# * Surface classification uses the **convex-hull** algorithm (dispatched
#   automatically by ``find_surface_atoms``).
# * ``G.graph["pbc"]`` will be ``[False, False, False]`` (no PBC bonds cross
#   the vacuum box).
# * There are **no frozen layers** for a nanoparticle; pass
#   ``frozen_indices=None`` throughout.
# * The site enumeration covers the full convex-hull surface — all facets
#   ((111), (100), (110) …) are found simultaneously.
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Wulff-construct a Cu nanoparticle |
# | 2 | Classify surface atoms via convex hull |
# | 3 | Build connectivity graph |
# | 4 | Build CO reactant (gas-phase energy stored) |
# | 5 | `find_adsorbate_sites(..., prune_stable_only=True, calculator=calc)` |
# | 6 | `run_kmc_steps(...)` — adsorption ⇌ desorption KMC loop |
# | 7 | Visualise the final occupied configuration |

# %% ── 0. Imports ─────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc import (
    build_nanoparticle,
    find_surface_atoms,
    build_graph,
    build_reactant,
    find_adsorbate_sites,
    run_kmc_steps,
)

# %% ── 1. Calculator ──────────────────────────────────────────────────────────
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path(""))
    / _MODEL_FILE
)
print(f"Device : {_DEVICE}   Model : {_MODEL_FILE}")


def make_calc():
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )


calc = make_calc()


# %% ── 2. KMC / build parameters ────────────────────────────────────────────
TARGET_ATOMS    : int   = 201       # approximate NP size (WulffPack target)
TEMPERATURE_K   : float = 500.0    # simulation temperature (K)
N_KMC_STEPS     : int   = 10_000  # max KMC events
FMAX            : float = 0.05     # ML force convergence (eV/Å)
MAX_OPT_STEPS   : int   = 500      # max LBFGS steps per relaxation
RANDOM_SEED     : int   = 69

# Wulff surface energies for Cu (DFT-fitted, J/m²)
# (111) is the close-packed facet; (100) and (110) are higher-energy.
SURFACE_ENERGIES = {
    (1, 1, 1): 1.10,
    (1, 0, 0): 1.29,
    (1, 1, 0): 1.51,
}

_BAR = "─" * 64


def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")


# %% ── 3. Build Cu nanoparticle ───────────────────────────────────────────────
_section("STAGE 1 — Wulff-construct Cu nanoparticle")

atoms = build_nanoparticle(
    composition       = "Cu",
    crystal_structure = "fcc",
    lattice_constant  = 3.615,      # Å — known EMT-optimised value
    surface_energies  = SURFACE_ENERGIES,
    target_atoms      = TARGET_ATOMS,
    vacuum            = 10.0,       # Å vacuum padding on each side
    calculator        = calc,
    verbose           = True,
)

pos  = atoms.get_positions()
syms = np.array(atoms.get_chemical_symbols())
print(f"Nanoparticle: {len(atoms)} atoms")
print(f"Cell diag   : {np.diag(atoms.get_cell()).round(2)} Å")

# Nanoparticles have no frozen layers.
frozen_indices: list[int] | None = None


# %% ── 4. Surface atoms + graph ───────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms (convex hull) & build graph")

surf_result = find_surface_atoms(atoms, tag_atoms=True)
surf_mask   = surf_result.mask
bulk_mask   = ~surf_mask

print(f"Method        : {surf_result.method}")   # should be "convexhull"
print(f"Surface atoms : {surf_mask.sum()} / {len(atoms)}")
print(f"Bulk atoms    : {bulk_mask.sum()}")

G = build_graph(atoms)
print(f"Graph : {G.number_of_nodes()} nodes   {G.number_of_edges()} edges")
print(f"PBC   : {G.graph['pbc'].tolist()}")      # [False, False, False]


# %% ── 5. Visualise raw nanoparticle ──────────────────────��──────────────────
_ELEM_COL = {"Cu": "#B87333", "Pt": "#C0C0C0", "Au": "#FFD700"}
_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_COL  = {"C": "#2ECC71", "O": "#E74C3C"}


def _ecolor(sym: str) -> str:
    return _ELEM_COL.get(sym, "#888888")


def _atom_scatter3d(pos_arr, colors, sizes, names,
                    opacity=1.0, symbol="circle",
                    line_color="black", line_width=0.5,
                    name="", showlegend=True):
    return go.Scatter3d(
        x=pos_arr[:, 0], y=pos_arr[:, 1], z=pos_arr[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=colors, symbol=symbol, opacity=opacity,
                    line=dict(color=line_color, width=line_width)),
        text=names,
        hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
        name=name, showlegend=showlegend,
    )


def _bond_lines3d(p_from, p_to,
                  color="rgba(160,160,160,0.4)", width=1,
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
        title=dict(text=title, font=dict(size=14)), showlegend=True,
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0), template="plotly_white",
    )


fig = go.Figure()
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=4, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.25, name="bulk Cu",
    ))
surf_pos = pos[surf_mask]
fig.add_trace(_atom_scatter3d(
    surf_pos, colors=_SURF_COL,
    sizes=8, names=[f"surf {i}" for i in np.where(surf_mask)[0]],
    opacity=0.9, name="surface Cu",
))
fig.update_layout(**_layout(
    f"Stage 1/2 — Cu NP ({len(atoms)} atoms) · surface classification"
))
fig.show()


# %% ── 6. CO reactant ─────────────────────────────────────────────────────────
_section("STAGE 3 — Build CO reactant (relax in gas phase)")

co = build_reactant(
    "[C-]#[O+]",
    add_hydrogens = False,
    calculator    = calc,
)
print(f"Formula      : {co.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {co.anchor_atoms}")
print(f"Bond length  : {co.atoms.get_all_distances()[0, 1]:.3f} Å")
print(f"Gas-phase E  : {co.energy:.4f} eV  ← used as E_gas in ΔE")


# %% ── 7. Adsorbate sites (with ML stability pruning) ────────────────────────
_section("STAGE 4 — Adsorbate sites  [enumerate → geom opt → ML prune → propagate]")

adsorbate_sites = find_adsorbate_sites(
    G, co,
    prune_stable_only = True,
    calculator        = calc,
    frozen_indices    = frozen_indices,   # None for NP
    prune_fmax        = FMAX,
    prune_max_steps   = MAX_OPT_STEPS,
    verbose           = True,
)

total_members = sum(len(s.members) for s in adsorbate_sites)
print(f"\nStable CO iso-classes : {len(adsorbate_sites)}   "
      f"total members : {total_members}")
for s in adsorbate_sites:
    print(f"  iso {s.iso_class:2d}   members={len(s.members):<4d}   "
          f"rep_pos[C]={np.round(s.positions[0], 2)}")


# %% ── 8. Run the KMC loop ────────────────────────────────────────────────────
_section("STAGE 5 — KMC loop (adsorption ⇌ desorption)")

summary = run_kmc_steps(
    G, adsorbate_sites, calc,
    reactants                = co,
    temperature              = TEMPERATURE_K,
    n_steps                  = N_KMC_STEPS,
    transmission_coefficient = 1.0,
    frozen_indices           = frozen_indices,
    fmax                     = FMAX,
    max_steps                = MAX_OPT_STEPS,
    rng                      = RANDOM_SEED,
    log_every                = 1,
    verbose                  = True,
)

print("\nKMC summary:")
print(f"  steps executed  : {summary['steps_executed']}")
print(f"  total time      : {summary['time']:.4e} s")
print(f"  reaction counts : {summary['reaction_counts']}")
print(f"  occupancy/iso   : {summary['final_occupancy']}")


# %% ── 9. Visualise the final occupied configuration ─────────────────────────
_section("STAGE 6 — Final occupied CO configuration on the nanoparticle")

ads_nodes_occ = [
    (n, d) for n, d in G.nodes(data=True)
    if d.get("type") == "adsorbate"
    and d.get("reactant") == co.smiles
    and d.get("occupied", False)
]

fig = go.Figure()

# NP backdrop
if bulk_mask.any():
    fig.add_trace(_atom_scatter3d(
        pos[bulk_mask], colors=_BULK_COL,
        sizes=3, names=[f"bulk {i}" for i in np.where(bulk_mask)[0]],
        opacity=0.15, name="bulk Cu",
    ))
fig.add_trace(_atom_scatter3d(
    surf_pos, colors=_SURF_COL,
    sizes=7, names=[f"surf {i}" for i in np.where(surf_mask)[0]],
    opacity=0.55, name="surface Cu",
))

# Intra-molecular bonds (C–O sticks)
drawn_pairs: set = set()
for n, d in ads_nodes_occ:
    for nb in G.neighbors(n):
        if not G.edges[n, nb].get("intra_adsorbate"):
            continue
        if not G.nodes[nb].get("occupied", False):
            continue
        key = (min(n, nb), max(n, nb))
        if key in drawn_pairs:
            continue
        drawn_pairs.add(key)
        pa = np.asarray(d["position"])
        pb = np.asarray(G.nodes[nb]["position"])
        fig.add_trace(_bond_lines3d(
            pa[np.newaxis], pb[np.newaxis],
            color="black", width=4, showlegend=False,
        ))

# Adsorbate atom markers, grouped by element
grp: dict = {}
for n, d in ads_nodes_occ:
    grp.setdefault(d["element"], []).append(d["position"])
for sym, pts in grp.items():
    pts_arr = np.asarray(pts)
    fig.add_trace(_atom_scatter3d(
        pts_arr,
        colors=_ADS_COL.get(sym, "#999999"),
        sizes=10,
        names=[f"CO {sym}"] * len(pts_arr),
        symbol="diamond" if sym == "C" else "cross",
        line_color="black", line_width=1,
        name=f"CO {sym}",
    ))

n_occ_final = sum(len(s.occupied_member_indices(G)) for s in adsorbate_sites)
coverage    = n_occ_final / total_members if total_members else 0.0
fig.update_layout(**_layout(
    f"Final state — {n_occ_final}/{total_members} CO occupied "
    f"(θ = {coverage:.2f})  "
    f"after {summary['steps_executed']} KMC steps  "
    f"t = {summary['time']:.2e} s"
))
fig.show()


# %% ── 10. Reaction history summary ──────────────────────────────────────────
_section("STAGE 7 — Reaction history")

hist = summary["history"]
print(f"  {'step':>4}  {'time / s':>11}  {'kind':<11}  "
      f"{'iso':>3}  {'m':>3}  {'lat':>3}  "
      f"{'ΔE / eV':>9}  {'Ea / eV':>8}  {'rate / Hz':>11}")
print("  " + "─" * 78)
for (step, t, kind, iso, m, lat, dE, Ea, rate) in hist:
    arrow = "↓" if dE < 0 else "↑"
    print(f"  {step:>4}  {t:>11.3e}  {kind:<11}  "
          f"{iso:>3}  {m:>3}  {lat:>3}  "
          f"{dE:>+9.4f}{arrow}  {Ea:>8.4f}  {rate:>11.3e}")

print("\nDone.")


