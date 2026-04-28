# %% [markdown]
# # AutoKMC — CO Adsorption / Desorption KMC on Cu(111)
#
# A complete, end-to-end **KMC** example for CO on Cu(111) using the
# library API exposed by :mod:`autokmc`.  Mirrors the structure of
# `get_adsorbate_sites_pruned.py` and extends it with the BKL/Gillespie
# loop from :mod:`autokmc.kmc_simulation`.
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Build Cu(111) slab |
# | 2 | Find surface atoms |
# | 3 | Build connectivity graph |
# | 4 | Build CO reactant (gas-phase **energy** stored on the Reactant) |
# | 5 | `find_adsorbate_sites(..., prune_stable_only=True, calculator=calc)` |
# | 6 | `run_kmc_steps(...)` — adsorption ⇌ desorption KMC loop |
# | 7 | Visualise the final occupied configuration |
#
# Energetics for adsorption / desorption
# --------------------------------------
# Each lateral class on every iso-class member is ML-relaxed *once*, giving
# `E_occ` and `E_unocc` (the surface energy with / without that adsorbate
# present in its current lateral environment).  Reaction energies include
# the gas-phase reactant energy `E_gas = co.energy`:
#
# ```
# Adsorption :  ΔE = E_occ - (E_unocc + E_gas)
# Desorption :  ΔE = (E_unocc + E_gas) - E_occ
# Barrier    :  Ea = max(0.1, ΔE + 0.1)                    # eV
# Rate       :  k  = κ · (k_B·T / h) · exp(-Ea / kT)      # Eyring, κ = 1
# ```

# %% ── 0. Imports ────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from nequip.ase import NequIPCalculator

from autokmc import (
    build_surface,
    find_surface_atoms,
    build_graph,
    build_reactant,
    find_adsorbate_sites,
    run_kmc_steps,
)

# %% ── 1. Calculator ────────────────────────────────────────────────────────
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path("."))
    / _MODEL_FILE
)
print(f"Device : {_DEVICE}   Model : {_MODEL_FILE}")


def make_calc():
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )


calc = make_calc()


# %% ── 2. KMC parameters ────────────────────────────────────────────────────
TEMPERATURE_K   : float = 500.0     # simulation temperature (K)
N_KMC_STEPS     : int   = 10000        # max KMC events
FMAX            : float = 0.05      # ML force convergence (eV/Å)
MAX_OPT_STEPS   : int   = 500       # max LBFGS steps per relaxation
RANDOM_SEED     : int   = 69

_BAR = "─" * 64
def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")


# %% ── 3. Build Cu(111) slab ────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition       = "Cu",
    crystal_structure = "fcc",
    miller_index      = (1, 1, 1),
    lattice_constant  = 3.615,
    min_slab_size     = 12.0,
    min_vacuum_size   = 12.0,
    goal_x            = 20.0,
    goal_y            = 20.0,
    n_freeze_layers   = 2,
    calculator        = calc,
    verbose           = True,
)
print(f"Slab: {len(atoms)} atoms   cell diag: {np.diag(atoms.get_cell()).round(3)} Å")

frozen_indices: list[int] | None = list(atoms.info.get("frozen_indices", []) or [])
if not frozen_indices:
    frozen_indices = None


# %% ── 4. Surface atoms + graph ─────────────────────────────────────────────
_section("STAGE 2 — Find surface atoms & build graph")

find_surface_atoms(atoms, tag_atoms=True)
G = build_graph(atoms)
print(f"Graph: {G.number_of_nodes()} nodes   {G.number_of_edges()} edges")


# %% ── 5. CO reactant (gas-phase energy stored on Reactant) ─────────────────
_section("STAGE 3 — Build CO reactant (relax in gas phase to set co.energy)")

co = build_reactant(
    "[C-]#[O+]",
    add_hydrogens = False,
    calculator    = calc,           # relax + get gas-phase energy
)
print(f"Formula      : {co.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {co.anchor_atoms}")
print(f"Bond length  : {co.atoms.get_all_distances()[0, 1]:.3f} Å")
print(f"Gas-phase E  : {co.energy:.4f} eV  ← used as E_gas in ΔE")


# %% ── 6. Adsorbate sites (with ML stability pruning) ──────────────────────
_section("STAGE 4 — Adsorbate sites  [enumerate → geom opt → ML prune → propagate]")

adsorbate_sites = find_adsorbate_sites(
    G, co,
    prune_stable_only = True,
    calculator        = calc,
    frozen_indices    = frozen_indices,
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


# %% ── 7. Run the KMC loop ──────────────────────────────────────────────────
_section("STAGE 5 — KMC loop (adsorption ⇌ desorption)")

summary = run_kmc_steps(
    G, adsorbate_sites, calc,
    reactants                = co,          # gas-phase E taken from co.energy
    temperature              = TEMPERATURE_K,
    n_steps                  = N_KMC_STEPS,
    transmission_coefficient = 1.0,         # Eyring κ = 1 → ν = k_B·T/h
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


# %% ── 8. Visualise the final occupied configuration ───────────────────────
_section("STAGE 6 — Final occupied CO configuration")

pos        = atoms.get_positions()
surf_mask  = atoms.arrays.get("surface", np.zeros(len(atoms), dtype=bool)).astype(bool)
bulk_mask  = ~surf_mask

_BULK_COL = "#aaaaaa"
_SURF_COL = "#4A90D9"
_ADS_COL  = {"C": "#2ECC71", "O": "#E74C3C"}

fig = go.Figure()

if bulk_mask.any():
    fig.add_trace(go.Scatter3d(
        x=pos[bulk_mask, 0], y=pos[bulk_mask, 1], z=pos[bulk_mask, 2],
        mode="markers",
        marker=dict(size=3, color=_BULK_COL, opacity=0.2),
        name="bulk Cu",
    ))
fig.add_trace(go.Scatter3d(
    x=pos[surf_mask, 0], y=pos[surf_mask, 1], z=pos[surf_mask, 2],
    mode="markers",
    marker=dict(size=6, color=_SURF_COL, opacity=0.55),
    name="surface Cu",
))

ads_nodes = [(n, d) for n, d in G.nodes(data=True)
             if d.get("type") == "adsorbate"
             and d.get("reactant") == co.smiles
             and d.get("occupied", False)]

drawn_pairs: set = set()
for n, d in ads_nodes:
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
        fig.add_trace(go.Scatter3d(
            x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]],
            mode="lines",
            line=dict(color="black", width=4),
            hoverinfo="skip", showlegend=False,
        ))

grp: dict = {}
for n, d in ads_nodes:
    grp.setdefault(d["element"], []).append(d["position"])
for sym, pts in grp.items():
    pts_arr = np.asarray(pts)
    fig.add_trace(go.Scatter3d(
        x=pts_arr[:, 0], y=pts_arr[:, 1], z=pts_arr[:, 2],
        mode="markers",
        marker=dict(size=10, color=_ADS_COL.get(sym, "#999999"),
                    line=dict(color="black", width=1)),
        name=f"CO {sym}",
    ))

n_occ_final = sum(len(s.occupied_member_indices(G)) for s in adsorbate_sites)
coverage    = n_occ_final / total_members if total_members else 0.0
fig.update_layout(
    title=dict(
        text=(f"Final state — {n_occ_final}/{total_members} CO occupied "
              f"(θ = {coverage:.2f})  "
              f"after {summary['steps_executed']} KMC steps  "
              f"t = {summary['time']:.2e} s"),
        font=dict(size=14),
    ),
    scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
               aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0),
    template="plotly_white",
    showlegend=True,
)
fig.show()


# %% ── 9. Reaction history summary ─────────────────────────────────────────
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

