# %% [markdown]
# # AutoKMC — CO Diffusion on Cu(111) (debug notebook)
#
# A step-by-step script for inspecting the diffusion pipeline.  Every
# stage is a separate ``# %%`` cell so it can be converted to a Jupyter
# notebook with jupytext or run section-by-section in an IDE.
#
# The script is deliberately **verbose at every step** so you can see
# exactly where a hang or error originates.  Set ``SMALL_SLAB = True``
# (default) to use a compact cell that runs fast; flip to ``False`` for
# the full 20 Å × 20 Å slab used in production.
#
# Pipeline
# --------
# | Stage | What happens |
# |---|---|
# | 1 | Build Cu(111) slab |
# | 2 | Find surface atoms + build graph |
# | 3 | Relax CO in gas phase (get E_gas) |
# | 4 | Find + prune adsorbate sites |
# | 5 | **find_diffusion_sites** — enumerate hop pairs |
# | 6 | **check_diffusion_site_lateral** — lateral class for member 0 of iso 0 |
# | 7 | **check_diffusion_stability** — relax both endpoints + run NEB |
# | 8 | Plot the NEB energy profile |
# | 9 | Run a short KMC loop with diffusion enabled |
# | 10 | Print the reaction history |

# %% ── 0. Imports ─────────────────────────────────────────────────────────────
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
    find_diffusion_sites,
    run_kmc_steps,
)
from autokmc.check_diffusion_sites import (
    check_diffusion_site_lateral,
    check_diffusion_stability,
    DiffusionStabilityError,
)

# %% ── 1. Configuration ──────────────────────────────────────────────────────
# Set SMALL_SLAB = True for a fast ~12×12 Å debug cell.
SMALL_SLAB = True

TEMPERATURE_K   : float = 500.0
N_KMC_STEPS     : int   = 50        # keep short — mainly for checking diffusion fires
FMAX_ADS        : float = 0.05      # adsorbate pruning convergence (eV/Å)
FMAX_NEB        : float = 0.05      # NEB / endpoint convergence (eV/Å)
MAX_OPT_STEPS   : int   = 500
NEB_N_IMAGES    : int   = 5         # intermediate images (small for speed)
NEB_CLIMB       : bool  = True
NEB_SPRING_K    : float = 0.1
NEB_INTERP      : str   = "linear"  # "linear" or "idpp"
RANDOM_SEED     : int   = 69

_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path("."))
    / _MODEL_FILE
)

_BAR = "─" * 64
def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")

print(f"Device : {_DEVICE}   Model : {_MODEL_FILE}")
print(f"Small slab : {SMALL_SLAB}   NEB images : {NEB_N_IMAGES}   "
      f"interpolation : {NEB_INTERP}")

# %% ── 2. Calculator ─────────────────────────────────────────────────────────
_section("Calculator")

def make_calc():
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )

calc = make_calc()
print("Calculator ready.")

# %% ── 3. Build Cu(111) slab ─────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

_goal = 15.0

atoms = build_surface(
    composition       = "Cu",
    crystal_structure = "fcc",
    miller_index      = (1, 1, 1),
    lattice_constant  = 3.615,
    min_slab_size     = 8.0,
    min_vacuum_size   = 12.0,
    goal_x            = _goal,
    goal_y            = _goal,
    n_freeze_layers   = 2,
    calculator        = calc,
    verbose           = True,
)
print(f"Slab: {len(atoms)} atoms   cell diag: {np.diag(atoms.get_cell()).round(3)} Å")

frozen_indices: list[int] | None = list(atoms.info.get("frozen_indices", []) or [])
if not frozen_indices:
    frozen_indices = None

# %% ── 4. Surface atoms + graph ───────────────────────────────────────────────
_section("STAGE 2 — Surface atoms + graph")

find_surface_atoms(atoms, tag_atoms=True)
G = build_graph(atoms)
print(f"Graph: {G.number_of_nodes()} nodes   {G.number_of_edges()} edges")
print(f"PBC: {G.graph['pbc']}")

# %% ── 5. CO reactant (gas-phase energy) ─────────────────────────────────────
_section("STAGE 3 — CO reactant (gas-phase relaxation)")

co = build_reactant(
    "[C]=O",
    add_hydrogens = False,
    calculator    = calc,
)
print(f"Formula      : {co.atoms.get_chemical_formula()}")
print(f"Anchor atoms : {co.anchor_atoms}")
print(f"Gas-phase E  : {co.energy:.4f} eV")

# %% ── 6. Adsorbate sites (with ML stability pruning) ────────────────────────
_section("STAGE 4 — Adsorbate sites [enumerate → ML prune → propagate]")

adsorbate_sites = find_adsorbate_sites(
    G, co,
    prune_stable_only = True,
    calculator        = calc,
    frozen_indices    = frozen_indices,
    prune_fmax        = FMAX_ADS,
    prune_max_steps   = MAX_OPT_STEPS,
    verbose           = True,
)

total_members = sum(len(s.members) for s in adsorbate_sites)
print(f"\nStable CO iso-classes : {len(adsorbate_sites)}   "
      f"total members : {total_members}")
for s in adsorbate_sites:
    print(f"  iso {s.iso_class:2d}  members={len(s.members):<4d}  "
          f"rep_pos[C]={np.round(s.positions[0], 2)}")

# %% ── 7. Diffusion sites (hop-pair enumeration) ──────────────────────────────
_section("STAGE 5 — find_diffusion_sites")

diff_by_smiles = find_diffusion_sites(
    G, adsorbate_sites,
    max_hops      = 0,
    n_shells_pair = 1,
    verbose       = True,
)

diff_sites_flat = [ds for dss in diff_by_smiles.values() for ds in dss]
total_diff_members = sum(len(ds.members) for ds in diff_sites_flat)

print(f"\nDiffusion iso-classes : {len(diff_sites_flat)}   "
      f"total hop-pair members : {total_diff_members}")
for ds in diff_sites_flat:
    site_a0, m_a0, site_b0, m_b0 = ds.members[0]
    _g = ds.ego_graph
    _nn = _g.number_of_nodes() if _g is not None else -1
    _ne = _g.number_of_edges() if _g is not None else -1
    print(f"  diff_iso {ds.iso_class:2d}  "
          f"members={len(ds.members):<4d}  "
          f"ego(n={_nn}, e={_ne})  "
          f"representative: (ads_iso={site_a0.iso_class}, m={m_a0}) "
          f"↔ (ads_iso={site_b0.iso_class}, m={m_b0})")

if not diff_sites_flat:
    raise RuntimeError(
        "No diffusion iso-classes found — check max_hops / slab size."
    )

# %% ── 7b. Plot the ego-graph of every unique DiffusionSite ────────────────
_section("STAGE 5b — DiffusionSite ego-graph visualisation")

from plotly.subplots import make_subplots

_n = len(diff_sites_flat)
_cols = min(3, _n)
_rows = (_n + _cols - 1) // _cols

fig_egos = make_subplots(
    rows=_rows, cols=_cols,
    subplot_titles=[
        (f"diff_iso {ds.iso_class}  "
         f"(ads {ds.members[0][0].iso_class}↔{ds.members[0][2].iso_class}, "
         f"members={len(ds.members)})")
        for ds in diff_sites_flat
    ],
    horizontal_spacing=0.04, vertical_spacing=0.08,
)

def _node_xy(g, nid):
    p = g.nodes[nid].get("position")
    if p is None:
        return 0.0, 0.0
    return float(p[0]), float(p[1])

for _idx, ds in enumerate(diff_sites_flat):
    g = ds.ego_graph
    if g is None:
        continue
    r = _idx // _cols + 1
    c = _idx %  _cols + 1

    # Edges — split by kind so they can be styled.
    edge_kinds = {
        "surface":       dict(color="#888888", width=1.0, dash="solid"),
        "anchor_bond":   dict(color="#1F77B4", width=2.0, dash="solid"),
        "intra_adsorbate": dict(color="#2CA02C", width=2.0, dash="dot"),
    }
    seg_x = {k: [] for k in edge_kinds}
    seg_y = {k: [] for k in edge_kinds}
    for u, v, ed in g.edges(data=True):
        if ed.get("anchor_bond"):
            kind = "anchor_bond"
        elif ed.get("intra_adsorbate"):
            kind = "intra_adsorbate"
        else:
            kind = "surface"
        x0, y0 = _node_xy(g, u)
        x1, y1 = _node_xy(g, v)
        # MIC-style guard: skip edges that wrap across the cell so the plot
        # doesn't grow long stretched lines (purely cosmetic).
        if abs(x1 - x0) > 8.0 or abs(y1 - y0) > 8.0:
            continue
        seg_x[kind] += [x0, x1, None]
        seg_y[kind] += [y0, y1, None]

    for kind, style in edge_kinds.items():
        if not seg_x[kind]:
            continue
        fig_egos.add_trace(go.Scatter(
            x=seg_x[kind], y=seg_y[kind], mode="lines",
            line=style, hoverinfo="skip", showlegend=(_idx == 0), name=kind,
        ), row=r, col=c)

    # Nodes — colour / size by type+role.
    surf_x, surf_y, surf_text = [], [], []
    ep_x, ep_y, ep_text, ep_col = [], [], [], []
    for n, d in g.nodes(data=True):
        x, y = _node_xy(g, n)
        if d.get("type") == "surface":
            surf_x.append(x); surf_y.append(y)
            surf_text.append(f"surf #{n} {d.get('element','?')}")
        else:
            ep_x.append(x); ep_y.append(y)
            ep_text.append(
                f"{d.get('element','?')} #{n} "
                f"iso={d.get('iso_class','?')} "
                f"role={d.get('endpoint_role','?')}"
            )
            ep_col.append("#E74C3C" if d.get("element") == "C" else "#F1C40F")

    fig_egos.add_trace(go.Scatter(
        x=surf_x, y=surf_y, mode="markers",
        marker=dict(size=10, color="#BDC3C7", line=dict(width=0.5, color="#555")),
        text=surf_text, hoverinfo="text",
        showlegend=(_idx == 0), name="surface",
    ), row=r, col=c)
    fig_egos.add_trace(go.Scatter(
        x=ep_x, y=ep_y, mode="markers",
        marker=dict(size=14, color=ep_col, symbol="diamond",
                    line=dict(width=1, color="black")),
        text=ep_text, hoverinfo="text",
        showlegend=(_idx == 0), name="endpoint",
    ), row=r, col=c)

    fig_egos.update_xaxes(scaleanchor=f"y{_idx + 1 if _idx else ''}",
                          scaleratio=1.0, row=r, col=c, showgrid=False,
                          zeroline=False, visible=False)
    fig_egos.update_yaxes(showgrid=False, zeroline=False, visible=False,
                          row=r, col=c)

fig_egos.update_layout(
    title=f"DiffusionSite ego-graphs — {_n} iso-class(es)",
    height=320 * _rows, width=380 * _cols,
    template="plotly_white",
    margin=dict(l=20, r=20, t=70, b=20),
)
fig_egos.show()

# %% ── 8. Lateral classification for diff_iso 0, member 0 ───────────────────
_section("STAGE 6 — check_diffusion_site_lateral  (diff_iso=0, member=0)")

TARGET_DS  = diff_sites_flat[0]
TARGET_M   = 0   # change to inspect a different hop-pair member

print(f"diff_iso={TARGET_DS.iso_class}  member={TARGET_M}")
site_a, m_a, site_b, m_b = TARGET_DS.members[TARGET_M]
a_nids, b_nids = TARGET_DS.member_node_ids[TARGET_M]
print(f"  endpoint A: ads_iso={site_a.iso_class}  m={m_a}  "
      f"nodes={list(a_nids)[:4]}{'…' if len(a_nids) > 4 else ''}")
print(f"  endpoint B: ads_iso={site_b.iso_class}  m={m_b}  "
      f"nodes={list(b_nids)[:4]}{'…' if len(b_nids) > 4 else ''}")

lc = check_diffusion_site_lateral(G, TARGET_DS, TARGET_M)
print(f"  → lateral_class={lc.lateral_class}   "
      f"ego nodes={lc.ego_graph.number_of_nodes() if lc.ego_graph else 'n/a'}   "
      f"stable={lc.stable}")

# %% ── 9. Endpoint relaxation + NEB for lateral class 0 ─────────────────────
_section("STAGE 7 — check_diffusion_stability  (CI-NEB)")

# Artificially mark endpoint A occupied and B empty so the NEB geometry
# makes physical sense (the stability check does not depend on current
# occupancy — it relaxes both configurations independently).
print(f"Running NEB: {NEB_N_IMAGES} intermediate images, "
      f"fmax={FMAX_NEB} eV/Å, climb={NEB_CLIMB}, interp={NEB_INTERP}")
print("(This is often the slow step — watch for per-step output below)")

try:
    E_a, E_b, E_ts = check_diffusion_stability(
        G, TARGET_DS, TARGET_M, lc, calc,
        frozen_indices   = frozen_indices,
        fmax             = FMAX_NEB,
        max_steps        = MAX_OPT_STEPS,
        n_images         = NEB_N_IMAGES,
        climb            = NEB_CLIMB,
        spring_k         = NEB_SPRING_K,
        interpolation    = NEB_INTERP,
        persist_neb_path = True,   # keep all images for plotting below
        verbose          = True,
    )
except DiffusionStabilityError as _neb_err:
    print(f"\n  ⚠  NEB failed: {type(_neb_err).__name__}: {_neb_err}")
    # Use whatever partial values check_diffusion_stability managed to store
    # in lc before raising (endpoints and/or TS may already be populated).
    E_a   = lc.energy_a  if lc.energy_a  is not None else 0.0
    E_b   = lc.energy_b  if lc.energy_b  is not None else 0.0
    E_ts  = lc.energy_ts if lc.energy_ts is not None else (max(E_a, E_b) + 0.1)
    print(f"     → using partial results from lc: "
          f"E_a={E_a:+.4f}  E_b={E_b:+.4f}  E_ts={E_ts:+.4f} eV")

Ea_fwd = E_ts - E_a
Ea_rev = E_ts - E_b
print(f"\nEnergies:")
print(f"  E_a  = {E_a:+.4f} eV  (endpoint A occupied)")
print(f"  E_b  = {E_b:+.4f} eV  (endpoint B occupied)")
print(f"  E_ts = {E_ts:+.4f} eV  (transition state)")
print(f"  ΔE (A→B)    = {E_b - E_a:+.4f} eV")
print(f"  Ea fwd (raw)= {Ea_fwd:+.4f} eV  "
      f"→ KMC uses max(0.1, {Ea_fwd:.4f}) = {max(0.1, Ea_fwd):.4f} eV")
print(f"  Ea rev (raw)= {Ea_rev:+.4f} eV  "
      f"→ KMC uses max(0.1, {Ea_rev:.4f}) = {max(0.1, Ea_rev):.4f} eV")

# Write the two optimised endpoint geometries to disk so they can be
# inspected in VESTA / OVITO without running again.
from ase.io import write as ase_write
from pathlib import Path

_out = Path("diffusion_debug")
_out.mkdir(exist_ok=True)
if lc.atoms_a is not None:
    ase_write(_out / "endpoint_a.extxyz", lc.atoms_a, format="extxyz")
    print(f"  Written {_out / 'endpoint_a.extxyz'}")
if lc.atoms_b is not None:
    ase_write(_out / "endpoint_b.extxyz", lc.atoms_b, format="extxyz")
    print(f"  Written {_out / 'endpoint_b.extxyz'}")
if lc.atoms_ts is not None:
    ase_write(_out / "ts.extxyz", lc.atoms_ts, format="extxyz")
    print(f"  Written {_out / 'ts.extxyz'}")

# %% ── 10. Plot the NEB energy profile ─────────────────────────────────────
_section("STAGE 8 — NEB energy profile")

neb_images = lc.atoms_neb_path        # set by persist_neb_path=True
neb_energies = lc.neb_path_energies  # captured before images lost their calculators
if neb_images and neb_energies:
    image_energies = np.array(neb_energies)
    image_energies -= image_energies[0]   # shift so A = 0

    # Also write the full band as a multi-frame extxyz.
    ase_write(_out / "neb_path.extxyz", neb_images, format="extxyz")
    print(f"  Written {_out / 'neb_path.extxyz'}  ({len(neb_images)} frames)")

    # Write each image as a separate file for easy per-frame inspection.
    _frames_dir = _out / "neb_frames"
    _frames_dir.mkdir(exist_ok=True)
    for _i, _im in enumerate(neb_images):
        _label = (
            "A"   if _i == 0 else
            "B"   if _i == len(neb_images) - 1 else
            f"{_i:02d}"
        )
        _fname = _frames_dir / f"image_{_label}.extxyz"
        ase_write(_fname, _im, format="extxyz")
    print(f"  Written {len(neb_images)} individual frames → {_frames_dir}/")
    reaction_coords = np.linspace(0, 1, len(image_energies))

    fig_neb = go.Figure()
    fig_neb.add_trace(go.Scatter(
        x=reaction_coords, y=image_energies * 1000,
        mode="lines+markers",
        marker=dict(size=8, color=[
            "#2ECC71" if i == 0 else
            "#E74C3C" if i == len(image_energies) - 1 else
            "#F39C12" if image_energies[i] == image_energies[1:-1].max() else
            "#4A90D9"
            for i in range(len(image_energies))
        ]),
        line=dict(color="#4A90D9", width=2),
        text=[
            f"Image {i}<br>ΔE = {e*1000:.1f} meV"
            for i, e in enumerate(image_energies)
        ],
        hoverinfo="text",
    ))
    fig_neb.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig_neb.update_layout(
        title=dict(
            text=(f"NEB energy profile — diff_iso={TARGET_DS.iso_class}  "
                  f"m={TARGET_M}  lat={lc.lateral_class}<br>"
                  f"Ea(fwd)={max(0.1, Ea_fwd)*1000:.0f} meV  "
                  f"Ea(rev)={max(0.1, Ea_rev)*1000:.0f} meV  "
                  f"(KMC floor = 100 meV)"),
            font=dict(size=13),
        ),
        xaxis_title="Reaction coordinate",
        yaxis_title="Energy relative to A (meV)",
        template="plotly_white",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig_neb.show()
else:
    print("No NEB path stored (neb_images or neb_energies missing) — "
          "set persist_neb_path=True to capture it.")

# %% ── 11. Short KMC run with diffusion enabled ──────────────────────────────
_section("STAGE 9 — KMC loop (adsorption ⇌ desorption ⇌ diffusion)")

# Start from a partially occupied surface so hops fire from the first step.
# Occupy the first member of the first adsorbate iso-class.
if adsorbate_sites and adsorbate_sites[0].member_node_ids:
    first_nids = adsorbate_sites[0].member_node_ids[0]
    for nid in first_nids:
        if nid in G:
            G.nodes[nid]["occupied"] = True
    print(f"Pre-occupied: iso=0, member=0 (node ids {list(first_nids)[:2]}…)")

diffusion_kwargs = dict(
    fmax             = FMAX_NEB,
    max_steps        = MAX_OPT_STEPS,
    n_images         = NEB_N_IMAGES,
    climb            = NEB_CLIMB,
    spring_k         = NEB_SPRING_K,
    interpolation    = NEB_INTERP,
    persist_neb_path = False,
)

summary = run_kmc_steps(
    G, adsorbate_sites, calc,
    reactants                = co,
    temperature              = TEMPERATURE_K,
    n_steps                  = N_KMC_STEPS,
    transmission_coefficient = 1.0,
    frozen_indices           = frozen_indices,
    fmax                     = FMAX_ADS,
    max_steps                = MAX_OPT_STEPS,
    rng                      = RANDOM_SEED,
    log_every                = 1,
    verbose                  = True,
    diffusion_sites          = diff_sites_flat,
    diffusion_kwargs         = diffusion_kwargs,
)

print("\nKMC summary:")
print(f"  steps executed  : {summary['steps_executed']}")
print(f"  total time      : {summary['time']:.4e} s")
print(f"  reaction counts : {summary['reaction_counts']}")
print(f"  occupancy/iso   : {summary['final_occupancy']}")

# %% ── 12. Reaction history ──────────────────────────────────────────────────
_section("STAGE 10 — Reaction history")

hist = summary["history"]
print(f"  {'step':>4}  {'time / s':>11}  {'kind':<11}  "
      f"{'iso':>3}  {'m':>3}  {'lat':>3}  "
      f"{'ΔE / eV':>9}  {'Ea / eV':>8}  {'rate / Hz':>11}")
print("  " + "─" * 79)
for (step, t, kind, iso, m, lat, dE, Ea, rate) in hist:
    arrow = "↓" if dE < 0 else "↑"
    tag   = "↔" if kind == "diffusion" else ("↓" if kind == "adsorption" else "↑")
    print(f"  {step:>4}  {t:>11.3e}  {kind:<11}  "
          f"{iso:>3}  {m:>3}  {lat:>3}  "
          f"{dE:>+9.4f}{arrow}  {Ea:>8.4f}  {rate:>11.3e}  {tag}")

diffusion_steps = [row for row in hist if row[2] == "diffusion"]
print(f"\nDiffusion events fired: {len(diffusion_steps)} / {len(hist)}")
if diffusion_steps:
    print("  First diffusion event:")
    step, t, kind, iso, m, lat, dE, Ea, rate = diffusion_steps[0]
    print(f"    step={step}  t={t:.3e} s  diff_iso={iso}  m={m}  "
          f"lat={lat}  ΔE={dE:+.4f} eV  Ea={Ea:.4f} eV  k={rate:.3e} Hz")

print("\nDone.")

