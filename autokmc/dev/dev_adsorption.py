"""
dev_adsorption.py
=================

End-to-end example: H, C, O on Cu(111) with NequIP calculator.

Demonstrates the three-step workflow:
1. Find sites – geometry only, no calculator.
2. Classify – group into symmetry-equivalent classes via graph isomorphism.
3. Optimize + check connectivity – relax each unique class and detect migrations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))

from collections import Counter
from autokmc.structure import build_surface
from autokmc.surface import find_surface_atoms
from autokmc.graph import build_graph
from autokmc.default_sites import (
    find_sites_for_element,
    reduce_sites_by_isomorphism,
    optimise_site_positions,
)
from autokmc.adsorbate import (
    optimise_unique_sites,
    SiteOptResult,
    ConnectivityStatus,
)

# =============================================================================
# Load the Allegro/NequIP calculator
# =============================================================================

import torch
from nequip.ase import NequIPCalculator

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__) if "__file__" in dir() else ".", _MODEL_FILE
)
print(f"Device : {_DEVICE}")
print(f"Model  : {_MODEL_FILE}")


def make_calc():
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )


calc = make_calc()
print(f"Calculator ready: {calc.__class__.__name__}")

# =============================================================================
# Build a Cu(111) slab
# =============================================================================

slab = build_surface(
    composition="Cu",
    crystal_structure="fcc",
    miller_index=(1, 1, 1),
    calculator=make_calc(),
    min_slab_size=8.0,
    min_vacuum_size=12.0,
    goal_x=12.0,
    goal_y=12.0,
    n_freeze_layers=2,
    verbose=True,
    orthogonalise=True,
)

print(f"\nFormula : {slab.get_chemical_formula()}")
print(f"Atoms   : {len(slab)}")
cell = slab.get_cell()
print(f"Cell    : a={cell[0,0]:.3f}  b={cell[1,1]:.3f}  c={cell[2,2]:.3f} Å")

# =============================================================================
# Identify surface atoms and build the graph
# =============================================================================

surface_mask, surface_indices, method = find_surface_atoms(
    slab, which="top", tag_atoms=True
)
print(f"Surface detection : {method}")
print(f"Surface atoms     : {surface_mask.sum()} / {len(slab)}")

graph = build_graph(slab)
print(f"Graph             : {graph.number_of_nodes()} nodes, "
      f"{graph.number_of_edges()} edges")
type_counts = Counter(d["type"] for _, d in graph.nodes(data=True))
for t, n in sorted(type_counts.items()):
    print(f"  {t:10s} : {n}")

# =============================================================================
# Find sites, reduce to iso-classes, and pre-optimize positions
# =============================================================================
# These are the prerequisites for site_opt.optimise_unique_sites.

_N_SHELLS = 1

for element in ["H", "C", "O"]:
    find_sites_for_element(graph, element, verbose=True)
    reduce_sites_by_isomorphism(graph, element, n_shells=_N_SHELLS, verbose=True)
    optimise_site_positions(graph, element, verbose=True)
    print()

# =============================================================================
# Summary of unique sites before calculator optimization
# =============================================================================

print(f"{'':=<55}")
print(f"  Unique sites at n_shells={_N_SHELLS}")
print(f"{'':=<55}")
_LABELS = {1: "top", 2: "bridge", 3: "hollow"}
for element in ["H", "C", "O"]:
    unique = graph.graph["unique_sites"][element][_N_SHELLS]
    total = sum(len(v) for v in unique.values())
    print(f"\n  {element}  ({total} unique classes)")
    for k, classes in sorted(unique.items()):
        label = _LABELS.get(k, f"{k}-fold")
        print(f"    k={k}  {label:8s}  {len(classes)} classes")

# =============================================================================
# H on Cu(111)
# =============================================================================

print("\n" + "=" * 75)
print("H on Cu(111)")
print("=" * 75)

results_H = optimise_unique_sites(
    graph,
    element="H",
    atoms=slab,
    calculator=calc,
    n_shells=_N_SHELLS,
    bond_factor=1.1,
    fmax=0.05,
    steps=500,
    verbose=True,
)

# Analyse H results
_stat_labels = {
    ConnectivityStatus.OK: "OK",
    ConnectivityStatus.LOST_BOND: "lost bond",
    ConnectivityStatus.NEW_BOND: "new bond",
    ConnectivityStatus.MIGRATED: "migrated",
}
print(f"\n{'H on Cu(111)':=<55}")
for r in results_H:
    label = _LABELS.get(r.iso_class.k, f"{r.iso_class.k}-fold")
    print(
        f"  k={r.iso_class.k} ({label:6s})  cls={r.iso_class.iso_class}"
        f"  E={r.energy:>10.4f} eV"
        f"  disp={r.displacement:.3f} Å"
        f"  conv={'✓' if r.converged else '✗'}"
        f"  [{_stat_labels[r.connectivity]}]"
    )

# =============================================================================
# DEBUG: Visualize H site positions pre- and post-optimization
# =============================================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

_ELEM_COLOURS = {
    "Cu": "#B87333", "Au": "#FFD700", "Ag": "#C0C0C0",
    "H": "#FFFFFF", "C": "#202020", "O": "#FF4444",
}

# Create comparison plots for each H result
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=("Pre-optimization", "Post-optimization"),
    horizontal_spacing=0.05,
)

if results_H:
    r = results_H[0]  # First result for debugging

    # Pre-optimization structure
    pre_atoms = r.atoms_initial
    pre_pos = pre_atoms.get_positions()

    # Post-optimization structure
    post_atoms = r.atoms_final
    post_pos = post_atoms.get_positions()

    # Group by symbol for pre
    pre_by_sym = defaultdict(list)
    for i, atom in enumerate(pre_atoms):
        pre_by_sym[atom.symbol].append(pre_pos[i])

    # Group by symbol for post
    post_by_sym = defaultdict(list)
    for i, atom in enumerate(post_atoms):
        post_by_sym[atom.symbol].append(post_pos[i])

    # Plot pre-optimization (left)
    for sym, positions in sorted(pre_by_sym.items()):
        arr = np.array(positions)
        size = 12 if sym == "H" else 6
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode="markers", name=sym,
            legendgroup=sym, showlegend=True,
            marker=dict(
                size=size,
                color=_ELEM_COLOURS.get(sym, "#AAAAAA"),
                opacity=0.9 if sym == "H" else 0.5,
                line=dict(
                    width=2 if sym == "H" else 0.5,
                    color="black",
                ),
            ),
        ), row=1, col=1)

    # Plot post-optimization (right)
    for sym, positions in sorted(post_by_sym.items()):
        arr = np.array(positions)
        size = 12 if sym == "H" else 6
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode="markers", name=sym,
            legendgroup=sym, showlegend=False,
            marker=dict(
                size=size,
                color=_ELEM_COLOURS.get(sym, "#AAAAAA"),
                opacity=0.9 if sym == "H" else 0.5,
                line=dict(
                    width=2 if sym == "H" else 0.5,
                    color="black",
                ),
            ),
        ), row=1, col=2)

    # Update layout
    camera = dict(eye=dict(x=0, y=-1.6, z=1.1))
    axis_style = dict(showbackground=False, showgrid=False,
                      zeroline=False, showticklabels=False)
    scene_cfg = dict(aspectmode="data", camera=camera,
                     xaxis=axis_style, yaxis=axis_style, zaxis=axis_style)

    fig.update_layout(
        title=dict(
            text=f"H site comparison (class {r.iso_class.iso_class}, k={r.iso_class.k})<br>"
                 f"E={r.energy:.4f} eV | disp={r.displacement:.3f} Å",
            x=0.5
        ),
        height=500, width=1000,
        scene=scene_cfg,
        scene2=scene_cfg,
    )

    fig.show()

    # Print displacement details
    h_idx = r.ads_index
    print(f"\nH atom displacement details:")
    print(f"  Pre-position  : {pre_pos[h_idx]}")
    print(f"  Post-position : {post_pos[h_idx]}")
    print(f"  Displacement  : {r.displacement:.4f} Å")
    print(f"  Connectivity  : {r.connectivity.value}")
    print(f"  Actual clique : {r.actual_clique}")

# =============================================================================
# C on Cu(111)
# =============================================================================

print("\n" + "=" * 75)
print("C on Cu(111)")
print("=" * 75)

results_C = optimise_unique_sites(
    graph,
    element="C",
    atoms=slab,
    calculator=calc,
    n_shells=_N_SHELLS,
    bond_factor=1.1,
    fmax=0.05,
    steps=500,
    verbose=True,
)

# Analyse C results
print(f"\n{'C on Cu(111)':=<55}")
for r in results_C:
    label = _LABELS.get(r.iso_class.k, f"{r.iso_class.k}-fold")
    print(
        f"  k={r.iso_class.k} ({label:6s})  cls={r.iso_class.iso_class}"
        f"  E={r.energy:>10.4f} eV"
        f"  disp={r.displacement:.3f} Å"
        f"  conv={'✓' if r.converged else '✗'}"
        f"  [{_stat_labels[r.connectivity]}]"
    )

# =============================================================================
# O on Cu(111)
# =============================================================================

print("\n" + "=" * 75)
print("O on Cu(111)")
print("=" * 75)

results_O = optimise_unique_sites(
    graph,
    element="O",
    atoms=slab,
    calculator=calc,
    n_shells=_N_SHELLS,
    bond_factor=1.1,
    fmax=0.05,
    steps=500,
    verbose=True,
)

# Analyse O results
print(f"\n{'O on Cu(111)':=<55}")
for r in results_O:
    label = _LABELS.get(r.iso_class.k, f"{r.iso_class.k}-fold")
    print(
        f"  k={r.iso_class.k} ({label:6s})  cls={r.iso_class.iso_class}"
        f"  E={r.energy:>10.4f} eV"
        f"  disp={r.displacement:.3f} Å"
        f"  conv={'✓' if r.converged else '✗'}"
        f"  [{_stat_labels[r.connectivity]}]"
    )

# =============================================================================
# COMBINED SUMMARY
# =============================================================================

print("\n" + "=" * 75)
print("COMBINED SUMMARY")
print("=" * 75)

import pandas as pd

rows = []
for element, results in [("H", results_H), ("C", results_C), ("O", results_O)]:
    for r in results:
        rows.append(
            {
                "element": element,
                "k": r.iso_class.k,
                "site_type": _LABELS.get(r.iso_class.k, f"{r.iso_class.k}-fold"),
                "iso_class": r.iso_class.iso_class,
                "energy_eV": r.energy,
                "disp_A": r.displacement,
                "converged": r.converged,
                "connectivity": r.connectivity.value,
            }
        )

df = pd.DataFrame(rows)
print("\nEnergy summary table across all elements:")
print(df.to_string(index=False))

# =============================================================================
# Connectivity summary
# =============================================================================

print(f"\n{'Connectivity summary':=<45}")
for element, results in [("H", results_H), ("C", results_C), ("O", results_O)]:
    counts = Counter(r.connectivity for r in results)
    print(f"\n  {element}:")
    for status, n in sorted(counts.items(), key=lambda x: x[0].value):
        print(f"    {status.value:12s} : {n}")

print("\n" + "=" * 75)
print("Done!")
print("=" * 75)

