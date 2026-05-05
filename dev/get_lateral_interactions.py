# %% [markdown]
# # AutoKMC — Adsorption / Desorption KMC Loop
#
# Demonstrates a minimal on-the-fly KMC workflow for CO on Cu(111):
#
# 1. **Build** a Cu(111) FCC slab and connectivity graph.
# 2. **Find** all stable CO adsorbate sites — `find_adsorbate_sites` with
#    `prune_stable_only=True` runs the full pipeline internally:
#    - Stage A  : enumerate placements → materialise nodes on G
#    - Stage B-1: calc-free rigid-body optimisation + Kabsch propagation
#    - Stage B-2: ML-potential stability check per iso-class; prune unstable;
#                 update representative positions; propagate to all members
#    Only iso-classes that are stable in isolation survive to the KMC loop,
#    which greatly reduces the number of sites and lateral-class checks.
# 3. **KMC loop**:
#    - For every member of every stable iso-class, classify its lateral
#      environment on-the-fly with `check_adsorbate_site_lateral`.
#    - If the lateral class (neighbour occupancy pattern) has not yet been
#      stability-checked, run `check_site_stability` to get E_occ / E_unocc.
#    - Build reaction list: unoccupied stable → adsorption;
#      occupied stable → desorption.
#    - Choose a reaction (Boltzmann-weighted) and execute it.
#    - Repeat.

# %% ── 0. Imports ────────────────────────────────────────────────────────────
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from nequip.ase import NequIPCalculator

from autokmc.structure import build_surface, find_surface_atoms
from autokmc.core.graph import build_graph
from autokmc.species.reactant import build_reactant
from autokmc.sites.adsorbate import (
    AdsorbateSite,
    AdsorbateSiteLateral,
    find_adsorbate_sites,
)
from autokmc.sites.stability.adsorption import (
    check_adsorbate_site_lateral,
    check_site_stability,
    SiteStabilityError,
)

# %% ── 1. Calculator setup ───────────────────────────────────────────────────
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_FILE = "asehcocuau.nequip.pt2" if _DEVICE == "cuda" else "cpuhcocuau.nequip.pth"
_MODEL_PATH = str(
    (Path(__file__).resolve().parent if "__file__" in globals() else Path(""))
    / _MODEL_FILE
)
print(f"Device : {_DEVICE}   Model : {_MODEL_FILE}")


def make_calc():
    """Return a fresh NequIPCalculator loaded from the model file."""
    return NequIPCalculator.from_compiled_model(
        compile_path=_MODEL_PATH,
        device=_DEVICE,
    )


calc = make_calc()

# %% ── 2. KMC parameters ──────────────────────────────────────────────────────
TEMPERATURE_K : float = 500.0   # simulation temperature (K)
MAX_STEPS     : int   = 30      # maximum KMC steps before stopping
FMAX          : float = 0.05    # force convergence for ML relaxation (eV/Å)
MAX_OPT_STEPS : int   = 500     # max LBFGS steps per relaxation
RANDOM_SEED   : int   = 69      # reproducibility

_KB_EV = 8.617_333_262e-5       # Boltzmann constant in eV / K
_kT    = _KB_EV * TEMPERATURE_K

rng = random.Random(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

_BAR  = "─" * 64

def _section(title: str) -> None:
    print(f"\n{_BAR}\n  {title}\n{_BAR}")


# %% ── 3. Build surface ───────────────────────────────────────────────────────
_section("STAGE 1 — Build Cu(111) FCC slab")

atoms = build_surface(
    composition      = "Cu",
    crystal_structure= "fcc",
    miller_index     = (1, 1, 1),
    lattice_constant = 3.615,
    min_slab_size    = 10.0,
    min_vacuum_size  = 12.0,
    goal_x           = 16.0,
    goal_y           = 16.0,
    n_freeze_layers  = 2,
    calculator       = calc,
    verbose          = True,
)
print(f"Slab: {len(atoms)} atoms   cell diag: {np.diag(atoms.get_cell()).round(3)} Å")

# Preserve frozen indices for the stability calculations.
frozen_indices: list[int] | None = list(atoms.info.get("frozen_indices", []) or [])
if not frozen_indices:
    frozen_indices = None

# %% ── 4. Surface atoms + graph ───────────────────────────────────────────────
_section("STAGE 2 — Surface atoms & graph")

find_surface_atoms(atoms, tag_atoms=True)
G = build_graph(atoms)
print(f"Graph: {G.number_of_nodes()} nodes   {G.number_of_edges()} edges")

# %% ── 5. Reactant ────────────────────────────────────────────────────────────
_section("STAGE 3 — CO reactant")

co = build_reactant("[C-]#[O+]", add_hydrogens=False)
print(f"Formula: {co.atoms.get_chemical_formula()}   "
      f"anchor atoms: {co.anchor_atoms}   "
      f"bond length: {co.atoms.get_all_distances()[0, 1]:.3f} Å")

# %% ── 6. Adsorbate sites (with ML stability pruning) ────────────────────────
_section("STAGE 4 — Adsorbate sites  [enumerate → geom opt → ML prune → propagate]")

# `find_adsorbate_sites` with prune_stable_only=True runs the full pipeline:
#   A  : enumerate all placements → materialise nodes on G
#   B-1: calc-free rigid-body optimisation → Kabsch propagation to all members
#   B-2: ML-relax representative per iso-class → prune unstable →
#        update ms.positions from relaxed Atoms → Kabsch-propagate to members
#
# Anchor sites are computed lazily inside find_adsorbate_sites.
# Only iso-classes that are stable in isolation reach the KMC loop.
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


# %% ── 7. Reaction dataclass ──────────────────────────────────────────────────

@dataclass
class Reaction:
    """One possible adsorption or desorption event.

    Attributes
    ----------
    kind : str
        ``"adsorption"`` or ``"desorption"``.
    site : AdsorbateSite
        Parent iso-class.
    member_index : int
        Index into ``site.member_node_ids``.
    lateral_class : AdsorbateSiteLateral
        Lateral environment under which the energies were computed.
    delta_e : float
        E_final − E_initial (eV).
        Adsorption: E_occ − E_unocc (negative = exothermic).
        Desorption: E_unocc − E_occ (positive = endothermic).
    rate : float
        Boltzmann-suppressed rate proxy:
        ``exp(−max(delta_e, 0) / kT)``.
    """
    kind          : str
    site          : AdsorbateSite
    member_index  : int
    lateral_class : AdsorbateSiteLateral
    delta_e       : float
    rate          : float


# %% ── 9. Core helpers ────────────────────────────────────────────────────────

def _set_occupied(G, site: AdsorbateSite, member_index: int, value: bool) -> None:
    """Toggle the ``occupied`` flag on all nodes of one member."""
    for nid in site.member_node_ids[member_index]:
        if nid in G:
            G.nodes[nid]["occupied"] = value


def _is_occupied(G, site: AdsorbateSite, member_index: int) -> bool:
    """Return True if any node of this member is marked occupied on G."""
    return site._member_is_occupied(G, site.member_node_ids[member_index])


def _is_clique_blocked(G, site: AdsorbateSite, member_index: int) -> bool:
    """Return True if any OTHER occupied adsorbate shares an exact bonding clique.

    Two adsorbate atoms that bind to exactly the same set of surface atoms
    cannot physically co-exist.  If the current member's bonding clique(s) are
    already claimed by an occupied neighbour, the site is blocked and neither
    adsorption (for an empty site) nor any lateral-class classification should
    be attempted.

    This check is O(adsorbate nodes on G) and is very cheap compared with an
    ML relaxation, so it is always applied before ``check_adsorbate_site_lateral``.
    """
    node_ids      = site.member_node_ids[member_index]
    member_id_set = frozenset(node_ids)

    # Collect the surface cliques bonded by this member's atoms.
    member_cliques: set[frozenset] = set()
    for nid in node_ids:
        if nid not in G:
            continue
        clq = G.nodes[nid].get("clique")
        if clq is not None:
            member_cliques.add(frozenset(clq))

    if not member_cliques:
        return False

    # Scan every other occupied adsorbate node for a clique collision.
    for n, d in G.nodes(data=True):
        if n in member_id_set:
            continue
        if d.get("type") != "adsorbate":
            continue
        if not d.get("occupied", False):
            continue
        clq = d.get("clique")
        if clq is not None and frozenset(clq) in member_cliques:
            return True

    return False


def get_possible_reactions(
    G,
    adsorbate_sites: list[AdsorbateSite],
    calculator,
    *,
    frozen_indices: list[int] | None = None,
    fmax: float = FMAX,
    max_steps: int = MAX_OPT_STEPS,
    verbose: bool = False,
) -> list[Reaction]:
    """Enumerate all stable adsorption / desorption events for the current G.

    For every member of every :class:`AdsorbateSite`:

    1. Calls :func:`check_adsorbate_site_lateral` to get (or retrieve) the
       lateral-interaction class for the current neighbour occupancy pattern.
    2. If the lateral class has not yet been stability-checked
       (``lc.stable is None``), calls :func:`check_site_stability` with the
       provided *calculator*.  On failure the lateral class is marked
       ``stable=False`` and skipped.
    3. Builds a :class:`Reaction` for each stable member:
       unoccupied → adsorption candidate, occupied → desorption candidate.

    Parameters
    ----------
    G : nx.Graph
    adsorbate_sites : list[AdsorbateSite]
    calculator : ASE calculator
        Used only for newly-encountered (unchecked) lateral classes.
    frozen_indices : list[int] | None
        Passed through to :func:`check_site_stability`.
    fmax, max_steps :
        Relaxation parameters forwarded to :func:`check_site_stability`.
    verbose : bool

    Returns
    -------
    list[Reaction]
        Sorted by ``abs(delta_e)`` descending (most energetically significant
        first) within each kind.
    """
    reactions: list[Reaction] = []
    n_new_checks = 0
    n_unstable   = 0
    n_blocked    = 0

    for site in adsorbate_sites:
        for m_idx in range(len(site.member_node_ids)):
            # ── Clique-collision guard ───────────────────────────────────
            # If any other occupied adsorbate is bonded to the same surface
            # atom(s) as this member, the site is physically blocked — skip
            # it immediately without any lateral classification or ML work.
            if _is_clique_blocked(G, site, m_idx):
                n_blocked += 1
                if verbose:
                    print(f"  ⛔ iso={site.iso_class} m={m_idx}: "
                          f"clique blocked by occupied neighbour — skipped")
                continue

            # ── Lateral classification ───────────────────────────────────
            try:
                lc = check_adsorbate_site_lateral(G, site, m_idx)
            except (ValueError, IndexError) as exc:
                if verbose:
                    print(f"  ⚠  iso={site.iso_class} m={m_idx}: "
                          f"lateral check skipped ({exc})")
                continue

            # ── Stability check (only for new lateral classes) ───────────
            if lc.stable is None:
                n_new_checks += 1
                try:
                    check_site_stability(
                        G, site, m_idx, lc, calculator,
                        frozen_indices = frozen_indices,
                        fmax           = fmax,
                        max_steps      = max_steps,
                        verbose        = verbose,
                    )
                except SiteStabilityError as exc:
                    lc.stable = False
                    n_unstable += 1
                    if verbose:
                        print(f"  ✗  iso={site.iso_class} m={m_idx} "
                              f"lat={lc.lateral_class}: "
                              f"{type(exc).__name__}: {exc}")
                    continue

            if not lc.stable:
                continue

            # Guard: energies must be present (set by check_site_stability).
            if lc.energy_occupied is None or lc.energy_unoccupied is None:
                continue

            # ── Build reaction ────────────────────────────────────────────
            occ = _is_occupied(G, site, m_idx)
            if occ:
                delta_e = lc.energy_unoccupied - lc.energy_occupied
                kind    = "desorption"
            else:
                delta_e = lc.energy_occupied - lc.energy_unoccupied
                kind    = "adsorption"

            rate = float(np.exp(-max(delta_e, 0.0) / _kT))
            reactions.append(Reaction(kind, site, m_idx, lc, delta_e, rate))

    if verbose and (n_new_checks or n_blocked):
        print(f"  → {n_new_checks} new lateral classes checked  "
              f"({n_unstable} unstable)  {n_blocked} site(s) clique-blocked")

    # Sort: adsorption first (most exothermic first), then desorption
    reactions.sort(key=lambda r: (r.kind != "adsorption", r.delta_e))
    return reactions


def execute_reaction(G, reaction: Reaction) -> None:
    """Toggle the occupied state of a site member on *G*."""
    new_state = (reaction.kind == "adsorption")
    _set_occupied(G, reaction.site, reaction.member_index, new_state)


def choose_reaction(reactions: list[Reaction], rng: random.Random) -> Reaction:
    """Pick one reaction with Boltzmann-weighted probability.

    The rate of each reaction is ``exp(−max(ΔE, 0) / kT)``, making
    downhill events equally likely and suppressing uphill ones.
    """
    weights = [r.rate for r in reactions]
    total   = sum(weights)
    pick    = rng.random() * total
    cumsum  = 0.0
    for r, w in zip(reactions, weights):
        cumsum += w
        if cumsum >= pick:
            return r
    return reactions[-1]


def print_reactions(reactions: list[Reaction], *, max_show: int = 10) -> None:
    """Pretty-print the reaction list."""
    n_ads  = sum(1 for r in reactions if r.kind == "adsorption")
    n_des  = sum(1 for r in reactions if r.kind == "desorption")
    print(f"\n  {len(reactions)} reaction(s): "
          f"{n_ads} adsorption, {n_des} desorption")
    header = (f"  {'#':>3}  {'kind':<12}  {'iso':>3}  {'member':>6}  "
              f"{'lat':>3}  {'ΔE /eV':>9}  {'rate':>8}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, r in enumerate(reactions[:max_show]):
        arrow = "↓" if r.delta_e < 0 else "↑"
        print(
            f"  {i:>3}  {r.kind:<12}  {r.site.iso_class:>3}  "
            f"{r.member_index:>6}  {r.lateral_class.lateral_class:>3}  "
            f"{r.delta_e:>+9.4f}{arrow}  {r.rate:>8.4f}"
        )
    if len(reactions) > max_show:
        print(f"  … ({len(reactions) - max_show} more not shown)")


def print_occupancy_summary(G, adsorbate_sites: list[AdsorbateSite]) -> None:
    """Print a one-line occupancy count per iso-class."""
    print("\n  Occupancy state:")
    for site in adsorbate_sites:
        n_occ = len(site.occupied_member_indices(G))
        n_tot = len(site.members)
        bar   = "█" * n_occ + "░" * (n_tot - n_occ)
        print(f"    iso {site.iso_class:2d}  [{bar}]  "
              f"{n_occ}/{n_tot} occupied")


# %% ── 10. KMC loop ───────────────────────────────────────────────────────────
_section("KMC LOOP")

print(f"  T = {TEMPERATURE_K} K   kT = {_kT*1000:.2f} meV   "
      f"max steps = {MAX_STEPS}")
print(f"  Stability check: fmax={FMAX} eV/Å   max_opt={MAX_OPT_STEPS} steps")
print(f"\n  Initial state: all {total_members} sites unoccupied.")

print_occupancy_summary(G, adsorbate_sites)

for step in range(1, MAX_STEPS + 1):
    print(f"\n{'─'*64}")
    print(f"  Step {step} / {MAX_STEPS}")

    reactions = get_possible_reactions(
        G, adsorbate_sites, calc,
        frozen_indices = frozen_indices,
        fmax           = FMAX,
        max_steps      = MAX_OPT_STEPS,
        verbose        = False,
    )

    if not reactions:
        print("  No stable reactions available — stopping.")
        break

    print_reactions(reactions)

    chosen = choose_reaction(reactions, rng)
    execute_reaction(G, chosen)

    n_occ_total = sum(
        len(s.occupied_member_indices(G)) for s in adsorbate_sites
    )
    print(
        f"\n  ► Executed: {chosen.kind.upper()}  "
        f"iso={chosen.site.iso_class}  member={chosen.member_index}  "
        f"lat={chosen.lateral_class.lateral_class}  "
        f"ΔE={chosen.delta_e:+.4f} eV"
        f"\n  ► Total occupied sites: {n_occ_total}"
    )

    print_occupancy_summary(G, adsorbate_sites)

else:
    print(f"\n  Reached max steps ({MAX_STEPS}).")


# %% ── 11. Final summary ──────────────────────────────────────────────────────
_section("SUMMARY")

all_lateral: list[tuple] = []
for site in adsorbate_sites:
    for lc in site.lateral_classes:
        all_lateral.append((site.iso_class, lc))

print(f"  Lateral classes discovered : {len(all_lateral)}")
for iso_class, lc in sorted(all_lateral, key=lambda x: (x[0], x[1].lateral_class)):
    e_occ  = f"{lc.energy_occupied:.4f}"   if lc.energy_occupied   is not None else "  (not checked)"
    e_unocc= f"{lc.energy_unoccupied:.4f}" if lc.energy_unoccupied is not None else "  (not checked)"
    stable = {True: "✓", False: "✗", None: "?"}[lc.stable]
    print(f"  iso={iso_class:2d}  lat={lc.lateral_class:2d}  "
          f"{stable}  E_occ={e_occ}  E_unocc={e_unocc} eV  "
          f"members={len(lc.members)}")

n_occ_final = sum(len(s.occupied_member_indices(G)) for s in adsorbate_sites)
print(f"\n  Final occupied sites : {n_occ_final} / {total_members}")
print("\nDone.")


