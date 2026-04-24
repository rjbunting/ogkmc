# PLAN — Multi-atom adsorbate placement and discovery

## 1. Goal & scope

Extend `autokmc` from single-atom adsorbates (current `optimise_unique_sites` /
`register_adsorption` in `autokmc/adsorbate.py`) to **multi-atom molecular
adsorbates** specified by a SMILES string and built by
`reactants.build_reactant`. Each reactant is placed by snapping its
**convex-hull-exposed anchor atoms** onto the slab's per-element
`G.graph["unique_sites"][element][1]` cliques, validated by a clash filter,
relaxed with the calculator, iso-deduplicated at `n_shells = 1`, and made
available to the on-the-fly KMC discovery loop. **Out of scope**: gas-phase
chemistry, intramolecular bond formation/breaking on the surface (those are
KMC reaction events handled by `disreax_kmc`), `n_shells != 1` (raises
`NotImplementedError` in v1).

## 2. Definitions

- **Anchor atom** — a reactant atom that the convex hull marks as exposed
  and therefore eligible to bond to a slab clique.
- **Anchor orbit** — equivalence class of anchors under the reactant's
  intramolecular automorphism group (NetworkX `vf2.GraphMatcher`,
  element-matched). Used to deduplicate identical anchor permutations.
- **Configuration** — an assignment `{anchor_atom_idx → slab_clique}` for
  every anchor of a single reactant, plus the rigid-aligned positions of
  all non-anchor atoms.
- **Combined subgraph** — the `n_shells = 1` ego expansion around the union
  of all anchor cliques, plus *every* reactant node and intramolecular
  edge. Hashed for iso-deduplication (mirrors
  `default_sites.reduce_sites_by_isomorphism`, augmented per
  `dev/PLAN_adsorption_sites.md` §11).
- **Stable / reactive / migrated_to** — same semantics as the single-atom
  case (see `AdsorptionSite` docstring in `autokmc/adsorbate.py`),
  generalised per-anchor.

## 3. Reactant object — upgrades to `autokmc/reactants.py`

Current `Reactant` carries only `smiles`, `atoms`, `graph`. Extend the
dataclass with all data needed for downstream placement:

- `energy: float` — `atoms.get_potential_energy()` after the optional
  calculator relax in `_optimise` (skip / `nan` if no calculator passed).
- `unique_nodes: dict[str, list[list[int]]]` — orbits of intramolecular
  automorphisms keyed by element. Built by a new `find_unique_atoms`
  using `networkx.algorithms.isomorphism.GraphMatcher` on `Reactant.graph`
  with `node_match = categorical_node_match("element", "X")`; collect
  orbits as the union of images of each node under all automorphisms.
- `anchor_atoms: list[int]` — convex-hull-exposed atom indices computed
  by a new `find_anchor_atoms`.
- `anchor_orbit: dict[int, int]` — anchor index → orbit id, derived from
  `unique_nodes` (so equivalent anchors collapse during enumeration).

Algorithm for `find_anchor_atoms(reactant, hull_tol=0.1)`:
1. `pts = reactant.atoms.get_positions()`. Try
   `scipy.spatial.ConvexHull(pts)`; on `QHullError` (single atom, linear
   molecule, planar molecule with <4 atoms) return **all atom indices**
   as anchors (degenerate fallback).
2. Hull vertices are anchors. Additionally, an atom is exposed if its
   covalent-radius sphere protrudes through any hull facet — for each
   non-vertex atom `i`, compute the signed distance to every facet
   plane; if `dist >= -r_cov_i + hull_tol` for any facet, mark exposed.
3. Return sorted unique anchor indices.

Energy storage: in `build_reactant`, after the LBFGS step set
`reactant.energy = float(atoms.get_potential_energy())` while the
calculator is still attached, mirroring the
`optimise_structure`-returns-a-copy convention in `AGENTS.md`.

Public additions in `reactants.py` (signatures only):

- `find_unique_atoms(reactant: Reactant) -> dict[str, list[list[int]]]`
- `find_anchor_atoms(reactant: Reactant, *, hull_tol: float = 0.1) -> list[int]`
- `build_reactant(smiles, *, calculator=None, …) -> Reactant`  (extended return — backwards compatible additive fields).

## 4. Site-mapping for multi-anchor adsorbates

Reuse the existing per-element machinery — for **every** anchor element
look up `G.graph["unique_sites"][element][1]` (the `n_shells = 1`
mandate; raise `NotImplementedError` for any other depth).

Enumeration (`enumerate_configurations`) builds candidate configurations
by:

a. **Anchor ordering** — pick a deterministic anchor order (e.g. orbit-id
   ascending, then atom index) so two automorphism-equivalent
   permutations produce the same configuration.

b. **First anchor placement** — for each clique instance of every allowed
   unique-site class for the first anchor's element (top / bridge /
   hollow per `k`, taken from `G.graph["sites"][element]` filtered by
   iso-class membership), compute the rigid-body transform mapping the
   reactant's first anchor onto the clique's `IsoClass.position`
   (translation only for top sites; full SO(3) alignment of the anchor
   normal for bridges / hollows using the surface-normal estimate from
   the clique's bonded slab atoms).

c. **Remaining anchors** — for each subsequent anchor:
   1. Predict its target Cartesian position by applying the rigid-body
      transform from (b) to its relaxed reactant coordinates.
   2. Search `G.graph["sites"][anchor_element]` for cliques whose
      `site_position` lies within `anchor_distance_tol` (default 0.5 Å,
      MIC) of the prediction. Use `_mic_dist_vec` style distances
      (`autokmc/adsorbate.py`).
   3. Cap candidates per anchor at `k_candidates` (default 8 nearest by
      MIC distance) to bound combinatorics.

d. **Clash filter** (mirror `_build_co_bond_graph`'s convention in
   `default_sites.py`):
   - For every pair `(reactant_atom, slab_atom)` not already in the
     reactant's intramolecular bond list, reject the configuration if
     MIC distance `< clash_factor * (r_cov_a + r_cov_b)` with
     `clash_factor` default `0.75` (looser than the bond cutoff to
     allow legitimate bonded contacts).
   - Also reject configurations where two anchors map to the same
     clique (unless their reactant graph is bonded *and* the clique is
     a `k>=2` shared site — degenerate case, defer to open question §12).

e. **Single-anchor degenerate case** (e.g. `*OH` bound only via O):
   non-anchor atoms inherit positions from the rigid-body alignment;
   only the clash filter constrains them.

## 5. Multi-atom `AdsorptionConfiguration` (parallel to `AdsorptionSite`)

New dataclass in `autokmc/adsorbate.py`, deliberately **separate** from
`AdsorptionSite` so single-atom code paths and tests stay untouched:

```
AdsorptionConfiguration:
    reactant_smiles    : str
    reactant           : Reactant                          # full record
    anchor_clique_map  : dict[int, frozenset[int]]         # anchor atom → slab clique
    subgraph           : nx.Graph                           # see §6
    iso_class          : IsoClass | hashable                # WL hash repr.
    ads_positions      : dict[int, np.ndarray]              # reactant atom idx → xyz
    adsorption_energy  : float
    energy             : float
    occupied           : bool                              = False
    stable             : bool                              = True
    reactive           : bool                              = True
    migrated_to        : dict[int, frozenset]              = {}     # per-anchor
    neighbour_sites    : list[frozenset[frozenset[int]]]   = []     # see §7
    context_results    : dict[hashable, ConfigOptResult]   = {}
    current_result     : ConfigOptResult | None            = None
```

Storage keys (none collide with single-atom keys):

- `G.graph["adsorption_configs"][smiles][1]` →
  `dict[config_key, AdsorptionConfiguration]` keyed by
  `frozenset(frozenset(c) for c in anchor_clique_map.values())`.
- `G.graph["unique_configs"][smiles][1]` → list of iso-class
  representatives (analogue of `G.graph["unique_sites"]`).
- `G.graph["config_opt"][smiles][1]` → `list[ConfigOptResult]`
  (analogue of `G.graph["site_opt"]`).
- `G.graph["context_cache_multi"][smiles][1]` → iso-deduplicated
  context cache (analogue of `G.graph["context_cache"]`).

The `subgraph` field follows the `AdsorptionSite.subgraph` convention
(`adsorbate.py::_build_adsorbate_subgraph`, adsorbate node at
`len(slab)`) but appends **every** reactant atom at keys `len(slab)+i`
(`i` = reactant atom index), preserves intramolecular edges from
`reactant.graph`, and adds slab-edges from each anchor node to its
clique's surface atoms. The slab portion of the subgraph is a *copy*;
the original `G` is never mutated.

## 6. Iso-class deduplication at `n_shells = 1` for combined sites

Reuse `_build_clique_ego` and `_iso_prefilter_key` from
`default_sites.py`, augmented per `dev/PLAN_adsorption_sites.md` §11 and
`adsorbate.py::_build_context_ego`:

1. Build the **augmented graph** = `G.copy()` ∪ all reactant nodes
   (including non-anchor) ∪ all intramolecular edges ∪ anchor↔clique
   edges (one edge per anchor-to-each-clique-member pair).
2. Seed set = union over all anchor cliques of their surface atoms,
   plus every reactant node.
3. Expand exactly **one shell** through the augmented graph; element +
   `surface` (0/1/2) labels are matched. (Mirror the WL/categorical hash
   used in `reduce_sites_by_isomorphism`.)
4. Two configurations are duplicates iff their combined ego subgraphs
   are isomorphic under `categorical_node_match("element", "X")` AND
   their `_iso_prefilter_key` matches.
5. Store iso-class → representative configuration in
   `G.graph["unique_configs"][smiles][1]`.

## 7. Optimisation pipeline `optimise_unique_configurations`

Mirror `optimise_unique_sites` (`autokmc/adsorbate.py:748`), looping
over iso-class representative configurations only:

1. **Reference energies** — reuse `calculate_gas_phase_energy`
   semantics, but the gas-phase reference is the relaxed `reactant.energy`
   from `build_reactant`. `e_surface` is computed once exactly as in
   `optimise_unique_sites`.
2. **Trial assembly** — `slab.copy() + reactant_atoms` rigid-aligned to
   the anchors (re-use the transform from §4b/c); preserve
   `slab.info["frozen_indices"]` `FixAtoms` constraint via
   `Atoms.copy`.
3. **Relax** with LBFGS, re-instantiating the calculator class per
   `AGENTS.md` ("optimise_structure always returns a copy and
   re-instantiates the calculator").
4. **Connectivity** — generalise `check_surface_connectivity` /
   `find_actual_clique` to all anchors:
   - For each anchor, run `find_actual_clique(trial, anchor_idx, G,
     r_cov_anchor, …)`; record per-anchor `actual_clique`.
   - `migrated_to[anchor] = actual_clique` if it differs from the
     planned clique.
   - Verify the **intramolecular** bond graph (excluding surface bonds)
     of the relaxed adsorbate matches `reactant.graph` — reject
     configurations where reactant bonds break or new internal bonds
     form (those are KMC events, out of scope per §1).
   - `stable = (all anchors OK) and (intramolecular graph unchanged)`.
5. Build `ConfigOptResult` (parallel to `SiteOptResult`) carrying
   `atoms_initial`, `atoms_final`, `energy`, `adsorption_energy`,
   `converged`, `n_steps`, per-anchor `connectivity`, per-anchor
   `actual_clique`, `matched_iso_class`, `displacement`. Cache in
   `G.graph["config_opt"][smiles][1]`.
6. Stage-3 expansion: every clique-permutation in the iso-class shares
   the representative's `ConfigOptResult`; populate
   `G.graph["adsorption_configs"][smiles][1]` exactly as
   `optimise_unique_sites` does for `adsorption_sites`. Seed
   `context_results[frozenset()] = result` and
   `current_result = result`.
7. Call a new `compute_neighbour_configurations(G, smiles, n_shells=1)`
   to populate `AdsorptionConfiguration.neighbour_sites` (see §8).

## 8. On-the-fly KMC discovery extension

New entry points (parallel to `register_adsorption` /
`register_desorption` / `discover_context_site` in `adsorbate.py`):

- `register_adsorption_multi(G, smiles, config_key, host_atoms,
  calculator, *, discover=True, **discover_kwargs) -> set[config_key]`
  - Mark every clique in `config.anchor_clique_map.values()` as occupied
    in `G.graph["adsorption_sites"][element][1]` (so single-atom
    discovery sees them as blockers).
  - Mark `config.occupied = True`, `reactive = False`.
  - If `discover`, iterate each *neighbour configuration* (other
    `AdsorptionConfiguration`s whose anchor cliques share at least one
    surface atom with any clique of `config`) and call
    `discover_context_configuration` for those that are stable, vacant.
  - Return the set of `config_key`s whose `reactive` flag flipped (KMC
    engine refreshes its reaction list).

- `register_desorption_multi(G, smiles, config_key) -> set[config_key]`
  - Inverse: free every clique in `anchor_clique_map.values()`, recompute
    `reactive` for the changed config + neighbours.

- `discover_context_configuration(G, smiles, config_key, host_atoms,
  calculator, *, fmax=0.05, steps=500, …) -> ConfigOptResult`
  - Two-tier cache lookup analogous to `discover_context_site`:
    1. Per-config exact-occupancy hash hit
       (`config.context_results[occ_key]`).
    2. Global iso-deduplicated hit
       (`G.graph["context_cache_multi"][smiles][1]`).
    3. Miss → assemble trial Atoms (host + every adsorbed reactant at
       its `current_result` positions + the trial reactant), relax,
       build `ConfigOptResult`, store in both caches.

- `compute_neighbour_configurations(G, smiles, n_shells=1)` —
  configuration `A` and `B` are neighbours iff
  `(union(A.anchor_clique_map.values()) & union(B.anchor_clique_map.values())) ≠ ∅`,
  i.e. they share at least one surface atom (consistent with
  `compute_neighbour_sites` decision §10 of the single-atom plan).

- `update_reactive_flags_multi(G, smiles, n_shells, changed_key)` —
  mirror `update_reactive_flags`, configuration-keyed.

## 9. Combinatorial control / pruning heuristics

Defaults exposed as kwargs on `enumerate_configurations`:

- `max_anchors = 4` — hard cap; raise `ValueError` for larger reactants
  in v1 (defer multi-anchor scaling to follow-up plan).
- `k_candidates = 8` — per-anchor MIC-nearest clique cap (§4c).
- `anchor_distance_tol = 0.5` Å — MIC tolerance between predicted and
  candidate anchor position (§4c).
- `clash_factor = 0.75` — hard reject below this fraction of the bond
  cutoff (§4d).
- Early reject if predicted **anchor–anchor** MIC distance differs from
  the intramolecular distance in `reactant.atoms` by more than
  `± anchor_distance_tol`.

Expected complexity: `O(n_anchors! · k_candidates ^ n_anchors)`. Anchor
orbit deduplication divides by `|aut(reactant.graph)|`; the clash and
distance filters dominate in practice. Document in the docstring of
`enumerate_configurations`.

## 10. Public API surface (signatures only, no implementations)

In `autokmc/reactants.py`:

- `build_reactant(smiles: str, *, calculator=None, add_hydrogens=True,
  fmax=0.01, steps=500, nl_mult=1.1, hull_tol=0.1) -> Reactant`
  (extended return; new fields `energy`, `unique_nodes`, `anchor_atoms`,
  `anchor_orbit`).
- `find_unique_atoms(reactant: Reactant) -> dict[str, list[list[int]]]`
- `find_anchor_atoms(reactant: Reactant, *, hull_tol: float = 0.1) -> list[int]`

In `autokmc/adsorbate.py`:

- `enumerate_configurations(G, reactant, *, n_shells=1,
  max_anchors=4, k_candidates=8, anchor_distance_tol=0.5,
  clash_factor=0.75) -> list[AdsorptionConfiguration]`
- `optimise_unique_configurations(G, reactant, atoms, calculator, *,
  n_shells=1, fmax=0.01, steps=500, e_surface=None,
  e_reactant=None, logfile=None, verbose=True) -> list[ConfigOptResult]`
- `compute_neighbour_configurations(G, smiles, n_shells=1) ->
  dict[config_key, list[config_key]]`
- `discover_context_configuration(...)`,
  `register_adsorption_multi(...)`, `register_desorption_multi(...)`,
  `update_reactive_flags_multi(...)`.

`AdsorptionConfiguration` and `ConfigOptResult` dataclasses live next
to `AdsorptionSite` and `SiteOptResult` in `autokmc/adsorbate.py`.

## 11. Testing strategy (extend `autokmc/tests/`)

Default to EMT (no calculator file required); mark NequIP-dependent
integration cases with the existing `requires_nequip` skip
(`autokmc/tests/conftest.py`).

1. **`test_find_anchor_atoms`** — unit:
   - H2 (linear): both atoms → anchors (QHullError fallback).
   - H2O (bent, 3 atoms, planar): all atoms → anchors (degenerate
     hull fallback).
   - CH4: only the 4 H atoms → anchors (C is hull-interior).
   - CO: both atoms → anchors.
2. **`test_find_unique_atoms`** — H2O → `{H: [[1,2]], O: [[0]]}`;
   CH4 → 4 H in one orbit; H2 → 2 H in one orbit.
3. **`test_enumerate_configurations`** — Cu(111) `(4,4,4)` slab
   pre-built via the existing structure helpers + EMT relax; H2 and CO
   reactants. Assert: number of configurations after dedup matches the
   number of inequivalent ordered anchor-clique pairs predicted by
   hand for the (4,4,4) Cu(111) symmetry.
4. **`test_optimise_unique_configurations_emt`** — H on Cu(111) via
   the H2 reactant (gas-phase H2 → adsorbed *H + *H), assert non-NaN
   `adsorption_energy` and `stable=True` for at least one
   configuration.
5. **`test_register_adsorption_multi_roundtrip`** — bootstrap, place
   one config, assert (a) every clique of the placed config is marked
   occupied in `adsorption_sites`, (b) `context_cache_multi` has at
   least one entry, (c) neighbour_sites of the placed config equals
   the union of single-anchor neighbour cliques.
6. **`requires_nequip` integration** — full
   `enumerate → optimise → register_adsorption_multi → desorption`
   round trip on the deployed model
   (`autokmc/dev/cpuhcocuau.nequip.pth`); assert cache hit on a
   symmetry-equivalent second placement (use the
   `CountingCalculator` fixture from `conftest.py`).
7. **Fixture** — add `multiatom_bootstrap` to
   `autokmc/tests/conftest.py` analogous to `bootstrap`: builds a
   Cu(111) slab, runs the single-atom bootstrap for every adsorbate
   element used by the test reactants, and pre-builds a couple of
   common `Reactant` objects.

## 12. Migration / backwards compatibility

- Single-atom code paths
  (`optimise_unique_sites`, `register_adsorption`,
  `register_desorption`, `discover_context_site`,
  `update_reactive_flags`) are **untouched**. They keep writing to
  `G.graph["adsorption_sites"]`, `["site_opt"]`,
  `["context_cache"]`.
- All multi-atom state lives in *new* graph keys
  (`adsorption_configs`, `unique_configs`, `config_opt`,
  `context_cache_multi`) so existing tests in
  `autokmc/tests/test_adsorption_sites.py` continue to pass.
- `optimise_unique_configurations` raises
  `NotImplementedError` for `n_shells != 1` (per user spec).
- `register_adsorption_multi` *does* mutate the single-atom
  `adsorption_sites[element][1]` `occupied` flags so the existing
  single-atom discovery loop sees multi-atom occupancy as blocking
  (see §13). Document this cross-write explicitly in the docstring.
- A future dedicated KMC engine consumes `adsorption_configs` /
  `context_cache_multi` directly; `disreax_kmc` is *not* updated for
  multi-atom support (see §13.5).

## 13. Resolved design decisions

The previous open questions have been resolved as follows:

1. **Degenerate hulls** — the all-atoms-are-anchors fallback is
   accepted for v1. Diatomics (CO, H₂), single atoms, and 3-atom
   planar molecules all hit `QHullError` and treat every atom as a
   candidate anchor. A future revision may add a dedicated 1D/2D
   codepath, but no special handling is needed now.
2. **Rigid alignment** — the reactant is treated as a **rigid body**
   at the trial-assembly stage. All internal degrees of freedom (bond
   lengths, angles, dihedrals) are left to LBFGS during the
   subsequent calculator relaxation. No pre-bending of dihedrals.
3. **Multi-element anchor cliques** — each anchor is resolved
   **independently** against its own element's site list, i.e.
   `G.graph["unique_sites"][anchor_element][1]`. Mixed-element bridges
   on alloys are *not* searched as a single combined site type;
   instead, anchor *A* (element X) is placed on an X-clique and
   anchor *B* (element Y) on a Y-clique, with the clash filter and
   anchor-distance filter enforcing geometric compatibility.
4. **Configuration iso-class collisions** — collapsed by the combined
   iso-class hash defined in §6: two configurations whose combined
   1-shell ego subgraphs are graph-isomorphic (element + `surface`
   labels matched) are deduplicated, regardless of which orbit
   permutation produced them. A regression test (§11) covers this.
5. **KMC engine integration** — `disreax_kmc` is *not* the target
   engine; a new engine will be developed separately. The
   multi-atom data layer commits only to publishing a stable lookup
   surface on the graph (`G.graph["adsorption_configs"]`,
   `G.graph["unique_configs"]`, `G.graph["context_cache_multi"]`)
   plus the event hooks in §8. The new engine consumes these via the
   same pattern the current `register_adsorption` /
   `register_desorption` API uses; design of the engine itself is
   out of scope for this plan.

### Cross-write to single-atom `adsorption_sites`

`register_adsorption_multi` continues to flip
`G.graph["adsorption_sites"][element][1][clique].occupied = True`
for every clique covered by an anchor. This is *required* so that
the existing on-the-fly single-atom discovery loop
(`discover_context_site` / `_build_context_ego`) correctly treats
multi-atom occupancy as blocking. The new KMC engine will read both
the single-atom and multi-atom occupancy maps through this shared
`adsorption_sites` dict.
