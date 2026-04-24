# PLAN — On-the-fly adsorption sites for KMC

## 1. Problem statement

The current `autokmc` pipeline relaxes one representative per iso-class on a
**clean** surface and reuses that energy for every clique instance. For
on-the-fly KMC we instead need: (a) an explicit per-clique record on `G` of
which sites are *stable* vs *unstable* after the calculator relaxation,
(b) a *reactive* flag tracking whether a stable, vacant site has been
optimised in the **current** neighbour-occupancy context, and (c) a
discovery loop that, when an adsorption event occurs, re-optimises (or
cache-hits) every neighbouring site under its new occupancy pattern. Single
atoms only for the proof of concept; multi-atom support reuses
`AdsorptionSite.subgraph`.

## 2. Definitions

- **stable site** — an `AdsorptionSite` whose owning `IsoClass`
  representative relaxed with `ConnectivityStatus.OK` (set in
  `optimise_unique_sites`, `adsorbate.py` ≈ line 925, via
  `check_connectivity`).
- **unstable site** — connectivity ≠ `OK` after relaxation
  (`LOST_BOND` / `NEW_BOND` / `MIGRATED`).  `SiteOptResult.matched_iso_class`
  (set ≈ line 935) names the iso-class the adsorbate actually landed in
  (or `None` for novel topology).
- **reactive site** — a stable, currently-vacant site whose
  `(iso_class, neighbour_occupancy_pattern)` key already has a cached
  `SiteOptResult` matching the live surface state. Initially (clean
  surface) every stable site is reactive.
- **neighbour-coupled iso-class** — the cache key
  `(iso_class_id, frozenset[occupied_neighbour_iso_class_id, …])` (or its
  ego-graph isomorphism representative) under which a context-specific
  relaxation result is stored on `AdsorptionSite.context_results`.

## 3. Data-model changes

Extend `AdsorptionSite` in `autokmc/adsorbate.py`:

- `stable: bool` — mirrors the iso-class's `ConnectivityStatus == OK`.
- `reactive: bool` — recomputed after every adsorption / desorption event.
- `neighbour_sites: list[frozenset]` — cliques sharing surface atoms with
  this one (computed once at Stage-3 build time; see §4a).
- `context_results: dict[frozenset[frozenset], SiteOptResult]` — keyed by
  the sorted frozenset of currently-occupied neighbour cliques (or by an
  iso-deduplicated key, see §4d).
- `migrated_to: Optional[frozenset]` — convenience pointer to
  `result.actual_clique` when `stable is False` (so KMC can redirect the
  adsorption event to where the atom actually ended up).

`SiteOptResult` is unchanged; it already carries `connectivity`,
`matched_iso_class`, `actual_clique`.  No new top-level graph keys are
strictly required — existing
`G.graph["adsorption_sites"][element][n_shells][clique]` entries gain the
new fields.  An optional `G.graph["context_cache"][element][n_shells]`
keyed by ego-graph iso-class can deduplicate context relaxations across
symmetry-equivalent cliques.

For multi-atom adsorbates the existing `AdsorptionSite.subgraph`
(adsorbate appended at node `len(slab)` in `_build_adsorbate_subgraph`) is
already the right abstraction — no schema change needed, only the
placement / connectivity helpers (see §7).

A lightweight new dataclass `IsoClassContext` may be introduced to hold
the cache key + ego-graph fingerprint:

```
IsoClassContext(iso_class_id, occupancy_iso_key, ego_graph)
```

## 4. Algorithm

### a. Bootstrap (clean surface) — extend `optimise_unique_sites`

After the existing per-iso-class relax / `check_connectivity` block in
`adsorbate.py`:

1. Tag each `IsoClass` with a derived `stable = (result.connectivity == OK)`
   (store on the `SiteOptResult` or on the `IsoClass` directly — pick one,
   suggest on a new `IsoClass.stable` field to keep `SiteOptResult`
   immutable).
2. Propagate to every `AdsorptionSite` instance built in Stage 3
   (members of an iso-class share `stable`).
3. For unstable iso-classes, populate `AdsorptionSite.migrated_to` from
   `result.matched_iso_class` so KMC redirects rather than discards.
   Do **not** drop unstable sites from `adsorption_sites`.
4. Compute `neighbour_sites` for every clique using the **shared
   surface atom** rule (decision §10): two cliques are neighbours iff
   `clique_i & clique_j` is non-empty.  Implement as
   `compute_neighbour_sites(G, element, n_shells)` over
   `G.graph["sites"][element]`.  Self is excluded.

### b. Reactive flag initialisation

On a clean surface, `reactive = stable`.  No occupied neighbours →
`context_results` is keyed by `frozenset()` and points at the bootstrap
`SiteOptResult`.

### c. On-the-fly discovery loop (called by the KMC engine)

Triggered when KMC adsorbs an atom at site `S`:

1. `S.occupied = True`; `S.reactive = False`.
2. For each `N in S.neighbour_sites` that is `stable` and `not occupied`:
   1. Build the context key:
      `key = frozenset(n.clique for n in N.neighbour_sites if n.occupied)`
      (optionally reduced to its ego-iso representative — see §4d).
   2. Look up `N.context_results[key]`.
   3. **Hit** → reuse cached `adsorption_energy` and final position;
      mark `N.reactive = True`.
   4. **Miss** → call `discover_context_site(G, element, N.clique,
      occupied_cliques, atoms, calculator, …)` which:
      - Builds a trial `Atoms` = bare slab + every currently-adsorbed atom
        (at its cached position) + trial atom at `N.ads_position`.
      - Runs LBFGS with the same `fmax` / `steps` defaults as
        `optimise_unique_sites`.
      - Runs `check_connectivity` for the **trial** atom at `N` and
        verifies that previously-adsorbed atoms have not migrated
        (re-check their actual cliques).
      - Constructs a `SiteOptResult`, stores it in
        `N.context_results[key]`, and (if iso-deduplication is enabled)
        also in `G.graph["context_cache"]`.
      - Sets `N.reactive` according to the trial-atom connectivity.

3. On desorption from `S`: cached entries remain valid (they are keyed by
   occupancy pattern); only `update_reactive_flags` needs to re-evaluate
   neighbours of `S` against the new occupancy state.

### d. Iso-class deduplication of contexts (`n_shells = 1`)

Many neighbour-occupancy patterns are equivalent by graph isomorphism
(same physics as `reduce_sites_by_isomorphism`).  Re-use
`_build_clique_ego` and `_iso_prefilter_key` in `default_sites.py` but:

1. Build the **augmented graph** = `G` ∪ adsorbate nodes from every
   occupied neighbour's `AdsorptionSite.subgraph` (single atom today;
   full reactant subgraphs once §11 lands).
2. Seed the ego expansion with the union of (target clique surface
   atoms) ∪ (every adsorbate node from every occupied neighbour
   subgraph).  This is the §11 expansion rule.
3. Expand exactly **one shell** (`n_shells = 1`, decision §10) through
   the augmented graph.
4. Two contexts hashing to the same `(prefilter_key, isomorphic ego)`
   share one cache entry.

This avoids N×M redundant relaxations on symmetric surfaces.

## 5. API additions (in `autokmc/adsorbate.py`)

- `compute_neighbour_sites(G, element, n_shells) -> dict[frozenset, list[frozenset]]`
  — return per-clique neighbour list using the chosen definition; also
  cached on each `AdsorptionSite.neighbour_sites`.
- `mark_stable_unstable(G, element, n_shells) -> None`
  — populate `stable` / `migrated_to` on every `AdsorptionSite` from its
  `SiteOptResult.connectivity` and `matched_iso_class`.
- `discover_context_site(G, element, target_clique, occupied_cliques, atoms, calculator, *, n_shells, fmax=0.05, steps=500) -> SiteOptResult`
  — on-the-fly entry point; idempotent via `context_results` cache.
- `update_reactive_flags(G, element, n_shells, changed_clique) -> set[frozenset]`
  — recompute `reactive` for `changed_clique` and its neighbours; returns
  the set of cliques whose flag flipped (KMC engine uses this to refresh
  reaction lists).
- `_context_key(G, site, occupied_neighbours, n_shells) -> frozenset`
  — internal helper for the iso-deduplicated cache key.

## 6. Graph contract additions

Existing keys (`G.graph["sites"]`, `["unique_sites"]`,
`["site_positions"]`, `["adsorption_sites"]`, `["site_opt"]`) are
**unchanged**.  New per-site fields live on the existing
`AdsorptionSite` objects already stored in
`G.graph["adsorption_sites"][element][n_shells]`.

Optional new top-level key:

- `G.graph["context_cache"][element][n_shells]` —
  `dict[(iso_class_id, occupancy_iso_key) → SiteOptResult]` for
  cross-clique iso-deduplicated lookup.

## 7. Multi-atom adsorbate forward compatibility

The `AdsorptionSite.subgraph` already represents an adsorbate as one or
more nodes attached to the clique.  Discovery-loop code paths only need
two extensions when multi-atom adsorbates land:

- `place_adsorbate` (`adsorbate.py`) → variant taking a `Reactant`
  subgraph (from `reactants.build_reactant`) and inserting all its atoms.
- `find_actual_clique` / `check_connectivity` → check the bonding atom(s)
  of the reactant rather than a single index.

`compute_neighbour_sites`, `discover_context_site`, the cache key, and
`update_reactive_flags` are agnostic to adsorbate cardinality.  Out of
scope for the proof of concept; flagged here so the API is shaped for it.

## 8. Testing strategy (extend `autokmc/tests/`)

The new on-the-fly tests use the **deployed NequIP ML potential** at
`autokmc/dev/cpuhcocuau.nequip.pth` (decision §12); skip cleanly if the
file or the `nequip` package is missing.  Existing structure / surface
tests continue to use EMT.

1. **`test_compute_neighbour_sites`** — Cu(111) `(4,4,4)` slab + O.
   Each fcc hollow has exactly 3 neighbour fcc hollows under the
   shared-surface-atom rule; bridges have 4; tops have 6.
2. **`test_mark_stable_unstable`** — synthetic `SiteOptResult` instances
   with mocked `connectivity`; assert `AdsorptionSite.stable` and
   `migrated_to` propagate correctly to all clique members (no
   calculator needed).
3. **`test_discover_context_site_nequip`** — O on Cu(111),
   `n_shells = 1`, ML calculator.  Verify (a) bootstrap: all hollows
   stable, (b) after occupying one fcc hollow, calling
   `discover_context_site` on each neighbour fcc produces a valid
   `SiteOptResult`, (c) calling it again at a symmetry-equivalent
   neighbour with the same occupancy pattern is a cache hit (assert via
   a counter on a wrapping calculator).
4. **`test_update_reactive_flags`** — adsorb / desorb sequence; assert
   the returned changed-set matches the neighbour set of the toggled
   clique.  Pure graph operations, no calculator.

## 9. Implementation order (smallest commits first)

1. Add `stable`, `reactive`, `migrated_to` fields to `AdsorptionSite` and
   populate them from existing `SiteOptResult.connectivity` /
   `matched_iso_class` inside the Stage-3 loop of
   `optimise_unique_sites`.  No behaviour change for existing callers.
2. Implement `compute_neighbour_sites` + unit tests; populate
   `AdsorptionSite.neighbour_sites`.
3. Implement `discover_context_site` (no caching, no iso-dedup) +
   integration test.
4. Add `context_results` cache + `update_reactive_flags`.
5. Add iso-deduplicated `G.graph["context_cache"]` and route lookups
   through it; extend tests to assert dedup hits across symmetry-
   equivalent cliques.
6. Write the event-handler signature documentation that
   `disreax_kmc` will call (no cross-package import; pure docstring +
   example notebook update in `dev_adsorption.ipynb`).

## 10. Decisions (locked)

- **Neighbour definition** — **(A) shared surface atom**: two cliques are
  neighbours iff `clique_i & clique_j` is non-empty.  No kwarg override.
- **Relaxation scope during context discovery** — **fully relax
  everything** (slab atoms minus the existing `FixAtoms`, all
  previously-adsorbed atoms, and the new trial atom).  Same `FixAtoms`
  constraint that came in on `atoms` is honoured automatically by
  `place_adsorbate` (which deep-copies constraints).
- **Unstable-site policy** — keep them in
  `G.graph["adsorption_sites"]` with `stable=False` and `migrated_to`
  populated.  They serve as a record that the site was tried and failed;
  the KMC engine is free to ignore them when building reaction lists.
- **`n_shells` for context keys** — **`n_shells = 1`**: the immediate
  bonded neighbours of the surface atoms in the target clique.  This is
  hard-coded in the context-key builder, independent of the (separate)
  `n_shells` used by `reduce_sites_by_isomorphism`.
- **Unstable context relaxations** — cache them with `stable=False` and
  `migrated_to` set; mirrors the bootstrap behaviour and prevents repeat
  expensive relaxations of known-unstable contexts.
- **Cache invalidation** — **never**.  Cache entries are pure functions
  of (context key + calculator).  Swapping calculators requires
  rebuilding `G` from scratch.

## 11. Adsorbate subgraph expansion in connectivity / context keys

For multi-atom adsorbates (out of scope for the proof of concept but
shaping the API now): when computing the **context ego-graph** at
`n_shells=1`, the seed set is *not* just the surface atoms in the target
clique — it must also include **every node of every occupied
adsorbate subgraph** that touches a clique surface atom.  Concretely, an
occupied neighbour clique is replaced by the full adsorbate node set
from its `AdsorptionSite.subgraph` (which is already the right
abstraction — see §3).  The ego expansion then walks one shell out
through the *combined* slab + adsorbate graph.

For a single-atom adsorbate the adsorbate subgraph adds exactly one
node, so this collapses to the simple case and there is no behavioural
difference.  Capture this in `_context_key` with a helper
`_collect_adsorbate_nodes(site)` that returns the adsorbate-typed nodes
of the subgraph; the implementation is trivial today and ready for
multi-atom reactants tomorrow.

## 12. Test calculator

Tests use the **deployed ML potential** at
`autokmc/dev/cpuhcocuau.nequip.pth` (loaded via NequIP's ASE
calculator), not EMT.  This matches how the dev notebooks exercise the
pipeline and tests the path that production will actually take.  Add a
shared pytest fixture `ml_calc` in `autokmc/tests/conftest.py` that
constructs the calculator once per session and a
`requires_nequip = pytest.mark.skipif(not _NEQUIP_AVAILABLE, …)` marker
so CI without the model file skips cleanly.  EMT remains the calculator
for the existing `test_structure.py` / `test_surface.py` suites — only
the new on-the-fly tests use the ML potential.

