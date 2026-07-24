# ISAAC Reaction Database

The reaction database avoids repeating expensive endpoint, vibrational, and
NEB calculations. Record folders are the source of truth; SQLite is only a
rebuildable search index.

## Layout

```text
calculation_cache/
  database_manifest.json
  index.sqlite3
  records/<ULID>/
    isaac_record.json
    reaction_graph.json
    occupied.extxyz              # adsorption
    unoccupied.extxyz
    state_a.extxyz               # diffusion
    state_b.extxyz
    state_ab.extxyz              # bond
    state_c.extxyz
    ts.extxyz                    # diffusion and bond
    neb_path.extxyz              # optional
```

`database_manifest.json` records every run UUID that has written into the
database and the current writer. Each record receives a real ULID derived from
its creation time plus deterministic entropy from the calculation key.

## ISAAC record

`isaac_record.json` is validated against the vendored official ISAAC v1.05
JSON schema before it becomes visible. AutoKMC records are computation evidence
records containing:

- material identity derived from the catalyst graph,
- calculator technique and method,
- operation and compatibility parameters,
- endpoint energies and optional thermochemistry,
- activation barriers for diffusion and bond processes,
- AutoKMC/database schema identifiers,
- run UUID when the record was produced during a configured run,
- assets with URI, media type, role, and SHA-256.

See the upstream
[ISAAC AI-ready scientific record wiki](https://github.com/ISAAC-DOE/isaac-ai-ready-record/wiki)
for the general record format. AutoKMC vendors the schema used at runtime under
`autokmc/schema/` so validation does not depend on network access.

Structures are not embedded as large JSON coordinate arrays. Required
structures are external `.extxyz` assets, which makes the geometries directly
readable by ASE and preserves the exact atomic state behind the calculation.

The run-level `isaac_records.json` export is a JSON array of complete ISAAC
records. Its asset URIs are rewritten relative to the export location. Keep the
corresponding `calculation_cache/records/` tree with the export.

## Lookup sequence

AutoKMC uses two lookup paths:

1. Exact calculation-key lookup.
2. Fallback lookup using reaction kind, operation identity, parameter hash,
   a Weisfeiler-Lehman graph fingerprint, a normalized scientific-input
   fingerprint, and an explicit calculator/model digest.

Every fallback candidate must then pass full labelled graph isomorphism. Before
loading, AutoKMC validates the ISAAC document, resolves every asset within the
record directory, verifies every SHA-256, checks the required state set, and
reads the `.extxyz` structures.

The local-geometry fingerprint uses element/fixed-state-labelled pair
distances under the minimum-image convention and the full lattice Gram matrix.
It also includes initial charges, magnetic moments, tags, custom per-atom
arrays, partial-periodicity flags, and structure metadata. It is invariant to
rigid translation, cell wrapping, atom order, and rigid rotation, while
rejecting strained local structures or different calculator-relevant atomic
state. Model checkpoint files and directory-valued model artifacts are
identified recursively by SHA-256 content rather than a machine-local path;
identical copied artifacts match and modified artifacts do not.

The scientific-input fingerprint replaces each `Atoms` input with that
invariant geometry description and retains every other input recursively.
Scalar gas energies, charge, spin declarations, and similar inputs therefore
cannot change without causing a miss. Only an explicit allowlist of run-local
enumeration identifiers (for example node ids, `iso_class`, and
`lateral_class`) is omitted.

For gas-product bond reactions, the gas molecule geometry and every gas
thermochemistry value consumed by the endpoint calculation are scientific
inputs. The feed partial pressure is intentionally not cached: it is restamped
from the current reactant and used only when the live KMC rate is evaluated.

Invariant geometry proves that two requests describe the same local shape; it
does not define how stored output coordinates should be moved into a new
coordinate frame. Records with endpoint structures or NEB paths therefore
also require an exact input-frame fingerprint before portable reuse. A rigidly
translated, rotated, wrapped, or atom-permuted query is conservatively
recomputed rather than receiving structures in the wrong frame.

Calculator/optimizer/NEB/free-energy settings participate in compatibility.
Temperature- and pressure-dependent KMC rates are recomputed for the current
run; the database stores primitive structures and energetics rather than a
rate frozen to old conditions.

Bare diffusion and bond calculations retain their optimized NEB band in the
calculation cache even when reaction-folder path persistence is disabled. A
lateral calculation that uses this band records the seed policy, projection
scope, and exact source-band fingerprint as cache inputs, so a result cannot be
reused against a different initial path.

Calculator constructor and factory arguments are retained even when the
constructed backend object does not expose them. Resolvable files and
directories nested under backend-specific parameter names are content-hashed,
and installed distribution versions for top-level and nested calculator entry
points participate in the cache identity. An opaque calculator whose entry
point cannot be versioned remains process-local rather than matching an
unverifiable result across runs. Remote model aliases remain literal because
AutoKMC cannot inspect their contents; production configurations should pin an
immutable revision or digest, not a mutable alias.

## Matcher compatibility

Reaction-database schema v2 adds geometry, normalized scientific-input,
input-frame, and calculator/model digests to the fallback contract. Older
record folders remain valid, exportable, and loadable through an explicitly
known exact key. Records missing any required compatibility digest are
deliberately excluded from portable fallback. Recomputing a result writes all
current v2 compatibility evidence.

Run-local graph node ids, `iso_class`, and `lateral_class` counters are also
ignored because they are enumeration artifacts rather than portable chemical
labels.

## Index recovery

If `index.sqlite3` is absent or unreadable, lookup automatically rebuilds it by
scanning record folders and accepting only records whose schemas and assets
verify. Invalid or tampered records are skipped.

Manual recovery uses:

```bash
autokmc rebuild-index RUN_DIR/calculation_cache
```

The rebuilt SQLite file is constructed separately and atomically replaces the
old index. Record folders are not modified.

## Database configuration

```yaml
output:
  calculation_cache_enabled: true
  calculation_cache_lookup_enabled: false
  calculation_cache_dir: calculation_cache
  isaac_export_enabled: false
  isaac_export_filename: isaac_records.json
```

Disable database reads/writes with `calculation_cache_enabled: false`. This
does not disable reaction-folder `.extxyz` persistence in the run directory.
By default, `calculation_cache_lookup_enabled: false` skips calculation-record
lookups while continuing to write ISAAC records and update the SQLite index.
Set it to `true` to reuse matching records. The default write-only mode is
useful when generating a database for later upload or reuse elsewhere.
The portable, potentially expensive aggregate export is opt-in; set
`isaac_export_enabled: true` when an `isaac_records.json` bundle is needed.

## Portability checklist

For an archival or manuscript artifact, retain:

- `run_manifest.json`
- `events.jsonl`
- `summary.json`
- `kmc.extxyz`
- `isaac_records.json`
- `calculation_cache/database_manifest.json`
- `calculation_cache/records/`
- relevant reaction folders and optional NEB paths

The SQLite index is convenient but not essential because it can be rebuilt.
