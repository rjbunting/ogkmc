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
   and a Weisfeiler-Lehman graph fingerprint.

Every fallback candidate must then pass full labelled graph isomorphism. Before
loading, AutoKMC validates the ISAAC document, resolves every asset within the
record directory, verifies every SHA-256, checks the required state set, and
reads the `.extxyz` structures.

Calculator/optimizer/NEB/free-energy settings participate in compatibility.
Temperature- and pressure-dependent KMC rates are recomputed for the current
run; the database stores primitive structures and energetics rather than a
rate frozen to old conditions.

## Deliberate geometry limitation

Fallback graph matching intentionally ignores Cartesian geometry. It compares
chemical identity, endpoint roles, elements, bond roles, and topology. A graph
can therefore match a stored record even if its coordinates differ.

This is a limitation of the method and must remain allowed. The database does
not claim that fallback matching proves geometric equivalence. The accepted
record's checksummed `.extxyz` files make the reused geometry explicit and
auditable.

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
  calculation_cache_dir: calculation_cache
  isaac_export_filename: isaac_records.json
```

Disable database reads/writes with `calculation_cache_enabled: false`. This
does not disable reaction-folder `.extxyz` persistence in the run directory.

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
