# autokmc CLI — examples

Run bundled CO/Cu(111) examples end-to-end:

```bash
pip install -e ".[cli,test]"

# EMT (fast smoke test, no ML deps):
autokmc validate-config example/co_cu111_emt.yaml
autokmc run example/co_cu111_emt.yaml

# NequIP / Allegro (realistic):
autokmc validate-config example/co_cu111_allegro.yaml
autokmc run example/co_cu111_allegro.yaml
```

## Output layout

```
runs/co_cu111_*/
├── events.jsonl              # one JSON line per executed KMC event
├── summary.json              # per-reaction-type aggregates + run metadata
├── kmc.extxyz                # extended-XYZ trajectory (initial + every Nth state)
└── reactions/                # one folder per unique (iso_class, lateral_class)
    ├── iso0_lat0/
    │   ├── reaction.json     # description + energies + ΔE / barrier / rate / count
    │   ├── occupied.extxyz   # relaxed atoms used to compute E_occupied
    │   └── unoccupied.extxyz # relaxed atoms used to compute E_unoccupied
    ├── iso0_lat1/
    │   └── …
    └── …
```

Every event in `events.jsonl` has a `reaction_dir` field pointing at its
per-(iso, lat) folder.  The `.extxyz` files are the **as-calculated**
relaxed structures from the stability check — re-attach any ASE
calculator to re-evaluate:

```python
from ase.io import read
from ase.calculators.vasp import Vasp        # any ASE calculator works
atoms = read("runs/co_cu111_emt/reactions/iso0_lat0/occupied.extxyz")
atoms.calc = Vasp(...)                        # VASP, CP2K, NequIP, MACE, …
e = atoms.get_potential_energy()
```

The full KMC trajectory is in `kmc.extxyz` (every Nth state, configurable
via `output.trajectory_dump_every`):

```python
from ase.io import read
frames = read("runs/co_cu111_emt/kmc.extxyz", index=":")
print(f"{len(frames)} frames, last KMC step = {frames[-1].info['kmc_step']}")
```

## Plugging in a different calculator

Edit the `calculator:` block — anything that follows the ASE calculator
protocol works.  Use `import_path` for direct instantiation, or
`factory` for class methods (e.g. NequIP / MACE need a factory).

```yaml
# VASP
calculator:
  import_path: ase.calculators.vasp.Vasp
  kwargs:
    xc: PBE
    encut: 400

# CP2K
calculator:
  import_path: ase.calculators.cp2k.CP2K
  kwargs:
    inp: |
      &GLOBAL
        RUN_TYPE ENERGY_FORCE
      &END GLOBAL

# NequIP (factory)
calculator:
  factory: nequip.ase.NequIPCalculator.from_compiled_model
  factory_kwargs:
    compile_path: ./models/cpuhcocuau.nequip.pth
    device: cpu

# MACE / Allegro (factory)
calculator:
  factory: mace.calculators.MACECalculator.from_model
  factory_kwargs:
    model_path: ./mace_models/MACE-Small.model
    device: cuda

# EMT (built-in, no deps)
calculator:
  import_path: ase.calculators.emt.EMT
  kwargs: {}
```
