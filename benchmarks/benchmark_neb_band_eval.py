"""A/B benchmark: NEB band evaluation ``images`` vs ``batched``.

Runs identical endpoint pairs through :func:`autokmc.sites.stability.neb.run_neb`
with both band-evaluation modes on the same calculator and reports, per case:
barrier, |dEa|, wall time, speedup, and model-call counts.  This is the
verification driver for the ``optimization.neb_band_eval`` config option.

Endpoints come either from built-in fcc(111) test hops (O/CO on Cu/Au) or
from a directory of persisted NEB paths harvested from a real run
(``--endpoints-dir`` expects ``<case>_initial.traj`` / ``<case>_final.traj``
pairs; ``persist_neb_path: true`` runs write usable band snapshots).

Examples
--------
CPU smoke test (EMT, plumbing only, no physics claim)::

    python dev/benchmark_neb_band_eval.py --backend emt

Production comparison (UMA, one GPU)::

    python dev/benchmark_neb_band_eval.py --backend uma --device cuda \
        --fmax 0.05 --n-images 8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import LBFGS

from autokmc.sites.stability.neb import run_neb


class BenchNEBFailed(RuntimeError):
    pass


@dataclass
class Case:
    name: str
    initial: Atoms
    final: Atoms
    frozen: list[int]


def _slab_case(
    name: str,
    metal: str,
    adsorbate: str,
    site_a: str,
    site_b: str,
    *,
    size=(3, 3, 4),
    height: float = 1.1,
) -> Case:
    initial = fcc111(metal, size=size, vacuum=8.0)
    frozen = [i for i, atom in enumerate(initial) if atom.tag > size[2] // 2]
    final = initial.copy()
    add_adsorbate(initial, adsorbate, height=height, position=site_a)
    add_adsorbate(final, adsorbate, height=height, position=site_b)
    return Case(name=name, initial=initial, final=final, frozen=frozen)


def builtin_cases() -> list[Case]:
    return [
        _slab_case("O_Cu111_fcc_hcp", "Cu", "O", "fcc", "hcp"),
        _slab_case("O_Au111_fcc_hcp", "Au", "O", "fcc", "hcp"),
        _slab_case("CO_Cu111_fcc_hcp", "Cu", "C", "fcc", "hcp", height=1.9),
    ]


def load_endpoint_cases(directory: Path) -> list[Case]:
    cases = []
    for initial_path in sorted(directory.glob("*_initial.traj")):
        name = initial_path.name[: -len("_initial.traj")]
        final_path = directory / f"{name}_final.traj"
        if not final_path.exists():
            continue
        initial = read(initial_path)
        final = read(final_path)
        frozen: list[int] = []
        for constraint in initial.constraints:
            if isinstance(constraint, FixAtoms):
                frozen = [int(i) for i in constraint.index]
        cases.append(Case(name=name, initial=initial, final=final, frozen=frozen))
    if not cases:
        raise SystemExit(f"no *_initial.traj/*_final.traj pairs in {directory}")
    return cases


class CountingCalculator:
    """Count single-structure and band evaluations on any calculator."""

    def __init__(self, inner):
        self.inner = inner
        self.single_calls = 0
        self.band_calls = 0
        self.band_images = 0
        self._band_evaluator = None

    def get_potential_energy(self, atoms, force_consistent=False):
        self.single_calls += 1
        return self.inner.get_potential_energy(atoms)

    def get_forces(self, atoms):
        self.single_calls += 1
        return self.inner.get_forces(atoms)

    def evaluate_band(self, images):
        if self._band_evaluator is None:
            from autokmc.sites.stability.band_eval import (
                resolve_band_evaluator,
            )
            self._band_evaluator = resolve_band_evaluator(self.inner)
            if self._band_evaluator is None:
                raise AttributeError(
                    "inner calculator supports no band evaluation"
                )
        self.band_calls += 1
        self.band_images += len(images)
        return self._band_evaluator.evaluate_band(images)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def build_calculator(backend: str, device: str):
    if backend == "emt":
        from ase.calculators.emt import EMT

        class _BandEMT(EMT):
            def evaluate_band(self, images):
                results = []
                for image in images:
                    energy = float(self.get_potential_energy(image))
                    forces = np.asarray(self.get_forces(image), dtype=float)
                    results.append((energy, forces))
                return results

        return _BandEMT()
    if backend == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predict_unit = pretrained_mlip.get_predict_unit(
            "uma-s-1p2", device=device
        )
        return FAIRChemCalculator(predict_unit, task_name="oc20")
    raise SystemExit(f"unknown backend {backend!r}")


def relax_endpoints(case: Case, calculator, fmax: float) -> None:
    for atoms in (case.initial, case.final):
        atoms.set_constraint(FixAtoms(indices=case.frozen))
        atoms.calc = calculator
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
        atoms.calc = None


def run_mode(case: Case, calculator, mode: str, args) -> dict:
    counter = CountingCalculator(calculator)
    start = time.perf_counter()
    result = run_neb(
        case.initial,
        case.final,
        calculator=counter if mode == "batched" else calculator,
        purpose=f"benchmark {mode}",
        n_images=args.n_images,
        interpolation="linear",
        spring_k=5.0,
        climb=True,
        frozen_indices=case.frozen,
        fmax=args.fmax,
        max_steps=args.max_steps,
        verbose=False,
        not_converged_error=BenchNEBFailed,
        capture_path=True,
        band_eval=mode,
    )
    walltime = time.perf_counter() - start
    e_initial = result.path_energies[0]
    e_final = result.path_energies[-1]
    barrier = result.energy_ts - min(e_initial, e_final)
    return {
        "mode": mode,
        "barrier_eV": barrier,
        "walltime_s": walltime,
        "optimizer_steps": result.optimizer_steps,
        "single_calls": counter.single_calls if mode == "batched" else None,
        "band_calls": counter.band_calls if mode == "batched" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="emt", choices=["emt", "uma"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--endpoints-dir", type=Path, default=None)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    calculator = build_calculator(args.backend, args.device)
    cases = (
        load_endpoint_cases(args.endpoints_dir)
        if args.endpoints_dir
        else builtin_cases()
    )

    rows = []
    for case in cases:
        relax_endpoints(case, calculator, args.fmax)
        results = {}
        for mode in ("images", "batched"):
            try:
                results[mode] = run_mode(case, calculator, mode, args)
            except BenchNEBFailed as exc:
                print(f"[{case.name}] {mode}: NOT CONVERGED ({exc})")
        if len(results) != 2:
            continue
        images, batched = results["images"], results["batched"]
        dea_mev = abs(batched["barrier_eV"] - images["barrier_eV"]) * 1000.0
        speedup = images["walltime_s"] / max(batched["walltime_s"], 1e-12)
        row = {
            "case": case.name,
            "Ea_images_eV": round(images["barrier_eV"], 4),
            "Ea_batched_eV": round(batched["barrier_eV"], 4),
            "dEa_meV": round(dea_mev, 2),
            "t_images_s": round(images["walltime_s"], 2),
            "t_batched_s": round(batched["walltime_s"], 2),
            "speedup": round(speedup, 2),
            "steps_images": images["optimizer_steps"],
            "steps_batched": batched["optimizer_steps"],
            "band_calls": batched["band_calls"],
        }
        rows.append(row)
        print(
            f"[{case.name}] Ea {row['Ea_images_eV']:.4f} vs "
            f"{row['Ea_batched_eV']:.4f} eV (|d| {row['dEa_meV']:.2f} meV) | "
            f"{row['t_images_s']:.2f}s -> {row['t_batched_s']:.2f}s "
            f"({row['speedup']:.1f}x) | band_calls={row['band_calls']}"
        )

    if args.json and rows:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
