# OGKMC Roadmap

This file tracks unresolved scientific or architectural work. Implemented
behavior is documented under [`docs/`](docs/index.md).

## Catalyst and surface models

- Extend oxide support so lattice oxygen can participate in reactions while
  metal atoms retain the catalyst-surface role.
- Improve anchor enumeration for cases that generate unexpectedly large
  cliques. Candidate approaches include blocker checks, an explicit `k_max`,
  and calibration of the covalent-radius factor.
- Raise a clear error when a requested local shell exceeds the usable surface
  graph instead of producing a misleading site class.

## Adsorbate placement and stability

- Calibrate geometric tolerances with a broader set of molecules and surfaces.
- Go beyond rigid-molecule placement so internal bonds may stretch or rearrange
  during adsorption, particularly for oxygen-containing species.
- Define a physically useful path for weakly adsorbing molecules that form no
  explicit surface bond but may still activate or return to the gas phase.
- Investigate additional site-pruning descriptors without discarding stable
  candidates before calculator-based validation.

## Calculators and transition paths

- Review calculator allocation for DFT NEB workflows where wavefunction reuse
  between images may matter. The current independent-calculator pool is aimed
  primarily at non-deepcopyable machine-learning calculators.
- Expand transition-state validation beyond the current connectivity,
  endpoint, energy, and image-position checks where a system-specific
  vibrational or reaction-coordinate test is needed.

## New chemistry

- Add transfer reactions in which atoms move between two adsorbed species.
  This requires transfer templates, placement enumeration, endpoint mapping,
  stability/NEB handling, event transitions, and offline lineage support.

## Deliberate non-goal

Reaction-database fallback matching is topology based and intentionally ignores
Cartesian geometry. This is a method limitation, not an open bug; see
[ISAAC Reaction Database](docs/reaction-database.md#deliberate-geometry-limitation).

## TS validation

Transition states should be verified in more ways that they currently are.
This will be explored further at a later date.

## Speedup
Can do different methods for batching for the TS searches.
Must look into parallelisation and the calculators too.

## Stiffness
Need to develop algorithm to prevent sampling of fast steps.

## TS pruning
Check the pruning to make sure it is functioning as intended

## Process workflow
Need to make sure that a transition state is found. Initial and final states are good, just making the TS search consistent is the next big step.