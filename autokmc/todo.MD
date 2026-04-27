structure.py: 
Need to consider oxides where the oxygen can react i.e. make the metal the surface and the oxygen the adsorbate

find_anchors.py: 
Sometimes large k values are found. Need to tinker with this. Options are: check for blocking atoms set k_max manually reduce the factor
Sometimes n_shell can be too big for the surface. Need to raise error when this happens

find_adsorbate_sites.py:
Set the tolerance to something reasonable. Can test this more later (probably too small) Solve beyond rigid molecule - can have variable orbits? maybe? IMPORTANT: Weakly adsorbing molecules (like CH4) will form no bonds to surface. Need way to still activate them or release into gas in 1 step

find_adsorbate_sites.py (chain-placement performance):
Re-introduce the per-element ``_AnchorKDTree`` annulus-query prefilter
in front of the inner ``_recurse`` loop.  The previous implementation
(orthogonal periodic → cKDTree(boxsize); non-orthogonal periodic →
±1-image tiling; NP → plain tree) replaced the current O(N) scan with
an O(log N + h) range query around the most recently placed anchor and
verified the remaining pairwise constraints explicitly on the small
candidate set.  It produced incorrect placements on at least one test
case (suspected MIC / wrap / candidate-deduplication bug), so the
optimisation has been removed in favour of the easier-to-audit O(N)
backtracking.  Re-introduce once the bug is understood.


find_anchors.py (_optimise_position slab branch):
Currently uses L-BFGS-B with a hard z-floor (``z ≥ max(b_pos[:, 2])``)
which assumes the surface normal is aligned with +z.  This holds for all
slabs produced by :mod:`autokmc.structure` (which orthogonalises the cell)
but is fragile for tilted inputs.  A local-outward-normal approach
(project lateral displacements onto the plane ⊥ n_out, apply half-space
SLSQP constraint along n_out) was implemented but produced incorrect
optimised positions in practice and was reverted.  Revisit when a clear
test case can be used to validate the normal computation.

structure.py (optimise_structure):
The calculator is deep-copied on every call so loaded ML models (e.g.
NequIP) are preserved. For tight inner-loop workflows that re-relax
the same Atoms hundreds of times this is wasteful. Add an opt-in
``copy_calculator: bool = True`` kwarg so callers can short-circuit
the copy when they know the calculator is safe to share.
