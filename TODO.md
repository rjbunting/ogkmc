structure.py:
Need to consider oxides where the oxygen can react i.e. make the metal the surface and the oxygen the adsorbate

find_anchors.py:
Sometimes large k values are found. Need to tinker with this. Options are: check for blocking atoms set k_max manually reduce the factor
Sometimes n_shell can be too big for the surface. Need to raise error when this happens

find_adsorbate_sites.py:
Set the tolerance to something reasonable. Can test this more later (probably too small) Solve beyond rigid molecule -
can have variable orbits? maybe? IMPORTANT: Weakly adsorbing molecules (like CH4) will form no bonds to surface.
Need way to still activate them or release into gas in 1 step

check_adsorbate_sites.py (adsorbate site stability):
Need to do cases when adsorbate bonds will stretch on the surface (oxygen)

check_diffusion_sites.py:
Need to prune further... there has to be some way, but I'm just unsure!

check_diffusion_sites.py (_check_ts_validity endpoint-collapse):
The endpoint-collapse check only raises ``TransitionStateInvalidError`` for
images 1 (adjacent to A) or n_interior (adjacent to B).  If image 2 is the
highest-energy point and its energy is nearly identical to E_a, no exception
is raised.  A more robust check would test ``abs(E_ts - E_a) < energy_tol``
or ``abs(E_ts - E_b) < energy_tol`` regardless of index, or require that
E_ts exceeds both endpoints by at least ``energy_tol``.

find_transfer_sites.py:
This is a new module that I haven't implemented yet, but it will be responsible for finding transfer sites for reactions
that involve the transfer of atoms from one species to another.

####
