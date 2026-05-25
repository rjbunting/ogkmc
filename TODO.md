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

sites.py:
Need to prune further... there has to be some way, but I'm just unsure!
Shared-calculator NEB is now used to support non-deepcopyable ML calculators.
This may need to change for DFT calculators where passing wavefunctions
between separate image calculators is important.

transition states:
Need to add transition state validation with a variety of methods

transfer.py:
This is a new module that I haven't implemented yet, but it will be responsible for finding transfer sites for reactions
that involve the transfer of atoms from one species to another.

####
