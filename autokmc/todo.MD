structure.py: 
Need to consider oxides where the oxygen can react i.e. make the metal the surface and the oxygen the adsorbate

find_anchors.py: 
Sometimes large k values are found. Need to tinker with this. Options are: check for blocking atoms set k_max manually reduce the factor
Sometimes n_shell can be too big for the surface. Need to raise error when this happens

find_adsorbate_sites.py: 
Set the tolerance to something reasonable. Can test this more later (probably too small) Solve beyond rigid molecule - can have variable orbits? maybe? IMPORTANT: Weakly adsorbing molecules (like CH4) will form no bonds to surface. Need way to still activate them or release into gas in 1 step

surface.py:
Fix bottom two layers with reverse ray casting method (do surface, remove, do it again = two layers)