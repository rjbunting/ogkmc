structure.py:
Need to consider passing an .extxyz manually
Need to consider oxides where the oxygen can react i.e. make the metal the surface and the oxygen the adsorbate

default_sites.py:
Sometimes large k values are found. Need to tinker with this. Options are:
check for blocking atoms
set k_max manually
reduce the k_opt factor
Need to optimise setting the position of the unique sites relative to the graph

graph.py:
Maybe add another flag to associate it with an adsorbate molecule... not sure yet

default_multisite.py:
Implement geometrically possible sites for multisite adsorption based on the default sites.