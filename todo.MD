structure.py:
Need to consider passing an .extxyz manually
Need to consider oxides where the oxygen can react i.e. make the metal the surface and the oxygen the adsorbate

default_sites.py:
Sometimes large k values are found. Need to tinker with this. Options are:
check for blocking atoms
set k_max manually
reduce the factor

find_multisite.py:
Can combine with the default sites code once complete

Can optimise setting the position of the unique sites relative to the graph

graph.py:
Maybe add another flag to associate it with an adsorbate molecule... not sure yet

Multi site:
Set the tolerance to something reasonable. Can test this more later (probably too small)

Solve beyond rigid molecule - can have variable orbits? maybe?