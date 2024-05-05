import vrplib
from genetic import genetic_algorithm

instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")

# Assuming you have already loaded these from your instance
depot_loc = instance['node_coord'][0]
node_loc = instance['node_coord']
demand = instance['demand']
capacity = instance['capacity']

# ----------------------------------------------
