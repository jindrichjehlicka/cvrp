import time

import numpy as np
import vrplib


# Adjusted function to handle objects already in dict format and NumPy arrays
# ecursively converts objects, including NumPy arrays, dictionaries, and lists, into a format that can be easily serialized into JSON.
def obj_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Process each key-value pair in the dictionary
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj


# Assuming the read_instance and read_solution return dictionaries
# Read VRPLIB formatted instances and solutions
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")

# Convert instance and solutions, handling any nested NumPy arrays
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)


def nearest_neighbor_vrp(depot_loc, node_loc, demand, capacity):
    """Solves a CVRP instance using the Nearest Neighbor Algorithm with capacity constraints."""
    num_customers = len(node_loc)
    routes = []
    visited = set()

    while len(visited) < num_customers:
        route = []
        current_load = 0
        current_location = depot_loc
        while True:
            nearest_distance = float('inf')
            nearest_customer = None
            for i, location in enumerate(node_loc):
                if i not in visited and current_load + demand[i] <= capacity:
                    distance = euclidean_distance(current_location, location)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = i
            if nearest_customer is None:
                break
            visited.add(nearest_customer)
            route.append(nearest_customer)
            current_load += demand[nearest_customer]
            current_location = node_loc[nearest_customer]

        if route:  # Only add non-empty routes
            routes.append(route)

    return routes


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Correctly set the depot location based on the assumption it's the first in `node_coord`
depot_loc = instance_dict['node_coord'][0]  # Assuming the first coordinate is the depot

# Adjust node locations and demands for the first 10 nodes (excluding the depot if desired)
# Note: Adjust the slicing as needed based on whether you include the depot in the node list or not
node_loc = instance_dict['node_coord']  # Assuming we skip the depot and take the next 10
demand = instance_dict['demand']  # Corresponding demands, adjusting indices as needed

# The capacity remains unchanged
capacity = instance_dict['capacity']

print("hyhy", solutions_dict)


def calculate_route_cost(route, node_loc, depot_loc):
    """Calculate the total cost (distance) of a single route."""
    cost = 0
    last_location = depot_loc
    for node in route:
        cost += euclidean_distance(last_location, node_loc[node])
        last_location = node_loc[node]
    cost += euclidean_distance(last_location, depot_loc)  # Return to depot
    return cost


def calculate_total_cost(routes, node_loc, depot_loc):
    """Calculate the total cost (distance) of all routes."""
    return sum(calculate_route_cost(route, node_loc, depot_loc) for route in routes)


# Následně můžete tento algoritmus použít stejně jako greedy_vrp:
start_time_nn = time.time()  # Start timer for Nearest Neighbor
nn_routes = nearest_neighbor_vrp(depot_loc, node_loc, demand, capacity)
end_time_nn = time.time()  # End timer for Nearest Neighbor

print("Execution time (Nearest Neighbor):", end_time_nn - start_time_nn, "seconds")
total_cost_nn = calculate_total_cost(nn_routes, node_loc, depot_loc)
print("Total cost (Nearest Neighbor):", total_cost_nn)

optimal_cost = 29888

# Porovnání s optimálním řešením, pokud je k dispozici
difference_nn = total_cost_nn - optimal_cost
print("Difference between nearest neighbor and optimal cost:", difference_nn)
