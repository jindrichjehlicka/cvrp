import numpy as np


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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
