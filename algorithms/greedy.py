import numpy as np


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_total_cost(routes, node_loc, depot_loc):
    total_cost = 0
    for route in routes:
        route_cost = 0
        last_location = depot_loc
        for node in route:
            route_cost += euclidean_distance(last_location, node_loc[node])
            last_location = node_loc[node]
        route_cost += euclidean_distance(last_location, depot_loc)  # Return to depot
        total_cost += route_cost
    return total_cost


def dijkstra_cvrp(depot_loc, node_loc, demand, capacity):
    num_customers = len(node_loc)
    routes = []
    visited = set()

    while len(visited) < num_customers:
        route = []
        current_load = 0
        current_location = depot_loc
        while True:
            min_distance = float('inf')
            next_customer = None
            for i, location in enumerate(node_loc):
                if i not in visited and current_load + demand[i] <= capacity:
                    distance = euclidean_distance(current_location, location)
                    if distance < min_distance:
                        min_distance = distance
                        next_customer = i
            if next_customer is None:
                break  # No valid next node, end route
            # Update for next iteration
            visited.add(next_customer)
            route.append(next_customer)
            current_load += demand[next_customer]
            current_location = node_loc[next_customer]

        if route:  # Only add non-empty routes
            routes.append(route)

    total_cost = calculate_total_cost(routes, node_loc, depot_loc)
    return routes, total_cost
