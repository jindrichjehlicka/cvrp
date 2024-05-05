import numpy as np


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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


def greedy_vrp(depot_loc, node_loc, demand, capacity):
    """Solves a CVRP instance using a Greedy Algorithm."""
    num_customers = len(node_loc)  # Total number of customers
    routes = []  # List to store routes
    visited = set()  # Set to track visited customers

    while len(visited) < num_customers:  # Continue until all customers are visited
        route = []  # Current route
        current_load = 0  # Load of the current vehicle
        current_location = depot_loc  # Start from the depot
        while True:
            nearest_distance = float('inf')  # Initialize with infinity
            nearest_customer = None  # To store the nearest customer's index
            for i, location in enumerate(node_loc):  # Iterate through customer locations
                if i not in visited and current_load + demand[i] <= capacity:  # Check if the customer can be visited
                    distance = euclidean_distance(current_location, location)  # Calculate distance
                    if distance < nearest_distance:  # Check if this is the nearest customer so far
                        nearest_distance = distance
                        nearest_customer = i
            if nearest_customer is None:
                break  # Break if no further customers can be added
            visited.add(nearest_customer)  # Mark the customer as visited
            route.append(nearest_customer)  # Add to the current route
            current_load += demand[nearest_customer]  # Update load
            current_location = node_loc[nearest_customer]  # Update current location

        routes.append(route)  # Add the current route to the list of routes

    # Calculate the total cost of the routes
    total_cost = calculate_total_cost(routes, node_loc, depot_loc)

    return routes, total_cost
