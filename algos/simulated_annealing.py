import random
import math


def estimate_initial_temperature(node_loc, demand, capacity, num_samples=100):
    import numpy as np
    costs = []
    for _ in range(num_samples):
        # Generate two random solutions
        solution1 = generate_initial_solution(node_loc, demand, capacity)
        solution2 = generate_initial_solution(node_loc, demand, capacity)

        cost1 = calculate_total_distance(solution1, node_loc)
        cost2 = calculate_total_distance(solution2, node_loc)

        # Calculate the absolute difference
        costs.append(abs(cost1 - cost2))

    # Use the mean of cost differences as a base for initial temperature
    mean_difference = np.mean(costs)
    initial_temperature = mean_difference * 0.2  # Adjust the factor based on acceptance needs
    return initial_temperature

def calculate_total_distance(routes, node_loc):
    """ Calculate the total distance of the vehicle routes, including return to the depot """
    total_distance = 0
    for route in routes:
        route_distance = 0
        last_node = 0  # Start at the depot
        for node in route:
            route_distance += math.dist(node_loc[last_node], node_loc[node])
            last_node = node
        route_distance += math.dist(node_loc[last_node], node_loc[0])  # Return to depot
        total_distance += route_distance
    return total_distance


def generate_initial_solution(node_loc, demand, capacity):
    """ Generate an initial feasible solution using a simple greedy heuristic """
    routes = []
    current_route = []
    current_load = 0
    depot = 0  # Assuming the depot is the first node

    for node in range(1, len(node_loc)):  # Start from 1 as 0 is the depot
        if current_load + demand[node] <= capacity:
            current_route.append(node)
            current_load += demand[node]
        else:
            routes.append(current_route)
            current_route = [node]
            current_load = demand[node]
    if current_route:
        routes.append(current_route)

    # Adding depot to the start and end of each route
    routes = [[depot] + route + [depot] for route in routes]
    return routes


def get_neighbor(solution):
    """ Generate a neighbor solution by making a small change in the current solution """
    if len(solution) > 1:
        route1, route2 = random.sample(solution, 2)
        if route1 and route2:
            # Ensure not to pick the depot
            node1 = random.choice(route1[1:-1])
            node2 = random.choice(route2[1:-1])
            # Swap nodes
            idx1, idx2 = route1.index(node1), route2.index(node2)
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return solution


def simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand, capacity):
    """ Perform simulated annealing to find a solution to the CVRP """
    current_solution = generate_initial_solution(node_loc, demand, capacity)
    current_cost = calculate_total_distance(current_solution, node_loc)
    temperature = initial_temperature

    for _ in range(max_iterations):
        neighbor = get_neighbor([route[:] for route in current_solution])  # Deep copy of current solution
        neighbor_cost = calculate_total_distance(neighbor, node_loc)
        if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / temperature):
            current_solution, current_cost = neighbor, neighbor_cost
        temperature *= cooling_rate

    return current_solution, current_cost
