import numpy as np
import random


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


def is_feasible(route, demand, capacity):
    return sum(demand[node] for node in route if node != 0) <= capacity


def generate_neighborhood(current_solution, demand, capacity, neighborhood_size):
    neighbors = []
    n_routes = len(current_solution)

    while len(neighbors) < neighborhood_size:
        if n_routes > 1:
            r1, r2 = random.sample(range(n_routes), 2)
            route1, route2 = current_solution[r1], current_solution[r2]

            if len(route1) > 2 and len(route2) > 2:
                i1, i2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
                new_route1, new_route2 = route1[:], route2[:]
                new_route1[i1], new_route2[i2] = new_route2[i2], new_route1[i1]

                if is_feasible(new_route1, demand, capacity) and is_feasible(new_route2, demand, capacity):
                    new_solution = current_solution[:]
                    new_solution[r1], new_solution[r2] = new_route1, new_route2
                    neighbors.append(new_solution)

        route_index = random.randint(0, n_routes - 1)
        route = current_solution[route_index]
        if len(route) > 3:
            start, end = sorted(random.sample(range(1, len(route) - 1), 2))
            new_route = route[:start] + route[start:end + 1][::-1] + route[end + 1:]
            if is_feasible(new_route, demand, capacity):
                new_solution = current_solution[:]
                new_solution[route_index] = new_route
                neighbors.append(new_solution)

    return neighbors


def calculate_cost(solution, distance_matrix):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


def generate_initial_solution(n_nodes, demand, capacity):
    nodes = list(range(1, n_nodes))
    solution = []
    route = [0]
    current_load = 0

    for node in nodes:
        if current_load + demand[node] <= capacity:
            route.append(node)
            current_load += demand[node]
        else:
            route.append(0)
            solution.append(route)
            route = [0, node]
            current_load = demand[node]
    route.append(0)
    solution.append(route)
    return solution


def tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand, capacity):
    distance_matrix = calculate_distance_matrix(node_loc)
    n_nodes = len(node_loc)
    current_solution = generate_initial_solution(n_nodes, demand, capacity)
    best_solution = current_solution
    best_cost = calculate_cost(current_solution, distance_matrix)
    tabu_list = []

    for iteration in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution, demand, capacity, neighborhood_size)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list and calculate_cost(neighbor, distance_matrix) < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = calculate_cost(neighbor, distance_matrix)

        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost

        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_solution = best_neighbor

    return best_solution, best_cost
