import numpy as np
import random
import math
from concurrent.futures import ThreadPoolExecutor


def euc_2d(c1, c2):
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


def calculate_cost(routes, depot_loc, node_loc):
    total_cost = 0
    for route in routes:
        if route:
            route_cost = euc_2d(depot_loc, node_loc[route[0]])
            for i in range(1, len(route)):
                route_cost += euc_2d(node_loc[route[i - 1]], node_loc[route[i]])
            route_cost += euc_2d(node_loc[route[-1]], depot_loc)
            total_cost += route_cost
    return total_cost


def initialize_pheromone(n, initial_pheromone):
    return np.full((n, n), initial_pheromone)


def aco_algorithm(depot_loc, node_loc, demand, capacity, num_ants=5, iterations=20, decay=0.05, alpha=1, beta=2):
    num_nodes = len(node_loc)
    pheromone = initialize_pheromone(num_nodes, 1.0)
    all_nodes = list(range(1, num_nodes))  # Exclude depot
    best_cost = float('inf')
    best_solution = []

    for iteration in range(iterations):
        solutions = []
        for ant in range(num_ants):
            solution = []
            remaining_nodes = set(all_nodes)
            while remaining_nodes:
                route = []
                current_node = 0  # Start from depot
                load = 0
                while remaining_nodes:
                    probabilities = []
                    for node in remaining_nodes:
                        if load + demand[node] <= capacity:
                            probability = (pheromone[current_node][node] ** alpha) * (
                                        (1.0 / euc_2d(node_loc[current_node], node_loc[node])) ** beta)
                            probabilities.append((probability, node))
                    if not probabilities:
                        break
                    probabilities.sort(reverse=True)
                    selected_node = probabilities[0][1]
                    route.append(selected_node)
                    remaining_nodes.remove(selected_node)
                    load += demand[selected_node]
                    current_node = selected_node
                solution.append(route)
            solutions.append(solution)
            cost = calculate_cost(solution, depot_loc, node_loc)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        for i, j in np.ndindex(pheromone.shape):
            pheromone[i][j] *= (1 - decay)
        for solution in solutions:
            route_cost = calculate_cost(solution, depot_loc, node_loc)
            for route in solution:
                for i in range(len(route) - 1):
                    pheromone[route[i]][route[i + 1]] += 1.0 / route_cost

    return best_solution, best_cost


def generate_neighborhood(solution, neighborhood_size=20):
    neighbors = []
    for _ in range(neighborhood_size):
        neighbor = [route[:] for route in solution]
        if len(neighbor) > 1:
            r1, r2 = random.sample(range(len(neighbor)), 2)
            if neighbor[r1] and neighbor[r2]:
                node1 = random.choice(neighbor[r1])
                node2 = random.choice(neighbor[r2])
                idx1, idx2 = neighbor[r1].index(node1), neighbor[r2].index(node2)
                neighbor[r1][idx1], neighbor[r2][idx2] = neighbor[r2][idx2], neighbor[r1][idx1]
        neighbors.append(neighbor)
    return neighbors


def tabu_search(initial_solution, node_loc, demand, capacity, max_iterations=100, tabu_size=50, neighborhood_size=20):
    best_solution = initial_solution
    best_cost = calculate_cost(initial_solution, node_loc[0], node_loc)
    tabu_list = []

    current_solution = initial_solution
    current_cost = best_cost

    for _ in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution, neighborhood_size)
        neighborhood = sorted(neighborhood, key=lambda x: calculate_cost(x, node_loc[0], node_loc))

        for neighbor in neighborhood:
            neighbor_cost = calculate_cost(neighbor, node_loc[0], node_loc)
            if neighbor not in tabu_list and neighbor_cost < current_cost:
                current_solution = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
                break

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_cost


def aco_ts_hybrid(depot_loc, node_loc, demand, capacity, num_ants=5, iterations=20, decay=0.05, alpha=1, beta=2,
                  max_tabu_iterations=100, tabu_size=50, neighborhood_size=20):
    aco_solution, aco_cost = aco_algorithm(depot_loc, node_loc, demand, capacity, num_ants, iterations, decay, alpha,
                                           beta)
    ts_solution, ts_cost = tabu_search(aco_solution, node_loc, demand, capacity, max_tabu_iterations, tabu_size,
                                       neighborhood_size)
    return ts_solution, ts_cost
