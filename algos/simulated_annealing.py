import numpy as np
import random
import math
from greedy import greedy_vrp


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


def calculate_cost(solution, distance_matrix):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


def initial_solution(depot_loc, node_loc, demand, capacity):
    return greedy_vrp(depot_loc, node_loc, demand, capacity)


def is_feasible(route, demand, capacity):
    return sum(demand[node] for node in route if node != 0) <= capacity


def generate_neighborhood(current_solution, demand, capacity):
    neighbors = []
    for i in range(len(current_solution)):
        for j in range(i + 1, len(current_solution)):
            neighbor_solution = current_solution[:]
            if len(current_solution[i]) > 1 and len(current_solution[j]) > 1:
                node1 = random.choice(current_solution[i][1:-1])  # Exclude depot
                node2 = random.choice(current_solution[j][1:-1])  # Exclude depot
                neighbor_solution[i][current_solution[i].index(node1)] = node2
                neighbor_solution[j][current_solution[j].index(node2)] = node1
                if is_feasible(neighbor_solution[i], demand, capacity) and is_feasible(neighbor_solution[j], demand,
                                                                                       capacity):
                    neighbors.append(neighbor_solution)
    return neighbors


def simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand,
                        capacity):
    distance_matrix = calculate_distance_matrix(node_loc)
    current_solution = initial_solution(node_loc[0], node_loc, demand, capacity)
    best_solution = current_solution
    best_cost = calculate_cost(current_solution, distance_matrix)
    temperature = initial_temperature

    for iteration in range(max_iterations):
        neighbor_solution = generate_neighborhood(current_solution, demand, capacity)
        neighbor_cost = calculate_cost(neighbor_solution, distance_matrix)

        if neighbor_cost < best_cost:
            best_solution = neighbor_solution
            best_cost = neighbor_cost
        elif math.exp(
                (calculate_cost(current_solution, distance_matrix) - neighbor_cost) / temperature) > random.random():
            current_solution = neighbor_solution

        temperature *= cooling_rate
        print(f"Iteration {iteration + 1}, Best cost: {best_cost}")

    return best_solution, best_cost
