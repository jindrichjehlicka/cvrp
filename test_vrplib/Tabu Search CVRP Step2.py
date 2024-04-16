
import numpy as np
import random
import copy

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_total_distance(routes, distance_matrix):
    total_distance = 0
    for route in routes:
        for i in range(1, len(route)):
            total_distance += distance_matrix[route[i-1], route[i]]
        total_distance += distance_matrix[route[-1], route[0]]  # Return to depot
    return total_distance

def swap_2opt(route, i, k):
    assert i < k
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def get_neighbors(solution, distance_matrix):
    neighbors = []
    for route_index, route in enumerate(solution):
        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                new_route = swap_2opt(route, i, k)
                new_solution = copy.deepcopy(solution)
                new_solution[route_index] = new_route
                neighbors.append(new_solution)
    return neighbors

def tabu_search(initial_solution, distance_matrix, num_iterations, tabu_tenure):
    current_solution = initial_solution
    best_solution = initial_solution
    best_cost = calculate_total_distance(initial_solution, distance_matrix)
    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = get_neighbors(current_solution, distance_matrix)
        neighbors_costs = [calculate_total_distance(neighbor, distance_matrix) for neighbor in neighbors]
        best_neighbor_idx = np.argmin(neighbors_costs)
        best_neighbor = neighbors[best_neighbor_idx]

        if best_neighbor not in tabu_list:
            current_solution = best_neighbor
            current_cost = neighbors_costs[best_neighbor_idx]
            if current_cost < best_cost:
                best_solution = best_neighbor
                best_cost = current_cost
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        if iteration % 10 == 0:  # Dynamic tabu tenure adjustment (simplified example)
            tabu_tenure = random.randint(5, 15)

        print(f"Iteration {iteration+1}: Best Cost = {best_cost}")

    return best_solution, best_cost

# Placeholder for auxiliary functions like generate_initial_solution
