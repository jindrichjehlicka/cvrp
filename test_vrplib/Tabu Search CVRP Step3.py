
import numpy as np
import random
import copy

# Assuming auxiliary functions from previous steps are defined...

def intensify(current_solution, best_solutions, distance_matrix):
    # Placeholder for intensification logic
    return current_solution

def diversify(current_solution, distance_matrix, iteration):
    # Placeholder for diversification logic
    if iteration % 20 == 0:  # Every 20 iterations, perform a diversification
        random_route = random.choice(current_solution)
        random.shuffle(random_route)  # Example: Shuffle a random route
        current_solution[current_solution.index(random_route)] = random_route
    return current_solution

def dynamic_tabu_tenure(iteration, stuck_counter):
    if stuck_counter > 10:
        return min(20, 5 + iteration % 10)  # Increase tenure to explore more deeply
    else:
        return max(5, 10 - iteration % 5)  # Decrease tenure to allow more frequent changes

def tabu_search_with_strategies(initial_solution, distance_matrix, num_iterations, initial_tabu_tenure):
    current_solution = initial_solution
    best_solution = initial_solution
    best_cost = calculate_total_distance(initial_solution, distance_matrix)
    tabu_list = []
    stuck_counter = 0  # Tracks iterations without improvement

    for iteration in range(num_iterations):
        tabu_tenure = dynamic_tabu_tenure(iteration, stuck_counter)
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
                stuck_counter = 0  # Reset counter on improvement
            else:
                stuck_counter += 1  # Increment counter if no improvement
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        current_solution = diversify(current_solution, distance_matrix, iteration)

        print(f"Iteration {iteration+1}: Best Cost = {best_cost}, Tabu Tenure = {tabu_tenure}")

    return best_solution, best_cost
