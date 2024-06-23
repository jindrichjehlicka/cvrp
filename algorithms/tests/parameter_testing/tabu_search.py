import numpy as np
import random
import vrplib
import itertools
import time
import csv
from joblib import Parallel, delayed,cpu_count
from datetime import datetime


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


# Function to load instance names from a file
def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names


# Define the parameter grid for Tabu Search
param_grid = {
    'max_iterations': [100, 250, 500],
    'tabu_size': [5, 10, 20, 30],
    'neighborhood_size': [5, 10, 20, 30]
}


# Function to evaluate a single parameter combination on a single dataset
def evaluate(instance_name, params, n_runs=20):
    max_iterations, tabu_size, neighborhood_size = params
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    costs = []
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _, cost = tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand, capacity)
        end_time = time.time()
        costs.append(cost)
        times.append(end_time - start_time)

    avg_cost = np.mean(costs)
    avg_time = np.mean(times)

    return {
        "instance_name": instance_name,
        "max_iterations": max_iterations,
        "tabu_size": tabu_size,
        "neighborhood_size": neighborhood_size,
        "optimal_cost": optimal_cost,
        "final_cost": avg_cost,
        "time": avg_time,
        "runs": n_runs
    }


# Main function to perform the grid search and save results to CSV
def main():
    filename = "../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)
    instances = instance_names[:200]

    # Create the list of all parameter combinations
    param_combinations = list(
        itertools.product(param_grid['max_iterations'], param_grid['tabu_size'], param_grid['neighborhood_size']))

    # Evaluate all parameter combinations on all instances in parallel
    results = Parallel(n_jobs=10)(delayed(evaluate)(instance_name, params)
                                  for instance_name in instances
                                  for params in param_combinations)

    # Save results to CSV
    csv_filename = f"tabu_search_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ["instance_name", "max_iterations", "tabu_size", "neighborhood_size",
                      "optimal_cost", "final_cost", "time", "runs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
