import numpy as np
import pandas as pd
import time
import vrplib
import random
from joblib import Parallel, delayed


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
    random.shuffle(nodes)
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
    history = []

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
        history.append((iteration, best_cost))

    return best_solution, best_cost, history


def run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, max_iterations,
                         tabu_size, neighborhood_size, runs=20):
    instance_results = []
    for run in range(runs):
        start_time = time.time()
        best_solution, best_cost, history = tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand,
                                                        capacity)
        elapsed_time = time.time() - start_time
        for iteration, cost in history:
            cost_difference = cost - optimal_cost
            instance_results.append(
                ['Tabu Search', instance_name, run + 1, iteration, cost_difference, elapsed_time,
                 f'max_iter={max_iterations},tabu_size={tabu_size},neighborhood_size={neighborhood_size}'])
    return instance_results


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()
    return instance_names


filename = "../instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instance_names = instance_names[101:601]


def process_instance(instance_name, max_iterations, tabu_size, neighborhood_size):
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    return run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, max_iterations,
                                tabu_size, neighborhood_size)


max_iterations = 50
tabu_size = 5
neighborhood_size = 10

for i in range(0, len(instance_names), 100):
    chunk_instance_names = instance_names[i:i + 100]

    results_list = Parallel(n_jobs=-1)(
        delayed(process_instance)(instance_name, max_iterations, tabu_size, neighborhood_size) for instance_name in
        chunk_instance_names)

    flattened_results = [item for sublist in results_list for item in sublist]

    results = pd.DataFrame(flattened_results,
                           columns=['Algorithm', 'Instance', 'Run', 'Iteration', 'Cost Difference', 'Time',
                                    'Parameters'])

    chunk_number = (i // 100) + 1
    results.to_csv(f'tabu_search_performance_chunk_{chunk_number}.csv', index=False)
