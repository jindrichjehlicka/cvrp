import numpy as np
import pandas as pd
import time
import vrplib
import random
import math
from joblib import Parallel, delayed


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


def generate_initial_solution(node_loc, demand, capacity):
    nodes = list(range(1, len(node_loc)))
    random.shuffle(nodes)
    routes = []
    current_route = [0]  # Start with depot
    current_load = 0

    for node in nodes:
        if current_load + demand[node] <= capacity:
            current_route.append(node)
            current_load += demand[node]
        else:
            current_route.append(0)  # Return to depot
            routes.append(current_route)
            current_route = [0, node]
            current_load = demand[node]
    current_route.append(0)  # Return to depot for the last route
    routes.append(current_route)
    return routes


def get_neighbor(solution):
    neighbor = [route[:] for route in solution]
    if len(neighbor) > 1:
        route1, route2 = random.sample(neighbor, 2)
        if len(route1) > 3 and len(route2) > 3:  # Ensure the routes have more than depot and one customer
            idx1, idx2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return neighbor


def simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand, capacity, epsilon=1e-10):
    distance_matrix = calculate_distance_matrix(node_loc)
    current_solution = generate_initial_solution(node_loc, demand, capacity)
    current_cost = calculate_cost(current_solution, distance_matrix)
    temperature = initial_temperature
    history = []

    for iteration in range(max_iterations):
        neighbor = get_neighbor(current_solution)
        neighbor_cost = calculate_cost(neighbor, distance_matrix)
        if neighbor_cost < current_cost or random.random() < math.exp(
                (current_cost - neighbor_cost) / (temperature + epsilon)):
            current_solution, current_cost = neighbor, neighbor_cost
        temperature *= cooling_rate
        history.append((iteration, current_cost))

    return current_solution, current_cost, history


def run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, max_iterations,
                         initial_temperature, cooling_rate, runs=20):
    instance_results = []
    for run in range(runs):
        start_time = time.time()
        best_solution, best_cost, history = simulated_annealing(max_iterations, initial_temperature, cooling_rate,
                                                                node_loc, demand, capacity)
        elapsed_time = time.time() - start_time
        for iteration, cost in history:
            cost_difference = cost - optimal_cost
            instance_results.append(
                ['Simulated Annealing', instance_name, run + 1, iteration, cost_difference, elapsed_time,
                 f'max_iter={max_iterations},init_temp={initial_temperature},cooling_rate={cooling_rate}'])
    return instance_results


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()
    return instance_names


filename = "../instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instance_names = instance_names[101:601]


def process_instance(instance_name, max_iterations, initial_temperature, cooling_rate):
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    return run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, max_iterations,
                                initial_temperature, cooling_rate)


initial_temperature = 1000
cooling_rate = 0.95
max_iterations = 150

for i in range(0, len(instance_names), 100):
    chunk_instance_names = instance_names[i:i + 100]

    results_list = Parallel(n_jobs=-1)(
        delayed(process_instance)(instance_name, max_iterations, initial_temperature, cooling_rate) for instance_name in
        chunk_instance_names)

    flattened_results = [item for sublist in results_list for item in sublist]

    results = pd.DataFrame(flattened_results,
                           columns=['Algorithm', 'Instance', 'Run', 'Iteration', 'Cost Difference', 'Time',
                                    'Parameters'])

    chunk_number = (i // 100) + 1
    results.to_csv(f'simulated_annealing_performance_chunk_{chunk_number}.csv', index=False)
