import random
import math
import vrplib
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from itertools import product
from joblib import Parallel, delayed


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


def estimate_initial_temperature(node_loc, demand, capacity, num_samples=100):
    costs = []
    for _ in range(num_samples):
        solution1 = generate_initial_solution(node_loc, demand, capacity)
        solution2 = generate_initial_solution(node_loc, demand, capacity)

        cost1 = calculate_total_distance(solution1, node_loc)
        cost2 = calculate_total_distance(solution2, node_loc)

        costs.append(abs(cost1 - cost2))

    mean_difference = np.mean(costs)
    initial_temperature = mean_difference * 0.2
    return initial_temperature


def calculate_total_distance(routes, node_loc):
    total_distance = 0
    for route in routes:
        route_distance = 0
        last_node = 0
        for node in route:
            route_distance += math.dist(node_loc[last_node], node_loc[node])
            last_node = node
        route_distance += math.dist(node_loc[last_node], node_loc[0])
        total_distance += route_distance
    return total_distance


def generate_initial_solution(node_loc, demand, capacity):
    routes = []
    current_route = []
    current_load = 0
    depot = 0

    for node in range(1, len(node_loc)):
        if current_load + demand[node] <= capacity:
            current_route.append(node)
            current_load += demand[node]
        else:
            routes.append(current_route)
            current_route = [node]
            current_load = demand[node]
    if current_route:
        routes.append(current_route)

    routes = [[depot] + route + [depot] for route in routes]
    return routes


def get_neighbor(solution):
    if len(solution) > 1:
        route1, route2 = random.sample(solution, 2)
        if route1 and route2:
            node1 = random.choice(route1[1:-1])
            node2 = random.choice(route2[1:-1])
            idx1, idx2 = route1.index(node1), route2.index(node2)
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return solution


def simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand, capacity, optimal_cost, epsilon=1e-10):
    current_solution = generate_initial_solution(node_loc, demand, capacity)
    current_cost = calculate_total_distance(current_solution, node_loc)
    temperature = initial_temperature

    cost_over_time = []

    for _ in range(max_iterations):
        neighbor = get_neighbor([route[:] for route in current_solution])
        neighbor_cost = calculate_total_distance(neighbor, node_loc)
        if neighbor_cost < current_cost or random.random() < math.exp(
                (current_cost - neighbor_cost) / (temperature + epsilon)):
            current_solution, current_cost = neighbor, neighbor_cost
        temperature *= cooling_rate
        cost_over_time.append(optimal_cost - current_cost)

    return current_solution, current_cost, cost_over_time


def process_epoch_data_for_param_combo(X, max_iterations, initial_temperature, cooling_rate):
    epoch_data = []
    for train_index, test_index in KFold(n_splits=3).split(X):
        for instance in [X[i] for i in train_index]:
            node_loc, demand, capacity, optimal_cost = instance
            _, _, cost_over_time = simulated_annealing(
                max_iterations, initial_temperature, cooling_rate, node_loc, demand, capacity, optimal_cost)
            epoch_data.append({
                "max_iterations": max_iterations,
                "initial_temperature": initial_temperature,
                "cooling_rate": cooling_rate,
                "epoch_data": cost_over_time
            })
    return epoch_data


def process_and_save_epoch_data(instance_names_chunk, chunk_number):
    data = []
    for instance_name in instance_names_chunk:
        instance = vrplib.read_instance(f"../../../Vrp-Set-XML100/instances/{instance_name}.vrp")
        solution = vrplib.read_solution(f"../../../Vrp-Set-XML100/solutions/{instance_name}.sol")
        optimal_cost = solution['cost']
        node_loc = instance['node_coord']
        depot_loc = node_loc[0]
        demand = instance['demand']
        capacity = instance['capacity']
        data.append((node_loc, demand, capacity, optimal_cost))

    X = data

    param_grid = {
        'max_iterations': [500,1000,1500],
        'initial_temperature': [500, 1000, 1500],
        'cooling_rate': [0.99, 0.97, 0.9]
    }

    param_combinations = list(
        product(param_grid['max_iterations'], param_grid['initial_temperature'], param_grid['cooling_rate']))
    epoch_data_list = Parallel(n_jobs=-1)(
        delayed(process_epoch_data_for_param_combo)(X, max_iter, init_temp, cool_rate) for max_iter, init_temp, cool_rate in
        param_combinations
    )

    epoch_data = [item for sublist in epoch_data_list for item in sublist]

    epoch_filename = f"simulated_annealing_epoch_data_chunk_{chunk_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["max_iterations", "initial_temperature", "cooling_rate", "epoch_data"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in epoch_data:
            writer.writerow(entry)

    print(f"Epoch data for chunk {chunk_number} saved to {epoch_filename}")


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names


def main():
    filename = "../../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)

    Parallel(n_jobs=-1)(
        delayed(process_and_save_epoch_data)(instance_names[i:i + 10], (i // 10) + 1)
        for i in range(0, 100, 10)
    )


if __name__ == "__main__":
    main()
