import random
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

    cost_over_time = []

    for iteration in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution, demand, capacity, neighborhood_size)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighborhood:
            neighbor_cost = calculate_cost(neighbor, distance_matrix)
            if neighbor not in tabu_list and neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost

        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_solution = best_neighbor
        cost_over_time.append(best_cost)

    return best_solution, best_cost, cost_over_time


def process_epoch_data_for_param_combo(X, max_iterations, tabu_size, neighborhood_size):
    epoch_data = []
    for train_index, test_index in KFold(n_splits=3).split(X):
        for instance in [X[i] for i in train_index]:
            node_loc, demand, capacity, optimal_cost = instance
            _, _, cost_over_time = tabu_search(
                max_iterations, tabu_size, neighborhood_size, node_loc, demand, capacity)
            diff_over_runs = [optimal_cost - run_cost for run_cost in cost_over_time]
            epoch_data.append({
                "max_iterations": max_iterations,
                "tabu_size": tabu_size,
                "neighborhood_size": neighborhood_size,
                "epoch_data": diff_over_runs
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
        'max_iterations': [200, 300, 500],
        'tabu_size': [5, 15, 30, ],
        'neighborhood_size': [5, 10, 15, ]
    }

    param_combinations = list(
        product(param_grid['max_iterations'], param_grid['tabu_size'], param_grid['neighborhood_size']))
    epoch_data_list = Parallel(n_jobs=-1)(
        delayed(process_epoch_data_for_param_combo)(X, max_iter, tabu_sz, neigh_sz) for max_iter, tabu_sz, neigh_sz in
        param_combinations
    )

    epoch_data = [item for sublist in epoch_data_list for item in sublist]  # Flatten the list of lists

    epoch_filename = f"tabu_search_epoch_data_chunk_{chunk_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["max_iterations", "tabu_size", "neighborhood_size", "epoch_data"]
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
