import csv
import numpy as np
import random
import vrplib
from datetime import datetime
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold


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


def tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand, capacity, runs=10):
    distance_matrix = calculate_distance_matrix(node_loc)
    n_nodes = len(node_loc)
    best_solution_overall = None
    best_cost_overall = float('inf')
    all_costs = []

    for _ in range(runs):
        current_solution = generate_initial_solution(n_nodes, demand, capacity)
        best_solution = current_solution
        best_cost = calculate_cost(current_solution, distance_matrix)
        tabu_list = []
        costs = []

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
            costs.append(best_cost)

        all_costs.append(costs)

        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_solution_overall = best_solution

    return best_solution_overall, best_cost_overall, all_costs


class TabuSearchCV(BaseEstimator, RegressorMixin):
    def __init__(self, max_iterations=100, tabu_size=10, neighborhood_size=20, runs=10):
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.neighborhood_size = neighborhood_size
        self.runs = runs
        self.cost_over_runs_ = []

    def fit(self, X, y=None):
        costs = []
        self.cost_over_runs_ = []

        for instance in X:
            node_loc, demand, capacity, optimal_cost, instance_name = instance
            _, cost, cost_over_runs = tabu_search(
                self.max_iterations, self.tabu_size, self.neighborhood_size, node_loc, demand, capacity, self.runs)
            costs.append(cost)
            # Calculate the difference between optimal cost and algorithm cost over time
            diff_over_runs = [[optimal_cost - run_cost for run_cost in run] for run in cost_over_runs]
            self.cost_over_runs_.append(diff_over_runs)

        self.best_cost_ = np.mean(costs)
        return self

    def predict(self, X):
        return np.array([self.best_cost_] * len(X))


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names


param_grid = {
    'max_iterations': [50, 100, 150],
    'tabu_size': [5, 10, 15],
    'neighborhood_size': [10, 20, 30]
}


def main():
    filename = "../../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)
    instances = instance_names[:100]

    data = []
    for instance_name in instances:
        instance = vrplib.read_instance(f"../../../Vrp-Set-XML100/instances/{instance_name}.vrp")
        solution = vrplib.read_solution(f"../../../Vrp-Set-XML100/solutions/{instance_name}.sol")
        optimal_cost = solution['cost']
        node_loc = instance['node_coord']
        depot_loc = node_loc[0]
        demand = instance['demand']
        capacity = instance['capacity']
        data.append((node_loc, demand, capacity, optimal_cost, instance_name))

    X = data
    y = [d[3] for d in data]  # Optimal costs as targets

    ts = TabuSearchCV()
    grid_search = GridSearchCV(estimator=ts, param_grid=param_grid, cv=3, n_jobs=-1, error_score='raise')
    grid_search.fit(X, y)

    # Save epoch data
    epoch_data = []
    for params in grid_search.cv_results_['params']:
        for train_index, test_index in KFold(n_splits=3).split(X):
            ts = TabuSearchCV(**params)
            ts.fit([X[i] for i in train_index])
            for run_data in ts.cost_over_runs_:
                for run in run_data:
                    epoch_data.append({
                        "max_iterations": params['max_iterations'],
                        "tabu_size": params['tabu_size'],
                        "neighborhood_size": params['neighborhood_size'],
                        "epoch_data": run,
                        "instance_name": X[train_index[0]][4]  # Adding instance name
                    })

    epoch_filename = f"tabu_search_epoch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["max_iterations", "tabu_size", "neighborhood_size", "epoch_data", "instance_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in epoch_data:
            writer.writerow(entry)

    print(f"Epoch data saved to {epoch_filename}")


if __name__ == "__main__":
    main()
