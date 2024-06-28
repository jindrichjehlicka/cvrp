import random
import vrplib
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


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


def initial_population(size, node_loc, demand, capacity):
    return [generate_initial_solution(node_loc, demand, capacity) for _ in range(size)]


def calculate_cost(solution, distance_matrix):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


def simulated_annealing(node_loc, demand, capacity, max_iterations, initial_temperature, cooling_rate):
    distance_matrix = calculate_distance_matrix(node_loc)
    current_solution = generate_initial_solution(node_loc, demand, capacity)
    current_cost = calculate_cost(current_solution, distance_matrix)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature
    cost_over_time = []

    for _ in range(max_iterations):
        neighbor = get_neighbor(current_solution)
        neighbor_cost = calculate_cost(neighbor, distance_matrix)
        if neighbor_cost < current_cost or random.random() < np.exp((current_cost - neighbor_cost) / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        temperature *= cooling_rate
        cost_over_time.append(best_cost)

    return best_solution, best_cost, cost_over_time


def get_neighbor(solution):
    neighbor = [route[:] for route in solution]
    if len(neighbor) > 1:
        route1, route2 = random.sample(neighbor, 2)
        if len(route1) > 3 and len(route2) > 3:
            idx1, idx2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return neighbor


class SimulatedAnnealingCV(BaseEstimator, RegressorMixin):
    def __init__(self, max_iterations=1000, initial_temperature=100, cooling_rate=0.99, runs=10):
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.runs = runs
        self.cost_over_runs_ = []

    def fit(self, X, y=None):
        costs = []
        self.cost_over_runs_ = []

        for instance in X:
            node_loc, demand, capacity, optimal_cost, instance_name = instance
            _, cost, cost_over_runs = simulated_annealing(
                node_loc, demand, capacity, self.max_iterations, self.initial_temperature, self.cooling_rate)
            costs.append(cost)
            # Calculate the difference between optimal cost and algorithm cost over time
            diff_over_runs = [optimal_cost - run_cost for run_cost in cost_over_runs]
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
    'max_iterations': [500, 1000, 1500],
    'initial_temperature': [50, 100, 150],
    'cooling_rate': [0.85, 0.9, 0.95]
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
    y = [d[3] for d in data]

    sa = SimulatedAnnealingCV()
    grid_search = GridSearchCV(estimator=sa, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    # Save epoch data
    epoch_data = []
    for params in grid_search.cv_results_['params']:
        for train_index, test_index in KFold(n_splits=3).split(X):
            sa = SimulatedAnnealingCV(**params)
            sa.fit([X[i] for i in train_index])
            for run_data in sa.cost_over_runs_:
                epoch_data.append({
                    "max_iterations": params['max_iterations'],
                    "initial_temperature": params['initial_temperature'],
                    "cooling_rate": params['cooling_rate'],
                    "epoch_data": run_data,
                    "instance_name": X[train_index[0]][4]  # Adding instance name
                })

    epoch_filename = f"simulated_annealing_epoch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["max_iterations", "initial_temperature", "cooling_rate", "epoch_data", "instance_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in epoch_data:
            writer.writerow(entry)

    print(f"Epoch data saved to {epoch_filename}")


if __name__ == "__main__":
    main()
