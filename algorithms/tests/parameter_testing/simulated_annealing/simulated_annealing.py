import random
import math
import vrplib
import csv
import numpy as np
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
    """ Calculate the total distance of the vehicle routes, including return to the depot """
    total_distance = 0
    for route in routes:
        route_distance = 0
        last_node = 0  # Start at the depot
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


def simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand, capacity, epsilon=1e-10,
                        runs=10):
    best_solution_overall = None
    best_cost_overall = float('inf')
    all_costs = []

    for _ in range(runs):
        current_solution = generate_initial_solution(node_loc, demand, capacity)
        current_cost = calculate_total_distance(current_solution, node_loc)
        temperature = initial_temperature

        for _ in range(max_iterations):
            neighbor = get_neighbor([route[:] for route in current_solution])  # Deep copy of current solution
            neighbor_cost = calculate_total_distance(neighbor, node_loc)
            if neighbor_cost < current_cost or random.random() < math.exp(
                    (current_cost - neighbor_cost) / (temperature + epsilon)):
                current_solution, current_cost = neighbor, neighbor_cost
            temperature *= cooling_rate

        all_costs.append(current_cost)

        if current_cost < best_cost_overall:
            best_cost_overall = current_cost
            best_solution_overall = current_solution

    return best_solution_overall, best_cost_overall, all_costs


class SimulatedAnnealingCV(BaseEstimator, RegressorMixin):
    def __init__(self, max_iterations=100, initial_temperature=1000, cooling_rate=0.95, runs=10):
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.runs = runs
        self.cost_over_runs_ = []

    def fit(self, X, y=None):
        costs = []
        self.cost_over_runs_ = []

        for instance in X:
            node_loc, demand, capacity, optimal_cost = instance
            initial_temp = estimate_initial_temperature(node_loc, demand, capacity)
            _, cost, cost_over_runs = simulated_annealing(
                self.max_iterations, initial_temp, self.cooling_rate, node_loc, demand, capacity, runs=self.runs)
            costs.append(cost)
            self.cost_over_runs_.append(cost_over_runs)

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
    'initial_temperature': [500, 1000, 1500],
    'cooling_rate': [0.85, 0.90, 0.95]
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
        data.append((node_loc, demand, capacity, optimal_cost))

    X = data
    y = [d[3] for d in data]  # Optimal costs as targets

    sa = SimulatedAnnealingCV()
    grid_search = GridSearchCV(estimator=sa, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    results = pd.DataFrame(grid_search.cv_results_)
    csv_filename = f"simulated_annealing_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    best_params = grid_search.best_params_
    best_params_filename = f"best_simulated_annealing_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([best_params]).to_csv(best_params_filename, index=False)
    print(f"Best parameters saved to {best_params_filename}")


if __name__ == "__main__":
    main()
