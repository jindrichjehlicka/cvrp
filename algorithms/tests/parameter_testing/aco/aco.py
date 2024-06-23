import random
import vrplib
import itertools
import time
import csv
from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

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

class CVRPProblem(Problem):
    def __init__(self, node_loc, demand, capacity):
        super().__init__(n_var=len(node_loc) - 1, n_obj=1, n_constr=0, xl=0, xu=len(node_loc) - 1, type_var=int)
        self.node_loc = node_loc
        self.demand = demand
        self.capacity = capacity

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [self.evaluate_route(route) for route in x]

    def evaluate_route(self, route):
        distance_matrix = calculate_distance_matrix(self.node_loc)
        # Ensure route is a list of integers and within valid range
        route = [max(0, min(int(r), len(self.node_loc) - 1)) for r in route]
        solution = [[0] + route + [0]]
        return calculate_cost(solution, distance_matrix)


class AntColonyOptimizationCV(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=50, generations=100, alpha=1.0, beta=2.0, rho=0.5):
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.cost_over_time_ = []

    def fit(self, X, y=None):
        costs = []
        self.cost_over_time_ = []  # Initialize cost over time
        max_generations = self.generations

        for instance in X:
            node_loc, demand, capacity, optimal_cost = instance
            problem = CVRPProblem(node_loc, demand, capacity)

            algorithm = GA(pop_size=self.population_size, eliminate_duplicates=True)

            res = minimize(problem, algorithm, ('n_gen', self.generations), verbose=False)

            best_cost = res.F[0]
            costs.append(best_cost)
            self.cost_over_time_.append([x[0] for x in res.history['pop']])

        self.best_cost_ = np.mean(costs)
        return self

    def predict(self, X):
        return np.array([self.best_cost_] * len(X))

def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names

param_grid = {
    'population_size': [50, 100, 150],
    'generations': [25, 50, 100],
    'alpha': [1.0, 2.0, 3.0],
    'beta': [2.0, 5.0, 10.0],
    'rho': [0.1, 0.5, 0.9]
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

    aco = AntColonyOptimizationCV()
    grid_search = GridSearchCV(estimator=aco, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    # Save grid search results
    results = pd.DataFrame(grid_search.cv_results_)
    csv_filename = f"aco_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Save best parameters
    best_params = grid_search.best_params_
    best_params_filename = f"best_aco_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([best_params]).to_csv(best_params_filename, index=False)
    print(f"Best parameters saved to {best_params_filename}")

    # Save epoch data
    epoch_data = []
    for params in grid_search.cv_results_['params']:
        fold_epoch_data = []
        for train_index, test_index in KFold(n_splits=3).split(X):
            aco = AntColonyOptimizationCV(**params)
            aco.fit([X[i] for i in train_index])
            fold_epoch_data.append(aco.cost_over_time_)

        # Calculate average epoch data across folds
        max_generations = max(len(epoch) for fold in fold_epoch_data for epoch in fold)
        avg_epoch_data = np.mean([np.pad(epoch, (0, max_generations - len(epoch)), 'edge')
                                  for fold in fold_epoch_data for epoch in fold], axis=0)
        epoch_data.append(avg_epoch_data.tolist())

    epoch_filename = f"aco_epoch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["population_size", "generations", "alpha", "beta", "rho", "epoch_data"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for params, epochs in zip(grid_search.cv_results_['params'], epoch_data):
            writer.writerow({
                "population_size": params['population_size'],
                "generations": params['generations'],
                "alpha": params['alpha'],
                "beta": params['beta'],
                "rho": params['rho'],
                "epoch_data": epochs
            })

    print(f"Epoch data saved to {epoch_filename}")

if __name__ == "__main__":
    main()
