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


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


def generate_solution(node_loc, demand, capacity):
    nodes = list(range(1, len(node_loc)))  # start from 1 to skip depot
    random.shuffle(nodes)
    solution = []
    current_capacity = capacity
    route = [0]  # start from depot

    for node in nodes:
        if demand[node] <= current_capacity:
            route.append(node)
            current_capacity -= demand[node]
        else:
            route.append(0)  # return to depot
            solution.append(route)
            route = [0, node]
            current_capacity = capacity - demand[node]
    route.append(0)  # end last route at depot
    solution.append(route)
    return solution


def initial_population(size, node_loc, demand, capacity):
    return [generate_solution(node_loc, demand, capacity) for _ in range(size)]


def calculate_cost(solution, distance_matrix):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


def fitness(solution, distance_matrix):
    return 1 / calculate_cost(solution, distance_matrix)  # Higher fitness is better


def select(population, fitnesses, k=3):
    selected = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [x for x, _ in selected[:k]]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:]


def mutate(solution, mutation_rate=0.01):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(solution) - 1)
            solution[i], solution[j] = solution[j], solution[i]
    return solution


def genetic_algorithm(population_size, generations, node_loc, demand, capacity, mutation_rate):
    distance_matrix = calculate_distance_matrix(node_loc)
    pop = initial_population(population_size, node_loc, demand, capacity)
    best_solution = None
    best_cost = float('inf')
    cost_over_time = []

    for generation in range(generations):
        fitnesses = [fitness(sol, distance_matrix) for sol in pop]
        current_best_index = np.argmax(fitnesses)
        current_best_solution = pop[current_best_index]
        current_best_solution_cost = calculate_cost(current_best_solution, distance_matrix)

        if current_best_solution_cost < best_cost:
            best_cost = current_best_solution_cost
            best_solution = current_best_solution

        cost_over_time.append(best_cost)

        selected = select(pop, fitnesses, k=int(population_size / 2))
        next_gen = []
        while len(next_gen) < population_size:
            p1, p2 = random.sample(selected, 2)
            offspring = crossover(p1, p2)
            offspring = mutate(offspring, mutation_rate)
            next_gen.append(offspring)
        pop = next_gen

    return best_solution, best_cost, cost_over_time


class GeneticAlgorithmCV(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=50, generations=100, mutation_rate=0.01):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.cost_over_time_ = []

    def fit(self, X, y=None):
        costs = []
        self.cost_over_time_ = []  # Initialize cost over time
        max_generations = self.generations

        for instance in X:
            node_loc, demand, capacity, optimal_cost = instance
            _, cost, cost_over_time = genetic_algorithm(
                self.population_size, self.generations, node_loc, demand, capacity, self.mutation_rate)
            costs.append(cost)

            # Pad cost_over_time to max_generations length
            if len(cost_over_time) < max_generations:
                cost_over_time += [cost_over_time[-1]] * (max_generations - len(cost_over_time))
            self.cost_over_time_.append(cost_over_time)

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
    'mutation_rate': [0.01, 0.05, 0.1]
}


def main():
    filename = "../../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)
    instances = instance_names[:200]

    data = []
    for instance_name in instances:
        instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
        solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
        optimal_cost = solution['cost']
        node_loc = instance['node_coord']
        depot_loc = node_loc[0]
        demand = instance['demand']
        capacity = instance['capacity']
        data.append((node_loc, demand, capacity, optimal_cost))

    X = data
    y = [d[3] for d in data]  # Optimal costs as targets

    ga = GeneticAlgorithmCV()
    grid_search = GridSearchCV(estimator=ga, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    # Save grid search results
    results = pd.DataFrame(grid_search.cv_results_)
    csv_filename = f"genetic_algorithm_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Save best parameters
    best_params = grid_search.best_params_
    best_params_filename = f"best_genetic_algorithm_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([best_params]).to_csv(best_params_filename, index=False)
    print(f"Best parameters saved to {best_params_filename}")

    # Save epoch data
    epoch_data = []
    for params in grid_search.cv_results_['params']:
        fold_epoch_data = []
        for train_index, test_index in KFold(n_splits=3).split(X):
            ga = GeneticAlgorithmCV(**params)
            ga.fit([X[i] for i in train_index])
            fold_epoch_data.append(ga.cost_over_time_)

        # Calculate average epoch data across folds
        max_generations = max(len(epoch) for fold in fold_epoch_data for epoch in fold)
        avg_epoch_data = np.mean([np.pad(epoch, (0, max_generations - len(epoch)), 'edge')
                                  for fold in fold_epoch_data for epoch in fold], axis=0)
        epoch_data.append(avg_epoch_data.tolist())

    epoch_filename = f"genetic_algorithm_epoch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["population_size", "generations", "mutation_rate", "epoch_data"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for params, epochs in zip(grid_search.cv_results_['params'], epoch_data):
            writer.writerow({
                "population_size": params['population_size'],
                "generations": params['generations'],
                "mutation_rate": params['mutation_rate'],
                "epoch_data": epochs
            })

    print(f"Epoch data saved to {epoch_filename}")


if __name__ == "__main__":
    main()
