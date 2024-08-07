import random
import vrplib
import csv
from datetime import datetime
import pandas as pd
import numpy as np
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


def generate_initial_solution(node_loc, demand, capacity):
    nodes = list(range(1, len(node_loc)))
    random.shuffle(nodes)
    routes = []
    current_route = [0]
    current_load = 0

    for node in nodes:
        if current_load + demand[node] <= capacity:
            current_route.append(node)
            current_load += demand[node]
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, node]
            current_load = demand[node]
    current_route.append(0)
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


def fitness(solution, distance_matrix):
    return 1 / calculate_cost(solution, distance_matrix)


def select(population, fitnesses, k=3):
    selected = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [x for x, _ in selected[:k]]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(solution, mutation_rate=0.01):
    for route in solution:
        if len(route) > 3 and random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
    return solution


def get_neighbor(solution):
    neighbor = [route[:] for route in solution]
    if len(neighbor) > 1:
        route1, route2 = random.sample(neighbor, 2)
        if len(route1) > 3 and len(route2) > 3:
            idx1, idx2 = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return neighbor


def simulated_annealing(solution, distance_matrix, max_iterations=500, initial_temperature=100, cooling_rate=0.95):
    current_solution = solution
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
        cost_over_time.append(current_cost)

    return best_solution, best_cost, cost_over_time


def ga_sa_hybrid(population_size, generations, node_loc, demand, capacity, max_iterations=500, initial_temperature=100, cooling_rate=0.95):
    distance_matrix = calculate_distance_matrix(node_loc)
    pop = initial_population(population_size, node_loc, demand, capacity)
    best_solution = min(pop, key=lambda sol: calculate_cost(sol, distance_matrix))
    best_cost = calculate_cost(best_solution, distance_matrix)
    cost_over_time = []

    for generation in range(generations):
        fitnesses = [fitness(sol, distance_matrix) for sol in pop]
        selected = select(pop, fitnesses, k=int(population_size / 2))
        next_gen = []

        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            next_gen.extend([mutate(child1), mutate(child2)])

        pop = next_gen[:population_size]
        current_best_solution = min(pop, key=lambda sol: calculate_cost(sol, distance_matrix))
        current_best_cost = calculate_cost(current_best_solution, distance_matrix)

        if current_best_cost < best_cost:
            best_solution = current_best_solution
            best_cost = current_best_cost

        sa_solution, sa_cost, sa_cost_over_time = simulated_annealing(current_best_solution, distance_matrix,
                                                                      max_iterations, initial_temperature, cooling_rate)
        if sa_cost < best_cost:
            best_solution = sa_solution
            best_cost = sa_cost

        cost_over_time.append([best_cost - cost for cost in sa_cost_over_time])

    return best_solution, best_cost, cost_over_time


class GASAHybridCV(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=50, generations=50, max_iterations=500, initial_temperature=100, cooling_rate=0.95, runs=1):
        self.population_size = population_size
        self.generations = generations
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.runs = runs
        self.cost_over_runs_ = []

    def fit(self, X, y=None):
        self.cost_over_runs_ = []

        for instance in X:
            node_loc, demand, capacity, optimal_cost, instance_name = instance
            _, cost, cost_over_runs = ga_sa_hybrid(
                self.population_size, self.generations, node_loc, demand, capacity, self.max_iterations,
                self.initial_temperature, self.cooling_rate)
            diff_over_runs = [[optimal_cost - run_cost for run_cost in run] for run in cost_over_runs]
            self.cost_over_runs_.append(diff_over_runs)

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
    'initial_temperature': [50, 100, 150],
    'cooling_rate': [0.95, 0.99, 0.995],
}


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
        data.append((node_loc, demand, capacity, optimal_cost, instance_name))

    X = data
    y = [d[3] for d in data]

    epoch_data = []
    for population_size, generations, max_iterations, initial_temperature, cooling_rate in product(
            param_grid['population_size'], param_grid['generations'], param_grid['max_iterations'],
            param_grid['initial_temperature'], param_grid['cooling_rate']):
        for train_index, test_index in KFold(n_splits=3).split(X):
            ga_sa = GASAHybridCV(
                population_size=population_size,
                generations=generations,
                max_iterations=max_iterations,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate
            )
            ga_sa.fit([X[i] for i in train_index])
            for run_data in ga_sa.cost_over_runs_:
                for run in run_data:
                    epoch_data.append({
                        "population_size": population_size,
                        "generations": generations,
                        "max_iterations": max_iterations,
                        "initial_temperature": initial_temperature,
                        "cooling_rate": cooling_rate,
                        "epoch_data": str(run),
                        "instance_name": X[train_index[0]][4]
                    })

    epoch_filename = f"ga_sa_hybrid_epoch_data_chunk_2_{chunk_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(epoch_filename, 'w', newline='') as csvfile:
        fieldnames = ["population_size", "generations", "max_iterations", "initial_temperature", "cooling_rate",
                      "epoch_data", "instance_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in epoch_data:
            writer.writerow(entry)

    print(f"Epoch data for chunk {chunk_number} saved to {epoch_filename}")
    return epoch_filename


def main():
    filename = "../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)

    results = Parallel(n_jobs=-1)(
        delayed(process_and_save_epoch_data)(instance_names[i:i + 10], chunk_number=(i // 10) + 1)
        for i in range(0, 200, 20)
    )


    print("Processing complete. Results:", results)


if __name__ == "__main__":
    main()
