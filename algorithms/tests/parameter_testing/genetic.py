import numpy as np
import random
import vrplib
import itertools
import time
import csv
from joblib import Parallel, delayed
from datetime import datetime


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

    for generation in range(generations):
        fitnesses = [fitness(sol, distance_matrix) for sol in pop]
        current_best_index = np.argmax(fitnesses)
        current_best_solution = pop[current_best_index]
        current_best_solution_cost = calculate_cost(current_best_solution, distance_matrix)

        if current_best_solution_cost < best_cost:
            best_cost = current_best_solution_cost
            best_solution = current_best_solution

        selected = select(pop, fitnesses, k=int(population_size / 2))
        next_gen = []
        while len(next_gen) < population_size:
            p1, p2 = random.sample(selected, 2)
            offspring = crossover(p1, p2)
            offspring = mutate(offspring, mutation_rate)
            next_gen.append(offspring)
        pop = next_gen

    return best_solution, best_cost


# Function to load instance names from a file
def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names


# Define the parameter grid for Genetic Algorithm
param_grid = {
    'population_size': [50, 100],
    'generations': [100, 200],
    'mutation_rate': [0.01, 0.05, 0.1]
}


# Function to evaluate a single parameter combination on a single dataset
def evaluate(instance_name, params, n_runs=20):
    population_size, generations, mutation_rate = params
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    costs = []
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _, cost = genetic_algorithm(population_size, generations, node_loc, demand, capacity, mutation_rate)
        end_time = time.time()
        costs.append(cost)
        times.append(end_time - start_time)

    avg_cost = np.mean(costs)
    avg_time = np.mean(times)

    return {
        "instance_name": instance_name,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "optimal_cost": optimal_cost,
        "final_cost": avg_cost,
        "time": avg_time,
        "runs": n_runs
    }


# Main function to perform the grid search and save results to CSV
def main():
    filename = "../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)
    instances = instance_names[:50]  # You can change the number of instances here (50, 100, 200)

    # Create the list of all parameter combinations
    param_combinations = list(
        itertools.product(param_grid['population_size'], param_grid['generations'], param_grid['mutation_rate']))

    # Evaluate all parameter combinations on all instances in parallel
    results = Parallel(n_jobs=-1)(delayed(evaluate)(instance_name, params)
                                  for instance_name in instances
                                  for params in param_combinations)

    # Save results to CSV
    csv_filename = f"genetic_algorithm_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ["instance_name", "population_size", "generations", "mutation_rate",
                      "optimal_cost", "final_cost", "time", "runs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
