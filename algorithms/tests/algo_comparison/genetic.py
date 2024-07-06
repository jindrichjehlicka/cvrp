import numpy as np
import pandas as pd
import time
import vrplib
import random
from joblib import Parallel, delayed


def calculate_distance_matrix(node_loc):
    n_nodes = len(node_loc)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
    return distance_matrix


def generate_solution(node_loc, demand, capacity):
    nodes = list(range(1, len(node_loc)))
    random.shuffle(nodes)
    solution = []
    current_capacity = capacity
    route = [0]

    for node in nodes:
        if demand[node] <= current_capacity:
            route.append(node)
            current_capacity -= demand[node]
        else:
            route.append(0)
            solution.append(route)
            route = [0, node]
            current_capacity = capacity - demand[node]
    route.append(0)
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
    return 1 / calculate_cost(solution, distance_matrix)


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


def genetic_algorithm(population_size, generations, node_loc, demand, capacity):
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
            offspring = mutate(offspring)
            next_gen.append(offspring)
        pop = next_gen

        yield generation, best_cost


def run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, population_size,
                         generations, mutation_rate, runs=20):
    instance_results = []
    for run in range(runs):
        start_time = time.time()
        best_cost = float('inf')
        for generation, cost in genetic_algorithm(population_size, generations, node_loc, demand, capacity):
            if cost < best_cost:
                best_cost = cost
            elapsed_time = time.time() - start_time
            cost_difference = best_cost - optimal_cost
            instance_results.append(['Genetic', instance_name, run + 1, generation, cost_difference, elapsed_time,
                                     f'pop_size={population_size},gen={generations},mut_rate={mutation_rate}'])
    return instance_results


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()
    return instance_names


filename = "../instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instance_names = instance_names[602:1102]


def process_instance(instance_name, population_size, generations, mutation_rate):
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    return run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, population_size,
                                generations, mutation_rate)


population_size = 50
generations = 100
mutation_rate = 0.01

for i in range(0, len(instance_names), 100):
    chunk_instance_names = instance_names[i:i + 100]

    results_list = Parallel(n_jobs=-1)(
        delayed(process_instance)(instance_name, population_size, generations, mutation_rate) for instance_name in
        chunk_instance_names)

    flattened_results = [item for sublist in results_list for item in sublist]

    results = pd.DataFrame(flattened_results,
                           columns=['Algorithm', 'Instance', 'Run', 'Iteration', 'Cost Difference', 'Time',
                                    'Parameters'])

    chunk_number = (i // 100) + 1
    results.to_csv(f'genetic_algorithm_performance_chunk_2_{chunk_number}.csv', index=False)
