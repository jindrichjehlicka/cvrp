import numpy as np
import random


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

    return best_solution, best_cost
