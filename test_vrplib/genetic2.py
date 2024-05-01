import random
import math
import sys
import vrplib
import numpy as np


class Node:
    def __init__(self, idx, demand, posX, posY):
        self.idx = idx
        self.demand = demand
        self.posX = posX
        self.posY = posY


def read_vrp_data(instance):
    nodes = []
    # Assuming that instance['node_coord'] and instance['demand'] are lists and correctly aligned
    for idx, ((x, y), d) in enumerate(zip(instance['node_coord'], instance['demand']), start=1):
        if idx == 1:  # Assuming the first node is the depot with a demand of 0
            nodes.append(Node(idx=0, demand=0, posX=x, posY=y))  # Depot initialization
        else:
            nodes.append(Node(idx=idx-1, demand=d, posX=x, posY=y))
    vrp = {
        'capacity': instance['capacity'],
        'nodes': nodes
    }
    return vrp


def distance(n1, n2):
    return math.sqrt((n1.posX - n2.posX) ** 2 + (n1.posY - n2.posY) ** 2)


def fitness(route, nodes):
    cost = distance(nodes[0], nodes[route[0]])  # From depot to first node
    for i in range(1, len(route)):
        cost += distance(nodes[route[i - 1]], nodes[route[i]])
    cost += distance(nodes[route[-1]], nodes[0])  # From last node back to depot
    return cost


def adjust_route(route, nodes, capacity):
    new_route = []
    current_capacity = 0
    for node_idx in route:
        if nodes[node_idx].demand + current_capacity > capacity:
            new_route.append(0)  # Insert depot
            current_capacity = 0
        new_route.append(node_idx)
        current_capacity += nodes[node_idx].demand
    if new_route[-1] != 0:
        new_route.append(0)  # End at depot
    return new_route


def generate_initial_population(size, nodes):
    pop = []
    for _ in range(size):
        route = list(range(1, len(nodes)))  # all nodes except depot
        random.shuffle(route)
        route = adjust_route(route, nodes, nodes[0].demand)  # depot's demand is capacity
        pop.append(route)
    return pop


def crossover(parent1, parent2, nodes):
    cut = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:cut] + [n for n in parent2 if n not in parent1[:cut]]
    return adjust_route(child, nodes, nodes[0].demand)


def mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]


def run_genetic_algorithm(instance_dict, pop_size, generations):
    vrp = read_vrp_data(instance_dict)
    population = generate_initial_population(pop_size, vrp['nodes'])
    best_route = None
    best_fitness = float('inf')

    for _ in range(generations):
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = crossover(parent1, parent2, vrp['nodes'])
            child2 = crossover(parent2, parent1, vrp['nodes'])
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        for route in population:
            route_cost = fitness(route, vrp['nodes'])
            if route_cost < best_fitness:
                best_fitness = route_cost
                best_route = route

    print("Best cost:", best_fitness)

    print("Best route:", [vrp['nodes'][idx].idx for idx in best_route if idx != 0])



# Helper function to handle various data formats
def obj_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj


# Example usage:
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)

run_genetic_algorithm(instance_dict, pop_size=50, generations=100)
