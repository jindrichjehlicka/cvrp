import random
import math
import vrplib


def distance(n1, n2):
    """Calculate Euclidean distance between two nodes."""
    dx = n2['posX'] - n1['posX']
    dy = n2['posY'] - n1['posY']
    return math.sqrt(dx ** 2 + dy ** 2)


def fitness(route, vrp):
    """Calculates the total distance of a route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(vrp['nodes'][route[i]], vrp['nodes'][route[i + 1]])
    return total_distance

def adjust(route, vrp):
    """Adjusts routes to ensure capacity constraints are met."""
    new_route = [0]  # Start at depot
    current_load = 0
    for node_index in route[1:]:  # skip initial depot
        node_demand = vrp['nodes'][node_index]['demand']
        if current_load + node_demand > vrp['capacity']:
            new_route.append(0)  # Return to depot
            current_load = 0  # Reset load
        new_route.append(node_index)
        current_load += node_demand
    if new_route[-1] != 0:
        new_route.append(0)  # Ensure ends at depot
    return new_route

def initialize_population(vrp, popsize):
    """Generates a population of random routes."""
    population = []
    node_indices = list(range(1, len(vrp['nodes'])))  # Exclude depot (assumed index 0)
    for _ in range(popsize):
        random_route = random.sample(node_indices, len(node_indices))
        population.append([0] + random_route + [0])  # Start and end at the depot
    return population

def genetic_algorithm(vrp, popsize, iterations):
    # Initialize population with feasible routes
    pop = [[i for i in range(1, len(vrp['nodes']))] for _ in range(popsize)]
    for i in range(len(pop)):
        random.shuffle(pop[i])
        pop[i] = adjust(pop[i], vrp)

    for _ in range(iterations):
        next_pop = []
        while len(next_pop) < popsize:
            parents = random.sample(pop, 4)
            parent1 = min(parents[:2], key=lambda p: fitness(p, vrp))
            parent2 = min(parents[2:], key=lambda p: fitness(p, vrp))
            idx1, idx2 = sorted(random.sample(range(1, len(parent1)), 2))
            child1 = parent1[:idx1] + parent2[idx1:idx2] + parent1[idx2:]
            child2 = parent2[:idx1] + parent1[idx1:idx2] + parent2[idx2:]
            child1, child2 = adjust(child1, vrp), adjust(child2, vrp)
            next_pop.extend([child1, child2])
        if random.randint(1, 15) == 1:
            to_mutate = random.choice(next_pop)
            i1, i2 = random.sample(range(len(to_mutate)), 2)
            to_mutate[i1], to_mutate[i2] = to_mutate[i2], to_mutate[i1]
            next_pop.append(adjust(to_mutate, vrp))
        pop = next_pop

    best_route = min(pop, key=lambda p: fitness(p, vrp))
    return best_route, fitness(best_route, vrp)


# VRP instance example:
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
# instance_dict = obj_to_dict(instance)
# Parameters from the instance
# depot_loc = instance_dict['node_coord'][0]  # Assuming the first coordinate is the depot
# node_loc = instance_dict['node_coord']
# demand = instance_dict['demand']
# capacity = instance_dict['capacity']


vrp = {
    'nodes': [{'label': f'Node {i}', 'demand': demand, 'posX': coord[0], 'posY': coord[1]}
              for i, (demand, coord) in enumerate(zip(instance['demand'], instance['node_coord']))],
    'capacity': instance['capacity']
}

popsize = 50
iterations = 100
best_route, best_cost = genetic_algorithm(vrp, popsize, iterations)
print("Best route:", best_route)
print("Optimal cost:", best_cost)
