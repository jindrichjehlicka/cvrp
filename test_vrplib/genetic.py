import numpy as np
import random
from copy import deepcopy
import vrplib
import time


class Node:
    def __init__(self, idx, x, y, demand):
        self.idx = idx
        self.x = x
        self.y = y
        self.demand = demand

class Vehicle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.route = []

def euclidean_distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

class GeneticAlgorithm:
    def __init__(self, nodes, vehicle_capacity, n_vehicles, n_chromosomes, generations):
        self.nodes = nodes
        self.vehicle_capacity = vehicle_capacity
        self.n_vehicles = n_vehicles
        self.n_chromosomes = n_chromosomes
        self.generations = generations
        self.population = []
        self.best_solution = None
        self.best_cost = float('inf')
        self.initialize_population()

    def initialize_population(self):
        for _ in range(self.n_chromosomes):
            random_solution = random.sample(self.nodes[1:], len(self.nodes) - 1)  # Exclude depot
            self.population.append([self.nodes[0]] + random_solution + [self.nodes[0]])  # Include depot as start/end

    def evaluate_solution(self, solution):
        total_cost = 0
        vehicle_index = 0
        current_load = 0
        current_route = [solution[0]]
        for node in solution[1:]:
            if current_load + node.demand <= self.vehicle_capacity:
                current_route.append(node)
                current_load += node.demand
            else:
                current_route.append(solution[0])  # Return to depot
                total_cost += sum(euclidean_distance(current_route[i], current_route[i+1]) for i in range(len(current_route)-1))
                current_route = [solution[0], node]
                current_load = node.demand
        current_route.append(solution[0])
        total_cost += sum(euclidean_distance(current_route[i], current_route[i+1]) for i in range(len(current_route)-1))
        return total_cost

    def select_parents(self):
        tournament_size = 5
        parents = random.sample(self.population, tournament_size)
        parents.sort(key=self.evaluate_solution)
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        cut_point = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child = parent1[:cut_point] + [node for node in parent2 if node not in parent1[:cut_point]]
        return child

    def mutate(self, chromosome, mutation_rate=0.02):
        for i in range(1, len(chromosome) - 1):  # Exclude depot
            if random.random() < mutation_rate:
                j = random.randint(1, len(chromosome) - 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    def run(self):
        for generation in range(self.generations):
            new_population = []
            while len(new_population) < self.n_chromosomes:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                if child not in new_population:
                    new_population.append(child)
            self.population = new_population
            current_best = min(self.population, key=self.evaluate_solution)
            current_cost = self.evaluate_solution(current_best)
            if current_cost < self.best_cost:
                self.best_solution = current_best
                self.best_cost = current_cost
            print(f"Generation {generation}: Best Cost = {self.best_cost}")


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


# Reading and preparing the data
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)

# Parameters from the instance
depot_loc = instance_dict['node_coord'][0]  # Assuming the first coordinate is the depot
node_loc = instance_dict['node_coord']
demand = instance_dict['demand']
capacity = instance_dict['capacity']

# Assuming you have a Node class and GeneticAlgorithm class defined properly as earlier discussed
nodes = [Node(idx=i, x=loc[0], y=loc[1], demand=d) for i, (loc, d) in enumerate(zip(node_loc, demand), start=1)]
n_vehicles = len([d for d in demand if d <= capacity])  # Simplistic heuristic for vehicle count
n_chromosomes = 100  # Example starting population size
n_generations = 200  # Example number of generations

# Create and run the genetic algorithm
ga = GeneticAlgorithm(nodes=nodes, vehicle_capacity=capacity, n_vehicles=n_vehicles, n_chromosomes=n_chromosomes,
                      generations=n_generations)
ga.run()

#
# # Running the Genetic Algorithm
# start_time_ga = time.time()
# best_solution, best_cost =
# end_time_ga = time.time()

# # Printing results
# print("Execution time (Genetic Algorithm):", end_time_ga - start_time_ga, "seconds")
# print("Total cost (Genetic Algorithm):", best_cost)
#
# # Comparison with the optimal solution
# optimal_cost = 29888  # Replace with actual optimal cost if available
# print("Difference between GA and optimal cost:", best_cost - optimal_cost)
