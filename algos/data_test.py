import vrplib
import time

from algos.greedy import greedy_vrp
from algos.nearest_neighbor import nearest_neighbor_vrp
from algos.simulated_annealing import simulated_annealing, estimate_initial_temperature
from algos.tabu_search import tabu_search
from genetic import genetic_algorithm


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()  # This reads the file and splits it by lines into a list
    return instance_names


filename = "instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instances = instance_names[9000:9010]
for instance_name in instances:
    instance = vrplib.read_instance(
        "./Vrp-Set-XML100/instances/{instance_name}.vrp".format(instance_name=instance_name))
    solution = vrplib.read_solution(
        "./Vrp-Set-XML100/solutions/{instance_name}.sol".format(instance_name=instance_name))
    optimal_cost = solution['cost']
    depot_loc = instance['node_coord'][0]
    node_loc = instance['node_coord']
    demand = instance['demand']
    capacity = instance['capacity']

    # TEST ALGOS
    # ----------------------------------------------

    start_time = time.time()  # Start the timer

    # Genetic Algorithm
    # population_size = 150
    # generations = 100
    # routes, total_cost = genetic_algorithm(population_size, generations, node_loc, demand, capacity)
    #

    # Greedy
    # routes, total_cost = greedy_vrp(depot_loc, node_loc, demand, capacity)

    # Tabu Search
    # max_iterations = 300
    # tabu_size = 100
    # neighborhood_size = 100
    # routes, total_cost = tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand,
    #                                  capacity)

    # Nearest Neighbor
    # routes, total_cost = nearest_neighbor_vrp(depot_loc, node_loc, demand, capacity)

    # Simulated Annealing
    max_iterations = 15000
    initial_temperature = estimate_initial_temperature(node_loc, demand, capacity)
    cooling_rate = 0.96
    routes, total_cost = simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc, demand,
                                             capacity)

    difference = total_cost - optimal_cost
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    print(instance_name)
    print("Optimal cost:", optimal_cost)
    print("Total cost:", total_cost)
    print("Difference between cost and optimal cost:", difference)
    print("Execution time:", execution_time, "seconds")
    print("----------------------------------------------")
