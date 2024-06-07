import vrplib
import time
import csv
import numpy as np

from algorithms.ant_colony_optimization import aco_algorithm
from algorithms.greedy import dijkstra_cvrp
from algorithms.nearest_neighbor import nearest_neighbor_vrp
from algorithms.simulated_annealing import simulated_annealing, estimate_initial_temperature
from algorithms.tabu_search import tabu_search
from genetic import genetic_algorithm

# Hybrid Algorithms
from algorithms.aco_ts import aco_ts_hybrid
from algorithms.ga_sa import ga_sa_hybrid


def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()
    return instance_names


def save_results_to_csv(filename, results):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


filename = "instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instances = instance_names[:1]

algorithms = [
    {"name": "Genetic Algorithm", "func": genetic_algorithm, "params": {"population_size": 150, "generations": 100}},
    {"name": "Greedy", "func": dijkstra_cvrp, "params": {}},
    {"name": "Tabu Search", "func": tabu_search,
     "params": {"max_iterations": 300, "tabu_size": 100, "neighborhood_size": 100}},
    {"name": "Nearest Neighbor", "func": nearest_neighbor_vrp, "params": {}},
    {"name": "ACO", "func": aco_algorithm,
     "params": {"num_ants": 5, "iterations": 10, "decay": 0.05, "alpha": 1, "beta": 2}},
    {"name": "Simulated Annealing", "func": simulated_annealing,
     "params": {"max_iterations": 15000, "cooling_rate": 0.96}},
    {"name": "GA-SA Hybrid", "func": ga_sa_hybrid,
     "params": {"population_size": 50, "generations": 100, "max_iterations": 1000, "cooling_rate": 0.99}},
    {"name": "ACO-TS Hybrid", "func": aco_ts_hybrid,
     "params": {"num_ants": 3, "iterations": 20, "decay": 0.1, "alpha": 1, "beta": 2,
                "max_iterations": 50, "tabu_size": 25, "neighborhood_size": 10}}

]

results = []

for instance_name in instances:
    instance = vrplib.read_instance(f"./Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"./Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    for algo in algorithms:
        start_time = time.time()

        if algo["name"] == "Genetic Algorithm":
            routes, total_cost = algo["func"](algo["params"]["population_size"], algo["params"]["generations"],
                                              node_loc, demand, capacity)
        elif algo["name"] == "Simulated Annealing":
            initial_temperature = estimate_initial_temperature(node_loc, demand, capacity)
            routes, total_cost = algo["func"](algo["params"]["max_iterations"], initial_temperature,
                                              algo["params"]["cooling_rate"], node_loc, demand, capacity)
        elif algo["name"] == "Tabu Search":
            routes, total_cost = algo["func"](algo["params"]["max_iterations"], algo["params"]["tabu_size"],
                                              algo["params"]["neighborhood_size"], node_loc, demand, capacity)
        elif algo["name"] == "GA-SA Hybrid":
            routes, total_cost = algo["func"](algo["params"]["population_size"], algo["params"]["generations"],
                                              node_loc, demand, capacity, algo["params"]["max_iterations"],
                                              algo["params"]["cooling_rate"])
        elif algo["name"] == "ACO-TS Hybrid":
            routes, total_cost = algo["func"](depot_loc, node_loc, demand, capacity, algo["params"]["num_ants"],
                                              algo["params"]["iterations"], algo["params"]["decay"],
                                              algo["params"]["alpha"],
                                              algo["params"]["beta"], algo["params"]["max_iterations"],
                                              algo["params"]["tabu_size"],
                                              algo["params"]["neighborhood_size"])
        else:
            routes, total_cost = algo["func"](depot_loc, node_loc, demand, capacity, **algo["params"])

        end_time = time.time()

        execution_time = end_time - start_time
        difference = total_cost - optimal_cost

        result = {
            "Instance": instance_name,
            "Algorithm": algo["name"],
            "Optimal Cost": optimal_cost,
            "Total Cost": total_cost,
            "Difference": difference,
            "Execution Time (s)": execution_time
        }
        results.append(result)
        print(result)

save_results_to_csv("algorithm_comparison_results.csv", results)
