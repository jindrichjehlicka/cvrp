import vrplib
import time
import csv
import numpy as np
from algos.aco_ts import aco_ts_hybrid


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
instances = instance_names[:100]

# Define the parameter grid
param_grid = [
    {"num_ants": 5, "iterations": 10, "decay": 0.05, "alpha": 1, "beta": 2, "max_iterations": 100, "tabu_size": 50,
     "neighborhood_size": 50},
    {"num_ants": 10, "iterations": 20, "decay": 0.05, "alpha": 1, "beta": 2, "max_iterations": 100, "tabu_size": 50,
     "neighborhood_size": 50},
    {"num_ants": 5, "iterations": 10, "decay": 0.1, "alpha": 2, "beta": 5, "max_iterations": 100, "tabu_size": 50,
     "neighborhood_size": 50},
    {"num_ants": 10, "iterations": 20, "decay": 0.1, "alpha": 2, "beta": 5, "max_iterations": 100, "tabu_size": 50,
     "neighborhood_size": 50},
    {"num_ants": 5, "iterations": 20, "decay": 0.05, "alpha": 1, "beta": 5, "max_iterations": 100, "tabu_size": 50,
     "neighborhood_size": 50}
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

    for params in param_grid:
        for _ in range(20):  # Run each dataset 20 times
            start_time = time.time()
            routes, total_cost = aco_ts_hybrid(depot_loc, node_loc, demand, capacity, params["num_ants"],
                                               params["iterations"], params["decay"], params["alpha"],
                                               params["beta"], params["max_iterations"], params["tabu_size"],
                                               params["neighborhood_size"])
            end_time = time.time()

            execution_time = end_time - start_time
            difference = total_cost - optimal_cost

            result = {
                "Instance": instance_name,
                "Algorithm": "ACO-TS Hybrid",
                "Parameters": params,
                "Optimal Cost": optimal_cost,
                "Total Cost": total_cost,
                "Difference": difference,
                "Execution Time (s)": execution_time
            }
            results.append(result)

save_results_to_csv("aco_ts_hybrid_results.csv", results)
