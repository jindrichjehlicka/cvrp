import vrplib
import time
import csv
import numpy as np
from algorithms.ant_colony_optimization import aco_algorithm


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
instances = instance_names[:200]

# Parameters to test for ACO
num_ants_list = [5, 10, 15]
iterations_list = [10, 20, 30]
decay_list = [0.01, 0.05, 0.1]

results = []

for instance_name in instances:
    instance = vrplib.read_instance(f"./Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"./Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    for num_ants in num_ants_list:
        for iterations in iterations_list:
            for decay in decay_list:
                for _ in range(20):  # Run each dataset 20 times
                    start_time = time.time()

                    routes, total_cost = aco_algorithm(depot_loc, node_loc, demand, capacity, num_ants=num_ants,
                                                       iterations=iterations, decay=decay, alpha=1, beta=2)

                    end_time = time.time()

                    execution_time = end_time - start_time
                    difference = total_cost - optimal_cost

                    params = {
                        "num_ants": num_ants,
                        "iterations": iterations,
                        "decay": decay
                    }

                    result = {
                        "Instance": instance_name,
                        "Algorithm": "ACO",
                        "Parameters": params,
                        "Optimal Cost": optimal_cost,
                        "Total Cost": total_cost,
                        "Difference": difference,
                        "Execution Time (s)": execution_time
                    }
                    results.append(result)

save_results_to_csv("aco_algorithm_comparison_results.csv", results)
