import vrplib
import time
import csv
import numpy as np
from algorithms.tabu_search import tabu_search


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
instances = instance_names[:500]

# Parameters to test for Tabu Search
max_iterations_list = [100, 200, 300]
tabu_size_list = [50, 75, 100]
neighborhood_size_list = [50, 75, 100]

results = []

for instance_name in instances:
    instance = vrplib.read_instance(f"./Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"./Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    for max_iterations in max_iterations_list:
        for tabu_size in tabu_size_list:
            for neighborhood_size in neighborhood_size_list:
                start_time = time.time()

                routes, total_cost = tabu_search(max_iterations, tabu_size, neighborhood_size, node_loc, demand,
                                                 capacity)

                end_time = time.time()

                execution_time = end_time - start_time
                difference = total_cost - optimal_cost

                params = {
                    "max_iterations": max_iterations,
                    "tabu_size": tabu_size,
                    "neighborhood_size": neighborhood_size
                }

                result = {
                    "Instance": instance_name,
                    "Algorithm": "Tabu Search",
                    "Parameters": params,
                    "Optimal Cost": optimal_cost,
                    "Total Cost": total_cost,
                    "Difference": difference,
                    "Execution Time (s)": execution_time
                }
                results.append(result)
                print(result)

save_results_to_csv("tabu_search_algorithm_comparison_results.csv", results)
