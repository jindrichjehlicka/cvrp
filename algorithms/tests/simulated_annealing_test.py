import vrplib
import time
import csv
import numpy as np

from algorithms.simulated_annealing import estimate_initial_temperature, simulated_annealing


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

# Parameters to test for Simulated Annealing
max_iterations_list = [5000, 10000, 15000]
cooling_rate_list = [0.9, 0.95, 0.99]

results = []

for instance_name in instances:
    instance = vrplib.read_instance(f"../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    for max_iterations in max_iterations_list:
        for cooling_rate in cooling_rate_list:
            for _ in range(20):  # Run each parameter set 20 times
                start_time = time.time()

                initial_temperature = estimate_initial_temperature(node_loc, demand, capacity)
                routes, total_cost = simulated_annealing(max_iterations, initial_temperature, cooling_rate, node_loc,
                                                         demand, capacity)

                end_time = time.time()

                execution_time = end_time - start_time
                difference = total_cost - optimal_cost

                params = {
                    "max_iterations": max_iterations,
                    "cooling_rate": cooling_rate
                }

                result = {
                    "Instance": instance_name,
                    "Algorithm": "Simulated Annealing",
                    "Parameters": params,
                    "Optimal Cost": optimal_cost,
                    "Total Cost": total_cost,
                    "Difference": difference,
                    "Execution Time (s)": execution_time
                }
                results.append(result)
                print(result)

save_results_to_csv("simulated_annealing_algorithm_comparison_results.csv", results)
