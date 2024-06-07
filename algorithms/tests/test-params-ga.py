import vrplib
import time
import csv
import numpy as np

from algorithms.genetic import genetic_algorithm


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

# Parameters to test for Genetic Algorithm
population_sizes = [50, 100, 150]
generations_list = [50, 100, 150]

algorithms = [
    {"name": "Genetic Algorithm", "func": genetic_algorithm,
     "params_list": [{"population_size": ps, "generations": gen} for ps in population_sizes for gen in generations_list]}
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
        if "params_list" in algo:
            for params in algo["params_list"]:
                for _ in range(20):  # Run each dataset 20 times
                    start_time = time.time()
                    routes, total_cost = algo["func"](params["population_size"], params["generations"], node_loc, demand, capacity)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    difference = total_cost - optimal_cost

                    result = {
                        "Instance": instance_name,
                        "Algorithm": algo["name"],
                        "Parameters": params,
                        "Optimal Cost": optimal_cost,
                        "Total Cost": total_cost,
                        "Difference": difference,
                        "Execution Time (s)": execution_time
                    }
                    results.append(result)
                    # print(result)

save_results_to_csv("ga_params_results.csv", results)
