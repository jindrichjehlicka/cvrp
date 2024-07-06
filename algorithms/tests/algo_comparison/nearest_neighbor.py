import numpy as np
import pandas as pd
import time
import vrplib
import concurrent.futures

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_total_cost(routes, node_loc, depot_loc):
    total_cost = 0
    for route in routes:
        route_cost = 0
        last_location = depot_loc
        for node in route:
            route_cost += euclidean_distance(last_location, node_loc[node])
            last_location = node_loc[node]
        route_cost += euclidean_distance(last_location, depot_loc)
        total_cost += route_cost
    return total_cost

def nearest_neighbor_vrp(depot_loc, node_loc, demand, capacity):
    num_customers = len(node_loc)
    routes = []
    visited = set()

    while len(visited) < num_customers:
        route = []
        current_load = 0
        current_location = depot_loc
        while True:
            nearest_distance = float('inf')
            nearest_customer = None
            for i, location in enumerate(node_loc):
                if i not in visited and current_load + demand[i] <= capacity:
                    distance = euclidean_distance(current_location, location)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = i
            if nearest_customer is None:
                break
            visited.add(nearest_customer)
            route.append(nearest_customer)
            current_load += demand[nearest_customer]
            current_location = node_loc[nearest_customer]

        if route:
            routes.append(route)

    total_cost = calculate_total_cost(routes, node_loc, depot_loc)
    return routes, total_cost

def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = file.read().splitlines()
    return instance_names

results = pd.DataFrame(columns=['Algorithm', 'Instance', 'Run', 'Iteration', 'Cost Difference', 'Time', 'Parameters'])

def run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost, runs=20):
    instance_results = []
    for run in range(runs):
        start_time = time.time()
        routes, total_cost = nearest_neighbor_vrp(depot_loc, node_loc, demand, capacity)
        end_time = time.time()
        elapsed_time = end_time - start_time

        cost_difference = total_cost - optimal_cost

        instance_results.append(['NearestNeighbor', instance_name, run + 1, 'NA', cost_difference, elapsed_time, 'NA'])
    return instance_results

filename = "../instance_names.txt"
instance_names = load_instance_names_from_file(filename)
instance_names = instance_names[101:601]

def process_instance(instance_name):
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    return run_and_collect_data(instance_name, depot_loc, node_loc, demand, capacity, optimal_cost)

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_instance = {executor.submit(process_instance, instance_name): instance_name for instance_name in instance_names}
    for future in concurrent.futures.as_completed(future_to_instance):
        instance_name = future_to_instance[future]
        try:
            instance_results = future.result()
            for result in instance_results:
                results.loc[len(results)] = result
        except Exception as exc:
            print(f"{instance_name} generated an exception: {exc}")

results.to_csv('nearest_neighbor_performance.csv', index=False)
