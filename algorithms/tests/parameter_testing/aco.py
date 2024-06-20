import numpy as np
import vrplib
import itertools
import time
import csv
from joblib import Parallel, delayed
from datetime import datetime


def euc_2d(c1, c2, epsilon=1e-10):
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1]) + epsilon


def calculate_cost(routes, depot_loc, node_loc):
    total_cost = 0
    for route in routes:
        if route:
            route_cost = euc_2d(depot_loc, node_loc[route[0]])
            for i in range(1, len(route)):
                route_cost += euc_2d(node_loc[route[i - 1]], node_loc[route[i]])
            route_cost += euc_2d(node_loc[route[-1]], depot_loc)
            total_cost += route_cost
    return total_cost


def initialize_pheromone(n, initial_pheromone):
    return np.full((n, n), initial_pheromone)


def aco_algorithm(depot_loc, node_loc, demand, capacity, num_ants=5, iterations=50, decay=0.05, alpha=1, beta=2):
    num_nodes = len(node_loc)
    pheromone = initialize_pheromone(num_nodes, 1.0)
    all_nodes = list(range(num_nodes))
    best_cost = float('inf')
    best_solution = []

    for iteration in range(int(iterations)):
        solutions = []
        for ant in range(int(num_ants)):
            solution = []
            remaining_nodes = set(all_nodes)
            while remaining_nodes:
                route = []
                current_node = None
                load = 0
                while remaining_nodes:
                    if current_node is None:
                        current_node = np.random.choice(list(remaining_nodes))
                    next_node = max(remaining_nodes, key=lambda x: (pheromone[current_node][x] ** alpha) *
                                                                   ((1.0 / euc_2d(node_loc[current_node],
                                                                                  node_loc[x])) ** beta)
                    if (load + demand[x] <= capacity) else 0)
                    if load + demand[next_node] > capacity:
                        break
                    route.append(next_node)
                    remaining_nodes.remove(next_node)
                    load += demand[next_node]
                    current_node = next_node
                solution.append(route)
            solutions.append(solution)
            cost = calculate_cost(solution, depot_loc, node_loc)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        # Update pheromone
        for i, j in np.ndindex(pheromone.shape):
            pheromone[i][j] *= (1 - decay)
        for solution in solutions:
            route_cost = calculate_cost(solution, depot_loc, node_loc)
            for route in solution:
                for i in range(len(route) - 1):
                    pheromone[route[i]][route[i + 1]] += 1.0 / route_cost

    return best_solution, best_cost


# Function to load instance names from a file
def load_instance_names_from_file(filename):
    with open(filename, 'r') as file:
        instance_names = [line.strip() for line in file.readlines()]
    return instance_names


# Define the parameter grid
param_grid = {
    'num_ants': [5, 10, 15],
    'iterations': [20, 50],
    'decay': [0.01, 0.05, 0.1],
    'alpha': [1, 2, 3],
    'beta': [2, 5, 8]
}


# Function to evaluate a single parameter combination on a single dataset
def evaluate(instance_name, params, n_runs=10):
    num_ants, iterations, decay, alpha, beta = params
    instance = vrplib.read_instance(f"../../Vrp-Set-XML100/instances/{instance_name}.vrp")
    solution = vrplib.read_solution(f"../../Vrp-Set-XML100/solutions/{instance_name}.sol")
    optimal_cost = solution['cost']
    node_loc = instance['node_coord']
    depot_loc = node_loc[0]
    demand = instance['demand']
    capacity = instance['capacity']

    costs = []
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _, cost = aco_algorithm(depot_loc, node_loc, demand, capacity, num_ants=int(num_ants),
                                iterations=int(iterations), decay=decay, alpha=alpha, beta=beta)
        end_time = time.time()
        costs.append(cost)
        times.append(end_time - start_time)

    avg_cost = np.mean(costs)
    avg_time = np.mean(times)

    return {
        "instance_name": instance_name,
        "num_ants": num_ants,
        "iterations": iterations,
        "decay": decay,
        "alpha": alpha,
        "beta": beta,
        "optimal_cost": optimal_cost,
        "final_cost": avg_cost,
        "time": avg_time,
        "runs": n_runs
    }


# Main function to perform the grid search and save results to CSV
def main():
    filename = "../instance_names.txt"
    instance_names = load_instance_names_from_file(filename)
    instances = instance_names[:100]

    # Create the list of all parameter combinations
    param_combinations = list(itertools.product(param_grid['num_ants'], param_grid['iterations'],
                                                param_grid['decay'], param_grid['alpha'], param_grid['beta']))

    # Evaluate all parameter combinations on all instances in parallel
    results = Parallel(n_jobs=10)(delayed(evaluate)(instance_name, params)
                                  for instance_name in instances
                                  for params in param_combinations)

    # Save results to CSV
    csv_filename = f"aco_grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ["instance_name", "num_ants", "iterations", "decay", "alpha", "beta",
                      "optimal_cost", "final_cost", "time", "runs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
