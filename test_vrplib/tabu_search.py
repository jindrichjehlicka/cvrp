import time
import numpy as np
import vrplib


# Helper functions for data handling
def obj_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj


# Euclidean distance function
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Load and convert VRP data
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)

# Initial parameters
depot_loc = instance_dict['node_coord'][0]
node_loc = instance_dict['node_coord']
demand = instance_dict['demand']
capacity = instance_dict['capacity']


# Tabu Search VRP Function
def tabu_search_vrp(depot_loc, node_loc, demand, capacity, num_customers, max_iterations=100, tabu_tenure=10):
    routes = []
    tabu_list = set()
    current_solution = [0]  # Start route at the depot

    # Initial solution: simple greedy approach to start
    for i in range(1, num_customers):
        if demand[i] <= capacity:  # Simplistic capacity check
            current_solution.append(i)

    # Begin Tabu Search iterations
    for iteration in range(max_iterations):
        best_local_move = None
        best_local_cost = float('inf')
        # Explore neighbors of the current solution
        for i in range(1, len(current_solution) - 1):
            for j in range(i + 1, len(current_solution)):
                new_solution = current_solution[:]
                # 2-opt Swap
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_cost = calculate_total_cost([new_solution], node_loc, depot_loc)

                if (new_cost < best_local_cost and (i, j) not in tabu_list):
                    best_local_cost = new_cost
                    best_local_move = (i, j)
                    best_solution = new_solution

        # Update current solution
        if best_local_move:
            current_solution = best_solution
            tabu_list.add(best_local_move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop()

        routes = [current_solution]  # Simplified to single route management

    return routes


# Route cost calculation functions
def calculate_route_cost(route, node_loc, depot_loc):
    cost = 0
    last_location = depot_loc
    for node in route:
        cost += euclidean_distance(last_location, node_loc[node])
        last_location = node_loc[node]
    cost += euclidean_distance(last_location, depot_loc)  # Return to depot
    return cost


def calculate_total_cost(routes, node_loc, depot_loc):
    return sum(calculate_route_cost(route, node_loc, depot_loc) for route in routes)


# Run the Tabu Search and calculate performance
start_time = time.time()
num_customers = len(node_loc)
routes = tabu_search_vrp(depot_loc, node_loc, demand, capacity, num_customers)
end_time = time.time()
execution_time = end_time - start_time

total_cost = calculate_total_cost(routes, node_loc, depot_loc)
optimal_cost = 29888  # Replace with actual extraction

# Display results
print("Execution time (Tabu Search):", execution_time, "seconds")
print("Total cost (Tabu Search):", total_cost)
print("Optimal cost:", optimal_cost)
print("Difference between Tabu Search and optimal cost:", total_cost - optimal_cost)
