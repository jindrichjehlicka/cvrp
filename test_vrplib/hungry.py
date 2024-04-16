import time

import numpy as np
import vrplib


# Adjusted function to handle objects already in dict format and NumPy arrays
# ecursively converts objects, including NumPy arrays, dictionaries, and lists, into a format that can be easily serialized into JSON.
def obj_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Process each key-value pair in the dictionary
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj


# Assuming the read_instance and read_solution return dictionaries
# Read VRPLIB formatted instances and solutions
instance = vrplib.read_instance("./Vrp-Set-XML100/instances/XML100_1111_01.vrp")
solutions = vrplib.read_solution("./Vrp-Set-XML100/solutions/XML100_1111_01.sol")

# Convert instance and solutions, handling any nested NumPy arrays
instance_dict = obj_to_dict(instance)
solutions_dict = obj_to_dict(solutions)


def greedy_vrp(depot_loc, node_loc, demand, capacity):
    """Solves a CVRP instance using a Greedy Algorithm."""
    num_customers = len(node_loc)  # Total number of customers
    routes = []  # List to store routes
    visited = set()  # Set to track visited customers

    while len(visited) < num_customers:  # Continue until all customers are visited
        route = []  # Current route
        current_load = 0  # Load of the current vehicle
        current_location = depot_loc  # Start from the depot
        while True:
            nearest_distance = float('inf')  # Initialize with infinity
            nearest_customer = None  # To store the nearest customer's index
            for i, location in enumerate(node_loc):  # Iterate through customer locations
                if i not in visited and current_load + demand[i] <= capacity:  # Check if the customer can be visited
                    distance = euclidean_distance(current_location, location)  # Calculate distance
                    if distance < nearest_distance:  # Check if this is the nearest customer so far
                        nearest_distance = distance
                        nearest_customer = i
            if nearest_customer is None:
                break  # Break if no further customers can be added
            visited.add(nearest_customer)  # Mark the customer as visited
            route.append(nearest_customer)  # Add to the current route
            current_load += demand[nearest_customer]  # Update load
            current_location = node_loc[nearest_customer]  # Update current location

        routes.append(route)  # Add the current route to the list of routes
    return routes


# Correctly set the depot location based on the assumption it's the first in `node_coord`
depot_loc = instance_dict['node_coord'][0]  # Assuming the first coordinate is the depot

# Adjust node locations and demands for the first 10 nodes (excluding the depot if desired)
# Note: Adjust the slicing as needed based on whether you include the depot in the node list or not
node_loc = instance_dict['node_coord']  # Assuming we skip the depot and take the next 10
demand = instance_dict['demand']  # Corresponding demands, adjusting indices as needed

# The capacity remains unchanged
capacity = instance_dict['capacity']

print("hyhy", solutions_dict)


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


routes = greedy_vrp(depot_loc, node_loc, demand, capacity)
print("Routes:", routes)


def calculate_route_cost(route, node_loc, depot_loc):
    """Calculate the total cost (distance) of a single route."""
    cost = 0
    last_location = depot_loc
    for node in route:
        cost += euclidean_distance(last_location, node_loc[node])
        last_location = node_loc[node]
    cost += euclidean_distance(last_location, depot_loc)  # Return to depot
    return cost


def calculate_total_cost(routes, node_loc, depot_loc):
    """Calculate the total cost (distance) of all routes."""
    return sum(calculate_route_cost(route, node_loc, depot_loc) for route in routes)


# ... (load instance and solutions, define euclidean_distance, etc.)

start_time = time.time()  # Start the timer
# Generate routes using the greedy algorithm
routes = greedy_vrp(depot_loc, node_loc, demand, capacity)
end_time = time.time()  # End the timer
execution_time = end_time - start_time  # Calculate the elapsed time

print("Execution time (Greedy):", execution_time, "seconds")
# Calculate the total cost for the greedy solution
total_cost_greedy = calculate_total_cost(routes, node_loc, depot_loc)
print("Total cost (Greedy):", total_cost_greedy)

# Placeholder for optimal cost extraction from solutions_dict
# You will need to adjust this part to correctly access the optimal cost from solutions_dict
# Example: optimal_cost = solutions_dict['optimal_cost']
optimal_cost = 29888  # Replace with actual extraction from solutions_dict
print("Optimal cost:", optimal_cost)

# Compare the greedy solution cost with the optimal cost
difference = total_cost_greedy - optimal_cost
print("Difference between greedy and optimal cost:", difference)
