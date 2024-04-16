import numpy as np
import random


# Helper functions
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_total_distance(routes, distance_matrix):
    total_distance = 0
    for route in routes:
        for i in range(1, len(route)):
            total_distance += distance_matrix[route[i - 1], route[i]]
        # Return to depot
        total_distance += distance_matrix[route[-1], route[0]]
    return total_distance


# Initial solution generator (simple nearest neighbor)
def generate_initial_solution(depot, customers, vehicle_capacity, demand, distance_matrix):
    routes = []
    unvisited_customers = set(range(len(customers)))
    while unvisited_customers:
        route = [depot]
        current_load = 0
        while unvisited_customers and current_load < vehicle_capacity:
            last_customer = route[-1]
            nearest_customer = min(unvisited_customers, key=lambda x: distance_matrix[last_customer, x] if (
                    current_load + demand[x] <= vehicle_capacity) else float('inf'))
            if current_load + demand[nearest_customer] > vehicle_capacity:
                break
            route.append(nearest_customer)
            current_load += demand[nearest_customer]
            unvisited_customers.remove(nearest_customer)
        routes.append(route)
    return routes



def swap_2opt(route, i, k):
    """Performs a 2-opt swap by inverting the route segment between two indices."""
    assert i < k
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route


def get_neighbors(solution, distance_matrix):
    """Generates neighboring solutions by applying 2-opt swaps."""
    neighbors = []
    for i in range(1, len(solution) - 2):
        for k in range(i + 1, len(solution) - 1):
            for route in solution:
                new_route = swap_2opt(route, i, k)
                new_solution = solution[:]
                new_solution[solution.index(route)] = new_route
                neighbors.append(new_solution)
    return neighbors


def tabu_search(initial_solution, distance_matrix, num_iterations, tabu_tenure):
    current_solution = initial_solution
    best_solution = initial_solution
    best_cost = calculate_total_distance(initial_solution, distance_matrix)
    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = get_neighbors(current_solution, distance_matrix)
        neighbors_costs = [calculate_total_distance(neighbor, distance_matrix) for neighbor in neighbors]
        best_neighbor_idx = np.argmin(neighbors_costs)
        best_neighbor = neighbors[best_neighbor_idx]

        if best_neighbor not in tabu_list:
            current_solution = best_neighbor
            current_cost = neighbors_costs[best_neighbor_idx]
            if current_cost < best_cost:
                best_solution = best_neighbor
                best_cost = current_cost
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        # Dynamic tabu tenure adjustment (simplified example)
        if iteration % 10 == 0:
            tabu_tenure = random.randint(5, 15)  # Adjust tenure randomly for demonstration

        print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

    return best_solution, best_cost


# Assuming auxiliary functions from previous steps are defined...

def intensify(current_solution, best_solutions, distance_matrix):
    # Placeholder for intensification logic
    return current_solution


def diversify(current_solution, distance_matrix, iteration):
    # Placeholder for diversification logic
    if iteration % 20 == 0:  # Every 20 iterations, perform a diversification
        random_route = random.choice(current_solution)
        random.shuffle(random_route)  # Example: Shuffle a random route
        current_solution[current_solution.index(random_route)] = random_route
    return current_solution


def dynamic_tabu_tenure(iteration, stuck_counter):
    if stuck_counter > 10:
        return min(20, 5 + iteration % 10)  # Increase tenure to explore more deeply
    else:
        return max(5, 10 - iteration % 5)  # Decrease tenure to allow more frequent changes


def tabu_search_with_strategies(initial_solution, distance_matrix, num_iterations, initial_tabu_tenure):
    current_solution = initial_solution
    best_solution = initial_solution
    best_cost = calculate_total_distance(initial_solution, distance_matrix)
    tabu_list = []
    stuck_counter = 0  # Tracks iterations without improvement

    for iteration in range(num_iterations):
        tabu_tenure = dynamic_tabu_tenure(iteration, stuck_counter)
        neighbors = get_neighbors(current_solution, distance_matrix)
        neighbors_costs = [calculate_total_distance(neighbor, distance_matrix) for neighbor in neighbors]
        best_neighbor_idx = np.argmin(neighbors_costs)
        best_neighbor = neighbors[best_neighbor_idx]

        if best_neighbor not in tabu_list:
            current_solution = best_neighbor
            current_cost = neighbors_costs[best_neighbor_idx]
            if current_cost < best_cost:
                best_solution = best_neighbor
                best_cost = current_cost
                stuck_counter = 0  # Reset counter on improvement
            else:
                stuck_counter += 1  # Increment counter if no improvement
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        current_solution = diversify(current_solution, distance_matrix, iteration)

        print(f"Iteration {iteration + 1}: Best Cost = {best_cost}, Tabu Tenure = {tabu_tenure}")

    return best_solution, best_cost


# Example of generating an initial solution (to be modified for actual CVRP instances)
if __name__ == "__main__":
    depot = 0
    customers = list(range(1, 10))  # Example customers
    vehicle_capacity = 15
    demand = [0] + [2] * 9  # Example demands, depot has no demand
    distance_matrix = np.zeros((10, 10))  # Placeholder, to be replaced with actual distances

    # Generate an initial solution
    initial_routes = generate_initial_solution(depot, customers, vehicle_capacity, demand, distance_matrix)
    print("Initial routes:", initial_routes)
