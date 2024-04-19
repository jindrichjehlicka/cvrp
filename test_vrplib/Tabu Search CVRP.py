import numpy as np
import copy


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def total_distance(route, distance_matrix):
    total_dist = 0
    for i in range(len(route)):
        total_dist += distance_matrix[route[i - 1]][route[i]]
    return total_dist


def generate_initial_solution(node_locations):
    route = list(range(len(node_locations)))
    np.random.shuffle(route)
    return route


def find_neighbors(solution):
    neighbors = []
    for i in range(1, len(solution) - 2):
        for j in range(i + 1, len(solution)):
            if j - i == 1: continue  # We do not consider adjacent swaps
            neighbor = solution.copy()
            neighbor[i:j] = solution[i:j][::-1]
            neighbors.append(neighbor)
    return neighbors


def tabu_search(node_locations, distance_matrix, num_iterations, tabu_tenure):
    best_solution = generate_initial_solution(node_locations)
    best_cost = total_distance(best_solution, distance_matrix)
    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = find_neighbors(best_solution)
        neighbors_costs = [total_distance(neighbor, distance_matrix) for neighbor in neighbors]

        best_neighbor = neighbors[np.argmin(neighbors_costs)]
        best_neighbor_cost = min(neighbors_costs)

        if best_neighbor_cost < best_cost and best_neighbor not in tabu_list:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost
            tabu_list.append(best_solution)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

    return best_solution, best_cost


# Example usage
if __name__ == "__main__":
    # Example node locations (x, y)
    node_locations = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]

    # Generate a distance matrix
    num_nodes = len(node_locations)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i][j] = euclidean_distance(node_locations[i], node_locations[j])

    # Run Tabu Search
    num_iterations = 100
    tabu_tenure = 5
    best_solution, best_cost = tabu_search(node_locations, distance_matrix, num_iterations, tabu_tenure)
    print(f"Best Solution: {best_solution}\nBest Cost: {best_cost}")
