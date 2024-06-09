import numpy as np


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

    for iteration in range(iterations):
        solutions = []
        for ant in range(num_ants):
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
