import numpy as np
from collections import deque
import random


class Node:
    def __init__(self, demand, is_routed=False):
        self.demand = demand
        self.is_routed = is_routed


class Vehicle:
    def __init__(self, capacity, route=None):
        self.capacity = capacity
        self.route = route if route is not None else []
        self.load = 0
        self.cost = 0

    def calculate_cost(self, distance_matrix):
        self.cost = 0
        for i in range(1, len(self.route)):
            self.cost += distance_matrix[self.route[i - 1]][self.route[i]]
        return self.cost


class TabuSearch:
    def __init__(self, nodes, vehicles, distance_matrix, n_tabu, max_iterations):
        self.nodes = nodes
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.n_tabu = n_tabu
        self.max_iterations = max_iterations
        self.tabu_list = set()
        self.tabu_queue = deque(maxlen=n_tabu)

    def is_tabu(self, move):
        return move in self.tabu_list

    def aspiration(self, current_cost, new_cost):
        return new_cost < current_cost

    def solve(self):
        for vehicle in self.vehicles:
            vehicle.calculate_cost(self.distance_matrix)

        best_cost = sum(vehicle.cost for vehicle in self.vehicles)
        best_vehicles = self.vehicles[:]
        current_cost = best_cost

        for iteration in range(self.max_iterations):
            for vehicle in self.vehicles:
                for i in range(1, len(vehicle.route) - 1):
                    for vehicle2 in self.vehicles:
                        for j in range(len(vehicle2.route)):
                            # Simulate the move
                            node = vehicle.route[i]
                            if vehicle is vehicle2 and i == j:
                                continue
                            if vehicle2.load + self.nodes[node].demand > vehicle2.capacity:
                                continue

                            # Calculate the costs
                            cost_increase = self.distance_matrix[vehicle2.route[j - 1]][node] + \
                                            self.distance_matrix[node][vehicle2.route[j]] - \
                                            self.distance_matrix[vehicle2.route[j - 1]][vehicle2.route[j]]
                            cost_reduction = self.distance_matrix[vehicle.route[i - 1]][vehicle.route[i + 1]] - \
                                             self.distance_matrix[vehicle.route[i - 1]][node] - \
                                             self.distance_matrix[node][vehicle.route[i + 1]]

                            new_cost = current_cost + cost_increase - cost_reduction
                            move = (vehicle.route[i - 1], node, vehicle2.route[j - 1], node)

                            if not self.is_tabu(move) or self.aspiration(current_cost, new_cost):
                                # Perform the move
                                vehicle.route.remove(node)
                                vehicle2.route.insert(j, node)
                                vehicle.calculate_cost(self.distance_matrix)
                                vehicle2.calculate_cost(self.distance_matrix)
                                self.tabu_list.add(move)
                                self.tabu_queue.append(move)

                                # Update loads
                                vehicle.load -= self.nodes[node].demand
                                vehicle2.load += self.nodes[node].demand

                                # Update costs
                                current_cost = new_cost
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_vehicles = [v for v in self.vehicles]

        # Set best solution found
        self.vehicles = best_vehicles
        print(f'Final cost: {sum(v.cost for v in self.vehicles)}')
        return self.vehicles


# Example usage:
nodes = [Node(demand=10) for _ in range(10)]
vehicles = [Vehicle(capacity=50) for _ in range(3)]
distance_matrix = np.random.rand(10, 10) * 100
ts = TabuSearch(nodes, vehicles, distance_matrix, n_tabu=10, max_iterations=100)
ts.solve()
