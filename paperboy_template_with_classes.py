# Paperboy Problem Template with Classes
# This template provides a structured approach to solving the Paperboy Problem using classes.
# It is generic and can be adapted for various heuristics and solution methods.
# It includes classes for Location, Instance, and Solution, along with methods for reading instances,
# computing distances, and evaluating solutions.
# Use this template as a starting point if you are feeling comfortable with object-oriented programming in Python.
# - OR5 Course, 5 September 2025

# start with importing standard libraries
import random
import time

# next, import third-party libraries
import pandas as pd
import matplotlib.pyplot as plt

# Program constant: Excel file name
#EXCEL_FILE = "simple_instance.xlsx"
EXCEL_FILE = "Excel/paperboy_instance.xlsx"
NUM_PAPERBOYS = 4
DISTANCE_METRIC = "manhattan"

class Location:
    """
    Represents a geographical location with a name and coordinates.

    Attributes:
        name (str): The name of the location.
        x (float): The x-coordinate of the location.
        y (float): The y-coordinate of the location.

    Args:
        name (str): The name of the location.
        x (float): The x-coordinate.
        y (float): The y-coordinate.
    """
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class Instance:
    """
    Represents an instance of the Paperboy Problem, encapsulating the locations to be visited,
    the starting location index, the number of paperboys, and the distance matrix between locations.
    Attributes:
        locations (list): A list of location coordinates or identifiers.
        start_idx (int): The index of the starting location.
        number_of_paperboys (int): The number of paperboys available for delivery.
        distance_matrix (list of lists): A matrix containing distances between each pair of locations.
    Methods:
        number_of_locations():
            Returns the total number of locations in the instance.
        get_distance(i, j):
            Returns the distance between location i and location j using the distance matrix.
    """
    def __init__(self, locations, start_idx):
        self.locations = locations
        self.start_idx = start_idx
        self.number_of_paperboys = NUM_PAPERBOYS
        self.distance_matrix = compute_distance_matrix(self, DISTANCE_METRIC)

    def number_of_locations(self):
        return len(self.locations)

    def get_distance(self, i, j):
        return self.distance_matrix[i][j]

class Solution:
    """
    Represents a solution to the Paperboy Problem, encapsulating a set of delivery routes and their associated distances.
    Attributes:
        instance: An object providing access to distance calculations between delivery points.
        routes (list of list): A list of routes, where each route is a list of delivery point indices.
        max_route_distance (float): The maximum distance among all routes.
        route_distances (list of float): The distances for each route.
    Methods:
        __init__(instance, routes):
            Initializes the Solution with the given instance and routes, and computes route distances and the maximum route distance.
        route_distance(idx):
            Calculates the total distance of the route at the given index.
        calculate_solution_quality():
            Computes the maximum route distance and the list of all route distances.
        get_route(idx):
            Returns the route at the specified index.
        get_route_distance(idx):
            Returns the distance of the route at the specified index.
    """
    def __init__(self, instance, routes):
        self.instance = instance
        self.routes = routes
        self.max_route_distance, self.route_distances = self.calculate_solution_quality()

    def route_distance(self, idx):
        """
        Calculates the total distance of a specific route.

        Args:
            idx (int): The index of the route in the self.routes list.

        Returns:
            int or float: The total distance of the route, computed as the sum of distances between consecutive stops.

        Explanation:
            This method iterates through the stops in the selected route and sums up the distances between each consecutive pair of stops using the instance's get_distance method.
        """
        route = self.routes[idx]
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.instance.get_distance(route[i], route[i + 1])
        return total_distance

    def calculate_solution_quality(self):
        """
        Calculates the quality of the current solution by determining the maximum route distance
        among all routes and the individual distances for each route.

        Returns:
            tuple:
                - max_distance (float): The longest distance among all routes.
                - route_distances (list of float): The distances for each route.
        """
        max_distance = 0
        route_distances = []
        for route in self.routes:
            route_distances.append(self.route_distance(self.routes.index(route)))
            max_distance = max(max_distance, route_distances[-1])
        return max_distance, route_distances

    def get_route(self, idx):
        return self.routes[idx]
      
    def get_route_distance(self, idx):
        return self.route_distance(idx)

def read_instance(filename):
    """
    Reads an Excel file containing location data and constructs an Instance object.

    Args:
        filename (str): The path to the Excel file containing location data. The file should have columns
            'name', 'x', and 'y' for each location.

    Returns:
        Instance: An Instance object containing a list of Location objects and the index of the starting location.

    Explanation:
        The function reads the Excel file into a pandas DataFrame, iterates over each row to create Location
        objects, and identifies the index of the row where the 'name' column is 'start' (case-insensitive).
        It then returns an Instance object initialized with the list of locations and the start index.
    """
    df = pd.read_excel(filename)
    locations = []
    start_idx = None
    for idx, row in df.iterrows():
        loc = Location(row['name'], row['x'], row['y'])
        locations.append(loc)
        if isinstance(row['name'], str) and row['name'].lower() == 'start':
            start_idx = idx
    return Instance(locations, start_idx)

def manhattan_distance(loc1, loc2):
    return abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)

def euclidean_distance(loc1, loc2):
    return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5

def compute_distance_matrix(instance, metric='manhattan'):
    """
    Computes a distance matrix for the given instance using the specified metric.

    Args:
        instance: An object containing location data. Must have a method `number_of_locations()` 
                  that returns the number of locations, and an attribute `locations` which is 
                  a list of coordinates (e.g., tuples or lists).
        metric (str, optional): The distance metric to use. Supported values are 'manhattan' 
                  for Manhattan (L1) distance and any other value for Euclidean (L2) distance.
                  Defaults to 'manhattan'.

    Returns:
        list[list[float]]: A 2D list (matrix) where element [i][j] represents the distance 
                           between location i and location j according to the specified metric.

    Note:
        The function assumes the existence of `manhattan_distance` and `euclidean_distance` 
        functions that compute the distance between two locations.
    """
    n = instance.number_of_locations()
    matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if metric == 'manhattan':
                matrix[i][j] = manhattan_distance(instance.locations[i], instance.locations[j])
            else:
                matrix[i][j] = euclidean_distance(instance.locations[i], instance.locations[j])
    return matrix

def random_constructive_heuristic(instance):
    """
    Generates an initial solution for the paperboy problem using a random constructive heuristic.
    This function randomly assigns delivery locations to a specified number of paperboys, ensuring each route starts at the depot.
    The assignment is reproducible due to a fixed random seed. The function returns a Solution object containing the generated routes,
    along with the time taken to construct the solution.
    Args:
        instance: An object containing problem data, including locations, the depot index (start_idx), and the number of paperboys.
    Returns:
        tuple:
            - Solution: An object representing the constructed routes for each paperboy.
            - float: The running time (in seconds) taken to generate the solution.
    """
    start_time = time.time()
    # Randomly assign locations to paperboys
    random.seed(42)  # For reproducibility
    indices = list(range(len(instance.locations)))
    indices.remove(instance.start_idx)
    random.shuffle(indices)
    routes = [[] for _ in range(instance.number_of_paperboys)]
    for i, idx in enumerate(indices):
        routes[i % instance.number_of_paperboys].append(idx)
    # Each route starts at depot
    for route in routes:
        route.insert(0, instance.start_idx)

    end_time = time.time()
    running_time = end_time - start_time
    return Solution(instance, routes), running_time

def best_improvement_heuristic_random_initial(instance):
    """
    Applies the best improvement 2-opt heuristic to each route in a randomly constructed initial solution
    for the Paperboy Problem.
    This function first generates an initial solution using a random constructive heuristic. Then, for each route,
    it iteratively applies the 2-opt best improvement algorithm: for all possible pairs of edges, it reverses the
    segment between them if doing so results in a shorter route. The process repeats until no further improvement
    can be found for any route.
    During optimization, the function prints progress information for each route, including initial, current, and
    improved distances.
    Args:
        instance: An object representing the problem instance, which must provide a distance matrix and
                  be compatible with the random_constructive_heuristic and Solution classes.
    Returns:
        A tuple containing:
            - Solution: An object representing the improved solution with optimized routes.
            - running_time (float): The time taken to perform the optimization, in seconds.
    """
    initial_solution = random_constructive_heuristic(instance)
    routes = initial_solution[0].routes

    start_time = time.time()
    # Apply 2-opt best improvement to each route
    for r_idx in range(len(routes)):
        improved = True
        n = len(routes[r_idx])
        init_dist = initial_solution[0].get_route_distance(r_idx)
        while improved:
            improved = False
            route = routes[r_idx]
            orig_dist = sum(instance.distance_matrix[route[k]][route[k+1]] for k in range(n-1))
            best_gain = 0
            best_new_route = None
            # Try all pairs (i, j) with i >= 1, j <= n-2, j >= i+1
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    # Create new route with [i..j] reversed
                    new_route = (
                        route[:i] +
                        route[j:i-1:-1] +
                        route[j+1:]
                    )
                    # Compute distances
                    new_dist = 0
                    for k in range(n-1):
                        new_dist += instance.distance_matrix[new_route[k]][new_route[k+1]]
                    gain = orig_dist - new_dist
                    if gain > best_gain:
                        best_gain = gain
                        best_new_route = new_route
            if best_gain > 0 and best_new_route is not None:
                routes[r_idx] = best_new_route
                new_dist = sum(instance.distance_matrix[best_new_route[k]][best_new_route[k+1]] for k in range(n-1))
                improved = True
                print(f"\rInvestigating route for Paperboy {r_idx+1}, Initial distance: {init_dist:.1f}, Current distance: {orig_dist:.1f} -> New distance: {new_dist:.1f}", end=" ")
        print()

    end_time = time.time()
    running_time = end_time - start_time

    return Solution(instance, routes), running_time

def draw_solution(solution : Solution, method_name):
    """
    Visualizes the solution to the paperboy routing problem using matplotlib.

    This function plots the locations (including the start location) and the routes taken by each paperboy.
    Each route is displayed in a different color, and the start location is highlighted.
    The plot includes labels for each location, route distances, and a legend for clarity.

    Args:
        solution (Solution): An object containing the instance data, routes, route distances, and max route distance.
        method_name (str): The name of the method or algorithm used to generate the solution, displayed in the plot title.

    Displays:
        A matplotlib figure showing the locations, routes, and relevant statistics.
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(8, 8))
    # Plot locations
    for i, loc in enumerate(solution.instance.locations):
        if loc.name == "start":
            plt.scatter(loc.x, loc.y, c='orange', s=300, zorder=3, edgecolors='black', label='Start')
            plt.text(loc.x, loc.y, loc.name, fontsize=11, ha='center', va='center', color='black', fontweight='bold', zorder=4)
        else:
            plt.scatter(loc.x, loc.y, c='black', s=100, zorder=3)
            plt.text(loc.x, loc.y, loc.name, fontsize=9, ha='center', va='center', color='white', zorder=4)
    # Plot routes
    for idx, route in enumerate(solution.routes):
        color = colors[idx % len(colors)]
        xs = [solution.instance.locations[i].x for i in route]
        ys = [solution.instance.locations[i].y for i in route]
        plt.plot(xs, ys, color=color, linewidth=1, label=f'Paperboy {idx+1}, distance {solution.route_distances[idx]:.1f}', zorder=2)
    plt.title(f"{method_name}, {DISTANCE_METRIC}: Max route distance = {solution.max_route_distance:.1f}, Paperboys = {solution.instance.number_of_paperboys }")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_results(results):
    print("Results for different solution methods:")
    for name, metrics in results.items():
        metrics_str = ", ".join([f"{metric}: {value:<10}" for metric, value in metrics.items()])
        print(f"{name:50}: {metrics_str}")

def main():
    instance = read_instance(EXCEL_FILE)
    methods = {
        "Random Constructive Heuristic": random_constructive_heuristic,
        "Best Improvement Heuristic": best_improvement_heuristic_random_initial,
        # Here, you can add more heuristic methods
    }
    results = {}
    for name, method in methods.items():
        print(f"\nRunning {name}...")
        solution, running_time = method(instance)
        draw_solution(solution, name)
        results[name] = {'Max route distance': solution.max_route_distance, 'Running time': running_time}
        print(f"\n{name}: Max route distance = {solution.max_route_distance}, Running time = {running_time:.2f}")

    print_results(results)

if __name__ == "__main__":
    main()
