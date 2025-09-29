import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import numpy as np

EXCEL_FILE = "Excel/paperboy_instance.xlsx"

def read_instance(filename):
    """Read Excel instance and return structured data including distance matrix"""
    df = pd.read_excel(filename)
    locations = []
    depot_idx = None
    for idx, row in df.iterrows():
        loc = {'name': row['name'], 'x': row['x'], 'y': row['y']}
        locations.append(loc)
        if row['name'] == 'start':
            depot_idx = idx

    instance = {
        'locations': locations,
        'depot': depot_idx,
        'num_paperboys': NUM_PAPERBOYS,
    }
    instance['dist_matrix'] = compute_distance_matrix(instance)
    return instance

def compute_distance_matrix(instance):
    """Compute Manhattan distance matrix for all locations"""
    n = len(instance['locations'])
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(instance['locations'][i]['x'] - instance['locations'][j]['x']) + \
                   abs(instance['locations'][i]['y'] - instance['locations'][j]['y'])
            matrix[i][j] = matrix[j][i] = dist
    return matrix

def route_distance(instance, route):
    """Calculate total Manhattan distance for a route"""
    return sum(instance['dist_matrix'][route[i]][route[i + 1]]
               for i in range(len(route) - 1))

def evaluate_solution(instance, routes):
    """Calculate max route distance and individual route distances"""
    route_dists = [route_distance(instance, route) for route in routes]
    return max(route_dists), route_dists

def random_constructive(instance):
    """Generate solution using random assignment"""
    start_time = time.time()
    indices = [i for i in range(len(instance['locations'])) if i != instance['depot']]
    random.shuffle(indices)

    routes = [[] for _ in range(instance['num_paperboys'])]
    for i, idx in enumerate(indices):
        routes[i % instance['num_paperboys']].append(idx)

    for route in routes:
        route.insert(0, instance['depot'])

    max_dist, route_dists = evaluate_solution(instance, routes)
    solution = {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists
    }
    return solution, time.time() - start_time

def nearest_neighbor(instance):
    """Generate solution using nearest neighbor heuristic"""
    start_time = time.time()
    depot = instance['depot']
    unvisited = set(range(len(instance['locations']))) - {depot}
    routes = [[depot] for _ in range(instance['num_paperboys'])]
    current_positions = [depot] * instance['num_paperboys']

    while unvisited:
        min_dist = float('inf')
        best_paperboy, best_customer = None, None

        for pb_idx, pos in enumerate(current_positions):
            for customer in unvisited:
                dist = instance['dist_matrix'][pos][customer]
                if dist < min_dist:
                    min_dist = dist
                    best_paperboy, best_customer = pb_idx, customer

        routes[best_paperboy].append(best_customer)
        current_positions[best_paperboy] = best_customer
        unvisited.remove(best_customer)

    max_dist, route_dists = evaluate_solution(instance, routes)
    solution = {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists
    }
    return solution, time.time() - start_time

def nearest_neighbor_round_robin(instance):
    start_time = time.time()
    depot = instance['depot']
    unvisited = set(range(len(instance['locations']))) - {depot}
    m = instance['num_paperboys']
    routes = [[depot] for _ in range(m)]
    current_positions = [depot] * m
    pb = 0

    while unvisited:
        # pick nearest customer for the current paperboy
        pos = current_positions[pb]
        best_customer = min(unvisited, key=lambda c: instance['dist_matrix'][pos][c])
        routes[pb].append(best_customer)
        current_positions[pb] = best_customer
        unvisited.remove(best_customer)
        pb = (pb + 1) % m

    max_dist, route_dists = evaluate_solution(instance, routes)
    return {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists
    }, time.time() - start_time

def improve_route(instance, route):
    """Apply 2-opt improvement to a single route"""
    n = len(route)
    best_route = route
    best_dist = route_distance(instance, route)

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = (
                    route[:i] +
                    list(reversed(route[i:j + 1])) +
                    route[j + 1:]
            )
            new_dist = route_distance(instance, new_route)
            if new_dist < best_dist:
                best_route = new_route
                best_dist = new_dist
    return best_route, best_dist

def best_improvement(instance, initial_solution, method="best", num_random_moves=100):
    """
    Apply best or random improvement to all routes using 2-opt.

    Parameters:
    - instance: The problem instance containing locations, depot, etc.
    - initial_solution: The initial solution to improve.
    - method: The improvement method to use ("best" or "random").
    - num_random_moves: The number of random moves to attempt (only used if method="random").

    Returns:
    - solution: The improved solution.
    - time_taken: The time taken to perform the improvement.
    """
    start_time = time.time()
    routes = copy.deepcopy(initial_solution['routes'])
    max_dist, route_dists = evaluate_solution(instance, routes)

    if method == "best":
        # Apply best improvement to each route
        for i, route in enumerate(routes):
            new_route, _ = improve_route(instance, route)
            routes[i] = new_route
    elif method == "random":
        # Perform a number of random 2-opt moves
        for _ in range(num_random_moves):
            # Randomly select a route
            route_idx = random.randint(0, len(routes) - 1)
            route = routes[route_idx]

            # Skip if the route is too short to perform a 2-opt move
            if len(route) <= 3:
                continue

            # Randomly select two indices in the route (excluding the first and last depot)
            n = len(route)
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2)

            # Create a new route by reversing the segment between i and j
            new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]

            # Calculate the distance of the new route
            new_dist = route_distance(instance, new_route)
            old_dist = route_distance(instance, route)

            # If the new route is better, replace the old one
            if new_dist < old_dist:
                routes[route_idx] = new_route

    # Evaluate the improved solution
    max_dist, route_dists = evaluate_solution(instance, routes)
    solution = {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists
    }
    return solution, time.time() - start_time

def random_neighbor(instance, solution):
    # Randomly perturb one route: e.g., swap two cities, relocate, etc.
    new_solution = copy.deepcopy(solution)
    # Apply small random change
    route_idx = random.randint(0, len(new_solution['routes']) - 1)
    route = new_solution['routes'][route_idx]
    if len(route) > 2:
        i, j = random.sample(range(1, len(route)), 2)  # skip depot
        route[i], route[j] = route[j], route[i]
    # Recompute max_distance
    new_solution['max_distance'], route_dists = evaluate_solution(instance, new_solution['routes'])
    return new_solution

def simulated_annealing(instance, initial_solution, temp=100, cooling=0.99, iterations=15, method="random"):
    start_time = time.time()

    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    current_cost = best_solution['max_distance']

    steps = []
    current_costs = []
    best_costs = []
    temperatures = []
    step_counter = 0

    while temp > 1:
        for _ in range(iterations):
            #neighbor_solution, _ = best_improvement(instance, current_solution, method=method)
            neighbor_solution = random_neighbor(instance, current_solution)
            neighbor_cost = neighbor_solution['max_distance']

            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_solution['max_distance']:
                    best_solution = copy.deepcopy(current_solution)

            # Record progress
            steps.append(step_counter)
            current_costs.append(current_cost)
            best_costs.append(best_solution['max_distance'])
            temperatures.append(temp)
            step_counter += 1

        # Cool down the temperature
        temp *= cooling

    progress_data = {
        'iterations': steps,
        'current_costs': current_costs,
        'best_costs': best_costs,
        'temperatures': temperatures
    }
    return best_solution, time.time() - start_time, progress_data


def plot_annealing_progress(history, title="Simulated Annealing Progress"):
    """
    Plot current cost, best cost, and temperature from history.
    X-axis (iteration) is in logarithmic scale.
    """
    iters = np.array(history['iterations'])
    current_costs = np.array(history['current_costs'])
    best_costs = np.array(history['best_costs'])
    temperatures = np.array(history['temperatures'])

    # Filter out iteration 0 or negative values if any (log undefined)
    mask = iters > 0
    if not mask.any():
        mask = iters >= 0  # Fallback if all iterations start at 0

    iters = iters[mask]
    current_costs = current_costs[mask]
    best_costs = best_costs[mask]
    temperatures = temperatures[mask]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Use log scale for x-axis
    ax1.set_xscale('log')

    # Plot costs
    ax1.plot(iters, current_costs, label="Current Cost", color='tab:blue', alpha=0.7, marker='.', linestyle='-',
             markersize=2)
    ax1.plot(iters, best_costs, label="Best Cost", color='tab:green', linewidth=2)
    ax1.set_xlabel("Iteration (log scale)")
    ax1.set_ylabel("Max Distance (Cost)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5, which='both')

    # Secondary y-axis for temperature
    ax2 = ax1.twinx()
    ax2.plot(iters, temperatures, label="Temperature", color='tab:red', linestyle='--', alpha=0.8)
    ax2.set_ylabel("Temperature", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left' if len(iters) > 1 else 'upper right')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def export_solution(solution, filename):
    """Export solution routes to Excel file"""
    data = []
    for i, route in enumerate(solution['routes']):
        for j, loc_idx in enumerate(route):
            data.append({
                'Paperboy': i + 1,
                'Sequence': j + 1,
                'Customer': loc_idx
            })
    pd.DataFrame(data).to_excel(filename, index=False)

def visualize_solution(instance, solution, title):
    """Plot routes using Manhattan-style visualization"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    # Plot locations
    for idx, loc in enumerate(instance['locations']):
        plt.scatter(loc['x'], loc['y'], s=100, zorder=3)
        plt.text(loc['x'], loc['y'], loc['name'],
                 ha='center', va='center', fontsize=9)

    # Plot depot
    depot = instance['locations'][instance['depot']]
    plt.scatter(depot['x'], depot['y'], s=300, c='orange',
                edgecolors='black', zorder=4, label='Depot')
    plt.text(depot['x'], depot['y'], depot['name'],
             ha='center', va='center', fontsize=11, fontweight='bold')

    # Plot routes
    for i, route in enumerate(solution['routes']):
        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            x1, y1 = instance['locations'][route[j]]['x'], instance['locations'][route[j]]['y']
            x2, y2 = instance['locations'][route[j + 1]]['x'], instance['locations'][route[j + 1]]['y']

            # Draw Manhattan path
            plt.plot([x1, x2], [y1, y1], color=color, linewidth=1.5, zorder=2)
            plt.plot([x2, x2], [y1, y2], color=color, linewidth=1.5, zorder=2)

            # Add direction indicator
            if j < len(route) - 2:
                plt.scatter(x2, y1, color=color, s=30, zorder=5)

    plt.title(f"{title}\nMax Distance: {solution['max_distance']:.1f}, Paperboys: {instance['num_paperboys']}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    global NUM_PAPERBOYS
    NUM_PAPERBOYS = 4
    instance = read_instance(EXCEL_FILE)

    # Store results for comparison
    results = {}

    print("\nRunning Simulated Annealing with fresh random solution...")
    # Generate a fresh random solution
    random_solution, _ = random_constructive(instance)
    # Run simulated annealing starting from this random solution
    solution, time_taken, progress_data = simulated_annealing(instance, random_solution, 100, 0.99, 10, method="random")
    results['Simulated Annealing'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Simulated Annealing")
    plot_annealing_progress(progress_data)

    # 1. Random Constructive Heuristic
    print("Running Random Constructive Heuristic...")
    solution, time_taken = random_constructive(instance)
    results['Random Constructive'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Random Constructive")

    # 2. Best Improvement from Random Solution
    print("\nImproving Random Solution...")
    solution, time_taken = best_improvement(instance, solution)
    results['Best Improvement (Random)'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Best Improvement (Random Initial)")

    # 3. Nearest Neighbor Heuristic
    print("\nRunning Nearest Neighbor Heuristic...")
    solution, time_taken = nearest_neighbor(instance)
    results['Nearest Neighbor'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Nearest Neighbor")

    # 4. Best Improvement from NN
    print("\nImproving NN Solution...")
    solution, time_taken = best_improvement(instance, solution)
    results['Best Improvement (NN)'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Best Improvement (NN)")

    # 3. Nearest Neighbor Heuristic
    print("\nRunning Nearest Neighbor Heuristic with Round Robin...")
    solution, time_taken = nearest_neighbor_round_robin(instance)
    results['Nearest Neighbor RR'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Nearest Neighbo with Round Robinr")

    # 4. Best Improvement from NN
    print("\nImproving Nearest Neighbuor Solution with Round Robin...")
    solution, time_taken = best_improvement(instance, solution)
    results['Best Improvement (NN) RR'] = {
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, "Best Improvement (NN) RR")


    # 5. Simulated Annealing starting with a fresh random solution

    # Export best solution
    export_solution(solution, "optimized_routes.xlsx")

    # Display results
    print("\n" + "=" * 50)
    print("Experimental Results:")
    print("=" * 50)
    for method, metrics in results.items():
        print(f"{method + ':':<30} {metrics['Max Distance']:.1f} (Time: {metrics['Time']:.2f}s)")

if __name__ == "__main__":
    main()