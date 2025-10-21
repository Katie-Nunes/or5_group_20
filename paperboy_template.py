import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
from tqdm import tqdm

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

def nearest_neighbor(instance, round_robin=False):
    start_time = time.time()
    depot = instance['depot']
    unvisited = set(range(len(instance['locations']))) - {depot}
    m = instance['num_paperboys']
    routes = [[depot] for _ in range(m)]
    current_positions = [depot] * m

    if round_robin:
        pb = 0
        while unvisited:
            # Pick nearest customer for the current paperboy
            pos = current_positions[pb]
            best_customer = min(unvisited, key=lambda c: instance['dist_matrix'][pos][c])
            routes[pb].append(best_customer)
            current_positions[pb] = best_customer
            unvisited.remove(best_customer)
            pb = (pb + 1) % m
    else:
        while unvisited:
            # Find globally nearest customer-paperboy pair
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


def apply_inter_route_transfer(solution):
    """
    Moves one random customer from one route to another.
    Good for load balancing (minimizing max_distance).
    """
    routes = solution['routes']
    num_routes = len(routes)

    # 1. Identify routes that can essentially 'give' a customer (must have >1 location, i.e., depot + at least 1 customer)
    possible_sources = [i for i, r in enumerate(routes) if len(r) > 1]
    if not possible_sources:
        return False  # No moves possible

    # 2. Pick a source route and a customer to move (excluding depot at index 0)
    src_r_idx = random.choice(possible_sources)
    src_route = routes[src_r_idx]
    cust_idx = random.randint(1, len(src_route) - 1)
    customer = src_route.pop(cust_idx)

    # 3. Pick a DIFFERENT target route
    possible_targets = [i for i in range(num_routes) if i != src_r_idx]
    tgt_r_idx = random.choice(possible_targets)
    tgt_route = routes[tgt_r_idx]

    # 4. Insert customer at random valid position in target (from index 1 to end)
    insert_pos = random.randint(1, len(tgt_route))
    tgt_route.insert(insert_pos, customer)

    return True

def generate_2opt_neighbors(route: list):
    """Generate all possible 2-opt neighbors for a route"""
    neighbors = []
    n = len(route)

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            neighbors.append(route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:])
    return neighbors

def generate_swap_neighbors(route: list):
    """Generate all possible swap neighbors for a route"""
    neighbors = []
    n = len(route)

    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors

def generate_relocate_neighbors(route: list):
    """Generate all possible move neighbors (relocate within same route)"""
    neighbors = []
    n = len(route)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if i != j:
                new_route = route.copy()
                node = new_route.pop(i)
                new_route.insert(j, node)
                neighbors.append(new_route)
    return neighbors

def generate_neighbors(route_or_routes, neighborhood: str):
    """Generate all possible neighbors given a neighborhood structure"""
    if neighborhood == "2-opt":
        return generate_2opt_neighbors(route_or_routes)
    elif neighborhood == "swap":
        return generate_swap_neighbors(route_or_routes)
    elif neighborhood == "move":
        return generate_relocate_neighbors(route_or_routes)

def best_improvement_route(instance, route: list, neighborhood: str = "2-opt") -> tuple[list, float]:
    """Apply best improvement to a single route using specified neighborhood"""
    best_route = route
    best_dist = route_distance(instance, route)

    neighbors = generate_neighbors(route, neighborhood)

    for new_route in neighbors:
        new_dist = route_distance(instance, new_route)
        if new_dist < best_dist:
            best_route = new_route
            best_dist = new_dist
    return best_route, best_dist

def first_improvement_route(instance, route: list, neighborhood: str = "2-opt") -> tuple[list, float]:
    """Apply first improvement to a single route using specified neighborhood"""
    current_dist = route_distance(instance, route)

    neighbors = generate_neighbors(route, neighborhood)

    for new_route in neighbors:
        new_dist = route_distance(instance, new_route)
        if new_dist < current_dist:
            return new_route, new_dist
    return route, current_dist

def _apply_single_route_improvement(instance, solution, move_generator):
    routes = solution['routes']
    route_idx = random.randint(0, len(routes) - 1)
    route = routes[route_idx]

    if len(route) > 3:
        new_route = move_generator(route)
        routes[route_idx] = new_route
        solution['max_distance'], solution['route_distances'] = evaluate_solution(instance, routes)


def random_neighbor(instance, solution, structure: str = None) -> dict:
    """
    Generates exactly ONE neighbor solution efficiently.
    Crucial: Creates a DEEP COPY so we don't corrupt the current solution.
    """
    # CRITICAL: Work on a copy, otherwise you modify the 'current' solution
    # in the SA loop before deciding to accept it!
    new_solution = copy.deepcopy(solution)

    # If no structure specified, pick one.
    # Weighting 'transfer' heavily to ensure load balancing happens.
    if structure is None:
        structure = random.choices(
            ["transfer", "2-opt", "relocate"],
            weights=[0.4, 0.4, 0.2],  # 40% chance to try rebalancing routes
            k=1
        )[0]

    moved = False
    if structure == "transfer":
        moved = apply_inter_route_transfer(new_solution)
    elif structure == "2-opt":
        moved = generate_2opt_neighbors(new_solution)
    elif structure == "swap":
        moved = generate_swap_neighbors(new_solution)
    elif structure == "relocate":
        moved = generate_relocate_neighbors(new_solution)

    # If the chosen move wasn't possible (e.g., route too short),
    # try a transfer as fallback, or just return completely unchanged solution.
    if not moved:
        # fallback to ensure we don't get stuck if one operator fails
        apply_inter_route_transfer(new_solution)

        # Recalculate costs for the new state
    new_solution['max_distance'], new_solution['route_distances'] = \
        evaluate_solution(instance, new_solution['routes'])

    return new_solution

def improve_solution(instance, solution: dict, method: str = "best") -> tuple[dict, float]:
    start_time = time.time()
    routes = copy.deepcopy(solution["routes"])

    if method == "best":
        for idx, route in enumerate(routes):
            if len(route) > 3:  # Only improve if possible
                improved_route, _ = best_improvement_route(instance, route)
                routes[idx] = improved_route

    elif method == "first":
        for idx, route in enumerate(routes):
            if len(route) > 3:
                improved_route, _ = first_improvement_route(instance, route)
                routes[idx] = improved_route
    # Evaluate final solution
    max_dist, route_dists = evaluate_solution(instance, routes)
    improved_solution = {
        "routes": routes,
        "max_distance": max_dist,
        "route_distances": route_dists
    }
    return improved_solution, time.time() - start_time



def simulated_annealing(instance, initial_solution, temp=100, cooling=0.995, iterations=300, method="2-opt"):
    start_time = time.time()

    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    current_cost = best_solution['max_distance']

    steps = []
    current_costs = []
    best_costs = []
    temperatures = []
    step_counter = 0

    # Calculate total steps for progress bar
    n_loops = 0
    t = temp
    while t > 1:
        n_loops += 1
        t *= cooling
    total_steps = n_loops * iterations

    # Progress bar
    with tqdm(total=total_steps, desc="Simulated Annealing") as pbar:
        while temp > 0.01:
            for _ in range(iterations):
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

                pbar.update(1)  # Update progress bar

            temp *= cooling

    progress_data = {
        'iterations': steps,
        'current_costs': current_costs,
        'best_costs': best_costs,
        'temperatures': temperatures
    }
    return best_solution, time.time() - start_time, progress_data

def plot_annealing_progress(history):
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

    plt.title("Simulated Annealing Progress")
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

def run_heuristic(instance, heuristic, name, results):
    """Run a heuristic and store the results."""
    print(f"\nRunning {name}...")
    if heuristic == "nearest_neighbor_RR":
        solution, time_taken = nearest_neighbor(instance, round_robin=True)
    else:
        solution, time_taken = heuristic(instance)
    results[name] = {
        'solution': solution,
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, name)

    solution, time_taken = improve_solution(instance, solution)
    results[f'Best Improvement ({name})'] = {
        'solution': solution,
        'Max Distance': solution['max_distance'],
        'Time': time_taken
    }
    visualize_solution(instance, solution, f"Best Improvement ({name})")


def run_simulated_annealing(instance, results):
    print("\n" + "=" * 40)
    print("Running Simulated Annealing (10 Multi-Start)")
    print("=" * 40)

    best_overall_sol = None
    best_overall_cost = float('inf')
    best_overall_data = None
    total_sa_time = 0

    NUM_RUNS = 2

    for i in range(NUM_RUNS):
        print(f"\nRun {i + 1}/{NUM_RUNS}...")
        # Start from diverse random points for better exploration
        #initial_sol, _ = nearest_neighbor(instance)
        initial_sol, _ = random_constructive(instance)

        sol, time_taken, prog_data = simulated_annealing(instance, initial_sol)
        total_sa_time += time_taken
        print(f"  > Run {i + 1} finished. Cost: {sol['max_distance']:.1f}")

        if sol['max_distance'] < best_overall_cost:
            best_overall_cost = sol['max_distance']
            best_overall_sol = sol
            best_overall_data = prog_data

    print(f"\nBest SA Cost found: {best_overall_cost:.1f}")

    results['Simulated Annealing (Best of 10)'] = {
        'solution': best_overall_sol,
        'Max Distance': best_overall_sol['max_distance'],
        'Time': total_sa_time  # Total time of all 10 runs
    }

    print(best_overall_data)
    visualize_solution(instance, best_overall_sol, "Simulated Annealing (Best of 10 Runs)")
    plot_annealing_progress(best_overall_data)


def main():
    global NUM_PAPERBOYS
    NUM_PAPERBOYS = 4
    instance = read_instance(EXCEL_FILE)
    results = {}

    run_simulated_annealing(instance, results)

    heuristics = [
        {'heuristic': random_constructive, 'name': 'Random Constructive'},
        {'heuristic': nearest_neighbor, 'name': 'Nearest Neighbor'},
        {'heuristic': "nearest_neighbor_RR", 'name': 'Nearest Neighbor RR'}
    ]
    solution = None
    for heuristic in heuristics:
        run_heuristic(instance, heuristic['heuristic'], heuristic['name'], results)

    # Export best solution
    best_solution = min(results, key=lambda x: results[x]['Max Distance'])
    export_solution(results[best_solution]['solution'], "optimized_routes.xlsx")

    # Display results
    print("\n" + "=" * 50)
    print("Experimental Results:")
    print("=" * 50)
    for method, metrics in results.items():
        print(f"{method + ':':<30} {metrics['Max Distance']:.1f} (Time: {metrics['Time']:.2f}s)")

if __name__ == "__main__":
    main()