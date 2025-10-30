# Best up to now is 1300 is inside current_best_route.xlsx
import copy
import math
import random
import time
from typing import List, Dict, Tuple, Any, Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# =========================================
# CONFIGURATION
# =========================================
CONFIG = {
    'excel_file': "Excel/paperboy_instance.xlsx",
    'num_paperboys': 4,
    'sa_iterations': 300,  # 300 Iterations per temperature level
    'sa_temp': 100,  # Starting temperature
    'sa_cooling': 0.995,  # Cooling rate
    'sa_min_temp': 0.01,  # Stopping temperature
    'num_sa_runs': 69,  # 100 Number of multi-starts for SA
    'show_plots': True  # Set to False to skip intermediate plots
}


# =========================================
# DATA MODEL & EVALUATION
# =========================================

def read_instance(filename: str, num_paperboys: int) -> Dict[str, Any]:
    """Read Excel instance and return structured data including distance matrix."""
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename}. Please check paths.")

    locations = []
    depot_idx = None
    for idx, row in df.iterrows():
        loc = {'name': row['name'], 'x': row['x'], 'y': row['y']}
        locations.append(loc)
        if row['name'] == 'start':
            depot_idx = idx

    if depot_idx is None:
        raise ValueError("No location named 'start' found in Excel file.")

    instance = {
        'locations': locations,
        'depot': depot_idx,
        'num_paperboys': num_paperboys,
        'num_locations': len(locations)
    }
    instance['dist_matrix'] = _compute_distance_matrix(instance)
    return instance


def _compute_distance_matrix(instance: Dict[str, Any]) -> List[List[float]]:
    """Compute Manhattan distance matrix for all locations."""
    n = len(instance['locations'])
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(instance['locations'][i]['x'] - instance['locations'][j]['x']) + \
                   abs(instance['locations'][i]['y'] - instance['locations'][j]['y'])
            matrix[i][j] = matrix[j][i] = dist
    return matrix


def get_route_distance(instance: Dict[str, Any], route: List[int]) -> float:
    """Calculate total Manhattan distance for a single route."""
    dist_matrix = instance['dist_matrix']
    return sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def evaluate_solution(instance: Dict[str, Any], routes: List[List[int]]) -> Tuple[float, List[float]]:
    """
    Calculate max route distance (objective) and individual route distances.
    Objective: Minimax (minimize the longest single route).
    """
    route_dists = [get_route_distance(instance, route) for route in routes]
    return (max(route_dists) if route_dists else 0.0), route_dists

# =========================================
# CONSTRUCTIVE HEURISTICS
# =========================================

def random_constructive(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Generate solution using random assignment."""
    start_time = time.time()
    depot = instance['depot']
    customers = [i for i in range(instance['num_locations']) if i != depot]
    random.shuffle(customers)

    routes = [[depot] for _ in range(instance['num_paperboys'])]
    for i, cust_idx in enumerate(customers):
        routes[i % instance['num_paperboys']].append(cust_idx)

    max_dist, route_dists = evaluate_solution(instance, routes)
    return {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists,
        'time': time.time() - start_time
    }


def nearest_neighbor(instance: Dict[str, Any], round_robin: bool = False) -> Dict[str, Any]:
    """
    Nearest Neighbor construction.
    If round_robin=True: Paperboys take turns picking their nearest available customer.
    If round_robin=False: The globally nearest (paperboy, customer) pair is chosen iteratively.
    """
    start_time = time.time()
    depot = instance['depot']
    unvisited = set(range(instance['num_locations'])) - {depot}
    m = instance['num_paperboys']
    routes = [[depot] for _ in range(m)]
    current_positions = [depot] * m
    dist_matrix = instance['dist_matrix']

    if round_robin:
        pb_idx = 0
        while unvisited:
            pos = current_positions[pb_idx]
            best_customer = min(unvisited, key=lambda c: dist_matrix[pos][c])
            routes[pb_idx].append(best_customer)
            current_positions[pb_idx] = best_customer
            unvisited.remove(best_customer)
            pb_idx = (pb_idx + 1) % m
    else:
        while unvisited:
            best_dist = float('inf')
            best_pb_idx, best_customer = -1, -1
            for pb_idx in range(m):
                pos = current_positions[pb_idx]
                for customer in unvisited:
                    dist = dist_matrix[pos][customer]
                    if dist < best_dist:
                        best_dist = dist
                        best_pb_idx, best_customer = pb_idx, customer
            routes[best_pb_idx].append(best_customer)
            current_positions[best_pb_idx] = best_customer
            unvisited.remove(best_customer)

    max_dist, route_dists = evaluate_solution(instance, routes)
    return {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists,
        'time': time.time() - start_time
    }

# =========================================
# LOCAL SEARCH & METAHEURISTICS
# =========================================

def _generate_2opt_neighbors(route: List[int]) -> List[List[int]]:
    """Generate all 2-opt neighbors for a route (deterministic)."""
    neighbors = []
    n = len(route)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            neighbors.append(route[:i] + route[i:j + 1][::-1] + route[j + 1:])
    return neighbors


def improve_solution_deterministically(instance: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    """Applies Best-Improvement 2-opt to EACH route individually until local optimum."""
    start_time = time.time()
    routes = copy.deepcopy(solution["routes"])

    for r_idx, route in enumerate(routes):
        if len(route) < 4: continue
        improved = True
        while improved:
            improved = False
            best_dist = get_route_distance(instance, route)
            best_route = route
            for neighbor in _generate_2opt_neighbors(route):
                dist = get_route_distance(instance, neighbor)
                if dist < best_dist - 1e-6:
                    best_dist = dist
                    best_route = neighbor
                    improved = True
            route = best_route
        routes[r_idx] = route

    max_dist, route_dists = evaluate_solution(instance, routes)
    return {
        'routes': routes,
        'max_distance': max_dist,
        'route_distances': route_dists,
        'time': time.time() - start_time + solution.get('time', 0)
    }


def _random_transfer(routes: List[List[int]]) -> bool:
    """Move a customer from one route to another."""
    donors = [i for i, r in enumerate(routes) if len(r) > 2]
    if len(donors) < 1 or len(routes) < 2: return False
    src = random.choice(donors)
    dest = random.choice([i for i in range(len(routes)) if i != src])
    cust = routes[src].pop(random.randint(1, len(routes[src]) - 1))
    routes[dest].insert(random.randint(1, len(routes[dest])), cust)
    return True


def _random_2opt(routes: List[List[int]]) -> bool:
    """Apply 2-opt to a single random route."""
    candidates = [i for i, r in enumerate(routes) if len(r) >= 4]
    if not candidates: return False
    r_idx = random.choice(candidates)
    route = routes[r_idx]
    i, j = sorted(random.sample(range(len(route)), 2))
    if i == 0: i = 1
    if i >= j: return False
    routes[r_idx] = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    return True


def simulated_annealing(instance: Dict[str, Any], initial_solution: Dict[str, Any]) -> Tuple[
    Dict[str, Any], Dict[str, List]]:
    current_sol = copy.deepcopy(initial_solution)
    best_sol = copy.deepcopy(initial_solution)
    current_cost = current_sol['max_distance']
    best_cost = best_sol['max_distance']

    temp = CONFIG['sa_temp']
    history = {'iterations': [], 'current_costs': [], 'best_costs': [], 'temperatures': []}
    step = 0
    total_steps = int(math.log(CONFIG['sa_min_temp'] / temp) / math.log(CONFIG['sa_cooling'])) * CONFIG['sa_iterations']

    with tqdm(total=total_steps, desc="Simulated Annealing") as pbar:
        while temp > CONFIG['sa_min_temp']:
            for _ in range(CONFIG['sa_iterations']):
                neighbor = copy.deepcopy(current_sol)
                # Higher weight on transfer for load balancing
                move_type = random.choices(['transfer', '2opt'], weights=[0.6, 0.4], k=1)[0]

                if move_type == 'transfer':
                    if not _random_transfer(neighbor['routes']): _random_2opt(neighbor['routes'])
                else:
                    if not _random_2opt(neighbor['routes']): _random_transfer(neighbor['routes'])

                n_max, n_dists = evaluate_solution(instance, neighbor['routes'])
                neighbor['max_distance'] = n_max
                neighbor['route_distances'] = n_dists

                delta = n_max - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_sol = neighbor
                    current_cost = n_max
                    if current_cost < best_cost:
                        best_sol = copy.deepcopy(current_sol)
                        best_cost = current_cost

                if step % 50 == 0:
                    history['iterations'].append(step)
                    history['current_costs'].append(current_cost)
                    history['best_costs'].append(best_cost)
                    history['temperatures'].append(temp)
                step += 1
                pbar.update(1)
            temp *= CONFIG['sa_cooling']
    return best_sol, history

# =========================================
# VISUALIZATION & IO
# =========================================

def visualize_solution(instance: Dict[str, Any], solution: Dict[str, Any], title: str):
    if not CONFIG['show_plots']: return
    fig = plt.figure(figsize=(10, 8))
    cmap = matplotlib.colormaps.get_cmap('tab10')

    for loc in instance['locations']:
        plt.scatter(loc['x'], loc['y'], c='grey', alpha=0.3, s=30)
    for i, route in enumerate(solution['routes']):
        color = cmap(i % 10)
        for j in range(len(route) - 1):
            p1, p2 = instance['locations'][route[j]], instance['locations'][route[j + 1]]
            plt.plot([p1['x'], p2['x']], [p1['y'], p1['y']], color=color, lw=2, alpha=0.8)
            plt.plot([p2['x'], p2['x']], [p1['y'], p2['y']], color=color, lw=2, alpha=0.8)
        rx = [instance['locations'][n]['x'] for n in route if n != instance['depot']]
        ry = [instance['locations'][n]['y'] for n in route if n != instance['depot']]
        plt.scatter(rx, ry, color=color, s=40, zorder=3, label=f'P{i + 1}: {solution["route_distances"][i]:.0f}')
    depot = instance['locations'][instance['depot']]
    plt.scatter(depot['x'], depot['y'], s=150, c='yellow', edgecolors='k', marker='*', zorder=5)
    plt.title(f"{title}\nMax Dist: {solution['max_distance']:.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_sa_progress(history: Dict[str, List]):
    if not CONFIG['show_plots'] or not history: return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history['iterations'], history['current_costs'], color='tab:blue', alpha=0.8, label='Current')
    ax1.plot(history['iterations'], history['best_costs'], color='navy', lw=2, label='Best Found')
    ax1.set_ylabel('Max Distance', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(history['iterations'], history['temperatures'], color='tab:red', ls='--', alpha=0.5, label='Temp')
    ax2.set_ylabel('Temperature', color='tab:red')
    plt.title("SA Progress")
    plt.show()
    return fig


def plot_combined_sa_progress(
    histories: List[Dict[str, List]],
    best_idx: int
):
    """
    Plot the 'current_costs' over iterations for multiple SA runs in one figure,
    with the run at histories[best_idx] highlighted.
    """
    if not CONFIG['show_plots'] or not histories:
        return

    # Set up dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    cmap = matplotlib.colormaps.get_cmap('tab10')

    for run_idx, history in enumerate(histories):
        color = cmap(run_idx % 10)
        is_best = (run_idx == best_idx)

        ax1.plot(
            history['iterations'],
            history['current_costs'],
            color=color,
            lw= 3 if is_best else 1,
            alpha=0.9 if is_best else 0.4,
            label=f"{'Best' if is_best else None}"
        )

        ax2.plot(
            history['iterations'],
            history['temperatures'],
            color='tab:red',
            ls='--',
            lw=1,
            alpha=1
        )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Current Max‐Distance", color='tab:blue')
    ax2.set_ylabel("Temperature", color='tab:red')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    plt.title("SA Progress Across Runs (best run highlighted)")
    plt.tight_layout()
    plt.show()
    return fig

# =========================================
# MAIN
# =========================================

def main():
    random.seed(42)
    try:
        instance = read_instance(CONFIG['excel_file'], CONFIG['num_paperboys'])
    except Exception as e:
        print(f"Error: {e}")
        return

    results = {}

    # --- 1. Run Constructive Baselines ---
    print("\n" + "=" * 40 + "\nRunning Baselines\n" + "=" * 40)
    heuristics = [
        ("Random", random_constructive),
        ("NN (Greedy)", lambda inst: nearest_neighbor(inst, round_robin=False)),
        ("NN (Round Robin)", lambda inst: nearest_neighbor(inst, round_robin=True))
    ]

    for name, func in heuristics:
        print(f"Running {name}...")
        sol = func(instance)
        visualize_solution(instance, sol, name)

        # Apply local search to baseline
        imp_sol = improve_solution_deterministically(instance, sol)
        visualize_solution(instance, imp_sol, f"{name} + 2-opt")
        results[f"{name} + 2-opt"] = imp_sol
        print(f"  -> Raw: {sol['max_distance']:.1f} | Improved: {imp_sol['max_distance']:.1f}")

    # --- 2. Run Metaheuristic (SA) ---
    print("\n" + "=" * 40 + "\nRunning Simulated Annealing\n" + "=" * 40)
    best_sa_sol = {'max_distance': float('inf')}
    best_sa_hist = None
    sa_histories: List[Dict[str, List]] = []  # <-- collect each run's history

    # Use the best baseline as starting point for SA
    start_sol = min(results.values(), key=lambda x: x['max_distance'])

    for i in range(CONFIG['num_sa_runs']):
        print(f"SA Run {i + 1}/{CONFIG['num_sa_runs']}...")
        # If multiple runs, maybe randomize start for i>0
        current_start = start_sol if i == 0 else random_constructive(instance)
        sa_sol, hist = simulated_annealing(instance, current_start)

        sa_histories.append(hist)
        if sa_sol['max_distance'] < best_sa_sol['max_distance']:
            best_sa_sol = sa_sol
            best_sa_hist = hist

    with open('sa_history_file.pkl', 'wb') as f:
        pickle.dump(sa_histories, f)

    results["Simulated Annealing"] = best_sa_sol
    plt = plot_sa_progress(best_sa_hist)
    plt.savefig('Images/best_sa_progress.png')

    final_bests = [h['best_costs'][-1] for h in sa_histories]
    best_idx = int(np.argmin(final_bests))

    # now call:
    plt = plot_combined_sa_progress(sa_histories, best_idx)
    plt.savefig('Images/combined_sa_progress.png')
    plt = visualize_solution(instance, best_sa_sol, "Best SA Solution")
    plt.savefig('Images/best_sa_visualization.png')

    # --- 3. Final Comparison ---
    print("\n" + "=" * 50 + "\nFinal Results Comparison\n" + "=" * 50)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['max_distance'])
    for name, data in sorted_results:
        print(
            f"{name:<25} | Max Dist: {data['max_distance']:.1f} | Routes: {[int(d) for d in data['route_distances']]}")

    best_overall = sorted_results[0][1]
    pd.DataFrame([{'Paperboy': i + 1, 'Sequence': j + 1, 'Location': loc}
                  for i, r in enumerate(best_overall['routes']) for j, loc in enumerate(r)]) \
        .to_excel("Excel/current_bestest_of_best_route_1286.xlsx", index=False)
    print(f"\nBest solution exported to Excel/current_bestest_of_best_route_1286.xlsx (Cost: {best_overall['max_distance']:.1f})")


if __name__ == "__main__":
    main()