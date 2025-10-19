import os
import sys
import numpy as np
import pandas as pd
import threading
import concurrent.futures
import matplotlib.image as mpimg
from datetime import datetime
from nsga_utils import (
    cache_exists,
    get_bounding_box,
    load_cache,
    save_cache,
    initialize_data,
    get_param_hash,
    ensure_output_dir,
    fetch_osm_static_map,
    plot_all_candidates_with_demand,
    log_with_timestamp,
    monitor_progress,
)
from nsga_algorithm import (
    nsga_ii_optimization_with_label,
)
from matplotlib import pyplot as plt

# --- Evaluation ---
def plot_pareto_fronts_with_parents(F, parents, iteration, label=""):
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap('tab10')(np.arange(10))
    for i, front in enumerate(F):
        if not front:
            continue
        costs = np.array([ind.Cost for ind in front])
        plt.scatter(costs[:, 0], costs[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=40)
    parent_costs = np.array([ind.Cost for ind in parents])
    plt.scatter(parent_costs[:, 0], parent_costs[:, 1], color='black', marker='*', s=120, label='Selected Parents')
    plt.title(f"Pareto Fronts and Selected Parents (Iteration {iteration}) - {label}")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'pareto_front_iter_{label}_{iteration:03d}.png'))
    plt.close()

def plot_pareto_front(pareto_front, label=""):
    output_dir = ensure_output_dir()
    costs = np.array([ind.Cost for ind in pareto_front])
    f1 = costs[:, 0]
    f2 = costs[:, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, color='blue', marker='o', s=50, label='Pareto Solutions')
    min_f1_idx = np.argmin(f1)
    min_f2_idx = np.argmin(f2)
    plt.scatter(f1[min_f1_idx], f2[min_f1_idx], color='red', marker='s', s=100, label='Min Cost Solution')
    plt.scatter(f1[min_f2_idx], f2[min_f2_idx], color='green', marker='^', s=100, label='Min Deviation Solution')
    plt.title(f"Final Pareto Front {label}")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'pareto_front_{label}.png'))
    plt.close()

def plot_objective_evolution(evolution_stats, label="", objective_idx=0):
    """
    Plot the evolution of the tracked solutions for a single objective over iterations.
    objective_idx: 0 for cost, 1 for stddev/equity
    """
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    iterations = np.arange(1, len(evolution_stats['cost']) + 1)
    obj_name = "Total Cost ($f_1$)" if objective_idx == 0 else "Standard Deviation ($f_2$, Equity)"
    # Plot cost-optimized
    plt.plot(iterations, [v[objective_idx] for v in evolution_stats['cost']], 'b-', marker='o', label='Min Cost')
    # Plot stddev-optimized
    plt.plot(iterations, [v[objective_idx] for v in evolution_stats['stddev']], 'r-', marker='s', label='Min Stddev')
    # Plot balanced
    plt.plot(iterations, [v[objective_idx] for v in evolution_stats['balanced']], 'g-', marker='^', label='Balanced')
    plt.xlabel("Iteration")
    plt.ylabel(obj_name)
    plt.title(f"Pareto Objective Evolution ({label}) - {obj_name}")
    plt.legend()
    plt.grid(True)
    fname = f'pareto_evolution_obj{objective_idx+1}_{label}.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def plot_spatial_solution(best_solution, data, label="", candidate_610=None, candidate_955=None, map_path=None):
    output_dir = ensure_output_dir()
    xy_candidate = data['xy_candidate']
    xy_community = data['xy_community']
    Q = data['Q']
    built_idx = np.where(np.any(best_solution.x, axis=1))[0]

    if len(built_idx) == 0:
        log_with_timestamp(f"Warning: No facilities are open in the solution for {label}. Skipping spatial plot.", data.get('log_file'))
        return

    facility_coords = xy_candidate[built_idx, :]
    facility_types = np.argmax(best_solution.x[built_idx, :], axis=1)
    #facility_capacities = Q[facility_types]
    type_colors = ['#337AFF', '#FF9900', '#33CC33']  # blue, orange, green

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Calculate bounding box for extent
    margin = 0.05
    min_x, max_x, min_y, max_y = get_bounding_box(xy_candidate, xy_community, margin=margin)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Add map background if map_path exists
    if map_path is not None and os.path.exists(map_path):
        static_img = mpimg.imread(map_path)
        plt.imshow(static_img, extent=(min_x, max_x, min_y, max_y), aspect='auto', alpha=0.3)

    # Only plot candidates for the current label
    if "610" in label and candidate_610 is not None:
        plt.scatter(candidate_610[:, 0], candidate_610[:, 1], 
                    marker='x', s=30, color='green', alpha=0.3, label='610 Candidates')
    elif "955" in label and candidate_955 is not None:
        plt.scatter(candidate_955[:, 0], candidate_955[:, 1], 
                    marker='x', s=30, color='red', alpha=0.3, label='955 Candidates')
    else:
        # For baseline or other labels, show both
        if candidate_610 is not None:
            plt.scatter(candidate_610[:, 0], candidate_610[:, 1], 
                        marker='x', s=30, color='green', alpha=0.3, label='610 Candidates')
        if candidate_955 is not None:
            plt.scatter(candidate_955[:, 0], candidate_955[:, 1], 
                        marker='x', s=30, color='red', alpha=0.3, label='955 Candidates')

    # Plot community demand bubbles
    Em_norm = data['E_L'].mean()
    bubble_sizes = data['E_L'].flatten() / Em_norm * 500 
    plt.scatter(xy_community[:, 0], xy_community[:, 1], 
                s=bubble_sizes, color='gray', alpha=0.3, label='Community Demand (Size=Demand)')

    # Plot built facilities (colored circles, size by capacity)
    for t, cap, color in zip(range(len(Q)), Q, type_colors):
        type_mask = (facility_types == t)
        plt.scatter(facility_coords[type_mask, 0], facility_coords[type_mask, 1], 
                    marker='o', s=cap*8, edgecolors='black', 
                    color=color, label=f'Type {t+1} (Cap: {cap})')

    # Plot allocation lines
    allocation_matrix = best_solution.y
    for j in range(data['n_community']):
        assigned_facilities = np.where(allocation_matrix[:, j] > 0)[0]
        for i in assigned_facilities:
            plt.plot([xy_community[j, 0], xy_candidate[i, 0]], 
                     [xy_community[j, 1], xy_candidate[i, 1]], 
                     'k-', linewidth=0.5, alpha=0.5)

    plt.title(f"Facility Locations and Community Demands (Best Solution) {label}")
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.legend(scatterpoints=1)
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(output_dir, f'spatial_solution_{label}.png'))
    plt.close()

def plot_pareto_evolution(all_fronts, label=""):
    """
    Plot all points from the best Pareto front of each iteration.
    Show only 5 colors for legend: first, 1/4, middle, 3/4, last iteration.
    """
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    n_iters = len(all_fronts)
    cmap = plt.colormaps.get_cmap('coolwarm')
    # Indices for legend: first, 1/4, middle, 3/4, last
    legend_idxs = [0, n_iters//4, n_iters//2, 3*n_iters//4, n_iters-1]
    legend_idxs = sorted(set([i for i in legend_idxs if i < n_iters]))
    handles = []
    labels = []
    # Plot all iterations, but only label the selected ones
    for i in reversed(range(n_iters)):
        front = all_fronts[i]
        if not front:
            continue
        costs = np.array([ind.Cost for ind in front])
        color = cmap(i / max(n_iters - 1, 1))
        legend_label = None
        if i in legend_idxs:
            legend_label = f'Iter {i+1}'
        sc = plt.scatter(costs[:, 0], costs[:, 1], color=color, alpha=0.7, s=30, label=legend_label)
        if legend_label:
            handles.append(sc)
            labels.append(legend_label)
    plt.title(f"Pareto Front Evolution ({label})")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.grid(True)
    if handles:
        plt.legend(handles, labels, title="Iteration", loc="best")
    plt.savefig(os.path.join(output_dir, f'pareto_evolution_{label}.png'))
    plt.close()


def export_allocation_summary(solution, data, label):
    output_dir = ensure_output_dir()
    Q = data['Q']
    y = solution.y
    x = solution.x
    n_candidate, n_types = x.shape
    n_community = data['n_community']

    rows = []
    for i in range(n_candidate):
        if np.any(x[i, :]):
            nh_type = np.argmax(x[i, :]) + 1
            capacity = Q[nh_type - 1]
            allocated_comms = []
            allocated_people = []
            for j in range(n_community):
                if y[i, j] > 0:
                    allocated_comms.append(str(j + 1))  # 1-based index for communities
                    allocated_people.append(str(int(y[i, j])))
            if allocated_comms:
                rows.append({
                    "Location": i + 1,  # 1-based index for location
                    "Nursing Home Type": nh_type,
                    "Capacity": capacity,
                    "Allocated communities": ",".join(allocated_comms),
                    "The Number of Allocated People": ",".join(allocated_people)
                })

    df = pd.DataFrame(rows, columns=[
        "Location", "Nursing Home Type", "Capacity",
        "Allocated communities", "The Number of Allocated People"
    ])
    # Save to CSV, TXT, and Excel
    df.to_csv(os.path.join(output_dir, f"allocation_summary_{label}.csv"), index=False)
    df.to_csv(os.path.join(output_dir, f"allocation_summary_{label}.txt"), sep='\t', index=False)
    df.to_excel(os.path.join(output_dir, f"allocation_summary_{label}.xlsx"), index=False)
    print(f"Saved allocation summary for {label} to allocation_summary_{label}.csv, .txt, and .xlsx")
    print(df.head(10).to_string(index=False))  # Print preview


def run_and_plot_nsga(data, label, candidate_610=None, candidate_955=None, return_front=False,
                      MaxIt=200, nPop=80, pCrossover=0.7, pMutation=0.15, patience=20, map_path=None):
    log_file = f'output/nsga_log_{label}.txt'
    log_with_timestamp(f"Starting NSGA-II optimization for {label}.", log_file)

    pareto_front, all_fronts, evolution_stats = nsga_ii_optimization_with_label(
        data, label, MaxIt, nPop, pCrossover, pMutation, patience, log_file=log_file)

    log_with_timestamp(f"Completed NSGA-II optimization for {label}.", log_file)

    plot_pareto_front(pareto_front, label)
    log_with_timestamp(f"Plotted Pareto front for {label}.", log_file)

    plot_pareto_evolution(all_fronts, label)
    log_with_timestamp(f"Plotted Pareto evolution for {label}.", log_file)

    plot_objective_evolution(evolution_stats, label, objective_idx=0)
    log_with_timestamp(f"Plotted objective evolution (Total Cost) for {label}.", log_file)

    plot_objective_evolution(evolution_stats, label, objective_idx=1)
    log_with_timestamp(f"Plotted objective evolution (Standard Deviation) for {label}.", log_file)

    costs = np.array([ind.Cost for ind in pareto_front])
    min_cost_idx = np.argmin(costs[:, 0])
    best_solution = pareto_front[min_cost_idx]

    plot_spatial_solution(best_solution, data, label, candidate_610=candidate_610, candidate_955=candidate_955, map_path=map_path)
    log_with_timestamp(f"Plotted spatial solution for {label}.", log_file)

    export_allocation_summary(best_solution, data, label)
    log_with_timestamp(f"Exported allocation summary for {label}.", log_file)

    if return_front:
        return pareto_front
    return best_solution

def show_parameters(data, label, log_file=None, run_label=""):
    """
    Print and optionally log the main parameters for a candidate set.
    """
    params = [
        f"--- Parameters for {label} - {run_label} ---",
        f"Number of Candidates: {data['n_candidate']}",
        f"Number of Communities: {data['n_community']}",
        f"Capacity Options (Q): {data['Q']}",
        f"Construction Costs (C): {data['C']}",
        f"Travel Cost per Unit (Cp): {data['Cp']}",
        f"Penalty Cost (Cpp): {data['Cpp']}",
        f"Number of Facilities to Open (U): {data['U']}",
        f"Minimum Distance Constraint (D): {data['D']}",
        f"Fuzzy Parameter (lambda): {data['lambda_']}",
        f"Lower Bound Demand (E_L): {np.round(data['E_L'].flatten(), 2)}",
        f"Upper Bound Demand (E_U): {np.round(data['E_U'].flatten(), 2)}",
        "-----------------------------"
    ]
    for line in params:
        print(line)
        if log_file is not None:
            log_with_timestamp(line, log_file)

def plot_final_solution_comparison(pareto_610, pareto_955, label_610="610", label_955="955", run_label=""):
    output_dir = ensure_output_dir()
    plt.figure(figsize=(10, 8))

    cost_scale = 1e6  # Scale cost to millions

    # Get and scale costs for 610
    costs_610 = np.array([ind.Cost for ind in pareto_610])
    costs_610_scaled = costs_610.copy()
    costs_610_scaled[:, 0] = costs_610_scaled[:, 0] / cost_scale
    min_cost_idx_610 = np.argmin(costs_610_scaled[:, 0])
    min_stddev_idx_610 = np.argmin(costs_610_scaled[:, 1])
    rand_idx_610 = np.random.randint(len(costs_610_scaled))
    points_610 = [
        (costs_610_scaled[min_cost_idx_610], f"{label_610} Min Cost"),
        (costs_610_scaled[min_stddev_idx_610], f"{label_610} Min Stddev"),
        (costs_610_scaled[rand_idx_610], f"{label_610} Random")
    ]

    # Get and scale costs for 955
    costs_955 = np.array([ind.Cost for ind in pareto_955])
    costs_955_scaled = costs_955.copy()
    costs_955_scaled[:, 0] = costs_955_scaled[:, 0] / cost_scale
    min_cost_idx_955 = np.argmin(costs_955_scaled[:, 0])
    min_stddev_idx_955 = np.argmin(costs_955_scaled[:, 1])
    rand_idx_955 = np.random.randint(len(costs_955_scaled))
    points_955 = [
        (costs_955_scaled[min_cost_idx_955], f"{label_955} Min Cost"),
        (costs_955_scaled[min_stddev_idx_955], f"{label_955} Min Stddev"),
        (costs_955_scaled[rand_idx_955], f"{label_955} Random")
    ]

    # Plot and annotate with fixed offsets and rounded values
    colors_610 = ['blue', 'green', 'black']
    markers_610 = ['o', 's', '^']
    offsets_610 = [(-60, 0), (-30, -15), (10, 15)]  # blue middle left, green bottom middle, black offset is top right

    colors_955 = ['orange', 'red', 'purple']
    markers_955 = ['o', 's', '^']
    offsets_955 = [(60, 0), (30, -15), (-10, 15)]  # orange middle right, red bottom middle, purple offset is top left

    for i, (pt, name) in enumerate(points_610):
        plt.scatter(pt[0], pt[1], color=colors_610[i], marker=markers_610[i], s=120, label=name)
        plt.annotate(f"{pt[0]:.2f}, {pt[1]:.2f}", (pt[0], pt[1]),
                     textcoords="offset points", xytext=offsets_610[i], fontsize=8, color=colors_610[i],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for i, (pt, name) in enumerate(points_955):
        plt.scatter(pt[0], pt[1], color=colors_955[i], marker=markers_955[i], s=120, label=name)
        plt.annotate(f"{pt[0]:.2f}, {pt[1]:.2f}", (pt[0], pt[1]),
                     textcoords="offset points", xytext=offsets_955[i], fontsize=8, color=colors_955[i],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.title(f"Final Solution Comparison: 610 vs 955 - {run_label}")
    plt.xlabel("Objective 1: Total Cost ($f_1$) [Millions]")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'final_solution_comparison_{run_label}.png'))
    plt.close()


def plot_final_pareto_comparison(pareto_610, pareto_955, run_label=""):
    """
    Plot the final Pareto fronts of 610 and 955 candidates on one figure for comparison.
    """
    if not pareto_610 or not pareto_955:
        raise ValueError("One or both Pareto fronts are empty. Cannot plot comparison.")
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    costs_610 = np.array([ind.Cost for ind in pareto_610])
    costs_955 = np.array([ind.Cost for ind in pareto_955])
    plt.scatter(costs_610[:, 0], costs_610[:, 1], color="orange", marker="o", s=50, label="610 Candidates")
    plt.scatter(costs_955[:, 0], costs_955[:, 1], color="red", marker="s", s=50, label="955 Candidates")
    plt.title(f"Final Pareto Front Comparison (610 vs 955 Candidates)-{run_label}")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"pareto_final_comparison-{run_label}.png"))
    plt.close()


def export_final_solution_comparison(pareto_610, pareto_955, run_label):
    """
    Export a summary comparison of the best solutions from both Pareto fronts to CSV and TXT.
    """
    output_dir = ensure_output_dir()

    def get_best_solutions(pareto_front):
        if not pareto_front:
            raise ValueError("Pareto front is empty. Cannot determine best solutions.")
        costs = np.array([ind.Cost for ind in pareto_front])
        min_cost_idx = np.argmin(costs[:, 0])
        min_stddev_idx = np.argmin(costs[:, 1])
        balanced_idx = np.argmin(
            np.abs(
                (costs[:, 0] - costs[:, 0].min()) / (costs[:, 0].max() - costs[:, 0].min() + 1e-8)
                - (costs[:, 1] - costs[:, 1].min()) / (costs[:, 1].max() - costs[:, 1].min() + 1e-8)
            )
        )
        return [pareto_front[min_cost_idx], pareto_front[min_stddev_idx], pareto_front[balanced_idx]]

    best_610 = get_best_solutions(pareto_610)
    best_955 = get_best_solutions(pareto_955)

    rows = []
    labels = ["Min Cost", "Min Stddev", "Balanced"]
    for i, sol in enumerate(best_610):
        rows.append(
            {
                "Candidate Set": "610",
                "Type": labels[i],
                "Total Cost": sol["Cost"][0],
                "Stddev (Equity)": sol["Cost"][1],
            }
        )
    for i, sol in enumerate(best_955):
        rows.append(
            {
                "Candidate Set": "955",
                "Type": labels[i],
                "Total Cost": sol["Cost"][0],
                "Stddev (Equity)": sol["Cost"][1],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"final_solution_comparison_{run_label}.csv"), index=False)
    df.to_csv(os.path.join(output_dir, f"final_solution_comparison_{run_label}.txt"), sep="\t", index=False)
    print(f"Saved final solution comparison to final_solution_comparison_{run_label}.csv and .txt")
    print(df.to_string(index=False))

def run_nsga_with_cache(args):
    data, label, candidate_610, candidate_955, cache_file, MaxIt, nPop, pCrossover, pMutation, patience, map_path = args
    if cache_exists(cache_file):
        return load_cache(cache_file)
    else:
        result = run_and_plot_nsga(data, label, candidate_610, candidate_955, return_front=True,
                                   MaxIt=MaxIt, nPop=nPop, pCrossover=pCrossover, pMutation=pMutation, patience=patience, map_path=map_path)
        save_cache(result, cache_file)
        return result
    

def main():
    # Get label from command line or set default
    if len(sys.argv) > 1:
        run_label = sys.argv[1]
    else:
        run_label = "default_run"

    # Prepare log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"output/console_log_{run_label}_{timestamp}.txt"
    ensure_output_dir()
    with open(log_file, "w") as f:
        f.write("NSGA-II Supply Chain Optimization Log\n")

    # Set NSGA-II parameters
    MaxIt = 10  # Recommended value from 150 to 300
    nPop = 5  # Recommended value from 50 to 100
    pCrossover = 0.7
    pMutation = 0.15
    patience = 20

    log_with_timestamp(f"NSGA-II Parameters: MaxIt={MaxIt}, nPop={nPop}, pCrossover={pCrossover}, pMutation={pMutation}, patience={patience}", log_file)

    try: 
        # Load demand data
        demand_df = pd.read_excel("data/demand_data.xlsx")
        Em = np.array(demand_df["demand"]).reshape(-1, 1)
        xy_community = demand_df[["x", "y"]].values

        # Load candidate locations
        candidate_610 = np.array(pd.read_excel("data/candidate_location_610.xlsx")[["x", "y"]])
        candidate_955 = np.array(pd.read_excel("data/candidate_location_955.xlsx")[["x", "y"]])
    except FileNotFoundError as e:
        log_with_timestamp(f"Error: {e}. Ensure all required data files are in the 'data/' directory.", log_file)
        sys.exit(1)
    except ValueError as e:
        log_with_timestamp(f"Error: {e}. Check the format of the input files.", log_file)
        sys.exit(1)

    log_with_timestamp("Successfully loaded demand and candidate location data.", log_file)

    # Initialize data for each candidate set
    data_610 = initialize_data("data/candidate_location_610.xlsx", Em, xy_community)
    data_955 = initialize_data("data/candidate_location_955.xlsx", Em, xy_community)

    log_with_timestamp("Initialized data for candidate sets.", log_file)

    # Fetch a realistic map if possible
    min_x, max_x, min_y, max_y = data_610["xy_candidate"][:, 0].min(), data_610["xy_candidate"][:, 0].max(), data_610[
        "xy_candidate"
    ][:, 1].min(), data_610["xy_candidate"][:, 1].max()
    try:
        ensure_output_dir()  # Ensure the output directory exists
        img = fetch_osm_static_map(min_x, max_x, min_y, max_y, width=800, height=600)
        if img is not None:
            img.save("output/static_map.png")
            map_path = "output/static_map.png"
            log_with_timestamp("Fetched and saved OSM static map.", log_file)
        else:
            log_with_timestamp("Warning: Failed to fetch the map. Proceeding without a map.", log_file)
            map_path = None
    except Exception as e:
        log_with_timestamp(f"Could not fetch OSM static map: {e}. The optimization will proceed without a map.", log_file)
        map_path = None

    # Plot all candidates with demand
    E_L = data_610["E_L"]  # or data_955['E_L'], they should be the same
    plot_all_candidates_with_demand(candidate_610, candidate_955, xy_community, E_L, map_path=map_path)
    log_with_timestamp("Plotted all candidates with demand.", log_file)

    # Create parameter hashes for cache filenames
    param_hash_610 = get_param_hash(data_610, MaxIt, nPop, pCrossover, pMutation, patience)
    param_hash_955 = get_param_hash(data_955, MaxIt, nPop, pCrossover, pMutation, patience)
    cache_file_610 = f"output/nsga_610_cache_{param_hash_610}_{run_label}.pkl"
    cache_file_955 = f"output/nsga_955_cache_{param_hash_955}_{run_label}.pkl"

    # Prepare arguments for parallel run
    args_610 = (data_610, f"nsga_610_{run_label}", candidate_610, candidate_955, cache_file_610, MaxIt, nPop, pCrossover, pMutation, patience, map_path)
    args_955 = (data_955, f"nsga_955_{run_label}", candidate_610, candidate_955, cache_file_955, MaxIt, nPop, pCrossover, pMutation, patience, map_path)
    args_list = [args_610, args_955]

    pareto_front_610 = None
    pareto_front_955 = None

    labels = [f"nsga_610_{run_label}", f"nsga_955_{run_label}"]
    monitor_thread = threading.Thread(target=monitor_progress, args=(labels, MaxIt))
    monitor_thread.daemon = True
    monitor_thread.start()

    log_with_timestamp("Started NSGA-II optimization.", log_file)

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_to_label = {}
        futures = []
        for args in args_list:
            future = executor.submit(run_nsga_with_cache, args)
            future_to_label[future] = args[1]  # args[1] is the label ("nsga_610" or "nsga_955")
            futures.append(future)
        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                label = future_to_label[f]
                if "610" in label:
                    pareto_front_610 = result
                else:
                    pareto_front_955 = result
                log_with_timestamp(f"Completed optimization for {label}.", log_file)
            except Exception as e:
                log_with_timestamp(f"Error in task {future_to_label[f]}: {e}", log_file)

    monitor_thread.join()
    if not pareto_front_610 or not pareto_front_955:
        log_with_timestamp("Error: One or both Pareto fronts are empty. Exiting.", log_file)
        sys.exit(1)

    plot_final_pareto_comparison(pareto_front_610, pareto_front_955, run_label)
    log_with_timestamp("Plotted final Pareto front comparison.", log_file)

    export_final_solution_comparison(pareto_front_610, pareto_front_955, run_label)
    log_with_timestamp("Exported final solution comparison.", log_file)


if __name__ == "__main__":
    main()