import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Global Data Setup (Converted from initial_data.m, demand_data.m, coordinate_data.m) ---

# ...existing code...

def initialize_data(candidate_file):
    # Parameters from initial_data.m
    n_community = 27
    Q = np.array([30, 40, 50])  # Capacity options
    C = np.array([45, 65, 80])  # Construction costs
    Cp = 20  # Travel cost per unit demand per unit distance
    Cpp = 50 # Penalty cost
    alpha = 1
    beta = 1
    U = 25   # Fixed number of facilities to open
    D = 200  # Minimum distance constraint between facilities
    landa = 0.5 # Fuzzy parameter

    # Load demand data (communities) from Excel
    demand_df = pd.read_excel("data/demand_data.xlsx")
    Em = np.array(demand_df["demand"]).reshape(-1, 1)  # Assuming column is named 'demand'

    Ep = 0.6 * Em
    Eo = 1.4 * Em
    E_L = (landa / 2 * (Em + Eo) / 2) + ((1 - landa / 2) * (Ep + Em) / 2)
    E_U = ((1 - landa / 2) * (Em + Eo) / 2) + (landa / 2 * (Ep + Em) / 2)

    # Load candidate locations from Excel
    candidate_df = pd.read_excel(candidate_file)
    xy_candidate = candidate_df[["x", "y"]].values
    n_candidate = xy_candidate.shape[0]

    # Load community locations from demand_data.xlsx (assuming columns 'x', 'y')
    xy_community = demand_df[["x", "y"]].values

    # Calculate dpp (distance between candidates)
    dpp = np.linalg.norm(xy_candidate[:, None, :] - xy_candidate[None, :, :], axis=2)

    # Calculate d (distance between candidates and communities)
    d = np.linalg.norm(xy_candidate[:, None, :] - xy_community[None, :, :], axis=2)

    gama = np.zeros((n_candidate, n_community)) # Placeholder value; must be defined by user data

    data = {
        'n_candidate': n_candidate, 'n_community': n_community, 'Q': Q, 'C': C, 'Cp': Cp, 'Cpp': Cpp,
        'alpha': alpha, 'beta': beta, 'U': U, 'D': D, 'landa': landa, 'E_L': E_L, 'E_U': E_U,
        'dpp': dpp, 'd': d, 'gama': gama, 'xy_candidate': xy_candidate, 'xy_community': xy_community
    }
    return data

class Individual:
    def __init__(self, x=None, E=None, y=None, Cost=None, Rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None):
        self.x = x # Location/Type matrix (n_candidate x n_types)
        self.E = E # Sampled Fuzzy Demand (n_community x 1)
        self.y = y # Allocation matrix (n_candidate x n_community)
        self.Cost = Cost # [f1, f2]
        self.Rank = Rank
        self.DominationSet = DominationSet if DominationSet is not None else []
        self.DominatedCount = DominatedCount if DominatedCount is not None else 0
        self.CrowdingDistance = CrowdingDistance
        
# --- Allocation Functions ---

def random_allocation(x, E, Q, d):
    # This is the original, non-deterministic allocation used for benchmark only.
    n_candidate, n_types = x.shape
    n_community, _ = E.shape
    y = np.zeros((n_candidate, n_community))
    remaining_E = E.copy().flatten()

    cap = np.zeros(n_candidate)
    for i in range(n_candidate):
        if np.any(x[i,:]):
            cap[i] = Q[np.argmax(x[i,:])]

    for j in range(n_community):
        comm_demand = remaining_E[j]
        while comm_demand > 0:
            available = np.where(cap > 0)[0]
            if len(available) == 0:
                break
            
            # Randomly pick an available center
            i = np.random.choice(available)
            assign = min(comm_demand, cap[i])
            
            y[i, j] += assign
            cap[i] -= assign
            comm_demand -= assign
            
    return y

def greedy_allocation(x, E, Q, d):
    # This is the corrected, deterministic, distance-minimizing allocation for NSGA-II fitness.
    n_candidate, n_types = x.shape
    n_community, _ = E.shape
    y = np.zeros((n_candidate, n_community))
    comm_demand = E.copy().flatten()

    # Pre-calculate capacity for open facilities
    cap = np.zeros(n_candidate)
    open_facilities = []
    for i in range(n_candidate):
        if np.any(x[i,:]):
            center_type = np.argmax(x[i,:])
            cap[i] = Q[center_type]
            open_facilities.append(i)

    # For each community, allocate to the closest open facility with capacity
    for j in range(n_community):
        current_demand = comm_demand[j]
        
        # Sort open facilities by distance to community j
        distances = d[:, j]
        sorted_indices = np.argsort(distances)
        
        for i in sorted_indices:
            if i in open_facilities: # Only consider open facilities
                assign = min(current_demand, cap[i])
                if assign > 0:
                    y[i, j] += assign
                    cap[i] -= assign
                    current_demand -= assign
                    if current_demand <= 0:
                        break
        
        # If demand remains unassigned (should not happen if total capacity > total demand)
        # We do not penalize here, penalty is handled in OF (gamma cost)
        
    return y

# --- Objective Function ---

def objective_function(x, y, E, C, Cp, Cpp, d, gama):
    n_candidate, n_types = x.shape
    
    # Objective 1: Minimize Total Cost (f1)
    cost_build = np.sum(x * C)
    cost_travel = np.sum(Cp * (d * y))
    
    # Penalty cost: assumes gama is a matrix of penalties for unassigned or poorly assigned demand
    # The MATLAB OF.m uses 'y' in the penalty, which suggests penalty is based on *assigned* volume,
    # or gama represents an issue with the assignment itself (e.g., exceeding a distance threshold)
    # Assuming gama is a cost modifier for assignment:
    cost_penalty = np.sum(Cpp * (gama * y)) 
    
    f1 = cost_build + cost_travel + cost_penalty
    
    # Objective 2: Maximize Equity (Minimize Standard Deviation of Assigned Distances) (f2)
    distances_flat = []
    for i in range(n_candidate):
        for j in range(E.size):
            # Repmat equivalent: repeat the distance d(i,j) by the volume y(i,j) allocated
            if y[i, j] > 0:
                distances_flat.extend([d[i, j]] * int(round(y[i, j])))
                
    if len(distances_flat) > 1:
        f2 = np.std(distances_flat)
    else:
        f2 = 0
        
    return np.array([f1, f2])

# --- NSGA-II Core Functions ---

def dominates(p_cost, q_cost):
    # Returns True if solution p dominates solution q
    return np.all(p_cost <= q_cost) and np.any(p_cost < q_cost)

def non_dominated_sorting(pop):
    nPop = len(pop)
    
    for i in range(nPop):
        pop[i].DominationSet = []
        pop[i].DominatedCount = 0

    F = [[]] # Fronts (F[0] is the first front F1)
    
    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]
            
            if dominates(p.Cost, q.Cost):
                p.DominationSet.append(j)
                q.DominatedCount += 1
            elif dominates(q.Cost, p.Cost):
                q.DominationSet.append(i)
                p.DominatedCount += 1
                
            pop[i] = p
            pop[j] = q

        if pop[i].DominatedCount == 0:
            F[0].append(i)
            pop[i].Rank = 1
    
    k = 0
    while k < len(F) and len(F[k]) > 0:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j_idx in p.DominationSet:
                q = pop[j_idx]
                q.DominatedCount -= 1
                if q.DominatedCount == 0:
                    Q.append(j_idx)
                    q.Rank = k + 2
                pop[j_idx] = q
        k += 1
        if len(Q) > 0:
            F.append(Q)
            
    # Convert list of indices to list of Individuals
    F_individuals = [[pop[i] for i in front_indices] for front_indices in F]
    return pop, F_individuals

def calculate_crowding_distance(pop, F_individuals):
    nObj = pop[0].Cost.size # 2 objectives
    
    for k, Fk in enumerate(F_individuals):
        if not Fk:
            continue
        
        n = len(Fk)
        if n <= 2:
            for ind in Fk:
                ind.CrowdingDistance = float('inf')
            continue

        Costs = np.array([ind.Cost for ind in Fk]).T
        d = np.zeros((n, nObj)) # Crowding distance matrix
        
        for j in range(nObj): # Loop over objectives (j=0 for f1, j=1 for f2)
            # Sort by objective j
            order = np.argsort(Costs[j, :])
            cj_sorted = Costs[j, order]
            
            # Boundary solutions get infinite distance
            d[order[0], j] = float('inf')
            d[order[-1], j] = float('inf')
            
            # Calculate distance for interior solutions
            obj_range = cj_sorted[-1] - cj_sorted[0]
            if obj_range == 0:
                 # If all points are the same, distance is inf
                for i in range(1, n - 1):
                    d[order[i], j] = float('inf')
            else:
                for i in range(1, n - 1):
                    # Normalized distance: |f(i+1) - f(i-1)| / Range
                    d[order[i], j] = abs(cj_sorted[i+1] - cj_sorted[i-1]) / obj_range

        # Update the CrowdingDistance in the population objects
        for i in range(n):
            Fk[i].CrowdingDistance = np.sum(d[i, :])
            
    return pop

def sort_population(pop):
    # Sort first by Rank (ascending), then by CrowdingDistance (descending)
    
    # 1. Sort by CrowdingDistance (descending)
    pop.sort(key=lambda x: x.CrowdingDistance, reverse=True)
    # 2. Sort by Rank (ascending) - Python's stable sort ensures CrowdingDistance order is preserved within the same Rank
    pop.sort(key=lambda x: x.Rank)
    
    return pop

# --- Genetic Operators ---

def generate_random_solution(data):
    n_candidate = data['n_candidate']
    n_community = data['n_community']
    n_types = len(data['Q'])
    U = data['U']
    D = data['D']
    dpp = data['dpp']
    E_L = data['E_L']
    E_U = data['E_U']
    
    # 1. Select U locations respecting the D constraint
    x = np.zeros((n_candidate, n_types))
    selected = random.sample(range(n_candidate), n_candidate)
    count = 0
    
    for loc in selected:
        if count >= U: break
        
        # Check D constraint against already selected locations
        too_close = False
        for j in range(n_candidate):
            if np.any(x[j, :]) and dpp[loc, j] < D:
                too_close = True
                break
        
        if not too_close:
            t = random.randint(0, n_types - 1)
            x[loc, t] = 1
            count += 1
            
    # If the loop finished before U facilities were selected, it means the D constraint is too tight.
    # In a real run, this should be handled better, but here we enforce U by filling up randomly if possible.
    if count < U:
         # Simplified handling: if constraint is too strict, just break and proceed with current count.
         # For this template, we assume D is loose enough.
         pass


    # 2. Generate fuzzy demand (E)
    E = (E_U - E_L) * np.random.rand(n_community, 1) + E_L
    E = np.floor(E) # Demand is integer

    # 3. Allocation (using the CORRECT greedy allocation)
    y = greedy_allocation(x, E, data['Q'], data['d'])
    
    # 4. Calculate cost
    cost = objective_function(x, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])
    
    return Individual(x=x, E=E, y=y, Cost=cost)

def crossover(p1, p2, data):
    n_candidate = data['n_candidate']
    n_types = len(data['Q'])
    U = data['U']
    D = data['D']
    dpp = data['dpp']
    
    # Start with the intersection (minimum) of parents' facility locations
    x = np.minimum(p1.x, p2.x)
    total_built = np.sum(x)
    
    # Find locations not yet assigned    
    remaining_indices = np.where(np.sum(x, axis=1) == 0)[0]
    remaining_indices = list(remaining_indices)  # Convert to list for shuffling
    random.shuffle(remaining_indices)


    attempt = 0
    while total_built < U and len(remaining_indices) > 0 and attempt < 500:
        loc = remaining_indices[0] # Pick the first available location
        
        # Check D constraint
        conflict = False
        for j in range(n_candidate):
            if np.any(x[j,:]) and dpp[loc, j] < D:
                conflict = True
                break
        
        if not conflict:
            t = random.randint(0, n_types - 1)
            x[loc, t] = 1
            total_built += 1
            
        remaining_indices = remaining_indices[1:] # Remove the location whether successful or not
        attempt += 1

    # Simple implementation returns only one offspring
    return Individual(x=x)

def mutate(p_x, data):
    n_candidate, n_types = p_x.shape
    D = data['D']
    dpp = data['dpp']
    
    x = p_x.copy()
    built_idx = np.where(np.any(x, axis=1))[0]
    empty_idx = np.where(~np.any(x, axis=1))[0]
    
    if len(built_idx) == 0 or len(empty_idx) == 0:
        return x
    
    attempt = 0
    while attempt < 500:
        i_old = random.choice(built_idx)
        i_new = random.choice(empty_idx)
        
        # Check D constraint for the new facility i_new
        conflict = False
        for j in built_idx:
            if j != i_old and dpp[i_new, j] < D:
                conflict = True
                break
        
        if not conflict:
            # Successful swap
            x[i_old, :] = 0 # Remove old facility
            t = random.randint(0, n_types - 1)
            x[i_new, t] = 1 # Add new facility with random type
            return x
        
        attempt += 1
        
    return p_x # Return unchanged if mutation failed to find a valid swap

# --- Main NSGA-II Loop ---

def nsga_ii_optimization_with_label(data, label):
    MaxIt = 10
    nPop = 10
    pCrossover = 0.7
    pMutation = 0.15
    nCrossover = int(round(pCrossover * nPop / 2) * 2)
    nMutation = int(round(pMutation * nPop))
    pop = [generate_random_solution(data) for _ in range(nPop)]
    all_fronts = []  # <-- Store Pareto fronts here
    for it in range(MaxIt):
        print(f"Iteration {it+1}/{MaxIt}")
        pop, F = non_dominated_sorting(pop)
        pop = calculate_crowding_distance(pop, F)
        popc = []
        while len(popc) < nCrossover:
            i1, i2 = random.sample(range(nPop), 2)
            p1 = pop[i1]
            p2 = pop[i2]
            x_new = crossover(p1, p2, data).x
            E = (data['E_U'] - data['E_L']) * np.random.rand(data['n_community'], 1) + data['E_L']
            E = np.floor(E)
            y = greedy_allocation(x_new, E, data['Q'], data['d'])
            cost = objective_function(x_new, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])
            popc.append(Individual(x=x_new, E=E, y=y, Cost=cost))
        popm = []
        for _ in range(nMutation):
            i1 = random.randint(0, nPop - 1)
            p = pop[i1]
            x_new = mutate(p.x, data)
            E = (data['E_U'] - data['E_L']) * np.random.rand(data['n_community'], 1) + data['E_L']
            E = np.floor(E)
            y = greedy_allocation(x_new, E, data['Q'], data['d'])
            cost = objective_function(x_new, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])
            popm.append(Individual(x=x_new, E=E, y=y, Cost=cost))
        pop_combined = pop + popc + popm
        pop_combined, F_combined = non_dominated_sorting(pop_combined)
        pop_combined = calculate_crowding_distance(pop_combined, F_combined)
        new_pop = []
        k = 0
        while len(new_pop) + len(F_combined[k]) <= nPop:
            new_pop.extend(F_combined[k])
            k += 1
        if len(new_pop) < nPop:
            F_k = F_combined[k]
            F_k.sort(key=lambda x: x.CrowdingDistance, reverse=True)
            remaining_slots = nPop - len(new_pop)
            new_pop.extend(F_k[:remaining_slots])
        plot_pareto_fronts_with_parents(F_combined, new_pop, it+1, label)
        all_fronts.append(F_combined[0])  # <-- Store the best Pareto front for this iteration
        pop = new_pop
    pop, F_final = non_dominated_sorting(pop)
    pop = calculate_crowding_distance(pop, F_final)
    return F_final[0], data, all_fronts
'''
def nsga_ii_optimization(data):
    # Parameters
    MaxIt = 50 # Using the optimal value from Table 12 it should be 300
    nPop = 10   # Using the optimal value from Table 12 it should be 50
    pCrossover = 0.7
    pMutation = 0.15
    
    nCrossover = int(round(pCrossover * nPop / 2) * 2)
    nMutation = int(round(pMutation * nPop))
    
    # Initial Population
    pop = [generate_random_solution(data) for _ in range(nPop)]
    all_fronts = []  # <-- Store Pareto fronts here
    # Main Loop
    for it in range(MaxIt):
        print(f"Iteration {it+1}/{MaxIt}")
        
        # Non-Dominated Sorting and Crowding Distance Calculation for Current Population
        pop, F = non_dominated_sorting(pop)
        pop = calculate_crowding_distance(pop, F)
        
        # Crossover
        popc = []
        while len(popc) < nCrossover:
            i1, i2 = random.sample(range(nPop), 2)
            p1 = pop[i1]
            p2 = pop[i2]
            
            x_new = crossover(p1, p2, data).x
            
            # Recalculate fitness for the new solution
            E = (data['E_U'] - data['E_L']) * np.random.rand(data['n_community'], 1) + data['E_L']
            E = np.floor(E)
            y = greedy_allocation(x_new, E, data['Q'], data['d'])
            cost = objective_function(x_new, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])
            
            popc.append(Individual(x=x_new, E=E, y=y, Cost=cost))
            
        # Mutation
        popm = []
        for _ in range(nMutation):
            i1 = random.randint(0, nPop - 1)
            p = pop[i1]
            
            x_new = mutate(p.x, data)
            
            # Recalculate fitness for the new solution
            E = (data['E_U'] - data['E_L']) * np.random.rand(data['n_community'], 1) + data['E_L']
            E = np.floor(E)
            y = greedy_allocation(x_new, E, data['Q'], data['d'])
            cost = objective_function(x_new, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])
            
            popm.append(Individual(x=x_new, E=E, y=y, Cost=cost))
            
        # Merge Parent and Offspring Populations (R_t = P_t U Q_t)
        pop_combined = pop + popc + popm
        
        # Non-Dominated Sorting and Crowding Distance Calculation
        pop_combined, F_combined = non_dominated_sorting(pop_combined)
        pop_combined = calculate_crowding_distance(pop_combined, F_combined)
        
        # Selection (P_{t+1})
        new_pop = []
        k = 0 # Front index
        while len(new_pop) + len(F_combined[k]) <= nPop:
            new_pop.extend(F_combined[k])
            k += 1
            
        # The last front F_k to be included may cause the population size to exceed nPop.
        # We must select the best from F_k based on Crowding Distance (descending)
        if len(new_pop) < nPop:
            F_k = F_combined[k]
            # Sort F_k by Crowding Distance (descending)
            F_k.sort(key=lambda x: x.CrowdingDistance, reverse=True)
            
            # Fill the remaining slots
            remaining_slots = nPop - len(new_pop)
            new_pop.extend(F_k[:remaining_slots])
            
        # --- Added this line to plot fronts and parents ---
        plot_pareto_fronts_with_parents(F_combined, new_pop, it+1)
        all_fronts.append(F_combined[0])  # <-- Store the best Pareto front for this iteration
        pop = new_pop # P_{t+1}
        
    # Final non-dominated sort to get the Pareto front
    pop, F_final = non_dominated_sorting(pop)
    pop = calculate_crowding_distance(pop, F_final)
    
    return F_final[0], data, all_fronts  # <-- Return all_fronts for evolution plot
'''


# --- Plotting Functions ---

def ensure_output_dir():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

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
    plt.title(f"Pareto Fronts and Selected Parents (Iteration {iteration})")
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

def plot_spatial_solution(best_solution, data, label="", candidate_610=None, candidate_955=None):
    output_dir = ensure_output_dir()
    xy_candidate = data['xy_candidate']
    xy_community = data['xy_community']
    Q = data['Q']
    built_idx = np.where(np.any(best_solution.x, axis=1))[0]
    facility_coords = xy_candidate[built_idx, :]
    facility_types = np.argmax(best_solution.x[built_idx, :], axis=1)
    facility_capacities = Q[facility_types]
    type_colors = ['#337AFF', '#FF9900', '#33CC33']  # blue, orange, green

    plt.figure(figsize=(10, 8))

    # Plot all candidate locations as background crosses
    if candidate_610 is not None:
        plt.scatter(candidate_610[:, 0], candidate_610[:, 1], 
                    marker='x', s=30, color='yellow', alpha=0.3, label='610 Candidates')
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
    Older fronts are lighter, newer fronts are darker.
    """
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    n_iters = len(all_fronts)
    base_color = np.array([0.1, 0.5, 0.8])  # blueish
    for i, front in enumerate(all_fronts):
        if not front:
            continue
        costs = np.array([ind.Cost for ind in front])
        # Fade color for earlier iterations
        alpha = 0.2 + 0.8 * (i + 1) / n_iters
        color = base_color * alpha + (1 - alpha)
        plt.scatter(costs[:, 0], costs[:, 1], color=color, alpha=alpha, s=40, label=f'Iter {i+1}' if i in [0, n_iters-1] else None)
        # If you want to connect nearest neighbors within each front, you can use scipy.spatial for Delaunay or MST, but it's not standard for Pareto plots.
    plt.title(f"Pareto Front Evolution ({label})")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'pareto_evolution_{label}.png'))
    plt.close()

def plot_final_pareto_comparison(pareto_610, pareto_955):
    """
    Plot the final Pareto fronts of 610 and 955 candidates on one figure for comparison.
    """
    output_dir = ensure_output_dir()
    plt.figure(figsize=(8, 6))
    costs_610 = np.array([ind.Cost for ind in pareto_610])
    costs_955 = np.array([ind.Cost for ind in pareto_955])
    plt.scatter(costs_610[:, 0], costs_610[:, 1], color='orange', marker='o', s=50, label='610 Candidates')
    plt.scatter(costs_955[:, 0], costs_955[:, 1], color='red', marker='s', s=50, label='955 Candidates')
    plt.title("Final Pareto Front Comparison (610 vs 955 Candidates)")
    plt.xlabel("Objective 1: Total Cost ($f_1$)")
    plt.ylabel("Objective 2: Standard Deviation ($f_2$, Equity)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pareto_final_comparison.png'))
    plt.close()

# --- Evaluation ---

def baseline_solution(data):
    # Generate a baseline solution using random allocation and minimum cost
    n_candidate = data['n_candidate']
    n_community = data['n_community']
    n_types = len(data['Q'])
    U = data['U']
    D = data['D']
    dpp = data['dpp']
    E_L = data['E_L']
    E_U = data['E_U']

    # Select U locations randomly, ignoring D constraint for baseline
    x = np.zeros((n_candidate, n_types))
    selected = random.sample(range(n_candidate), U)
    for loc in selected:
        t = random.randint(0, n_types - 1)
        x[loc, t] = 1

    # Use average fuzzy demand for baseline
    E = np.floor((E_U + E_L) / 2)

    # Allocation using random allocation
    y = random_allocation(x, E, data['Q'], data['d'])

    # Calculate cost
    cost = objective_function(x, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gama'])

    return Individual(x=x, E=E, y=y, Cost=cost)

def run_and_plot_baseline(data, label):
    print(f"\n--- Running Baseline ({label}) ---")
    sol = baseline_solution(data)
    plot_spatial_solution(sol, data, label)
    print(f"Saved Baseline spatial plot for {label} to spatial_solution_{label}.png")
    if sol.Cost is not None:
        print(f"Baseline cost: {sol.Cost[0]:.2f}, equity stddev: {sol.Cost[1]:.2f}")
    else:
        print("Baseline cost or equity stddev is not available (Cost is None).")
    return sol

def run_and_plot_nsga(data, label):
    print(f"\n--- Running NSGA-II ({label}) ---")
    pareto_front, data, all_fronts = nsga_ii_optimization_with_label(data, label)
    plot_pareto_front(pareto_front, label)
    plot_pareto_evolution(all_fronts, label)
    costs = np.array([ind.Cost for ind in pareto_front])
    min_cost_idx = np.argmin(costs[:, 0])
    best_solution = pareto_front[min_cost_idx]
    plot_spatial_solution(best_solution, data, label)
    print(f"Saved NSGA-II spatial plot for {label} to spatial_solution_{label}.png")
    print(f"NSGA-II best cost: {best_solution.Cost[0]:.2f}, equity stddev: {best_solution.Cost[1]:.2f}")
    return best_solution


def main():
    data_610 = initialize_data("data/candidate_location_610.xlsx")
    data_955 = initialize_data("data/candidate_location_955.xlsx")
    run_and_plot_baseline(data_610, "baseline_610")
    run_and_plot_nsga(data_610, "nsga_610")
    run_and_plot_baseline(data_955, "baseline_955")
    run_and_plot_nsga(data_955, "nsga_955")

if __name__ == "__main__":
    main()