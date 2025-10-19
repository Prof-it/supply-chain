import numpy as np
import random
import multiprocessing
import time
from nsga_utils import log_with_timestamp, ensure_output_dir, Individual

# --- Objective Function ---
def objective_function_optimized(x, y, E, C, Cp, Cpp, d, gamma=1.0):
    """
    Compute the two-objective function for location optimization:
    - f1: total cost (construction + travel + penalty)
    - f2: weighted standard deviation of travel distances (equity)
    """
    # Ensure y is integer (rounding as in original implementation)
    y_int = np.round(y).astype(int)

    # Total cost
    f1 = np.sum(x * C.reshape(-1, 1)) + np.sum(Cp * (d * y)) + np.sum(Cpp * (gamma * y))

    # Weighted standard deviation (equity)
    total_y = np.sum(y_int)
    if total_y <= 1:
        f2 = 0.0
    else:
        dm = np.sum(y_int * d) / total_y
        f2 = np.sqrt(np.sum(y_int * (d - dm)**2) / (total_y - 1))

    return np.array([f1, f2])

# --- Generate Random Solution ---
def generate_random_solution(data):
    """
    Generate a random solution that satisfies the constraints.
    """
    n_candidate = data['n_candidate']
    n_community = data['n_community']
    Q = data['Q']
    E = data['E_L']
    U = data['U']

    # Ensure U is not greater than n_candidate
    if U > n_candidate:
        print(f"Warning: U ({U}) exceeds n_candidate ({n_candidate}). Adjusting U to {n_candidate}.")
        U = n_candidate

    # Randomly select facilities to open (respecting the budget limit)
    x = np.zeros((n_candidate, len(Q)))
    open_facilities = random.sample(range(n_candidate), U)
    for i in open_facilities:
        x[i, random.randint(0, len(Q) - 1)] = 1

    # Randomly assign demand to facilities
    y = np.zeros((n_candidate, n_community))
    for j in range(n_community):
        remaining_demand = E[j]
        attempts = 0  # Add a counter to prevent infinite loops
        max_attempts = 100  # Set a maximum number of attempts
        while remaining_demand > 0 and attempts < max_attempts:
            i = random.choice(open_facilities)
            capacity = np.sum(x[i, :] * Q) - np.sum(y[i, :])
            assign = min(remaining_demand, capacity)
            if assign <= 0:  # If no capacity is available, skip to the next iteration
                attempts += 1
                continue
            y[i, j] += assign.item()
            remaining_demand -= assign
        if remaining_demand > 0:
            print(f"Warning: Could not fully assign demand for community {j}. Remaining demand: {remaining_demand}")

    # Repair the solution to ensure constraints are satisfied
    x, y = repair_solution(x, y, E, Q, data['dpp'], data['D'], U, data['C'])
    cost = objective_function_optimized(x, y, E, data['C'], data['Cp'], data['Cpp'], data['d'], data['gamma'])
    return Individual(x=x, E=E, y=y, Cost=cost)  # Return an Individual object

# --- Constraint Validation ---
def validate_constraints(x, y, E, Q, dpp, D, U):
    """
    Validate all constraints for a solution.
    """
    #print("Debugging validate_constraints:")
    #print(f"x:\n{x}")
    #print(f"y:\n{y}")
    #print(f"E:\n{E}")
    #print(f"Q:\n{Q}")
    #print(f"dpp:\n{dpp}")
    #print(f"D: {D}")
    #print(f"U: {U}")

    # Equation 4: Demand Coverage
    for j in range(E.size):
        if not np.isclose(np.sum(y[:, j]), E[j]):
            print(f"Demand coverage failed for community {j}.")
            return False

    # Equation 5: Capacity Limit
    for i in range(x.shape[0]):
        if np.sum(y[i, :]) > np.sum(x[i, :] * Q[:x.shape[1]]):
            print(f"Capacity limit failed for facility {i}.")
            return False

    # Equation 6: Minimum Separation
    for i in range(x.shape[0]):
        for k in range(i + 1, x.shape[0]):
            if dpp[i, k] <= D and np.sum(x[i, :]) + np.sum(x[k, :]) > 1:
                print(f"Minimum separation failed between facilities {i} and {k}.")
                return False

    # Equation 7: Budget Limit
    if np.sum(x) > U:
        print(f"Budget limit failed. Total facilities: {np.sum(x)}, U: {U}.")
        return False

    return True

# --- Apply Penalty ---
def apply_penalty(x, y, E, Q, dpp, D, U, penalty_weight=1e6):
    """
    Apply penalties for constraint violations.
    """
    penalty = 0

    # Equation 4: Demand Coverage
    for j in range(E.size):
        if not np.isclose(np.sum(y[:, j]), E[j]):
            penalty += abs(np.sum(y[:, j]) - E[j]) * penalty_weight

    # Equation 5: Capacity Limit
    for i in range(x.shape[0]):
        if np.sum(y[i, :]) > np.sum(x[i, :] * Q):
            penalty += (np.sum(y[i, :]) - np.sum(x[i, :] * Q)) * penalty_weight

    # Equation 6: Minimum Separation
    for i in range(x.shape[0]):
        for k in range(i + 1, x.shape[0]):
            if dpp[i, k] <= D and np.sum(x[i, :]) + np.sum(x[k, :]) > 1:
                penalty += penalty_weight

    # Equation 7: Budget Limit
    if np.sum(x) > U:
        penalty += (np.sum(x) - U) * penalty_weight

    return penalty

# --- Repair Solution with Logging ---
def repair_solution(x, y, E, Q, dpp, D, U, C, log_file=None):
    """
    Repair a solution to ensure it satisfies all constraints.
    """
    log_with_timestamp("Repairing solution...", log_file)

    # Ensure x is binary
    x = np.round(x).astype(int)

    # Ensure y is non-negative and integer
    y = np.maximum(0, np.round(y).astype(int))

    # Ensure the budget limit (Equation 7)
    while np.sum(x) > U:
        open_facilities = np.where(np.any(x, axis=1))[0]
        # Close the facility with the highest cost
        costs = [np.sum(x[i, :] * C.reshape(-1, 1)) for i in open_facilities]
        to_close = open_facilities[np.argmax(costs)]
        x[to_close, :] = 0
        y[to_close, :] = 0
        log_with_timestamp(f"Closed facility {to_close} to meet budget limit.", log_file)

    # Ensure minimum separation (Equation 6)
    for i in range(x.shape[0]):
        if np.any(x[i, :]):
            too_close = np.where((dpp[i, :] <= D) & (np.sum(x, axis=1) > 0))[0]
            for k in too_close:
                if k != i:
                    x[k, :] = 0
                    y[k, :] = 0
                    log_with_timestamp(f"Closed facility {k} due to minimum separation constraint with facility {i}.", log_file)

    # Ensure capacity limits (Equation 5)
    for i in range(x.shape[0]):
        if np.sum(y[i, :]) > np.sum(x[i, :] * Q[:x.shape[1]]):
            y[i, :] = np.minimum(y[i, :], np.sum(x[i, :] * Q[:x.shape[1]]))
            log_with_timestamp(f"Adjusted assignments for facility {i} to meet capacity limits.", log_file)

    # Ensure demand coverage (Equation 4)
    for j in range(E.size):
        remaining_demand = E[j] - np.sum(y[:, j])
        if remaining_demand > 0:
            open_facilities = np.where(np.any(x, axis=1))[0]
            for i in open_facilities:
                assign = min(remaining_demand, np.sum(x[i, :] * Q[:x.shape[1]]) - np.sum(y[i, :]))
                y[i, j] += assign
                remaining_demand -= assign
                log_with_timestamp(f"Assigned {assign} demand to facility {i} for community {j}.", log_file)
                if remaining_demand <= 0:
                    break

    log_with_timestamp("Solution repair completed.", log_file)
    return x, y

# --- Final Validation of Pareto Front ---
def validate_pareto_front(pareto_front, data, log_file=None):
    """
    Validate all solutions in the Pareto front to ensure they satisfy constraints.
    Apply penalties to infeasible solutions instead of raising errors.
    """
    log_with_timestamp("Validating Pareto front solutions...", log_file)
    for ind in pareto_front:
        x, y = ind.x, ind.y
        if not validate_constraints(x, y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U']):
            log_with_timestamp(f"Solution {ind} is infeasible. Applying penalty...", log_file)
            penalty = apply_penalty(x, y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U'])
            ind.Cost[0] += penalty  # Add penalty to the first objective (total cost)
    log_with_timestamp("Pareto front validation completed.", log_file)


def dominates(p_cost, q_cost, p_feasible, q_feasible):
    if p_feasible and not q_feasible:
        return True
    if not p_feasible and q_feasible:
        return False
    return np.all(p_cost <= q_cost) and np.any(p_cost < q_cost)

def non_dominated_sorting(pop, data):
    """
    Perform non-dominated sorting on the population.
    Classifies solutions into Pareto fronts and assigns ranks.
    """
    nPop = len(pop)
    for i in range(nPop):
        pop[i].DominationSet = []
        pop[i].DominatedCount = 0

    F = [[]]  # Fronts (F[0] is the first Pareto front)

    # Compare each solution with every other solution
    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]

            # Check feasibility of solutions
            p_feasible = validate_constraints(p.x, p.y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U'])
            q_feasible = validate_constraints(q.x, q.y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U'])

            if dominates(p.Cost, q.Cost, p_feasible, q_feasible):
                p.DominationSet.append(j)
                q.DominatedCount += 1
            elif dominates(q.Cost, p.Cost, q_feasible, p_feasible):
                q.DominationSet.append(i)
                p.DominatedCount += 1

        # If no solution dominates `p`, it belongs to the first front
        if pop[i].DominatedCount == 0:
            F[0].append(i)
            pop[i].Rank = 1

    # Generate subsequent fronts
    k = 0
    while k < len(F) and len(F[k]) > 0:
        Q = []  # Next front
        for i in F[k]:
            p = pop[i]
            for j_idx in p.DominationSet:
                q = pop[j_idx]
                q.DominatedCount -= 1
                if q.DominatedCount == 0:
                    Q.append(j_idx)
                    q.Rank = k + 2
        k += 1
        if len(Q) > 0:
            F.append(Q)

    # Convert list of indices to list of individuals
    F_individuals = [[pop[i] for i in front_indices] for front_indices in F]
    return pop, F_individuals

# --- Crossover ---
def crossover_worker(args):
    """
    Perform crossover between two parent solutions.
    """
    parent1, parent2, data = args
    n_candidate = data['n_candidate']
    n_community = data['n_community']

    # Ensure parent1 is valid
    if parent1 is None or not isinstance(parent1, Individual) or parent1.x is None:
        print("Warning: parent1 is invalid. Replacing with a random solution.")
        parent1 = generate_random_solution(data)

    # Ensure parent2 is valid
    if parent2 is None or not isinstance(parent2, Individual) or parent2.x is None:
        print("Warning: parent2 is invalid. Replacing with a random solution.")
        parent2 = generate_random_solution(data)

    # Uniform crossover for x
    if parent1.x is not None and parent2.x is not None and parent1.y is not None and parent2.y is not None:
        child_x = np.zeros_like(parent1.x)
        for i in range(n_candidate):
            if random.random() < 0.5:
                child_x[i, :] = parent1.x[i, :]
            else:
                child_x[i, :] = parent2.x[i, :]

        # Uniform crossover for y
        child_y = np.zeros_like(parent1.y)
        for j in range(n_community):
            if random.random() < 0.5:
                child_y[:, j] = parent1.y[:, j]
            else:
                child_y[:, j] = parent2.y[:, j]

        # Repair the solution to ensure constraints are satisfied
        child_x, child_y = repair_solution(child_x, child_y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U'], data['C'])
    else:
        child_x = generate_random_solution(data).x
        child_y = generate_random_solution(data).y 

    return Individual(x=child_x, y=child_y, Cost=objective_function_optimized(child_x, child_y, data['E_L'], data['C'], data['Cp'], data['Cpp'], data['d'], data['gamma']))
# --- Mutation ---
def mutation_worker(args):
    """
    Perform mutation on a solution.
    """
    parent, data = args
    n_candidate = data['n_candidate']
    n_community = data['n_community']
    pMutation = data.get('pMutation', 0.1)

    # Mutate x (open/close facilities)
    child_x = parent.x.copy()  # Create a copy of the parent's x
    for i in range(n_candidate):
        if random.random() < pMutation:  # Mutation probability
            child_x[i, :] = 0 if np.any(child_x[i, :]) else random.choice([1, 0])

    # Mutate y (reassign demand)
    child_y = parent.y.copy()  # Create a copy of the parent's y
    for j in range(n_community):
        if random.random() < pMutation:  # Mutation probability
            child_y[:, j] = 0  # Clear current demand assignments
            remaining_demand = data['E_L'][j]

            # Get open facilities and their available capacities
            open_facilities = np.where(np.any(child_x, axis=1))[0]
            if len(open_facilities) == 0:
                # Skip mutation if no facilities are open
                log_with_timestamp(f"Warning: No open facilities available for community {j}. Skipping mutation.", data.get('log_file'))
                continue

            capacities = np.array([np.sum(child_x[i, :] * data['Q']) - np.sum(child_y[i, :]) for i in open_facilities])
            costs = np.array([np.sum(child_x[i, :] * data['C']) for i in open_facilities])

            if len(costs) == 0 or len(capacities) == 0:
                log_with_timestamp(f"Warning: Empty costs or capacities for community {j}. Skipping mutation.", data.get('log_file'))
                continue

            # Normalize costs and capacities
            norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-8)
            norm_capacities = (capacities - capacities.min()) / (capacities.max() - capacities.min() + 1e-8)

            # Calculate a weighted score (e.g., prioritize capacity over cost)
            scores = 0.7 * norm_capacities - 0.3 * norm_costs  # Higher capacity and lower cost are better

            # Assign demand to facilities in order of scores
            for i in open_facilities[np.argsort(-scores)]:  # Sort by descending score
                if remaining_demand <= 0:
                    break
                assign = min(remaining_demand, capacities[i])
                child_y[i, j] += assign
                remaining_demand -= assign

            # Fallback: Log a warning if demand cannot be fully reassigned
            if remaining_demand > 0:
                log_with_timestamp(f"Warning: Could not reassign {remaining_demand} demand for community {j}.", data.get('log_file'))

    # Repair the solution to ensure constraints are satisfied
    child_x, child_y = repair_solution(child_x, child_y, data['E_L'], data['Q'], data['dpp'], data['D'], data['U'], data['C'])

    # Return the mutated solution as an Individual object
    return Individual(x=child_x, y=child_y, Cost=objective_function_optimized(child_x, child_y, data['E_L'], data['C'], data['Cp'], data['Cpp'], data['d'], data['gamma']))


# --- Crowding Distance ---
def calculate_crowding_distance(pop, fronts):
    """
    Calculate the crowding distance for each individual in the population.
    """
    for front in fronts:
        if len(front) < 2:
            for ind in front:
                ind.CrowdingDistance = float('inf')
            continue

        n_objectives = len(front[0].Cost)
        for m in range(n_objectives):
            front.sort(key=lambda x: x.Cost[m])
            front[0].CrowdingDistance = float('inf')
            front[-1].CrowdingDistance = float('inf')
            for i in range(1, len(front) - 1):
                front[i].CrowdingDistance += (front[i + 1].Cost[m] - front[i - 1].Cost[m])

    return pop

# --- NSGA-II Optimization ---
def nsga_ii_optimization_with_label(data, label, MaxIt, nPop, pCrossover, pMutation, patience, log_file=None):
    """
    NSGA-II optimization loop with constraint handling and logging.
    """
    # Initialization
    nCrossover = int(round(pCrossover * nPop / 2) * 2)
    nMutation = int(round(pMutation * nPop))
    pop = [generate_random_solution(data) for _ in range(nPop)]
    all_fronts = []
    no_improve_count = 0
    best_f1 = float('inf')
    best_f2 = float('inf')
    new_pop = []  # Ensure new_pop is always defined
    start_time = time.time()
    for it in range(MaxIt):
        log_with_timestamp(f"Iteration {it+1}/{MaxIt} started.", log_file)
        pop, F = non_dominated_sorting(pop, data)
        pop = calculate_crowding_distance(pop, F)

        # Crossover and mutation
        with multiprocessing.Pool(processes=4) as pool:
            crossover_args = [(pop[random.randint(0, nPop - 1)], pop[random.randint(0, nPop - 1)], data) for _ in range(nCrossover)]
            popc = pool.map(crossover_worker, crossover_args)
            mutation_args = [(pop[random.randint(0, nPop - 1)], data) for _ in range(nMutation)]
            popm = pool.map(mutation_worker, mutation_args)

        # Combine populations
        pop_combined = pop + popc + popm
        pop_combined, F_combined = non_dominated_sorting(pop_combined, data)
        pop_combined = calculate_crowding_distance(pop_combined, F_combined)

        # Select next generation

        k = 0
        while k < len(F_combined) and len(new_pop) + len(F_combined[k]) <= nPop:
            new_pop.extend(F_combined[k])
            k += 1
        if len(new_pop) < nPop:
            F_k = F_combined[k]
            F_k.sort(key=lambda x: x.CrowdingDistance, reverse=True)
            new_pop.extend(F_k[:nPop - len(new_pop)])

        # Early stopping check
        costs = np.array([ind.Cost for ind in new_pop])
        current_best_f1 = np.min(costs[:, 0])
        current_best_f2 = np.min(costs[:, 1])
        if current_best_f1 < best_f1 or current_best_f2 < best_f2:
            best_f1 = min(best_f1, current_best_f1)
            best_f2 = min(best_f2, current_best_f2)
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            log_with_timestamp(f"Early stopping: No improvement for {patience} generations.", log_file)
            break
        # Validate the final Pareto front
        validate_pareto_front(new_pop, data, log_file)

        # Fallback if new_pop is empty
        if not new_pop:
            log_with_timestamp("Warning: new_pop is empty. Using best solutions from pop_combined.", log_file)
            new_pop = sorted(pop_combined, key=lambda x: x['Cost'][0])[:nPop]
        pop = new_pop

    end_time = time.time()
    log_with_timestamp(f"NSGA-II optimization completed in {end_time - start_time:.2f} seconds.", log_file)

    

    return new_pop, data, all_fronts