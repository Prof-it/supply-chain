import os
import random
import time
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Individual:
    def __init__(self, x=None, E=None, y=None, Cost=None, Rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None):
        self.x = x # Location/Type matrix (n_candidate x n_types)
        self.E = E # Sampled Fuzzy Demand (n_community x 1)
        self.y = y # Allocation matrix (n_candidate x n_community)
        self.Cost = Cost # [f1, f2]
        self.Rank = Rank
        self.DominationSet = DominationSet if DominationSet is not None else []
        self.DominatedCount = DominatedCount if DominatedCount is not None else 0
        self.CrowdingDistance = CrowdingDistance if CrowdingDistance is not None else 0


# --- Logging and monitoring Function ---
def log_with_timestamp(message, log_file=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)  # Always print to the console for debugging
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")
def monitor_progress(labels, MaxIt, interval=2):
    last_progress = {label: -1 for label in labels}
    while True:
        bars = []
        all_done = True
        changed = False
        for label in labels:
            progress_file = f"output/nsga_progress_{label}.txt"
            try:
                with open(progress_file, "r") as f:
                    it = int(f.read().strip())
            except Exception:
                it = 0
            bar = f"{label}: [{('#' * it).ljust(MaxIt)}] {it}/{MaxIt}"
            bars.append(bar)
            if it < MaxIt:
                all_done = False
            if last_progress[label] != it:
                changed = True
                last_progress[label] = it
        if changed:
            print("\r" + " | ".join(bars), end="", flush=True)
        if all_done:
            break
        time.sleep(interval)
    print()  # Newline after done

# --- Caching Functions ---
def cache_exists(cache_file):
    """
    Check if a cache file exists.
    """
    return os.path.exists(cache_file)

def save_cache(obj, cache_file):
    """
    Save an object to a cache file using pickle.
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(obj, f)

def load_cache(cache_file):
    """
    Load an object from a cache file using pickle.
    """
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def get_param_hash(data, MaxIt, nPop, pCrossover, pMutation, patience):
    """
    Generate a hash for the parameters to use as a cache key.
    """
    param_keys = ['n_candidate', 'n_community', 'Q', 'C', 'Cp', 'Cpp', 'U', 'D', 'lambda_']
    param_dict = {k: (data[k].tolist() if isinstance(data[k], np.ndarray) else data[k]) for k in param_keys if k in data}
    param_dict['MaxIt'] = MaxIt
    param_dict['nPop'] = nPop
    param_dict['pCrossover'] = pCrossover
    param_dict['pMutation'] = pMutation
    param_dict['patience'] = patience
    param_str = json.dumps(param_dict, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

# --- Data Preparation ---
def initialize_data(candidate_file, Em, xy_community):
    """
    Initialize the data dictionary for NSGA-II optimization.
    """
    # Parameters from initial_data.m
    n_community = Em.shape[0]  # Number of communities
    Q = np.array([30, 40, 50])  # Capacity options
    C = np.array([45, 65, 80])  # Construction costs
    Cp = 20  # Travel cost per unit demand per unit distance
    Cpp = 50  # Penalty cost
    U = 25  # Fixed number of facilities to open
    D = 200  # Minimum distance constraint between facilities
    lambda_ = 0.5  # Fuzzy parameter

    # Calculate E_L and E_U based on Em
    Ep = 0.6 * Em
    Eo = 1.4 * Em
    E_L = (lambda_ / 2 * (Em + Eo) / 2) + ((1 - lambda_ / 2) * (Ep + Em) / 2)
    E_U = ((1 - lambda_ / 2) * (Em + Eo) / 2) + (lambda_ / 2 * (Ep + Em) / 2)

    # Load candidate locations from Excel
    candidate_df = pd.read_excel(candidate_file)
    xy_candidate = candidate_df[["x", "y"]].values
    n_candidate = xy_candidate.shape[0]

    # Calculate dpp (distance between candidates)
    dpp = np.linalg.norm(xy_candidate[:, None, :] - xy_candidate[None, :, :], axis=2)

    # Calculate d (distance between candidates and communities)
    d = np.linalg.norm(xy_candidate[:, None, :] - xy_community[None, :, :], axis=2)

    # Calculate gamma (penalty coefficient based on distance bounds)
    gamma = np.zeros((n_candidate, n_community))
    for i in range(n_candidate):
        for j in range(n_community):
            if d[i, j] < E_L[j].item():
                gamma[i, j] = (E_L[j].item() - d[i, j]) / E_L[j].item()  # Penalty for being below lower bound
            elif d[i, j] > E_U[j].item():
                gamma[i, j] = (d[i, j] - E_U[j].item()) / E_U[j].item()  # Penalty for exceeding upper bound

    # Prepare the data dictionary
    data = {
        'n_candidate': n_candidate,
        'n_community': n_community,
        'Q': Q,
        'C': C,
        'Cp': Cp,
        'Cpp': Cpp,
        'U': U,
        'D': D,
        'lambda_': lambda_,
        'E_L': E_L,
        'E_U': E_U,
        'dpp': dpp,
        'd': d,
        'gamma': gamma,
        'xy_candidate': xy_candidate,
        'xy_community': xy_community
    }

    return data

# --- Output Directory ---
def ensure_output_dir():
    """
    Ensure the output directory exists.
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# --- Map Fetching ---
def fetch_osm_static_map(min_x, max_x, min_y, max_y, width=800, height=600):
    """
    Fetch a static map from OpenStreetMap for the given bounding box.
    """
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    url = f"https://staticmap.openstreetmap.de/staticmap.php?center={center_y},{center_x}&zoom=12&size={width}x{height}&maptype=mapnik"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching map: {e}")
        return None
# --- Bounding Box Calculation ---
def get_bounding_box(xy_candidate, xy_community, margin=0.05):
    """
    Calculate the bounding box for the given candidate and community coordinates.
    """
    all_x = np.concatenate([xy_candidate[:, 0], xy_community[:, 0]])
    all_y = np.concatenate([xy_candidate[:, 1], xy_community[:, 1]])
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    x_margin = margin * (max_x - min_x)
    y_margin = margin * (max_y - min_y)
    return min_x - x_margin, max_x + x_margin, min_y - y_margin, max_y + y_margin



def plot_all_candidates_with_demand(candidate_610, candidate_955, xy_community, E_L, map_path=None):
    output_dir = ensure_output_dir()
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Calculate bounding box for extent
    margin = 0.1
    min_x, max_x, min_y, max_y = get_bounding_box(candidate_955, xy_community, margin=margin)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Add map background if map_path exists
    if map_path is not None and os.path.exists(map_path):
        static_img = mpimg.imread(map_path)
        plt.imshow(static_img, extent=(min_x, max_x, min_y, max_y), aspect='auto', alpha=0.3)
    # Plot 610 candidates as green crosses
    plt.scatter(candidate_610[:, 0], candidate_610[:, 1], marker='x', s=30, color='green', alpha=0.3, label='610 Candidates')
    # Plot 955 candidates as red crosses
    plt.scatter(candidate_955[:, 0], candidate_955[:, 1], marker='x', s=30, color='red', alpha=0.3, label='955 Candidates')
    # Plot community demand bubbles
    Em_norm = E_L.mean()
    bubble_sizes = E_L.flatten() / Em_norm * 500 
    plt.scatter(xy_community[:, 0], xy_community[:, 1], 
                s=bubble_sizes, color='gray', alpha=0.3, label='Community Demand (Size=Demand)')
    plt.title("All Candidate Locations and Community Demands")
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.legend(scatterpoints=1)
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'all_candidates_with_demand.png'))
    plt.close()


