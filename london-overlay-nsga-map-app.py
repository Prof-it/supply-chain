import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import time

# Set page config to wide mode
st.set_page_config(layout="wide")

# London bounds
LAT_MIN, LAT_MAX = 51.3, 51.7
LON_MIN, LON_MAX = -0.5, 0.3
CENTER_LAT, CENTER_LON = 51.5074, -0.1278

# Simulate population as patients: 1000 people randomly distributed
np.random.seed(42)
patients = np.column_stack([
    np.random.uniform(LAT_MIN, LAT_MAX, 1000),
    np.random.uniform(LON_MIN, LON_MAX, 1000)
])

# Parameters for center capacity
CENTER_CAPACITY = 150  # max patients a center can serve (simulated limit)
NUM_CENTERS = 3  # or any number you want

# Define the problem with capacity and equity constraints
class LondonNursingProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, xl=np.array([LAT_MIN, LON_MIN]), xu=np.array([LAT_MAX, LON_MAX]))

    def _evaluate(self, x, out, *args, **kwargs):
        lat, lon = x
        distances = np.sqrt((patients[:, 0] - lat)**2 + (patients[:, 1] - lon)**2)
        sorted_dists = np.sort(distances)
        
        # Simulate limited capacity: only closest patients served
        served_distances = sorted_dists[:CENTER_CAPACITY]
        unserved_penalty = np.mean(sorted_dists[CENTER_CAPACITY:]) if len(sorted_dists) > CENTER_CAPACITY else 0

        # Objective 1: cost/utility
        utility_penalty = np.mean(served_distances) + unserved_penalty

        # Objective 2: equity (std dev of served distances)
        equity_penalty = np.std(served_distances)

        out["F"] = [utility_penalty, equity_penalty]

class LondonMultiCenterProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=NUM_CENTERS * 2,
            n_obj=2,
            xl=np.array([LAT_MIN, LON_MIN] * NUM_CENTERS),
            xu=np.array([LAT_MAX, LON_MAX] * NUM_CENTERS),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        centers = x.reshape(NUM_CENTERS, 2)
        # Assign each patient to nearest center
        dists = np.linalg.norm(patients[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        # Capacity constraint: can be added here if needed
        out["F"] = [np.mean(min_dists), np.std(min_dists)]

@st.cache_data
def run_ga():
    problem = LondonNursingProblem()
    algorithm = NSGA2(pop_size=50)
    result = minimize(problem, algorithm, ("n_gen", 20), save_history=True, verbose=False)
    history_coords = [gen.pop.get("X") for gen in result.history]
    return history_coords

@st.cache_data
def run_ga_multi():
    problem = LondonMultiCenterProblem()
    algorithm = NSGA2(pop_size=50)
    result = minimize(problem, algorithm, ("n_gen", 20), save_history=True, verbose=False)
    history_coords = [gen.pop.get("X") for gen in result.history]
    return history_coords

# --- Run GAs ---
coords_per_gen_single = run_ga()
coords_per_gen_multi = run_ga_multi()
num_gens = min(len(coords_per_gen_single), len(coords_per_gen_multi))

# Streamlit app UI
st.title("NSGA-II: Optimal Nursing Center Placement in London")

# --- Animation Logic (must be before any UI rendering!) ---
if "generation_slider" not in st.session_state:
    st.session_state.generation_slider = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

if st.session_state.is_playing:
    time.sleep(2)
    next_gen = (st.session_state.generation_slider + 1) % num_gens
    st.session_state.generation_slider = next_gen
    st.rerun()
    st.stop()  # Prevents double rendering

# --- Play Button (top, single-step) ---
play_step = st.button("▶️ Step")
if play_step:
    next_gen = (st.session_state.generation_slider + 1) % num_gens
    st.session_state.generation_slider = next_gen
    st.rerun()

# --- Slider (single, controls both) ---
generation = st.slider(
    "Generation", 0, num_gens - 1, st.session_state.generation_slider, key="generation_slider"
)
# Do NOT assign to st.session_state.generation_slider after this line!

# --- Data for current generation ---
candidates_df_single = pd.DataFrame(coords_per_gen_single[st.session_state.generation_slider], columns=["lat", "lon"])
all_centers = []
for sol in coords_per_gen_multi[st.session_state.generation_slider]:
    centers = np.array(sol).reshape(NUM_CENTERS, 2)
    for lat, lon in centers:
        all_centers.append({"lat": lat, "lon": lon})
candidates_df_multi = pd.DataFrame(all_centers)

patient_df = pd.DataFrame(patients, columns=["lat", "lon"])
patient_df["count"] = 1

layer_patients = pdk.Layer(
    "HeatmapLayer",
    patient_df,
    get_position="[lon, lat]",
    aggregation="SUM",
    get_weight="count",
    radiusPixels=60,
    intensity=0.6,
    opacity=0.7,
    color_range=[
        [255, 255, 255, 0],
        [173, 216, 230, 60],
        [144, 238, 144, 100],
        [255, 255, 102, 140],
        [30, 144, 255, 180],
        [0, 0, 139, 220],
    ]
)

layer_candidates_single = pdk.Layer(
    "ScatterplotLayer",
    candidates_df_single,
    get_position="[lon, lat]",
    get_fill_color="[220, 20, 60, 220]",  # Bright red, more opaque
    get_radius=250,                      # Bigger points
)

layer_candidates_multi = pdk.Layer(
    "ScatterplotLayer",
    candidates_df_multi,
    get_position="[lon, lat]",
    get_fill_color="[220, 20, 60, 180]",  # Bright red, slightly less opaque
    get_radius=250,                      # Bigger points
)

view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10)

# --- Show maps side by side ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Single Center Optimization")
    st.pydeck_chart(pdk.Deck(
        layers=[layer_patients, layer_candidates_single],
        initial_view_state=view_state,
        tooltip={"text": "Location: [{lat}, {lon}]"},
        map_style="light"
    ))

with col2:
    st.markdown("### Multi-Center Optimization")
    st.pydeck_chart(pdk.Deck(
        layers=[layer_patients, layer_candidates_multi],
        initial_view_state=view_state,
        tooltip={"text": "Location: [{lat}, {lon}]"},
        map_style="light"
    ))

st.markdown("""
### Model Details:
- Patients are randomly distributed over London.
- Capacity constraint simulated: Each center can only serve 150 closest patients.
- Objective 1: Minimize average travel distance (including penalty for unserved).
- Objective 2: Minimize standard deviation of travel distance (equity).
- Use the play button to animate, or the slider to navigate generations of NSGA-II.
""")