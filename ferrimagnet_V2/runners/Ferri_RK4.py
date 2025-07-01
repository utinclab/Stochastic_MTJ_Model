import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp # <--- Import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formulations import *

def euler_integrate(ode_func, t_span, y0, dt):
    t0, t1 = t_span
    num_steps = int((t1 - t0) / dt)
    t_vals = np.linspace(t0, t1, num_steps + 1)
    y_vals = np.zeros((num_steps + 1, 3))
    y_vals[0] = y0
    m = y0.copy()

    for i in range(num_steps):
        dmdt = ode_func(t_vals[i], m)
        m += dt * dmdt
        m = m / np.linalg.norm(m)  # Normalize to keep m a unit vector
        y_vals[i + 1] = m

    return t_vals, y_vals

# Random initial conditions
dt = 5e-15  # Smaller time step for stability
steps = 200000000  # More time steps for better resolution

# --- Time Integration using solve_ivp ---

# Define the time span for integration
t_start = 0.0
t_end = steps * dt
t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, steps + 1)

ferri = ferrimagnetic_device(**GdFeCo_MTJ)
m0_aniso = np.array([0.4, 0.1, np.sqrt(1-0.4**2-0.1**2)]) # Start at arbitrary angle

# damp = ferrimagnetic_device(**pure_damp)
# m0_damp = np.array([0.4, 0.1, np.sqrt(1-0.4**2-0.1**2)]) # Start at arbitrary angle

# noise = ferrimagnetic_device(**thermal_noise_test)
# m0_random = random_initial_magnetization() # Initial condition (unit vector)

# kwon = ferrimagnetic_device(**ferromagnet_test, ferro=True)
m0_random = random_initial_magnetization() # Initial condition (unit vector)

m0 = m0_random # Choose between m0_aniso or m0_random
ode_func = ferri.ferri_LLG # Define the ODE function

t_vals, m_trajectory = euler_integrate(ode_func, t_span=(0, 5e-9), y0=m0, dt=1e-12)
print("Euler integration finished.")



# --- Plotting Setup ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio

ax.set_xlabel("m_x")
ax.set_ylabel("m_y")
ax.set_zlabel("m_z")
ax.set_title("3D Magnetization Trajectory (GdFeCo)")

# Initialize the line object for the magnetization path
line, = ax.plot([], [], [], label="Magnetization Path", color="black", lw=1.5)

# Plot Start/End points (static)
if m_trajectory.shape[0] > 0: # Check if trajectory exists
    ax.scatter(m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", s=50, label="Start", depthshade=False)
    ax.scatter(m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", s=50, label="End", depthshade=False)
    
    # Initialize quivers (arrows) for Start and End directions (static)
    start_arrow = None
    end_arrow = ax.quiver(0, 0, 0, m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", length=np.linalg.norm(m_trajectory[-1]), normalize=False, label="End Vec")
else:
    print("Warning: m_trajectory is empty, cannot plot points or arrows.")
    # Initialize dummy arrows if needed for update function structure
    start_arrow = ax.quiver(0, 0, 0, 0, 0, 0) 
    # end_arrow = ax.quiver(0, 0, 0, 0, 0, 0) # End arrow doesn't need update

# --- Animation Setup ---

# Get the total number of points/frames available in the trajectory
total_data_points = m_trajectory.shape[0]

# Define how many data points to skip between animation frames
# Example: frame_skip = 10 will display data point 0, 10, 20, etc.
# Adjust this value: larger skip means faster animation, less smooth.
frame_skip = 10  # Or 5, 20, 50 etc.

# Generate the sequence of trajectory indices to actually use for animation frames
# We use range(start, stop, step)
frame_indices = range(0, total_data_points, frame_skip)

# Store the dynamically changing arrow
start_arrow_container = {"arrow": None}

def update(frame_idx):
    line.set_data(m_trajectory[:frame_idx+1, 0], m_trajectory[:frame_idx+1, 1])
    line.set_3d_properties(m_trajectory[:frame_idx+1, 2])

    # Remove previous arrow if exists
    if start_arrow_container["arrow"] is not None:
        start_arrow_container["arrow"].remove()

    # Add new arrow at current frame
    if m_trajectory.shape[0] > frame_idx:
        current_m = m_trajectory[frame_idx]
        arrow = ax.quiver(0, 0, 0,
                          current_m[0], current_m[1], current_m[2],
                          color="red",
                          length=np.linalg.norm(current_m),
                          normalize=False)
        start_arrow_container["arrow"] = arrow

    return line

# Create the animation
print(f"Creating animation with {len(frame_indices)} frames (skipping {frame_skip-1} data points between frames)...")
ani = FuncAnimation(fig, update, frames=frame_indices, interval=20, blit=False, repeat=False)

# Show the plot
ax.legend()
plt.tight_layout()
plt.show()