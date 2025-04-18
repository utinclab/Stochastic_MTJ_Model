import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp # <--- Import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formulations import *

# Random initial conditions
dt = 1e-12  # Smaller time step for stability
steps = 20000  # More time steps for better resolution

# --- Time Integration using solve_ivp ---

# Define the time span for integration
t_start = 0.0
t_end = steps * dt
t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, steps + 1)

ferri = ferrimagnetic_device(**GdFeCo_MTJ)
m0_aniso = np.array([0.4, 0.1, np.sqrt(1-0.4**2-0.1**2)]) # Start at arbitrary angle

damp = ferrimagnetic_device(**pure_damp)
m0_damp = np.array([0.4, 0.1, np.sqrt(1-0.4**2-0.1**2)]) # Start at arbitrary angle

m0_random = random_initial_magnetization() # Initial condition (unit vector)



m0 = m0_random # Choose between m0_aniso or m0_random
ode_func = ferri.ferri_LLG # Define the ODE function

print("Starting ODE integration with solve_ivp...")
# Choose a method - 'RK45' is a good default (similar to RK4/5)
# Other options: 'BDF', 'LSODA' (good for stiff problems)
sol = solve_ivp(
    fun=ode_func,
    t_span=t_span,
    y0=m0,
    method='RK45',  # Or 'BDF', 'LSODA' etc.
    t_eval=t_eval,
    dense_output=False # Set True if you need interpolation between steps
)
print("Integration finished.")

# Check if the solver was successful
if not sol.success:
    print(f"WARNING: solve_ivp failed! Message: {sol.message}")
    # Decide how to handle failure (e.g., exit, use partial results)
    # For now, we'll try to plot what we got
    m_trajectory = sol.y.T # Transpose to get shape (n_points, 3)
    if m_trajectory.shape[0] < 2:
       print("Error: Not enough data points to plot.")
       sys.exit()
else:
    # The solution is stored in sol.y, with shape (n_dimensions, n_time_points)
    # We need to transpose it to match the shape expected by the plotting code: (n_points, 3)
    m_trajectory = sol.y.T
    print(f"Trajectory calculated with shape: {m_trajectory.shape}")

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
frame_skip = 20  # Or 5, 20, 50 etc.

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

# Pass the 'frame_indices' sequence to the 'frames' argument
# Adjust 'interval' (milliseconds) for delay between displayed frames (e.g., 20ms is 50fps)
ani = FuncAnimation(fig, update, frames=frame_indices, interval=20, blit=False, repeat=False)

# Show the plot
ax.legend()
plt.tight_layout()
plt.show()

# Optional: Save the animation
# print("Saving animation...")
# ani.save('magnetization_trajectory_skipped.gif', writer='imagemagick', fps=30) # Adjust fps
# print("Animation saved.")