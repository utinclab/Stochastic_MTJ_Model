import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formulations import *

# Random initial conditions
m0 = random_initial_magnetization()  # Random initial magnetization
dt = 5e-12  # Smaller time step for stability
steps = 500  # More time steps for better resolution

# Define the ferrimagnetic device
ferri_mtj = ferrimagnetic_device(**devices.GdFeCo_MTJ)

# Solve for trajectory
m_trajectory = bdf2_implicit(ferri_mtj.ferri_LLG_implicit, m0, dt, steps)

# Set up the figure for the 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set fixed limits for all axes to maintain constant radius 1
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio square

ax.set_xlabel("m_x")
ax.set_ylabel("m_y")
ax.set_zlabel("m_z")
ax.set_title("3D Magnetization Trajectory (GdFeCo)")

# Initialize the line object for the magnetization path (to update in the animation)
line, = ax.plot([], [], [], label="Magnetization Path", color="black")
ax.scatter(m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", label="Start")
ax.scatter(m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", label="End")

# Initialize quivers (arrows) without assigning them to variables initially
start_arrow = ax.quiver(0, 0, 0, m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", length=1, label="Start Arrow")
end_arrow = ax.quiver(0, 0, 0, m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", length=1, label="End Arrow")

# Animation update function
def update(num):
    # Update the magnetization trajectory
    line.set_data(m_trajectory[:num, 0], m_trajectory[:num, 1])
    line.set_3d_properties(m_trajectory[:num, 2])

    # Update the quivers (arrows) for start and end using set_data and set_3d_properties
    start_arrow.set_segments([[[0, 0, 0], [m_trajectory[num, 0], m_trajectory[num, 1], m_trajectory[num, 2]]]])
    end_arrow.set_segments([[[0, 0, 0], [m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2]]]])

    return line, start_arrow, end_arrow

# Create the animation
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

# Show the animation
plt.legend()
plt.show()
