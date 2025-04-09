import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formulations import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Random initial conditions
m0 = random_initial_magnetization()  # Random initial magnetization
dt = 5e-11  # Smaller time step for stability
steps = 5000  # More time steps for better resolution

# Define the ferrimagnetic device
ferri_mtj = ferrimagnetic_device(**devices.GdFeCo_MTJ)

# Solve for trajectory
midpoint_trajectory = midpoint(ferri_mtj.ferri_LLG, m0, dt, steps)
rk4_trajectory = rk4(ferri_mtj.ferri_LLG, m0, dt, steps)
bdf2_trajectory = bdf2(ferri_mtj.ferri_LLG, m0, dt, steps)

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
ax.set_title("3D Magnetization Trajectories (GdFeCo)")

# Initialize the lines for the three magnetization paths (to update in the animation)
line_midpoint, = ax.plot([], [], [], label="Midpoint Trajectory", color="green")
line_rk4, = ax.plot([], [], [], label="RK4 Trajectory", color="purple")
line_bdf2, = ax.plot([], [], [], label="BDF2 Trajectory", color="orange")

# Initialize quivers (arrows) for the start and end of each trajectory
start_arrow_midpoint = ax.quiver(0, 0, 0, midpoint_trajectory[0, 0], midpoint_trajectory[0, 1], midpoint_trajectory[0, 2], color="green", length=1, label="Midpoint Start Arrow")
end_arrow_midpoint = ax.quiver(0, 0, 0, midpoint_trajectory[-1, 0], midpoint_trajectory[-1, 1], midpoint_trajectory[-1, 2], color="green", length=1, label="Midpoint End Arrow")

start_arrow_rk4 = ax.quiver(0, 0, 0, rk4_trajectory[0, 0], rk4_trajectory[0, 1], rk4_trajectory[0, 2], color="purple", length=1, label="RK4 Start Arrow")
end_arrow_rk4 = ax.quiver(0, 0, 0, rk4_trajectory[-1, 0], rk4_trajectory[-1, 1], rk4_trajectory[-1, 2], color="purple", length=1, label="RK4 End Arrow")

start_arrow_bdf2 = ax.quiver(0, 0, 0, bdf2_trajectory[0, 0], bdf2_trajectory[0, 1], bdf2_trajectory[0, 2], color="orange", length=1, label="BDF2 Start Arrow")
end_arrow_bdf2 = ax.quiver(0, 0, 0, bdf2_trajectory[-1, 0], bdf2_trajectory[-1, 1], bdf2_trajectory[-1, 2], color="orange", length=1, label="BDF2 End Arrow")

# Animation update function
def update(num):
    # Update the magnetization trajectories
    line_midpoint.set_data(midpoint_trajectory[:num, 0], midpoint_trajectory[:num, 1])
    line_midpoint.set_3d_properties(midpoint_trajectory[:num, 2])

    line_rk4.set_data(rk4_trajectory[:num, 0], rk4_trajectory[:num, 1])
    line_rk4.set_3d_properties(rk4_trajectory[:num, 2])

    line_bdf2.set_data(bdf2_trajectory[:num, 0], bdf2_trajectory[:num, 1])
    line_bdf2.set_3d_properties(bdf2_trajectory[:num, 2])

    # Update the quivers (arrows) for each trajectory
    start_arrow_midpoint.set_segments([[[0, 0, 0], [midpoint_trajectory[num, 0], midpoint_trajectory[num, 1], midpoint_trajectory[num, 2]]]])
    end_arrow_midpoint.set_segments([[[0, 0, 0], [midpoint_trajectory[-1, 0], midpoint_trajectory[-1, 1], midpoint_trajectory[-1, 2]]]])

    start_arrow_rk4.set_segments([[[0, 0, 0], [rk4_trajectory[num, 0], rk4_trajectory[num, 1], rk4_trajectory[num, 2]]]])
    end_arrow_rk4.set_segments([[[0, 0, 0], [rk4_trajectory[-1, 0], rk4_trajectory[-1, 1], rk4_trajectory[-1, 2]]]])

    start_arrow_bdf2.set_segments([[[0, 0, 0], [bdf2_trajectory[num, 0], bdf2_trajectory[num, 1], bdf2_trajectory[num, 2]]]])
    end_arrow_bdf2.set_segments([[[0, 0, 0], [bdf2_trajectory[-1, 0], bdf2_trajectory[-1, 1], bdf2_trajectory[-1, 2]]]])

    return line_midpoint, line_rk4, line_bdf2, start_arrow_midpoint, end_arrow_midpoint, start_arrow_rk4, end_arrow_rk4, start_arrow_bdf2, end_arrow_bdf2

# Create the animation
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

# Show the animation
plt.legend()
plt.show()