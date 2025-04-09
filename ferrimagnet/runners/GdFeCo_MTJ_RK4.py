import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formulations import *
import matplotlib.pyplot as plt

# Random initial conditions
m0 = random_initial_magnetization()  # Random initial magnetization
dt = 5e-10  # Smaller time step for stability
steps = 5000  # More time steps for better resolution

# Define the ferrimagnetic device
ferri_mtj = ferrimagnetic_device(**devices.GdFeCo_MTJ)

# Solve for trajectory
m_trajectory = rk4(ferri_mtj.ferri_LLG, m0, dt, steps)

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(m_trajectory[:, 0], m_trajectory[:, 1], m_trajectory[:, 2], label="Magnetization Path")
ax.scatter(m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", label="Start")
ax.scatter(m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", label="End")

# Plot arrows from origin to start and end
ax.quiver(0, 0, 0, m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", length=1, label="Start Arrow")
ax.quiver(0, 0, 0, m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", length=1, label="End Arrow")

# Set fixed limits for all axes to maintain constant radius 1
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio square

ax.set_xlabel("m_x")
ax.set_ylabel("m_y")
ax.set_zlabel("m_z")
ax.set_title("3D Magnetization Trajectory (GdFeCo)")
ax.legend()

plt.show()
