import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root
from RK4 import *
# Random initial conditions
m_a0 = np.random.randn(3)
m_a0 /= np.linalg.norm(m_a0)

m_b0 = -1 * m_a0

# Solve for trajectory
dt = 5e-12
steps = 5000
m_a_trajectory, m_b_trajectory = rk4_coupled(ferri_LLG_a_b, m_a0, m_b0, dt, steps)
print("Done!")
# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(m_a_trajectory[:, 0], m_a_trajectory[:, 1], m_a_trajectory[:, 2], label="Magnetization A Path")
ax.plot(m_b_trajectory[:, 0], m_b_trajectory[:, 1], m_b_trajectory[:, 2], label="Magnetization B Path", linestyle="dashed")

ax.scatter(m_a_trajectory[0, 0], m_a_trajectory[0, 1], m_a_trajectory[0, 2], color="red", label="Start A")
ax.scatter(m_b_trajectory[0, 0], m_b_trajectory[0, 1], m_b_trajectory[0, 2], color="orange", label="Start B")

ax.scatter(m_a_trajectory[-1, 0], m_a_trajectory[-1, 1], m_a_trajectory[-1, 2], color="blue", label="End A")
ax.scatter(m_b_trajectory[-1, 0], m_b_trajectory[-1, 1], m_b_trajectory[-1, 2], color="purple", label="End B")

# Set fixed limits for all axes to maintain constant radius 1
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])  # Keep aspect ratio square

ax.set_xlabel("m_x")
ax.set_ylabel("m_y")
ax.set_zlabel("m_z")
ax.set_title("3D Coupled Magnetization Trajectories")
ax.legend()
plt.show()