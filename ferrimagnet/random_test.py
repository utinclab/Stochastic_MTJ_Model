import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def llg(t, M, H_eff, gamma, alpha):
    """
    Landau-Lifshitz-Gilbert (LLG) equation.
    Parameters:
    - t: time (unused in this case but required by ODE solvers)
    - M: Magnetization vector (3D)
    - H_eff: Effective magnetic field (3D)
    - gamma: Gyromagnetic ratio
    - alpha: Damping constant
    
    Returns the derivative of magnetization
    """
    # Compute cross products for the LLG equation
    cross1 = np.cross(M, H_eff)
    cross2 = np.cross(M, cross1)
    
    # LLG equation components
    dMdt = -gamma * cross1 + alpha * np.cross(M, cross2)
    return dMdt

# Define parameters
gamma = 2.21e5  # Gyromagnetic ratio (1/s/T)
alpha = 0.01    # Damping constant
H_eff = np.array([0.0, 0.0, 1.0])  # Example effective field (along z-axis)

# Initial magnetization (normalized)
M0 = np.array([0.0, 1.0, 1.0])

# Time span for simulation
t_span = (0, 10e-9)  # 10 ns
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve the ODE using solve_ivp (for vectorized solvers)
sol = solve_ivp(llg, t_span, M0, t_eval=t_eval, args=(H_eff, gamma, alpha))

# Extract the solution
m_trajectory = sol.y.T  # Transpose to match (time, Mx, My, Mz)

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

