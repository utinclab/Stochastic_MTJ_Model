import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants (GdFeCo Ferrimagnet)
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
e = 1.60217662e-19    # Electron charge (C)
tf = 2.0e-9           # Film thickness (m)
J = 5.0e10            # Spin current density (A/m²)
theta_eff = 0.1       # Spin-Hall angle (dimensionless)
gamma_a = 1.76e11     # Gyromagnetic ratio for Fe sublattice (rad/(s·T))
gamma_b = -1.31e11    # Gyromagnetic ratio for Gd sublattice (rad/(s·T))
u0 = 4 * np.pi * 1e-7 # Vacuum permeability (T·m/A)
c = 0.5                 # Exchange coupling (A/m)
p = q = 0.5

# Sublattice magnetizations
M_a = 1.0e6  # A/m (FeCo)
M_b = 2.0e5  # A/m (Gd)
M = M_a - M_b  # Net magnetization

# Damping parameters
alpha_a = 0.015
alpha_b = 0.008

# Anisotropy constant (uniaxial)
K = 3.0e5  # J/m³

# External field
H_ext = np.array([0.01, 0, 0])  # Tesla

# Spin-Hall torque
T_a = hbar * theta_eff / (2 * e * tf * M_a)
T_b = hbar * theta_eff / (2 * e * tf * M_b)

# Effective parameters
alpha_eff = ((alpha_a * M_a) / gamma_a + (alpha_b * M_b) / gamma_b) / (M_a / gamma_a + M_b / gamma_b)
K_eff = K - u0 * (M_a - M_b) ** 2 / 2
gamma_eff = (M_a - M_b) / (M_a / gamma_a + M_b / gamma_b)

def ferri_LLG(m, mdot):
    """Computes the right-hand side of the ferrimagnetic LLG equation."""
    m = m / np.linalg.norm(m)  # Normalize to ensure |m|=1
    dE_ani = np.array([0, 0, 2 * m[2] * K_eff])
    
    term1 = -u0 * gamma_eff * np.cross(m, H_ext)
    term2 = gamma_eff / M * np.cross(m, np.cross(m, dE_ani))
    term3 = alpha_eff * np.cross(m, mdot)
    term4 = -gamma_eff * ((hbar * theta_eff) / (2 * e * tf * M)) * J * np.cross(m, np.cross(np.array([0, 1, 0]), m))
    print(mdot)
    return term1 + term2 + term3 + term4

def rk4(f, m0, dt, steps):
    """Runge-Kutta 4th order solver for LLG."""
    trajectory = np.zeros((steps, 3))
    m = m0
    m = m.astype(np.float64) # Ensure float64 for precision
    mdot = np.zeros(3)  # Initial derivative
    for i in range(steps):
        trajectory[i] = m

        # Compute derivatives dynamically
        k1 = f(m, mdot)
        k2 = f(m + k1 * dt / 2, (k1 * dt / 2))  # Update mdot dynamically
        k3 = f(m + k2 * dt / 2, (k2 * dt / 2))
        k4 = f(m + k3 * dt, (k3 * dt))
        
        mdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        m += mdot * dt
        m = m / np.linalg.norm(m)  # Normalize magnetization

    return trajectory

def ferri_LLG_a_b(m_a, m_b, m_a_dot, m_b_dot):
    """Computes the right-hand side of the ferrimagnetic LLG equation."""
    m_a = m_a / np.linalg.norm(m_a)  # Normalize to ensure |m|=1
    m_b = m_b / np.linalg.norm(m_b)  # Normalize to ensure |m|=1
    dE_ani_a = np.array([0, 0, 2 * m_a[2] * K_eff])
    dE_ani_b = np.array([0, 0, 2 * m_b[2] * K_eff])

    coefficient_a = gamma_a / M_a
    term1_a = np.cross(-M_a * m_a, u0 * (H_ext + c*M_b*m_b))
    term2_a = np.cross(M_a * m_a, dE_ani_a)
    term3_a = alpha_a / M_a / gamma_a * np.cross(M_a * m_a, m_a_dot)
    term4_a = -p * T_a * np.cross(M_a * m_a, np.cross(np.array([0, 1, 0]), m_a))

    coefficient_b = gamma_b / M_b
    term1_b = np.cross(-M_b * m_b, u0 * (H_ext - c*M_a*m_a))
    term2_b = np.cross(M_b * m_b, -dE_ani_b)  
    term3_b = alpha_b / M_b / gamma_b * np.cross(M_b * m_b, m_b_dot)
    term4_b = -q * T_b * np.cross(M_b * m_b, np.cross(np.array([0, 1, 0]), m_b))

    new_m_a = coefficient_a * (term1_a + term2_a + term3_a + term4_a)
    new_m_b = coefficient_b * (term1_b + term2_b + term3_b + term4_b)
    print(new_m_a, new_m_b)
    return new_m_a, new_m_b

def rk4_coupled(f, m_a0, m_b0, dt, steps):
    """Runge-Kutta 4th order solver for coupled ferrimagnetic LLG equations."""
    trajectory_a = np.zeros((steps, 3))
    trajectory_b = np.zeros((steps, 3))

    m_a = m_a0.astype(np.float64)
    m_b = m_b0.astype(np.float64)
    m_b_dot = np.zeros(3)  # Initial derivative for m_b
    m_a_dot = np.zeros(3)  # Initial derivative for m_a
    for i in range(steps):
        trajectory_a[i] = m_a
        trajectory_b[i] = m_b

        # Compute k1 (initial derivatives)
        k1_a, k1_b = f(m_a, m_b, m_a_dot, m_b_dot)
        k2_a, k2_b = f(m_a + k1_a * dt / 2, m_b + k1_b * dt / 2, k1_a * dt / 2, k1_b * dt / 2)
        k3_a, k3_b = f(m_a + k2_a * dt / 2, m_b + k2_b * dt / 2, k2_a * dt / 2, k2_b * dt / 2)
        k4_a, k4_b = f(m_a + k3_a * dt, m_b + k3_b * dt, k3_a * dt, k3_b * dt)

        #update m_a_dot and m_b_dot
        m_a_dot = (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6
        m_b_dot = (k1_b + 2 * k2_b + 2 * k3_b + k4_b) / 6

        # Update m_a and m_b
        m_a += m_a_dot * dt
        m_b += m_b_dot * dt

        # Normalize to maintain |m|=1
        m_a /= np.linalg.norm(m_a)
        m_b /= np.linalg.norm(m_b)

    return trajectory_a, trajectory_b

# Randomize initial magnetization state
def random_initial_magnetization():
    """Generate a random unit vector for the initial magnetization state."""
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1))  # Random polar angle
    
    # Convert spherical coordinates to Cartesian coordinates
    m0 = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return m0

# Random initial conditions
m0 = random_initial_magnetization()  # Random initial magnetization
dt = 5e-12  # Smaller time step for stability
steps = 5000  # More time steps for better resolution

# Solve for trajectory
m_trajectory = rk4(ferri_LLG, m0, dt, steps)

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
