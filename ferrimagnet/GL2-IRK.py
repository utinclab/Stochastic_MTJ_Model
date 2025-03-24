import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

# Constants
hbar = 1.0545718e-34
e = 1.60217662e-19
tf = 1.1e-9
J = 0
theta_eff = 0.1  
gamma = 2.211e5
u0 = 4 * np.pi * 1e-7
alpha_a = 0.01
alpha_b = 0.01
M_a = 1.2e6
M_b = 0
M = M_a - M_b
gamma_a = gamma / u0
gamma_b = gamma / u0
K = 1.0e5
H_ext = np.array([0.0, 20, 1.0e5])  

alpha_eff = ((alpha_a * M_a) / gamma_a + (alpha_b * M_b) / gamma_b) / (M_a / gamma_a + M_b / gamma_b)
K_eff = K - u0 * (M_a - M_b) ** 2 / 2
gamma_eff = (M_a - M_b) / (M_a / gamma_a + M_b / gamma_b)

def ferri_LLG(m, mdot):
    """Computes the right-hand side of the ferrimagnetic LLG equation."""
    m = m / np.linalg.norm(m)  
    dE_ani = np.array([0, 0, 2 * m[2] * K_eff])
    
    term1 = -u0 * gamma_eff * np.cross(m, H_ext)
    term2 = gamma_eff / M * np.cross(m, np.cross(m, dE_ani))
    term3 = alpha_eff * np.cross(m, mdot)
    term4 = -gamma_eff * ((hbar * theta_eff) / (2 * e * tf * M)) * J * np.cross(m, np.cross(np.array([0, 1, 0]), m))
    
    return term1 + term2 + term3 + term4 

def gauss_legendre_2nd_order(m0, dt):
    """Solves the LLG equation using the 2-stage Gauss-Legendre implicit Runge-Kutta method."""
    A = np.array([[0.5, 0.5 - np.sqrt(3) / 6], 
                  [0.5 + np.sqrt(3) / 6, 0.5]])
    b = np.array([0.5, 0.5])
    c = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])

    def residual(K):
        """Computes the residual for Newton's method."""
        K = K.reshape(2, 3)
        
        # Update intermediate stages
        m1 = m0 + dt * (A[0, 0] * K[0] + A[0, 1] * K[1])
        m2 = m0 + dt * (A[1, 0] * K[0] + A[1, 1] * K[1])
        
        m1 /= np.linalg.norm(m1)  
        m2 /= np.linalg.norm(m2)

        mdot1 = ferri_LLG(m1, K[0])
        mdot2 = ferri_LLG(m2, K[1])

        return np.hstack([K[0] - mdot1, K[1] - mdot2])

    K0 = np.zeros((2, 3))
    K_sol = fsolve(residual, K0.flatten()).reshape(2, 3)

    # Compute new m
    m_new = m0 + dt * (b[0] * K_sol[0] + b[1] * K_sol[1])
    m_new /= np.linalg.norm(m_new)  

    return m_new

# Simulation parameters
dt = 5e-11
num_steps = 500  
m_init = np.array([0, 1, 0])  

# Store trajectory
m_trajectory = np.zeros((num_steps, 3))
m_trajectory[0] = m_init

for i in range(1, num_steps):
    m_trajectory[i] = gauss_legendre_2nd_order(m_trajectory[i-1], dt)

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(m_trajectory[:, 0], m_trajectory[:, 1], m_trajectory[:, 2], label="Magnetization Path")
ax.scatter(m_trajectory[0, 0], m_trajectory[0, 1], m_trajectory[0, 2], color="red", label="Start")
ax.scatter(m_trajectory[-1, 0], m_trajectory[-1, 1], m_trajectory[-1, 2], color="blue", label="End")

# Set fixed limits for all axes
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Ensure equal aspect ratio
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("m_x")
ax.set_ylabel("m_y")
ax.set_zlabel("m_z")
ax.set_title("3D Magnetization Trajectory")
ax.legend()
plt.show()
