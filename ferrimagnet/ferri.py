import numpy as np
from scipy.optimize import root
# Removed unused import
import matplotlib.pyplot as plt

def rk4_step_implicit(f, x, dt):
    """
    Perform a single step of the 4th-order Runge-Kutta method for implicit equations.

    Parameters:
        f: function
            The derivative function f(x, xdot) that returns 0 when solved for xdot.
        x: ndarray or float
            The current state.
        t: float
            The current time.
        dt: float
            The time step.

    Returns:
        ndarray or float
            The state after one RK4 step.
    """
    def solve_xdot(x, xdot_guess):
        """
        Solve for xdot using a root-finding method.
        """
        sol = root(lambda xdot: xdot - f(x, xdot), xdot_guess)
        if not sol.success:
            raise RuntimeError("Root-finding failed for xdot.")
        return sol.x

    # Initial guess for xdot (can be zero or the previous xdot)
    xdot_guess = np.zeros_like(x)

    # Solve for k1
    try:
        k1 = solve_xdot(x, xdot_guess)
        k2 = solve_xdot(x + 0.5 * dt * k1, k1)
        k3 = solve_xdot(x + 0.5 * dt * k2, k2)
        k4 = solve_xdot(x + dt * k3, k3)
    except RuntimeError as e:
        print(f"Root-finding failed: {e}")
        return x  # Return the current state if root-finding fails

    # Update x
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

# Example usage:
# Define your function f(x, xdot) in the form 0 = xdot - f(x, xdot)
def example_f(x, xdot):
    # Example: A simpler implicit system
    return np.cross(x, xdot) - x

# eq. 2 Je et. al. 2018
def ferri_LLG(m, mdot):
    # Constants
    hbar = 1.0545718e-34
    e = 1.60217662e-19
    tf = 1.1e-9
    J = 1.0e-12
    theta_eff = 0.0
    gamma = 2.211e5
    u0 = 4 * np.pi * 1e-7
    alpha_a = 0.01
    alpha_b = 0.01
    M_a = 1.2e6
    M_b = 1.2e5
    M = M_a - M_b
    gamma_a = gamma / u0
    gamma_b = gamma / u0
    K = 1.0e5
    H_ext = np.array([0.0, 0.0, 0.0])

    # effective dampling constant
    alpha_eff = ((alpha_a * M_a)/gamma_a + (alpha_b * M_b)/gamma_b) / (M_a/gamma_a + M_b/gamma_b)
    # total effective anisotropy
    K_eff = K - u0 * (M_a - M_b) * (M_a - M_b) / 2
    # effective gyromagnetic ratio
    gamma_eff = (M_a - M_b) / (M_a/gamma_a + M_b/gamma_b)
    # total anisotrpy energy
    dE_ani = np.array([0, 0, 2 * m[2] * K_eff])

    # ferri LLG equation
    term1 = -u0 * gamma_eff * np.cross(m, H_ext)
    term2 = gamma_eff / M * np.cross(m, np.cross(m, dE_ani))
    term3 = alpha_eff * np.cross(m, mdot)
    term4 = -gamma_eff * (1) * ((hbar * theta_eff) / (2 * e * tf * M)) * J * np.cross(m, np.cross(np.array([0, 1, 0]), m))
    return term1 + term2 + term3 + term4


# Initial conditions for a 3D vector
x0 = np.array([1.0, 0.0, 0])
t0 = 0.0
dt = 0.01

# Perform one RK4 step
x1 = rk4_step_implicit(example_f, x0, dt)
print(f"Next state: {x1}")

# Time range for simulation
t_end = 10.0
time_steps = int(t_end / dt)
time = np.linspace(t0, t_end, time_steps)

# Initialize state array
states = np.zeros((time_steps, len(x0)))
states[0] = x0

# Perform RK4 steps over the time range
for i in range(1, time_steps):
    states[i] = rk4_step_implicit(ferri_LLG, states[i - 1], dt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, states[:, 0], label='x[0]')
plt.plot(time, states[:, 1], label='x[1]')
plt.plot(time, states[:, 2], label='x[2]')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('System Evolution Over Time')
plt.legend()
plt.grid()
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Trajectory')

# Label the axes
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('x[2]')
ax.set_title('3D System Evolution Over Time')

# Add a legend and grid
ax.legend()
ax.grid()

# Show the plot
plt.show()