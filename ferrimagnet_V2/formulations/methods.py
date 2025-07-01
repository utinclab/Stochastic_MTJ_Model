import numpy as np

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


# def rk4(f, m0, dt, steps):
#     """Runge-Kutta 4th order solver for equation of form mdot = f(m, mdot)."""
#     trajectory = np.zeros((steps, 3))
#     m = m0
#     m = m.astype(np.float64) # Ensure float64 for precision
#     mdot = np.zeros(3)  # Initial derivative
#     for i in range(steps):
#         trajectory[i] = m

#         # Compute derivatives dynamically
#         k1 = f(m, mdot)
#         k2 = f(m + k1 * dt / 2, (k1 * dt / 2))  # Update mdot dynamically
#         k3 = f(m + k2 * dt / 2, (k2 * dt / 2))
#         k4 = f(m + k3 * dt, (k3 * dt))
        
#         mdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
#         m += mdot * dt
#         m = m / np.linalg.norm(m)  # Normalize magnetization

#     return trajectory


# def bdf2(f, m0, dt, steps, tol=1e-6, max_iter=10):
#     """Backward Differentiation Formula (BDF2) solver for equation of form mdot = f(m, mdot)."""
#     trajectory = np.zeros((steps, 3))
#     m = np.array(m0, dtype=np.float64)  # Ensure float64 precision
#     mdot = np.zeros(3)  # Initial derivative
#     m_prev = m.copy()  # Store previous step

#     for i in range(steps):
#         trajectory[i] = m

#         # Compute the derivative dynamically using previous mdot
#         mdot = f(m, mdot)

#         if i == 0:
#             # First step: Use implicit Euler (BDF1)
#             m_next = m + mdot * dt
#         else:
#             # BDF2 implicit update (solve nonlinearly)
#             m_next = m.copy()

#             for _ in range(max_iter):
#                 mdot_new = f(m_next, mdot)  # Update mdot using new m_next
#                 m_new = (4 / 3) * m - (1 / 3) * m_prev + (2 / 3) * mdot_new * dt

#                 if np.linalg.norm(m_new - m_next) < tol:
#                     break  # Converged
                
#                 m_next = m_new  # Update estimate

#             m_prev = m.copy()  # Store previous step
#             m = m_next / np.linalg.norm(m_next)  # Normalize magnetization

#     return trajectory


# def bdf2_implicit(f, m0, dt, steps, tol=1e-9, max_iter=10):
#     """Backward Differentiation Formula (BDF2) solver for the LLG equation with implicit damping."""
#     trajectory = np.zeros((steps, 3))
#     m = np.array(m0, dtype=np.float64)  # Initial magnetization (ensure float64 precision)
#     mdot = np.zeros(3)  # Initial derivative (mdot = f(m, mdot) as needed)
#     m_prev = m.copy()  # Store previous magnetization for BDF2

#     for i in range(steps):
#         trajectory[i] = m

#         if i == 0:
#             # First step: Use implicit Euler (BDF1)
#             mdot = f(m, m, dt)  # Initially, we use m for both arguments in the function
#             m_next = m + mdot * dt
#         else:
#             # BDF2 implicit update (solve nonlinearly)
#             m_next = m.copy()

#             for _ in range(max_iter):
#                 # Update mdot using the new m_next
#                 mdot_new = f(m, m_next, dt)  # Call ferri_LLG with current m and m_next
#                 # Apply BDF2 formula for magnetization update
#                 m_new = (4 / 3) * m - (1 / 3) * m_prev + (2 / 3) * mdot_new * dt

#                 # Check for convergence of m_new and m_next
#                 if np.linalg.norm(m_new - m_next) < tol:
#                     break  # Converged
                
#                 m_next = m_new  # Update the estimate of m_next

#             m_prev = m.copy()  # Store previous magnetization
#             m = m_next / np.linalg.norm(m_next)  # Normalize the magnetization

#         # Store the magnetization trajectory at this time step
#         trajectory[i] = m

#     return trajectory


# def midpoint(f, m0, dt, steps, tol=1e-6, max_iter=10):
#     """Implicit Midpoint Method for equation of form mdot = f(m, mdot)."""
#     trajectory = np.zeros((steps, 3))
#     m = np.array(m0, dtype=np.float64)  # Ensure precision

#     for i in range(steps):
#         trajectory[i] = m
#         m_next = m.copy()  # Initial guess for iteration

#         for _ in range(max_iter):
#             # Compute mdot using midpoint approximation
#             m_mid = (m + m_next) / 2  # Midpoint for implicit update
#             mdot = f(m_mid, (m_next - m) / dt)  # Use estimated mdot

#             # Update m_next using implicit midpoint rule
#             m_new = m + dt * mdot

#             # Check for convergence
#             if np.linalg.norm(m_new - m_next) < tol:
#                 break

#             m_next = m_new  # Update guess

#         # Normalize to ensure |m| = 1
#         m = m_next / np.linalg.norm(m_next)

#     return trajectory


# def forward_euler(f, m0, dt, steps):
#         """Solve the LLG equation using Forward Euler method."""
#         trajectory = np.zeros((steps, 3))
#         m = np.array(m0, dtype=np.float64)  # Initial magnetization
#         mdot = np.zeros(3)  # Initial time derivative

#         for i in range(steps):
#             trajectory[i] = m

#             # Calculate the right-hand side of LLG
#             mdot = f(m, mdot)

#             # Update magnetization using Forward Euler method
#             m = m + mdot * dt

#             # Normalize magnetization after the update
#             m = m / np.linalg.norm(m)

#         return trajectory