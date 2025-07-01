import numpy as np
from numba import njit

class ferrimagnetic_device():
    # constants
    hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
    e = 1.60217662e-19    # Electron charge (C)
    u0 = 4 * np.pi * 1e-7 # Vacuum permeability (T·m/A)
    k_B = 1.380649e-23    # Boltzmann constant (J/K)
    
    def laplacian(self, f, dx, dy, dz):
        d2f_dx2 = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2
        d2f_dy2 = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2
        d2f_dz2 = (np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / dz**2
        return d2f_dx2 + d2f_dy2 + d2f_dz2

    def __init__(self, tf, J, theta_eff, gamma_a, gamma_b, c, p, q, M_a, M_b, alpha_a, alpha_b, K, H_ext, dt, T=300, ferro=False):
        self.tf = tf                    # Film thickness (m)
        self.J = J                      # Spin current density (A/m²)
        self.theta_eff = theta_eff      # Spin-Hall angle (dimensionless)
        self.gamma_a = gamma_a          # Gyromagnetic ratio for Fe sublattice (rad/(s·T))
        self.gamma_b = gamma_b          # Gyromagnetic ratio for Gd sublattice (rad/(s·T))
        self.c = c                      # Exchange coupling (A/m)
        self.p = p
        self.q = q
        self.M_a = M_a                  # magnetization saturation a
        self.M_b = M_b                  # magnetization saturation b
        self.M = M_a if ferro else M_a - M_b              # Net magnetization
        self.alpha_a = alpha_a          # Damping parameter for a sublattice
        self.alpha_b = alpha_b          # Damping parameter for b sublattice
        self.K = K                      # Anisotropy constant (uniaxial)
        self.H_ext = H_ext              # External field
        self.T = T                      # Temperature (K)
        self.dt = dt                    # Time step (s)
        self.count = 0
        self.a = 50e-9
        self.b = 50e-9
        self.v = self.tf*np.pi*self.b*self.a/4
        self.easy_axis = np.array([0, 0, 1])  # Default easy axis along z-direction
        self.K_u = 889e3 # Uniaxial anisotropy constant (J/m³)
        self.A_x = 1e-12  # Exchange stiffness (J/m)

        # Effective parameters
        print(f"[DEBUG] alpha_a: {self.alpha_a}, M_a: {self.M_a}, gamma_a: {self.gamma_a}")
        print(f"[DEBUG] alpha_b: {self.alpha_b}, M_b: {self.M_b}, gamma_b: {self.gamma_b}")

        numerator = (self.alpha_a * self.M_a) / self.gamma_a + (self.alpha_b * self.M_b) / self.gamma_b
        denominator = (self.M_a / self.gamma_a) - (self.M_b / self.gamma_b)

        print(f"[DEBUG] Numerator: {numerator}")
        print(f"[DEBUG] Denominator: {denominator}")

        if denominator == 0:
            print("[ERROR] Denominator is zero! Check M_a, M_b, gamma_a, gamma_b values.")
            # Optionally raise error or handle it
            raise ZeroDivisionError("Division by zero in alpha_eff calculation.")

        self.alpha_eff = numerator / denominator
        self.K_eff = K - self.u0 * (M_a - M_b) ** 2 / 2
        self.gamma_eff = (M_a - M_b) / (M_a / gamma_a - M_b / gamma_b)
        # Print all initialized parameters
        print("Initialized Parameters:")
        print(f"  Film thickness (tf): {self.tf} m")
        print(f"  Spin current density (J): {self.J} A/m²")
        print(f"  Spin-Hall angle (theta_eff): {self.theta_eff}")
        print(f"  Gyromagnetic ratio (gamma_a): {self.gamma_a} rad/(s·T)")
        print(f"  Gyromagnetic ratio (gamma_b): {self.gamma_b} rad/(s·T)")
        print(f"  Exchange coupling (c): {self.c} A/m")
        print(f"  Magnetization saturation (M_a): {self.M_a} A/m")
        print(f"  Magnetization saturation (M_b): {self.M_b} A/m")
        print(f"  Net magnetization (M): {self.M} A/m")
        print(f"  Damping parameter (alpha_a): {self.alpha_a}")
        print(f"  Damping parameter (alpha_b): {self.alpha_b}")
        print(f"  Anisotropy constant (K): {self.K}")
        print(f"  External field (H_ext): {self.H_ext}")
        print(f"  Effective damping (alpha_eff): {self.alpha_eff}")
        print(f"  Effective anisotropy (K_eff): {self.K_eff}")
        print(f"  Effective gyromagnetic ratio (gamma_eff): {self.gamma_eff} rad/(s·T)")

    def __str__(self):
        """Returns a formatted string representation of the device parameters."""
        return (
            f"Ferrimagnetic Device Parameters:\n"
            f"  Film thickness (tf): {self.tf} m\n"
            f"  Spin current density (J): {self.J} A/m**2\n"
            f"  Spin-Hall angle (theta_eff): {self.theta_eff}\n"
            f"  Gyromagnetic ratio (gamma_a): {self.gamma_a} rad/(s*T)\n"
            f"  Gyromagnetic ratio (gamma_b): {self.gamma_b} rad/(s*T)\n"
            f"  Exchange coupling (c): {self.c} A/m\n"
            f"  Magnetization saturation (M_a): {self.M_a} A/m\n"
            f"  Magnetization saturation (M_b): {self.M_b} A/m\n"
            f"  Net magnetization (M): {self.M} A/m\n"
            f"  Damping parameter (alpha_a): {self.alpha_a}\n"
            f"  Damping parameter (alpha_b): {self.alpha_b}\n"
            f"  Anisotropy constant (K): {self.K}\n"
            f"  External field (H_ext): {self.H_ext}\n"
            f"  Effective damping (alpha_eff): {self.alpha_eff}\n"
            f"  Effective anisotropy (K_eff): {self.K_eff}\n"
            f"  Effective gyromagnetic ratio (gamma_eff): {self.gamma_eff} rad/(s*T)"
        )


    # ferrimagnetic LLG equation in implicit form: mdot = f(m, mdot)
    def ferri_LLG(self, t, m):
        """Computes the right-hand side of the ferrimagnetic LLG equation."""
        # --- 0. Normalize m ---
        norm_m = np.linalg.norm(m)
        if norm_m < 1e-9: 
            return np.zeros_like(m)
        m_normalized = m / norm_m

        if t > 8e-10:
            J = 0
        else:
            J = self.J

        # Thermal Noise field
        Thermal_coefficient = np.sqrt((2 * self.k_B * self.T * self.alpha_eff) / (self.M * self.gamma_eff * self.v * self.dt)) # tesla
        random_vector = np.random.normal(0, 1, 3)
        normalized_random_vector = random_vector / np.linalg.norm(random_vector)
        H_therm_raw = normalized_random_vector * Thermal_coefficient
        H_therm = H_therm_raw * self.gamma_eff

        # Anisotropy Field (Uniaxial)
        H_ani = np.zeros(3)
        if self.K_eff is not None:
            H_ani = 2 * self.K_u / self.M * (np.dot(m_normalized, self.easy_axis) * self.easy_axis)

        # --- 2. Calculate TOTAL Effective Field ---
        H_eff = (self.H_ext + H_ani)

        coefficient = -self.gamma_eff / (1 + self.alpha_eff**2)
        term1 = np.cross(m, H_eff) * coefficient
        term2 = self.alpha_eff * np.cross(m, term1) * coefficient
       
        print("dmdt:", f" {term1 + term2} count: {self.count}")
        self.count += 1
        return term1 + term2