import numpy as np
from numba import njit

class ferrimagnetic_device():
    # constants
    hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
    e = 1.60217662e-19    # Electron charge (C)
    u0 = 4 * np.pi * 1e-7 # Vacuum permeability (T·m/A)
    k_B = 1.380649e-23    # Boltzmann constant (J/K)

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

        # Effective parameters
        self.alpha_eff = alpha_a if ferro else ((alpha_a * M_a) / gamma_a + (alpha_b * M_b) / gamma_b) / (M_a / gamma_a + M_b / gamma_b)
        self.K_eff = -K if ferro else K - self.u0 * (M_a - M_b) ** 2 / 2
        self.gamma_eff = gamma_a if ferro else (M_a - M_b) / (M_a / gamma_a + M_b / gamma_b)
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
        # print(t, J)
        # External Field
        H_ext_term = -self.u0 * self.gamma_eff * self.H_ext 
        # H_ext_raw = self.H_ext

        # Thermal Noise field
        Thermal_coefficient = np.sqrt((2 * self.k_B * self.T * self.alpha_eff) / (self.M * self.gamma_eff * 10e-18 * self.u0 * self.dt))
        H_therm_raw = np.random.normal(0, 1, 3) * Thermal_coefficient
        H_therm_term = H_therm_raw * self.gamma_eff


        # Anisotropy Field (Uniaxial)
        H_k_term = H_k_raw = np.zeros(3)
        if self.K_eff is not None:
            dE_ani = np.array([0, 0, 2 * m[2] * self.K_eff])
            H_k_term = self.gamma_eff/self.M * dE_ani
            # H_k_raw = dE_ani / (self.u0 * self.M)

        # Cuppling Field (Exchange)
        H_c_term = H_c_raw = np.zeros(3)
        if self.theta_eff is not None:
            coefficient = -self.gamma_eff * (self.p + self.q) * self.hbar * self.theta_eff / 2 / self.e / self.M / self.tf * J
            H_c_term = coefficient * np.cross(np.array([0, 1, 0]), m_normalized)

        # --- 2. Calculate TOTAL Effective Field ---
        H_eff = (H_ext_term + H_k_term + H_c_term + H_therm_term)
        # print("H_ext_field:", H_ext_raw)
        # print("H_k_field:", H_k_raw)
        # print("H_c_field:", H_c_raw)
        # print("H_therm_field:", H_therm_raw)



        precondition = 1 / (1 + self.alpha_eff**2)
        term1 = np.cross(m, H_eff) * precondition
        term2 = self.alpha_eff * np.cross(m, term1) * precondition
       
        print("dmdt:", f" {term1 + term2} count: {self.count}")
        self.count += 1
        return term1 + term2

import numpy as np

class NormalLLG:
    # constants
    hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
    e = 1.60217662e-19    # Electron charge (C)
    u0 = 4 * np.pi * 1e-7 # Vacuum permeability (T·m/A)

    def __init__(self, gamma, alpha, M_s, H_ext=None, 
                 K_u=None, easy_axis=None, 
                 N_d=None, 
                 J=0.0, p_pol=None, free_layer_thickness=None, P_spin_pol=0.0):
        """
        Initialize the LLG equation parameters including common effective field terms.

        Args:
            gamma (float): Gyromagnetic ratio (rad s^-1 T^-1). NOTE: Provide the correct sign! 
                           Typically negative for electrons (~ -1.76e11).
            alpha (float): Gilbert damping coefficient (dimensionless).
            M_s (float): Saturation magnetization (A/m).
            H_ext (array-like, optional): External applied field [Hx, Hy, Hz] (A/m). Defaults to [0,0,0].
            K_u (float, optional): Uniaxial anisotropy constant (J/m^3). Defaults to None (no anisotropy).
                                   Positive K_u means easy_axis is preferred.
            easy_axis (array-like, optional): Unit vector [kx, ky, kz] for uniaxial anisotropy. 
                                             Required if K_u is specified. Defaults to None.
            N_d (array-like, optional): Demagnetizing factors [Nx, Ny, Nz] (diagonal tensor assumed). 
                                        Dimensionless, Nx+Ny+Nz=1. Defaults to None (no demag field).
            J (float, optional): Perpendicular current density (A/m^2). Positive for current flowing 
                                 in +z direction (check convention). Defaults to 0.
            p_pol (array-like, optional): Unit vector [px, py, pz] for spin polarization direction 
                                         (fixed layer magnetization). Required for STT. Defaults to None.
            free_layer_thickness (float, optional): Thickness 't' of the free layer (m). Required for STT. 
                                                   Defaults to None.
            P_spin_pol (float, optional): Spin polarization factor of the current (dimensionless, 0 to 1). 
                                         Required for STT. Defaults to 0.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.M_s = M_s
        
        self.H_ext = np.asarray(H_ext) if H_ext is not None else np.zeros(3)
        
        # Anisotropy parameters
        self.K_u = K_u
        self.easy_axis = np.asarray(easy_axis)/np.linalg.norm(easy_axis) if easy_axis is not None else None
        if self.K_u is not None and self.easy_axis is None:
            raise ValueError("easy_axis must be provided if K_u is specified.")
            
        # Demagnetizing parameters (assuming diagonal tensor)
        self.N_d_diag = np.asarray(N_d) if N_d is not None else None
        if self.N_d_diag is not None and len(self.N_d_diag) != 3:
             raise ValueError("N_d must be a list/array of 3 diagonal factors [Nx, Ny, Nz].")

        # STT parameters
        self.J = J
        self.p_pol = np.asarray(p_pol)/np.linalg.norm(p_pol) if p_pol is not None else None
        self.t = free_layer_thickness
        self.P = P_spin_pol
        
        # Check if STT parameters are consistent
        self.stt_active = self.J != 0 and self.p_pol is not None and self.t is not None and self.P != 0
        if self.J != 0 and not self.stt_active:
             print("Warning: Current density J is non-zero, but other STT parameters (p_pol, t, P) are missing or zero. STT will be inactive.")
        
        # Pre-calculate STT coefficient beta if active (check formula for your specific model!)
        # beta = (hbar / (2 * e * M_s * t)) * P 
        self.stt_beta = 0.0
        if self.stt_active:
             if self.M_s <= 0 or self.t <=0:
                  raise ValueError("M_s and free_layer_thickness must be positive for STT.")
             self.stt_beta = (self.hbar / (2 * self.e * self.M_s * self.t)) * self.P
             print(f"STT active. Beta factor: {self.stt_beta:.2e}")
    
    def llg(self, t, m):
        """
        Computes the right-hand side of the LLG equation, including H_k, H_d, and STT terms.
        Uses the corrected damping term and handles gamma sign.
        """
        m = np.asarray(m) 
        
        # --- 0. Normalize m ---
        norm_m = np.linalg.norm(m)
        if norm_m < 1e-9: 
            return np.zeros_like(m)
        m_normalized = m / norm_m
        
        # External Field
        H_ext_term = self.H_ext 

        # Anisotropy Field (Uniaxial)
        H_k_term = np.zeros(3)
        if self.K_u is not None and self.easy_axis is not None:
            # H_k = (2*Ku / (mu0*Ms)) * (m . k) * k
            # Ensure M_s is positive to avoid division errors
            if self.M_s > 0:
                 H_k_magnitude = (2 * self.K_u) / (self.u0 * self.M_s)
                 m_dot_k = np.dot(m_normalized, self.easy_axis)
                 H_k_term = H_k_magnitude * m_dot_k * self.easy_axis
            else:
                 # Handle M_s=0 case if necessary, though unlikely physically
                 pass 

        # Demagnetizing Field (Diagonal Tensor)
        H_d_term = np.zeros(3)
        if self.N_d_diag is not None:
             # H_d = - N_d * M = - diag(Nx,Ny,Nz) * Ms * m_normalized
             # Element-wise multiplication for diagonal tensor
             H_d_term = - self.N_d_diag * self.M_s * m_normalized

        # --- 2. Calculate TOTAL Effective Field ---
        H_eff = (H_ext_term + H_k_term + H_d_term)
        
        # --- 3. Check for Trivial Cases ---
        if np.linalg.norm(H_eff) < 1e-14: # Use a small tolerance
            # If H_eff is zero, only STT might act (if present)
            pass # Continue to STT calculation below

        m_cross_Heff = np.cross(m_normalized, H_eff)
    
        gamma_eff = self.gamma / (1 + self.alpha**2)

        # Precessional Torque Term (using TOTAL H_eff)
        term1_llg = - gamma_eff * self.u0 * m_cross_Heff

        # Damping Torque Term (Corrected - NO '/ M_s')
        term2_llg = - gamma_eff * self.u0 * self.alpha * np.cross(m_normalized, m_cross_Heff)

        # --- 6. Combine all terms for dm/dt ---
        dmdt = term1_llg + term2_llg

        return dmdt
