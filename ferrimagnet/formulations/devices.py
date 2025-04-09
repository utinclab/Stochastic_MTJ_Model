import numpy as np

# Dictionary to define all constants
GdFeCo_MTJ = {
    "tf": 2.0e-9,           # Film thickness (m)
    "J": 5.0e10,            # Spin current density (A/m²)
    "theta_eff": 0.1,       # Spin-Hall angle (dimensionless)
    "gamma_a": 1.76e11,     # Gyromagnetic ratio for Fe sublattice (rad/(s·T))
    "gamma_b": -1.31e11,    # Gyromagnetic ratio for Gd sublattice (rad/(s·T))
    "c": 0.5,               # Exchange coupling (A/m)
    "p": 0.5,               # Parameter p
    "q": 0.5,               # Parameter q

    # Sublattice magnetizations
    "M_a": 1.0e6,           # A/m (FeCo)
    "M_b": 2.0e5,           # A/m (Gd)

    # Damping parameters
    "alpha_a": 0.015,
    "alpha_b": 0.008,

    # Anisotropy constant (uniaxial)
    "K": 3.0e5,             # J/m³

    # External field
    "H_ext": np.array([0.01, 0, 0])  # Tesla
}

GPT_MTJ = {
    "gamma": 2.21e5,           # Gyromagnetic ratio (in CGS units)
    "alpha": 0.01,             # Damping coefficient
    "H_ext": np.array([0, 0, 1]),  # External magnetic field (along the z-axis)
    "M_s": 1.0e6,            # Saturation magnetization (A/m)
}