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
    "H_ext": np.array([0.01, 0, 0]),  # Tesla

    # time step for integration
    "dt": 5e-15,            # seconds

    # temperature
    "T": 300,               # Kelvin
}

# Dictionary to define all constants
pure_precess = {
    "tf": 2.0e-9,           # Film thickness (m)
    "J": 0,            # Spin current density (A/m²)
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
    "alpha_a": 0,
    "alpha_b": 0,

    # Anisotropy constant (uniaxial)
    "K": 0,             # J/m³

    # External field
    "H_ext": np.array([0.0, 0, 10000]),  # Tesla

    # time step for integration
    "dt": 1e-12,            # seconds

    # temperature
    "T": 300,               # Kelvin
}

# Dictionary to define all constants
pure_damp = {
    "tf": 2.0e-9,           # Film thickness (m)
    "J": 1,            # Spin current density (A/m²)
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
    "alpha_a": 0.005,
    "alpha_b": 0,

    # Anisotropy constant (uniaxial)
    "K": 0,             # J/m³

    # External field
    "H_ext": np.array([0.0, 0, 10000]),  # Tesla

        # time step for integration
    "dt": 1e-12,            # seconds

    # temperature
    "T": 300,               # Kelvin
}

# Dictionary to define all constants
thermal_noise_test = {
    "tf": 2.0e-9,           # Film thickness (m)
    "J": 0,            # Spin current density (A/m²)
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
    "H_ext": np.array([0.01, 0, 0]),  # Tesla

    # time step for integration
    "dt": 1e-12,            # seconds

    # temperature
    "T": 300,               # Kelvin
}

# Dictionary to define all constants
ferromagnet_test = {
    "tf": 1.1e-9,           # Film thickness (m)
    "J": 0,            # Spin current density (A/m²)
    "theta_eff": 0.1,       # Spin-Hall angle (dimensionless)
    "gamma_a": 1.76e11,     # Gyromagnetic ratio for Fe sublattice (rad/(s·T))
    "gamma_b": 1.76e11,    # Gyromagnetic ratio for Gd sublattice (rad/(s·T))
    "c": 0,               # Exchange coupling (A/m)
    "p": 0.5,               # Parameter p
    "q": 0.5,               # Parameter q

    # Sublattice magnetizations
    "M_a": 1.2e6,           # A/m (FeCo)
    "M_b": 1.2e6,           # A/m (Gd)

    # Damping parameters
    "alpha_a": 0.03,
    "alpha_b": 0.03,

    # Anisotropy constant (uniaxial)
    "K": 4.0e5,             # J/m³

    # External field
    "H_ext": np.array([0.01, 0, 0]),  # A/m

    # time step for integration
    "dt": 5e-15,            # seconds

    # temperature
    "T": 150,               # Kelvin
}