# parameters/base_params.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MTJParameters:
    # Physical constants
    uB: float    = 9.274e-24        # Bohr magneton (J/T)
    h_bar: float = 1.054e-34        # Reduced Planck constant (J*s)
    u0: float    = np.pi * 4e-7     # Permeability of free space (T*m/A)
    e: float     = 1.6e-19          # Elementary charge (C)
    kb: float    = 1.38e-23         # Boltzmann constant (J/K)

    # Simulation settings
    t_step: float   = 5e-11         # Time step (s)
    v_pulse: float  = 0             # Voltage during pulse (V)
    t_pulse: float  = 50e-9         # Pulse duration (s)
    t_relax: float  = 50e-9         # Relaxation time after pulse (s)
    Happl: float    = 0             # Applied field (A/m)
    Hshe: float     = 0             # Spin Hall field (A/m)
    J_stt: float    = 0             # Spin transfer torque current density (A/m^2)
    J_she: float    = 0             # Spin Hall current density (A/m^2)
    vhold: float    = 0             # Voltage during hold (V)
    temperature: float  = 300       # Temperature (K)

    # MTJ geometry
    a: float    = 50e-9
    b: float    = 50e-9
    tf: float   = 1.1e-9
    tox: float  = 1.5e-9
    alphar: float = 10e-30

    # VCMA / SHE
    ksi: float  = 75e-15
    Vh: float   = 0.5
    TMR: float  = 1.2
    Rp: float   = 5e3
    eta: float  = 0.3
    w: float    = 100e-9
    l: float    = 100e-9
    d: float    = 3e-9
    rho: float  = 200e-8

    # Damping / spin
    alpha: float = 0.03
    gamma: float = field(init=False)
    gammap: float = field(init=False)
    F: float    = field(init=False)
    gamma_b: float = field(init=False)
    Ki: float   = 1.0056364e-3
    Ms: float   = 1.2e6
    P : float   = 0.6

    # Derived geometry fields
    v: float    = field(init=False)
    A: float    = field(init=False)
    A2: float   = field(init=False)
    R2: float   = field(init=False)

    # Field parameters
    Htherm: float = field(init=False)
    Hx: float = 0
    Hy: float = field(init=False)
    Hz: float = field(init=False)

    # Demagnetization
    Nx: float   = 0.010613177892974
    Ny: float   = 0.010613177892974
    Nz: float   = 0.978773644214052

    def compute_derived(self):
        """Compute all derived parameters for a normal MTJ."""
        self.gamma = 2*self.u0*self.uB/self.h_bar
        self.gamma_b = self.gamma / self.u0
        self.Bsat = self.Ms * self.u0
        self.gammap = self.gamma / (1 + self.alpha**2)
        self.v = self.tf * np.pi * self.b * self.a / 4
        self.A = self.a * self.b * np.pi / 4
        self.A2 = self.d * self.w
        self.R2 = self.rho * self.l / (self.w * self.d)
        self.Hy = self.Hshe
        self.Hz = self.Happl
        self.Htherm = np.sqrt(
            (2 * self.u0 * self.alpha * self.kb * self.temperature) / 
            (self.Bsat * self.gamma_b * self.t_step * self.v)
        ) / self.u0
        self.F = (self.gamma * self.h_bar) / (2 * self.u0 * self.e * self.tf * self.Ms)

    def print_params(self):
            """Print all parameters for ferrimagnetic MTJ."""
            print("Ferrimagnetic MTJ Parameters:")
            for field_name, field_def in self.__dataclass_fields__.items():
                value = getattr(self, field_name)
                if isinstance(value, float):
                    print(f"  {field_name}: {value:.4e}")
                else:
                    print(f"  {field_name}: {value}")
    
    def to_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }