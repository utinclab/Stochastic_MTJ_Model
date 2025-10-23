# parameters/ferri_params.py
from dataclasses import dataclass, field
import numpy as np
from .base_params import MTJParameters
from utils.utils import MaterialUtils

@dataclass
class FerriParameters(MTJParameters):
    # Ferrimagnetic-specific material parameters
    M_a0: float = 9.3e5
    M_b0: float = 4.5e5
    Tc_a: float = 424
    Tc_b: float = 424
    beta_a: float = 0.8
    beta_b: float = 0.21
    alpha_a: float = 0.072
    alpha_b: float = 0.078
    gamma_a: float = 5.35e6
    gamma_a2: float = 0.864e6
    Ki_base: float = 3.8e4  # Base interfacial anisotropy

    # faster pulses for ferri
    t_pulse: float = 0.8e-9
    t_relax: float = 0.2e-9
    t_step: float = 1e-12

    # geometry changes
    tf: float = 1e-9

    # Override Ms to be computed
    Ms: float = field(init=False)
    gamma: float = field(init=False)
    gamma_b: float = field(init=False)
    alpha: float = field(init=False)
    gammap: float = field(init=False)
    F: float = field(init=False)
    Htherm: float = field(init=False)
    P: float = field(init=False)
    Ki: float = field(init=False)

    def compute_derived(self):
        """Compute derived fields for ferrimagnetic MTJ."""
        super().compute_derived()  # compute geometry, A, A2, R2, etc.
        # recompute ferri-specific quantities
        self.M_a = MaterialUtils.Ms_T_dependence(self.temperature, self.beta_a, self.Tc_a, self.M_a0)
        self.M_b = MaterialUtils.Ms_T_dependence(self.temperature, self.beta_b, self.Tc_b, self.M_b0)
        self.P = 0.6 if self.M_a > self.M_b else -0.6
        self.Ms = self.M_a - self.M_b
        self.gamma = (self.M_a - self.M_b) / (self.M_a/self.gamma_a - self.M_b/self.gamma_a2)
        self.gamma_b = self.gamma / self.u0
        self.alpha = (self.alpha_a * self.M_a / self.gamma_a + self.alpha_b * self.M_b / self.gamma_a2) / \
                    (self.M_a/self.gamma_a - self.M_b/self.gamma_a2)
        self.Bsat = self.Ms * self.u0
        self.gammap = self.gamma / (1 + self.alpha**2)
        self.Ki = self.Ki_base * self.tf - self.tf * self.u0 * self.Ms**2 / 2
        self.Htherm = np.sqrt((2 * self.u0 * self.alpha * self.kb * self.temperature) / 
                            (self.Bsat * self.gamma_b * self.t_step * self.v)) / self.u0
        self.F = (self.gamma * self.h_bar) / (2 * self.u0 * self.e * self.tf * self.Ms)

    def to_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }