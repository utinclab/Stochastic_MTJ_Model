from .base import DeviceBase
from parameters.base_params import MTJParameters
import numpy as np
from numba import njit

# -----------------------
# numba-compiled helpers
# -----------------------

@njit
def compute_effective_field_jit(theta, phi,
                                Ki, tf, Ms, u0, ksi, tox,
                                Hx, Hy, Hz, Nx, Ny, Nz,
                                V, Htherm,
                                noise_ax, noise_ay, noise_az, out):
    """
    Return Ax, Ay, Az as a 3-element array.
    noise_* are pre-drawn normal variates (floats).
    """
    Hk = (2.0 * Ki) / (tf * Ms * u0) - (2.0 * ksi * V) / (u0 * Ms * tox * tf)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    Ax = Hx - Nx * Ms * sin_t * cos_p + noise_ax * Htherm
    Ay = Hy - Ny * Ms * sin_t * sin_p + noise_ay * Htherm
    Az = Hz - Nz * Ms * cos_t + Hk * cos_t + noise_az * Htherm

    out[0] = Ax
    out[1] = Ay
    out[2] = Az
    return out

@njit
def update_state_jit(theta, phi, energy,
                     Ax, Ay, Az,
                     gammap, alpha,
                     J_she, F, eta, P, J_stt,
                     t_step,
                     Rp, Vh, TMR, v, R2, A2, A,
                     V):
    """
    Update LLG state and compute R, power, energy_new.
    Returns (theta_new, phi_new, energy_new, power, R).
    """
    alpha2 = 1.0 + alpha * alpha
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    # dtheta (following original expression)
    dtheta = gammap * (
        Ax * (alpha * cos_t * cos_p - sin_p)
        + Ay * (alpha * cos_t * sin_p + cos_p)
        - Az * alpha * sin_t
    ) - J_she * F * eta * (cos_p * cos_t + (alpha * sin_p) / alpha2) \
      + ((F * P * J_stt) * sin_t / alpha2)

    # dphi
    # watch for division by zero -- keep original form
    denom = sin_t if sin_t != 0.0 else 1e-16
    dphi = gammap * (
        Ax * (-cos_t * cos_p - alpha * sin_p)
        + Ay * (-cos_t * sin_p + alpha * cos_p)
        + Az * sin_t
    ) / denom + J_she * F * eta * (sin_p - alpha * cos_p * cos_t) / (denom * alpha2) \
      - ((alpha * F * P * J_stt) / alpha2)

    theta_new = theta + t_step * dtheta
    phi_new = phi + t_step * dphi

    # Resistance R and power, energy
    # Keep original algebra (converted to float math)
    num = 1.0 + (V / Vh) ** 2 + TMR
    denomR = 1.0 + (v / Vh) ** 2 + TMR * (1.0 + (np.sin(theta) * np.cos(phi))) / 2.0
    R = Rp * (num / denomR)

    # power expression using given variables
    power = v * v / R + R2 * (abs(J_she * A2)) ** 2 + R * (J_stt * A) ** 2
    energy_new = energy + power * t_step

    return theta_new, phi_new, energy_new, power, R

# -----------------------
# Class using the compiled helpers
# -----------------------

class ParallelCPUDevice(DeviceBase):
    """Abstract base class for all MTJ-type devices."""

    def __init__(self, params: MTJParameters, device_id=0):
        self.params = params
        self.params.compute_derived()  # Ensure derived parameters are computed
        self.device_id = device_id

    def compute_effective_field(self, state, V):
        """Return effective field components based on device physics."""
        theta, phi, _, _, _ = state
        p = self.params

        # Draw thermal noise in Python and pass as floats to the njit function
        noise_ax = np.random.normal()
        noise_ay = np.random.normal()
        noise_az = np.random.normal()
        tmp_out = np.empty(3, dtype=np.float64)
        field_arr = compute_effective_field_jit(
            theta, phi,
            p.Ki, p.tf, p.Ms, p.u0, p.ksi, p.tox,
            p.Hx, p.Hy, p.Hz, p.Nx, p.Ny, p.Nz,
            V, p.Htherm,
            noise_ax, noise_ay, noise_az, tmp_out
        )

        # return as a small dict-like object to keep original calling code compatible
        H_eff = {'Ax': field_arr[0], 'Ay': field_arr[1], 'Az': field_arr[2]}
        return H_eff

    def update_state(self, state, H_eff, V):
        """Update magnetization state using LLG equation."""
        theta, phi, energy, _, _ = state
        p = self.params

        theta_new, phi_new, energy_new, power, R = update_state_jit(
            theta, phi, energy,
            H_eff['Ax'], H_eff['Ay'], H_eff['Az'],
            p.gammap, p.alpha,
            p.J_she, p.F, p.eta, p.P, p.J_stt,
            p.t_step,
            p.Rp, p.Vh, p.TMR, p.v, p.R2, p.A2, p.A,
            V
        )

        new_state = (theta_new, phi_new, energy_new, power, R)
        return new_state
