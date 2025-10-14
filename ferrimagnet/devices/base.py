from abc import ABC
from parameters.base_params import MTJParameters
import numpy as np

class DeviceBase(ABC):
    """Abstract base class for all MTJ-type devices."""

    def __init__(self, params: MTJParameters, device_id=0):
        self.params = params
        self.params.compute_derived()  # Ensure derived parameters are computed
        self.device_id = device_id

    def compute_effective_field(self, state, V):
        """Return effective field components based on device physics."""
        theta, phi, _, _, _ = state
        p = self.params
        Hk = (2*p.Ki)/(p.tf*p.Ms*p.u0)-(2*p.ksi*V)/(p.u0*p.Ms*p.tox*p.tf)
        H_eff = {
            'Ax': p.Hx-p.Nx*p.Ms*np.sin(theta)*np.cos(phi)+np.random.normal()*p.Htherm,
            'Ay': p.Hy-p.Ny*p.Ms*np.sin(theta)*np.sin(phi)+np.random.normal()*p.Htherm,
            'Az': p.Hz-p.Nz*p.Ms*np.cos(theta)+Hk*np.cos(theta)+np.random.normal()*p.Htherm
        }
        return H_eff

    def update_state(self, state, H_eff, V):
        """Update magnetization state using LLG equation."""
        theta, phi, energy, _, _ = state
        p = self.params

        dtheta = p.gammap*(
            H_eff['Ax']*(p.alpha*np.cos(theta)*np.cos(phi)-np.sin(phi))
            + H_eff['Ay']*(p.alpha*np.cos(theta)*np.sin(phi)+np.cos(phi))
            - H_eff['Az']*p.alpha*np.sin(theta))                                                                              \
            - p.J_she*p.F*p.eta*(np.cos(phi)*np.cos(theta)+(p.alpha*np.sin(phi))/(1+p.alpha*p.alpha))                        \
            + ((p.F*p.P*p.J_stt)*np.sin(theta)/(1+p.alpha*p.alpha))
        
        dphi = p.gammap*(
            H_eff['Ax']*(-np.cos(theta)*np.cos(phi)-p.alpha*np.sin(phi))
            + H_eff['Ay']*(-np.cos(theta)*np.sin(phi)+p.alpha*np.cos(phi))
            + H_eff['Az']*np.sin(theta))/(np.sin(theta))                                                                           \
            + p.J_she*p.F*p.eta*(np.sin(phi)-p.alpha*np.cos(phi)*np.cos(theta))/(np.sin(theta)*(1+p.alpha*p.alpha))     \
            - ((p.alpha*p.F*p.P*p.J_stt)/(1+p.alpha*p.alpha))
        
        theta_new = theta + p.t_step * dtheta
        phi_new = phi + p.t_step * dphi
        
        R = p.Rp*(1+(V/p.Vh)**2+p.TMR)/(1+(p.v/p.Vh)**2+p.TMR*(1+(np.sin(theta)*np.cos(phi)))/2)
        power = p.v**2/R + p.R2*(np.abs(p.J_she*p.A2))**2 + R*(p.J_stt*p.A)**2
        energy_new = energy + power * p.t_step

        new_state = (theta_new, phi_new, energy_new, power, R)
        return new_state                                                                                                        