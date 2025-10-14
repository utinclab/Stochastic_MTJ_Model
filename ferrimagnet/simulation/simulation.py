from devices.base import DeviceBase
import numpy as np
from tqdm import tqdm

class MTJSimulation:
    def __init__(self, device: DeviceBase):
        self.device = device

    def run(self, state_init, j_stt, j_she, flips):
        """Run full time evolution for one flip."""
        state = state_init
        t_pulse_steps = int(self.device.params.t_pulse/self.device.params.t_step)
        t_relax_steps = int(self.device.params.t_relax/self.device.params.t_step)
        total_steps = (t_pulse_steps + t_relax_steps) * flips + 1

        history = {
            "bitstream": np.empty(flips, dtype=np.float64),
            "theta": np.empty(total_steps, dtype=np.float64), 
            "phi": np.empty(total_steps, dtype=np.float64), 
            "energy": np.empty(total_steps, dtype=np.float64), 
            "power": np.empty(total_steps, dtype=np.float64), 
            "R": np.empty(total_steps, dtype=np.float64),
            "J_she": np.empty(total_steps, dtype=np.float64),
            "t": np.arange(0,total_steps*self.device.params.t_step,self.device.params.t_step)
        }

        # first flip
        t = 0
        history["theta"][t] = state[0]
        history["phi"][t] = state[1]
        history["energy"][t] = state[2]
        history["power"][t] = state[3]
        history["R"][t] = state[4]
        history["J_she"][t] = self.device.params.J_she
        t += 1
        for i in tqdm(range(flips),ncols=80,leave=False):
            for _ in range(t_pulse_steps):
                self.device.params.J_she = 0
                self.device.params.J_stt = j_stt
                H_eff = self.device.compute_effective_field(state, self.device.params.v_pulse)
                state = self.device.update_state(state, H_eff, self.device.params.v_pulse)
                history["theta"][t] = state[0]
                history["phi"][t] = state[1]
                history["energy"][t] = state[2]
                history["power"][t] = state[3]
                history["R"][t] = state[4]
                history["J_she"][t] = self.device.params.J_she
                t += 1

            for _ in range(t_relax_steps):
                self.device.params.J_she = j_she
                self.device.params.J_stt = -j_stt
                H_eff = self.device.compute_effective_field(state, self.device.params.vhold)
                state = self.device.update_state(state, H_eff, self.device.params.vhold)
                history["theta"][t] = state[0]
                history["phi"][t] = state[1]
                history["energy"][t] = state[2]
                history["power"][t] = state[3]
                history["R"][t] = state[4]
                history["J_she"][t] = self.device.params.J_she
                t += 1

            history["bitstream"][i] = 1 if np.cos(state[0]) > 0 else -1

        return history