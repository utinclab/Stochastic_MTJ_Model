from devices.base import DeviceBase
import numpy as np
import os

class MTJSimulation:
    def __init__(self, device: DeviceBase, tmp_dir: str = "./tmp"):
        self.device = device
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    def run(self, state_init, j_stt, j_she, flips, index, x, save_simulation_data=False):
        """
        Run the full MTJ simulation and save results into a single file.
        The file is named 'sim_i_j.npz' where index = (i, j).
        """
        i, j = index
        os.makedirs(self.tmp_dir, exist_ok=True)

        state = state_init
        params = self.device.params
        t_pulse_steps = int(params.t_pulse / params.t_step)
        t_relax_steps = int(params.t_relax / params.t_step)

        arrays = ["theta", "phi", "energy", "power", "R"]
        results = {k: [] for k in arrays}
        bitstream = np.empty(flips, dtype=np.float64)

        # Initialize state
        for k, val in zip(arrays, state[:6]):
            results[k].append(val)

        for flip_idx in range(flips):
            # Pulse period
            for _ in range(t_pulse_steps):
                params.J_she = 0
                params.J_stt = j_stt
                H_eff = self.device.compute_effective_field(state, params.v_pulse)
                state = self.device.update_state(state, H_eff, params.v_pulse)
                current = [state[0], state[1], state[2], state[3], state[4]]
                for k, val in zip(arrays, current):
                    results[k].append(val)

            # Relaxation period
            for _ in range(t_relax_steps):
                params.J_she = j_she
                params.J_stt = -j_stt
                H_eff = self.device.compute_effective_field(state, params.vhold)
                state = self.device.update_state(state, H_eff, params.vhold)
                current = [state[0], state[1], state[2], state[3], state[4]]
                for k, val in zip(arrays, current):
                    results[k].append(val)

            bitstream[flip_idx] = 1 if np.cos(state[0]) > 0 else -1

        # Convert to numpy arrays
        for k in arrays:
            results[k] = np.array(results[k], dtype=np.float64)

        if save_simulation_data:
            # Save each array separately
            for k, arr in results.items():
                np.save(os.path.join(self.tmp_dir, f"{k}/{k}_{i}_{j}_{x}.npy"), arr)

            # Also save bitstream
            np.save(os.path.join(self.tmp_dir, f"bitstream/bitstream_{i}_{j}_{x}.npy"), bitstream)

        return np.mean(bitstream)