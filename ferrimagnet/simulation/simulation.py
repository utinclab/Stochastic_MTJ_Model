from devices.base import DeviceBase
import numpy as np
from tqdm import tqdm
import os

class MTJSimulation:
    def __init__(self, device: DeviceBase, tmp_dir: str = "./tmp"):
        self.device = device
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    def run(self, state_init, j_stt, j_she, flips, chunk_size=10000):
        """Stream results to disk instead of keeping all in memory."""
        state = state_init
        t_pulse_steps = int(self.device.params.t_pulse/self.device.params.t_step)
        t_relax_steps = int(self.device.params.t_relax/self.device.params.t_step)
        total_steps = (t_pulse_steps + t_relax_steps) * flips + 1

        # Temporary binary files for streaming arrays
        arrays = ["theta", "phi", "energy", "power", "R", "J_she"]
        tmp_files = {k: os.path.join(self.tmp_dir, f"{k}_{os.getpid()}.npy") for k in arrays}
        buffers = {k: [] for k in arrays}

        # Write bitstream separately
        bitstream = np.empty(flips, dtype=np.float64)

        t_step = self.device.params.t_step
        t_array = np.arange(0, total_steps * t_step, t_step)

        # Initialize state
        t = 0
        current = [state[0], state[1], state[2], state[3], state[4], self.device.params.J_she]

        for k, val in zip(arrays, current):
            buffers[k].append(val)

        t += 1
        chunk_idx = 0

        for i in tqdm(range(flips), ncols=80, leave=False):
            for _ in range(t_pulse_steps):
                self.device.params.J_she = 0
                self.device.params.J_stt = j_stt
                H_eff = self.device.compute_effective_field(state, self.device.params.v_pulse)
                state = self.device.update_state(state, H_eff, self.device.params.v_pulse)
                current = [state[0], state[1], state[2], state[3], state[4], self.device.params.J_she]
                for k, val in zip(arrays, current):
                    buffers[k].append(val)
                t += 1

                # Stream to disk periodically
                if t % chunk_size == 0:
                    self._flush_to_disk(buffers, tmp_files, chunk_idx)
                    chunk_idx += 1
                    for k in arrays:
                        buffers[k] = []

            for _ in range(t_relax_steps):
                self.device.params.J_she = j_she
                self.device.params.J_stt = -j_stt
                H_eff = self.device.compute_effective_field(state, self.device.params.vhold)
                state = self.device.update_state(state, H_eff, self.device.params.vhold)
                current = [state[0], state[1], state[2], state[3], state[4], self.device.params.J_she]
                for k, val in zip(arrays, current):
                    buffers[k].append(val)
                t += 1
                if t % chunk_size == 0:
                    self._flush_to_disk(buffers, tmp_files, chunk_idx)
                    chunk_idx += 1
                    for k in arrays:
                        buffers[k] = []

            bitstream[i] = 1 if np.cos(state[0]) > 0 else -1

        # Flush remaining data
        self._flush_to_disk(buffers, tmp_files, chunk_idx)

        return {
            "tmp_files": tmp_files,
            "bitstream": bitstream,
            "t": t_array,
            "total_steps": total_steps
        }
    

    def _flush_to_disk(self, buffers, tmp_files, chunk_idx):
        """Append buffered data to .npy files incrementally."""
        for key, buf in buffers.items():
            arr = np.array(buf, dtype=np.float64)
            path = tmp_files[key]
            if os.path.exists(path):
                # Append mode
                with open(path, "ab") as f:
                    np.save(f, arr)
            else:
                # First write
                with open(path, "wb") as f:
                    np.save(f, arr)