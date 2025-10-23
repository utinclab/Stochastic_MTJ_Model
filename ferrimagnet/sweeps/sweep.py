import os
import pickle
import numpy as np
import multiprocessing
from simulation.simulation import MTJSimulation
from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_and_save(task, save_path):
    """Worker function that runs simulation and writes result directly to disk."""
    self_ref, dev_class, ps, j_stt_arr, j_she, flips = task
    result = self_ref.run_one((dev_class, ps, j_stt_arr, j_she, flips))
    # result contains theta, phi, etc.
    with open(save_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    del result
    return {"file": save_path, "status": "ok"}


def load_all_chunks(path):
    arrays = []
    with open(path, "rb") as f:
        while True:
            try:
                arrays.append(np.load(f))
            except EOFError:
                break
    return np.concatenate(arrays)


def reconstruct_history(tmp_files, bitstream, t):
    """Load streamed .npy files and merge into a single dict."""
    tmp_files = tmp_files
    data = {}
    for key, path in tmp_files.items():
        data[key] = load_all_chunks(path)
    data["bitstream"] = bitstream
    data["t"] = t
    return data


class DeviceSweep:
    def __init__(self, device_classes, param_sets, tmp_folder='./tmp/'):
        self.device_classes = device_classes
        self.param_sets = param_sets
        self.tmp_folder = tmp_folder


    def run_one(self, args):
        device_class, param_set, j_stt_arr, j_she, flips = args
        bitstream_averages = []
        sim_out = []

        for j_stt in j_stt_arr:
            dev = device_class(param_set)
            sim = MTJSimulation(dev, tmp_dir=self.tmp_folder)
            initial_state = (np.pi/2, 0, 0, 0, 0)

            # Run and stream to disk
            stream_info = sim.run(state_init=initial_state, j_she=j_she, j_stt=j_stt, flips=flips)

            # Reconstruct only when needed
            result = reconstruct_history(stream_info["tmp_files"], stream_info["bitstream"], stream_info["t"])
            sim_out.append({
                "device": device_class.__name__,
                "params": param_set,
                "j_stt": j_stt,
                "result": result
            })
            bitstream_averages.append(np.mean(result["bitstream"]))

        return {"simulation_output": sim_out, "bitstream_averages": bitstream_averages, "j_stt_arr": j_stt_arr}


    def run_all_parallel(self, j_stt_arr, j_she, flips):
        os.makedirs(self.tmp_folder, exist_ok=True)

        tasks = [
            (self, dev_class, ps, j_stt_arr, j_she, flips)
            for dev_class in self.device_classes
            for ps in self.param_sets
        ]

        max_workers = max(1, multiprocessing.cpu_count() - 2)
        temp_files = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, t in enumerate(tasks):
                temp_path = os.path.join(self.tmp_folder, f"result_{i}.pkl")
                futures.append(executor.submit(_run_and_save, t, temp_path))

            for future in as_completed(futures):
                meta = future.result()
                temp_files.append(meta["file"])

        # You can return just the file paths (to reload later)
        return temp_files


    def run_all(self, j_stt_arr, j_she, flips):
        tasks = [
            (dev_class, ps, j_stt_arr, j_she, flips)
            for dev_class in self.device_classes
            for ps in self.param_sets
        ]

        results = []
        for t in tasks:
            result = self.run_one(t)
            results.append(result)
        return results