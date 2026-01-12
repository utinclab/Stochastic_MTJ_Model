import os
import numpy as np
import multiprocessing
from simulation.simulation import MTJSimulation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_and_save(task):
    """Worker function that runs simulation and writes result directly to disk."""
    self_ref, dev_class, ps, j_stt_arr, j_she, flips, index, save_data = task
    result = self_ref.run_one((dev_class, ps, j_stt_arr, j_she, flips, index, save_data))
    return result


class DeviceSweep:
    def __init__(self, device_classes, param_sets, tmp_folder='./tmp/'):
        self.device_classes = device_classes
        self.param_sets = param_sets
        self.tmp_folder = tmp_folder


    def run_one(self, args):
        device_class, param_set, j_stt_arr, j_she, flips, index, save_data = args
        bitstream_averages = []
        results_files = []
        for x, j_stt in enumerate(j_stt_arr):
            dev = device_class(param_set)
            sim = MTJSimulation(dev, tmp_dir=self.tmp_folder)
            initial_state = (np.pi/2, 0, 0, 0, 0)

            # Run and stream to disk
            bitstream_average = sim.run(state_init=initial_state, j_she=j_she, j_stt=j_stt, flips=flips, index=index, x=x, save_simulation_data=save_data)
            bitstream_averages.append(bitstream_average)
        if save_data:
            np.save(os.path.join(self.tmp_folder, f"bitstream_avg/bitstream_avg_{index}.npy"), bitstream_average)
            np.save(os.path.join(self.tmp_folder, f"j_stt/j_stt_{index}.npy"), j_stt_arr)
        return {index: bitstream_averages}


    def run_all_parallel(self, j_stt_arr, j_she, flips, save_data=True):
        os.makedirs(self.tmp_folder, exist_ok=True)

        tasks = [
            (self, dev_class, ps, j_stt_arr, j_she, flips, tuple([i,j]), save_data)
            for i, dev_class in enumerate(self.device_classes)
            for j, ps in enumerate(self.param_sets)
        ]

        max_workers = 6
        results = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, t in enumerate(tasks):
                futures.append(executor.submit(_run_and_save, t))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running tasks"):
                meta = future.result()
                results.update(meta)
        return results


    def run_all(self, j_stt_arr, j_she, flips, save_data=True):
        tasks = [
            (self, dev_class, ps, j_stt_arr, j_she, flips, tuple([i,j]), save_data)
            for i, dev_class in enumerate(self.device_classes)
            for j, ps in enumerate(self.param_sets)
        ]

        results = {}
        for t in tasks:
            result = self.run_one(t)
            results.update(result)
        return results