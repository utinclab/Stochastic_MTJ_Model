from sweeps.sweep import DeviceSweep
from devices.parallel_cpu import ParallelCPUDevice
from utils.utils import MaterialUtils
from parameters.ferri_params import FerriParameters, MTJParameters
import numpy as np
from graphing.plotter import Plotter
import time, atexit
import os
import shutil

def main():
    start_time = time.perf_counter()
    def _print_elapsed():
        elapsed = time.perf_counter() - start_time
        print(f"Total execution time: {elapsed:.4f} s")
    atexit.register(_print_elapsed)

    date = '11122025'
    test_name = 'LUT_test_325_1'
    figure_path = f'./LUT/{date}/{test_name}/'
    data_folder = f'./LUT/{date}/{test_name}/data/'
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(os.path.join(data_folder, "bitstream/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "theta/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "phi/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "energy/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "power/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "R/"), exist_ok=True)

    device_classes = [ParallelCPUDevice]  # Add other device classes if needed
    param_sets=[FerriParameters(t_step=1e-12, temperature=325, t_pulse=0.8e-9, t_relax=0.5e-9)]
    sweep = DeviceSweep(device_classes, param_sets, tmp_folder=data_folder)

    j_stt_arr = np.linspace(-1e11,1e11,100)
    j_she = -4e11  # Example spin Hall current density in A/m^2
    flips = 1000 # Number of stochastic flips to simulate per j_stt value

    results = sweep.run_all_parallel(j_stt_arr, j_she, flips, False)

    # Plot the results using the Plotter class
    plotter = Plotter(figure_path, results, device_classes, param_sets, j_stt_arr, j_she, flips, data_folder, [])
    plotter.plot_scurve()
    plotter.save_parameters()
    plotter.save_bitstream_averages()
if __name__ == "__main__":
    main()