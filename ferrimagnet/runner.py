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

    date = '10282025'
    test_name = 'ferro_bitstream_test2'
    figure_path = f'./figures/{date}/{test_name}/'
    data_folder = f'./figures/{date}/{test_name}/data/'
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(os.path.join(data_folder, "bitstream/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "theta/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "phi/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "energy/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "power/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "R/"), exist_ok=True)

    # Define the device classes and parameter sets to sweep
    device_classes = [ParallelCPUDevice]  # Add other device classes if needed
    # Sweep Alpha and Temperature
    param_sets=[MTJParameters(t_step=1e-15)]
    # for i in range(10):
    #     for j in range(50):
    #         param_sets.append(
    #             FerriParameters(
    #                 alpha_b=(0.001 + i*0.01),
    #                 temperature=(275 + j)
    #             )
    #         )
    sweep = DeviceSweep(device_classes, param_sets, tmp_folder=data_folder)

    # Define the range of j_stt values to sweep
    # j_stt_arr = np.linspace(-1e11,1e11,50)
    j_stt_arr = np.linspace(0,0,1)

    # Run the sweep with specified j_she and number of flips
    j_she = -4e11  # Example spin Hall current density in A/m^2
    flips = 1 # Number of stochastic flips to simulate per j_stt value

    results = sweep.run_all_parallel(j_stt_arr, j_she, flips, True)

    # Plot the results using the Plotter class
    plotter = Plotter(figure_path, results, device_classes, param_sets, j_stt_arr, j_she, flips, data_folder, [])
    # plotter.plot_scurve()
    # plotter.plot_combined_scurve()
    plotter.save_parameters()
    plotter.plot_bittrace()
    plotter.plot_3D_trajectory()
if __name__ == "__main__":
    main()