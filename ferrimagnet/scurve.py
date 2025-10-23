from sweeps.sweep import DeviceSweep
from devices.parallel_cpu import ParallelCPUDevice
from utils.utils import MaterialUtils
from parameters.ferri_params import FerriParameters, MTJParameters
import numpy as np
from graphing.plotter import Plotter
import time, atexit
import shutil

def main():
    start_time = time.perf_counter()
    def _print_elapsed():
        elapsed = time.perf_counter() - start_time
        print(f"Total execution time: {elapsed:.4f} s")
    atexit.register(_print_elapsed)

    # empty tmp folder
    tmp_folder = './tmp/'
    shutil.rmtree(tmp_folder, ignore_errors=True)

    # Define the device classes and parameter sets to sweep
    device_classes = [ParallelCPUDevice]  # Add other device classes if needed


    # Sweep Alpha and Temperature
    param_sets=[]
    for i in range(10):
        for j in range(1):
            param_sets.append(
                FerriParameters(
                    alpha_a=(0.050 + i*0.001),
                    temperature=(275 + j)
                )
            )
    sweep = DeviceSweep(device_classes, param_sets, tmp_folder='./tmp/')

    # Define the range of j_stt values to sweep
    j_stt_arr = np.linspace(-1e11,1e11,21)

    # Run the sweep with specified j_she and number of flips
    j_she = -4e11  # Example spin Hall current density in A/m^2
    flips = 100  # Number of stochastic flips to simulate per j_stt value

    tmp_files = sweep.run_all_parallel(j_stt_arr, j_she, flips)
    results = MaterialUtils.load_all_results(tmp_folder='./tmp/')
    print(f"Loaded {len(results)} results from sweep.")
    print(results[0].keys())
    print(results[0]['simulation_output'][0].keys())

    # Plot the results using the Plotter class
    plotter = Plotter(figure_path='./figures/10222025/alpha_a_sweep2/', results=results, sweep_variables=["temperature", "alpha_a"])
    reduced_plotter = Plotter(figure_path='./figures/10222025/alpha_a_sweep2/', results=results, sweep_variables=["temperature", "alpha_a"])
    plotter.plot_scurve()
    plotter.save_parameters()
    reduced_plotter.plot_bittrace()
if __name__ == "__main__":
    main()