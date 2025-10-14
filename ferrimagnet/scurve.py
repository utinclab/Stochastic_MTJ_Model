from sweeps.sweep import DeviceSweep
from devices.base import DeviceBase
from devices.parallel_cpu import ParallelCPUDevice
from parameters.ferri_params import FerriParameters, MTJParameters
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec

def main():
    # Define the device classes and parameter sets to sweep
    device_classes = [ParallelCPUDevice]  # Add other device classes if needed
    param_sets = [
        FerriParameters(temperature=280),
        FerriParameters(temperature=320),
        MTJParameters(temperature=280)
    ]

    # Create a DeviceSweep instance
    sweep = DeviceSweep(device_classes, param_sets)

    # Define the range of j_stt values to sweep
    j_stt_arr = np.linspace(-1e11,1e11,10)

    # Run the sweep with specified j_she and number of flips
    j_she = -4e11  # Example spin Hall current density in A/m^2
    flips = 1000  # Number of stochastic flips to simulate per j_stt value

    results = sweep.run_all(j_stt_arr, j_she, flips)

    # save results as a json file
    for res in results:
        for sim_out in res["simulation_output"]:

            device = sim_out["device"]
            params = sim_out["params"]
            j_stt = sim_out["j_stt"]
            result_data = sim_out["result"]

            mx = np.sin(result_data["theta"]) * np.cos(result_data["phi"])
            my = np.sin(result_data["theta"]) * np.sin(result_data["phi"])
            mz = np.cos(result_data["theta"])
            # print(len(mz))
            # print(len(result_data["t"]))
            
            """ 2D bittrace plot """
            # plt.figure(figsize=(6, 6))
            # gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)  # Two rows, shared x-axis

            # # Top plot (new y-axis, same x)
            # ax1 = plt.subplot(gs[0])
            # ax1.plot(result_data["t"] * 1e9, result_data["J_she"], color='tab:orange')
            # ax1.set_ylabel('SOT current (A/m^2)', fontsize=15)
            # ax1.tick_params(labelbottom=False)  # Hide x-axis labels here

            # # Bottom plot (original plot)
            # ax2 = plt.subplot(gs[1], sharex=ax1)
            # ax2.plot(result_data["t"] * 1e9, mz, color='tab:blue')
            # ax2.set_xlabel('Time (ns)', fontsize=15)
            # ax2.set_ylabel('Mz', fontsize=15)
            # plt.show()
            # plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(res["j_stt_arr"], res["bitstream_averages"], marker='o', linestyle='-', color='tab:blue')
        plt.xlabel('STT bias current (A/m^2)', fontsize=15)
        plt.ylabel('bitstream average', fontsize=15)
        plt.title(f'S-Curve', fontsize=15)
        plt.grid()
        plt.show()
        plt.close() 

        # Save the S-curve data
        np.save(f'ferri_s_J_stt', res["j_stt_arr"])
        np.save(f'ferri_s_bitstr_avg', res["bitstream_averages"])
if __name__ == "__main__":
    main()