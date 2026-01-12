import numpy as np
import os
import matplotlib.pyplot as plt
from parameters.base_params import MTJParameters
from devices.base import DeviceBase
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.gridspec import GridSpec
import json
from collections import defaultdict

class Plotter:
    def __init__(self, figure_path:str, results:dict, device_classes:list[DeviceBase], param_sets:list[MTJParameters], j_stt_arr:list[float], j_she, flips:int, tmp_folder:str, sweep_variables:list[str]=["temperature"]):
        self.figure_path = figure_path
        os.makedirs(self.figure_path, exist_ok=True)
        self.results = results
        self.sweep_variables = sweep_variables
        self.device_classes = device_classes
        self.param_sets = param_sets
        self.j_stt_arr = j_stt_arr
        self.flips = flips
        self.tmp_folder = tmp_folder
        self.j_she = j_she


    def set_title_and_filename(self, params, title_prefix, filename_prefix, j_stt, index, show_values=True, sweep_vars=None):
        title_parts = []
        filename_parts = []
        i, j, k = index
        if sweep_vars is None:
            sweep_vars = self.sweep_variables
        for var in sweep_vars:
            if show_values:
                title_parts.append(f"{var.capitalize()}: {getattr(params, var)}")
                filename_parts.append(f"{var}_{getattr(params, var)}")
            else:
                title_parts.append(f"{var.capitalize()}")
                filename_parts.append(f"{var}")
        title = title_prefix + f"J_STT: {j_stt:.3f}, " + ", ".join(title_parts)
        filename = filename_prefix + f"{i}_{j}_{k}_" + "_".join(filename_parts) + ".png"
        return title, filename


    def plot_bittrace(self):
        os.makedirs(os.path.join(self.figure_path, "bittraces/"), exist_ok=True)
        for i, _ in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                t_pulse_steps = int(params.t_pulse / params.t_step)
                t_relax_steps = int(params.t_relax / params.t_step)
                t = np.arange(0,(t_pulse_steps + t_relax_steps) * self.flips + 1)
                j_she = np.tile(
                    np.concatenate([
                        np.zeros(t_pulse_steps, dtype=np.float64), 
                        np.full(t_relax_steps, self.j_she, dtype=np.float64)
                    ]),
                    self.flips
                )
                j_she = np.insert(j_she, 0, 0.0)
                for k, j_stt in enumerate(self.j_stt_arr):
                    theta = np.load(os.path.join(self.tmp_folder, f"theta/theta_{i}_{j}_{k}.npy"))
                    mz = np.cos(theta)
                    
                    
                    title, filename = self.set_title_and_filename(params, title_prefix="Mz Transience - ", filename_prefix="bittrace_", j_stt=j_stt, index = (i,j,k), show_values=True)

                    plt.figure(figsize=(6, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)  # Two rows, shared x-axis

                    # Top plot (new y-axis, same x)
                    ax1 = plt.subplot(gs[0])
                    ax1.plot(t * 1e9, j_she, color='tab:orange')
                    ax1.set_ylabel('SOT current (A/m^2)', fontsize=15)
                    ax1.tick_params(labelbottom=False)  # Hide x-axis labels here
                    ax1.set_title(title, fontsize=12)

                    # Bottom plot (original plot)
                    ax2 = plt.subplot(gs[1], sharex=ax1)
                    ax2.plot(t * 1e9, mz, color='tab:blue')
                    ax2.set_xlabel('Time (ns)', fontsize=15)
                    ax2.set_ylabel('Mz', fontsize=15)
                    plt.savefig(os.path.join(self.figure_path, "bittraces/", filename), dpi=300, bbox_inches='tight')
                    plt.close()


    def plot_3D_trajectory(self):
        os.makedirs(os.path.join(self.figure_path, "trajectories/"), exist_ok=True)
        for i, _ in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                t_pulse_steps = int(params.t_pulse / params.t_step)
                t_relax_steps = int(params.t_relax / params.t_step)
                t = np.arange(0,(t_pulse_steps + t_relax_steps) * self.flips + 1)
                for k, j_stt in enumerate(self.j_stt_arr):
                    title, filename = self.set_title_and_filename(params, title_prefix="Magnetization Trajectory - ", filename_prefix="mtj_trajectory_", j_stt=j_stt, index = (i,j,k), show_values=True)
                    phi = np.load(os.path.join(self.tmp_folder, f"phi/phi_{i}_{j}_{k}.npy"))
                    theta = np.load(os.path.join(self.tmp_folder, f"theta/theta_{i}_{j}_{k}.npy"))

                    mx = np.sin(theta) * np.cos(phi)
                    my = np.sin(theta) * np.sin(phi)
                    mz = np.cos(theta)
                    t = np.arange(0,len(mz)*params.t_step,params.t_step)

                    # Normalize the time or other variable to colormap
                    norm = Normalize(vmin=t.min(), vmax=t.max())
                    cmap = cm.viridis

                    # Create line segments
                    points = np.array([mx, my, mz]).T.reshape(-1, 1, 3)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Create color array
                    colors = cmap(norm(t[:-1]))

                    lc = Line3DCollection(segments, colors=colors, linewidth=2)
                    fig = plt.figure(figsize=(8, 5))
                    ax = fig.add_subplot(projection='3d')
                    ax.add_collection3d(lc)
                    ax.set_xlim([min(mx), max(mx)])
                    ax.set_ylim([min(my), max(my)])
                    ax.set_zlim([min(mz), max(mz)])
                    ax.set_xlabel('mx', fontsize=15)
                    ax.set_ylabel('my', fontsize=15)
                    ax.set_zlabel('mz', fontsize=15)
                    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                    mappable.set_array(t)
                    fig.colorbar(mappable, ax=ax, label='Time (s)')
                    plt.title(title, fontsize=15)

                    # Save the figure
                    plt.savefig(os.path.join(self.figure_path, "trajectories/", filename), dpi=300, bbox_inches='tight')
                    plt.close()


    def plot_scurve(self):
        os.makedirs(os.path.join(self.figure_path, "scurves/"), exist_ok=True)
        for i, device in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                bitstream_averages = self.results[(i,j)]
                title, filename = self.set_title_and_filename(params=params, title_prefix="S-Curve - ", filename_prefix="scurve_", j_stt=0, index = (i,j,0), show_values=True)
                plt.figure(figsize=(6, 6))
                plt.plot(self.j_stt_arr, bitstream_averages, marker='o', linestyle='-', color='tab:blue')
                plt.xlabel('STT bias current (A/m^2)', fontsize=15)
                plt.ylabel('bitstream average', fontsize=15)
                plt.title(title, fontsize=15)
                plt.grid()
                plt.savefig(os.path.join(self.figure_path, "scurves/", filename), dpi=300, bbox_inches='tight')
                plt.close() 


    def save_bitstream_averages(self):
        os.makedirs(os.path.join(self.figure_path, "bitstream_averages/"), exist_ok=True)
        for i, _ in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                bitstream_averages = self.results[(i,j)]
                _, bitstream_filename = self.set_title_and_filename(params=params, title_prefix="S-Curve - ", filename_prefix="bitstream_", j_stt=0, index = (i,j,0), show_values=True)
                bitstream_filename = os.path.join(self.figure_path, "bitstream_averages/", bitstream_filename.replace(".png",".npy"))
                j_stt_filename = "j_stt_arr.npy"
                np.save(bitstream_filename, np.array(bitstream_averages))
                np.save(os.path.join(self.figure_path, "bitstream_averages/", j_stt_filename), np.array(self.j_stt_arr))
                    

    def plot_combined_scurve(self):
        if len(self.sweep_variables) != 2: 
            raise ValueError("plot_scurve method requires exactly two sweep variables.")
        os.makedirs(os.path.join(self.figure_path, "combined-scurves/"), exist_ok=True)
        all_scurves = defaultdict(list)
        for i, device in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                bitstream_averages = self.results[(i,j)]
                var1 = getattr(params, self.sweep_variables[0])
                var2 = getattr(params, self.sweep_variables[1])
                all_scurves[var2].append((var1, bitstream_averages))
                # --- Step 2: Normalize colors by temperature ---0
        for var2, data in all_scurves.items():
            title = f"S-Curves at {self.sweep_variables[1].capitalize()}: {var2}"
            filename = f"combined_scurve_{self.sweep_variables[1]}_{var2}.png"
            all_temps = set([item[0] for item in data])
            collected_data = []
            for var1, bitstream_averages in data:
                collected_data.append((var1, self.j_stt_arr, bitstream_averages))

            if not collected_data:
                continue

            plt.figure(figsize=(6, 6))

            # --- Step 2: Normalize colors by temperature ---
            norm = plt.Normalize(min(all_temps), max(all_temps))
            cmap = plt.cm.inferno

            # --- Step 3: Plot S-curves (x=J_stt, y=bitstream avg, color=T) ---
            for (var1, J_stt, bitstr_avg) in sorted(collected_data, key=lambda x: x[0]):
                color = cmap(norm(var1))
                plt.plot(J_stt, bitstr_avg, "-", ".", color=color, label=f"{var1}")

            # --- Step 4: Labels, colorbar, save ---
            plt.xlabel('STT bias current (A/m²)', fontsize=14)
            plt.ylabel('Bitstream average', fontsize=14)
            plt.title(title, fontsize=15)
            plt.grid(True, linestyle=':')
            plt.tight_layout()

            # Colorbar for temperature
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label=f"{self.sweep_variables[0].capitalize()}")

            save_path = os.path.join(self.figure_path, f"combined-scurves/", filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


    def plot_combined_scurve_single(self):
        if len(self.sweep_variables) != 1: 
            raise ValueError("plot_scurve method requires exactly one sweep variable.")
        os.makedirs(os.path.join(self.figure_path, "combined-scurves-single/"), exist_ok=True)
        all_scurves = []
        for i, device in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                bitstream_averages = self.results[(i,j)]
                var1 = getattr(params, self.sweep_variables[0])
                all_scurves.append((var1, bitstream_averages))
                # --- Step 2: Normalize colors by temperature ---0

        title = f"S-Curves"
        filename = f"combined_scurve.png"
        all_temps = set([item[0] for item in all_scurves])
        collected_data = []
        for var1, bitstream_averages in all_scurves:
            collected_data.append((var1, self.j_stt_arr, bitstream_averages))

        if not collected_data:
            return

        plt.figure(figsize=(6, 6))

        # --- Step 2: Normalize colors by temperature ---
        norm = plt.Normalize(min(all_temps), max(all_temps))
        cmap = plt.cm.inferno

        # --- Step 3: Plot S-curves (x=J_stt, y=bitstream avg, color=T) ---
        for (var1, J_stt, bitstr_avg) in sorted(collected_data, key=lambda x: x[0]):
            color = cmap(norm(var1))
            plt.plot(J_stt, bitstr_avg, "-", ".", color=color, label=f"{var1}")

        # --- Step 4: Labels, colorbar, save ---
        plt.xlabel('STT bias current (A/m²)', fontsize=14)
        plt.ylabel('Bitstream average', fontsize=14)
        plt.title(title, fontsize=15)
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        # Colorbar for temperature
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=f"{self.sweep_variables[0].capitalize()}")

        save_path = os.path.join(self.figure_path, f"combined-scurves-single/", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    def save_parameters(self):
        os.makedirs(os.path.join(self.figure_path, "parameters/"), exist_ok=True)
        for i, device in enumerate(self.device_classes):
            for j, params in enumerate(self.param_sets):
                filename = f"params_{i}_{j}.json"
                param_dict = params.to_dict()
                param_filepath = os.path.join(self.figure_path, "parameters/", filename)
                with open(param_filepath, 'w') as f:
                    json.dump(param_dict, f, indent=4, default=str)