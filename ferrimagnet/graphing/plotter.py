import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.gridspec import GridSpec
import json


class Plotter:
    def __init__(self, figure_path='figures', results='results', sweep_variables:list[str]=["temperature"]):
        self.figure_path = figure_path
        os.makedirs(self.figure_path, exist_ok=True)
        self.results = results
        self.sweep_variables = sweep_variables


    def set_title_and_filename(self, params, title_prefix, filename_prefix, j_stt, show_values=True):
        title_parts = []
        filename_parts = []
        
        for var in self.sweep_variables:
            if show_values:
                title_parts.append(f"{var.capitalize()}: {getattr(params, var)}")
                filename_parts.append(f"{var}_{getattr(params, var)}")
            else:
                title_parts.append(f"{var.capitalize()}")
                filename_parts.append(f"{var}")
        title = title_prefix + f"J_STT: {j_stt:.3f}, " + ", ".join(title_parts)
        filename = filename_prefix + f"jstt_{j_stt:.3f}_" + "_".join(filename_parts) + ".png"
        return title, filename


    def plot_bittrace(self):
        os.makedirs(os.path.join(self.figure_path, "bittraces/"), exist_ok=True)
        for res in self.results:
            for sim_out in res["simulation_output"]:
                params = sim_out["params"]
                j_stt = sim_out["j_stt"]
                _, filename = self.set_title_and_filename(params, title_prefix="", filename_prefix="bittrace_", j_stt=j_stt)
                result_data = sim_out["result"]
                mz = np.cos(result_data["theta"])

                plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)  # Two rows, shared x-axis

                # Top plot (new y-axis, same x)

                ax1 = plt.subplot(gs[0])
                print(len(result_data["J_she"]))
                size = len(result_data["t"])
                ax1.plot(result_data["t"] * 1e9, result_data["J_she"][-size:], color='tab:orange')
                ax1.set_ylabel('SOT current (A/m^2)', fontsize=15)
                ax1.tick_params(labelbottom=False)  # Hide x-axis labels here

                # Bottom plot (original plot)
                ax2 = plt.subplot(gs[1], sharex=ax1)
                ax2.plot(result_data["t"] * 1e9, mz[-size:], color='tab:blue')
                ax2.set_xlabel('Time (ns)', fontsize=15)
                ax2.set_ylabel('Mz', fontsize=15)
                plt.savefig(os.path.join(self.figure_path, "bittraces/", filename), dpi=300, bbox_inches='tight')
                plt.close()


    def plot_3D_trajectory(self):
        os.makedirs(os.path.join(self.figure_path, "trajectories/"), exist_ok=True)
        for res in self.results:
            for sim_out in res["simulation_output"]:
                params = sim_out["params"]
                title, filename = self.set_title_and_filename(params, title_prefix="Magnetization Trajectory - ", filename_prefix="mtj_trajectory_", j_stt=sim_out["j_stt"])
                result_data = sim_out["result"]

                mx = np.sin(result_data["theta"]) * np.cos(result_data["phi"])
                my = np.sin(result_data["theta"]) * np.sin(result_data["phi"])
                mz = np.cos(result_data["theta"])
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
        for res in self.results:
            params = res["simulation_output"][0]["params"]
            title, filename = self.set_title_and_filename(params=params, title_prefix="S-Curve - ", filename_prefix="scurve_", j_stt=0, show_values=True)
            plt.figure(figsize=(6, 6))
            plt.plot(res["j_stt_arr"], res["bitstream_averages"], marker='o', linestyle='-', color='tab:blue')
            plt.xlabel('STT bias current (A/m^2)', fontsize=15)
            plt.ylabel('bitstream average', fontsize=15)
            plt.title(title, fontsize=15)
            plt.grid()
            plt.savefig(os.path.join(self.figure_path, "scurves/", filename), dpi=300, bbox_inches='tight')
            plt.close() 

            # Save the S-curve data
            np.save(os.path.join(self.figure_path, "scurves/", f'{filename}_jstt'), res["j_stt_arr"])
            np.save(os.path.join(self.figure_path, "scurves/", f'{filename}_bitstream'), res["bitstream_averages"])

    
    def save_parameters(self):
        os.makedirs(os.path.join(self.figure_path, "parameters/"), exist_ok=True)
        for res in self.results:
            for sim_out in res["simulation_output"]:
                params = sim_out["params"]
                j_stt = sim_out["j_stt"]
                _, filename = self.set_title_and_filename(params, title_prefix="", filename_prefix="params_", j_stt=j_stt, show_values=True)
                param_dict = params.to_dict()
                param_filepath = os.path.join(self.figure_path, "parameters/", filename.replace(".png", ".json"))
                with open(param_filepath, 'w') as f:
                    json.dump(param_dict, f, indent=4, default=str)