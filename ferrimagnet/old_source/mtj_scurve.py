from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mtj_mod import mtj_mod
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec
matplotlib.use('Agg')  # No display — saves faster
from tqdm import tqdm
import json

import numpy as np
import os
from numba.typed import Dict as NumbaDict

def make_json_safe(obj):
    if isinstance(obj, NumbaDict):  # Convert numba Dict to normal dict
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def run_iteration(k):
    steps = 100 # 10
    t_step = 1e-12
    v_pulse = 0
    vhold = 0
    t_pulse = 0.8e-9
    t_relax = 0.2e-9
    Happl = np.linspace(0,0,steps)
    Hshe = 0 # 300Oe=2.4e4 200Oe=1.6e4 100Oe=8e3
    J_stt = np.linspace(-1e11,1e11,steps)
    J_she = -0.8e12
    cycles = 1000
    reps = 1

    # figure switches
    plot_3d = True
    plot_2d = True
    scurve = True

    iterations = 100
    temperature_range = 100
    name = 'mtj_gamma_sweep4'
    os.makedirs(f'./figures/{name}/', exist_ok=True)
    parent_path = f'./figures/{name}/'

    os.makedirs(f'{parent_path}{k}/', exist_ok=True)
    figure_path = f'{parent_path}{k}/'
    gamma_a = 0.35e6 + k*0.1e5
    temps = []
    for i in range(temperature_range):
        r_avg = []
        g_avg = []
        bitstr_avg = []
        mx_avg = []
        my_avg = []
        mz_avg = []
        energy_avg = []
        M_a = M_b = None
        for id,j in enumerate(J_stt):
            print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}')
            theta = np.pi/2
            phi = 0
            T = 250 + i
            temps.append(T)
            t_arr = []
            r_arr = []
            g_arr = []
            mx_arr = []
            my_arr = []
            mz_arr = []
            bitstr_arr = []
            energy_arr = []
            ms_arr = []
            gamma_arr = []
            alpha_arr = []
            j_she = []
            for cy in tqdm(range(cycles),ncols=80,leave=False):
                theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, T, figure_path, flips=1, gamma_a=gamma_a)
                # print(f'M_a: {M_a}, M_b: {M_b}')
                t_arr.append(t)
                r_arr.append(r)
                g_arr.append(g)
                mx_arr.append(mx)
                my_arr.append(my)
                mz_arr.append(mz)
                bitstr_arr.append(bitstr)
                energy_arr.append(energy)
                ms_arr.append(Ms)
                gamma_arr.append(gamma)
                alpha_arr.append(alpha)
            r_avg.append(np.mean(r_arr))
            g_avg.append(np.mean(g_arr))
            mz_avg.append(np.mean(mz_arr))
            bitstr_avg.append(np.mean(bitstr_arr))
            energy_avg.append(np.sum(energy_arr)/cycles)
            print(f'mz_avg = {mz_avg[-1]}; bitstr_avg = {(bitstr_avg[-1]+1)/2}; energy_avg = {energy_avg[-1]}')
            print('---------------')
        # Save only if file doesn't exist
        filename = figure_path+str(gamma_a)+"_params.json"
        if not os.path.exists(filename):
            params_safe = make_json_safe(params)
            with open(filename, "w") as f:
                json.dump(params_safe, f, indent=4)
            print(f"✅ Parameters saved to {filename}")
        else:
            print(f"⚠️ Parameters file '{filename}' already exists. Skipping save.")

        # 2D Plot for Mz vs. Time
        if plot_2d:
            plt.figure(figsize=(6, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)  # Two rows, shared x-axis

            # Top plot (new y-axis, same x)
            ax1 = plt.subplot(gs[0])
            ax1.plot(t * 1e9, j_she, color='tab:orange')
            ax1.set_ylabel('SOT current (A/m^2)', fontsize=15)
            ax1.tick_params(labelbottom=False)  # Hide x-axis labels here

            # Bottom plot (original plot)
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(t * 1e9, mz, color='tab:blue')
            ax2.set_xlabel('Time (ns)', fontsize=15)
            ax2.set_ylabel('Mz', fontsize=15)
            plt.savefig(f'{figure_path}{T}_{gamma_a:03}_mtj_2d_plot.png', dpi=300, bbox_inches='tight')
            plt.close()


        # # 3D Plot with colormap
        if plot_3d:
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
            # info_text = f'T={T}  Ms={ms_arr[-1]:.2e}\nGamma={gamma_arr[-1]:.2e}  Alpha={alpha_arr[-1]:.2e}'
            # ax.text(0.5, 0.05, 1, info_text, transform=ax1.transAxes,
            #         fontsize=10, verticalalignment='top',
            #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(t)
            fig.colorbar(mappable, ax=ax, label='Time (s)')
            plt.title(f'Ferrimagnetic MTJ RNG at T={T} gamma_a={gamma_a:03}', fontsize=15)
            # Add annotation box to ax1

            # plt.show()
            # Save the figure
            plt.savefig(f'{figure_path}{T}_{gamma_a:03}_mtj_3d_plot.png', dpi=300, bbox_inches='tight')
            plt.close()


        # scurve
        if scurve:
            plt.figure(figsize=(6, 6))
            plt.plot(J_stt, bitstr_avg, marker='o', linestyle='-', color='tab:blue')
            plt.xlabel('STT bias current (A/m^2)', fontsize=15)
            plt.ylabel('bitstream average', fontsize=15)
            plt.title(f'S-Curve at T={T} gamma={gamma_a:03}', fontsize=15)
            plt.grid()
            plt.savefig(f'{figure_path}{T}_{gamma_a:03}_mtj_scurve.png', dpi=300, bbox_inches='tight')
            plt.close()

            data_path = figure_path
            np.save(f'{data_path}{T}_{gamma_a:03}_ferri_s_J_stt',J_stt)
            np.save(f'{data_path}{T}_{gamma_a:03}_ferri_s_bitstr_avg',bitstr_avg)

    return k


def main():
    max_workers = max_workers = max(1, multiprocessing.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_iteration, k) for k in range(42, 101, 1)]
        for f in as_completed(futures):
            print(f"✅ Finished gamma iteration {f.result()}")


if __name__ == "__main__":
    main()