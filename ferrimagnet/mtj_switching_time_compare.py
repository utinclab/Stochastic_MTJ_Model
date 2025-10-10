import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mtj_mod import mtj_mod
from mtj_mod_ferro import ferro_mtj_mod
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
    
def extract_relaxation_times(data, zero_thresh=0.01, one_thresh=0.09):
    data = np.array(data)
    times_to_one = []
    times_to_zero = []
    state = None
    start_index = None

    for i, val in enumerate(data):
        abs_val = abs(val)
        
        # Near zero: |val| < zero_thresh
        if abs_val < zero_thresh:
            if state == "from_one":
                times_to_zero.append(i - start_index)
                state = None
            else:
                state = "from_zero"
                start_index = i

        # Near |1|: |val| > one_thresh
        elif abs_val > one_thresh:
            if state == "from_zero":
                times_to_one.append(i - start_index)
                state = None
            else:
                state = "from_one"
                start_index = i

    avg_time_to_one = np.mean(times_to_one) if times_to_one else None
    avg_time_to_zero = np.mean(times_to_zero) if times_to_zero else None

    return {
        "times_to_one": times_to_one,
        "times_to_zero": times_to_zero,
        "avg_time_to_one": avg_time_to_one,
        "avg_time_to_zero": avg_time_to_zero
    }

def ferrimagnet_simulation(figure_path, t_step = 1e-12):
    steps = 1 # 10
    t_step = t_step
    v_pulse = 0
    vhold = 0
    t_pulse = 0.8e-9
    t_relax = 0.2e-9
    Happl = np.linspace(0,0,steps)
    Hshe = 0 # 300Oe=2.4e4 200Oe=1.6e4 100Oe=8e3
    J_stt = np.linspace(0,0,steps)
    J_she = -0.8e12
    cycles = 100
    reps = 1
    flips = 10

    # figure switches
    animation = False
    plot_3d = True
    plot_2d = True
    data = True
    plot_relaxation = True

    iterations = 100
    relaxation_time_avg = []
    relaxation_time_avg_zero = []
    temps = []
    sublattice_magnetizations = []
    for i in range(iterations):
        r_avg = []
        g_avg = []
        bitstr_avg = []
        mx_avg = []
        my_avg = []
        mz_avg = []
        energy_avg = []
        M_a = M_b = None
        for rep in range(reps):
            for id,j in enumerate(J_stt):
                print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
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
                relaxation_time = []
                relaxation_time_zero = []
                j_she = []
                for cy in tqdm(range(cycles),ncols=80,leave=False):
                    theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, T, figure_path, flips)
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
                    results = extract_relaxation_times(mz)
                    relaxation_time.append(results['avg_time_to_one'])
                    relaxation_time_zero.append(results['avg_time_to_zero'])
                r_avg.append(np.mean(r_arr))
                g_avg.append(np.mean(g_arr))
                mz_avg.append(np.mean(mz_arr))
                bitstr_avg.append(np.mean(bitstr_arr))
                energy_avg.append(np.sum(energy_arr)/cycles)
                relaxation_time_avg.append(np.mean(relaxation_time))
                relaxation_time_avg_zero.append(np.mean(relaxation_time_zero))
                print(f'mz_avg = {mz_avg[-1]}; bitstr_avg = {(bitstr_avg[-1]+1)/2}; energy_avg = {energy_avg[-1]}')
                print('---------------')
        # Assuming you have your data arrays t, mx, my, mz

        # Save only if file doesn't exist
        filename = figure_path+str(T)+"_ferri_params.json"
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
            plt.savefig(f'{figure_path}{T:03}_ferri_mtj_2d_plot.png', dpi=300, bbox_inches='tight')
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
            plt.title(f'Ferrimagnetic MTJ RNG at T={T}', fontsize=15)
            # Add annotation box to ax1

            # plt.show()
            # Save the figure
            plt.savefig(f'{figure_path}{T:03}_ferri_mtj_3d_plot.png', dpi=300, bbox_inches='tight')
            plt.close()


        if data:
            np.save('tm',t)
            np.save('mz',mz_arr)
            data_path = figure_path
            np.save(f'{data_path}{T:03}_ferri_tm',t)
            np.save(f'{data_path}{T:03}_ferri_mz',mz)
            np.save(f'{data_path}{T:03}_ferri_mx',mx)
            np.save(f'{data_path}{T:03}_ferri_my',my)

        # Animation of the MTJ dynamics
        if animation:
            # ====== USER PARAMETERS ======
            frame_interval = 0.000001      # ms between frames
            frame_skip = 2
            # =============================

            # Downsample
            mx_ds = mx[::frame_skip]
            my_ds = my[::frame_skip]
            mz_ds = mz[::frame_skip]
            t_ds = t[::frame_skip]
            j_she_ds = j_she[::frame_skip]

            # Set up figure
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 2, height_ratios=[3, 1, 1])

            ax3d_net = fig.add_subplot(gs[0, 0], projection='3d')
            ax3d_sub = fig.add_subplot(gs[0, 1], projection='3d')
            ax_plot = fig.add_subplot(gs[1, :])
            ax_mz = fig.add_subplot(gs[2, :])  # <- New subplot for Mz vs time

            # === Net magnetization vector plot ===
            ax3d_net.set_xlim([min(mx), max(mx)])
            ax3d_net.set_ylim([min(my), max(my)])
            ax3d_net.set_zlim([min(mz), max(mz)])
            ax3d_net.set_xlabel('mx')
            ax3d_net.set_ylabel('my')
            ax3d_net.set_zlabel('mz')
            ax3d_net.set_title('Net Magnetization')

            path_line, = ax3d_net.plot([], [], [], color='blue', linewidth=2)
            vector_line, = ax3d_net.plot([], [], [], color='red', linewidth=3)
            dot, = ax3d_net.plot([], [], [], 'ro')
            time_text = ax3d_net.text2D(0.05, 0.9, '', transform=ax3d_net.transAxes)

            # === Sublattice vector plot ===
            ax3d_sub.set_xlim([min(mx), max(mx)])
            ax3d_sub.set_ylim([min(my), max(my)])
            ax3d_sub.set_zlim([min(mz), max(mz)])
            ax3d_sub.set_xlabel('x')
            ax3d_sub.set_ylabel('y')
            ax3d_sub.set_zlabel('z')
            ax3d_sub.set_title('Sublattice Decomposition')

            Ma_line, = ax3d_sub.plot([], [], [], color='orange', linewidth=2, label='M_a path')
            Mb_line, = ax3d_sub.plot([], [], [], color='green', linewidth=2, label='M_b path')
            Ma_vector, = ax3d_sub.plot([], [], [], color='orange', linestyle='--', linewidth=3)
            Mb_vector, = ax3d_sub.plot([], [], [], color='green', linestyle='--', linewidth=3)
            Ma_dot = ax3d_sub.plot([], [], [], 'o', color='orange')[0]
            Mb_dot = ax3d_sub.plot([], [], [], 'o', color='green')[0]
            ax3d_sub.legend()

            # === j_she time plot ===
            ax_plot.set_xlim([t_ds[1], t_ds[-1]])
            ax_plot.set_ylim([-9e11, 9e11])
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('j_she (A/m²)')
            plot_line, = ax_plot.plot([], [], color='purple')

            # === mz vs time plot ===
            ax_mz.set_xlim([t_ds[1], t_ds[-1]])
            ax_mz.set_ylim([-1.1, 1.1])
            ax_mz.set_xlabel('Time (s)')
            ax_mz.set_ylabel('mz')
            mz_line, = ax_mz.plot([], [], color='black')

            def update(frame):
                # --- Net m ---
                mx_, my_, mz_ = mx_ds[frame], my_ds[frame], mz_ds[frame]
                m_vec = np.array([mx_, my_, mz_])
                m_unit = m_vec / np.linalg.norm(m_vec)

                path_line.set_data(mx_ds[:frame+1], my_ds[:frame+1])
                path_line.set_3d_properties(mz_ds[:frame+1])

                vector_line.set_data([0, mx_], [0, my_])
                vector_line.set_3d_properties([0, mz_])
                dot.set_data([mx_], [my_])
                dot.set_3d_properties([mz_])
                time_text.set_text(f'Time: {t_ds[frame]:.2e} s')

                # --- Sublattices ---
                Ma_mag = M_a / (M_a + M_b)
                Mb_mag = M_b / (M_a + M_b)

                Ma_dir = 1 if M_a > M_b else -1
                Mb_dir = -1 * Ma_dir

                Ma_vec = Ma_dir * Ma_mag * m_unit
                Mb_vec = Mb_dir * Mb_mag * m_unit

                Ma_line.set_data(Ma_dir * mx_ds[:frame+1] * Ma_mag, Ma_dir * my_ds[:frame+1] * Ma_mag)
                Ma_line.set_3d_properties(Ma_dir * mz_ds[:frame+1] * Ma_mag)

                Mb_line.set_data(Mb_dir * mx_ds[:frame+1] * Mb_mag, Mb_dir * my_ds[:frame+1] * Mb_mag)
                Mb_line.set_3d_properties(Mb_dir * mz_ds[:frame+1] * Mb_mag)

                Ma_dot.set_data([Ma_vec[0]], [Ma_vec[1]])
                Ma_dot.set_3d_properties([Ma_vec[2]])

                Mb_dot.set_data([Mb_vec[0]], [Mb_vec[1]])
                Mb_dot.set_3d_properties([Mb_vec[2]])

                Ma_vector.set_data([0, Ma_vec[0]], [0, Ma_vec[1]])
                Ma_vector.set_3d_properties([0, Ma_vec[2]])

                Mb_vector.set_data([0, Mb_vec[0]], [0, Mb_vec[1]])
                Mb_vector.set_3d_properties([0, Mb_vec[2]])

                # --- j_she plot ---
                plot_line.set_data(t_ds[:frame+1], j_she_ds[:frame+1])

                # --- mz vs time plot ---
                mz_line.set_data(t_ds[:frame+1], mz_ds[:frame+1])

                return (
                    path_line, vector_line, dot, time_text,
                    Ma_line, Mb_line, Ma_vector, Mb_vector,
                    plot_line, mz_line,
                    Ma_dot, Mb_dot
                )

            ani = FuncAnimation(fig, update, frames=len(t_ds), interval=frame_interval, blit=False)

            print(f"M_a: {M_a}, M_b: {M_b}")
            print(f"net MS = {M_a-M_b}")
            plt.tight_layout()
            # plt.show()
            # Save the animation
            ani.save(f'{figure_path}{T:03}_mtj_animation.gif', writer='pillow', fps=30, dpi=100)
            plt.close()

    return relaxation_time_avg, relaxation_time_avg_zero, temps

def ferromagnet_simulation(figure_path, t_step = 5e-11):
    steps = 1 # 10
    t_step = t_step
    v_pulse = 0
    vhold = 0
    t_pulse = 50e-9
    t_relax = 50e-9
    Happl = np.linspace(0,0,steps)
    Hshe = 0 # 300Oe=2.4e4 200Oe=1.6e4 100Oe=8e3
    J_stt = np.linspace(0,0,steps)
    J_she = -4e11
    cycles = 100
    reps = 1
    flips = 10

    # figure switches
    animation = False
    plot_3d = True
    plot_2d = True
    data = True
    plot_relaxation = True

    iterations = 100
    relaxation_time_avg = []
    relaxation_time_avg_zero = []
    temps = []
    sublattice_magnetizations = []
    for i in range(iterations):
        r_avg = []
        g_avg = []
        bitstr_avg = []
        mx_avg = []
        my_avg = []
        mz_avg = []
        energy_avg = []
        M_a = M_b = None
        for rep in range(reps):
            for id,j in enumerate(J_stt):
                print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
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
                relaxation_time = []
                relaxation_time_zero = []
                j_she = []
                for cy in tqdm(range(cycles),ncols=80,leave=False):
                    theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = ferro_mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, T, figure_path, flips)
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
                    results = extract_relaxation_times(mz)
                    relaxation_time.append(results['avg_time_to_one'])
                    relaxation_time_zero.append(results['avg_time_to_zero'])
                r_avg.append(np.mean(r_arr))
                g_avg.append(np.mean(g_arr))
                mz_avg.append(np.mean(mz_arr))
                bitstr_avg.append(np.mean(bitstr_arr))
                energy_avg.append(np.sum(energy_arr)/cycles)
                relaxation_time_avg.append(np.mean(relaxation_time))
                relaxation_time_avg_zero.append(np.mean(relaxation_time_zero))
                print(f'mz_avg = {mz_avg[-1]}; bitstr_avg = {(bitstr_avg[-1]+1)/2}; energy_avg = {energy_avg[-1]}')
                print('---------------')
        # Assuming you have your data arrays t, mx, my, mz

        # Save only if file doesn't exist
        filename = figure_path+str(T)+"_ferro_params.json"
        if not os.path.exists(filename):
            params_safe = make_json_safe(params)
            with open(filename, "w") as f:
                json.dump(params_safe, f, indent=4)
            print(f"✅ Parameters saved to {filename}")
        else:
            print(f"⚠️ Parameters file '{filename}' already exists. Skipping save.")

        if data:
            np.save('tm',t)
            np.save('mz',mz_arr)
            data_path = figure_path
            np.save(f'{data_path}{T:03}_ferro_tm',t)
            np.save(f'{data_path}{T:03}_ferro_mz',mz)
            np.save(f'{data_path}{T:03}_ferro_mx',mx)
            np.save(f'{data_path}{T:03}_ferro_my',my)

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
            plt.savefig(f'{figure_path}{T:03}_ferro_mtj_2d_plot.png', dpi=300, bbox_inches='tight')
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
            plt.title(f'Ferromagnetic MTJ RNG at T={T}', fontsize=15)
            # Add annotation box to ax1

            # plt.show()
            # Save the figure
            plt.savefig(f'{figure_path}{T:03}_ferro_mtj_3d_plot.png', dpi=300, bbox_inches='tight')
            plt.close()


        # Animation of the MTJ dynamics
        if animation:
            # ====== USER PARAMETERS ======
            frame_interval = 0.000001      # ms between frames
            frame_skip = 2
            # =============================

            # Downsample
            mx_ds = mx[::frame_skip]
            my_ds = my[::frame_skip]
            mz_ds = mz[::frame_skip]
            t_ds = t[::frame_skip]
            j_she_ds = j_she[::frame_skip]

            # Set up figure
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 2, height_ratios=[3, 1, 1])

            ax3d_net = fig.add_subplot(gs[0, 0], projection='3d')
            ax3d_sub = fig.add_subplot(gs[0, 1], projection='3d')
            ax_plot = fig.add_subplot(gs[1, :])
            ax_mz = fig.add_subplot(gs[2, :])  # <- New subplot for Mz vs time

            # === Net magnetization vector plot ===
            ax3d_net.set_xlim([min(mx), max(mx)])
            ax3d_net.set_ylim([min(my), max(my)])
            ax3d_net.set_zlim([min(mz), max(mz)])
            ax3d_net.set_xlabel('mx')
            ax3d_net.set_ylabel('my')
            ax3d_net.set_zlabel('mz')
            ax3d_net.set_title('Net Magnetization')

            path_line, = ax3d_net.plot([], [], [], color='blue', linewidth=2)
            vector_line, = ax3d_net.plot([], [], [], color='red', linewidth=3)
            dot, = ax3d_net.plot([], [], [], 'ro')
            time_text = ax3d_net.text2D(0.05, 0.9, '', transform=ax3d_net.transAxes)

            # === Sublattice vector plot ===
            ax3d_sub.set_xlim([min(mx), max(mx)])
            ax3d_sub.set_ylim([min(my), max(my)])
            ax3d_sub.set_zlim([min(mz), max(mz)])
            ax3d_sub.set_xlabel('x')
            ax3d_sub.set_ylabel('y')
            ax3d_sub.set_zlabel('z')
            ax3d_sub.set_title('Sublattice Decomposition')

            Ma_line, = ax3d_sub.plot([], [], [], color='orange', linewidth=2, label='M_a path')
            Mb_line, = ax3d_sub.plot([], [], [], color='green', linewidth=2, label='M_b path')
            Ma_vector, = ax3d_sub.plot([], [], [], color='orange', linestyle='--', linewidth=3)
            Mb_vector, = ax3d_sub.plot([], [], [], color='green', linestyle='--', linewidth=3)
            Ma_dot = ax3d_sub.plot([], [], [], 'o', color='orange')[0]
            Mb_dot = ax3d_sub.plot([], [], [], 'o', color='green')[0]
            ax3d_sub.legend()

            # === j_she time plot ===
            ax_plot.set_xlim([t_ds[1], t_ds[-1]])
            ax_plot.set_ylim([-9e11, 9e11])
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('j_she (A/m²)')
            plot_line, = ax_plot.plot([], [], color='purple')

            # === mz vs time plot ===
            ax_mz.set_xlim([t_ds[1], t_ds[-1]])
            ax_mz.set_ylim([-1.1, 1.1])
            ax_mz.set_xlabel('Time (s)')
            ax_mz.set_ylabel('mz')
            mz_line, = ax_mz.plot([], [], color='black')

            def update(frame):
                # --- Net m ---
                mx_, my_, mz_ = mx_ds[frame], my_ds[frame], mz_ds[frame]
                m_vec = np.array([mx_, my_, mz_])
                m_unit = m_vec / np.linalg.norm(m_vec)

                path_line.set_data(mx_ds[:frame+1], my_ds[:frame+1])
                path_line.set_3d_properties(mz_ds[:frame+1])

                vector_line.set_data([0, mx_], [0, my_])
                vector_line.set_3d_properties([0, mz_])
                dot.set_data([mx_], [my_])
                dot.set_3d_properties([mz_])
                time_text.set_text(f'Time: {t_ds[frame]:.2e} s')

                # --- Sublattices ---
                Ma_mag = M_a / (M_a + M_b)
                Mb_mag = M_b / (M_a + M_b)

                Ma_dir = 1 if M_a > M_b else -1
                Mb_dir = -1 * Ma_dir

                Ma_vec = Ma_dir * Ma_mag * m_unit
                Mb_vec = Mb_dir * Mb_mag * m_unit

                Ma_line.set_data(Ma_dir * mx_ds[:frame+1] * Ma_mag, Ma_dir * my_ds[:frame+1] * Ma_mag)
                Ma_line.set_3d_properties(Ma_dir * mz_ds[:frame+1] * Ma_mag)

                Mb_line.set_data(Mb_dir * mx_ds[:frame+1] * Mb_mag, Mb_dir * my_ds[:frame+1] * Mb_mag)
                Mb_line.set_3d_properties(Mb_dir * mz_ds[:frame+1] * Mb_mag)

                Ma_dot.set_data([Ma_vec[0]], [Ma_vec[1]])
                Ma_dot.set_3d_properties([Ma_vec[2]])

                Mb_dot.set_data([Mb_vec[0]], [Mb_vec[1]])
                Mb_dot.set_3d_properties([Mb_vec[2]])

                Ma_vector.set_data([0, Ma_vec[0]], [0, Ma_vec[1]])
                Ma_vector.set_3d_properties([0, Ma_vec[2]])

                Mb_vector.set_data([0, Mb_vec[0]], [0, Mb_vec[1]])
                Mb_vector.set_3d_properties([0, Mb_vec[2]])

                # --- j_she plot ---
                plot_line.set_data(t_ds[:frame+1], j_she_ds[:frame+1])

                # --- mz vs time plot ---
                mz_line.set_data(t_ds[:frame+1], mz_ds[:frame+1])

                return (
                    path_line, vector_line, dot, time_text,
                    Ma_line, Mb_line, Ma_vector, Mb_vector,
                    plot_line, mz_line,
                    Ma_dot, Mb_dot
                )

            ani = FuncAnimation(fig, update, frames=len(t_ds), interval=frame_interval, blit=False)

            print(f"M_a: {M_a}, M_b: {M_b}")
            print(f"net MS = {M_a-M_b}")
            plt.tight_layout()
            # plt.show()
            # Save the animation
            ani.save(f'{figure_path}{T:03}_mtj_animation.gif', writer='pillow', fps=30, dpi=100)
            plt.close()

    return relaxation_time_avg, relaxation_time_avg_zero, temps

def ferrimagnet_simulation_scurve(figure_path, t_step = 1e-12):
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
    data = True

    iterations = 100
    temps = []
    for i in range(iterations):
        r_avg = []
        g_avg = []
        bitstr_avg = []
        mz_avg = []
        energy_avg = []
        M_a = M_b = None
        for rep in range(reps):
            for id,j in enumerate(J_stt):
                print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
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
                    theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, T, figure_path)
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
        # Assuming you have your data arrays t, mx, my, mz

        # Save only if file doesn't exist
        filename = figure_path+str(T)+"_ferri_s_params.json"
        if not os.path.exists(filename):
            params_safe = make_json_safe(params)
            with open(filename, "w") as f:
                json.dump(params_safe, f, indent=4)
            print(f"✅ Parameters saved to {filename}")
        else:
            print(f"⚠️ Parameters file '{filename}' already exists. Skipping save.")


        if data:
            data_path = figure_path
            np.save(f'{data_path}{T:03}_ferri_s_J_stt',J_stt)
            np.save(f'{data_path}{T:03}_ferri_s_bitstr_avg',bitstr_avg)

        plt.figure(figsize=(6, 6))
        plt.plot(J_stt, bitstr_avg, marker='o', linestyle='-', color='tab:blue')
        plt.xlabel('STT bias current (A/m^2)', fontsize=15)
        plt.ylabel('bitstream average', fontsize=15)
        plt.title(f'Ferrimagnetic S-Curve at T={T}', fontsize=15)
        plt.grid()
        plt.savefig(f'{figure_path}{T:03}_ferri_mtj_scurve.png', dpi=300, bbox_inches='tight')
        plt.close()

def ferromagnet_simulation_scurve(figure_path, t_step = 1e-12):
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
    data = True

    iterations = 100
    temps = []
    sublattice_magnetizations = []
    for i in range(iterations):
        r_avg = []
        g_avg = []
        bitstr_avg = []
        mx_avg = []
        my_avg = []
        mz_avg = []
        energy_avg = []
        M_a = M_b = None
        for rep in range(reps):
            for id,j in enumerate(J_stt):
                print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
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
                    theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = ferro_mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold, T, figure_path)
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
        # Assuming you have your data arrays t, mx, my, mz

        # Save only if file doesn't exist
        filename = figure_path+str(T)+"_ferro_s_params.json"
        if not os.path.exists(filename):
            params_safe = make_json_safe(params)
            with open(filename, "w") as f:
                json.dump(params_safe, f, indent=4)
            print(f"✅ Parameters saved to {filename}")
        else:
            print(f"⚠️ Parameters file '{filename}' already exists. Skipping save.")

        if data:
            data_path = figure_path
            np.save(f'{data_path}{T:03}_ferro_s_J_stt',J_stt)
            np.save(f'{data_path}{T:03}_ferro_s_bitstr_avg',bitstr_avg)

        plt.figure(figsize=(6, 6))
        plt.plot(J_stt, bitstr_avg, marker='o', linestyle='-', color='tab:blue')
        plt.xlabel('STT bias current (A/m^2)', fontsize=15)
        plt.ylabel('bitstream average', fontsize=15)
        plt.title(f'Ferromagnetic S-Curve at T={T}', fontsize=15)
        plt.grid()
        plt.savefig(f'{figure_path}{T:03}_ferro_mtj_scurve.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    name = 'mtj_relaxation_test10'
    os.makedirs(f'./figures/{name}/', exist_ok=True)
    figure_path = f'./figures/{name}/'
    ferri_t_step = 1e-12
    ferri_relaxation_time_avg, ferri_relaxation_time_avg_zero, ferri_temps = ferrimagnet_simulation(figure_path, ferri_t_step)
    ferro_t_step = 5e-11
    ferro_relaxation_time_avg, ferro_relaxation_time_avg_zero, ferro_temps = ferromagnet_simulation(figure_path, ferro_t_step)
    # 2D Plot for Mz vs. Time
    plt.figure()
    for i in range(len(ferri_relaxation_time_avg)):
        ferri_relaxation_time_avg[i] = ferri_relaxation_time_avg[i] * ferri_t_step / 1e-9
        # ferri_relaxation_time_avg_zero[i] = ferri_relaxation_time_avg_zero[i] * t_step / 1e-9
    for i in range(len(ferro_relaxation_time_avg)):
        ferro_relaxation_time_avg[i] = ferro_relaxation_time_avg[i] * ferro_t_step / 1e-9
    plt.plot(ferri_temps, ferri_relaxation_time_avg, label='Ferrimagnetic Relaxation Time')
    plt.plot(ferro_temps, ferro_relaxation_time_avg, label='Ferromagnetic Relaxation Time')
    # plt.plot(temps, relaxation_time_avg_zero, label='Switching Time to 0')
    plt.xlabel('Tempurature (K)', fontsize=15)
    plt.ylabel('Time (ns)', fontsize=15)
    plt.legend()
    plt.title('Ferri vs Ferro-magnetic Switching Times', fontsize=15)
    plt.grid()
    plt.savefig(f'{figure_path}mtj_2d_plot_relax.png', dpi=300, bbox_inches='tight')
    plt.close()

    np.save(f'{figure_path}_ferri_relaxation_time_avg',ferri_relaxation_time_avg)
    np.save(f'{figure_path}_ferro_relaxation_time_avg',ferro_relaxation_time_avg)

    # ferrimagnet_simulation_scurve(figure_path, ferri_t_step)
    # ferromagnet_simulation_scurve(figure_path, ferro_t_step)

main()