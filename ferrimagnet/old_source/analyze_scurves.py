import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

figure_path = "figures/mtj_gamma_sweep4/"
scurves_path = f'{figure_path}scurves/'
os.makedirs(scurves_path, exist_ok=True)
# Magnetic types to overlay
types = ["ferro", "ferri"]

# Marker styles
styles = {
    "ferro": {"marker": ".", "linestyle": "-"},
    "ferri": {"marker": ".", "linestyle": "--"}
}

# Find all numeric subdirectories
subdirs = sorted(
    [d for d in os.listdir(figure_path) if os.path.isdir(os.path.join(figure_path, d)) and d.isdigit()],
    key=lambda x: int(x)
)

print(f"Found {len(subdirs)} subdirectories: {subdirs}")

for subdir in subdirs:
    sub_path = os.path.join(figure_path, subdir)
    print(f"\nProcessing subdirectory: {sub_path}")

    plt.figure(figsize=(7, 6))
    all_temps = []
    gamma_value = None
    collected_data = []

    # --- Step 1: Load all data ---
    for t in types:
        j_files = sorted(glob.glob(f"{sub_path}/*_{t}_s_J_stt.npy"))
        if not j_files:
            continue

        print(f"  {t}: {len(j_files)} files found.")
        for jf in j_files:
            # Filename pattern: 250_380000.0_ferri_s_J_stt.npy
            match = re.search(r"(\d+)_([\d.]+)_\w+_s_J_stt.npy", os.path.basename(jf))
            if not match:
                continue

            T = float(match.group(1))
            gamma = float(match.group(2))
            gamma_value = gamma  # same gamma within subdir

            bf = os.path.join(sub_path, f"{int(T)}_{gamma}_{t}_s_bitstr_avg.npy")
            if not os.path.exists(bf):
                print(f"    Missing {bf}, skipping.")
                continue

            J_stt = np.load(jf)
            bitstr_avg = np.load(bf)

            collected_data.append((T, t, J_stt, bitstr_avg))
            all_temps.append(T)

    if not collected_data:
        print(f"  No valid data in {subdir}, skipping.")
        plt.close()
        continue

    # --- Step 2: Normalize colors by temperature ---
    norm = plt.Normalize(min(all_temps), max(all_temps))
    cmap = plt.cm.inferno

    # --- Step 3: Plot S-curves (x=J_stt, y=bitstream avg, color=T) ---
    for (T, t, J_stt, bitstr_avg) in sorted(collected_data, key=lambda x: x[0]):
        color = cmap(norm(T))
        plt.plot(J_stt, bitstr_avg, styles[t]["linestyle"], marker=styles[t]["marker"], color=color, label=f"{T} K")

    # --- Step 4: Labels, colorbar, save ---
    plt.xlabel('STT bias current (A/m²)', fontsize=14)
    plt.ylabel('Bitstream average', fontsize=14)
    plt.title(f"S-Curve Sweep — γ = {gamma_value:.2e} A/m", fontsize=15)
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # Colorbar for temperature
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Temperature (K)")

    save_path = os.path.join(scurves_path, f"scurve_gamma_{gamma_value:.0f}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {save_path}")
