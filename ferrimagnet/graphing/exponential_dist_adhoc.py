import numpy as np
import matplotlib.pyplot as plt
import os

date = '11122025'
test_name = 'corrected_exponential_dist_test_6_8_1'
figure_path = f'../figures/{date}/{test_name}/'
data_folder = f'../figures/{date}/{test_name}/data/'
lmda = 0.01

# --- Load intstream ---
intstream_path = os.path.join(data_folder, "intstream/exponential_intstream.npy")
intstream = np.load(intstream_path)

# --- Define range (0 to 255 for 8-bit values) ---
x_vals = np.arange(0, 256)

# --- Histogram as frequency ---
counts, _ = np.histogram(intstream, bins=np.arange(257))

# --- Ideal exponential distribution (scaled to match total count) ---
ideal_pdf = lmda * np.exp(-lmda * x_vals)
ideal_pdf = ideal_pdf / ideal_pdf.sum() * counts.sum()

# --- Powers of two markers ---
powers_of_two = [2**i for i in range(0, 8)]  # 1..128


# ============================================================
#   FIRST PLOT — LINEAR Y SCALE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(x_vals, counts, label="Empirical Frequency", linewidth=2)
plt.plot(x_vals, ideal_pdf, label="Ideal Exponential (λ=0.01)", linestyle="--", linewidth=2)

# Vertical lines at powers of two
for p in powers_of_two:
    plt.axvline(x=p, color='gray', linestyle=':', linewidth=1)
    plt.text(
        p, max(counts)*0.9, f"2^{int(np.log2(p))}",
        rotation=90, va='top', ha='right', fontsize=8
    )

plt.title("Empirical Frequency vs Ideal Exponential (Linear Y)")
plt.xlabel("Value (0–255)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# Save
save_dir = os.path.join(figure_path, "intstream/")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "exponential_frequency_linear.png"), dpi=300)

plt.show()



# ============================================================
#   SECOND PLOT — LOG Y SCALE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(x_vals, counts, label="Empirical Frequency", linewidth=2)
plt.plot(x_vals, ideal_pdf, label="Ideal Exponential (λ=0.01)", linestyle="--", linewidth=2)

plt.yscale("log")

# Vertical lines at powers of two
for p in powers_of_two:
    plt.axvline(x=p, color='gray', linestyle=':', linewidth=1)
    plt.text(
        p, counts.max(), f"2^{int(np.log2(p))}",
        rotation=90, va='top', ha='right', fontsize=8
    )

plt.title("Empirical Frequency vs Ideal Exponential (Log Y)")
plt.xlabel("Value (0–255)")
plt.ylabel("Frequency (log scale)")
plt.legend()
plt.grid(True, which='both')

# Save
plt.savefig(os.path.join(save_dir, "exponential_frequency_log.png"), dpi=300)

plt.show()

# --- Ideal exponential distribution (scaled to match total count) ---
ideal_pdf = lmda * np.exp(-lmda * x_vals)
ideal_pdf = ideal_pdf / ideal_pdf.sum() * counts.sum()

# --- R^2 Goodness of Fit ---
y = counts
y_hat = ideal_pdf

sse = np.sum((y - y_hat)**2)
sst = np.sum((y - np.mean(y))**2)
r2 = 1 - sse/sst

print(f"R^2 between empirical and ideal exponential: {r2:.6f}")

