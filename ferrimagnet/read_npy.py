import numpy as np

# Load the .npy file
data = np.load("./figures/mtj_relaxation_test10/_ferro_relaxation_time_avg.npy")

print("Shape:", data.shape)
print("Data type:", data.dtype)
print("First few values:", data[:10])  # show first 10 entries if it's 1D
print("Max value:", np.max(data))
print("Min value:", np.min(data))
