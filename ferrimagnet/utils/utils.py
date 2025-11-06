import os
import pickle
from scipy import interpolate
import numpy as np

class MaterialUtils:
    @staticmethod
    def Ms_T_dependence(T, beta, Tc, M0):
        """
        Saturation magnetization as a function of temperature.
        
        Parameters:
        T (float): Temperature in K
        beta (float): Critical exponent
        Tc (float): Curie temperature
        M0 (float): Saturation magnetization at 0 K
        
        Returns:
        float: Saturation magnetization at temperature T
        """
        return M0 * (1 - T / Tc) ** beta
    
    @staticmethod
    def load_all_results(tmp_folder):
        results = []
        for filename in sorted(os.listdir(tmp_folder)):
            if filename.endswith(".pkl"):
                path = os.path.join(tmp_folder, filename)
                try:
                    with open(path, "rb") as f:
                        results.append(pickle.load(f))
                except EOFError:
                    print(f"[WARN] Skipping incomplete file: {path}")
        return results
    
    @staticmethod
    def j_stt_bitavg_lut(bitavg, bitstream_avg_path, j_stt_path):
        bitavg_arr = np.load(bitstream_avg_path)
        J_lut = np.load(j_stt_path)
        if bitavg > 0.9988:
            return -6.6e9
        elif bitavg < 0.002:
            return 6.6e9
        else:
            f = interpolate.interp1d(bitavg_arr, J_lut)
            return f(bitavg)