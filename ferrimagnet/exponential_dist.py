from simulation.simulation import MTJSimulation
from devices.parallel_cpu import ParallelCPUDevice
from utils.utils import MaterialUtils
from parameters.ferri_params import FerriParameters
import numpy as np
import time, atexit
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def mtj_sample(simulation, j_stt, index, save=False):
    initial_state = (np.pi/2, np.random.rand()*2*np.pi, 0, 0, 0)
    j_she = -4e11
    return(simulation.run(state_init=initial_state, j_she=j_she, j_stt=j_stt, flips=1, index=(index,0), x=0, save_simulation_data=save))


def cdf(x, lmda):
    return 1-np.exp(-lmda*x)


def rng(sim,k,lmda,index, save=False):
        x2 = 2**k
        x0 = 1
        x1 = (x2+x0)/2
        temp = 0
        bits = ""
        bitstream_path = "./LUT/10292025/LUT_test1/bitstream_averages/bitstream_0_0_0_.npy"
        j_stt_arr_path = "./LUT/10292025/LUT_test1/bitstream_averages/j_stt_arr.npy"

        for i in range(k):
            pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
            bias = MaterialUtils.j_stt_bitavg_lut(pright, bitstream_path, j_stt_arr_path)
            out = mtj_sample(sim, bias, index, save)
            bits += '1' if out == -1 else '0'
            out = 1 if out == -1 else 0
            if out == 1:
                x0 = x1
            elif out == 0:
                x2 = x1
            x1 = (x2+x0)/2
            temp += out*2**(k-i-1)
        return temp, bits


def main():
    start_time = time.perf_counter()
    def _print_elapsed():
        elapsed = time.perf_counter() - start_time
        print(f"Total execution time: {elapsed:.4f} s")
    atexit.register(_print_elapsed)

    date = '10302025'
    test_name = 'exponential_dist_test1'
    figure_path = f'./figures/{date}/{test_name}/'
    data_folder = f'./figures/{date}/{test_name}/data/'
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(os.path.join(data_folder, "bitstream/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "intstream/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "theta/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "phi/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "energy/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "power/"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "R/"), exist_ok=True)

    dev = ParallelCPUDevice(FerriParameters(t_step=1e-12, temperature=300, t_pulse=0.8e-9, t_relax=0.5e-9))
    sim = MTJSimulation(dev, tmp_dir=data_folder)

    intstream = np.empty(100000, dtype=np.int32)
    for i in tqdm(range(100000), desc="Sampling Exponential Distribution", unit="it"):
        lmda = .01
        temp, _ = rng(sim,8,lmda,i,save=False)
        intstream[i] = temp
    
    np.save(os.path.join(data_folder, f"intstream/exponential_intstream.npy"), intstream)

    plt.hist(intstream, bins=50, density=True)
    plt.title("Exponential Distribution from MTJ-based RNG")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    os.makedirs(os.path.join(figure_path, "intstream/"), exist_ok=True)
    plt.savefig(os.path.join(figure_path, "intstream/exponential_distribution.png"))

    


if __name__ == "__main__":
    main()