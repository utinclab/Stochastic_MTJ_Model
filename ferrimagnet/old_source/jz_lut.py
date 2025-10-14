from scipy import interpolate
import numpy as np
#FIXME: missing vcma and write 

J_lut = np.linspace(-6e9,6e9,41)

bit_avg_she = [9.99105e-01, 9.98360e-01, 9.97475e-01, 9.96170e-01, 9.93635e-01, 9.89935e-01, 
    9.84885e-01, 9.78295e-01, 9.68750e-01, 9.56170e-01, 9.39425e-01, 9.20085e-01,
    8.92535e-01, 8.60480e-01, 8.21865e-01, 7.78710e-01, 7.33545e-01, 6.78290e-01,
    6.21895e-01, 5.62025e-01, 5.01395e-01, 4.39310e-01, 3.79450e-01, 3.22725e-01,
    2.69530e-01, 2.17580e-01, 1.76170e-01, 1.38400e-01, 1.07065e-01, 8.08900e-02,
    5.98750e-02, 4.41350e-02, 3.16700e-02, 2.22750e-02, 1.47200e-02, 9.44500e-03,
    6.08000e-03, 4.18500e-03, 2.45000e-03, 1.64500e-03, 9.20000e-04]
energy_avg_she = [3.14221939e-13, 3.12707915e-13, 3.11269558e-13, 3.09910931e-13, 
 3.08634702e-13, 3.07436795e-13, 3.06318216e-13, 3.05276851e-13,
 3.04315629e-13, 3.03430865e-13, 3.02622444e-13, 3.01881945e-13,
 3.01219603e-13, 3.00624175e-13, 3.00098545e-13, 2.99637855e-13,
 2.99246975e-13, 2.98933609e-13, 2.98700329e-13, 2.98551614e-13,
 2.98500753e-13, 2.98554705e-13, 2.98725601e-13, 2.99018771e-13,
 2.99444795e-13, 3.00012497e-13, 3.00720096e-13, 3.01574758e-13,
 3.02575311e-13, 3.03723404e-13, 3.05015398e-13, 3.06448679e-13,
 3.08023904e-13, 3.09737667e-13, 3.11594298e-13, 3.13587114e-13,
 3.15714314e-13, 3.17973944e-13, 3.20375351e-13, 3.22910408e-13,
 3.25585426e-13]

bit_avg_she_fromfile = []

def read_bit_avg_she():
    f = open("results_Scurve/weightData_short_0000.txt", "r")
    for i in range(41):
        dev_i_bits = float(f.readline())
        bit_avg_she_fromfile.append(dev_i_bits)

    f.close()

def jz_lut_she(weight):
    if weight > 0.9988:
        return -6.6e9
    elif weight < 0.002:
        return 6.6e9
    else:
        f = interpolate.interp1d(bit_avg_she, J_lut)
        return f(weight)

def jz_lut_she_fromfile(weight):
    if weight > 0.9988:
        return -6.6e9
    elif weight < 0.002:
        return 6.6e9
    else:
        f = interpolate.interp1d(bit_avg_she_fromfile, J_lut)
        return f(weight)


def energy_she(weight):
    if weight > 0.9988:
        return 1.9e-13
    elif weight < 0.002:
        return 1.9e-13
    else:
        f = interpolate.interp1d(bit_avg_she, energy_avg_she)
        return f(weight)
