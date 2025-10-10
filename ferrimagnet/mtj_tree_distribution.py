import os
import sys
import csv
import time
import argparse
import cProfile, pstats, io
import numpy as np
import matplotlib.pyplot as plt
from mtj_mod import mtj_mod


def mtj_sample(j_stt):
  figure_path = "."
  theta = np.pi/2
  phi = 0
  T = 310
  t_step = 1e-12
  v_pulse = 0
  vhold = 0
  t_pulse = 0.8e-9
  t_relax = 0.2e-9
  Happl = 0
  Hshe = 0
  J_she = -0.8e12
  theta,phi,t,r,g,mx,my,mz,bitstr,_,energy, Ms, gamma, alpha, j_she, M_a, M_b, params = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl,Hshe,j_stt,J_she,vhold, T, figure_path, flips=1)
  print(bitstr)


# =====================================================================
#            to be called directly in serial operation
# =====================================================================
def single_run_serial(dev,k,init,lmda,\
                      hist,bitstream,energy_avg,mag_view_flag,iterator_never_equal_6):
      temp,bits,energies = rng(dev,k,init,lmda,mag_view_flag,iterator_never_equal_6)
      return temp,bits,energies

def rng(dev,k,init,lmda,mag_view_flag,proc_ID):
    x2 = 2**k
    x0 = 1
    x1 = (x2+x0)/2
    theta = init
    phi = np.random.rand()*2*np.pi
    # NOTE: new method
    dev.set_mag_vector(phi,theta)
    temp = 0
    bits = []
    energies = []

    for i in range(k):
      pright = (cdf(x2,lmda)-cdf(x1,lmda))/(cdf(x2,lmda)-cdf(x0,lmda))
      val = jz_lut_she(pright)
      out,energy = mtj_sample(dev,val,mag_view_flag,proc_ID)
      bits.append(out)
      energies.append(energy)

      if out == 1:
        x0 = x1
      elif out == 0:
        x2 = x1
      x1 = (x2+x0)/2
      temp += out*2**(k-i-1)
    return temp,bits,energies

def cdf(x, lmda):
    return 1-np.exp(-lmda*x)

dir_check = lambda d: None if(os.path.isdir(d)) else(os.mkdir(d))
def check_output_paths() -> None:
    dir_check("./results")
    dir_check("./results/parameter_files")
    dir_check("./results/chi2Data")
    dir_check("./results/plots")
    dir_check("./results/plots/distribution_plots")
    dir_check("./results/plots/magnetization_plots")
    dir_check("./results/magPhi")
    dir_check("./results/bitstream_results")
    dir_check("./results/magTheta")
    dir_check("./results/energy_results")
    dir_check("./results/countData")
    dir_check("./results/bitData")


def mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, rtype, dev_agnostic, writeFile=None):

  dd = 1
  # NOTE: device init only takes in dev-to-dev variation flag
  dev = SHE_MTJ_rng(dd_flag=dd)
  # NOTE: parameter setting done manually or with set_vals method, 1 uses default values
  dev.set_vals(Ki=Ki,Ms=Ms,tf=tf,J_she=J_she,a=50e-9,b=50e-9,d=d,eta=eta,alpha=alpha,Rp=Rp,TMR=TMR)
  #print(dev) #NOTE: can print device to list all parameters

  k = 8
  lmda = 0.01
  init_t = 9*np.pi/10
  samples = 20
  hist = []
  bitstream = []
  energy_avg = []
  mag_view_flag = True

  parallel_flag = False #NOTE: dont use parallel, much slower
  parallel_batch_size = 1 # ==== NOTE:  None value defaults to the total number of cores on the CPU ====

  RA_tmp = 7e-12
  A_tmp = np.pi * (50e-9 *50e-9) / 4
  counts = np.zeros(2**k)
  for j in range(samples):
      temp_j,bits_j,energies_j = single_run_serial(dev,k,init_t,lmda,hist,bitstream,energy_avg,\
                                                          mag_view_flag,j+7)
      counts[temp_j] += 1
      # hist.append(temp_j) # this is the list of numbers
      bitstream.append(''.join(str(i) for i in bits_j))
      energy_avg.append(np.average(energies_j))

  # ==============================================================================================
  # Build an analytical exponential probability density function (PDF)
  xxis = []
  exp_pdf = []
  #exp_count, _ = np.histogram(hist,bins=256)
  for j in range(2**k):
    if (j == 0):
      temp = 1 # exp_count[0]
      xxis.append(j)
      exp_pdf.append(temp)

    if (j > 0):
      temp = lmda*np.exp(-lmda*j)
      temp = temp*exp_pdf[0]/lmda
      xxis.append(j)
      exp_pdf.append(temp)

  # Normalize exponential distribution
  expsum = 0
  for j in range(2**k):
    expsum += exp_pdf[j]

  exp_pdf = exp_pdf/expsum
  exp_pdf = exp_pdf*samples

  # Build plot 2, overlay computed distribution to ideal exponential
  plt.figure(2)
  plt.plot(xxis, counts, 'b-')
  plt.plot(xxis, exp_pdf,'k--')
  plt.xlabel("Generated Number")
  plt.ylabel("Normalized")
  distribution_plot_path = "results/plots/distribution_plots/distribution_plot_{}.png".format(run)
  #plt.savefig(distribution_plot_path)
  plt.clf()
  plt.close()


def main():
  for i in range(100):
    print(mtj_sample(1e11))


if __name__ == "__main__":
  start_time = time.time()
  main()
  print("--- %s seconds ---" % (time.time() - start_time))
  exit()
