# ===== handles fortran interface and batch parallelism =====
from interface_funcs import run_in_parallel_batch, mtj_sample
# ===========================================================
import os
import sys
import csv
import time
import argparse
import cProfile, pstats, io
import numpy as np
import matplotlib.pyplot as plt
from config_verify import config_verify
#from tqdm import tqdm
from mtj_types_v3 import SHE_MTJ_rng,draw_norm
from jz_lut import jz_lut_she,read_bit_avg_she,bit_avg_she_fromfile#,jz_lut_vcma,jz_lut_write

def profile(func):
  def inner(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    retval = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #FIXME
    #print(s.getvalue())
    return retval 
  return inner

# =====================================================================
#       NOTE: do not call this function directly —
#       arguments handeled by interface function "run_in_parllel_batch"
# =====================================================================
def single_run_parallel(dev,k,init,lmda,\
                      hist_queue,bitstream_queue,energy_avg_queue,mag_view_flag,proc_ID):
      temp,bits,energies = rng(dev,k,init,lmda,mag_view_flag,proc_ID)
      hist_queue.put(temp)
      bitstream_queue.put(''.join(str(i) for i in bits))
      energy_avg_queue.put(np.average(energies))

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
      #val = val - 0.006*val
      #val = 1/(2359*1.26*10**-15)
      #print('here is the first jz')
      #print(str(val))
      #exit()
      #fortran call wrapper in interface_funcs
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

@profile
def mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, rtype, dev_agnostic, writeFile=None):
  # if writeFile == None:
  #   csvFile = "MTJ_Results.csv"
  #   f = open(csvFile, "a")
  #   writeFile = csv.writer(f)

  dd = 1
  # NOTE: device init only takes in dev-to-dev variation flag
  dev = SHE_MTJ_rng(dd_flag=dd)
  # NOTE: parameter setting done manually or with set_vals method, 1 uses default values
  dev.set_vals(Ki=Ki,Ms=Ms,tf=tf,J_she=J_she,a=50e-9,b=50e-9,d=d,eta=eta,alpha=alpha,Rp=Rp,TMR=TMR)
  #print(dev) #NOTE: can print device to list all parameters

  #print("verifying device paramters")
  #nerr, mz1, mz2, PI = config_verify(dev)
  # ignoring warnings
  #if nerr == -1:
  #  print('numerical error, do not use parameters!')
  #elif PI == -1:
  #  print('PMA too strong')
  #elif PI == 1:
  #  print('IMA too strong')
  #else:
  #  print('parameters okay')
  #print("running application")

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

  # ===================== entry point to parallelized fortran interface ==========================
  # This function will spawn multiple processes of single_run_wrapper which then passes
  # the dynamical computation jobs to fortran. After this functions return, there are no
  # more diverges from this python script.
  #
  #FIXME: currently much slower than the serial version. To be fixed with MPI or OMP
  # ========================================
  if (parallel_flag):
      hist, energy_avg, bitstream = run_in_parallel_batch(single_run_parallel,samples,\
                                                            dev,k,init_t,lmda,hist,bitstream,energy_avg,\
                                                            mag_view_flag,parallel_batch_size)
  else:
      RA_tmp = 7e-12
      A_tmp = np.pi * (50e-9 *50e-9) / 4
      counts = np.zeros(2**k)
      for j in range(samples):
          # if ((j%100) == 0):
          #    print('working on sample: %s',j)

          #if (dev_agnostic):
              # set device parameters
              #dev.Rp = draw_norm(RA_tmp/A_tmp,1,0.05)
              #delta_tmp = draw_norm(70,1,0.03)
              #Eb_tmp = delta_tmp*1.38e-23*300
              #dev.Ki = Eb_tmp/A_tmp
              #dev.TMR = draw_norm(1.5,1,0.05)

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
  #counts, _ = np.histogram(hist,bins=256)

  #sys.exit()


  # Calculate the chi_square parameter
  chi2 = 0
  for j in range(2**k):
    chi2 += ((counts[j]-exp_pdf[j])**2)/exp_pdf[j]

  # check_output_paths()
  # File holds the Chi2 value
  if rtype == 1:
    chi2Data_path = "results_Rp/chi2Data/runData_{:04d}.txt".format(run)
    f = open(chi2Data_path,'w')
    f.write(str(dev.Rp)[1:-1])
    f.write('\n')
    f.write(str(dev.Ki))
    f.write('\n')
    f.write(str(dev.TMR))
  elif rtype == 2:
    chi2Data_path = "results_Ki/chi2Data/runData_{:04d}.txt".format(run)
    f = open(chi2Data_path,'w')
    f.write(str(dev.Rp))
    f.write('\n')
    f.write(str(dev.Ki)[1:-1])
    f.write('\n')
    f.write(str(dev.TMR))
  elif rtype == 3:
    chi2Data_path = "results_TMR/chi2Data/runData_{:04d}.txt".format(run)
    f = open(chi2Data_path,'w')
    f.write(str(dev.Rp))
    f.write('\n')
    f.write(str(dev.Ki))
    f.write('\n')
    f.write(str(dev.TMR)[1:-1])
  elif rtype == 4:
    chi2Data_path = "results_Mult/chi2Data/runData_{:04d}.txt".format(run)
    f = open(chi2Data_path,'w')
    f.write(str(dev.Rp)[1:-1])
    f.write('\n')
    f.write(str(dev.Ki)[1:-1])
    f.write('\n')
    f.write(str(dev.TMR)[1:-1])
  elif rtype == 5:
    chi2Data_path = "results_streamcheck/chi2Data/runData_{:04d}.txt".format(run)
    f = open(chi2Data_path,'w')
    f.write(str(dev.Rp))
    f.write('\n')
    f.write(str(dev.Ki))
    f.write('\n')
    f.write(str(dev.TMR))
  else:
    print('unknown file type found, will generate an error')

  f.write('\n')
  f.write(str(k))
  f.write('\n')
  f.write(str(lmda))
  f.write('\n')
  f.write(str(samples))
  f.write('\n')
  f.write(str(chi2))
  f.close

  counts = counts/samples
  exp_pdf = exp_pdf/samples

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

  # Build plot 3, will track the magnetization path of the first generated random number
  # look at this picture to see if IMA or PMA; if magnetization stays strongly in XY plane,
  # it is IMA. If magnetization never comes down from +Z axis, PMA is too strong
  # if simulation durations change, inner for loop (for in in range (500) will need to change
  xvals = []
  yvals = []
  zvals = []

  cnt = 0
  try:
    for j in range(10):
      for i in range(1000): # change range
        xvals.append(np.sin(dev.thetaHistory[j][i])*np.cos(dev.phiHistory[j][i]))
        yvals.append(np.sin(dev.thetaHistory[j][i])*np.sin(dev.phiHistory[j][i]))
        zvals.append(np.cos(dev.thetaHistory[j][i]))
        cnt += 1
  except IndexError:
    pass

  fig, axs = plt.subplots(1,2)
  axs[0].scatter(yvals,zvals)
  axs[0].axis('equal')
  axs[0].axis(xmin = -1, xmax = 1)
  axs[0].axis(ymin = -1, ymax = 1)
  axs[0].set_xlabel('Y')
  axs[0].set_ylabel('Z')
  axs[1].scatter(xvals,zvals)
  axs[1].axis('equal')
  axs[1].axis(xmin = -1, xmax = 1)
  axs[1].axis(ymin = -1, ymax = 1)
  axs[1].set_xlabel('X')
  axs[1].set_ylabel('Z')
  axs[1].yaxis.set_label_position("right")
  axs[1].yaxis.tick_right()
  fig.tight_layout()
  magnetization_plot_path = "results/plots/magnetization_plots/magnetization_plot_{}.png".format(run)
  #plt.savefig(magnetization_plot_path)
  plt.clf()
  plt.close()

  # Save bitstream and energy values
  #np.save(bitstream_path, np.array(bitstream))
  energy_path = "results/energy_results/energy_{}.npy".format(run)
  #np.save(energy_path, np.array(energy_avg))

  # File holds the number of times each number was generated; use for a histogram
  if rtype == 1:
    countData_path = "results_Rp/countData/countData_{:04d}.txt".format(run)
  elif rtype == 2:
    countData_path = "results_Ki/countData/countData_{:04d}.txt".format(run)
  elif rtype == 3:
    countData_path = "results_TMR/countData/countData_{:04d}.txt".format(run)
  elif rtype == 4:
    countData_path = "results_Mult/countData/countData_{:04d}.txt".format(run)
  elif rtype == 5:
    countData_path = "results_streamcheck/countData/countData_{:04d}.txt".format(run)
  else:
    print('unknown file type found, will generate an error')

  f = open(countData_path,'w')
  for i in range(2**k):
    f.write(str(counts[i]))
    f.write('\n')
  f.close

  # File holds average energy for each sample
#  if rtype == 1:
#    energyData_path = "results_Rp/energyData/energyData_{:04d}.txt".format(run)
#  elif rtype == 2:
#    energyData_path = "results_Ki/energyData/energyData_{:04d}.txt".format(run)
#  elif rtype == 3:
#    energyData_path = "results_TMR/energyData/energyData_{:04d}.txt".format(run)
#  elif rtype == 4:
#    energyData_path = "results_Mult/energyData/energyData_{:04d}.txt".format(run)
#  elif rtype == 5:
#    countData_path = "results_novar/countData/countData_{:04d}.txt".format(run)
#  else:
#    print('unknown file type found, will generate an error')

#  f = open(energyData_path,'w')
#  for i in range(2**k):
#    f.write(str(energy_avg[i]))
#    f.write('\n')
#  f.close


  # File holds the list of all random numbers generated at each sample
  if rtype == 1:
    bitData_path = "results_Rp/bitData/bitData_{:04d}.txt".format(run)
  elif rtype == 2:
    bitData_path = "results_Ki/bitData/bitData_{:04d}.txt".format(run)
  elif rtype == 3:
    bitData_path = "results_TMR/bitData/bitData_{:04d}.txt".format(run)
  elif rtype == 4:
    bitData_path = "results_Mult/bitData/bitData_{:04d}.txt".format(run)
  elif rtype == 5:
    bitData_path = "results_streamcheck/bitData"
  else:
    print('unknown file type found, will generate an error')

#  f = open(bitData_path,'w')
#  for i in range(samples):
#    f.write(str(hist[i]))
#    f.write('\n')
#  f.close

  # File holds the outputs of the magnetization path generating the first random number (only theta)
  magTheta_path = "results/magTheta/magTheta_{}_RL9_2.txt".format(run)
  f = open(magTheta_path,'w')
  thetshape = len(dev.thetaHistory[1])
  print(thetshape)
  try:
    for j in range(samples):
      for i in range(thetshape):
        f.write(str(dev.thetaHistory[j][i]))
        f.write('\n')
  except IndexError:
    pass
  finally:
    f.close()

  # File holds the outputs of the magnetization path generating the first random number (only phi)
  magPhi_path = "results/magPhi/magPhi_{}.txt".format(run)
  f = open(magPhi_path,'w')
  try:
    for j in range(samples):
      for i in range(thetshape):
        f.write(str(dev.phiHistory[j][i]))
        f.write('\n')
  except IndexError:
    pass
  finally:
    f.close

  #if writeFile == None:
  #  parameterFile_path = "results/parameter_files/parameterFile_{}.txt".format(run)
  #  with open(parameterFile_path, "w") as f:
  #    f.write("alpha: {}\n".format(str(alpha)))
  #    f.write("Ki: {}\n".format(str(Ki)))
  #    f.write("Ms: {}\n".format(str(Ms)))
  #    f.write("Rp: {}\n".format(str(Rp)))
  #    f.write("TMR: {}\n".format(str(TMR)))
  #    f.write("d: {}\n".format(str(d)))
  #    f.write("tf: {}\n".format(str(tf)))
  #    f.write("eta: {}\n".format(str(eta)))
  #    f.write("J_she: {}\n".format(str(J_she)))
  #    f.write("distribution_plot_path: {}\n".format(distribution_plot_path))
  #    f.write("magnetization_plot_path: {}\n".format(magnetization_plot_path))
  #    f.write("bitstream_path: {}\n".format(bitstream_path))
  #    f.write("energy_path: {}\n".format(energy_path))
  #    f.write("countData_path: {}\n".format(countData_path))
  #    f.write("bitData_path: {}\n".format(bitData_path))
  #    f.write("magTheta_path: {}\n".format(magTheta_path))
  #    f.write("magPhi_path: {}\n".format(magPhi_path))
  #    f.write("chi2Data_path: {}\n".format(chi2Data_path))
  #else:
  #  writeFile.writerow([alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she,
  #                      distribution_plot_path, magnetization_plot_path, bitstream_path,
  #                      energy_path, countData_path, bitData_path, magTheta_path, magPhi_path, chi2Data_path])


def main():

  rtype = int(sys.argv[1])
  pid = os.getpid()

  if rtype == 1:
    pidfloc = 'PIDfiles_Rp/'
  elif rtype == 2:
    pidfloc = 'PIDfiles_Ki/'
  elif rtype == 3:
    pidfloc = 'PIDfiles_TMR/'
  elif rtype == 4:
    pidfloc = 'PIDfiles_Mult/'
  elif rtype == 5:
    pidfloc = 'PIDfiles_novar/'
  else:
    print('unexpected runtype, will error')
    exit()

  pidfilename = pidfloc + str(pid) + '.txt'
  # this is to clear old names out of the folder

  os.system('rm ' + pidfloc + '*.txt >/dev/null 2>&1')
  time.sleep(2)

  pf = open(pidfilename,'w')
  pf.write(str(pid))
  pf.close

  time.sleep(2)

  pidfilelist = os.listdir(pidfloc)

  numpids = 0
  for filename in pidfilelist:
    numpids += 1

  pidList = np.zeros(numpids)
  f = 0
  for filename in pidfilelist:
    a = filename[:-4]
    pidList[f] = float(a)
    if(pidList[f] == pid):
        print(str(pid) + ' assigned to local rank ' + str(f))
        run = f
    f += 1

  # set device parameters
  RA = 7e-12
  A = np.pi * (50e-9 *50e-9) / 4
  if rtype == 1:
    Rp = draw_norm(RA/A,1,0.05)
    if run == 0:
      print('This is a Rp variation run')
  else:
    Rp = RA/A
 
  if rtype == 2:
    delta = draw_norm(70,1,0.03)
    if run == 0:
      print('this is a Ki variation run')
  else:
    delta = 69.59
  Eb = delta*1.38e-23*300
  Ki = Eb/(np.pi * 50e-9 * 50e-9 / 4)

  if rtype == 3:
    TMR = draw_norm(1.5,1,0.05)
    if run == 0:
      print('this is a TMR variation run')
  else:
    TMR = 3.0

  alpha = 0.01
  Ki = 0.0002
  Ms = 300000
  Rp = 500
  J_she = 2.47e12
  
  eta = 0.1
  d = 3e-9
  tf = 1.1e-9
  
  # run device
  if rtype == 4:
    mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, rtype, 1, writeFile=None)
  else:
    mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run, rtype, 0, writeFile=None)



if __name__ == "__main__":
  start_time = time.time()

  main()
  print("--- %s seconds ---" % (time.time() - start_time))
  exit()
