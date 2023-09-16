from interface_funcs import mtj_sample
import matplotlib.pyplot as plt
import numpy as np


def configuration_check(dev):
    steps = 1
    J_stt = np.linspace(0,0,steps)
    cycles = 100
    reps = 1
    t_step = 5e-11
    pulse_check_start = (1/5) * dev.t_pulse
    relax_check_start = (3/5) * dev.t_relax
    pulse_steps = int(np.floor(dev.t_pulse/t_step))
    relax_steps = int(np.floor(dev.t_relax/t_step))
    pulse_start = int(np.floor(pulse_check_start/t_step))
    relax_start = int(np.floor(relax_check_start/t_step))
    mz_avg = []
    mz_chk1_arr = np.zeros(pulse_steps)
    mz_chk2_arr = np.zeros(relax_steps)
    point2point_variation = 0
    for rep in range(reps):
        for id,j in enumerate(J_stt):
            print('starting')
            dev.set_mag_vector(0, np.pi/2)
            mz_arr = []
            for i in range(cycles):
                _,_ = mtj_sample(dev,0,1,1,i,1)
                mz = np.cos(dev.thetaHistory)
                mz_arr.append(mz)
                if i == 0:
                    print(mz)
                mz_chk1_arr = mz_chk1_arr + np.absolute(mz[0:pulse_steps])
                mz_chk2_arr = mz_chk2_arr + np.absolute(mz[pulse_steps:pulse_steps+relax_steps])
                point2point_variation = point2point_variation + np.sum(np.absolute(mz[1::] - mz[0:-1]))/(mz.size - 1)
            mz_avg.append(np.mean(mz_arr))

    mz_chk1_arr = mz_chk1_arr/cycles
    mz_chk2_arr = mz_chk2_arr/cycles
    point2point_variation = point2point_variation/cycles

    mz_chk1_val = np.average(mz_chk1_arr[pulse_start:pulse_steps])
    mz_chk2_val = np.average(mz_chk2_arr[relax_start:relax_steps])

    if point2point_variation > 0.25:
        numerical_err = -1
    else:
        numerical_err = 0

    if mz_chk1_val < 0.2:
        mz_chk1_res = 0
    elif mz_chk1_val < 0.5:
        mz_chk1_res = 1
    else:
        mz_chk1_res = -1

    if mz_chk2_val < 0.2:
        mz_chk2_res = -1
    elif mz_chk2_val < 0.5:
        mz_chk2_res = 1
    else:
        mz_chk2_res = 0

    if mz_chk1_res == -1:
        PMAIMA = -1 #if check 1 fails, then it is too PMA
    elif mz_chk2_res == -1:
        PMAIMA = 1 #if check 2 fails, it is not PMA enough
    else:
        PMAIMA = 0 #if both checks are just warnings or better, balance is good enough

    plt.plot(np.linspace(0,1,relax_steps+pulse_steps) ,mz_arr[0] )
    plt.show()
    return numerical_err, mz_chk1_res,mz_chk2_res,PMAIMA
