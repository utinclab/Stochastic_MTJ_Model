from mtj_mod_ferro import mtj_mod
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

steps = 1 # 10
t_step = 5e-11
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

r_avg = []
g_avg = []
bitstr_avg = []
mx_avg = []
my_avg = []
mz_avg = []
energy_avg = []
for rep in range(reps):
    for id,j in enumerate(J_stt):
        print(f'J = {j} A/m^2, H = {Happl[id]} A/m^2, point {id+1}/{steps}, rep {rep+1}/{reps}')
        theta = np.pi/2
        phi = 0
        t_arr = []
        r_arr = []
        g_arr = []
        mx_arr = []
        my_arr = []
        mz_arr = []
        bitstr_arr = []
        energy_arr = []
        for cy in tqdm(range(cycles),ncols=80,leave=False):
            theta,phi,t,r,g,mx,my,mz,bitstr,_,energy = mtj_mod(theta,phi,t_step,v_pulse,t_pulse,t_relax,Happl[id],Hshe,j,J_she,vhold)
            t_arr.append(t)
            r_arr.append(r)
            g_arr.append(g)
            mx_arr.append(mx)
            my_arr.append(my)
            mz_arr.append(mz)
            bitstr_arr.append(bitstr)
            energy_arr.append(energy)
        r_avg.append(np.mean(r_arr))
        g_avg.append(np.mean(g_arr))
        mz_avg.append(np.mean(mz_arr))
        bitstr_avg.append(np.mean(bitstr_arr))
        energy_avg.append(np.sum(energy_arr)/cycles)
        print(f'mz_avg = {mz_avg[-1]}; bitstr_avg = {(bitstr_avg[-1]+1)/2}; energy_avg = {energy_avg[-1]}')
        print('---------------')

# np.save('tm',t)
# np.save('mz',mz_arr)
# data_path = './Kwon_source/'
# np.save(data_path+'tm',t)
# np.save(data_path+'mz',mz)
# np.save(data_path+'mx',mx)
# np.save(data_path+'my',my)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

# Assuming you have your data arrays t, mx, my, mz

# 2D Plot for Mz vs. Time
plt.figure()
plt.plot(t * 1e9, mz)
plt.xlabel('Time (ns)', fontsize=15)
plt.ylabel('Mz', fontsize=15)

# 3D Plot with colormap
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')

# Normalize the time or other variable to colormap
norm = Normalize(vmin=t.min(), vmax=t.max())
cmap = cm.viridis

# Plot each segment with a color corresponding to the normalized value
for i in range(len(t) - 1):
    ax.plot(mx[i:i+2], my[i:i+2], mz[i:i+2], color=cmap(norm(t[i])), linewidth=2)

# Add labels
ax.set_xlabel('mx', fontsize=15)
ax.set_ylabel('my', fontsize=15)
ax.set_zlabel('mz', fontsize=15)

# Add a colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(t)
fig.colorbar(mappable, ax=ax, label='Time (s)')

plt.show()



# data_dir = './Kwon_source/'
# tx = np.load(data_dir + 'tm.npy')
# mz = np.load(data_dir + 'mz.npy')
# mx = np.load(data_dir + 'mx.npy')
# my = np.load(data_dir + 'my.npy')
# print(mz.shape); print(tx.shape)
# mz_rs = np.reshape(mz, (-1))
# print(mz_rs.shape)

# t_step = 5e-11
# tm1 = np.arange(0,len(mz_rs)*t_step,t_step)*1e9

# plt.figure(figsize=(6,4))
# plt.plot(tm1, mz_rs)
# plt.xlabel('Time (ns)', fontsize = 15)
# plt.ylabel('Mz', fontsize =15)

# plt.figure(figsize=(6,4))
# plt.plot(tx*1e9,mz)
# plt.xlabel('Time (ns)', fontsize = 15)
# plt.ylabel('Mz', fontsize =15)
        
# plt.figure(figsize=(6,5))
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(mx, my, mz, '-', label='dynamics')
# ax.set_xlabel('mx', fontsize=15)
# ax.set_ylabel('my', fontsize=15)
# ax.set_zlabel('mz', fontsize=15)
# ax.legend
# plt.show()

# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# plt.figure(figsize=(6,4))
# plt.plot(t*1e9,mz)
# plt.xlabel('Time (ns)', fontsize = 15)
# plt.ylabel('Mz', fontsize =15)
        
# plt.figure(figsize=(8,5))
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(mx, my, mz, '-', label='dynamics')
# ax.set_xlabel('mx', fontsize=15)
# ax.set_ylabel('my', fontsize=15)
# ax.set_zlabel('mz', fontsize=15)
# ax.legend
# plt.show()
