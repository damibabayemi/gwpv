import numpy as np
import h5py
import numba
import gwpv.waveform as gw

D = 20
num_points = 100
X = np.linspace(-D, D, num_points)
Y = X
Z = X
x, y, z = map(lambda arr: arr.flatten(order='F'), np.meshgrid(X, Y, Z, indexing='ij'))

# some randomly chosen timesteps
t = np.linspace(-1, 20, 100)

# the waveform
def waveform(x, y, z, t):
    return gw.waveformPlus(x,y,z,t,3,2,2,2) + 1j*gw.waveformCross(x,y,z,t,3,2,2,2)


h5f = h5py.File('timeseparated.h5', 'w')
gr = h5f.create_group('Extrapolated_N2.dir')
gr.create_dataset('t_values.dir', data=t)
for tea in t:
    t_data = waveform(x, y, z, tea)
    gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
