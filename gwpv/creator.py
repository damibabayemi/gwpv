import numpy as np
import h5py

D = 10
num_points = 100
X = np.linspace(-D, D, num_points)
Y = X
Z = X
x, y, z = map(lambda arr: arr.flatten(order='F'), np.meshgrid(X, Y, Z, indexing='ij'))

# some randomly chosen timesteps
t = np.linspace(-102, 1000, 100)

def phi(x, y):
    return np.arctan2(y, x)

def theta(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.arccos(z / r)

# dummy function
def waveform(x, y, z, t):
    return 100*(1 - 1j)*(np.cos(theta(x, y, z))*1/(t+101) + np.sin(theta(x, y, z))*1/((t+50)**2)*np.cos(0.1*phi(x, y)))


h5f = h5py.File('timeseparated.h5', 'w')
gr = h5f.create_group('Extrapolated_N2.dir')
gr.create_dataset('t_values.dir', data=t)
for tea in t:
    t_data = waveform(x, y, z, tea)
    gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
