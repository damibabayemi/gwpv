import h5py
import numpy as np

h5f = h5py.File('trajdata.h5', 'w')

grid = np.zeros((100, 4))
grid[:, 0] = np.linspace(0, 45, 100)
coordcenter = np.zeros((100, 4))
coordcenter[:, 0] = np.linspace(0, 45, 100)
grid[:, 3] = np.sin(np.linspace(0, 45, 100))
h5f.create_dataset('BH1', data=grid)
h5f.create_dataset('CoordCenterInertial.dat', data=coordcenter)
