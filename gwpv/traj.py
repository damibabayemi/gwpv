import h5py
import numpy as np

h5f = h5py.File('trajdata.h5', 'w')
gr = h5f.create_group('BH1.dir')
grid = np.zeros((100, 4))
grid[:, 0] = np.linspace(0, 45, 100)
coordcenter = grid
grid[:, 3] = np.sin(np.linspace(0, 45, 100))
gr.create_dataset('BH1', data=grid)
gr.create_dataset('CoordCenterInertial.dat', data=coordcenter)
