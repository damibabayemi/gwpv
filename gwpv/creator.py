import numpy as np
import h5py
import numba
import gwpv.waveform as gw
import argparse

D = 100
num_points = 100
X = np.linspace(-D, D, num_points)
Y = X
Z = X
x, y, z = map(lambda arr: arr.flatten(order='F'), np.meshgrid(X, Y, Z, indexing='ij'))

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bparameter", type=int,
                    help="choose scattering parametre b")
parser.add_argument("-g", "--gamma", type=int,
                    help="choose gamma factor")
parser.add_argument("-m1", "--mass1", type=int,
                    help="choose mass of black hole 1")
parser.add_argument("-m2", "--mass2", type=int,
                    help="choose mass of black hole 2")
args = parser.parse_args()
b = args.bparameter
g = args.gamma
m1 = args.mass1
m2 = args.mass2

# some randomly chosen timesteps
t = np.linspace(0, 5000, 100)

# the waveform
def waveform(x, y, z, t):
    return (gw.waveformPlus(x,y,z,t,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2))*0.01


h5f = h5py.File('timeseparated.h5', 'w')
gr = h5f.create_group('Extrapolated_N2.dir')
gr.create_dataset('t_values.dir', data=t)
for tea in t:
    t_data = waveform(x, y, z, 0.2*tea)
    gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
