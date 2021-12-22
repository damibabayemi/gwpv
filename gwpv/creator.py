import numpy as np
import h5py
import numba
import gwpv.waveform as gw
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bparameter", type=int,
                    help="choose scattering parametre b")
parser.add_argument("-g", "--gamma", type=int,
                    help="choose gamma factor")
parser.add_argument("-m1", "--mass1", type=int,
                    help="choose mass of black hole 1")
parser.add_argument("-m2", "--mass2", type=int,
                    help="choose mass of black hole 2")
parser.add_argument("-t", "--timesteps", type=int, default=100,
                    help="choose number of timesteps")
parser.add_argument("--r-scaling", default=False, action="store_true",
                   help="add 1 over r scaling")
parser.add_argument("--get-curvature", default=False, action="store_true",
                   help="create curvature data instead of waveform data")
parser.add_argument("-i", "--initial", type=int, default=0,
                    help="choose initial time")
parser.add_argument("-f", "--final", type=int, default=50,
                    help="choose final time")
parser.add_argument("-d", "--size", type=int, default=10,
                    help="choose size of the rendered scene")
                    
args = parser.parse_args()
b = args.bparameter
g = args.gamma
m1 = args.mass1
m2 = args.mass2
nt = args.timesteps
ti = args.initial
# timesteps got scaled for simpler handling
tf = args.final/6
D = args.size


num_points = 100
X = np.linspace(-D, D, num_points)
Y = X
Z = X
x, y, z = map(lambda arr: arr.flatten(order='F'), np.meshgrid(X, Y, Z, indexing='ij'))

t = np.linspace(ti, tf, nt)

# the waveform
def waveform(x, y, z, t):
    if args.r_scaling:
      r = np.sqrt(x**2 + y**2 + z**2)
      return np.divide(gw.waveformPlus(x,y,z,t,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2), r)*0.01
    else:
      return (gw.waveformPlus(x,y,z,t,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2))*0.01
# use numerical differentiation techniques to  calculate space curvature
def curvature(x, y, z, t):
    if args.r_scaling:
      h = 0.001
      r = np.sqrt(x**2 + y**2 + z**2)
      return np.divide((gw.waveformPlus(x,y,z,t+2*h,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t+2*h,b,g,m1,m2))
                       -2*(gw.waveformPlus(x,y,z,t+h,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t+h,b,g,m1,m2))
                       +(gw.waveformPlus(x,y,z,t,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2)),(h**2)*r)
    else:
      h = 0.001
      return np.divide((gw.waveformPlus(x,y,z,t+2*h,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t+2*h,b,g,m1,m2))
                       -2*(gw.waveformPlus(x,y,z,t+h,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t+h,b,g,m1,m2))
                       +(gw.waveformPlus(x,y,z,t,b,g,m1,m2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2)),(h**2))



h5f = h5py.File('timeseparated.h5', 'w')
gr = h5f.create_group('Extrapolated_N2.dir')
gr.create_dataset('t_values.dir', data=t)
if args.get_curvature:
  for tea in t:
      t_data = curvature(x, y, z, tea)
      gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
else:
  for tea in t:
      t_data = waveform(x, y, z, tea)
      gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
