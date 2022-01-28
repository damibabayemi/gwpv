import numpy as np
import h5py
import numba
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bparameter", type=float,
                    help="choose scattering parametre b")
parser.add_argument("-g", "--gamma", type=float,
                    help="choose gamma factor")
parser.add_argument("-m1", "--mass1", type=float,
                    help="choose mass of black hole 1")
parser.add_argument("-m2", "--mass2", type=float,
                    help="choose mass of black hole 2")
parser.add_argument("-s1", "--spin1", type=float, default=1,
                    help="choose spin of black hole 1")
parser.add_argument("-s2", "--spin2", type=float, default=1,
                    help="choose spin of black hole 2")
parser.add_argument("-t", "--timesteps", type=int, default=100,
                    help="choose number of timesteps")
parser.add_argument("--r-scaling", default=False, action="store_true",
                   help="add 1 over r scaling")
parser.add_argument("--use-spinwaveform", default=False, action="store_true",
                   help="use the waveform with spins")
parser.add_argument("--get-curvature", default=False, action="store_true",
                   help="create curvature data instead of waveform data")
parser.add_argument("-i", "--initial", type=int, default=0,
                    help="choose initial time")
parser.add_argument("-f", "--final", type=float, default=50,
                    help="choose final time")
parser.add_argument("-d", "--size", type=float, default=10,
                    help="choose size of the rendered scene")
                    
args = parser.parse_args()
b = args.bparameter
g = args.gamma
m1 = args.mass1
m2 = args.mass2
s1 = args.spin1
s2 = args.spin2
nt = args.timesteps
ti = args.initial
tf = args.final
D = args.size

if args.use_spinwaveform:
  import gwpv.spinwaveform as gw
else:
  import gwpv.waveform as gw

num_points = 100
X = np.linspace(-D, D, num_points)
Y = X
Z = X
x, y, z = map(lambda arr: arr.flatten(order='F'), np.meshgrid(X, Y, Z, indexing='ij'))

t = np.linspace(ti, tf, nt)

if args.use_spinwaveform:
  # the waveform
  def waveform(x, y, z, t):
      if args.r_scaling:
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.divide(gw.waveformPlus(x,y,z,t,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2,s1,s2), r)*0.01
      else:
        return (gw.waveformPlus(x,y,z,t,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2,s1,s2))*0.01
  # use numerical differentiation techniques to  calculate space curvature
  def curvature(x, y, z, t):
      if args.r_scaling:
        h = 0.001
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.divide((gw.waveformPlus(x,y,z,t+2*h,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t+2*h,b,g,m1,m2,s1,s2))
                         -2*(gw.waveformPlus(x,y,z,t+h,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t+h,b,g,m1,m2,s1,s2))
                         +(gw.waveformPlus(x,y,z,t,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2,s1,s2)),(h**2)*r)
      else:
        h = 0.001
        return np.divide((gw.waveformPlus(x,y,z,t+2*h,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t+2*h,b,g,m1,m2,s1,s2))
                         -2*(gw.waveformPlus(x,y,z,t+h,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t+h,b,g,m1,m2,s1,s2))
                         +(gw.waveformPlus(x,y,z,t,b,g,m1,m2,s1,s2) + 1j*gw.waveformCross(x,y,z,t,b,g,m1,m2,s1,s2)),(h**2))
else:
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
      t_data = 0.5*curvature(x, y, z, tea)
      gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
else:
  for tea in t:
      t_data = waveform(x, y, z, tea)
      gr.create_dataset('t_{}.dir'.format(tea), data=t_data)
