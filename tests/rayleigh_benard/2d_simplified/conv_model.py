

import numpy as np
import time
import shelve
from dedalus2.public import *


# set domain
x_basis = Fourier(32, interval=[0.,1.])
z_basis = Chebyshev(32, interval=[0.,1.])
domain = Domain([x_basis, z_basis])

Ra = 150
k = np.pi
omega = np.sqrt(Ra) - k**2
T = 1. / np.abs(omega)
print('Expected mode growth rate = 2(Ra^0.5 - k^2) = %f' %omega)
print('Expected mode time scale = %f' %T)
print('Expected energy growth rate = 2(Ra^0.5 - k^2) = %f' %(2*omega))
print('Expected energy time scale = %f' %(T/2.))
print('Expected theta amp when (|w|=1) = %f' %(1./np.sqrt(Ra)))

conv = Problem(['w','t','wz','tz'], 1)

# Equations:
#
# Dt w - Ra t - Dx^2 w + Dz wz = 0
# Dt t - w - Dx^2 t - Dz tz = 0
# Dz w - wz = 0
# Dz t - tz = 0
#

def M0(d_trans):
  return np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]])

conv.M0[0] = M0

def L0(d_trans):
  Dx = d_trans[0]
  return np.array([[-Dx**2,    -Ra,  0.,  0.],
                   [   -1., -Dx**2,  0.,  0.],
                   [    0.,     0., -1.,  0.],
                   [    0.,     0.,  0., -1.]])


conv.L0[0] = L0

def L1(d_trans):
  return np.array([[0., 0., -1.,  0.],
                   [0., 0.,  0., -1.],
                   [1., 0.,  0.,  0.],
                   [0., 1.,  0.,  0.]])

conv.L1[0] = L1

conv.LL = lambda d_trans: np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 0., 0.],
                                    [0., 0., 0., 0.]])

conv.LR = lambda d_trans: np.array([[0., 0., 0., 0.],
                                    [0., 0., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [0., 1., 0., 0.]])


pde = conv
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# initial conditions
x = domain.grids[0]
z = domain.grids[1]
w  = int.state['w']
wz = int.state['wz']
t = int.state['t']
tz = int.state['tz']
t['X'] = 1e-5 * (2*np.random.randn(*t['X'].shape)-1) * np.sin(np.pi*z)
tz['xk'] = t.differentiate(1)
#w['X']   = np.sin(np.pi * 2. * z) * np.sin(np.pi * 2. * x)
#wz['xk'] = w.differentiate(1)

# integrate parameters

int.dt = T / 10.
print('Timestep = %f' %int.dt)
int.sim_stop_time = int.dt * 200
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
w_list = [np.copy(w['X'])]
theta_list = [np.copy(t['X'])]
E_list = [np.sum(np.abs(w['X'])**2)]
copy_cadence = 1

# Main loop
start_time = time.time()
while int.ok:

  # advance
  int.advance()

  # update lists
  if int.iteration % copy_cadence == 0:
    t_list.append(int.time)
    w_list.append(np.copy(w['X']))
    theta_list.append(np.copy(t['X']))
    E_list.append(np.sum(np.abs(w['X'])**2))
    print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
  t_list.append(int.time)
  w_list.append(np.copy(w['X']))
  theta_list.append(np.copy(t['X']))
  E_list.append(np.sum(np.abs(w['X'])**2))

end_time = time.time()

# Print statistics
print('-' * 20)
print('Total time:', end_time - start_time)
print('Iterations:', int.iteration)
print('Average timestep:', int.time / int.iteration)
print('-' * 20)

# Write storage lists
shelf = shelve.open('data.db', flag='n')
shelf['t'] = np.array(t_list)
shelf['x'] = x
shelf['z'] = z
shelf['w'] = np.array(w_list)
shelf['theta'] = np.array(theta_list)
shelf['E'] = np.array(E_list)
shelf.close()






