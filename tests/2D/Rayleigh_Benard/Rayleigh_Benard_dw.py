

import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from dedalus2.public import *


# Set domain
x_basis = Fourier(16, interval=[0.,1.])
z_basis = Chebyshev(16, interval=[0.,1.])
domain = Domain([x_basis, z_basis])

Ra = 3000.
Pr = 1.
iPr = 1./Pr

rb = Problem(['p','u','w','t','uz','wz','tz'], 1)
# Equations:
#
# Dx u + wz = 0
# iPr Dt u + Dx p        - Dx Dx u - Dz uz = - iPr (u Dx u + w uz  )
# iPr Dt w + Dz p - Ra t - Dx Dx w - Dz wz = - iPr (u Dx w - w Dx u)
# Dt t - w - Dx Dx t - Dz tz = - u Dx t - w tz
# Dz u - uz = 0
# Dz w - wz = 0
# Dz t - tz = 0
#

def M(d_trans):
  return np.array([[0,   0,   0,   0, 0, 0, 0],
                   [0, iPr,   0,   0, 0, 0, 0],
                   [0,   0, iPr,   0, 0, 0, 0],
                   [0,   0,   0, 1.0, 0, 0, 0],
                   [0,   0,   0,   0, 0, 0, 0],
                   [0,   0,   0,   0, 0, 0, 0],
                   [0,   0,   0,   0, 0, 0, 0]])

rb.M0[0] = M

def L0(d_trans):
  Dx = d_trans[0]
  return np.array([[ 0,     Dx,      0,      0,   0,  1.,   0],
                   [Dx, -Dx*Dx,      0,      0,   0,   0,   0],
                   [ 0,      0, -Dx*Dx,    -Ra,   0,   0,   0],
                   [ 0,      0,    -1., -Dx*Dx,   0,   0,   0],
                   [ 0,      0,      0,      0, -1.,   0,   0],
                   [ 0,      0,      0,      0,   0, -1.,   0],
                   [ 0,      0,      0,      0,   0,   0, -1.]])


rb.L0[0] = L0

def L1(d_trans):
  return np.array([[  0,  0,  0,  0,   0,   0,   0],
                   [  0,  0,  0,  0, -1.,   0,   0],
                   [ 1.,  0,  0,  0,   0, -1.,   0],
                   [  0,  0,  0,  0,   0,   0, -1.],
                   [  0, 1.,  0,  0,   0,   0,   0],
                   [  0,  0, 1.,  0,   0,   0,   0],
                   [  0,  0,  0, 1.,   0,   0,   0]])

rb.L1[0] = L1

# rb.F[1] = "-1/Pr*(u*d_trans[0]*u+w*uz)"
# rb.F[2] = "-1/Pr*(u*d_trans[0]*w-w*d_trans[0]*u)"
# rb.F[3] = "-u*d_trans[0]*t-w*tz"

rb.LL = lambda d_trans: np.array([[ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0, 1.0,   0,   0, 0, 0],
                                  [ 0, 0,   0, 1.0,   0, 0, 0],
                                  [ 0, 0,   0,   0, 1.0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0]])

rb.LR = lambda d_trans: np.array([[ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0,   0,   0,   0, 0, 0],
                                  [ 0, 0, 1.0,   0,   0, 0, 0],
                                  [ 0, 0,   0, 1.0,   0, 0, 0],
                                  [ 0, 0,   0,   0, 1.0, 0, 0]])


pde = rb
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# initial conditions
x = domain.grids[0]
z = domain.grids[1]
u  = int.state['u']
uz = int.state['uz']
w  = int.state['w']
wz = int.state['wz']
T = int.state['t']
Tz = int.state['tz']
#T['X'] = 1e-10 * np.sin(np.pi * z) * np.random.randn(*T['X'].shape) + np.sin(np.pi*2.*z)
#Tz['xk'] = T.differentiate(1)
u['X']   = np.cos(1.*np.pi * z) * np.cos(2.*np.pi*x + np.pi/4.)
uz['xk'] = u.differentiate(1)
w['X']   = 2.*np.sin(1.*np.pi * z) * np.sin(2.*np.pi*x + np.pi/4.)
wz['xk'] = w.differentiate(1)

# wz = field_manager.get_field(domain)
# ux = field_manager.get_field(domain)

# wz['xk'] = w.differentiate(1)
# ux['K'] = u.differentiate(0)

#psi = field_manager.get_field(domain)

# integrate parameters

int.dt = 1e-2
int.sim_stop_time = 5.0
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
w_list = [np.copy(w['X'])]
T_list = [np.copy(T['X'])]
E_list = [np.sum(np.abs(w['X'])**2)]
copy_cadence = 5

# Main loop
start_time = time.time()
while int.ok:

  # advance
  int.advance()

  # update lists
  if int.iteration % copy_cadence == 0:
    t_list.append(int.time)
    u_list.append(np.copy(u['X']))
    w_list.append(np.copy(w['X']))
    T_list.append(np.copy(T['X']))
    E_list.append(np.sum(np.abs(w['X'])**2))
    print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
  t_list.append(int.time)
  u_list.append(np.copy(u['X']))
  w_list.append(np.copy(w['X']))
  T_list.append(np.copy(T['X']))
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
shelf['u'] = np.array(u_list)
shelf['w'] = np.array(w_list)
shelf['theta'] = np.array(T_list)
shelf['E'] = np.array(E_list)
shelf.close()





