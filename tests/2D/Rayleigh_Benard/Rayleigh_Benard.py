
import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from dedalus2.public import *

# set domain
x_basis = Fourier(32, interval=[0.,1.])
z_basis = Chebyshev(32, interval=[0.,1.])
domain = Domain([x_basis,z_basis])

Ra = 2000.
Pr = 1.
iPr = 1./Pr

rb = Problem(['p','u','w','t','uz','tz'], 1)

# Equations:
#
# Dx u + Dz w = 0
# iPr*Dt u + Dx p - Dx^2 u - Dz uz = - iPr * ( u Dx u + w uz )
# iPr*Dt w + Dz p - Ra t - Dx^2 w + Dx uz = - iPr * (u Dx w - w Dx u)
# Dt t - w - Dx^2 t - Dz tz = - u Dx t - w tz
# Dz u - uz = 0
# Dz t - tz = 0
# 


def M(d_trans):
  return np.array([[0,   0,   0,   0, 0, 0],
                   [0, iPr,   0,   0, 0, 0],
                   [0,   0, iPr,   0, 0, 0],
                   [0,   0,   0, 1.0, 0, 0],
                   [0,   0,   0,   0, 0, 0],
                   [0,   0,   0,   0, 0, 0]])

rb.M0[0] = M

def L0(d_trans):
  Dx = d_trans[0]
  return np.array([[ 0,     Dx,      0,      0,    0,    0],
                   [Dx, -Dx**2,      0,      0,    0,    0],
                   [ 0,      0, -Dx**2,    -Ra,   Dx,    0],
                   [ 0,      0,   -1.0, -Dx**2,    0,    0],
                   [ 0,      0,      0,      0, -1.0,    0],
                   [ 0,      0,      0,      0,    0, -1.0]])


rb.L0[0] = L0

def L1(d_trans):
  return np.array([[   0,   0, 1.0,   0,    0,    0],
                   [   0,   0,   0,   0, -1.0,    0],
                   [ 1.0,   0,   0,   0,    0,    0],
                   [   0,   0,   0,   0,    0, -1.0],
                   [   0, 1.0,   0,   0,    0,    0],
                   [   0,   0,   0, 1.0,    0,    0]])

rb.L1[0] = L1

# rb.F[1] = "-1/Pr*(u*d_trans[0]*u+w*uz)"
# rb.F[2] = "-1/Pr*(u*d_trans[0]*w-w*d_trans[0]*u)"
# rb.F[3] = "-u*d_trans[0]*t-w*tz"

rb.LL = lambda d_trans: np.array([[ 0, 0, 1.0,   0,   0, 0],
                                  [ 0, 0,   0, 1.0,   0, 0],
                                  [ 0, 0,   0,   0, 1.0, 0],
                                  [ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0,   0,   0,   0, 0]])

rb.LR = lambda d_trans: np.array([[ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0,   0,   0,   0, 0],                                     
                                  [ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0, 1.0,   0,   0, 0],
                                  [ 0, 0,   0, 1.0,   0, 0],
                                  [ 0, 0,   0,   0, 1.0, 0]])


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
temp = int.state['t']
u['X']   = np.cos(np.pi * 2. * z) * np.cos(np.pi * 2. * x)
uz['xk'] = u.differentiate(1)
w['X']   = np.sin(np.pi * 2. * z) * np.sin(np.pi * 2. * x)

#wz = field_manager.get_field(domain)
#ux = field_manager.get_field(domain)

#wz['xk'] = w.differentiate(1)
#ux['K'] = u.differentiate(0)

#psi = field_manager.get_field(domain)

# integrate parameters

int.dt = 2e-2
int.sim_stop_time = 1.0
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
w_list = [np.copy(w['X'])]
theta_list = [np.copy(temp['X'])]
copy_cadence = 1

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
    theta_list.append(np.copy(temp['X']))
    print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
  t_list.append(int.time)
  u_list.append(np.copy(u['X']))

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
shelf['temp'] = np.array(theta_list)
shelf.close()





