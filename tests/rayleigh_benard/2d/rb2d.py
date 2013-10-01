

import numpy as np
import time
import shelve
from dedalus2.public import *


# Set domain
x_basis = Fourier(32, interval=[0.,4.])
z_basis = Chebyshev(32, interval=[0.,1.])
domain = Domain([x_basis, z_basis])

Ra = 2000.
Pr = 1.
iPr = 1./Pr

rb = Problem(['p','u','w','t','uz','tz'], 1)
# Equations:
#
# Dx u + Dz w = 0
# iPr Dt u + Dx p        - Dx Dx u - Dz uz = - iPr (u Dx u + w uz  )
# iPr Dt w + Dz p - Ra t - Dx Dx w + Dx uz = - iPr (u Dx w - w Dx u)
# Dt t - w - Dx Dx t - Dz tz = - u Dx t - w tz
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
                   [Dx, -Dx*Dx,      0,      0,    0,    0],
                   [ 0,      0, -Dx*Dx,    -Ra,   Dx,    0],
                   [ 0,      0,   -1.0, -Dx*Dx,    0,    0],
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

D = operators.create_diff_operators(domain)

rb.parameters['Pr'] = Pr
rb.parameters['Dx'] = D[0]

rb.F[1] = "-1/Pr*(u*Dx(u)+w*uz)"
rb.F[2] = "-1/Pr*(u*Dx(w)-w*Dx(u))"
rb.F[3] = "-u*Dx(t)-w*tz"

rb.LL = lambda d_trans: np.array([[ 0, 0, 1.0,   0,   0, 0],
                                  [ 0, 0,   0, 1.0,   0, 0],
                                  [ 0, 0,   0,   0, 1.0, 0],
                                  [ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0,   0,   0,   0, 0],
                                  [ 0, 0,   0,   0,   0, 0]])

def LR(d_trans):
    Dx = d_trans[0]
    if Dx == 0:
        return np.array([[   0, 0, 0,   0,   0, 0],
                         [   0, 0, 0,   0,   0, 0],
                         [   0, 0, 0,   0,   0, 0],
                         [ 1.0, 0, 0,   0,   0, 0],
                         [   0, 0, 0, 1.0,   0, 0],
                         [   0, 0, 0,   0, 1.0, 0]])
    else:
        return np.array([[ 0, 0,   0,   0,   0, 0],
                         [ 0, 0,   0,   0,   0, 0],
                         [ 0, 0,   0,   0,   0, 0],
                         [ 0, 0, 1.0,   0,   0, 0],
                         [ 0, 0,   0, 1.0,   0, 0],
                         [ 0, 0,   0,   0, 1.0, 0]])

rb.LR = LR

# Build solver
ts = timesteppers.CNAB3
int = Integrator(rb, domain, ts)

# initial conditions
x = domain.grids[0]
z = domain.grids[1]

u  = int.state['u']
uz = int.state['uz']
w  = int.state['w']
T = int.state['t']

T['X'] = 1e-2 * np.sin(np.pi * z) * np.random.randn(*T['X'].shape)

# integrate parameters
int.dt = 1e-3
int.sim_stop_time = int.dt * 2000
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
w_list = [np.copy(w['X'])]
T_list = [np.copy(T['X'])]
E_list = [np.sum(u['X']**2+w['X']**2+T['X']**2)]
copy_cadence = 10

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
        E_list.append(np.sum(u['X']**2+w['X']**2+T['X']**2))
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    u_list.append(np.copy(u['X']))
    w_list.append(np.copy(w['X']))
    T_list.append(np.copy(T['X']))
    E_list.append(np.sum(u['X']**2+w['X']**2+T['X']**2))

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
shelf['T'] = np.array(T_list)
shelf['E'] = np.array(E_list)
shelf.close()

