

import numpy as np
import time
import shelve
from dedalus2.public import *


# Set domain
# z_basis = Chebyshev(32, interval=[-1., 1.])
# x_basis = Fourier(32, interval=[-1.,1.])
z_basis = Chebyshev(64, interval=[-1., 1.])
x_basis = Fourier(64, interval=[-1.,1.])
domain = Domain([x_basis, z_basis])

# Heat equation: y_t = y_xx + y_zz
#
# y_t - dy_z - y_xx = 0
# dy - y_z = 0
#
heat_eq = Problem(['y', 'dy'], 1)
heat_eq.M0[0] = lambda d_trans: np.array([[1., 0.],
                                          [0., 0.]])
heat_eq.L0[0] = lambda d_trans: np.array([[-d_trans[0]**2, 0.],
                                          [0.            , 1.]])
heat_eq.L1[0] = lambda d_trans: np.array([[ 0., -1.],
                                          [-1.,  0.]])
heat_eq.LL = lambda d_trans: np.array([[1., 0.],
                                       [0., 0.]])
heat_eq.LR = lambda d_trans: np.array([[0., 0.],
                                       [1., 0.]])

def b(d_trans):
  dx = d_trans[0]
  if dx == 0:
    return np.array([1., 1.])
  else:
    return np.array([0., 0.])

heat_eq.b = b

pde = heat_eq
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x, z = domain.grids
y = int.state['y']
dy = int.state['dy']
y['X'] = 1 + np.sin(np.pi*z)*np.cos(np.pi*x) + 1000.*np.sin(np.pi*8.*z)*np.cos(np.pi*8.*x)
dy['xk'] = y.differentiate(1)

# Integration parameters
c1 = (np.pi)**2 + (np.pi)**2
c8 = (8*np.pi)**2 + (8*np.pi)**2

int.dt = 1. / c8 # Resolve both scales
#int.dt = 2.399 / c8 # Resolve c1 scale.  c8 mode will decay correctly but oscillate
#int.dt = 1./ c1 # Resolve first scale.  c8 mode will oscillate AND decay too slowly.

print('h c1 = %f' %(int.dt*c1))
print('h c8 = %f' %(int.dt*c8))

int.sim_stop_time = int.dt * 100
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
y_list = [np.copy(y['X'])]
dy_list = [np.copy(dy['X'])]
E_list = [np.sum(np.abs(y['X'])**2)]
copy_cadence = 1

# Main loop
start_time = time.time()
while int.ok:

    # Advance
    int.advance()

    # Update storage lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        y_list.append(np.copy(y['X']))
        dy_list.append(np.copy(dy['X']))
        E_list.append(np.sum(np.abs(y['X'])**2))

    # Print progress
    if int.iteration % copy_cadence == 0:
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

# Store final state
if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    y_list.append(np.copy(y['X']))
    dy_list.append(np.copy(dy['X']))
    E_list.append(np.sum(np.abs(y['X'])**2))

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
shelf['y'] = np.array(y_list)
shelf['dy'] = np.array(dy_list)
shelf['E'] = np.array(E_list)
shelf.close()

