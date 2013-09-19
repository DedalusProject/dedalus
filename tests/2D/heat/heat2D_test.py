

import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from dedalus2.public import *


# Set domain
z_basis = Chebyshev(32, interval=[-1., 1.])
x_basis = Fourier(32,interval=[-1.,1.])
domain = Domain([x_basis,z_basis])

# Heat equation: y_t = y_xx + y_zz
#
# y_t - dy_z - y_xx = 0
# dy - y_z = 0
#
heat_equation_1d = Problem(['y', 'dy'], 1)
heat_equation_1d.M0[0] = lambda d_trans: np.array([[1., 0.],
                                                   [0., 0.]])
heat_equation_1d.L0[0] = lambda d_trans: np.array([[-d_trans[0]**2, 0.],
                                                   [0.            , 1.]])
heat_equation_1d.L1[0] = lambda d_trans: np.array([[ 0., -1.],
                                                   [-1.,  0.]])
heat_equation_1d.LL = lambda d_trans: np.array([[1., 0.],
                                                [0., 0.]])
heat_equation_1d.LR = lambda d_trans: np.array([[0., 0.],
                                                [1., 0.]])
heat_equation_1d.b = lambda d_trans: np.array([0., 0.])

pde = heat_equation_1d
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x = domain.grids[0]
z = domain.grids[1]
y = int.state['y']
dy = int.state['dy']
y['x'] = np.sin(np.pi * 2. * z)*np.cos(np.pi * 2. * x)
dy['k'] = y.differentiate(0)

# Integration parameters
int.dt = 1e-4
int.sim_stop_time = 0.1
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
y_list = [np.copy(y['x'])]
copy_cadence = 10

# Main loop
start_time = time.time()
while int.ok:

    # Advance
    int.advance()

    # Update storage lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        y_list.append(np.copy(y['x']))

    # Print progress
    if int.iteration % copy_cadence == 0:
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

# Store final state
if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    y_list.append(np.copy(y['x']))

end_time = time.time()

plt.contour(y['x'],x,z)
plt.colorbar()
plt.savefig('y_contour.png')


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
shelf['y'] = np.array(y_list)
shelf.close()

