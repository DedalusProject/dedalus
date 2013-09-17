

import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from dedalus2.public import *


# Set domain
x_basis = Fourier(128, interval=[-1., 1.])
domain = Domain([x_basis])

# Wave equation:  y_tt = c2 y_xx
#
# y_t - v = 0
# y_x - dy = 0
# v_t - c2 dy_x = 0
#
# y(left) = 0
# y(right) = 0
#
# Index ordering:  Matrix[Order][Equation][Variable]
#
wave_equation_1d = Problem(['y', 'dy', 'v'], 1)
c = 1

wave_equation_1d.M0[0][0][0] = 1.
wave_equation_1d.L0[0][0][2] = -1.

wave_equation_1d.L0[0][1][1] = -1.
wave_equation_1d.L1[0][1][0] = 1.

wave_equation_1d.M0[0][2][2] = 1.
wave_equation_1d.L1[0][2][1] = -c**2

pde = wave_equation_1d
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x = domain.grids[0]
y = int.state['y']
dy = int.state['dy']
v = int.state['v']

# y['x'] = np.cos(np.pi * 12. * x)
# dy['k'] = y.differentiate(0)
# v['x'] = 0.

a = 0.15
y['xspace'] = np.exp(-x**2/a**2)
dy['k'] = y.differentiate(0)
v['k'] = -y.differentiate(0)

# Integration parameters
int.dt = 1e-2
int.sim_stop_time = 8.
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
y_list = [np.copy(y['x'])]
copy_cadence = 5

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

