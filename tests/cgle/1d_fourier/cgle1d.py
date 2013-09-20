

import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from dedalus2.public import *


# Set domain
L = 300
x_basis = Chebyshev(512, interval=[-L/2., L/2.])
domain = Domain([x_basis])

mu = 1.
s = 0.0
b = 0.6
c = -1.4

MagSq = operators.MagSquared

# Complex Ginzburg-Landau equation
#
# A_t - mu A - s Ax - (1 + ib) A_xx = - (1 + ic) |A|^2 A
# A_x - Ax = 0
#
# A(0) = 0.
# A(L) = 0.
#
cgle = Problem(['A', 'Ax'], 1)
cgle.parameters['c'] = c
cgle.parameters['MagSq'] = MagSq

cgle.M0[0][0][0] = 1.
cgle.L0[0][0][0] = -mu
cgle.L0[0][0][1] = -s
cgle.L1[0][0][1] = -(1. + 1j*b)
cgle.F[0] = "-(1 + 1j*c) * MagSq(A) * A"

cgle.L1[0][1][0] = 1.
cgle.L0[0][1][1] = -1.

cgle.LL[0][0] = 1.
cgle.LR[1][0] = 1.

cgle.b[0] = 1.
cgle.b[1] = 1.

# Choose PDE and integrator
pde = cgle
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x = domain.grids[0]
A, Ax = int.state.fields.values()
qex = 2 * np.pi / L
gamma = 1.
A['x'] = np.exp(1j * (qex*x + np.pi/2. * np.tanh(gamma*x)))
Ax['k'] = A.differentiate(0)

# Integration parameters
int.dt = 0.1
int.sim_stop_time = 150
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
A_list = [np.copy(A['x'])]
copy_cadence = 10

# Main loop
start_time = time.time()
while int.ok:

    # Advance
    int.advance()

    # Update storage lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        A_list.append(np.copy(A['x']))

    # Print progress
    if int.iteration % copy_cadence == 0:
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

# Store final state
if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    A_list.append(np.copy(A['x']))

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
shelf['u'] = np.array(A_list).real
shelf['v'] = np.array(A_list).imag
shelf.close()

from dedalus2.data.field import field_manager
print(field_manager.field_count[domain], ' fields allocated')
