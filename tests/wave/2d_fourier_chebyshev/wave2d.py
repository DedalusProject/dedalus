

import numpy as np
import time
import shelve
from dedalus2.public import *


# Set domain
z_basis = Chebyshev(32, interval=[-1., 1.])
x_basis = Fourier(32, interval=[-1.,1.])
domain = Domain([x_basis, z_basis])

# Heat equation: u_tt = c^2 (u_xx + u_zz)
#
# ut - dt u = 0
# uz - dz u = 0
# ut_t - c^2 dx dx u - c^2 dz uz = 0
#
wave_eq = Problem(['u', 'uz', 'ut'], 1)
c = 1.
wave_eq.M0[0] = lambda d: np.array([[-1., 0., 0.],
                                    [0. , 0., 0.],
                                    [0. , 0., 1.]])
wave_eq.L0[0] = lambda d: np.array([[0., 0., 1.],
                                    [0., 1., 0.],
                                    [-c*c*d[0]*d[0], 0., 0.]])
wave_eq.L1[0] = lambda d: np.array([[0. , 0., 0.],
                                    [-1., 0., 0.],
                                    [0., -c*c, 0.]])
wave_eq.LL = lambda d: np.array([[0., 0., 0.],
                                 [1., 0., 0.],
                                 [0., 0., 0.]])
wave_eq.LR = lambda d: np.array([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [1., 0., 0.]])
wave_eq.b = lambda d: np.array([0., 0., 0.])

pde = wave_eq
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x, z = domain.grids
u = int.state['u']
uz = int.state['uz']
ut = int.state['ut']
u['X'] = np.sin(4*np.pi*z) * np.cos(4*np.pi*x)
uz['xk'] = u.differentiate(1)

# Integration parameters
int.dt = 1e-2
int.sim_stop_time = 0.9722
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
ut_list = [np.copy(ut['X'])]
#E_list = [np.sum(np.abs(u['X'])**2)]
copy_cadence = 1

# Main loop
start_time = time.time()
while int.ok:

    # Advance
    int.advance()

    # Update storage lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        u_list.append(np.copy(u['X']))
        ut_list.append(np.copy(ut['X']))
        #E_list.append(np.sum(np.abs(y['X'])**2))

    # Print progress
    if int.iteration % copy_cadence == 0:
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

# Store final state
if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    u_list.append(np.copy(u['X']))
    ut_list.append(np.copy(ut['X']))
    #E_list.append(np.sum(np.abs(y['X'])**2))

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
shelf['ut'] = np.array(ut_list)
#shelf['E'] = np.array(E_list)
shelf.close()

