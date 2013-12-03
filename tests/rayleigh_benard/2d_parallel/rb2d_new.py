

import numpy as np
np.seterr(all="warn")
import time
import shelve

from dedalus2.public import *


# Set domain
x_basis = Fourier(64, interval=[0.,4.])
z_basis = Chebyshev(65, interval=[0.,1.])
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

# Rayleigh-Benard convection
#
# Equations:
#
# Dx(u) + wz = 0
#   Dt(h) - DxDx(h) - Dz(hz)         - w    = -   u Dx(h) -   w hz
# P Dt(u) - DxDx(u) - Dz(uz) + Dx(p)        = - P u Dx(u) - P w uz
# P Dt(w) - DxDx(w) - Dz(wz) + Dz(p) - Ra h = - P u Dx(w) - P w wz
# Dz(h) - hz = 0
# Dz(u) - uz = 0
# Dz(w) - wz = 0
#
RBC = Problem(['p','h','u','w','hz','uz','wz'], 1)

Ra = 2000.
Pr = 1.

pert = 1e-2

P = 1./Pr
R = Ra

def M(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0,  P,  0,  0,  0,  0],
                     [  0,  0,  0,  P,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0]])
RBC.M0[0] = M

def L0(d_trans):
    Dx = d_trans[0]
    DD = Dx * Dx
    return np.array([[  0,  0, Dx,  0,  0,  0, 1.],
                     [  0,-DD,  0,-1.,  0,  0,  0],
                     [ Dx,  0,-DD,  0,  0,  0,  0],
                     [  0, -R,  0,-DD,  0,  0,  0],
                     [  0,  0,  0,  0,-1.,  0,  0],
                     [  0,  0,  0,  0,  0,-1.,  0],
                     [  0,  0,  0,  0,  0,  0,-1.]])
RBC.L0[0] = L0

def L1(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,-1.,  0,  0],
                     [  0,  0,  0,  0,  0,-1.,  0],
                     [ 1.,  0,  0,  0,  0,  0,-1.],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0,  0]])
RBC.L1[0] = L1

D = operators.create_diff_operators(domain)
RBC.parameters['P'] = P
RBC.parameters['Dx'] = D[0]
RBC.parameters['Dz'] = D[1]

RBC.F[1] = "- u*Dx(h) - w*hz"
RBC.F[2] = "- P*u*Dx(u) - P*w*uz"
RBC.F[3] = "- P*u*Dx(w) - P*w*wz"

def LL(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0, 1.,  0],
                     [  0,  0,  0, 1.,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0]])
RBC.LL = LL

def LR(d_trans):
    Dx = d_trans[0]
    if Dx == 0.:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0, 1.,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0, 1.,  0],
                         [  0,  0,  0,  0,  0,  0,  0]])
    else:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0, 1.,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0, 1.,  0],
                         [  0,  0,  0, 1.,  0,  0,  0]])
RBC.LR = LR

def LI(d_trans):
    Dx = d_trans[0]
    if Dx == 0.:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [ 1.,  0,  0,  0,  0,  0,  0]])
    else:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0]])
RBC.LI = LI

# Build solver
ts = timesteppers.Euler
int = Integrator(RBC, domain, ts)

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

h = int.state['h']

h['g'] = pert * np.sin(np.pi * z) * np.random.randn(*h['g'].shape)

# Integration parameters
int.dt = 1e-3
int.sim_stop_time = int.dt * 1000
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Storage lists
storage_list = ['p', 'h', 'u', 'w']
storage = {}
storage['time'] = [int.time]
for fn in storage_list:
    storage[fn] = [np.copy(int.state[fn]['g'])]
copy_cadence = 15

# Pre-Euler steps
presteps = 0
for i in range(presteps):
    int.advance(3.*int.dt / presteps)

    storage['time'].append(int.time)
    for fn in storage_list:
        storage[fn].append(np.copy(int.state[fn]['g']))
    print('Iteration: %i, Time: %e' %(int.iteration-presteps, int.time))

int.timestepper = timesteppers.Euler(int.pencilset, int.state, int.rhs)
int.iteration = 0

# Main loop
start_time = time.time()
while int.ok:

    # advance
    int.advance()

    # update lists
    if int.iteration % copy_cadence == 0:
        storage['time'].append(int.time)
        for fn in storage_list:
            storage[fn].append(np.copy(int.state[fn]['g']))
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
    storage['time'].append(int.time)
    for fn in storage_list:
        storage[fn].append(np.copy(int.state[fn]['g']))

end_time = time.time()

# Print statistics
print('-' * 20)
print('Total time:', end_time - start_time)
print('Iterations:', int.iteration)
print('Average timestep:', int.time / int.iteration)
print('-' * 20)

# Write storage lists
shelf = shelve.open('data_%i.db' %domain.distributor.rank, flag='n')
shelf['time'] = np.array(storage['time'])
shelf['x'] = x
shelf['z'] = z
for fn in storage_list:
    shelf[fn] = np.array(storage[fn])
shelf.close()

