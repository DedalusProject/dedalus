

import numpy as np
import time
import shelve

from dedalus2_burns.public import *


# Set domain
zbottom = -1.
ztop = 3.
x_basis = Fourier(128, interval=(-1., 1.))
z_basis = Chebyshev(129, interval=(zbottom, ztop))
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

# Equations:
#
# Dx u + wz = 0
# Dt h - P DxDx(h) - P Dz(hz)         = - u Dx(h) - w hz
# Dt u - R DxDx(u) - R Dz(uz) + Dx(p) = - u Dx(u) - w uz
# Dt w - R DxDx(w) - R Dz(wz) + Dz(p) = - u Dx(w) - w wz + F h h
# Dz h - hz = 0
# Dz u - uz = 0
# Dz w - wz = 0
#
BIH = Problem(['p','h','u','w','hz','uz','wz'], 1)

# Rayleigh-Benard convection
Ra = 1e6
Pr = 1.

P = 1.
F = Ra * Pr
R = Pr

def M(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0]])
BIH.M0[0] = M

def L0(d_trans):
    Dx = d_trans[0]
    PDD = P * Dx * Dx
    RDD = R * Dx * Dx
    return np.array([[  0,   0,  Dx,   0,  0,  0, 1.],
                     [  0,-PDD,   0,   0,  0,  0,  0],
                     [ Dx,   0,-RDD,   0,  0,  0,  0],
                     [  0,   0,   0,-RDD,  0,  0,  0],
                     [  0,   0,   0,   0,-1.,  0,  0],
                     [  0,   0,   0,   0,  0,-1.,  0],
                     [  0,   0,   0,   0,  0,  0,-1.]])
BIH.L0[0] = L0

def L1(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0, -P,  0,  0],
                     [  0,  0,  0,  0,  0, -R,  0],
                     [ 1.,  0,  0,  0,  0,  0, -R],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0,  0]])
BIH.L1[0] = L1

D = operators.create_diff_operators(domain)
BIH.parameters['Dx'] = D[0]
BIH.parameters['F'] = F

BIH.F[1] = "- u*Dx(h) - w*hz"
BIH.F[2] = "- u*Dx(u) - w*uz"
BIH.F[3] = "- u*Dx(w) - w*wz + F*h*h"

# def ML(d_trans):
#     return np.array([[  0,  0,  0,  0,  0,  0,  0],
#                      [  0, 1.,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0]])
# BIH.ML = ML

# def MR(d_trans):
#     return np.array([[  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0, 1.,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0],
#                      [  0,  0,  0,  0,  0,  0,  0]])
# BIH.MR = MR

def LL(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0,  0],
                     [  0, 1.,  0,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0,  0]])
BIH.LL = LL

def LR(d_trans):
    Dx = d_trans[0]
    if Dx == 0.:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0, 1.,  0,  0,  0,  0,  0],
                         [  0,  0, 1.,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0]])
    else:
        return np.array([[  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0,  0],
                         [  0, 1.,  0,  0,  0,  0,  0],
                         [  0,  0, 1.,  0,  0,  0,  0],
                         [  0,  0,  0, 1.,  0,  0,  0]])
BIH.LR = LR

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
BIH.LI = LI

def b(d_trans):
    Dx = d_trans[0]
    b = np.zeros(7)
    if Dx == 0.:
        b[1] = zbottom
        b[4] = ztop
    return b
BIH.b = b

# Build solver
ts = timesteppers.Euler
int = Integrator(BIH, domain, ts)

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
h = int.state['h']
hz = int.state['hz']

pert = 1e-3
profile = -(z - zbottom) * (z - ztop)
profile /= profile.max()
delta = pert * profile * np.random.randn(*h['g'].shape)
h['g'] = z + delta
h.differentiate(1, hz)

# Integration parameters
min_dt = 1e-3
int.sim_stop_time = 1e-1
int.wall_stop_time = 2*60*60
int.stop_iteration = 200

# Storage lists
storage_list = ['p', 'h', 'u', 'w']
storage = {}
storage['time'] = [int.time]
storage['r'] = [np.copy(-int.state['h']['g']**2)]
for fn in storage_list:
    storage[fn] = [np.copy(int.state[fn]['g'])]
copy_cadence = 15
dt_cadence = 5

# Pre-Euler steps
presteps = 10
for i in range(presteps):
    int.advance(min_dt/presteps)

    storage['time'].append(int.time)
    storage['r'].append(np.copy(-int.state['h']['g']**2))
    for fn in storage_list:
        storage[fn].append(np.copy(int.state[fn]['g']))
    print('Iteration: %i, Time: %e' %(int.iteration-presteps, int.time))

int.timestepper = timesteppers.MCNAB2(int.pencilset, int.state, int.rhs)
int.iteration = 0

def grid_spacing(grid):
    diff = np.diff(grid)
    dg = np.empty_like(grid)
    dg[0] = diff[0]
    dg[-1] = diff[-1]
    for i in range(1, grid.size-1):
        dg[i] = min(diff[i], diff[i-1])

    return dg

xg = x_basis.grid
zg = z_basis.grid
dx = grid_spacing(xg).reshape((xg.size, 1))
dz = grid_spacing(zg).reshape((1, zg.size))

def cfl_dt(safety=1.):
    minut = np.min(np.abs(dx / int.state['u']['g'].real))
    minwt = np.min(np.abs(dz / int.state['w']['g'].real))

    return safety * min(minut, minwt)

CFL = 0.2
int.dt = min(min_dt, cfl_dt(CFL))

# Main loop
start_time = time.time()
while int.ok:

    # advance
    int.advance()

    # update lists
    if int.iteration % copy_cadence == 0:
        storage['time'].append(int.time)
        storage['r'].append(np.copy(-int.state['h']['g']**2))
        for fn in storage_list:
            storage[fn].append(np.copy(int.state[fn]['g']))
        print('Iteration: %i, Time: %e, dt: %e' %(int.iteration, int.time, int.dt))

    if int.iteration % dt_cadence == 0:
        int.dt = min(min_dt, cfl_dt(CFL))


if int.iteration % copy_cadence != 0:
    storage['time'].append(int.time)
    storage['r'].append(np.copy(-int.state['h']['g']**2))
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
shelf = shelve.open('data.db', flag='n')
shelf['time'] = np.array(storage['time'])
shelf['x'] = x
shelf['z'] = z
shelf['r'] = np.array(storage['r'])
for fn in storage_list:
    shelf[fn] = np.array(storage[fn])
shelf.close()

