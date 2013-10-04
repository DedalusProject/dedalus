

import numpy as np
import time
import shelve
from dedalus2.public import *


# Set domain
x_basis = Fourier(16, interval=[0., 2.])
y_basis = Fourier(16, interval=[0., 2.])
z_basis = Chebyshev(32, interval=[0., 1.])
domain = Domain([x_basis, y_basis, z_basis])

Ra = 2000.
Pr = 1.
iPr = 1./Pr

rb = Problem(['p', 'u', 'v', 'w', 't', 'uz', 'vz', 'wz', 'tz'], 1)
# Equations:
#
# Dx u + Dy v + wz = 0
# iPr Dt u + Dx p        - (Dx Dx + Dy Dy) u - Dz uz = - iPr (u Dx u + v Dy u + w uz)
# iPr Dt v + Dy p        - (Dx Dx + Dy Dy) v - Dz vz = - iPr (u Dx v + v Dy v + w vz)
# iPr Dt w + Dz p - Ra t - (Dx Dx + Dy Dy) w - Dz wz = - iPr (u Dx w + v Dy w + w wz)
#     Dt t        -    w - (Dx Dx + Dy Dy) t - Dz tz = -     (u Dx t + v Dy t + w tz)
# Dz u - uz = 0
# Dz v - vz = 0
# Dz w - wz = 0
# Dz t - tz = 0
#

def M(d_trans):
    return np.array([[     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,   iPr,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,   iPr,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,   iPr,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,    1.,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0]])

rb.M0[0] = M

def L0(d_trans):
    Dx = d_trans[0]
    Dy = d_trans[1]
    DT2 = Dx*Dx + Dy*Dy
    return np.array([[     0,    Dx,    Dy,     0,     0,     0,     0,    1.,     0],
                     [    Dx,  -DT2,     0,     0,     0,     0,     0,     0,     0],
                     [    Dy,     0,  -DT2,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,  -DT2,   -Ra,     0,     0,     0,     0],
                     [     0,     0,     0,   -1.,  -DT2,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,   -1.,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,   -1.,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,   -1.,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,   -1.]])

rb.L0[0] = L0

def L1(d_trans):
    return np.array([[     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,   -1.,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,   -1.,     0,     0],
                     [    1.,     0,     0,     0,     0,     0,     0,   -1.,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,   -1.],
                     [     0,    1.,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,    1.,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,    1.,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,    1.,     0,     0,     0,     0]])

rb.L1[0] = L1

D = operators.create_diff_operators(domain)

rb.parameters['Pr'] = Pr
rb.parameters['Dx'] = D[0]
rb.parameters['Dy'] = D[1]

rb.F[1] = "-(u*Dx(u) + v*Dy(u) + w*uz) * (1./Pr)"
rb.F[2] = "-(u*Dx(v) + v*Dy(v) + w*vz) * (1./Pr)"
rb.F[3] = "-(u*Dx(w) + v*Dy(w) + w*wz) * (1./Pr)"
rb.F[4] = "-(u*Dx(t) + v*Dy(t) + w*tz)"

def LL(d_trans):
    return np.array([[     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,    1.,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,    1.,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,    1.,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,    1.,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                     [     0,     0,     0,     0,     0,     0,     0,     0,     0]])

rb.LL = LL

def LR(d_trans):
    Dx = d_trans[0]
    Dy = d_trans[1]
    if (Dx == 0) and (Dy == 0):
        return np.array([[     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    1.,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,    1.,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,    1.,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,    1.,     0,     0]])
    else:
        return np.array([[     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [     0,     0,     0,    1.,     0,     0,     0,     0,     0],
                         [     0,     0,     0,     0,    1.,     0,     0,     0,     0],
                         [     0,     0,     0,     0,     0,    1.,     0,     0,     0],
                         [     0,     0,     0,     0,     0,     0,    1.,     0,     0]])

rb.LR = LR

# Build solver
ts = timesteppers.CNAB3
int = Integrator(rb, domain, ts)

# initial conditions
x, y, z = domain.grids

u = int.state['u']
v = int.state['v']
w = int.state['w']
T = int.state['t']

T['X'] = 1e-2 * np.sin(np.pi * z) * np.random.randn(*T['X'].shape)

# integrate parameters
int.dt = 1e-3 / 2.
int.sim_stop_time = int.dt * 4000
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
v_list = [np.copy(v['X'])]
w_list = [np.copy(w['X'])]
T_list = [np.copy(T['X'])]
E_list = [np.sum(u['X']**2 + v['X']**2 + w['X']**2 + T['X']**2)]
copy_cadence = 20

# Main loop
start_time = time.time()
while int.ok:

    # advance
    int.advance()

    # update lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        u_list.append(np.copy(u['X']))
        v_list.append(np.copy(v['X']))
        w_list.append(np.copy(w['X']))
        T_list.append(np.copy(T['X']))
        E_list.append(np.sum(u['X']**2 + v['X']**2 + w['X']**2 + T['X']**2))
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    u_list.append(np.copy(u['X']))
    v_list.append(np.copy(v['X']))
    w_list.append(np.copy(w['X']))
    T_list.append(np.copy(T['X']))
    E_list.append(np.sum(u['X']**2 + v['X']**2 + w['X']**2 + T['X']**2))

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
shelf['y'] = y
shelf['z'] = z
shelf['u'] = np.array(u_list)
shelf['v'] = np.array(v_list)
shelf['w'] = np.array(w_list)
shelf['T'] = np.array(T_list)
shelf['E'] = np.array(E_list)
shelf.close()

