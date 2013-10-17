

import numpy as np
import time
import shelve

from dedalus2.public import *


# Set domain
x_basis = Fourier(64, interval=[0.,1.])
z_basis = Chebyshev(128, interval=[-1.,1.])
domain = Domain([x_basis, z_basis])

# Equations:
#
# Dx u + wz = 0
# Dt rho = - (u Dx rho + w Dz rho)
# Dt u + Dx p            - iRe Dx Dx u - iRe Dz uz = - (u Dx u + w uz)
# Dt w + Dz p + iFr2 rho - iRe Dx Dx w - iRe Dz wz = - (u Dx w + w wz)
# Dz u - uz = 0
# Dz w - wz = 0
#
# Boundary conditions:
#
# u(L) = u(R) = 0
# w(L) = w(R) = 0
#
rt = Problem(['p','rho','u','w','uz','wz'], 1)

Fr = 1e-3
Re = 1e6
At = 1e-6
width = 0.2
pert = 0.2

iFr2 = Fr**(-2)
iRe = Re**(-1)

def M(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0],
                     [  0, 1.,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0],
                     [  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0]])
rt.M0[0] = M

def L0(d_trans):
    Dx = d_trans[0]
    iRDD = iRe * Dx * Dx
    return np.array([[  0,   0,   Dx,    0,  0, 1.],
                     [  0,   0,    0,    0,  0,  0],
                     [ Dx,   0,-iRDD,    0,  0,  0],
                     [  0,iFr2,    0,-iRDD,  0,  0],
                     [  0,   0,    0,    0,-1.,  0],
                     [  0,   0,    0,    0,  0,-1.]])
rt.L0[0] = L0

def L1(d_trans):
    return np.array([[  0,  0,  0,  0,   0,   0],
                     [  0,  0,  0,  0,   0,   0],
                     [  0,  0,  0,  0,-iRe,   0],
                     [ 1.,  0,  0,  0,   0,-iRe],
                     [  0,  0, 1.,  0,   0,   0],
                     [  0,  0,  0, 1.,   0,   0]])
rt.L1[0] = L1

D = operators.create_diff_operators(domain)
rt.parameters['Dx'] = D[0]
rt.parameters['Dz'] = D[1]

rt.F[1] = "-(u*Dx(rho) + w*Dz(rho))"
rt.F[2] = "-(u*Dx(u) + w*uz)"
rt.F[3] = "-(u*Dx(w) + w*wz)"

def LL(d_trans):
    return np.array([[  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0],
                     [  0,  0, 1.,  0,  0,  0],
                     [  0,  0,  0, 1.,  0,  0],
                     [  0,  0,  0,  0,  0,  0],
                     [  0,  0,  0,  0,  0,  0]])
rt.LL = LL

def LR(d_trans):
    Dx = d_trans[0]
    if Dx == 0.:
        return np.array([[  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0, 1.,  0,  0,  0],
                         [ 1.,  0,  0,  0,  0,  0]])
    else:
        return np.array([[  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0,  0,  0,  0,  0],
                         [  0,  0, 1.,  0,  0,  0],
                         [  0,  0,  0, 1.,  0,  0]])
rt.LR = LR

# Build solver
ts = timesteppers.CNAB3
int = Integrator(rt, domain, ts)

# Initial conditions
x = domain.grids[0]
z = domain.grids[1]

u  = int.state['u']
w  = int.state['w']
rho = int.state['rho']

rho['X'] = At * np.tanh(z/width + pert*np.cos(2*np.pi*x))

# Integration parameters
int.dt = 2e-3
int.sim_stop_time = int.dt * 1500
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
w_list = [np.copy(w['X'])]
r_list = [np.copy(rho['X'])]
copy_cadence = 5

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
        r_list.append(np.copy(rho['X']))
        print('Iteration: %i, Time: %e' %(int.iteration, int.time))

if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    u_list.append(np.copy(u['X']))
    w_list.append(np.copy(w['X']))
    r_list.append(np.copy(rho['X']))

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
shelf['rho'] = np.array(r_list)
shelf.close()

