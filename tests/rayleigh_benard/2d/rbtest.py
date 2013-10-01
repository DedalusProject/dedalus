

import numpy as np
import time
import shelve
from dedalus2.public import *


# Set domain
x_basis = Fourier(32, interval=[0., 4.])
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
rb.parameters['d'] = D[0]

rb.F[1] = "-1/Pr*(u*d(u)+w*uz)"
rb.F[2] = "-1/Pr*(u*d(w)-w*d(u))"
rb.F[3] = "-u*d(t)-w*tz"

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

T['X'] = 1e-1 * np.sin(np.pi * z) * np.random.randn(*T['X'].shape)

# integrate parameters
int.dt = 5e-4
int.sim_stop_time = 1.25
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# storage lists
t_list = [int.time]
u_list = [np.copy(u['X'])]
w_list = [np.copy(w['X'])]
T_list = [np.copy(T['X'])]
E_list = [np.sum(u['X']**2+w['X']**2+T['X']**2)]
copy_cadence = 20


int.timestepper.update_pencils(int.dt, int.iteration)


plt.figure(1, figsize=(10,10))
plt.imshow(np.log10(np.abs(int.pencils[0].LHS.todense())))
for i in range(1, 6):
    p = i * 32
    plt.axhline(p-0.5, ls='solid', c='k')
    plt.axvline(p-0.5, ls='solid', c='k')
plt.xlabel('j')
plt.ylabel('i')
plt.title('Chebyshev RB k=0 LHS')
plt.savefig('chebyshev_rb_k0_lhs_MOD.png')

