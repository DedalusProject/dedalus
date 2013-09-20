

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from dedalus2.public import *


## Airy test: u_zz + (a + bz)u = 0
#
# u_z - du = 0
# du_z + (a + bz) u = 0
#
a = 0.
b = -300.
c = 1.
d = 1.
airy = Problem(['u', 'du'], 2)
airy.L0[0] = lambda d_trans: np.array([[0., -1.],
                                        [a, 0.]])
airy.L0[1] = lambda d_trans: np.array([[0., 0.],
                                        [b, 0.]])
airy.L1[0] = lambda d_trans: np.array([[1., 0.],
                                        [0., 1.]])
airy.LL = np.array([[1., 0.],
                    [0., 0.]])
airy.LR = np.array([[0., 0.],
                    [1., 0.]])
airy.b = np.array([c, d])

# Set domain
x_basis = Chebyshev(32, interval=[-1., 1.])
domain = Domain([x_basis])

# Choose PDE and integrator
pde = airy
ts = timesteppers.SimpleSolve

# Build solver
int = Integrator(pde, domain, ts)

# Solve/integrate
int.dt = 1.
int.sim_stop_time = np.inf
int.advance()

# Exact solution
def exact(z):
    arg = -(a + b*z) / (-b)**(2./3.)
    Ai, Aip, Bi, Bip = scipy.special.airy(arg)
    L = np.array([[Ai[0], Bi[0]],
                  [Ai[-1], Bi[-1]]])
    R = np.array([d, c])
    c1, c2 = np.linalg.solve(L, R)
    u = c1*Ai + c2*Bi
    print(c1, c2)

    return u

# Plot
z = x_basis.grid
z_dense = np.linspace(1, -1, 10000)
u = int.state['u']['x'].real

fig = plt.figure(1)
fig.clear()

ax1 = fig.add_subplot(211)
ax1.plot(z_dense, exact(z_dense), '-k', label='Exact')
ax1.plot(z, u, 'ob', label='Numerical (%i)' %z.size)
ax1.legend(fontsize='small')

ax2 = fig.add_subplot(212)
ax2.axhline(0, c='k', ls='dashed')
ax2.plot(z, u - exact(z), 'o-r')
ax2.set_xlabel(r'$z$')
ax2.set_ylabel('Error = Numerical - Exact', size='small')

fig.suptitle(r"$u'' + (a + b z) u = 0, \quad u(-1) = c, \quad u(1) = d$")
ax1.set_title(r"$a = %.1f, \quad b = %.1f, \quad c = %.1f, \quad d = %.1f$" %(a,b,c,d), size='small')

plt.savefig('airy.png', dpi=200)

