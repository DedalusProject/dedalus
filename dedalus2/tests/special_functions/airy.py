"""
Compute Airy functions by solving the Airy equation:

u_xx - (a + bx) u = 0
u(-1) = c
u(+1) = d

"""

import numpy as np
from scipy import special

from dedalus2.public import *


default_params = {}
default_params['a'] = 0.
default_params['b'] = 600.
default_params['c'] = 1.
default_params['d'] = 1.
default_params['N'] = 65


def dedalus_domain(N):
    """Construct Dedalus domain for solving the Airy equation."""

    x_basis = Chebyshev(N, interval=(-1., 1.))
    domain = Domain([x_basis], grid_dtype=np.float64)

    return domain


def dedalus_solution(a, b, c, d, N):
    """Use Dedalus to solve the Airy equation."""

    # First-order system:
    # dz(uz) - (a + bz) u = 0
    # dz(u) - uz = 0

    airy = Problem(['u', 'uz'], 2)

    airy.L0[0] = lambda d_trans: np.array([[-a, 0],
                                           [ 0,-1]])

    airy.L0[1] = lambda d_trans: np.array([[-b, 0],
                                           [ 0, 0]])

    airy.L1[0] = lambda d_trans: np.array([[0, 1],
                                           [1, 0]])

    # Boundary conditions:
    # u(L) = c
    # u(R) = d

    airy.LL = lambda d_trans: np.array([[1, 0],
                                        [0, 0]])

    airy.LR = lambda d_trans: np.array([[0, 0],
                                        [1, 0]])

    airy.b = lambda d_trans: np.array([c, d])

    # Solve
    domain = dedalus_domain(N)
    ts = timesteppers.SimpleSolve
    int = Integrator(airy, domain, ts)
    int.advance()
    u = int.state['u']

    return np.copy(u['g'])


def exact_solution(a, b, c, d, N):
    """Use scipy to construct exact solution to the Airy equation."""

    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)

    # Compute Airy functions on rescaled grid
    y = (a + b*x) * b**(-2/3)
    Ai, Aip, Bi, Bip = special.airy(y)

    # Solve for coefficients using boundary conditions
    L = np.array([[ Ai[0],  Bi[0]],
                  [Ai[-1], Bi[-1]]])
    R = np.array([d, c])
    c1, c2 = np.linalg.solve(L, R)
    u_exact = c1*Ai + c2*Bi

    return u_exact

