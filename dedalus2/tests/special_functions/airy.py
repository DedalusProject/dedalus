"""
Compute Airy functions by solving the Airy equation:

y_xx - (a + bx) y = 0
y(-1) = c
y(+1) = e

"""

import numpy as np
from scipy import special

from ...public import *


default_params = {}
default_params['a'] = 100.
default_params['b'] = 500.
default_params['c'] = 1.
default_params['e'] = 1.
default_params['N'] = 128


def dedalus_domain(N):
    """Construct Dedalus domain for solving the Airy equation."""

    x_basis = Chebyshev('x', N, interval=(-1., 1.))
    domain = Domain([x_basis], grid_dtype=np.float64)

    return domain


def dedalus_solution(a, b, c, e, N):
    """Use Dedalus to solve the Airy equation."""

    # Domain
    domain = dedalus_domain(N)

    # Problem
    airy = BVP(domain, variables=['y', 'yx'])
    airy.parameters['a'] = a
    airy.parameters['b'] = b
    airy.parameters['c'] = c
    airy.parameters['e'] = e
    airy.add_equation("dx(yx) - (a + b*x)*y = 0")
    airy.add_equation("yx - dx(y) = 0")
    airy.add_bc("left(y) = c")
    airy.add_bc("right(y) = e")

    # Solve
    bvp = solvers.LinearBVP(airy)
    bvp.solve()

    return np.copy(bvp.state['y']['g'])


def exact_solution(a, b, c, e, N):
    """Use scipy to construct exact solution to the Airy equation."""

    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)

    # Compute Airy functions on rescaled grid
    z = lambda x: (a + b*x) * b**(-2/3)
    Ai, Aip, Bi, Bip = special.airy(z(x))
    LAi, LAip, LBi, LBip = special.airy(z(-1))
    RAi, RAip, RBi, RBip = special.airy(z(1))

    # Solve for coefficients using boundary conditions
    L = np.array([[LAi, LBi],
                  [RAi, RBi]])
    R = np.array([c, e])
    c1, c2 = np.linalg.solve(L, R)
    y_exact = c1*Ai + c2*Bi

    return y_exact

