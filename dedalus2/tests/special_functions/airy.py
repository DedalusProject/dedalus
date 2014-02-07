"""
Compute Airy functions by solving the Airy equation:

y_xx - (a + bx) y = 0
y(-1) = c
y(+1) = d

"""

import numpy as np
from scipy import special

from ...public import *


default_params = {}
default_params['a'] = 100.
default_params['b'] = 500.
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

    # Problem
    airy = ParsedProblem(axis_names=['x',],
                         field_names=['y', 'yx'],
                         param_names=['a', 'b', 'c' ,'d'])
    airy.add_equation("dx(yx) - (a + b*x)*y = 0")
    airy.add_equation("yx - dx(y) = 0")
    airy.add_left_bc("y = c")
    airy.add_right_bc("y = d")

    # Domain
    domain = dedalus_domain(N)

    # Parameters
    airy.parameters['a'] = a
    airy.parameters['b'] = b
    airy.parameters['c'] = c
    airy.parameters['d'] = d
    airy.expand(domain, order=2)

    # Solve
    bvp = solvers.LinearBVP(airy, domain)
    bvp.solve()

    return np.copy(bvp.state['y']['g'])


def exact_solution(a, b, c, d, N):
    """Use scipy to construct exact solution to the Airy equation."""

    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)

    # Compute Airy functions on rescaled grid
    z = (a + b*x) * b**(-2/3)
    Ai, Aip, Bi, Bip = special.airy(z)

    # Solve for coefficients using boundary conditions
    L = np.array([[ Ai[0],  Bi[0]],
                  [Ai[-1], Bi[-1]]])
    R = np.array([d, c])
    c1, c2 = np.linalg.solve(L, R)
    y_exact = c1*Ai + c2*Bi

    return y_exact

