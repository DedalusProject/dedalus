"""
Compute Bessel function by solving the Bessel equation:

x**2 y_xx + x y_x + (x**2 - a**2) y = 0
y(0) = 0
y(30) = b

"""

import numpy as np
from scipy import special

from ...public import *


default_params = {}
default_params['a'] = 2.
default_params['b'] = 1.
default_params['N'] = 65


def dedalus_domain(N):
    """Construct Dedalus domain for solving the Airy equation."""

    x_basis = Chebyshev('x', N, interval=(0., 30.))
    domain = Domain([x_basis], grid_dtype=np.float64)

    return domain


def dedalus_solution(a, b, N):
    """Use Dedalus to solve the Airy equation."""

    # Domain
    domain = dedalus_domain(N)

    # Problem
    bessel = BVP(domain, variables=['y', 'yx'])
    bessel.parameters['a'] = a
    bessel.parameters['b'] = b
    bessel.add_equation("x**2*dx(yx) + x*yx + (x**2 - a**2)*y = 0")
    bessel.add_equation("dx(y) - yx = 0")
    bessel.add_bc("left(y) = 0")
    bessel.add_bc("right(y) = b")

    # Solve
    bvp = solvers.LinearBVP(bessel)
    bvp.solve()

    return np.copy(bvp.state['y']['g'])


def exact_solution(a, b, N):
    """Use scipy to construct exact solution to the Airy equation."""

    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)

    # Compute Bessel function on grid
    Jv = special.jv(a, x)

    # Solve for coefficient using boundary condition
    c = b / special.jv(a, 30)
    y_exact = c * Jv

    return y_exact

