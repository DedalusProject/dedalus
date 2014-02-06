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

    x_basis = Chebyshev(N, interval=(0., 30.))
    domain = Domain([x_basis], grid_dtype=np.float64)

    return domain


def dedalus_solution(a, b, N):
    """Use Dedalus to solve the Airy equation."""

    # Problem
    bessel = ParsedProblem(axis_names=['x',],
                           field_names=['y', 'yx'],
                           param_names=['a', 'b'])
    bessel.add_equation("x**2*dx(yx) + x*yx + (x**2 - a**2)*y = 0")
    bessel.add_equation("dx(y) - yx = 0")
    bessel.add_left_bc("y = 0")
    bessel.add_right_bc("y = b")

    # Domain
    domain = dedalus_domain(N)

    # Parameters
    bessel.parameters['a'] = a
    bessel.parameters['b'] = b
    bessel.expand(domain, order=3)

    # Solve
    bvp = LinearBVP(bessel, domain)
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
    c = b / Jv[0]
    y_exact = c * Jv

    return y_exact

