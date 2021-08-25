"""
Compute Bessel function by solving the Bessel equation:

x**2 f_xx + x f_x + (x**2 - A**2) f = 0
f(0) = 0
f(30) = B

"""

import numpy as np
from scipy import special

from ... import public as de


default_params = {}
default_params['A'] = 2.
default_params['B'] = 1.
default_params['N'] = 65


def dedalus_domain(N):
    """Construct Dedalus domain for solving the Airy equation."""
    x_basis = de.Chebyshev('x', N, interval=(0., 30.))
    domain = de.Domain([x_basis], grid_dtype=np.float64)
    return domain


def dedalus_solution(A, B, N):
    """Use Dedalus to solve the Airy equation."""
    # Domain
    domain = dedalus_domain(N)
    # Problem
    problem = de.LBVP(domain, variables=['f', 'fx'])
    problem.parameters['A'] = A
    problem.parameters['B'] = B
    problem.add_equation("x**2*dx(fx) + x*fx + (x**2 - A**2)*f = 0")
    problem.add_equation("dx(f) - fx = 0")
    problem.add_bc("left(f) = 0")
    problem.add_bc("right(f) = B")
    # Solve
    solver = problem.build_solver()
    solver.solve()
    return np.copy(solver.state['f']['g'])


def exact_solution(A, B, N):
    """Use scipy to construct exact solution to the Airy equation."""
    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)
    # Compute Bessel function on grid
    Jv = special.jv(A, x)
    # Solve for coefficient using boundary condition
    c = B / special.jv(A, 30)
    y_exact = c * Jv
    return y_exact


def test_bessel(params=default_params):
    """Compare Dedalus and exact results."""
    dedalus = dedalus_solution(**params)
    exact = exact_solution(**params)
    assert np.allclose(dedalus, exact)


