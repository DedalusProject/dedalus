"""
Compute Airy functions by solving the Airy equation:

f_xx - (A + Bx) f = 0
f(-1) = C
f(+1) = D

"""

import numpy as np
from scipy import special

from ... import public as de


default_params = {}
default_params['A'] = 100.
default_params['B'] = 500.
default_params['C'] = 1.
default_params['D'] = 1.
default_params['N'] = 128


def dedalus_domain(N):
    """Construct Dedalus domain for solving the Airy equation."""
    x_basis = de.Chebyshev('x', N, interval=(-1., 1.))
    domain = de.Domain([x_basis], grid_dtype=np.float64)
    return domain


def dedalus_solution(A, B, C, D, N):
    """Use Dedalus to solve the Airy equation."""
    # Domain
    domain = dedalus_domain(N)
    # Problem
    problem = de.LBVP(domain, variables=['f', 'fx'])
    problem.parameters['A'] = A
    problem.parameters['B'] = B
    problem.parameters['C'] = C
    problem.parameters['D'] = D
    problem.add_equation("dx(fx) - (A + B*x)*f = 0")
    problem.add_equation("fx - dx(f) = 0")
    problem.add_bc("left(f) = C")
    problem.add_bc("right(f) = D")
    # Solve
    solver = problem.build_solver()
    solver.solve()
    return np.copy(solver.state['f']['g'])


def exact_solution(A, B, C, D, N):
    """Use scipy to construct exact solution to the Airy equation."""
    # Setup Dedalus domain to get same grid
    domain = dedalus_domain(N)
    x = domain.grid(0)
    # Compute Airy functions on rescaled grid
    z = lambda x: (A + B*x) * B**(-2/3)
    Ai, Aip, Bi, Bip = special.airy(z(x))
    LAi, LAip, LBi, LBip = special.airy(z(-1))
    RAi, RAip, RBi, RBip = special.airy(z(1))
    # Solve for coefficients using boundary conditions
    L = np.array([[LAi, LBi],
                  [RAi, RBi]])
    R = np.array([C, D])
    c1, c2 = np.linalg.solve(L, R)
    y_exact = c1*Ai + c2*Bi
    return y_exact


def test_airy(params=default_params):
    """Compare Dedalus and exact results."""
    dedalus = dedalus_solution(**params)
    exact = exact_solution(**params)
    assert np.allclose(dedalus, exact)

