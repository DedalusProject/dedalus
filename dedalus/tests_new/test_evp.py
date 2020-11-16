"""
Test 1D EVP.
"""

import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier])
def test_heat_1d_periodic(x_basis_class, Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.EVP([u], a)
    problem.add_equation((a*u + dx(dx(u)), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem)
    solver.solve_dense(solver.subproblems[1])
    # Check solution
    print(solver.eigenvalues)
    raise
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)
