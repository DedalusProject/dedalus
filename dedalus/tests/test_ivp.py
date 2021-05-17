"""
Test 1D IVP with various timesteppers.
"""

import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('timestepper', timesteppers.schemes.values())
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier])
def test_heat_1d_periodic(x_basis_class, Nx, timestepper, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(xb,), dtype=dtype)
    F['g'] = -np.sin(x)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    dt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((-dt(u) + dx(dx(u)), F))
    # Solver
    solver = solvers.InitialValueSolver(problem, timestepper)
    dt = 1e-5
    iter = 10
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)
