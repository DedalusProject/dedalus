"""
Test 1D NLBVP.
"""
import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('basis_class', [basis.ChebyshevT, basis.Legendre])
def test_sin_nlbvp(basis_class, Nx, dtype):
    # Parameters
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis_class(c, size=Nx, bounds=(0, 1), dealias=2)
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    sqrt = lambda A: operators.UnaryGridFunctionField(np.sqrt, A)
    problem = problems.NLBVP([u], ncc_cutoff=ncc_cutoff)
    problem.add_equation((dx(u), sqrt(1-u*u)))
    problem.add_equation((u(x=0), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    u['g'] = x
    # Iterations
    pert = solver.perturbations.data
    pert.fill(1+tolerance)
    while np.sum(np.abs(pert)) > tolerance:
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    # Check solution
    u_true = np.sin(x)
    u.require_scales(1)
    assert np.allclose(u['g'], u_true)
