"""
Test 1D EVP.
"""

import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier, basis.RealFourier])
def test_heat_1d_periodic(x_basis_class, Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.EVP([u], a)
    problem.add_equation((a*u + dx(dx(u)), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem, matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])
    # Check solution
    k = xb.wavenumbers
    assert np.allclose(solver.eigenvalues, k**2)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_waves_1d(x_basis_class, Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    τ1 = field.Field(name='τ1', dist=d, dtype=dtype)
    τ2 = field.Field(name='τ2', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    xb2 = dx(dx(u)).domain.bases[0]
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1]=1
    P2['c'][-2]=1

    problem = problems.EVP([u,τ1,τ2], a)
    problem.add_equation((a*u + dx(dx(u)) + P1*τ1 + P2*τ2, 0))
    problem.add_equation((u(x=0), 0))
    problem.add_equation((u(x=np.pi), 0))

    # Solver
    solver = solvers.EigenvalueSolver(problem, matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])
    i_sort = np.argsort(solver.eigenvalues)
    sorted_eigenvalues = solver.eigenvalues[i_sort]
    # Check solution
    Nmodes = Nx//4
    k = np.arange(Nmodes)+1
    assert np.allclose(sorted_eigenvalues[:Nmodes], k**2)
