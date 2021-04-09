"""
Test 1D IVP with various timesteppers.
"""

import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('dt', [1e-5])
@pytest.mark.parametrize('timestepper', timesteppers.schemes.values())
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier])
def test_heat_1d_periodic(x_basis_class, Nx, timestepper, dt, dtype):
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
    ddt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((-ddt(u) + dx(dx(u)), F))
    # Solver
    solver = solvers.InitialValueSolver(problem, timestepper)
    dt = 1e-5
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dt', [1e-5])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_heat_1d_periodic_chebyshev(x_basis_class, Nx, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(xb,), dtype=dtype)
    F['g'] = -np.sin(x)
    τu1 = field.Field(name='τu1', dist=d, dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, dtype=dtype)
    xb2 = xb._new_a_b(1.5,1.5)
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1] = 1
    P2['c'][-2] = 1
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    ddt = operators.TimeDerivative
    problem = problems.IVP([u,τu1,τu2])
    problem.add_equation((-ddt(u) + P1*τu1 + P2*τu2+ dx(dx(u)), F))
    problem.add_equation((u(x=0)-u(x=2*np.pi),0))
    problem.add_equation((dx(u)(x=0)-dx(u)(x=2*np.pi),0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dt', [1e-5,1e-6,1e-7])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('k_ncc', [4])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_heat_1d_periodic_ncc(x_basis_class, Nx, k_ncc, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(xb,), dtype=dtype)
    F['g'] = -np.sin(x)
    τu1 = field.Field(name='τu1', dist=d, dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, dtype=dtype)
    xb2 = xb._new_a_b(1.5,1.5)
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1] = 1
    P2['c'][-2] = 1
    # Problem
    ncc = field.Field(name='ncc', dist=d, bases=(xb,), dtype=dtype)
    ncc['g'] = k_ncc
    for ik in np.arange(1,k_ncc+1):
        ncc['g'] += np.sqrt(ik/k_ncc)*np.cos(ik*x)
    dx = lambda A: operators.Differentiate(A, c)
    ddt = operators.TimeDerivative
    problem = problems.IVP([u,τu1,τu2])
    problem.add_equation((-ncc*ddt(u) + P1*τu1 +  P2*τu2 + ncc*dx(dx(u)), ncc*F))
    problem.add_equation((u(x=0)-u(x=2*np.pi),0))
    problem.add_equation((dx(u)(x=0)-dx(u)(x=2*np.pi),0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('dt', [1e-4])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_wave_1d_periodic(x_basis_class, Nx, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    k = 4
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    ut = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    u['g'] = u0 = np.sin(k*x)
    ut['g'] = 1j*k*u0
    τu1 = field.Field(name='τu1', dist=d, dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, dtype=dtype)
    xb2 = xb._new_a_b(1.5,1.5)
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1] = 1
    P2['c'][-2] = 1
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    ddt = operators.TimeDerivative
    problem = problems.IVP([ut,u,τu1,τu2])
    problem.add_equation((ddt(u) - ut, 0))
    problem.add_equation((ddt(ut) + P1*τu1 + P2*τu2  - dx(dx(u)), 0))
    problem.add_equation((u(x=0)-u(x=2*np.pi),0))
    problem.add_equation((dx(u)(x=0)-dx(u)(x=2*np.pi),0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    u_true = u0*np.exp(1j*k*solver.sim_time)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('dt', [1e-4, 1e-5, 1e-6])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('k_ncc', [4])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_wave_1d_periodic_ncc(x_basis_class, Nx, k_ncc, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    k = 4
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    ut = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    u['g'] = u0 = np.sin(k*x)
    ut['g'] = 1j*k*u0
    τu1 = field.Field(name='τu1', dist=d, dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, dtype=dtype)
    xb2 = xb._new_a_b(1.5,1.5)
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1] = 1
    P2['c'][-2] = 1
    # Problem
    ncc = field.Field(name='ncc', dist=d, bases=(xb2,), dtype=dtype)
    ncc['g'] = k_ncc
    for ik in np.arange(1,k_ncc+1):
        ncc['g'] += np.sqrt(ik/k_ncc)*np.cos(ik*x)
    dx = lambda A: operators.Differentiate(A, c)
    ddt = operators.TimeDerivative
    problem = problems.IVP([ut,u,τu1,τu2])
    problem.add_equation((ncc*ddt(u) - ncc*ut, 0))
    problem.add_equation((ncc*ddt(ut) + P1*τu1 + P2*τu2  - ncc*dx(dx(u)), 0))
    problem.add_equation((u(x=0)-u(x=2*np.pi),0))
    problem.add_equation((dx(u)(x=0)-dx(u)(x=2*np.pi),0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    u_true = u0*np.exp(1j*k*solver.sim_time)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('dt', [1e-5])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_1d_periodic_chebyshev(x_basis_class, Nx, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    k = 4
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    ut = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    u['g'] = u0 = np.sin(k*x)
    ut['g'] = 1j*k*u0
    # Problem
    ddt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((ddt(u), ut))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    u_true = u0*np.exp(1j*k*solver.sim_time)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('dt', [1e-5])
@pytest.mark.parametrize('timestepper', timesteppers.schemes)
@pytest.mark.parametrize('k_ncc', [4])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_1d_periodic_chebyshev_ncc(x_basis_class, Nx, k_ncc, timestepper, dt, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    k = 4
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    ut = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    u['g'] = u0 = np.sin(k*x)
    ut['g'] = 1j*k*u0
    # Problem
    ncc = field.Field(name='ncc', dist=d, bases=(xb,), dtype=dtype)
    ncc['g'] = k_ncc
    for ik in np.arange(1,k_ncc+1):
        ncc['g'] += np.sqrt(ik/k_ncc)*np.cos(ik*x)
    ddt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((ncc*ddt(u), ncc*ut))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.schemes[timestepper])
    iter = 100
    for i in range(iter):
        solver.step(dt)
    # Check solution
    u_true = u0*np.exp(1j*k*solver.sim_time)
    assert np.allclose(u['g'], u_true)
