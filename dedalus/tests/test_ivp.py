"""
Test 1D IVP with various timesteppers.
"""

import pytest
import numpy as np
import functools
from dedalus import public as de


def bench_wrapper(test):
    @functools.wraps(test)
    def wrapper(benchmark, *args, **kw):
        benchmark.pedantic(test, args=(None,)+args, kwargs=kw)
    return wrapper


@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('timestepper', de.timesteppers.schemes.values())
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Fourier])
@bench_wrapper
def test_heat_1d_periodic(benchmark, x_basis_class, Nx, timestepper, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    F.meta['x']['parity'] = -1
    x = domain.grid(0)
    F['g'] = -np.sin(x)
    # Problem
    problem = de.IVP(domain, variables=['u'])
    problem.meta['u']['x']['parity'] = -1
    problem.parameters['F'] = F
    problem.add_equation("-dt(u) + dx(dx(u)) = F")
    # Solver
    solver = problem.build_solver(timestepper)
    dt = 1e-5
    iter = 10
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('timestepper', de.timesteppers.schemes.values())
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev])
@bench_wrapper
def test_heat_1d_nonperiodic(benchmark, x_basis_class, Nx, timestepper, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x = domain.grid(0)
    F['g'] = -np.sin(x)
    # Problem
    problem = de.IVP(domain, variables=['u','ux'])
    problem.parameters['F'] = F
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("-dt(u) + dx(ux) = F")
    problem.add_bc("left(u) - right(u) = 0")
    problem.add_bc("left(ux) - right(ux) = 0")
    # Solver
    solver = problem.build_solver(timestepper)
    dt = 1e-5
    iter = 10
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)

