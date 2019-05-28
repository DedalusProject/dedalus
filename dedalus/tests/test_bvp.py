"""
Test 2D BVP with various bases and dtypes.
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


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Ny', [64])
@pytest.mark.parametrize('Nx', [8])
@pytest.mark.parametrize('y_basis_class', [de.Fourier, de.SinCos])
@pytest.mark.parametrize('x_basis_class', [de.Fourier, de.SinCos])
@bench_wrapper
def test_poisson_2d_periodic(benchmark, x_basis_class, y_basis_class, Nx, Ny, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 2*np.pi))
    y_basis = y_basis_class('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    F.meta['x']['parity'] = -1
    F.meta['y']['parity'] = -1
    x, y = domain.grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.meta['u']['x']['parity'] = -1
    problem.meta['u']['y']['parity'] = -1
    problem.parameters['F'] = F
    problem.add_equation("dx(dx(u)) + dy(dy(u)) = F", condition="(nx != 0) or (ny != 0)")
    problem.add_equation("u = 0", condition="(nx == 0) and (ny == 0)")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.sin(x) * np.sin(y)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


def DoubleChebyshev(name, N, interval=(-1,1), dealias=1):
    N0 = int(N // 2)
    N1 = N - N0
    L = interval[1] - interval[0]
    int0 = (interval[0], interval[0] + L/2)
    int1 = (interval[0] + L/2, interval[1])
    b0 = de.Chebyshev('b0', N0, interval=int0, dealias=dealias)
    b1 = de.Chebyshev('b1', N1, interval=int1, dealias=dealias)
    return de.Compound(name, (b0, b1), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Ny', [64])
@pytest.mark.parametrize('Nx', [8])
@pytest.mark.parametrize('y_basis_class', [de.Chebyshev, DoubleChebyshev])
@pytest.mark.parametrize('x_basis_class', [de.Fourier, de.SinCos])
@bench_wrapper
def test_poisson_2d_nonperiodic(benchmark, x_basis_class, y_basis_class, Nx, Ny, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 2*np.pi))
    y_basis = y_basis_class('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    F.meta['x']['parity'] = -1
    x, y = domain.grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u','uy'])
    problem.meta['u']['x']['parity'] = -1
    problem.meta['uy']['x']['parity'] = -1
    problem.parameters['F'] = F
    problem.add_equation("uy - dy(u) = 0")
    problem.add_equation("dx(dx(u)) + dy(uy) = F")
    problem.add_bc("left(u) - right(u) = 0")
    problem.add_bc("left(uy) - right(uy) = 0", condition="nx != 0")
    problem.add_bc("left(u) = 0", condition="nx == 0")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.sin(x) * np.sin(y)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


def DoubleLaguerre(name, N, interval=(-1,1), dealias=1):
    N0 = int(N // 2)
    N1 = N - N0
    L = interval[1] - interval[0]
    C = (interval[0] + interval[1]) / 2
    int0 = (C, C + L/2)
    int1 = (C, C - L/2)
    b0 = de.Chebyshev('b0', N0, interval=int0, dealias=dealias)
    b1 = de.Chebyshev('b1', N1, interval=int1, dealias=dealias)
    return de.Compound(name, (b0, b1), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Hermite])
@bench_wrapper
def test_gaussian(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(-1, 1))
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x = domain.grid(0)
    F['g'] = -2*x*np.exp(-x**2)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['F'] = F
    problem.add_equation("dx(u) = F", tau=False)
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.exp(-x**2)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Laguerre])
@bench_wrapper
def test_exponential(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 1))
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.add_equation("dx(u) + u = 0")
    problem.add_bc("left(u) = 1")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    x = domain.grid(0)
    u_true = np.exp(-x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)

