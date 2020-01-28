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
    x, y = domain.all_grids()
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


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Ny', [64])
@pytest.mark.parametrize('Nx', [8])
@pytest.mark.parametrize('y_basis_class', [de.Fourier, de.SinCos])
@pytest.mark.parametrize('x_basis_class', [de.Fourier, de.SinCos])
@bench_wrapper
def test_poisson_2d_periodic_firstorder(benchmark, x_basis_class, y_basis_class, Nx, Ny, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, interval=(0, 2*np.pi))
    y_basis = y_basis_class('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    F.meta['x']['parity'] = -1
    F.meta['y']['parity'] = -1
    x, y = domain.all_grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u','ux','uy'])
    problem.meta['u']['x']['parity'] = -1
    problem.meta['u']['y']['parity'] = -1
    problem.meta['ux']['x']['parity'] = 1
    problem.meta['ux']['y']['parity'] = -1
    problem.meta['uy']['x']['parity'] = -1
    problem.meta['uy']['y']['parity'] = 1
    problem.parameters['F'] = F
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("uy - dy(u) = 0")
    problem.add_equation("dx(ux) + dy(uy) = F", condition="(nx != 0) or (ny != 0)")
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


def DoubleLegendre(name, N, interval=(-1,1), dealias=1):
    N0 = int(N // 2)
    N1 = N - N0
    L = interval[1] - interval[0]
    int0 = (interval[0], interval[0] + L/2)
    int1 = (interval[0] + L/2, interval[1])
    b0 = de.Legendre('b0', N0, interval=int0, dealias=dealias)
    b1 = de.Legendre('b1', N1, interval=int1, dealias=dealias)
    return de.Compound(name, (b0, b1), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Ny', [64])
@pytest.mark.parametrize('Nx', [8])
@pytest.mark.parametrize('y_basis_class', [de.Chebyshev, de.Legendre, DoubleChebyshev, DoubleLegendre])
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
    x, y = domain.all_grids()
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


def DoubleLaguerre(name, N, center=0.0, stretch=1.0, dealias=1):
    b0 = de.Laguerre('b0', int(N//2), edge=center, stretch=-stretch, dealias=dealias)
    b1 = de.Laguerre('b1', int(N//2), edge=center, stretch=stretch, dealias=dealias)
    return de.Compound(name, (b0, b1), dealias=dealias)


def LCCL(name, N, center=0.0, stretch=1.0, cwidth=1.0, dealias=1):
    b1 = de.Laguerre('b1', int(N//4), edge=center-cwidth, stretch=-stretch, dealias=dealias)
    b2 = de.Chebyshev('b2', int(N//4), interval=(center-cwidth, center), dealias=dealias)
    b3 = de.Chebyshev('b3', int(N//4), interval=(center, center+cwidth), dealias=dealias)
    b4 = de.Laguerre('b4', int(N//4), edge=center+cwidth, stretch=stretch, dealias=dealias)
    return de.Compound(name, (b1, b2, b3, b4), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [de.Hermite, DoubleLaguerre, LCCL])
@bench_wrapper
def test_gaussian_free(benchmark, x_basis_class, Nx, dtype):
    # Stretch Laguerres
    if x_basis_class is de.Hermite:
        stretch = 1.0
    else:
        stretch = 0.1
    # Bases and domain
    x_basis = x_basis_class('x', Nx, center=0, stretch=stretch)
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['pi'] = np.pi
    problem.add_equation("dx(u) + 2*x*u = 0", tau=True)
    problem.add_bc("integ(u) = sqrt(pi)")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    x = domain.grid(0)
    u_true = np.exp(-x**2)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [de.Hermite, DoubleLaguerre, LCCL])
@bench_wrapper
def test_gaussian_forced(benchmark, x_basis_class, Nx, dtype):
    # Stretch Laguerres
    if x_basis_class is de.Hermite:
        stretch = 1.0
    else:
        stretch = 0.1
    # Bases and domain
    x_basis = x_basis_class('x', Nx, center=0, stretch=stretch)
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


def ChebLag(name, N, edge=0.0, stretch=1.0, cwidth=1.0, dealias=1):
    b1 = de.Chebyshev('b1', int(N//2), interval=(edge, edge+cwidth), dealias=dealias)
    b2 = de.Laguerre('b2', int(N//2), edge=edge+cwidth, stretch=stretch, dealias=dealias)
    return de.Compound(name, (b1, b2), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Laguerre, ChebLag])
@bench_wrapper
def test_exponential_free(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, edge=0)
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


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Laguerre, ChebLag])
@bench_wrapper
def test_exponential_forced(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, edge=0)
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x = domain.grid(0)
    F['g'] = - np.exp(-x)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['F'] = F
    problem.add_equation("dx(u) = F")
    problem.add_bc("left(u) = 1")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    x = domain.grid(0)
    u_true = np.exp(-x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [DoubleLaguerre, LCCL])
@bench_wrapper
def test_double_exponential_free(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, center=0)
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # NCC
    x = domain.grid(0)
    sign_x = domain.new_field()
    sign_x.meta['x']['envelope'] = False
    sign_x['g'] = np.sign(x)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['sign_x'] = sign_x
    problem.add_equation("dx(u) + sign_x*u = 0", tau=True)
    problem.add_equation("integ(u) = 2")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.exp(-np.sign(x)*x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [DoubleLaguerre, LCCL])
@bench_wrapper
def test_double_exponential_forced(benchmark, x_basis_class, Nx, dtype):
    # Bases and domain
    x_basis = x_basis_class('x', Nx, center=0)
    domain = de.Domain([x_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x = domain.grid(0)
    F['g'] = -np.sign(x)*np.exp(-np.sign(x)*x)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['F'] = F
    problem.add_equation("dx(u) = F", tau=False)
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.exp(-np.sign(x)*x)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)

