"""
Test 1D EVP.
"""

import pytest
import numpy as np
import functools
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)


def bench_wrapper(test):
    @functools.wraps(test)
    def wrapper(benchmark, *args, **kw):
        benchmark.pedantic(test, args=(None,)+args, kwargs=kw)
    return wrapper


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


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev, de.Legendre, DoubleChebyshev, DoubleLegendre])
@bench_wrapper
def test_wave_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = int(Nx // 3)
    # Domain
    x_basis = x_basis_class('x', Nx, interval=(-1, 1))
    domain = de.Domain([x_basis], np.float64)
    # Problem
    problem = de.EVP(domain, variables=['u', 'ux'], eigenvalue='k2')
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("dx(ux) + k2*u = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("right(u) = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
    # Sort eigenmodes by eigenvalue
    order = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = ((1 + n) * np.pi / 2)**2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev, de.Legendre, DoubleChebyshev])
@bench_wrapper
def test_wave_sparse_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 5
    # Domain
    x_basis = x_basis_class('x', Nx, interval=(-1, 1))
    domain = de.Domain([x_basis], np.float64)
    # Problem
    problem = de.EVP(domain, variables=['u', 'ux'], eigenvalue='k2')
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("dx(ux) + k2*u = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("right(u) = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve_sparse(solver.pencils[0], n_comp, target=0)
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
    # Sort eigenmodes by eigenvalue
    order = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = ((1 + n) * np.pi / 2)**2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [de.Hermite])
@bench_wrapper
def test_qho_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 10
    # Stretch Laguerres
    if x_basis_class is de.Hermite:
        stretch = 1.0
    else:
        stretch = 0.4
    # Domain
    x_basis = x_basis_class('x', Nx, center=0, stretch=stretch)
    domain = de.Domain([x_basis], np.float64)
    # Problem
    problem = de.EVP(domain, variables=['ψ', 'ψx'], eigenvalue='E')
    problem.substitutions["V"] = "x**2 / 2"
    problem.substitutions["H(ψ,ψx)"] = "-dx(ψx)/2 + V*ψ"
    problem.add_equation("ψx - dx(ψ) = 0", tau=False)
    problem.add_equation("H(ψ,ψx) - E*ψ = 0", tau=False)
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
    # Sort eigenmodes by eigenvalue
    order = np.argsort(np.abs(solver.eigenvalues))
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = n + 1/2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)


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


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [DoubleLaguerre, LCCL])
@bench_wrapper
def test_qho_stretch_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 10
    # Domain
    x_basis = x_basis_class('x', Nx, center=0, stretch=1.0)
    domain = de.Domain([x_basis], np.float64)
    # Problem
    sign_x = domain.new_field()
    sign_x.meta['x']['envelope'] = False
    sign_x['g'] = np.sign(domain.grid(0))
    problem = de.EVP(domain, variables=['ψ', 'ψx'], eigenvalue='E')
    problem.parameters['sign_x'] = sign_x
    problem.substitutions["V"] = "abs(x) / 2"
    problem.substitutions["H(ψ,ψx)"] = "-(4*sign_x*x*dx(ψx) + 2*sign_x*ψx)/2 + V*ψ"
    problem.add_equation("ψx - dx(ψ) = 0", tau=False)
    problem.add_equation("H(ψ,ψx) - E*ψ = 0", tau=False)
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
    # Sort eigenmodes by eigenvalue
    order = np.argsort(np.abs(solver.eigenvalues))
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = n + 1/2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)


def ChebLag(name, N, edge=0.0, stretch=1.0, cwidth=1.0, dealias=1):
    b1 = de.Chebyshev('b1', int(N//2), interval=(edge, edge+cwidth), dealias=dealias)
    b2 = de.Laguerre('b2', int(N//2), edge=edge+cwidth, stretch=stretch, dealias=dealias)
    return de.Compound(name, (b1, b2), dealias=dealias)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [64])
@pytest.mark.parametrize('x_basis_class', [de.Laguerre, ChebLag])
@bench_wrapper
def test_half_qho_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 10
    stretch = 0.1
    # Domain
    x_basis = x_basis_class('x', Nx, edge=0, stretch=stretch)
    domain = de.Domain([x_basis], np.float64)
    # Problem
    problem = de.EVP(domain, variables=['ψ', 'ψx'], eigenvalue='E')
    problem.substitutions["V"] = "x**2 / 2"
    problem.substitutions["H(ψ,ψx)"] = "-dx(ψx)/2 + V*ψ"
    problem.add_equation("ψx - dx(ψ) = 0", tau=True)
    problem.add_equation("H(ψ,ψx) - E*ψ = 0", tau=False)
    problem.add_bc("left(ψ) = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
    # Sort eigenmodes by eigenvalue
    order = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = 2*n + 3/2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)

