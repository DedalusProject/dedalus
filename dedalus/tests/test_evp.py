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


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev])
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
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev])
@bench_wrapper
def test_wave_sparse_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 8
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
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Hermite])
@bench_wrapper
def test_qho_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = int(Nx // 3)
    # Domain
    x_basis = x_basis_class('x', Nx, interval=(-1, 1))
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
    order = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[order]
    solver.eigenvectors = solver.eigenvectors[:, order]
    # Check solution
    n = np.arange(n_comp)
    exact_eigenvalues = n + 1/2
    assert np.allclose(solver.eigenvalues[:n_comp], exact_eigenvalues)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [128])
@pytest.mark.parametrize('x_basis_class', [de.Laguerre])
@bench_wrapper
def test_half_qho_dense_evp(benchmark, x_basis_class, Nx, dtype):
    n_comp = 5
    # Domain
    x_basis = x_basis_class('x', Nx, interval=(0, 1))
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

