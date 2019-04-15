"""
Test 1D NLBVP with various dtypes.
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


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev])
@bench_wrapper
def test_sin_nlbvp(benchmark, x_basis_class, Nx, dtype):
    # Parameters
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Build domain
    x_basis = x_basis_class('x', Nx, interval=(0, 1), dealias=2)
    domain = de.Domain([x_basis], np.float64)
    # Setup problem
    problem = de.NLBVP(domain, variables=['u'], ncc_cutoff=ncc_cutoff)
    problem.add_equation("dx(u) = sqrt(1 - u**2)")
    problem.add_bc("left(u) = 0")
    # Setup initial guess
    solver = problem.build_solver()
    x = domain.grid(0)
    u = solver.state['u']
    u['g'] = x
    # Iterations
    pert = solver.perturbations.data
    pert.fill(1+tolerance)
    while np.sum(np.abs(pert)) > tolerance:
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    # Check solution
    u_true = np.sin(x)
    u.set_scales(1)
    assert np.allclose(u['g'], u_true)

