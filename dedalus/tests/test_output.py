"""
Test 1D IVP with various timesteppers.
"""

import pytest
import numpy as np
import functools
from dedalus import public as de
from dedalus.tools import post
import shutil
import h5py


def bench_wrapper(test):
    @functools.wraps(test)
    def wrapper(benchmark, *args, **kw):
        benchmark.pedantic(test, args=(None,)+args, kwargs=kw)
    return wrapper


@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('timestepper', [de.timesteppers.RK222])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [de.Chebyshev])
def test_1d_output(x_basis_class, Nx, timestepper, dtype):
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
    # Output
    output = solver.evaluator.add_file_handler('test_output', iter=1)
    output.add_task('u', layout='g', name='ug')
    # Loop
    dt = 1e-5
    iter = 10
    for i in range(iter):
        solver.step(dt)
    # Check solution
    post.merge_process_files('test_output')
    with h5py.File('test_output/test_output_s1.h5') as file:
        ug = file['tasks']['ug'][:]
        t = file['scales']['sim_time'][:]
    shutil.rmtree('test_output')
    amp = 1 - np.exp(-t[:, None])
    u_true = amp * np.sin(x[None, :])
    assert np.allclose(ug, u_true)

