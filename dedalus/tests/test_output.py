"""Test outputs of various dimensionality."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedFunction
import h5py
import tempfile
import pathlib


# Check if parallel h5py is available
parallel_range = ['gather', 'virtual']
if h5py.get_config().mpi:
    parallel_range.append('mpio')
else:
    parallel_range.append(pytest.param('mpio', marks=pytest.mark.xfail(reason="parallel h5py not available")))


dealias_range = [3/2]
dtype_range = [np.float64, np.complex128]
layout_range = ['g', 'c']
scales_range = [1, 3/2, pytest.param(1/2, marks=pytest.mark.xfail(reason="evaluator not copying correctly for scales < 1"))]


@pytest.mark.parametrize('N', [8])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('output_scales', scales_range)
@pytest.mark.parametrize('output_layout', layout_range)
@pytest.mark.parametrize('parallel', parallel_range)
def test_cartesian_output(N, dealias, dtype, output_scales, output_layout, parallel):
    """Test outputs of 0d/1d/2d/3d tasks in 3d FFC problem."""
    # Bases
    c = d3.CartesianCoordinates('x', 'y', 'z')
    d = d3.Distributor(c, dtype=dtype)
    xb = d3.Fourier(c.coords[0], size=N, bounds=(0, 1), dealias=dealias, dtype=dtype)
    yb = d3.Fourier(c.coords[1], size=N, bounds=(0, 1), dealias=dealias, dtype=dtype)
    zb = d3.Chebyshev(c.coords[2], size=N, bounds=(0, 1), dealias=dealias)
    x, y, z = d.local_grids(xb, yb, zb)
    # Fields
    u = d.Field(name='u', bases=(xb,yb,zb))
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    problem = d3.IVP([u])
    problem.add_equation("dt(u) = 0")
    solver = problem.build_solver("RK111")
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0)]
    with tempfile.TemporaryDirectory(dir='/tmp') as tempdir:
        output = solver.evaluator.add_file_handler(tempdir, iter=1, parallel=parallel)
        for task in tasks:
            output.add_task(task, layout=output_layout, name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        # Check solution
        with h5py.File(f'{tempdir}/{pathlib.Path(tempdir).stem}_s1.h5', mode='r') as file:
            for task in tasks:
                task_saved = file['tasks'][str(task)][-1]
                task = task.evaluate()
                task.change_scales(output_scales)
                assert np.allclose(task[output_layout], task_saved)


# stopped cleanup


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, k, dealias, dtype):
    radius_ball = 1.5
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor((c,))
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, k, dealias, dtype):
    radii_shell = (0.5, 1.5)
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor((c,))
    b = d3.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', [1])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('output_scales', [1, 3/2,
    pytest.param(1/2, marks=pytest.mark.xfail(reason="evaluator not copying correctly for scales < 1"))])
@pytest.mark.parametrize('output_layout', ['g'])
@pytest.mark.parametrize('parallel', parallel_range)
def test_spherical_output(Nphi, Ntheta, Nr, k, dealias, dtype, basis, output_scales, output_layout, parallel):
    # Basis
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    # Fields
    u = d3.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    u.preset_scales(b.domain.dealias)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = d3.TimeDerivative
    problem = d3.IVP([u])
    problem.add_equation((dt(u), 0))
    # Solver
    solver = problem.build_solver(d3.RK222, matrix_coupling=[False, False, True])
    # Output
    tasks = [u(phi=np.pi), u(theta=np.pi/2), u(r=1.0), u]
    with tempfile.TemporaryDirectory(dir='/tmp') as tempdir:
        output = solver.evaluator.add_file_handler(tempdir, iter=1, parallel=parallel)
        for task in tasks:
            output.add_task(task, layout=output_layout, name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        # Check solution
        errors = []
        with h5py.File(f'{tempdir}/{pathlib.Path(tempdir).stem}_s1.h5', mode='r') as file:
            for task in tasks:
                task_saved = file['tasks'][str(task)][-1]
                task = task.evaluate()
                if not isinstance(task, d3.LockedField):
                    task.change_scales(output_scales)
                errors.append(np.max(np.abs(task[output_layout] - task_saved)))
    assert np.allclose(errors, 0)

