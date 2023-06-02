"""Test outputs of various dimensionality."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers
import h5py
import tempfile
import pathlib
from dedalus.tools.cache import CachedFunction


# Check if parallel h5py is available
handler_options = ['gather', 'virtual']
if h5py.get_config().mpi:
    handler_options.append('mpio')
else:
    handler_options.append(pytest.param('mpio', marks=pytest.mark.xfail(reason="parallel h5py not available")))


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('output_scales', [1, 3/2, 2,
    pytest.param(1/2, marks=pytest.mark.xfail(reason="evaluator not copying correctly for scales < 1"))])
@pytest.mark.parametrize('output_layout', ['g', 'c'])
@pytest.mark.parametrize('parallel', handler_options)
def test_cartesian_output(dtype, dealias, output_scales, output_layout, parallel):
    Nx = Ny = Nz = 16
    Lx = Ly = Lz = 2 * np.pi
    # Bases
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,))
    Fourier = {np.float64: basis.RealFourier, np.complex128: basis.ComplexFourier}[dtype]
    xb = Fourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    yb = Fourier(c.coords[1], size=Ny, bounds=(0, Ly), dealias=dealias)
    zb = basis.ChebyshevT(c.coords[2], size=Nz, bounds=(0, Lz), dealias=dealias)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,yb,zb), dtype=dtype)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((dt(u), 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0)]
    with tempfile.TemporaryDirectory(dir='.') as tempdir:
        tempdir = pathlib.Path(tempdir).stem
        output = solver.evaluator.add_file_handler(tempdir, iter=1, parallel=parallel)
        for task in tasks:
            output.add_task(task, layout=output_layout, name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        # Check solution
        errors = []
        with h5py.File(f'{tempdir}/{tempdir}_s1.h5', mode='r') as file:
            for task in tasks:
                task_saved = file['tasks'][str(task)][-1]
                task = task.evaluate()
                task.change_scales(output_scales)
                errors.append(np.max(np.abs(task[output_layout] - task_saved)))
    assert np.allclose(errors, 0)


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, k, dealias, dtype):
    radius_ball = 1.5
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, k, dealias, dtype):
    radii_shell = (0.5, 1.5)
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('output_scales', [1, 3/2, 2,
    pytest.param(1/2, marks=pytest.mark.xfail(reason="evaluator not copying correctly for scales < 1"))])
@pytest.mark.parametrize('parallel', handler_options)
def test_spherical_output(Nphi, Ntheta, Nr, k, dealias, dtype, basis, output_scales, parallel):
    # Basis
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    u.preset_scales(b.domain.dealias)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((dt(u), 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222, matrix_coupling=[False, False, True])
    # Output
    tasks = [u(phi=np.pi), u(theta=np.pi/2), u(r=1.0), u]
    with tempfile.TemporaryDirectory(dir='.') as tempdir:
        tempdir = pathlib.Path(tempdir).stem
        output = solver.evaluator.add_file_handler(tempdir, iter=1, parallel=parallel)
        for task in tasks:
            output.add_task(task, layout='g', name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        # Check solution
        errors = []
        with h5py.File(f'{tempdir}/{tempdir}_s1.h5', mode='r') as file:
            for task in tasks:
                task_saved = file['tasks'][str(task)][-1]
                task = task.evaluate()
                if not isinstance(task, field.LockedField):
                    task.change_scales(output_scales)
                print(task['g'].shape, task_saved.shape, dealias, output_scales)
                errors.append(np.max(np.abs(task['g'] - task_saved)))
    assert np.allclose(errors, 0)

