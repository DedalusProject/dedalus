"""Test outputs of various dimensionality."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers
import h5py
import tempfile
import pathlib


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('output_scales', [1, 3/2])
def test_cartesian_output(dtype, dealias, output_scales):
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
    problem.add_equation((dt(u) + u, 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0)]
    with tempfile.TemporaryDirectory(dir='.') as tempdir:
        tempdir = pathlib.Path(tempdir).stem
        output = solver.evaluator.add_file_handler(tempdir, iter=1)
        for task in tasks:
            output.add_task(task, layout='g', name=str(task), scales=output_scales)
        solver.evaluator.evaluate_handlers([output])
        # Check solution
        #post.merge_process_files('test_output')
        errors = []
        with h5py.File(f'{tempdir}/{tempdir}_s1/{tempdir}_s1_p0.h5', mode='r') as file:
            for task in tasks:
                task_saved = file['tasks'][str(task)][-1]
                task = task.evaluate()
                task.require_scales(output_scales)
                errors.append(np.max(np.abs(task['g'] - task_saved)))
    assert np.allclose(errors, 0)

