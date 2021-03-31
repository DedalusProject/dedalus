"""Test outputs of various dimensionality."""

import os
print(os.__file__)
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
import shutil
import h5py


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
    u['g'] = np.sin(x)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((dt(u) + u, 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0)]
    output = solver.evaluator.add_file_handler('test_output', iter=1)
    for task in tasks:
        output.add_task(task, layout='g', name=str(task), scales=output_scales)
    solver.evaluator.evaluate_handlers([output])
    # Check solution
    #post.merge_process_files('test_output')
    errors = []
    with h5py.File('test_output/test_output_s1/test_output_s1_p0.h5', mode='r') as file:
        for task in tasks:
            task_saved = file['tasks'][str(task)][-1]
            task = task.evaluate()
            task.require_scales(output_scales)
            errors.append(np.max(np.abs(task['g'] - task_saved)))
    shutil.rmtree('test_output')
    assert np.allclose(errors, 0)

