"""Test outputs of various dimensionality."""

import os
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.parallel import Sync
from dedalus.tools.post import merge_virtual_analysis
import shutil
import h5py


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('output_scales', [1/2, 1, 3/2])
def test_cartesian_output(dtype, dealias, output_scales):
    Nx = Ny = Nz = 16
    Lx = Ly = Lz = 2 * np.pi
    # Bases
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,), mesh=(2,2))
    Fourier = {np.float64: basis.RealFourier, np.complex128: basis.ComplexFourier}[dtype]
    xb = Fourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    yb = Fourier(c.coords[1], size=Ny, bounds=(0, Ly), dealias=dealias)
    zb = Fourier(c.coords[2], size=Nz, bounds=(0, Lz), dealias=dealias)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,yb,zb), dtype=dtype)
    v = field.Field(name='v', dist=d, bases=(xb,yb,zb), tensorsig=(c,), dtype=dtype)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u, v])
    problem.add_equation((dt(u) + u, 0))
    problem.add_equation((dt(v) + v, 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0),
             v, v(x=0), v(y=0), v(z=0), v(x=0,y=0), v(x=0,z=0), v(y=0,z=0), v(x=0,y=0,z=0)]
    output = solver.evaluator.add_file_handler('test_output', iter=1)
    for task in tasks:
        output.add_task(task, layout='g', name=str(task), scales=output_scales)
    solver.evaluator.evaluate_handlers([output])
    # Check solution
    errors = []
    rank = d.comm.rank
    with h5py.File('test_output/test_output_s1/test_output_s1_p{}.h5'.format(rank), mode='r') as file:
        for task in tasks:
            task_saved = file['tasks'][str(task)][-1]
            task = task.evaluate()
            task.change_scales(output_scales)
            local_error = task['g'] - task_saved
            if local_error.size:
                errors.append(np.max(np.abs(task['g'] - task_saved)))
    with Sync() as sync:
        if sync.comm.rank == 0:
            shutil.rmtree('test_output')
    assert np.allclose(errors, 0)


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('output_scales', [1/2])
def test_cartesian_output_virtual(dtype, dealias, output_scales):
    Nx = Ny = Nz = 16
    Lx = Ly = Lz = 2 * np.pi
    # Bases
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,), mesh=(2,2))
    Fourier = {np.float64: basis.RealFourier, np.complex128: basis.ComplexFourier}[dtype]
    xb = Fourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    yb = Fourier(c.coords[1], size=Ny, bounds=(0, Ly), dealias=dealias)
    zb = Fourier(c.coords[2], size=Nz, bounds=(0, Lz), dealias=dealias)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,yb,zb), dtype=dtype)
    v = field.Field(name='v', dist=d, bases=(xb,yb,zb), tensorsig=(c,), dtype=dtype)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u, v])
    problem.add_equation((dt(u) + u, 0))
    problem.add_equation((dt(v) + v, 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0),
             v, v(x=0), v(y=0), v(z=0), v(x=0,y=0), v(x=0,z=0), v(y=0,z=0), v(x=0,y=0,z=0)]
    output = solver.evaluator.add_file_handler('test_output', iter=1, max_writes=1, virtual_file=True)
    for task in tasks:
        output.add_task(task, layout='g', name=str(task), scales=output_scales)
    solver.evaluator.evaluate_handlers([output])
    # Check solution
    errors = []
    d.comm.Barrier()
    with h5py.File('test_output/test_output_s1.h5', mode='r') as file:
        for task in tasks:
            task_name = str(task)
            task = task.evaluate()
            task.change_scales(output_scales)
            local_slices = (slice(None),) * len(task.tensorsig) + d.grid_layout.slices(task.domain, task.scales)
            task_saved = file['tasks'][task_name][-1]
            task_saved = task_saved[local_slices]
            local_error = task['g'] - task_saved
            if local_error.size:
                errors.append(np.max(np.abs(task['g'] - task_saved)))
    with Sync() as sync:
        if sync.comm.rank == 0:
            shutil.rmtree('test_output')
    assert np.allclose(errors, 0)

@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5, ])
@pytest.mark.parametrize('output_scales', [1/2, 1])
def test_cartesian_output_merged_virtual(dtype, dealias, output_scales):
    Nx = Ny = Nz = 16
    Lx = Ly = Lz = 2 * np.pi
    # Bases
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,), mesh=[2,2])
    Fourier = {np.float64: basis.RealFourier, np.complex128: basis.ComplexFourier}[dtype]
    xb = Fourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    yb = Fourier(c.coords[1], size=Ny, bounds=(0, Ly), dealias=dealias)
    zb = Fourier(c.coords[2], size=Nz, bounds=(0, Lz), dealias=dealias)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,yb,zb), dtype=dtype)
    v = field.Field(name='v', dist=d, bases=(xb,yb,zb), tensorsig=(c,), dtype=dtype)
    u['g'] = np.sin(x) * np.sin(y) * np.sin(z)
    # Problem
    dt = operators.TimeDerivative
    problem = problems.IVP([u, v])
    problem.add_equation((dt(u) + u, 0))
    problem.add_equation((dt(v) + v, 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
    # Output
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0,y=0), u(x=0,z=0), u(y=0,z=0), u(x=0,y=0,z=0),
             v, v(x=0), v(y=0), v(z=0), v(x=0,y=0), v(x=0,z=0), v(y=0,z=0), v(x=0,y=0,z=0)]
    output = solver.evaluator.add_file_handler('test_output', iter=1, max_writes=1, virtual_file=True)
    for task in tasks:
        output.add_task(task, layout='g', name=str(task), scales=output_scales)
    solver.evaluator.evaluate_handlers([output])
    # Check solution
    errors = []
    
    merge_virtual_analysis('test_output', cleanup=True)
    d.comm.Barrier()

    with h5py.File('test_output/test_output_s1.h5', mode='r') as file:
        for task in tasks:
            task_name = str(task)
            task = task.evaluate()
            task.change_scales(output_scales)
            local_slices = (slice(None),) * len(task.tensorsig) + d.grid_layout.slices(task.domain, task.scales)
            task_saved = file['tasks'][task_name][-1]
            task_saved = task_saved[local_slices]
            local_error = task['g'] - task_saved
            if local_error.size:
                errors.append(np.max(np.abs(task['g'] - task_saved)))
    with Sync() as sync:
        if sync.comm.rank == 0:
            shutil.rmtree('test_output')
    assert np.allclose(errors, 0)
