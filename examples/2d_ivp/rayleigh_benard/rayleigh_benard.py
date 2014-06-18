"""
Simulation script for 2D Rayleigh-Benard convection.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `join.py` and `plot.py`
scripts in this folder can be used to merge scripts from parallel runs and plot
the snapshots, respectively.

On a single processor, this should take a couple minutes to run.

"""

import os
import numpy as np
from mpi4py import MPI
import time

from dedalus2 import public as de

import logging
logger = logging.getLogger(__name__)


# 2D Boussinesq hydrodynamics
problem = de.ParsedProblem(axis_names=['x','z'],
                           field_names=['p','r','u','w','rz','uz','wz'],
                           param_names=['R','P','F'])
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(r) - P*(dx(dx(r)) + dz(rz))               = - u*dx(r) - w*rz")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)       = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) + F*r = - u*dx(w) - w*wz")
problem.add_equation("rz - dz(r) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_left_bc("r = -1")
problem.add_left_bc("u = 0")
problem.add_left_bc("w = 0")
problem.add_right_bc("r = 0")
problem.add_right_bc("u = 0")
problem.add_right_bc("w = 0", condition="(dx != 0)")
problem.add_int_bc("p = 0", condition="(dx == 0)")

# Parameters
Lx, Lz = (3., 1.)
Prandtl = 1.
Rayleigh = 1e6

# Create bases and domain
x_basis = de.Fourier(1024, interval=(0, Lx), dealias=2/3)
z_basis = de.Chebyshev(64, interval=(0, Lz), dealias=2/3)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Finalize problem
problem.parameters['P'] = 1.
problem.parameters['R'] = Prandtl
problem.parameters['F'] = Prandtl * Rayleigh
problem.expand(domain)

# Build solver
ts = de.timesteppers.SBDF3
solver = de.solvers.IVP(problem, domain, ts)
logger.info('Solver built.')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
r = solver.state['r']
rz = solver.state['rz']

δ =  1e-3 * np.random.standard_normal(domain.local_grid_shape) * np.sin(np.pi*z/Lz)
r['g'] = z/Lz - 1 - δ
r.differentiate('z', out=rz)

# Integration parameters
dt = 1e-5
solver.stop_sim_time = np.inf
solver.stop_wall_time = 5 * 60.
solver.stop_iteration = 2000.

# CFL routines
evaluator = solver.evaluator
evaluator.vars['grid_delta_x'] = domain.grid_spacing(0)
evaluator.vars['grid_delta_z'] = domain.grid_spacing(1)

cfl_cadence = 10
cfl_variables = evaluator.add_dictionary_handler(iter=cfl_cadence)
cfl_variables.add_task('u/grid_delta_x', name='f_u')
cfl_variables.add_task('w/grid_delta_z', name='f_w')

def cfl_dt():
    if z.size > 0:
        max_f_u = np.max(np.abs(cfl_variables.fields['f_u']['g']))
        max_f_w = np.max(np.abs(cfl_variables.fields['f_w']['g']))
    else:
        max_f_u = max_f_w = 0
    max_f = max(max_f_u, max_f_w)
    if max_f > 0:
        min_t = 1 / max_f
    else:
        min_t = np.inf
    return min_t

safety = 0.3
dt_array = np.zeros(1, dtype=np.float64)
def update_dt(dt):
    new_dt = max(0.5*dt, min(safety*cfl_dt(), 1.05*dt))
    if domain.distributor.size > 1:
        dt_array[0] = new_dt
        domain.distributor.comm_cart.Allreduce(MPI.IN_PLACE, dt_array, op=MPI.MIN)
        new_dt = dt_array[0]
    return new_dt

# Analysis
snapshots = evaluator.add_file_handler('snapshots', sim_dt=5e-4, max_writes=20)
snapshots.add_task("p")
snapshots.add_task("r")
snapshots.add_task("u")
snapshots.add_task("w")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % cfl_cadence == 0:
            dt = update_dt(dt)
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

