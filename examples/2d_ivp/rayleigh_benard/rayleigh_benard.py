"""
Simulation script for 2D Rayleigh-Benard convection.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `process.py` script in this
folder can be used to merge distributed save files from parallel runs and plot
the snapshots from the command line.

To run, join, and plot using 4 processes, for instance, you could use:
$ mpiexec -n 4 python3 rayleigh_benard.py
$ mpiexec -n 4 python3 process.py join snapshots
$ mpiexec -n 4 python3 process.py plot snapshots/*.h5

On a single process, this should take ~15 minutes to run.

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
                           field_names=['p','b','u','w','bz','uz','wz'],
                           param_names=['R','P','F'])
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz))             = - u*dx(b) - w*bz")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = - u*dx(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_left_bc("b = -F*z")
problem.add_left_bc("u = 0")
problem.add_left_bc("w = 0")
problem.add_right_bc("b = -F*z")
problem.add_right_bc("u = 0")
problem.add_right_bc("w = 0", condition="(dx != 0)")
problem.add_int_bc("p = 0", condition="(dx == 0)")

# Parameters
Lx, Lz = (4., 1.)
Prandtl = 1.
Rayleigh = 1e6

# Create bases and domain
x_basis = de.Fourier(512, interval=(0, Lx), dealias=2/3)
z_basis = de.Chebyshev(129, interval=(-Lz/2, Lz/2), dealias=2/3)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Finalize problem
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.expand(domain)

# Build solver
ts = de.timesteppers.SBDF3
solver = de.solvers.IVP(problem, domain, ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * np.random.standard_normal(domain.local_grid_shape) * (zt - z) * (z - zb)
b['g'] = -F*(z - pert)
b.differentiate('z', out=bz)

# Integration parameters
dt = 1e-2
solver.stop_sim_time = 30
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

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
    new_dt = max(0.5*dt, min(safety*cfl_dt(), 1.1*dt))
    if domain.distributor.size > 1:
        dt_array[0] = new_dt
        domain.distributor.comm_cart.Allreduce(MPI.IN_PLACE, dt_array, op=MPI.MIN)
        new_dt = dt_array[0]
    return new_dt

# Analysis
snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_task("p")
snapshots.add_task("b")
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

