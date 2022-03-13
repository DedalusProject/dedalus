"""
Dedalus script simulating librational instability in a disk by solving the
incompressible Navier-Stokes equations linearized around a background librating
flow. This script demonstrates solving an initial value problem in the disk.
It can be ran serially or in parallel, and uses the built-in analysis framework
to save data snapshots to HDF5 files. The `plot_disk.py` and `plot_scalars.py`
scripts can be used to produce plots from the saved data. The simulation should
take roughly 10 cpu-minutes to run.

The problem is non-dimesionalized using the disk radius and librational frequency,
so the resulting viscosity is related to the Ekman number as:

    nu = Ekman

For incompressible hydro in the disk, we need one tau term for the velocity.
Here we lift to the original (k=0) basis.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 libration.py
    $ mpiexec -n 4 python3 plot_disk.py snapshots/*.h5
    $ python3 plot_scalars.py scalars/*.h5
"""

import numpy as np
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)


# Parameters
Nphi, Nr = 32, 128
Ekman = 1 / 2 / 20**2
Ro = 40
dealias = 3/2
stop_sim_time = 50
timestepper = d3.SBDF2
timestep = 1e-3
dtype = np.float64

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
S1_basis = basis.S1_basis(radius=1)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
tau_u = dist.VectorField(coords, name='tau_u', bases=S1_basis)
tau_p = dist.Field(name='tau_p')

# Substitutions
phi, r = dist.local_grids(basis)
nu = Ekman
lift = lambda A: d3.Lift(A, basis, -1)

# Background librating flow
u0_real = dist.VectorField(coords, bases=basis)
u0_imag = dist.VectorField(coords, bases=basis)
u0_real['g'][0] = Ro * np.real(jv(1, (1-1j)*r/np.sqrt(2*Ekman)) / jv(1, (1-1j)/np.sqrt(2*Ekman)))
u0_imag['g'][0] = Ro * np.imag(jv(1, (1-1j)*r/np.sqrt(2*Ekman)) / jv(1, (1-1j)/np.sqrt(2*Ekman)))
t = dist.Field()
u0 = np.cos(t) * u0_real - np.sin(t) * u0_imag

# Problem
problem = d3.IVP([p, u, tau_u, tau_p], time=t, namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u0) - u0@grad(u)")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='standard_normal') # Random noise
u.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=20)
snapshots.add_task(-d3.div(d3.skew(u)), scales=(4, 1), name='vorticity')
scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
scalars.add_task(d3.integ(0.5*u@u), name='KE')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(u@u, name='u2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, timestep, max_u))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# Post-processing
if dist.comm.rank == 0:
    scalars.process_virtual_file()

