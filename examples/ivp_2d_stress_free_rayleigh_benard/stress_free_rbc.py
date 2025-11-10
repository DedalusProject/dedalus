"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard
convection with stress-free boundary conditions using Sine/Cosine
bases in z.  This script demonstrates solving a 2D Cartesian initial
value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved
data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

Note that unlike the no-slip Cheybshev example, here we solve for the
buoyancy *perturbation* instead of the total buoyancy field.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = 4, 1
Nx, Nz = 8, 8
Rayleigh = 2e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64
library = 'fftw' # 'scipy', 'matrix'
logger.info(f"Running with {library} DCT/DST library")
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zsbasis = d3.OddParity(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias, library=library)
zcbasis = d3.EvenParity(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias, library=library)

# Fields
p = dist.Field(name='p', bases=(xbasis,zcbasis))
b = dist.Field(name='b', bases=(xbasis,zsbasis))
u = dist.Field(name='u', bases=(xbasis,zcbasis))
w = dist.Field(name='w', bases=(xbasis,zsbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zcbasis)

dx = lambda A: d3.Differentiate(A,coords['x'])
dz = lambda A: d3.Differentiate(A,coords['z'])
lap = lambda A: dx(dx(A)) + dz(dz(A))

# Problem
problem = d3.IVP([p, b, u, w, tau_p], namespace=locals())
problem.add_equation("dx(u) + dz(w) + tau_p = 0")
problem.add_equation("dt(b) - kappa*lap(b) - w = - u*dx(b) - w*dz(b)")
problem.add_equation("dt(u) - nu*lap(u) + dx(p)         = - u*dx(u) - w*dz(u)")
problem.add_equation("dt(w) - nu*lap(w) + dz(p) - b  = - u*dx(w) - w*dz(w)")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(dz(u) - dx(w), name='vorticity')

# CFL
#CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
#             max_change=1.5, min_change=0.5, max_dt=max_timestep)
#CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u*u + w*w)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        #timestep = CFL.compute_timestep()
        timestep = 1e-3
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
