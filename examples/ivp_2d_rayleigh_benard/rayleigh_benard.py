"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

The simulation should take roughly 8 process-minutes to run.

"""

import numpy as np
import time
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.parsing import split_equation
from dedalus.tools import logging
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# TODO: add CFL
# TODO: maybe fix plotting to directly handle vector
# TODO: optimize and match d2 resolution

# Parameters
Lx, Lz = (4., 1.)
Nx, Nz = 64, 32
Prandtl = 1
Rayleigh = 1e6
dtype = np.float64
stop_sim_time = 30
dealias = 3/2
dt = 1e-2
timestepper = "RK222"

# Bases
c = coords.CartesianCoordinates('x', 'z')
d = distributor.Distributor((c,))
xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
p = field.Field(name='p', dist=d, bases=(xb,zb), dtype=dtype)
b = field.Field(name='b', dist=d, bases=(xb,zb), dtype=dtype)
u = field.Field(name='u', dist=d, bases=(xb,zb), dtype=dtype, tensorsig=(c,))

# Taus
zb1 = basis.ChebyshevU(c.coords[1], size=Nz, bounds=(0, Lz), alpha0=0)
tb1 = field.Field(name='t1', dist=d, bases=(xb,), dtype=dtype)
tb2 = field.Field(name='t2', dist=d, bases=(xb,), dtype=dtype)
tu1 = field.Field(name='t3', dist=d, bases=(xb,), dtype=dtype, tensorsig=(c,))
tu2 = field.Field(name='t4', dist=d, bases=(xb,), dtype=dtype, tensorsig=(c,))
P1 = field.Field(name='P1', dist=d, bases=(zb1,), dtype=dtype)
if rank == 0:
    P1['c'][0,-1] = 1

# Parameters and operators
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)

ex = field.Field(name='ex', dist=d, bases=(zb,), dtype=dtype, tensorsig=(c,))
ez = field.Field(name='ez', dist=d, bases=(zb,), dtype=dtype, tensorsig=(c,))
ex['g'][0] = 1
ez['g'][1] = 1

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)

dx = lambda A: operators.Differentiate(A, c.coords[0])
dz = lambda A: operators.Differentiate(A, c.coords[1])
P2 = dz(P1).evaluate()

Coeff = operators.Coeff
Conv = operators.Convert
integ = operators.Integrate

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, b, u, tb1, tb2, tu1, tu2])
problem.add_equation(eq_eval("div(u) - P1*dot(ez,tu2) = 0"))
problem.add_equation(eq_eval("ddt(b) - P*lap(b) + P1*tb1 + P*P2*tb2 = - Coeff(Conv(dot(u,grad(b)),zb))"))
problem.add_equation(eq_eval("ddt(u) - R*lap(u) + grad(p) + P1*tu1 + R*P2*tu2 - b*ez = - Coeff(Conv(dot(u,grad(u)),zb))"))
problem.add_equation(eq_eval("b(z=0) = Lz"))
problem.add_equation(eq_eval("u(z=0) = 0"))
problem.add_equation(eq_eval("b(z=Lz) = 0"))
problem.add_equation(eq_eval("u(z=Lz) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("dot(ex,u)(z=Lz) = 0"), condition="nx == 0")
problem.add_equation(eq_eval("p(z=Lz) = 0"), condition="nx == 0")

# Build solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
solver.stop_sim_time = stop_sim_time

# Random perturbations, initialized globally for same results in parallel
gshape = d.grid_layout.global_shape(b.domain, scales=1)
slices = d.grid_layout.slices(b.domain, scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = zb.bounds
b['g'] = Lz - z +  1e-3 * noise * (zt - z) * (z - zb)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_task(p)
snapshots.add_task(b)
snapshots.add_task(dot(u,ex), name='ux')
snapshots.add_task(dot(u,ez), name='uz')

# # CFL
# CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=10, safety=1,
#                      max_change=1.5, min_change=0.5, max_dt=0.1)
# CFL.add_velocities(u)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(dot(u,u))/R, name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*d.comm.size))
