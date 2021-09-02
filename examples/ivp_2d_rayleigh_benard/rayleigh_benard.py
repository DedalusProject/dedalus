"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. The simulation should take roughly 10
cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: maybe fix plotting to directly handle vector
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?
# TODO: timestepper strings
# TODO: field method for reproducible random noise?


# Parameters
Lx, Lz = 4, 1
Nx, Nz = 64, 32
Rayleigh = 1e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 30
timestepper = d3.RK222
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords.coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau1b = dist.Field(name='tau1b', bases=xbasis)
tau2b = dist.Field(name='tau2b', bases=xbasis)
tau1u = dist.VectorField(coords, name='tau1u', bases=xbasis)
tau2u = dist.VectorField(coords, name='tau2u', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

div = d3.Divergence
lap = lambda A: d3.Laplacian(A, coords)
grad = lambda A: d3.Gradient(A, coords)
dot = d3.DotProduct
ddt = d3.TimeDerivative
trace = d3.Trace
lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = grad(u) + ez*lift(tau1u,-1) # First-order reduction
grad_b = grad(b) + ez*lift(tau1b,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
def eq_eval(eq_str):
    return [eval(expr) for expr in d3.split_equation(eq_str)]
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u])
problem.add_equation(eq_eval("trace(grad_u) = 0"))
problem.add_equation(eq_eval("ddt(b) - kappa*div(grad_b) + lift(tau2b,-1) = - dot(u,grad(b))"))
problem.add_equation(eq_eval("ddt(u) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) - b*ez = - dot(u,grad(u))"))
problem.add_equation(eq_eval("b(z=0) = Lz"))
problem.add_equation(eq_eval("u(z=0) = 0"))
problem.add_equation(eq_eval("b(z=Lz) = 0"))
problem.add_equation(eq_eval("u(z=Lz) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("dot(ex,u)(z=Lz) = 0"), condition="nx == 0")
problem.add_equation(eq_eval("p(z=Lz) = 0"), condition="nx == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Random perturbations, initialized globally for same results in parallel
gshape = dist.grid_layout.global_shape(b.domain, scales=1)
slices = dist.grid_layout.slices(b.domain, scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
# Linear background + perturbations damped at walls
zb, zt = zbasis.bounds
b['g'] = Lz - z +  1e-3 * noise * (zt - z) * (z - zb)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_task(p)
snapshots.add_task(b)
snapshots.add_task(dot(u,ex), name='ux')
snapshots.add_task(dot(u,ez), name='uz')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(dot(u,u))/nu, name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*dist.comm.size))

