"""
Dedalus script simulating 2D centrifugal convection in an annulus. This script
demonstrates solving an initial value problem in the annulus. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots to
HDF5 files. The `plot_polar.py` and `plot_scalars.py` scripts can be used to produce
plots from the saved data. The simulation should take roughly 10 cpu-minutes to run.

The problem is non-dimesionalized using the mean radius L = (Ri + Ro)/2 and the
freefall time, so the resulting thermal diffusivity and viscosity are related to the
Prandtl and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

The radii ratio is given by eta = Ro/Ri. Since the problem is 2D, the Coriolis force
and Rossby number drop out of the problem.

For incompressible hydro in the annulus, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 centrifugal_convection.py
    $ mpiexec -n 4 python3 plot_polar.py snapshots/*.h5
    $ python3 plot_scalars.py scalars/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Nphi, Nr = 256, 64
eta = 3
Rayleigh = 1e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 30
timestepper = d3.RK222
max_timestep = 0.125
safety = 0.5
dtype = np.float64

# Derived parameters
Ri = 2 / (1 + eta)
Ro = 2 * eta / (1 + eta)

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
annulus = d3.AnnulusBasis(coords, shape=(Nphi, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
edge = annulus.outer_edge

# Fields
p = dist.Field(name='p', bases=annulus)
b = dist.Field(name='b', bases=annulus)
u = dist.VectorField(coords, name='u', bases=annulus)
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=edge)
tau_b2 = dist.Field(name='tau_b2', bases=edge)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=edge)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=edge)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
phi, r = dist.local_grids(annulus)
rvec = dist.VectorField(coords, bases=annulus.radial_basis)
rvec['g'][1] = r
lift_basis = annulus.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction
g = rvec * 2 * (eta - 1) / (eta + 1)

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + b*g + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(r=Ri) = 0")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("b(r=Ro) = 1")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
b['g'] += np.log(r/Ri) / np.log(Ro/Ri) # Add conductive background

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=20)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(b, name='buoyancy')
scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
scalars.add_task(d3.integ(0.5*u@u), name='KE')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, max_dt=max_timestep, safety=safety,
             cadence=10, threshold=0.1, max_change=1.5, min_change=0.5)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
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
    solver.log_stats()
