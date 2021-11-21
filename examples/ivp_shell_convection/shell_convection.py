"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_sphere.py` script can be used to produce plots from the
saved data. The simulation should take roughly 1 cpu-hour to run.

The problem is non-dimensionalized using the shell thickness and freefall time, so
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
    $ mpiexec -n 4 python3 shell_convection.py
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: fix "one" conversion
# TODO: get unit vectors from coords?


# Parameters
Ri, Ro = 14, 15
Nphi, Ntheta, Nr = 192, 96, 8
Rayleigh = 3000
Prandtl = 1
dealias = 3/2
stop_sim_time = 2000
timestepper = d3.RK222
max_timestep = 5
dtype = np.float64
mesh = None

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
phi, theta, r = dist.local_grids(basis)

# Fields
p = dist.Field(name='p', bases=basis)
b = dist.Field(name='b', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=s2_basis)
tau_b2 = dist.Field(name='tau_b2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)

# Substitutions
one = dist.Field(bases=basis.S2_basis(Ri))
one['g'] = 1

kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

er = dist.VectorField(coords, name='er', bases=basis.radial_basis)
er['g'][2] = 1

rvec = dist.VectorField(coords, name='er', bases=basis.radial_basis)
rvec['g'][2] = r

lift_basis = basis.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)

grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_b = d3.grad(b) + rvec*lift(tau_b1,-1) # First-order reduction

integ = lambda A: d3.Integrate(A, coords)

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2,-1) = - dot(u,grad(b))")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*er + lift(tau_u2,-1) = - dot(u,grad(u))")
problem.add_equation("b(r=Ri) = one")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("b(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
b['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # Add linear background

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=5, max_writes=10)
snapshots.add_task(b(r=(Ri+Ro)/2), scales=(4,4,1), name='bmid')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=1, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

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
