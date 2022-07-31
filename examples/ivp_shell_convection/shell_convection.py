"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.py` script can be used to produce plots from the
saved data. The simulation should take about 20 cpu-minutes to run.

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
    $ mpiexec -n 4 python3 plot_shell.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Ri, Ro = 14, 15
Nphi, Ntheta, Nr = 192, 96, 6
Rayleigh = 3500
Prandtl = 1
dealias = 3/2
stop_sim_time = 2000
timestepper = d3.SBDF2
max_timestep = 1
dtype = np.float64
mesh = None

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
sphere = shell.outer_surface

# Fields
p = dist.Field(name='p', bases=shell)
b = dist.Field(name='b', bases=shell)
u = dist.VectorField(coords, name='u', bases=shell)
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=sphere)
tau_b2 = dist.Field(name='tau_b2', bases=sphere)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
phi, theta, r = dist.local_grids(shell)
er = dist.VectorField(coords, bases=shell.radial_basis)
er['g'][2] = 1
rvec = dist.VectorField(coords, bases=shell.radial_basis)
rvec['g'][2] = r
lift_basis = shell.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*er + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(r=Ri) = 1")
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
flux = er @ (-kappa*d3.grad(b) + u*b)
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10, max_writes=10)
snapshots.add_task(b(r=(Ri+Ro)/2), scales=dealias, name='bmid')
snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')
snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')
snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')
snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
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
