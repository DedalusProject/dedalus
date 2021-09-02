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
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: fix "one" conversion
# TODO: timestepper strings
# TODO: get unit vectors from coords?
# TODO: move rank printing to solver method?
# TODO: field method for reproducible random noise?


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
phi, theta, r = basis.local_grids((1, 1, 1))

# Fields
p = dist.Field(name='p', bases=basis)
b = dist.Field(name='b', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
tau1b = dist.Field(name='tau1b', bases=s2_basis)
tau2b = dist.Field(name='tau2b', bases=s2_basis)
tau1u = dist.VectorField(coords, name='tau1u', bases=s2_basis)
tau2u = dist.VectorField(coords, name='tau2u', bases=s2_basis)

# Substitutions
one = dist.Field(bases=basis.S2_basis(Ri))
one['g'] = 1

kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

er = dist.VectorField(coords, name='er', bases=basis.radial_basis)
er['g'][2] = 1

rvec = dist.VectorField(coords, name='er', bases=basis.radial_basis)
rvec['g'][2] = r

div =  d3.Divergence
lap = lambda A: d3.Laplacian(A, coords)
grad = lambda A: d3.Gradient(A, coords)
dot = d3.DotProduct
ddt = d3.TimeDerivative
ang = d3.AngularComponent
trace = d3.Trace
lift_basis = basis.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = grad(u) + rvec*lift(tau1u,-1) # First-order reduction
grad_b = grad(b) + rvec*lift(tau1b,-1) # First-order reduction

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in d3.split_equation(eq_str)]
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u])
problem.add_equation(eq_eval("trace(grad_u) = 0"))
problem.add_equation(eq_eval("ddt(b) - kappa*div(grad_b) + lift(tau2b,-1) = - dot(u,grad(b))"))
problem.add_equation(eq_eval("ddt(u) - nu*div(grad_u) + grad(p) - b*er + lift(tau2u,-1) = - dot(u,grad(u))"))
problem.add_equation(eq_eval("b(r=Ri) = one"))
problem.add_equation(eq_eval("u(r=Ri) = 0"))
problem.add_equation(eq_eval("b(r=Ro) = 0"))
problem.add_equation(eq_eval("u(r=Ro) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("p(r=Ro) = 0"), condition="ntheta == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# # Check matrix rank
# for i, subproblem in enumerate(solver.subproblems):
#     M = subproblem.M_min @ subproblem.pre_right
#     L = subproblem.L_min @ subproblem.pre_right
#     A = (M + L).A
#     print(f"MPI rank: {MPI.COMM_WORLD.rank}, subproblem: {i}, group: {subproblem.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

# Initial conditions
# # Random perturbations, initialized globally for same results in parallel
gshape = dist.grid_layout.global_shape(b.domain, scales=1)
slices = dist.grid_layout.slices(b.domain, scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
# Linear background + perturbations damped at walls
b['g'] = (Ri - Ri*Ro/r) / (Ri - Ro) + 1e-3 * noise * (Ro - r) * (r - Ri)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=5, max_writes=10)
snapshots.add_task(b(r=(Ri+Ro)/2), scales=(4,4,1), name='bmid')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=1, threshold=0.1,
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

