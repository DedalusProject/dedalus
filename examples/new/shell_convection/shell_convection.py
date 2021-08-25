"""
Dedalus script for Boussinesq convection in a spherical shell.
This script should take approximately 1 cpu-hr to run.

"""

import numpy as np
import time
from mpi4py import MPI
from numpy.lib.stride_tricks import broadcast_to

rank = MPI.COMM_WORLD.rank

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.parsing import split_equation
from dedalus.tools import logging
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# TODO: add CFL
# TODO: fix "one" conversion
# TODO: check loading radial slices from virtual file

# Parameters
Nphi, Ntheta, Nr = 192, 96, 8
Ri = 14
Ro = 15
Prandtl = 1
Rayleigh = 3000
dtype = np.float64
stop_sim_time = 2000
dealias = 3/2
timestepper = "RK222"
dt = 1
mesh = (2, 4)

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
bc = basis.ShellBasis(c, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
phi, theta, r = bc.local_grids((1, 1, 1))

# Fields
p = field.Field(name='p', dist=d, bases=(bc,), dtype=dtype)
b = field.Field(name='b', dist=d, bases=(bc,), dtype=dtype)
u = field.Field(name='u', dist=d, bases=(bc,), dtype=dtype, tensorsig=(c,))

# Taus
bs = bc.S2_basis()
tb1 = field.Field(name='t1', dist=d, bases=(bs,), dtype=dtype)
tb2 = field.Field(name='t2', dist=d, bases=(bs,), dtype=dtype)
tu1 = field.Field(name='t3', dist=d, bases=(bs,), dtype=dtype, tensorsig=(c,))
tu2 = field.Field(name='t4', dist=d, bases=(bs,), dtype=dtype, tensorsig=(c,))

one = field.Field(name='t2', dist=d, bases=(bc.S2_basis(Ri),), dtype=dtype)
one['g'] = 1

# Parameters and operators
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)

er = field.Field(name='er', dist=d, bases=(bc.radial_basis,), dtype=dtype, tensorsig=(c,))
er['g'][2] = 1

rvec = field.Field(name='er', dist=d, bases=(bc.radial_basis,), dtype=dtype, tensorsig=(c,))
rvec['g'][2] = r

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
LT = lambda A, n: operators.LiftTau(A, bc, n)
ang = operators.AngularComponent
trace = operators.Trace

# Spherical first-order reduction
grad_u = grad(u) + rvec*LT(tu1,-1)
# trace(grad_u) = div(u) + rvec@LT(tu1,-1)
# div(grad_u) = lap(u) + div(rvec*LT(tu1,-1))
grad_b = grad(b) + rvec*LT(tb1,-1)
# div(grad_b) = lap(b) + div(LT(tb1,-1))

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, b, u, tb1, tb2, tu1, tu2])
problem.add_equation(eq_eval("trace(grad_u) = 0"))
problem.add_equation(eq_eval("ddt(b) - P*div(grad_b) + LT(tb2,-1) = - dot(u,grad(b))"))
problem.add_equation(eq_eval("ddt(u) - R*div(grad_u) + grad(p) - b*er + LT(tu2,-1) = - dot(u,grad(u))"))
problem.add_equation(eq_eval("b(r=Ri) = one"))
problem.add_equation(eq_eval("u(r=Ri) = 0"))
problem.add_equation(eq_eval("b(r=Ro) = 0"))
problem.add_equation(eq_eval("u(r=Ro) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("p(r=Ro) = 0"), condition="ntheta == 0")

# Build solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
solver.stop_sim_time = stop_sim_time

# Check matrix rank
for i, subproblem in enumerate(solver.subproblems):
    M = subproblem.M_min @ subproblem.pre_right
    L = subproblem.L_min @ subproblem.pre_right
    A = (M + L).A
    print(f"MPI rank: {MPI.COMM_WORLD.rank}, subproblem: {i}, group: {subproblem.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

# Random perturbations, initialized globally for same results in parallel
gshape = d.grid_layout.global_shape(bc.domain, scales=1)
slices = d.grid_layout.slices(bc.domain, scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
b['g'] = (Ri - Ri*Ro/r) / (Ri - Ro) + 1e-3 * noise * (Ro - r) * (r - Ri)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=5, max_writes=10, virtual_file=True)
#snapshots.add_task(b(r=(Ri+Ro)/2), scales=(4,4,1), name='bmid')
snapshots.add_task(b, scales=(2,2,1))

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

