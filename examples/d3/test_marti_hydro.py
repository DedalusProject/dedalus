import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)

dtype = np.float64
# Parameters
radius = 1
Lmax = 30
Nmax = 31
# load balancing for real variables and parallel runs
if Lmax % 2 == 1:
    nm = 2*(Lmax+1)
else:
    nm = 2*(Lmax+2)
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2

# Integration parameters
dt = 1e-2
t_end = 20

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]

L_dealias = N_dealias = 3/2
# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (nm, Lmax+1, Nmax+1), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=dtype)
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
τu = field.Field(dist=d, bases=(b.S2_basis(),), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
ν = field.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)

# create boundary conditions
u_BC = field.Field(dist=d, bases=(b.S2_basis(),), tensorsig=(c,), dtype=dtype)
u_BC['g'][2] = 0. # u_r = 0
u_BC['g'][1] = - u0*np.cos(theta)*np.cos(phi)
u_BC['g'][0] = u0*np.sin(phi)

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = operators.Grid(ez).evaluate()

ν['g'] = nu

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
power = lambda A, B: operators.Power(A, B)
LiftTau = lambda A, n: operators.LiftTau(A,b,n)
ddt = lambda A: operators.TimeDerivative(A)

# Problem
problem = problems.IVP([p, u, τu])
problem.add_equation((div(u), 0), condition = "ntheta != 0")
problem.add_equation((ddt(u) - ν*lap(u) + grad(p) + LiftTau(τu,-1), - dot(u,grad(u)) - Om*cross(ez, u)), condition = "ntheta != 0")
problem.add_equation((p, 0), condition = "ntheta == 0")
problem.add_equation((u, 0), condition = "ntheta == 0")
problem.add_equation((u(r=1), u_BC), condition = "ntheta != 0")
problem.add_equation((τu, 0), condition = "ntheta == 0")
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF4)


reducer = GlobalArrayReducer(d.comm_cart)
p.require_scales(3/2)
weight_theta = b.local_colatitude_weights(3/2)
weight_r = b.local_radial_weights(3/2)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(radius)
vol_correction = vol/vol_test

E0_marti = 0.06183074756
logger.info("target value: E0 = {}".format(E0_marti))
energy_report_cadence = 100
good_solution = True
while solver.sim_time < t_end and good_solution:
    if solver.iteration % energy_report_cadence == 0:
        #q = (ρ*power(u,2)).evaluate() # can't eval in parallel with ρ
        q = (power(u,2)).evaluate()
        E0 = np.sum(vol_correction*weight_r*weight_theta*0.5*q['g'])
        E0 *= (np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("iter: {:d}, dt={:.2e}, t={:.3e}, E0={:e}".format(solver.iteration, dt, solver.sim_time, E0))
        good_solution = np.isfinite(E0)
    solver.step(dt)
logger.info("target value:    E0_m = {}".format(E0_marti))
logger.info("equilibrated value E0 = {}".format(E0))
logger.info("difference:   E0-E0_m = {}".format(E0-E0_marti))
