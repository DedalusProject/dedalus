
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers, timesteppers
from dedalus.tools.parsing import split_equation
from dedalus.tools import logging
from dedalus.extras import flow_tools
from scipy.special import jv
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

Nphi = 32
Nr = 128
# might want to include this in the future
dealias = 3/2

Ekman = 1/2/20**2
Ro = 40.
radius = 1
dtype = np.float64
dt = 1.5e-3

c = coords.PolarCoordinates('phi','r')
d = distributor.Distributor((c,))
db = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=(dealias, dealias), dtype=dtype, azimuth_library='matrix')
phi, r = db.local_grids()
cb = db.S1_basis(radius=radius)

u = field.Field(name='u', dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(db,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(cb,), tensorsig=(c,), dtype=dtype)
tau_w = field.Field(dist=d, bases=(cb,), dtype=dtype)

# Parameters and operators
lap = lambda A: operators.Laplacian(A, c)
div = lambda A: operators.Divergence(A)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = lambda A,B: arithmetic.DotProduct(A, B)
cross = lambda A,B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
LiftTau = lambda A: operators.LiftTau(A, db, -1)
azimuthal = lambda A: operators.AzimuthalComponent(A)
integ = lambda A: operators.Integrate(A, c)
grid = operators.Grid

# Define librating flow
u0R = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
u0I = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
t = field.Field(dist=d, dtype=dtype)
u0R['g'][0] = Ro* np.real(jv(1, (1-1j)*r/np.sqrt(2*Ekman))/jv(1, (1-1j)/np.sqrt(2*Ekman)))
u0I['g'][0] = Ro* np.imag(jv(1, (1-1j)*r/np.sqrt(2*Ekman))/jv(1, (1-1j)/np.sqrt(2*Ekman)))

u0 = np.cos(t)*u0R - np.sin(t)*u0I

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, tau_u], time=t)
problem.add_equation(eq_eval("ddt(u) - Ekman*lap(u) + grad(p) + LiftTau(tau_u) = - dot(u, grad(u0)) - dot(u0, grad(u))"))
problem.add_equation(eq_eval("div(u) = 0"))

problem.add_equation(eq_eval("u(r=1) = 0"), condition='nphi != 0')
problem.add_equation(eq_eval("azimuthal(u(r=1)) = 0"), condition='nphi == 0')
problem.add_equation(eq_eval("p(r=1) = 0"), condition='nphi == 0')

logger.info('building solver')
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2)

# Noise Initial Conditions
seed = 42# + d.comm_cart.rank
rand = np.random.RandomState(seed=seed)

u['g'] = rand.standard_normal(u['g'].shape)
u['c']
u.require_scales(0.25)
u['g']
u['c']
u.require_scales(dealias)

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=20, virtual_file=True)
snapshots.add_task(u, name='u')

traces = solver.evaluator.add_file_handler('traces', sim_dt=0.01, virtual_file=True)
traces.add_task(integ(0.5*dot(u,u)), name='KE')

solver.stop_sim_time = 0.91

# Report maximum |u|
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property(np.sqrt(dot(u,u)), name='u')

# Main loop
start_time = time.time()
while solver.ok:

    solver.step(dt)

    # Impose Hermitian symmetry every 100 timesteps
    if solver.iteration % 100 in [0, 1]:
        for field in solver.state: field.require_grid_space()

    if solver.iteration % 100 == 0:
        logger.info("t = %f, max(u) = %f" %(solver.sim_time, flow.max('u')) )
end_time = time.time()
logger.info('Run time: %f' %(end_time-start_time))

# Make virtual file for traces
if MPI.COMM_WORLD.rank == 0:
    traces.process_virtual_file()

