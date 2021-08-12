"""

This script simulates internally-heated convection. We solve the Boussinesq equations with
a constant internal heating source term in the temperature equation.

dt(u) + grad(p) - 1/R*laplacian(u) - r*T = -u.grad(u)
div(u) = 0
dt(T) - 1/P*laplacian(T) = -u.grad(T) + S/P

where R = sqrt(Rayleigh*Prandtl) and P = sqrt(Rayleigh/Prandtl), and r is the radius vector.
We use stress-free boundary conditions, and maintain the temperature on the outer boundary
equal to 0. The convection is driven by the internal heating term S/P. The conductive
equilibrium is T=S/6*(1-r^2).

The simulation will run to t=10, about the time for the first convective plumes to hit the
top boundary. After running this initial simulation, you can restart the simulation by
running with keyword '--restart', i.e.,

mpirun -np 4 python3 internally_heated_conv.py --restart

"""

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras import flow_tools
import time
from mpi4py import MPI
import sys

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

# TODO:
# NCC basis
# CFL

restart = False
if len(sys.argv) > 1 and sys.argv[1] == '--restart':
    restart = True

# Parameters
Lmax = 63
L_dealias = 3/2
Nmax = 47
N_dealias = 3/2
if not restart:
    t_end = 10.01
else:
    t_end = 20
ts = timesteppers.SBDF2
dtype = np.float64
mesh = None#[4,8]
Rayleigh = 1e6
Prandtl = 1
R = np.sqrt(Rayleigh*Prandtl)
P = np.sqrt(Rayleigh/Prandtl)

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=1, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))
x = r*np.sin(theta)*np.cos(phi)
y = r*np.sin(theta)*np.sin(phi)
z = r*np.cos(theta) + 0*phi

# Fields
u = field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(name='p', dist=d, bases=(b,), dtype=dtype)
T = field.Field(name='T', dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(name='tau u', dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(name='tau T', dist=d, bases=(b_S2,), dtype=dtype)
# TODO: basis for NCC
r_vec = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
r_vec['g'][2] = r
T_source = field.Field(dist=d, bases=(b,), dtype=dtype)
T_source['g'] = 6
#T_source = 6

# Operators
div = operators.Divergence
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = arithmetic.DotProduct
cross = arithmetic.CrossProduct
ddt = operators.TimeDerivative
LiftTau = lambda A: operators.LiftTau(A, b, -1)
rad = operators.RadialComponent
ang = operators.AngularComponent
trans = operators.TransposeComponents
grid = operators.Grid

# Boundary conditions
strain_rate = grad(u) + trans(grad(u))
shear_stress = ang(rad(strain_rate(r=1)))

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, T, tau_u, tau_T])
problem.add_equation(eq_eval("ddt(u) - 1/R*lap(u) + grad(p) + LiftTau(tau_u) - r_vec*T = - cross(curl(u),u)"))
problem.add_equation(eq_eval("div(u) = 0"))
problem.add_equation(eq_eval("ddt(T) - 1/P*lap(T) + LiftTau(tau_T) = - dot(u,grad(T)) + grid(T_source)/P"))
problem.add_equation(eq_eval("rad(u(r=1)) = 0"), condition="ntheta != 0")  # no penetration
problem.add_equation(eq_eval("p(r=1) = 0"), condition="ntheta == 0")  # pressure gauge
problem.add_equation(eq_eval("shear_stress = 0"))  # stress free
problem.add_equation(eq_eval("T(r=1) = 0"))
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Initial condition
if not restart:
    T['g'] = 1-r**2
    #T['g'] += 0.04*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3
    #T['g'] += 0.3*np.exp(-((x+0.3)**2+y**2+z**2)/0.2**2)
    T['g'] += 0.5*np.random.rand(*T['g'].shape)
    mode = 'overwrite'
else:
    write, dt = solver.load_state('checkpoints/checkpoints_s11.h5')
    mode = 'append'

# Analysis
slices = solver.evaluator.add_file_handler('slices', sim_dt = 0.1, max_writes = 10, virtual_file=True, mode=mode)
slices.add_task(T(theta=np.pi/2), name='T eq')
slices.add_task(T(phi=0), name='T mer right')
slices.add_task(T(phi=np.pi), name='T mer left')
slices.add_task(T(r=1/2), name='T r=0.5')

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt = 1, max_writes = 1, virtual_file=True, mode=mode)
checkpoints.add_tasks(solver.state)

# Report maximum |u|
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(dot(u,u)), name='u')

# TODO: CFL
dt = 0.5e-2

hermitian_cadence = 100

# Main loop
start_time = time.time()
while solver.ok:

    # Impose hermitian symmetry on two consecutive timesteps because we are using a 2-stage timestepper
    if solver.iteration % hermitian_cadence in [0, 1]:
        for f in solver.state:
            f.require_grid_space()

    solver.step(dt)

    if solver.iteration % 10 == 0:
        logger.info("t = %f, |u|_max = %e" %(solver.sim_time, flow.max('u')))

end_time = time.time()
logger.info('Run time: %f' %(end_time-start_time))
