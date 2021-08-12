"""

2D periodic shear flow problem. We are solving the incompressible hydrodynamics equation
plus and equation for a passive tracer field s, used for visualization purposes. The
equations are

dt(u) + grad(p) - nu*laplacian(u) = -u.grad(u)
div(u) = 0
dt(s) - D*laplacian(s) = -u.grad(s)

The initial flow is in the x direction and depends only on z. There are two shear layers at
\pm 0.5. The velocity changes by 1 across the shear layer. The Reynolds number in this
simulation is 1/nu.

Simulation should run in less than 5 minutes on 4 cores.

"""

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

#from dedalus.tools.config import config
#config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

# Parameters
Lx, Lz = (1, 2)
Nx, Nz = 96, 192
#Nx, Nz = 32, 64
nu = 1e-4
D = 1e-4

dtype = np.float64

# Bases
c = coords.CartesianCoordinates('x', 'z')
d = distributor.Distributor((c,))
xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=3/2)
zb = basis.RealFourier(c.coords[1], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
p = field.Field(name='p', dist=d, bases=(xb,zb), dtype=dtype)
s = field.Field(name='s', dist=d, bases=(xb,zb), dtype=dtype)
u = field.Field(name='u', dist=d, bases=(xb,zb), dtype=dtype, tensorsig=(c,))

ez = field.Field(name='u', dist=d, bases=(xb,zb), dtype=dtype, tensorsig=(c,))
ez['g'][1] = 1.

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, s])
problem.add_equation(eq_eval("dt(u) + grad(p) - nu*lap(u) = -dot(u, grad(u))"))
problem.add_equation(eq_eval("div(u) = 0"), condition = '(nx != 0) or (nz != 0)')
# pressure gauge on averaged mode
problem.add_equation(eq_eval("p = 0"), condition = '(nx == 0) and (nz == 0)')
problem.add_equation(eq_eval("dt(s) - D*lap(s) = - dot(u, grad(s))"))
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
solver.stop_sim_time = 5.01

# Initial conditions
# component 0 is the x component
u['g'][0] = 1/2 + 1/2 * (np.tanh( (z-0.5)/0.1 ) - np.tanh( (z+0.5)/0.1 ))
# also initialize the scalar field s in the same way
s['g'] = 1/2 + 1/2 * (np.tanh( (z-0.5)/0.1 ) - np.tanh( (z+0.5)/0.1 ))
# add a small vertical perturbation to u localized to the shear layers
u['g'][1] = 0.1 * ( np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01) - np.sin(2*np.pi*x/Lx + np.pi) * np.exp(-(z+0.5)**2/0.01) )

# Output
# Save s every 0.05 time units
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt = 0.05, max_writes=10, virtual_file=True)
snapshots.add_task(s)

# Report maximum |w|
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property(dot(u,ez)**2, name='w2')

# TODO: CFL
dt = 5e-3

# Main loop
while solver.ok:
    solver.step(dt)

    if solver.iteration % 10 == 0:
        logger.info('Iteration: %i, t: %f, max w = %f' %(solver.iteration, solver.sim_time, np.sqrt(flow.max('w2'))))

