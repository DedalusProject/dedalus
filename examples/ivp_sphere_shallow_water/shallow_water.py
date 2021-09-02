import time
from scipy.special import sph_harm
import numpy as np
import dedalus.public as d3
from dedalus.core import timesteppers
from dedalus.core.basis import S2Skew
import logging
logger = logging.getLogger(__name__)

dtype = np.complex128

# Parameters
nphi = 128
ntheta = 64

V = 1
N = 2

Ampl = 1e-3
l_ic = 3
m_ic = 3
L_dealias = 1

# Bases
c = d3.S2Coordinates('phi', 'theta')
d = d3.Distributor((c,))
b = d3.SphereBasis(c, (nphi, ntheta), radius=1, dtype=dtype)
phi, theta = b.local_grids(b.domain.dealias)

# Fields
f = d3.Field(dist=d, bases=(b,), dtype=dtype)
v = d3.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
h = d3.Field(dist=d, bases=(b,), dtype=dtype)

# Parameters and operators
lap = lambda A: d3.Laplacian(A, c)
ddt = lambda A: d3.TimeDerivative(A)
grad = lambda A: d3.Gradient(A, c)
div = lambda A: d3.Divergence(A)
skew = lambda A: S2Skew(A)
H = 0.1
Omega = 1e-1

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

f['g'] = 2*Omega*np.cos(theta)
problem = d3.IVP([v,h])
problem.add_equation((ddt(v) + grad(h), -f*skew(v)))
problem.add_equation((ddt(h) + H*div(v), 0))
print("Problem built")

# Solver
solver = problem.build_solver(timesteppers.RK222)

# initial conditions
x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)

x0 = 0
y0 = 0
z0 = 1
eps2 = 0.1
ampl = 1

r2 = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
h['g'] = ampl * np.exp(-r2/eps2)

snapshots = solver.evaluator.add_file_handler('snapshots', iter=5, max_writes=10, virtual_file=True)
snapshots.add_task(h, name='height')

# Main loop
solver.stop_sim_time = 15
#solver.stop_wall_time = 60
dt = 5e-2
start_time = time.time()
while solver.ok:
    solver.step(dt)

    if solver.iteration % 10 == 0:
        logger.info("t = {:3.2f}".format(solver.sim_time))


end_time = time.time()
print('Run time:', end_time-start_time)

print("SUCCESS!")
