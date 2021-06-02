
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers, timesteppers
from dedalus.tools.parsing import split_equation
from dedalus.tools import logging
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy.special import jv
import time
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

Nphi = 64
Nr = 32
# might want to include this in the future
dealias = 1

Ekman = 1e-2
Ro = 0.0
f = 1
radius = 1
kz = 0
dtype = np.complex128
dt = 1e-2

c3D = coords.CylindricalCoordinates('phi', 'r', 'z')
c = c3D.polar_coordsys
d = distributor.Distributor((c3D,))
db = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=(dealias, dealias), dtype=dtype)
phi, r = db.local_grids()
cb = db.S1_basis(radius=radius)

u3D = field.Field(dist=d, bases=(db,), tensorsig=(c3D,), dtype=dtype)

#u = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
#w = field.Field(dist=d, bases=(db,), dtype=dtype)
p = field.Field(dist=d, bases=(db,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(cb,), tensorsig=(c,), dtype=dtype)
tau_w = field.Field(dist=d, bases=(cb,), dtype=dtype)

ez = field.Field(dist=d, bases=(db.radial_basis,), tensorsig=(c3D,), dtype=dtype)
ez['g'][2] = 1.

# Parameters and operators
lap = lambda A: operators.Laplacian(A, c)
div = lambda A: operators.Divergence(A)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = lambda A,B: arithmetic.DotProduct(A, B)
cross = lambda A,B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
LiftTau = lambda A: operators.LiftTau(A, db, -1)
dz = lambda A: 1j*kz*A
vertical = lambda A: operators.VerticalComponent(A)
horizontal = lambda A: operators.HorizontalComponent(A)
azimuthal = lambda A: operators.AzimuthalComponent(A)

u = horizontal(u3D)
w = vertical(u3D)

u0 = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
u0_func = lambda t: Ro* jv(0, (1-1j)*r)/jv(0, (1-1j)) * np.exp(1j*t)
u0['g'][0] = u0_func(0)

seed = 42 + d.comm_cart.rank
rand = np.random.RandomState(seed=seed)

u3D['g'] = rand.standard_normal(u3D['g'].shape)
u3D['c']
u3D.require_scales(0.25)
u3D['g']
u3D['c']
u3D.require_scales(dealias)

fig, ax = plt.subplots(1, 1)

x = r*np.cos(phi)
y = r*np.sin(phi)
ax.pcolormesh(x, y, u3D['g'][0,:,:].real)
ax.set_aspect(1)
plt.savefig('uphi.png', dpi=150)

cor = horizontal(cross(ez, u3D)).evaluate()

error_phi = np.max(np.abs(cor['g'][0] - u3D['g'][1]))
error_r = np.max(np.abs(cor['g'][1] + u3D['g'][0]))
logger.info('size of ur: %e' %np.max(np.abs(u3D['g'][1])))
logger.info('error in phi component of coriolis force: %e' %error_phi)
logger.info('error in r component of coriolis force: %e' %error_r)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u3D, p, tau_u, tau_w])
#problem.add_equation(eq_eval("ddt(u) - Ekman*(lap(u) + dz(dz(u))) + grad(p) + LiftTau(tau_u) = - f*horizontal(cross(ez, u3D)) - dot(u, grad(u0)) - dot(u0, grad(u))"))
problem.add_equation(eq_eval("ddt(u) - Ekman*(lap(u) + dz(dz(u))) + grad(p) + LiftTau(tau_u) + f*horizontal(cross(ez, u3D)) = - dot(u, grad(u0)) - dot(u0, grad(u))"))
#problem.add_equation(eq_eval("ddt(u) - Ekman*(lap(u) + dz(dz(u))) + grad(p) + LiftTau(tau_u) = 0"))
problem.add_equation(eq_eval("ddt(w) - Ekman*(lap(w) + dz(dz(w))) + dz(p) + LiftTau(tau_w) = 0."))
problem.add_equation(eq_eval("div(u) + dz(w) = 0"))

problem.add_equation(eq_eval("u3D(r=1) = 0"), condition='nphi != 0')
problem.add_equation(eq_eval("w(r=1) = 0"), condition='nphi == 0')
problem.add_equation(eq_eval("azimuthal(u(r=1)) = 0"), condition='nphi == 0')
problem.add_equation(eq_eval("p(r=1) = 0"), condition='nphi == 0')

logger.info('building solver')
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2)

solver.stop_sim_time = 10

weight_r = db.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r+0*p['g'])*np.pi/Nphi/dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = np.pi/vol_test

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*u3D['g'].real**2)
        E0 = 0.5*E0*(np.pi)/Nphi/dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        Ez = np.sum(vol_correction*weight_r*u3D['g'][2].real**2)
        Ez = 0.5*Ez*(np.pi)/Nphi/dealias
        Ez = reducer.reduce_scalar(Ez, MPI.SUM)
        logger.info("t = %f, E = %e, Ez = %e" %(solver.sim_time, E0, Ez))
    solver.step(dt)
    u0['g'][0] = u0_func(solver.sim_time)
end_time = time.time()
logger.info('Run time: %f' %(end_time-start_time))

ax.pcolormesh(x, y, u3D['g'][0,:,:].real)
ax.set_aspect(1)
plt.savefig('uphi_end.png', dpi=150)

ax.pcolormesh(x, y, u3D['g'][1,:,:].real)
ax.set_aspect(1)
plt.savefig('ur_end.png', dpi=150)


