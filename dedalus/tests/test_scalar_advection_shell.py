
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic, timesteppers_sphere
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
import dedalus_sphere
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
radii = (0.5, 3)
Lmax = 15
L_dealias = 1
Nmax = 15
N_dealias = 1

ts = timesteppers.SBDF2

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=None)
b = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radii=radii)
b_inner = b.S2_basis(radius=radii[0])
b_outer = b.S2_basis(radius=radii[1])
phi, theta, r = b.local_grids((1, 1, 1))
phig,thetag,rg= b.global_grids((1,1, 1))
theta_target = thetag[0,(Lmax+1)//2,0]
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
T = field.Field(dist=d, bases=(b,), dtype=np.complex128)

# solid body rotation
Omega = 2*np.pi
u['g'][0] = Omega*r*np.sin(theta)

# perturbation
x0 = 1.75
y0 = 0
z0 = 0
T['g'] = T_init = np.exp( -( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )/0.5**2 )

# Parameters and operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([T])

problem.add_equation(eq_eval("ddt(T) = - dot(u,grad(T))"))

logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts, basis_coupling=[False, False, True])
solver.stop_sim_time = 1.

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*T['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

report_cadence = 10

dt = 1e-3

plot_cadence = report_cadence
plot = theta_target in theta

var = T['g']
name = 'T'
if plot:
    i_theta = np.argmin(np.abs(theta[0,:,0] - theta_target))
    import matplotlib.pyplot as plt
    from dedalus.extras import plot_tools
    r = r[0,0,:]
    phi = phi[:,0,0]
    rm, phim = plot_tools.quad_mesh(r,phi)
    x = rm*np.cos(phim)
    y = rm*np.sin(phim)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(x,y,var[:,i_theta,:].real)
    plt.colorbar(im)
    plt.savefig('frames/%s_%04i.png' %(name, solver.iteration//plot_cadence))

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % report_cadence == 0 and solver.iteration > 0 :
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = T0*(np.pi)/(Lmax+1)/L_dealias
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        logger.info("iter = {:d}, t = {:f}, T = {:g}".format(solver.iteration, solver.sim_time, T0))

    if solver.iteration % plot_cadence == 0:
        var = T['g']
        if plot:
            im.set_array(np.ravel(var[:,i_theta,:].real))
            plt.savefig('frames/%s_%04i.png' %(name,solver.iteration//plot_cadence))


    solver.step(dt)
end_time = time.time()

if solver.iteration % plot_cadence == 0:
    var = T['g']
    if plot:
        im.set_array(np.ravel(var[:,i_theta,:].real))
        plt.savefig('frames/%s_%04i.png' %(name,solver.iteration//plot_cadence))

T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
T0 = T0*(np.pi)/(Lmax+1)/L_dealias
T0 = reducer.reduce_scalar(T0, MPI.SUM)

error = np.sum(vol_correction*weight_r*weight_theta*(T['g'].real-T_init)**2)
error = error*(np.pi)/(Lmax+1)/L_dealias
error = reducer.reduce_scalar(error, MPI.SUM)

logger.info("error is %e" % (error/T0))

