"""
1D Korteweg-de Vries / Burgers equation

This script should be ran serially (because it is 1D), and creates a space-time
plot of the computed solution.

"""

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)


# Bases and domain
x_basis = de.Fourier('x', 1024, interval=(-2, 8), dealias=3/2)
domain = de.Domain([x_basis], np.float64)

# Problem
problem = de.IVP(domain, variables=['u', 'ux', 'uxx'])
problem.parameters['a'] = 2e-4
problem.parameters['b'] = 1e-4
problem.add_equation("dt(u) - a*dx(ux) - b*dx(uxx) = -u*ux")
problem.add_equation("ux - dx(u) = 0")
problem.add_equation("uxx - dx(ux) = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_wall_time = 60
solver.stop_iteration = 5000

# Initial conditions
x = domain.grid(0)
u = solver.state['u']
ux = solver.state['ux']
uxx = solver.state['uxx']

n = 20
u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*x)**2) / (2*n)
u.differentiate(0, out=ux)
ux.differentiate(0, out=uxx)

# Store data for final plot
u.set_scales(1, keep_data=True)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 2e-3
while solver.ok:
    solver.step(dt)
    if solver.iteration % 20 == 0:
        u.set_scales(1, keep_data=True)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 100 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

# Create space-time plot
u_array = np.array(u_list)
t_array = np.array(t_list)
xmesh, ymesh = quad_mesh(x=x, y=t_array)
plt.figure()
plt.pcolormesh(xmesh, ymesh, u_array, cmap='RdBu_r')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('KdV-Burgers, (a,b)=(%g,%g)' %(problem.parameters['a'], problem.parameters['b']))
plt.savefig('kdv_burgers.png')

