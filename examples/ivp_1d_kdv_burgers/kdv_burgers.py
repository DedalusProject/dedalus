"""
Dedalus script solving the 1D 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should be ran serially and take
about 1 minute to complete.

We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

# TODO: Fix default matrix_coupling for 1D fourier problems

# Parameters
Nx = 1024
Lx = 10
a = 2e-4
b = 1e-4
timestepper = d3.SBDF2
timestep = 2e-3
dealias = 3/2
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.Field(name='u', bases=xbasis)

# Problem
problem = d3.IVP(variables=[u])
dx = lambda A: d3.Differentiate(A, xcoord)
dt = d3.TimeDerivative
problem.add_equation((dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))), -u*dx(u)))

# Initial conditions
x = xbasis.local_grid(1)
n = 20
u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)

# Solver
solver = problem.build_solver(timestepper, matrix_coupling=[True])
solver.stop_wall_time = 60
solver.stop_iteration = 5000

# Main loop
u.require_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]
while solver.ok:
    solver.step(timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, timestep))
    if solver.iteration % 25 == 0:
        u.require_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)

# Plot
u_array = np.array(u_list)
t_array = np.array(t_list)
plt.figure()
plt.pcolormesh(x.ravel(), t_array, u_array, cmap='RdBu_r', shading='nearest')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
plt.tight_layout()
plt.savefig('kdv_burgers.png')

