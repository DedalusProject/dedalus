"""
Simulation script for 2D Poisson equation.

This script is an example of how to do 2D linear boundary value problems.
It is best ran serially, and produces a plot of the solution using the included
plotting helpers.

On a single process, this should take just a few seconds to run.

"""

import os
import numpy as np

from dedalus2 import public as de
from dedalus2.extras import plot_tools


# Create bases and domain
x_basis = de.Fourier(256, interval=(0, 2*np.pi))
y_basis = de.Chebyshev(129, interval=(0, 1))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.ParsedProblem(axis_names=['x', 'y'],
                           field_names=['u', 'uy'])
problem.add_equation("dx(dx(u)) + dy(uy) = -10 * sin(x/2)**2 * (y - y**2)")
problem.add_equation("uy - dy(u) = 0")
problem.add_left_bc("u = sin(8*x)")
problem.add_right_bc("uy = 0")
problem.expand(domain)

# Build solver
solver = de.solvers.LinearBVP(problem, domain)
solver.solve()

# Plot solution
u = solver.state['u']
fig = plot_tools.plot_bot_2d(u)
fig.savefig('poisson.png')
