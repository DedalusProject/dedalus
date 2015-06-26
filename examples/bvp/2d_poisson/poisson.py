"""
Simulation script for 2D Poisson equation.

This script is an example of how to do 2D linear boundary value problems.
It is best ran serially, and produces a plot of the solution using the included
plotting helpers.

On a single process, this should take just a few seconds to run.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools


# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, 2*np.pi))
y_basis = de.Chebyshev('y', 128, interval=(0, 1))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Poisson equation
problem = de.LBVP(domain, variables=['u','uy'])
problem.add_equation("dx(dx(u)) + dy(uy) = -10 * sin(x/2)**2 * (y - y**2)")
problem.add_equation("uy - dy(u) = 0")
problem.add_bc("left(u) = left(sin(8*x))")
problem.add_bc("right(uy) = 0")

# Build solver
solver = problem.build_solver()
solver.solve()

# Plot solution
u = solver.state['u']
u.require_grid_space()
plot_tools.plot_bot_2d(u)
plt.savefig('poisson.png')
