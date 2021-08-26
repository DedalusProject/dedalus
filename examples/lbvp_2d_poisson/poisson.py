"""
Dedalus script solving the 2D Poisson equation with mixed boundary conditions.
This script demonstrates solving a 2D cartesian linear boundary value problem.
and produces a plot of the solution. It should be ran serially and take just a
few seconds to complete.

We use a Fourier(x) * Chebyshev(y) discretization to solve the LBVP:
    dx(dx(u)) + dy(dy(u)) = f
    u(y=0) = g
    dy(u)(y=Ly) = h
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

# TODO: make proper plotting using plotbot/xarray
# TODO: improve build_P or add LiftTau for Jacobi
# TODO: make parallel safe?

# Parameters
Nx = 256
Ny = 128
Lx = 2 * np.pi
Ly = 1
dtype = np.float64

# Bases
c = d3.CartesianCoordinates('x', 'y')
d = d3.Distributor((c,))
xb = d3.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx))
yb = d3.ChebyshevT(c.coords[1], size=Ny, bounds=(0, Ly))

# Fields
u = d3.Field(d, bases=(xb, yb), dtype=dtype)
t1 = d3.Field(d, bases=(xb,), dtype=dtype)
t2 = d3.Field(d, bases=(xb,), dtype=dtype)
f = d3.Field(d, bases=(xb, yb), dtype=dtype)
g = d3.Field(d, bases=(xb,), dtype=dtype)
h = d3.Field(d, bases=(xb,), dtype=dtype)
x = xb.local_grid()
y = yb.local_grid()
f['g'] = -10 * np.sin(x/2)**2 * (y - y**2)
g['g'] = np.sin(8*x)
h['g'] = 0

# Problem
dy = lambda A: d3.Differentiate(A, c.coords[1])
lap = lambda A: d3.Laplacian(A, c)
def build_P(n):
    yb2 = lap(u).domain.bases[1]
    P = d3.Field(d, bases=(yb2,), dtype=dtype)
    P['c'][0, n] = 1
    return P
LT = lambda A, n: A * build_P(n)
problem = d3.LBVP(variables=[u, t1, t2])
problem.add_equation((lap(u) + LT(t1,-1) + LT(t2,-2), f))
problem.add_equation((u(y=0), g))
problem.add_equation((dy(u)(y=Ly), h))

# Solver
solver = problem.build_solver()
solver.solve()

# Plot
plt.figure(figsize=(6, 4))
plt.imshow(u['g'].T)
plt.colorbar()
plt.tight_layout()
plt.savefig('poisson.png')
