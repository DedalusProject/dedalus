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
# TODO: indexing on coord systems by name or axis?

# Parameters
Nx = 256
Ny = 128
Lx = 2 * np.pi
Ly = 1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords.coords[0], size=Nx, bounds=(0, Lx))
ybasis = d3.ChebyshevT(coords.coords[1], size=Ny, bounds=(0, Ly))

# Fields
u = dist.Field(name='u', bases=(xbasis, ybasis))
tau1 = dist.Field(name='tau1', bases=xbasis)
tau2 = dist.Field(name='tau2', bases=xbasis)

# Forcing
x = xbasis.local_grid()
y = ybasis.local_grid()
f = dist.Field(bases=(xbasis, ybasis))
g = dist.Field(bases=xbasis)
h = dist.Field(bases=xbasis)
f['g'] = -10 * np.sin(x/2)**2 * (y - y**2)
g['g'] = np.sin(8*x)
h['g'] = 0

# Problem
dy = lambda A: d3.Differentiate(A, coords.coords[1])
lap = lambda A: d3.Laplacian(A, coords)
def build_P(n):
    yb2 = lap(u).domain.bases[1]
    P = dist.Field(bases=yb2)
    P['c'][0, n] = 1
    return P
LT = lambda A, n: A * build_P(n)
problem = d3.LBVP(variables=[u, tau1, tau2])
problem.add_equation((lap(u) + LT(tau1,-1) + LT(tau2,-2), f))
problem.add_equation((u(y=0), g))
problem.add_equation((dy(u)(y=Ly), h))

# Solver
solver = problem.build_solver()
solver.solve()

# Plot
plt.figure(figsize=(6, 4))
plt.imshow(u['g'].T)
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('poisson.png')
