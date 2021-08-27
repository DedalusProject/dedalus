"""
Dedalus script computing the eigenmodes of waves on a clamped string.
This script demonstrates solving a 1D eigenvalue problem and produces
a plot of the relative error of the eigenvalues.  It should be ran serially
and take just a few seconds to complete.

We use a Chebyshev basis to solve the EVP:
    s*u + dx(dx(u)) = 0
where s is the eigenvalue.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

# TODO: improve build_P or add LiftTau for Jacobi

# Parameters
Nx = 128
Lx = 1
dtype = np.complex128

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx))

# Fields
u = dist.Field(name='u', bases=xbasis)
tau1 = dist.Field(name='tau1')
tau2 = dist.Field(name='tau2')
s = dist.Field(name='s')

# Problem
dx = lambda A: d3.Differentiate(A, xcoord)
def build_P(n):
    b2 = dx(dx(u)).domain.bases[0]
    P = dist.Field(bases=b2)
    P['c'][n] = 1
    return P
LT = lambda A, n: A * build_P(n)
problem = d3.EVP(variables=[u, tau1, tau2], eigenvalue=s)
problem.add_equation((s*u + dx(dx(u)) + LT(tau1,-1) + LT(tau2,-2), 0))
problem.add_equation((u(x=0), 0))
problem.add_equation((u(x=Lx), 0))

# Solve
solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0])
evals = np.sort(solver.eigenvalues)
n = 1 + np.arange(evals.size)
true_evals = (n * np.pi / Lx)**2
relative_error = np.abs(evals - true_evals) / true_evals

# Plot
plt.figure(figsize=(6, 4))
plt.semilogy(n, relative_error, '.')
plt.xlabel("eigenvalue number")
plt.ylabel("relative eigenvalue error")
plt.tight_layout()
plt.savefig('eigenvalue_error.pdf')
