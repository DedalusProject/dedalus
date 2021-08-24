"""
Dedalus script to compute the eigenmodes of waves on a clamped string.
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
c = d3.Coordinate('x')
d = d3.Distributor((c,))
b = d3.Chebyshev(c, size=Nx, bounds=(0, Lx))

# Fields
u = d3.Field(d, bases=(b,), dtype=dtype)
t1 = d3.Field(d, dtype=dtype)
t2 = d3.Field(d, dtype=dtype)
s = d3.Field(d, dtype=dtype)

# Problem
dx = lambda A: d3.Differentiate(A, c)
def build_P(n):
    b2 = dx(dx(u)).domain.bases[0]
    P = d3.Field(d, bases=(b2,), dtype=dtype)
    P['c'][n] = 1
    return P
LT = lambda A, n: A * build_P(n)
problem = d3.EVP(variables=[u, t1, t2], eigenvalue=s)
problem.add_equation((s*u + dx(dx(u)) + LT(t1,-1) + LT(t2,-2), 0))
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
