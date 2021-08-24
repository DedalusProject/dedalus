"""
Dedalus script to compute the eigenmodes of waves on a clamped string.

We solve the eigenvalue problem:
    s*u + dx(dx(u)) = 0
where s is the eigenvalue, using a Chebyshev basis.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.dev as de

import logging
logger = logging.getLogger(__name__)

# TODO: improve build_P or add LiftTau for Jacobi

# Parameters
Nx = 128
Lx = 1
dtype = np.complex128

# Bases
c = de.coords.Coordinate('x')
d = de.distributor.Distributor((c,))
b = de.basis.ChebyshevT(c, size=Nx, bounds=(0, Lx))

# Fields
u = de.field.Field(d, bases=(b,), dtype=dtype)
t1 = de.field.Field(d, dtype=dtype)
t2 = de.field.Field(d, dtype=dtype)
s = de.field.Field(d, dtype=dtype)

# Problem
dx = lambda A: de.operators.Differentiate(A, c)
def build_P(n):
    b2 = dx(dx(u)).domain.bases[0]
    P = de.field.Field(d, bases=(b2,), dtype=dtype)
    P['c'][n] = 1
    return P
LT = lambda A, n: A * build_P(n)
problem = de.problems.EVP(variables=[u, t1, t2], eigenvalue=s)
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
