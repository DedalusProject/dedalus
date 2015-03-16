"""Dedalus script to compute eigenvalues and vectors for waves on a
string on a Chebyshev basis.

We solve

l * u + dx(dx(u)) = 0

where l is the eigenvalue. Careful! Don't use "lambda": this is a
python reserved word!

This is a somewhat challenging problem because the eigenvectors are
sines and cosines, but we expand them in Chebyshev polynomials.

reference: J. P. Boyd, "Chebyshev and Fourier Spectral Methods", 2nd
Edition (2001).

"""
import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

# Build domain
x_basis = de.Chebyshev('x', 128, interval=(-1, 1))
domain = de.Domain([x_basis], np.float64)

problem = de.EVP(domain, variables=['u', 'u_x'],eigenvalue='l')
problem.add_equation("l*u + dx(u_x) = 0")
problem.add_equation("u_x - dx(u) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

solver = problem.build_solver()

solver.solve(solver.pencils[0])

evalue_number = 1 + np.arange(len(solver.eigenvalues))
analytic_solution = np.pi**2/4. * (evalue_number)**2 

solver.eigenvalues.sort()

plt.semilogy(evalue_number,(np.abs(solver.eigenvalues - analytic_solution))/analytic_solution)
plt.xlabel("eigenvalue number")
plt.ylabel(r"$|\lambda - \lambda_{analytic}|/\lambda_{analytic}$")
plt.savefig('eval_error.png')
