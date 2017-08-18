"""
Dedalus script to compute eigenmodes for waves on a string in a Chebyshev basis.

We solve:
    l*u + dx(dx(u)) = 0
where l is the eigenvalue. Careful! Don't use "lambda": this is a
python reserved word!

This is a somewhat challenging problem because the eigenvectors are
sines and cosines, but we expand them in Chebyshev polynomials.

Reference:
    J. P. Boyd, "Chebyshev and Fourier Spectral Methods", 2nd edition (2001), sec 7.2.

"""

import dedalus.public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)


# Domain
Nx = 128
x_basis = de.Chebyshev('x', Nx, interval=(-1, 1))
domain = de.Domain([x_basis], np.float64)

# Problem
problem = de.EVP(domain, variables=['u', 'u_x'],eigenvalue='l')
problem.meta[:]['x']['dirichlet'] = True
problem.add_equation("l*u + dx(u_x) = 0")
problem.add_equation("u_x - dx(u) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

# Solver
solver = problem.build_solver()
t1 = time.time()
solver.solve_dense(solver.pencils[0])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Filter infinite/nan eigenmodes
finite = np.isfinite(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[finite]
solver.eigenvectors = solver.eigenvectors[:, finite]

# Sort eigenmodes by eigenvalue
order = np.argsort(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[order]
solver.eigenvectors = solver.eigenvectors[:, order]

# Plot error vs exact eigenvalues
mode_number = 1 + np.arange(len(solver.eigenvalues))
exact_eigenvalues = mode_number**2 * np.pi**2 / 4
eval_relative_error = (solver.eigenvalues - exact_eigenvalues) / exact_eigenvalues

plt.figure()
plt.semilogy(mode_number, np.abs(eval_relative_error))
plt.xlabel("Mode number")
plt.ylabel(r"$|\lambda - \lambda_{exact}|/\lambda_{exact}$")
plt.title(r"Eigenvalue relative error ($N_x=%i$)" %Nx)
plt.savefig('eval_error.png')
