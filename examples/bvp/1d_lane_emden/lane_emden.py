"""
Dedalus script for the Lane-Emden equation.

This is a 1D script and should be ran serially.  It should converge within
roughly a dozen iterations, and should take under a minute to run.

In astrophysics, the Lane–Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    dr(dr(f)) + (2/r)*dr(f) + f**n = 0
    f(r=0) = 1
    dr(f)(r=0) = 0
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r).

Following [2], we rescale the equation by defining r=R*x:
    dx(dx(f)) + (2/x)*dx(f) + (R**2)*(f**n) = 0
    f(x=0) = 1
    dx(f)(x=0) = 0
    f(x=1) = 0
This is a nonlinear eigenvalue problem over the interval x=[0,1], with the
additional boundary condition fixing the eigenvalue R.

References:
    [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).

"""

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)


# Parameters
n = 3.25
ncc_cutoff = 1e-10
tolerance = 1e-10

# Build domain
x_basis = de.Chebyshev('x', 128, interval=(0, 1), dealias=2)
domain = de.Domain([x_basis], np.float64)

# Setup problem
problem = de.NLBVP(domain, variables=['f', 'fx', 'R'], ncc_cutoff=ncc_cutoff)
problem.meta[:]['x']['dirichlet'] = True
problem.parameters['n'] = n
problem.add_equation("x*dx(fx) + 2*fx = -x*(R**2)*(f**n)")
problem.add_equation("fx - dx(f) = 0")
problem.add_equation("dx(R) = 0")
problem.add_bc("left(f) = 1")
problem.add_bc("left(fx) = 0")
problem.add_bc("right(f) = 0")

# Setup initial guess
solver = problem.build_solver()
x = domain.grid(0)
f = solver.state['f']
fx = solver.state['fx']
R = solver.state['R']
f['g'] = np.cos(np.pi/2 * x)*0.9
f.differentiate('x', out=fx)
R['g'] = 3

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration()
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    logger.info('R iterate: {}'.format(R['g'][0]))

# Compare to reference solutions from Boyd
R_ref = {0.0: np.sqrt(6),
         0.5: 2.752698054065,
         1.0: np.pi,
         1.5: 3.65375373621912608,
         2.0: 4.3528745959461246769735700,
         2.5: 5.355275459010779,
         3.0: 6.896848619376960375454528,
         3.25: 8.018937527,
         3.5: 9.535805344244850444,
         4.0: 14.971546348838095097611066,
         4.5: 31.836463244694285264}
logger.info('-'*20)
logger.info('Iterations: {}'.format(solver.iteration))
logger.info('Final R iteration: {}'.format(R['g'][0]))
if n in R_ref:
    logger.info('Error vs reference: {}'.format(R['g'][0]-R_ref[n]))


