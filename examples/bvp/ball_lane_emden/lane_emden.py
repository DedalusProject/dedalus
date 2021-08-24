"""
Dedalus script for solving the Lane-Emden equation. This script demonstrates
solving a spherically symmetric nonlinear boundary value problem inside the
ball. It should be ran serially, should converge within roughly a dozen
iterations, and should take just a few seconds to run.

In astrophysics, the Lane–Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    lap(f) + f**n = 0
    f(r=0) = 1
    f(r=R) = 0
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r).

Following [2], we rescale radius by 1/R, giving:
    lap(r) + (R**2)*(f**n) = 0
    f(r=0) = 1
    f(r=1) = 0
This is a nonlinear eigenvalue problem over the unit ball, with the additional
boundary condition fixing the eigenvalue R.

References:
    [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)


# Parameters
Nr = 128
n = 3.25
ncc_cutoff = 1e-10
tolerance = 1e-10
dealias = 2
dtype = np.float64

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,))
b = d3.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
f = d3.Field(dist=d, name='f', bases=(b,), dtype=dtype)
R = d3.Field(dist=d, name='R', dtype=dtype)
t = d3.Field(dist=d, name='t', bases=(b.S2_basis(),), dtype=dtype)

# Problem
lap = lambda A: d3.Laplacian(A, c)
LT = lambda A, n: d3.LiftTau(A, b, n)
problem = d3.NLBVP(variables=[f, R, t])
problem.add_equation((lap(f) + LT(t,-1), -R**2*f**n))
problem.add_equation((f(r=0), 1))
problem.add_equation((f(r=1), 0))

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

# Initial guess
f['g'] = np.cos(np.pi/2 * r)
R['g'] = 3

# Newton iterations
pert_norm = np.inf
while pert_norm > tolerance:
    solver.newton_iteration()
    pert_norm = np.sum([np.sum(np.abs(pert['c'])) for pert in solver.perturbations])
    logger.info(f'Perturbation norm: {pert_norm}')
    R0 = R['g'][0]
    logger.info(f'R iterate: {R0}')

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

