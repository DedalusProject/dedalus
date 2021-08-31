"""
Dedalus script solving the Lane-Emden equation. This script demonstrates
solving a spherically symmetric nonlinear boundary value problem inside the
ball. It should be ran serially, should converge within roughly a dozen
iterations, and should take just a few seconds to run.

In astrophysics, the Lane-Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    lap(f) + f**n = 0
    f(r=0) = 1
    f(r=R) = 0
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r).

Following [2], we rescale r by 1/R, giving:
    lap(f) + (R**2)*(f**n) = 0
    f(r=0) = 1
    f(r=1) = 0
This is a nonlinear eigenvalue problem over the unit ball, with the additional
boundary condition fixing the eigenvalue R.

We can eliminate R by rescaling f by R**(2/(n-1)), giving:
    lap(f) + f**n = 0
    f(r=1) = 0
and R can then be recovered from f(r=0) = R**(2/(n-1)).

For a scalar Laplacian in the ball, we need a single tau term. Here we choose
to lift it to the natural output (k=2) basis.

References:
    [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: print NCC bandwidths and optimize parameters
# TODO: make parallel safe


# Parameters
Nr = 64
n = 3.0
ncc_cutoff = 1e-3
tolerance = 1e-10
dealias = 2
dtype = np.float64

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.BallBasis(coords, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)

# Fields
f = dist.Field(name='f', bases=basis)
tau = dist.Field(name='tau', bases=basis.S2_basis(radius=1))

# Substitutions
lap = lambda A: d3.Laplacian(A, coords)
lift_basis = basis.clone_with(k=2) # Natural output basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)

# Problem
problem = d3.NLBVP([f, tau])
problem.add_equation((lap(f) + lift(tau,-1), -f**n))
problem.add_equation((f(r=1), 0))

# Initial guess
phi, theta, r = basis.local_grids((1, 1, 1))
R0 = 5
f['g'] = R0**(2/(n-1)) * (1 - r**2)**2

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
pert_norm = np.inf
while pert_norm > tolerance:
    solver.newton_iteration()
    pert_norm = np.sum([np.sum(np.abs(pert['c'])) for pert in solver.perturbations])
    logger.info(f'Perturbation norm: {pert_norm:.3e}')
    f0 = f(r=0).evaluate()['g'][0,0,0]
    Ri = f0**((n-1)/2)
    logger.info(f'R iterate: {Ri}')

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
logger.info(f'Iterations: {solver.iteration}')
logger.info(f'Final R iteration: {Ri}')
if n in R_ref:
    logger.info(f'Error vs reference: {Ri-R_ref[n]:.3e}')

