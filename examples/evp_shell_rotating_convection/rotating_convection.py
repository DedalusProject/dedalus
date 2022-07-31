"""
Dedalus script solving the linear stability eigenvalue problem for rotating
Rayleigh-Benard convection in a shell. This script demonstrates solving an
eigenvalue problem with non-constant coefficients that depend on both radius
and colatitude. It should take about a minute to run (serial only).

The aspect ratio of the shell is R_inner / R_outer = 0.35, and the problem is
non-dimensionalized using the outer radius and the viscous time. The script
calculates the eigenmodes for an Ekman number of 1e-5, where the critical
mode has an azimuthal wavenumber of m=13. At the critical Rayleigh number,
the imaginary part of the eigenvalue is zero.

Both stress-free (default) and no-slip boundary conditions are implemented.
For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and temperature. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

The eigenvalues are not fully converged at the given resolution and shift slightly
if the resolution is increased. For the given resolutions, the eigenvalues agree
with Table 1 of [1] to several digits of precision.

To run and print the calculated eigenvalues:
    $ python3 rotating_convection.py

References:
    [1]: P. Marti, M. A. Calkins, K. Julien, "A computationally
         efficient spectral method for modeling coredynamics,"
         Geochemistry, Geophysics, Geosystems (2016).
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Nphi = 28  # Critical mode has m=13
Ntheta = 64
Nr = 64
Ri = 0.35
Ro = 1
Prandtl = 1
Ekman = 1e-5
stress_free = True
dtype = np.complex128

# Critical Rayleigh numbers
if stress_free:
    Rayleigh = 2.1029e7
else:
    Rayleigh = 2.0732e7

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dtype=dtype)
sphere = shell.outer_surface
phi, theta, r = dist.local_grids(shell)

# Fields
om = dist.Field(name='om')
u = dist.VectorField(coords, name='u', bases=shell)
p = dist.Field(name='p', bases=shell)
T = dist.Field(name='T', bases=shell)
tau_u1 = dist.VectorField(coords, bases=sphere)
tau_u2 = dist.VectorField(coords, bases=sphere)
tau_T1 = dist.Field(bases=sphere)
tau_T2 = dist.Field(bases=sphere)
tau_p = dist.Field()

# Substitutions
dt = lambda A: -1j*om*A
rvec = dist.VectorField(coords, bases=shell.meridional_basis)
rvec['g'][2] = r
ez = dist.VectorField(coords, bases=shell.meridional_basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta)
lift_basis = shell.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1) # First-order reduction
strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))

# Problem
problem = d3.EVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p], eigenvalue=om, namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + (1/Ekman)*cross(ez, u) + grad(p) - Rayleigh*T*rvec - div(grad_u) + lift(tau_u2) = 0")
problem.add_equation("Prandtl*dt(T) - dot(rvec,u) - div(grad_T) + lift(tau_T2) = 0")
problem.add_equation("integ(p) = 0")
if stress_free:
    problem.add_equation("radial(u(r=Ri)) = 0")
    problem.add_equation("radial(u(r=Ro)) = 0")
    problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
    problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
else:
    problem.add_equation("u(r=Ri) = 0")
    problem.add_equation("u(r=Ro) = 0")
problem.add_equation("T(r=Ri) = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(ncc_cutoff=1e-10)

# Select m=13
subproblem = solver.subproblems_by_group[(13, None, None)]

# Find 10 eigenvalues closest to the target
if stress_free:
    target = 963.765
else:
    target = 731.753
solver.solve_sparse(subproblem, 10, target)

# Report results
logger.info(f"Predicted eigenvalue: {target+0j:f}")
logger.info(f"Calculated eigenvalue: {solver.eigenvalues[0]:f}")
logger.info("Ten eigenvalues closest to target:")
logger.info(solver.eigenvalues)
