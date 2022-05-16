"""
Script for solving the linear stability eigenvalue problem for rotating
Rayleigh-Benard convection in shell geometry. The script calculates linear
modes for a given azimuthal wavenumber m and Rayleigh number R. The aspect
ratio is given by R_i/R_o = 0.35. We non-dimensionalize lengths with the
outer radius. At the critical Rayleigh number, the imaginary part of the
eigenvalue is zero. We have implemented both stress-free and no-slip
boundary conditions.

This script is an example of an eigenvalue problem with non-constant
coefficient terms which depend on both r and theta. All non-constant
coefficient terms must be defined over the meriodional basis. The script
should be run in serial, and at the low resolution of Nmax = Lmax = 64
should take about a minute to run. The eigenvalues shift slightly if Nmax
or Lmax increase.

Our eigenvalues agree well with previous results in the literature. For
example, our critical Rayleigh numbers agree with Table 1 of [1] to
several digits of precision.

References:
    [1]: P. Marti, M. A. Calkins, K. Julien, "A computationally
         efficient spectral method for modeling coredynamics,"
         Geochemistry, Geophysics, Geosystems (2016).
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

# We use Ekman = 10^-5; the critical mode has m=13
Mmax = 14
Lmax = 64
Nmax = 64

L_dealias = 3/2
N_dealias = 3/2

dtype = np.complex128

# Problem Parameters
eta = 0.35
ri = eta
ro = 1    #Outer radius
radii = (ri, ro)

# parameters
P = 1 # Prandtl
tau = 1e5 # tau = 1/Ekman
stress_free = True

# Critical Rayleigh numbers
if stress_free:
    R = 2.1029e7
else:
    R = 2.0732e7

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), mesh=None, dtype=dtype)
b = d3.ShellBasis(c, (2*Mmax,Lmax,Nmax), radii=radii, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
b_inner = b.S2_basis(radius=ri) #For inner sphere boundary conditions
b_outer = b.S2_basis(radius=ro) #For outer sphere boundary conditions
phi, theta, r = b.local_grids((1, 1, 1)) #Get local coordinate arrays

# Fields
u = d.VectorField(c, bases=b)
p = d.Field(bases=b)
T = d.Field(bases=b)

# Eigenvalue
om = d.Field(name='om')

#Tau boundaries
tau_u_ri = d.VectorField(c, bases=b_inner)
tau_u_ro = d.VectorField(c, bases=b_outer)
tau_T_ri = d.Field(bases=b_inner)
tau_T_ro = d.Field(bases=b_outer)
tau_p = d.Field()

# Velocity boundary conditions
stress = d3.grad(u) + d3.transpose(d3.grad(u))

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# NCCs
rvec = d.VectorField(c, bases=b.meridional_basis)
rvec['g'][2] = r
ez = d.VectorField(c, bases=b.meridional_basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta)

# Substitutions
grad_u = d3.grad(u)
grad_T = d3.grad(T)
dt = lambda A: -1j*om*A

# Problem
problem = d3.EVP([p, u, T, tau_u_ri, tau_u_ro, tau_T_ri, tau_T_ro, tau_p], eigenvalue=om, namespace=locals())

problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + tau*cross(ez, u) + grad(p) - R*T*rvec - div(grad_u) + lift(tau_u_ri, -2) + lift(tau_u_ro, -1) = 0")
problem.add_equation("P*dt(T) - dot(rvec,u) - div(grad_T) + lift(tau_T_ri,-2) + lift(tau_T_ro, -1) = 0")
problem.add_equation("integ(p) = 0")

#Boundary conditions
if stress_free:
    problem.add_equation("radial(u(r=ri)) = 0")
    problem.add_equation("radial(u(r=ro)) = 0")
    problem.add_equation("angular(radial(stress(r=ri), 0), 0) = 0")
    problem.add_equation("angular(radial(stress(r=ro), 0), 0) = 0")
else:
    problem.add_equation("u(r=ri) = 0")
    problem.add_equation("u(r=ro) = 0")
problem.add_equation("T(r=ri) = 0")
problem.add_equation("T(r=ro) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(ncc_cutoff=1e-10)

if stress_free:
    target = 963.765
else:
    target = 731.753

# Select m=13
subproblem = solver.subproblems_by_group[(13, None, None)]

# Find 10 eigenvalues closest to the target
solver.solve_sparse(subproblem, 10, target)

# Report results
logger.info('predicted eigenvalue: ' + (0j+target).__format__('f'))
logger.info('calculated eigenvalue: ' + solver.eigenvalues[0].__format__('f'))
logger.info('ten eigenvalues closest to target:')
logger.info(solver.eigenvalues)
