"""
Dedalus script for calculating the maximum linear growth rates in no-slip
Rayleigh-Benard convection over a range of horizontal wavenumbers. This script
demonstrates solving a 1D eigenvalue problem in a Cartesian domain. It can
be ran serially or in parallel, and produces a plot of the highest growth rate
found for each horizontal wavenumber. It should take a few seconds to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard_evp.py
"""

import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def max_growth_rate(Rayleigh, Prandtl, kx, Nz, NEV=10, target=0):
    """Compute maximum linear growth rate."""

    # Parameters
    Nx = 2
    Lx = 2 * np.pi / kx
    Lz = 1

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx))
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

    # Fields
    omega = dist.Field(name='omega')
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
    dt = lambda A: -1j*omega*A

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals(), eigenvalue=omega)
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) - ez@u = 0")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = 0")
    problem.add_equation("b(z=0) = 0")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=target)
    return np.max(solver.eigenvalues.imag)


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    comm = MPI.COMM_WORLD

    # Parameters
    Nz = 64
    Rayleigh = 1710
    Prandtl = 1
    kx_global = np.linspace(3.0, 3.25, 50)
    NEV = 10

    # Compute growth rate over local wavenumbers
    kx_local = kx_global[comm.rank::comm.size]
    t1 = time.time()
    growth_local = np.array([max_growth_rate(Rayleigh, Prandtl, kx, Nz, NEV=NEV) for kx in kx_local])
    t2 = time.time()
    logger.info('Elapsed solve time: %f' %(t2-t1))

    # Reduce growth rates to root process
    growth_global = np.zeros_like(kx_global)
    growth_global[comm.rank::comm.size] = growth_local
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
    else:
        comm.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

    # Plot growth rates from root process
    if comm.rank == 0:
        plt.figure(figsize=(6,4))
        plt.plot(kx_global, growth_global, '.')
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$\mathrm{Im}(\omega)$')
        plt.title(r'Rayleigh-Benard growth rates ($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f$)' %(Rayleigh, Prandtl))
        plt.tight_layout()
        plt.savefig('growth_rates.pdf')
