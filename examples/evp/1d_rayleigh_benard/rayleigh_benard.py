"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)


# Global parameters
Nz = 64
Prandtl = 1
Rayleigh = 1710
kx_global = np.linspace(3.0, 3.25, 50)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', Nz, interval=(-1/2, 1/2))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
problem = de.EVP(domain, variables=['p','b','u','w','bz','uz','wz'], eigenvalue='omega')
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['kx'] = 1
problem.substitutions['dx(A)'] = "1j*kx*A"
problem.substitutions['dt(A)'] = "-1j*omega*A"
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(b) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0")
solver = problem.build_solver()

# Create function to compute max growth rate for given kx
def max_growth_rate(kx):
    logger.info('Computing max growth rate for kx = %f' %kx)
    # Change kx parameter
    problem.namespace['kx'].value = kx
    # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
    solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True)
    # Return largest imaginary part
    return np.max(solver.eigenvalues.imag)

# Compute growth rate over local wavenumbers
kx_local = kx_global[CW.rank::CW.size]
t1 = time.time()
growth_local = np.array([max_growth_rate(kx) for kx in kx_local])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Reduce growth rates to root process
growth_global = np.zeros_like(kx_global)
growth_global[CW.rank::CW.size] = growth_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

# Plot growth rates from root process
if CW.rank == 0:
    plt.plot(kx_global, growth_global, '.')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$\mathrm{Im}(\omega)$')
    plt.title(r'Rayleigh-Benard growth rates ($\mathrm{Ra} = %.2f, \; \mathrm{Pr} = %.2f$)' %(Rayleigh, Prandtl))
    plt.savefig('growth_rates.png')
