"""
Runs an eigenvalue problem for the KH instability of a U = tanh(z) shear layer in MHD
for purposes of testing left eigenvectors

TODO: Remove this script before merging with main. This is just meant for testing.
"""

import time
import numpy as np
# import matplotlib.pyplot as plt
import dedalus.public as de
# from mpi4py import MPI
# CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)


# Global parameters
Nz = 128
Lz = 5.0*np.pi
kx = 0.2
MA = 40.0

# Create bases and domain
z_basis = de.Chebyshev('z', Nz, interval=(-0.5*Lz, 0.5*Lz))
domain = de.Domain([z_basis], grid_dtype=np.complex128)
z = domain.grid(0)

# Problem
problem = de.EVP(domain, variables=['phi', 'phi_z', 'psi', 'psi_z'], eigenvalue='omega')
problem.parameters['MA2inv'] = 1.0/MA**2.0
problem.parameters['kx'] = kx
nccU = domain.new_field(name='U')
nccDU = domain.new_field(name='DU')
nccU['g'] = np.tanh(z)
problem.parameters['U'] = nccU
nccDU['g'] = 1/(np.cosh(z)**2)
problem.parameters['DU'] = nccDU

problem.substitutions['dx(A)'] = "1.0j*kx*A"
problem.substitutions['dt(A)'] = "1.0j*omega*A"
problem.substitutions['w'] = "-dx(phi)"
problem.substitutions['Bz'] = "-dx(psi)"
problem.substitutions['Bx'] = "psi_z"
problem.substitutions['Jz'] = "dx(dx(psi)) + dz(Bx)"
problem.substitutions['zeta'] = "dx(dx(phi)) + dz(phi_z)"

problem.add_equation("dt(zeta) + U*dx(zeta) - dx(phi)*dz(DU) - MA2inv*dx(Jz) = 0")
problem.add_equation("dt(psi) + U*dx(psi) - dx(phi) = 0")
problem.add_equation("phi_z - dz(phi) = 0")
problem.add_equation("psi_z - dz(psi) = 0")

problem.add_bc("left(phi) = 0")
problem.add_bc("right(phi_z) = 0")
problem.add_bc("right(Bz) = 0")
solver = problem.build_solver()

solver.solve_sparse(solver.pencils[0], N=5, target=0.0, rebuild_coeffs=True, left=True)
# print(np.abs(np.matmul(np.conjugate(solver.modified_left_eigenvectors.T), solver.eigenvectors)))
print(np.abs(np.matmul(solver.modified_left_eigenvectors1.T, solver.eigenvectors)))
print(np.abs(np.matmul(np.conjugate(solver.modified_left_eigenvectors2.T), solver.eigenvectors)))
print(np.abs(np.matmul(np.conj(solver.modified_left_eigenvectors3), solver.eigenvectors)))