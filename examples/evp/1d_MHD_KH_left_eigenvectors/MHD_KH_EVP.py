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

# The following print statements can be uncommented to test biorthogonality of solver.modified_left_eigenvectors (1-3)
# and solver.eigenvectors
# print(np.abs(np.matmul(solver.modified_left_eigenvectors1.T, solver.eigenvectors)))
# print(np.abs(np.matmul(np.conjugate(solver.modified_left_eigenvectors2.T), solver.eigenvectors)))
# print(np.abs(np.matmul(np.conj(solver.modified_left_eigenvectors3), solver.eigenvectors)))
kronecker1 = np.matmul(solver.modified_left_eigenvectors1.T, solver.eigenvectors)

# Test biorthogonality using set_state:
for n in range(5):
    solver.set_state(n)
    phi_n = solver.state['phi']['c']
    if n==0:
        right_phis = np.zeros((5, len(phi_n)), dtype=np.complex128)
        right_psis = np.zeros_like(right_phis)
        right_phi_zs = np.zeros_like(right_phis)
        right_psi_zs = np.zeros_like(right_phis)
        modified_left_phis = np.zeros_like(right_phis)
        modified_left_psis = np.zeros_like(right_phis)
        modified_left_phi_zs = np.zeros_like(right_phis)
        modified_left_psi_zs = np.zeros_like(right_phis)
    right_phis[n] = phi_n
    right_psis[n] = solver.state['psi']['c']
    right_phi_zs[n] = solver.state['phi_z']['c']
    right_psi_zs[n] = solver.state['psi_z']['c']

    solver.set_state(n, modified_left=True)
    modified_left_phis[n] = solver.state['phi']['c']
    modified_left_psis[n] = solver.state['psi']['c']
    modified_left_phi_zs[n] = solver.state['phi_z']['c']
    modified_left_psi_zs[n] = solver.state['psi_z']['c']

rights = np.array([right_phis, right_phi_zs, right_psis, right_psi_zs])
modified_lefts = np.array([modified_left_phis, modified_left_phi_zs, modified_left_psis, modified_left_psi_zs])
kronecker2 = np.tensordot(modified_lefts, rights, axes=([0,2],[0,2]))
# Should produce True
print(np.allclose(kronecker1, kronecker2))
