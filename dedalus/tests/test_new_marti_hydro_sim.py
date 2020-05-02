

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic, timesteppers_sphere
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
import dedalus_sphere
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
radius = 1
Lmax = 15
L_dealias = 1
Nmax = 15
N_dealias = 1
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = field.Field(dist=d, bases=(b,), dtype=np.complex128)
taus = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)

# Boundary conditions
u_BC = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
u_BC['g'][2] = 0. # u_r = 0
u_BC['g'][1] = - u0*np.cos(theta)*np.cos(phi)
u_BC['g'][0] = u0*np.sin(phi)

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, taus])
problem.add_equation(eq_eval("div(u) = 0"))
problem.add_equation(eq_eval("dt(u) - nu*lap(u) + grad(p) = - dot(u,grad(u)) - Om*cross(ez, u)"))
problem.add_equation(eq_eval("u(r=1) = u_BC"))
print("Problem built")

# Integration parameters
dt = 0.01
t_end = 20

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.SBDF2)
solver.stop_sim_time = t_end

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (2,       ell+deg+0.5)
    return dedalus_sphere.jacobi128.coefficient_connection(N - ell//2,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    M = subproblem.M_min
    L = subproblem.L_min
    shape = M.shape
    subproblem.M_min[:,-3:] = 0
    subproblem.M_min.eliminate_zeros()
    N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)
    tau_columns = np.zeros((shape[0], 3))
    tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
    tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
    tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
    subproblem.L_min[:,-3:] = tau_columns
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

# Handle ell=0
for subproblem in solver.subproblems:
    if subproblem.group[1] == 0:
        shape = subproblem.M_min.shape
        subproblem.M_min = sparse.csr_matrix(shape, dtype=np.complex128)
        subproblem.L_min = sparse.identity(shape[0], format='csr', dtype=np.complex128)
        subproblem.expand_matrices(['M','L'])
        subproblem.rhs_map[:,:] = 0
        subproblem.rhs_map.eliminate_zeros()


t_list = []
E_list = []
reducer = GlobalArrayReducer(d.comm_cart)

# timestepping loop
start_time = time.time()

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

while solver.ok:
    if solver.iteration % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("t = %f, E = %e" %(solver.sim_time, E0))
        t_list.append(solver.sim_time)
        E_list.append(E0)
    solver.step(dt)
end_time = time.time()
print('Run time:', end_time-start_time)
