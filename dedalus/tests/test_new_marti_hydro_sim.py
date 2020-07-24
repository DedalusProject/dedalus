

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
dt = 0.02
t_end = 20
ts = timesteppers.SBDF2
dtype = np.float64
Nphi = 2*(Lmax+1)

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (Nphi, Lmax+1, Nmax+1), radius=radius, dtype=dtype)
bk2 = basis.BallBasis(c, (Nphi, Lmax+1, Nmax+1), k=2, radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
tau = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)

# Boundary conditions
u_BC = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
u_BC['g'][2] = 0. # u_r = 0
u_BC['g'][1] = - u0*np.cos(theta)*np.cos(phi)
u_BC['g'][0] = u0*np.sin(phi)

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, tau])
# Equations for ell != 0
problem.add_equation(eq_eval("div(u) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(u) - nu*lap(u) + grad(p) = - dot(u,grad(u)) - Om*cross(ez, u)"), condition="ntheta != 0")
problem.add_equation(eq_eval("u(r=1) = u_BC"), condition="ntheta != 0")
# Equations for ell == 0
problem.add_equation(eq_eval("p = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tau = 0"), condition="ntheta == 0")
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (2,       ell+deg+0.5)
    return dedalus_sphere.jacobi.coefficient_connection(N - ell//2 + 1,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    if subproblem.group[1] != 0:
        ell = subproblem.group[1]
        L = subproblem.L_min
        if dtype == np.complex128:
            N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)
            tau_columns = np.zeros((L.shape[0], 3))
            tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
            L[:,-3:] = tau_columns
        elif dtype == np.float64:
            NL = Nmax - ell//2 + 1
            N0, N1, N2, N3 = BC_rows(Nmax, ell, 4) * 2
            tau_columns = np.zeros((L.shape[0], 6))
            tau_columns[N0:N0+NL,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N1+NL,1] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N2+NL,2] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N0+NL:N0+2*NL,3] = tau_columns[N0:N0+NL,0]
            tau_columns[N1+NL:N1+2*NL,4] = tau_columns[N1:N1+NL,1]
            tau_columns[N2+NL:N2+2*NL,5] = tau_columns[N2:N2+NL,2]
            L[:,-6:] = tau_columns
        L.eliminate_zeros()
        subproblem.expand_matrices(['M','L'])

# Analysis
t_list = []
E_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radial_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

# Main loop
start_time = time.time()
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
logger.info('Run time:', end_time-start_time)
