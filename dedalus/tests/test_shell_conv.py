import numpy as np
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.core import arithmetic, timesteppers, problems, solvers
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import dedalus_sphere
from mpi4py import MPI
import time
from dedalus_sphere import ball, intertwiner
from dedalus_sphere import jacobi128 as jacobi

import logging
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank

Lmax = 15
Nmax = 15

# right now can't run with dealiasing
L_dealias = 1
N_dealias = 1

# parameters
Ekman = 1e-3
Prandtl = 1
Rayleigh = 100
r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,))
b    = de.basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii)
bk2  = de.basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=2, radii=radii)
b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = b.local_grids((1, 1, 1))

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_u_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
tau_T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
tau_u_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
tau_T_outer = de.field.Field(dist=d, bases=(b_outer,), dtype=np.complex128)

ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r/r_outer

T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
T_inner['g'] = 1.

# initial condition
A = 0.1
x = 2*r-r_inner-r_outer
T['g'] = r_inner*r_outer/r - r_inner + 210*A/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

# Parameters and operators
div = lambda A: de.operators.Divergence(A, index=0)
lap = lambda A: de.operators.Laplacian(A, c)
grad = lambda A: de.operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: de.operators.CrossProduct(A, B)
ddt = lambda A: de.operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, T, tau_u_inner, tau_T_inner, tau_u_outer, tau_T_outer])
problem.add_equation(eq_eval("Ekman*ddt(u) - Ekman*lap(u) + grad(p) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - 2*cross(ez, u)"), condition = "ntheta != 0")
problem.add_equation(eq_eval("u = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("div(u) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("p = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("ddt(T) - lap(T)/Prandtl = - dot(u,grad(T))"))
problem.add_equation(eq_eval("u(r=7/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_inner = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=7/13) = T_inner"))
problem.add_equation(eq_eval("u(r=20/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_outer = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=20/13) = 0"))
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.CNAB2)

# Add taus

# ChebyshevV
alpha_BC = (2-1/2, 2-1/2)

def C(N):
    ab = alpha_BC
    cd = (b.alpha[0]+2,b.alpha[1]+2)
    return dedalus_sphere.jacobi128.coefficient_connection(N,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list

for subproblem in solver.subproblems:
    if subproblem.group[1] != 0:
        ell = subproblem.group[1]
        M = subproblem.M_min
        L = subproblem.L_min
        shape = M.shape
        subproblem.M_min[:,-8:] = 0
        subproblem.M_min.eliminate_zeros()
        N0, N1, N2, N3, N4 = BC_rows(Nmax, 5)
        tau_columns = np.zeros((shape[0], 8))
        tau_columns[  :N0,0] = (C(Nmax))[:,-1]
        tau_columns[N0:N1,1] = (C(Nmax))[:,-1]
        tau_columns[N1:N2,2] = (C(Nmax))[:,-1]
        tau_columns[N3:N4,3] = (C(Nmax))[:,-1]
        tau_columns[  :N0,4] = (C(Nmax))[:,-2]
        tau_columns[N0:N1,5] = (C(Nmax))[:,-2]
        tau_columns[N1:N2,6] = (C(Nmax))[:,-2]
        tau_columns[N3:N4,7] = (C(Nmax))[:,-2]
        subproblem.L_min[:,-8:] = tau_columns
        subproblem.L_min.eliminate_zeros()
        subproblem.expand_matrices(['M','L'])
    else: # ell = 0
        M = subproblem.M_min
        L = subproblem.L_min
        shape = M.shape
        subproblem.M_min[:,-8:] = 0
        subproblem.M_min.eliminate_zeros()
        N0, N1, N2, N3, N4 = BC_rows(Nmax, 5)
        subproblem.L_min[N3:N4,N4+3] = (C(Nmax))[:,-1].reshape((N0,1))
        subproblem.L_min[N3:N4,N4+7] = (C(Nmax))[:,-2].reshape((N0,1))
        subproblem.L_min.eliminate_zeros()
        subproblem.expand_matrices(['M','L'])

reducer = GlobalArrayReducer(d.comm_cart)

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(r_outer**3-r_inner**3)
vol_correction = vol/vol_test

t = 0.

t_list = []
E_list = []

# timestepping loop
start_time = time.time()
iter = 0

# Integration parameters
dt = 1.e-4
t_end = 1.25
solver.stop_sim_time = t_end

while solver.ok:

    if iter % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias/vol
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = 0.5*T0*(np.pi)/(Lmax+1)/L_dealias/vol
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, T0={:e}".format(solver.iteration, dt, solver.sim_time, E0, T0))
        t_list.append(t)
        E_list.append(E0)

    solver.step(dt)

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_conv.dat',np.array([t_list,E_list]))

