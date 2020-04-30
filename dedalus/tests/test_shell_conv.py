import numpy as np
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.core import arithmetic, timesteppers_sphere, problems, solvers
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import dedalus_sphere
from mpi4py import MPI
import time
from dedalus_sphere import timesteppers, ball, intertwiner
from dedalus_sphere import jacobi128 as jacobi

import logging
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank

Lmax = 31
Nmax = 31

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

u_rhs = de.field.Field(dist=d, bases=(bk2,), tensorsig=(c,), dtype=np.complex128)
p_rhs = de.field.Field(dist=d, bases=(bk2,), dtype=np.complex128)
T_rhs = de.field.Field(dist=d, bases=(bk2,), dtype=np.complex128)

u_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
u_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
T_outer = de.field.Field(dist=d, bases=(b_outer,), dtype=np.complex128)
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
dt = lambda A: de.operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, T])
problem.add_equation(eq_eval("Ekman*dt(u) - Ekman*lap(u) + grad(p) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - 2*cross(ez, u)"))
#problem.add_equation(eq_eval("Ekman*dt(u) - Ekman*lap(u) + grad(p) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T"))
problem.add_equation(eq_eval("div(u) = 0"))
problem.add_equation(eq_eval("dt(T) - lap(T)/Prandtl = - dot(u,grad(T))"))
problem.add_equation(eq_eval("u(r=7/13) = 0"))
problem.add_equation(eq_eval("T(r=7/13) = T_inner"))
problem.add_equation(eq_eval("u(r=20/13) = 0"))
problem.add_equation(eq_eval("T(r=20/13) = 0"))
# pressure gauge?
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK111)

class StateVector:

    def __init__(self, args):
        solver, vars = args
        self.data = []
        for subproblem in solver.subproblems:
            self.data.append(subproblem.get_vector(vars))

    def pack(self, vars):
        for i,subproblem in enumerate(solver.subproblems):
            self.data[i] = subproblem.get_vector(vars)

    def unpack(self, vars):
        for i,subproblem in enumerate(solver.subproblems):
            subproblem.set_vector(vars, self.data[i])

state = StateVector( (solver, [u,p,T,tau_u_inner,tau_T_inner,tau_u_outer,tau_T_outer]) )
state.pack( (u,p,T,tau_u_inner,tau_T_inner,tau_u_outer,tau_T_outer) )
NL    = StateVector( (solver, [u,p,T,tau_u_inner,tau_T_inner,tau_u_outer,tau_T_outer]) )

# ChebyshevV
alpha_BC = (2-1/2, 2-1/2)

def C(N):
    ab = alpha_BC
    cd = (b.alpha[0]+2,b.alpha[1]+2)
    return dedalus_sphere.jacobi128.coefficient_connection(N,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list

def add_tau(solver):

    for subproblem in solver.subproblems:
        ell = subproblem.local_index[1]
        M = subproblem.M_min
        L = subproblem.L_min
        shape = M.shape

        tau_columns = np.zeros((shape[0], 8))
        subproblem.M_min = sparse.bmat([[M, tau_columns ]], dtype=np.complex128 ).tocsr()

        N0, N1, N2, N3, N4 = BC_rows(Nmax, 5)

        tau_columns[  :N0,0] = (C(Nmax))[:,-1]
        tau_columns[N0:N1,1] = (C(Nmax))[:,-1]
        tau_columns[N1:N2,2] = (C(Nmax))[:,-1]
        tau_columns[N3:N4,3] = (C(Nmax))[:,-1]
        tau_columns[  :N0,4] = (C(Nmax))[:,-2]
        tau_columns[N0:N1,5] = (C(Nmax))[:,-2]
        tau_columns[N1:N2,6] = (C(Nmax))[:,-2]
        tau_columns[N3:N4,7] = (C(Nmax))[:,-2]

        subproblem.L_min = sparse.bmat([[L, tau_columns ]], dtype=np.complex128).tocsr()

def set_ell_zero(solver):
    for subproblem in solver.subproblems:
        if subproblem.local_index[1] == 0:
            shape = subproblem.M_min.shape
            M = sparse.csr_matrix(shape, dtype=np.complex128)
            L = sparse.identity(shape[0], format='csr', dtype=np.complex128)

            N0, N1, N2, N3, N4 = BC_rows(Nmax, 5)

            M22 = de.operators.convert(T, (bk2,)).subproblem_matrix(subproblem)
            M[N3:N4,N3:N4] = M22

            L22 = -1/Prandtl*de.operators.Laplacian(T, c).subproblem_matrix(subproblem)
            L[N3:N4,N3:N4] = L22

            #BCs
            op = de.operators.interpolate(T,r=7/13)
            L[N4+3,N3:N4] = op.subproblem_matrix(subproblem)
            op = de.operators.interpolate(T,r=20/13)
            L[N4+7,N3:N4] = op.subproblem_matrix(subproblem)

            # taus
            L[N3:N4,N4+3] = (C(Nmax))[:,-1].reshape((Nmax+1,1))
            L[N3:N4,N4+7] = (C(Nmax))[:,-2].reshape((Nmax+1,1))
            L[N4+3,N4+3] = 0
            L[N4+7,N4+7] = 0

            subproblem.M_min = M.tocsr()
            subproblem.L_min = L.tocsr()

timestepper = timesteppers_sphere.SBDF2(StateVector, (solver, [u,p,T,tau_u_inner,tau_T_inner,tau_u_outer,tau_T_outer] ))

LU = [None]*len(solver.subproblems)

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack((u,p,T,tau_u_inner,tau_T_inner,tau_u_outer,tau_T_outer))
    u_rhs = problem.equations[0]['F'].evaluate()
    T_rhs = problem.equations[2]['F'].evaluate()
    if 0 in b.local_l:
        u_rhs['c'][:,:,0,:].fill(0) # very important to zero out the ell=0 RHS
    NL.pack((u_rhs,p_rhs,T_rhs,u_inner,T_inner,u_outer,T_outer))

add_tau(solver)
set_ell_zero(solver)

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

while t < t_end:

    nonlinear(state, NL, t)

    if iter % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias/vol
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = 0.5*T0*(np.pi)/(Lmax+1)/L_dealias/vol
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, T0={:e}".format(iter, dt, t, E0, T0))
        t_list.append(t)
        E_list.append(E0)

    timestepper.step(dt, state, solver, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_conv.dat',np.array([t_list,E_list]))

