

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
p_rhs = field.Field(dist=d, bases=(bk2,), dtype=np.complex128)

# create boundary conditions
u_BC = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
u_BC['g'][2] = 0. # u_r = 0
u_BC['g'][1] = - u0*np.cos(theta)*np.cos(phi)
u_BC['g'][0] = u0*np.sin(phi)
taus = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)

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
problem = problems.IVP([p, u])
problem.add_equation(eq_eval("div(u) = 0"))
problem.add_equation(eq_eval("dt(u) - nu*lap(u) + grad(p) = - dot(u,grad(u)) - Om*cross(ez, u)"))
problem.add_equation(eq_eval("u(r=1) = u_BC"))
# pressure gauge?
print("Problem built")

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

state = StateVector( (solver, [p,u,taus]) )
NL = StateVector( (solver, [p,u,taus]) )

def add_tau(solver, alpha_BC = 0):

    def C(N, ell, deg):
        ab = (alpha_BC,ell+deg+0.5)
        cd = (2,       ell+deg+0.5)
        return dedalus_sphere.jacobi128.coefficient_connection(N - ell//2,ab,cd)

    def BC_rows(N, ell, num_comp):
        N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
        return N_list

    for subproblem in solver.subproblems:
        ell = subproblem.local_index[1]
        M = subproblem.M_min
        L = subproblem.L_min
        shape = M.shape

        tau_columns = np.zeros((shape[0], 3))
        subproblem.M_min = sparse.bmat([[M, tau_columns ]], dtype=np.complex128 ).tocsr()

        N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)

        tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]

        subproblem.L_min = sparse.bmat([[L, tau_columns ]], dtype=np.complex128).tocsr()

def set_ell_zero(solver):
    for subproblem in solver.subproblems:
        if subproblem.local_index[1] == 0:
            shape = subproblem.M_min.shape
            subproblem.M_min = sparse.csr_matrix(shape, dtype=np.complex128)
            subproblem.L_min = sparse.identity(shape[0], format='csr', dtype=np.complex128)

timestepper = timesteppers_sphere.CNAB2(StateVector, (solver, [p,u,u_BC] ))

LU = [None]*len(solver.subproblems)

# calculate RHS terms from state vector
def nonlinear(state_vector, NL, t):

    # get U in coefficient space
    state_vector.unpack((p,u,taus))
    u_rhs = problem.equations[1]['F'].evaluate()
    if 0 in b.local_l:
        u_rhs['c'][:,:,0,:].fill(0) # very important to zero out the ell=0 RHS
    NL.pack((p_rhs,u_rhs,u_BC))

add_tau(solver)
set_ell_zero(solver)

# Integration parameters
dt = 0.02
t_end = 20

t_list = []
E_list = []

reducer = GlobalArrayReducer(d.comm_cart)

# timestepping loop
t = 0.
start_time = time.time()
iter = 0

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

while t < t_end:

    nonlinear(state, NL, t)

    if iter % 10 == 0:
        state.unpack((p,u,taus))
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("t = %f, E = %e" %(t,E0))
        t_list.append(t)
        E_list.append(E0)

    timestepper.step(dt, state, solver, NL, LU)

    t += dt
    iter += 1

end_time = time.time()

