

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
Lmax = 31
L_dealias = 1
Nmax = 31
N_dealias = 1
dt = 5e-6
t_end = 20
ts = timesteppers.SBDF2

Ekman = 5e-4
Roberts = 7
Rayleigh = 200
Rossby = 5/7*1e-4
Source = 3*Roberts

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=[8,8]) #[16,16])
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
A = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = field.Field(dist=d, bases=(b,), dtype=np.complex128)
φ = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
tau_A = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)
B = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)

r_vec = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r

T['g'] = 0.5*(1-r**2) + 0.1/8*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# initial toroidal velocity field
u['g'][1] = -10*r**2/7/np.sqrt(3)*np.cos(theta)*(  3*(-147+343*r**2-217*r**4+29*r**6)*np.cos(phi)
                                                 +14*(-9 - 125*r**2 +39*r**4+27*r**6)*np.sin(phi) )
u['g'][2] = -5*r/5544*( 7*(           (43700-58113*r**2-15345*r**4+1881*r**6+20790*r**8)*np.sin(theta)
                           +1485*r**2*(-9 + 115*r**2 - 167*r**4 + 70*r**6)*np.sin(3*theta) )
                       +528*np.sqrt(3)*r*np.cos(2*theta)*( 14*(-9-125*r**2+39*r**4+27*r**6)*np.cos(phi)
                                                           +3*(147-343*r**2+217*r**4-29*r**6)*np.sin(phi) ) )

T_source = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T_source['g'] = Source

# initial toroidal magnetic field
B['g'][0] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))
B['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))


# Boundary conditions
u_r_bc = operators.RadialComponent(operators.interpolate(u,r=1))

stress = operators.Gradient(u, c) + operators.TransposeComponents(operators.Gradient(u, c))
u_perp_bc = operators.RadialComponent(operators.AngularComponent(operators.interpolate(stress,r=1), index=1))

# Potential BC on B
r_out = 1
ell_func = lambda ell: ell+1
A_potential_bc = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=1)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=1)/r_out

#A_perp_bc = operators.AngularComponent(operators.interpolate(A,r=1))
# problem.add_equation(eq_eval("φ(r=1) = 0"), condition="ntheta != 0") # perfect conductor
# problem.add_equation(eq_eval("A_perp_bc = 0"), condition="ntheta != 0") # perfect conductor


# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
curl = lambda A: operators.Curl(A)

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

# Initial condtions on A
# # BVP for initial A
# solve for A via BVP; conduct on serial domain
# d_IC = distributor.Distributor((c,), comm=MPI.COMM_SELF)
#
# B_IC = field.Field(dist=d_IC, bases=(b,), tensorsig=(c,), dtype=np.complex128)
# V = field.Field(dist=d_IC, bases=(b,), dtype=np.complex128)
# tau_A_IC = field.Field(dist=d_IC, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
# A_IC = field.Field(dist=d_IC, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#
# A_potential_bc_IC = operators.RadialComponent(operators.interpolate(operators.Gradient(A_IC, c), r=1)) + operators.interpolate(operators.SphericalEllProduct(A_IC, c, ell_func), r=1)/r_out
#
# B_IC['g'] = B['g']
#
# BVP = problems.LBVP([A_IC, V, tau_A_IC])
#
# def eq_eval(eq_str):
#     return [eval(expr) for expr in split_equation(eq_str)]
# BVP.add_equation(eq_eval("curl(A_IC) + grad(V) = B_IC"), condition="ntheta != 0")
# BVP.add_equation(eq_eval("div(A_IC) = 0"), condition="ntheta != 0")
# BVP.add_equation(eq_eval("A_potential_bc_IC = 0"), condition="ntheta != 0")
# #BVP.add_equation(eq_eval("A_bc = 0"), condition="ntheta != 0")
# BVP.add_equation(eq_eval("A_IC = 0"), condition="ntheta == 0")
# BVP.add_equation(eq_eval("V = 0"), condition="ntheta == 0")
# BVP.add_equation(eq_eval("tau_A_IC = 0"), condition="ntheta == 0")
#
# solver = solvers.LinearBoundaryValueSolver(BVP)
#
# # Add taus
# alpha_BC = 0
#
# def C(N, ell, deg):
#     ab = (alpha_BC,ell+deg+0.5)
#     cd = (1,       ell+deg+0.5)
#     return dedalus_sphere.jacobi128.coefficient_connection(N - ell//2,ab,cd)
#
# def BC_rows(N, ell, num_comp):
#     N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
#     return N_list
#
# for subproblem in solver.subproblems:
#     ell = subproblem.group[1]
#     L = subproblem.L_min
#     shape = L.shape
#     tau_columns = np.zeros((shape[0], 3))
#     BCs         = np.zeros((3, shape[1]))
#     N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)
#     if ell != 0:
#         tau_columns[N0:N1,0] = (C(Nmax, ell, +1))[:,-1]
#         tau_columns[N1:N2,1] = (C(Nmax, ell,  0))[:,-1]
#         tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
#         subproblem.L_min[:,-3:] = tau_columns
#
#         # hand built potential field boundary condition
#         BCs[0,  :N0] = b.operator_matrix('r=R', ell, -1)
#         BCs[1,N0:N1] = b.operator_matrix('r=R', ell,  0, dk=1) @ b.operator_matrix('D-', ell, +1)
#         BCs[2,N1:N2] = b.operator_matrix('r=R', ell, -1, dk=1) @ b.operator_matrix('D-', ell,  0)
#         subproblem.L_min[-3:,:] = BCs
#     subproblem.L_min.eliminate_zeros()
#     subproblem.expand_matrices(['L'])
#
# logger.info("built BVP")
# solver.solve()
# logger.info("solved BVP")
#
# A['g'] = A_IC['g']

A_analytic_2 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +3/8*r**3*(2-7*r**2+9*r**4-4*r**6)
                   *(3*np.cos(theta)**2-1)
               +9/160*r**2*(-200/21*r+980/27*r**3-540/11*r**5+880/39*r**7)
                     *(3*np.cos(theta)**2-1)
               +9/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                    *(3*np.cos(theta)**2-1)
               +1/8*r*(-48/5*r+288/7*r**3-64*r**5+360/11*r**7)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_1 = (-27/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                        *np.cos(theta)*np.sin(theta)
                +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                    *np.cos(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_0 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *(np.cos(phi)+np.sin(phi)))

A['g'][0] = A_analytic_0
A['g'][1] = A_analytic_1
A['g'][2] = A_analytic_2

# Problem
problem = problems.IVP([p, u, φ, A, T, tau_u, tau_A, tau_T])

problem.add_equation(eq_eval("div(u) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("Rossby*ddt(u) - Ekman*lap(u) + grad(p) = - Rossby*dot(u,grad(u)) + Roberts*Rayleigh*r_vec*T - cross(ez, u) + dot(curl(A),grad(curl(A)))"), condition = "ntheta != 0")
problem.add_equation(eq_eval("div(A) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(A) - lap(A) + grad(φ) = cross(u, B)"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(T) - Roberts*lap(T) = - dot(u,grad(T)) + T_source"))
problem.add_equation(eq_eval("p = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("φ = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("A = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("u_r_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("A_potential_bc = 0"), condition="ntheta != 0") # placeholder
problem.add_equation(eq_eval("tau_u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tau_A = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("T(r=1) = 0"))

# Solver
solver = solvers.InitialValueSolver(problem, ts)
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
    subproblem.M_min[:,-7:] = 0
    subproblem.M_min.eliminate_zeros()
    N0, N1, N2, N3, N4, N5, N6, N7, N8 = BC_rows(Nmax, ell, 9)
    tau_columns = np.zeros((shape[0], 7))
    BCs         = np.zeros((3, shape[1]))
    if ell != 0:
        # nothing on p
        tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
        # nothing on phi
        tau_columns[N4:N5,3] = (C(Nmax, ell, -1))[:,-1]
        tau_columns[N5:N6,4] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N6:N7,5] = (C(Nmax, ell,  0))[:,-1]
        tau_columns[N7:N8,6] = (C(Nmax, ell,  0))[:,-1]
        subproblem.L_min[:,-7:] = tau_columns

        # hand built potential field boundary condition
        BCs[0,N4:N5] = b.operator_matrix('r=R', ell, -1)
        BCs[1,N5:N6] = b.operator_matrix('r=R', ell,  0, dk=1) @ b.operator_matrix('D-', ell, +1)
        BCs[2,N6:N7] = b.operator_matrix('r=R', ell, -1, dk=1) @ b.operator_matrix('D-', ell,  0)
        subproblem.L_min[-4:-1,:] = BCs
    else: # ell = 0
        tau_columns[N7:N8, 6] = (C(Nmax, ell, 0))[:,-1]
        subproblem.L_min[:,-1:] = tau_columns[:,6:]
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

logger.info("built IVP")

# Analysis
t_list = []
KE_list = []
ME_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

# Main loop
report_cadence = 1
start_time = time.time()
while solver.ok:
    if solver.iteration % report_cadence == 0:
        KE = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        KE = 0.5*KE*(np.pi)/(Lmax+1)/L_dealias
        KE = reducer.reduce_scalar(KE, MPI.SUM)
        B = (operators.Curl(A)).evaluate()
        ME = np.sum(vol_correction*weight_r*weight_theta*B['g'].real**2)
        ME = 0.5*ME*(np.pi)/(Lmax+1)/L_dealias
        ME /= Rossby
        ME = reducer.reduce_scalar(ME, MPI.SUM)
        logger.info("t = {:f}, KE = {:e},  ME = {:e}".format(solver.sim_time, KE, ME))
        t_list.append(solver.sim_time)
        KE_list.append(KE)
        ME_list.append(ME)
    solver.step(dt)
end_time = time.time()
print('Run time:', end_time-start_time)
