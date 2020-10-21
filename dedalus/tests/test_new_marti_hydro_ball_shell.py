

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
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
Nmax = 23
N_dealias = 1
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2
dt = 0.005
t_end = 20
dtype = np.float64
ts = timesteppers.SBDF4
mesh = None

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
c_S2 = c.S2coordsys
d = distributor.Distributor((c,), mesh=mesh)
bB = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius/2, dtype=dtype)
bS = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radii=(radius/2, radius), dtype=dtype)
bmid = bB.S2_basis(radius=radius/2)
btop = bS.S2_basis(radius=radius)
phi_B, theta_B, r_B = bB.local_grids((1, 1, 1))
phi_S, theta_S, r_S = bS.local_grids((1, 1, 1))

# Fields
uB = field.Field(dist=d, bases=(bB,), dtype=dtype, tensorsig=(c,))
pB = field.Field(dist=d, bases=(bB,), dtype=dtype)
tB = field.Field(dist=d, bases=(bmid,), dtype=dtype, tensorsig=(c,))
uS = field.Field(dist=d, bases=(bS,), dtype=dtype, tensorsig=(c,))
pS = field.Field(dist=d, bases=(bS,), dtype=dtype)
tS_ang = field.Field(dist=d, bases=(bmid,), dtype=dtype, tensorsig=(c_S2,))
tS_rad = field.Field(dist=d, bases=(bmid,), dtype=dtype)
tS2 = field.Field(dist=d, bases=(btop,), dtype=dtype, tensorsig=(c,))

# Boundary conditions
utop = field.Field(dist=d, bases=(btop,), dtype=dtype, tensorsig=(c,))
utop['g'][2] = 0. # u_r = 0
utop['g'][1] = - u0*np.cos(theta_S)*np.cos(phi_S)
utop['g'][0] = u0*np.sin(phi_S)

# Parameters and operators
ezB = field.Field(dist=d, bases=(bB,), dtype=dtype, tensorsig=(c,))
ezB['g'][1] = -np.sin(theta_B)
ezB['g'][2] =  np.cos(theta_B)
ezS = field.Field(dist=d, bases=(bS,), dtype=dtype, tensorsig=(c,))
ezS['g'][1] = -np.sin(theta_S)
ezS['g'][2] =  np.cos(theta_S)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
radcomp = lambda A: operators.RadialComponent(A)
angcomp = lambda A: operators.AngularComponent(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([pB, uB, pS, uS, tB, tS_ang, tS_rad, tS2])
# Equations for ell != 0, ball
problem.add_equation(eq_eval("div(uB) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(uB) - nu*lap(uB) + grad(pB) = - dot(uB,grad(uB)) - Om*cross(ezB, uB)"), condition="ntheta != 0")
# Equations for ell != 0, shell
problem.add_equation(eq_eval("div(uS) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(uS) - nu*lap(uS) + grad(pS) = - dot(uS,grad(uS)) - Om*cross(ezS, uS)"), condition="ntheta != 0")
# Boundary conditions for ell != 0
problem.add_equation(eq_eval("uB(r=1/2) - uS(r=1/2) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("angcomp(radcomp(grad(uB)(r=1/2) - grad(uS)(r=1/2))) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("pB(r=1/2) - pS(r=1/2) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("uS(r=1) = utop"), condition="ntheta != 0")
# Equations for ell == 0
problem.add_equation(eq_eval("pB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("uB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("pS = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("uS = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tS_ang = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tS_rad = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tS2 = 0"), condition="ntheta == 0")
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
alpha_BC_ball = 0

def C_ball(N, ell, deg):
    ab = (alpha_BC_ball,ell+deg+0.5)
    cd = (2,            ell+deg+0.5)
    return dedalus_sphere.jacobi.coefficient_connection(N - ell//2 + 1,ab,cd)

# ChebyshevV
alpha_BC_shell = (2-1/2, 2-1/2)

def C_shell(N):
    ab = alpha_BC_shell
    cd = (bS.radial_basis.alpha[0]+2,bS.radial_basis.alpha[1]+2)
    return dedalus_sphere.jacobi.coefficient_connection(N,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list

for subproblem in solver.subproblems:
    if subproblem.group[1] != 0:
        if dtype == np.complex128:
            ell = subproblem.group[1]
            L = subproblem.left_perm.T @ subproblem.L_min
            shape = L.shape
            tau_columns = np.zeros((shape[0], 9))
            N0, N1, N2, N3 = BC_rows(Nmax - ell//2, 4)
            tau_columns[N0:N1, -9] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N2, -8] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N3, -7] = (C_ball(Nmax, ell,  0))[:,-1]
            NS = bS.shape[-1]
            N4, N5, N6, N7 = N3 + BC_rows(NS-1, 4)
            tau_columns[N4:N5, -6] = (C_shell(NS))[:,-1]
            tau_columns[N4:N5, -3] = (C_shell(NS))[:,-2]
            tau_columns[N5:N6, -5] = (C_shell(NS))[:,-1]
            tau_columns[N5:N6, -2] = (C_shell(NS))[:,-2]
            tau_columns[N6:N7, -4] = (C_shell(NS))[:,-1]
            tau_columns[N6:N7, -1] = (C_shell(NS))[:,-2]
            L[:,-9:] = tau_columns
            L.eliminate_zeros()
        elif dtype == np.float64:
            ell = subproblem.group[1]
            L = subproblem.left_perm.T @ subproblem.L_min
            shape = L.shape
            tau_columns = np.zeros((shape[0], 18))
            NL = Nmax - ell//2 + 1
            N0, N1, N2, N3 = BC_rows(Nmax - ell//2, 4) * 2
            tau_columns[N0:N0+NL, 0] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N1+NL, 1] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N2+NL, 2] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N0+NL:N0+2*NL, 9] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N1+NL:N1+2*NL, 10] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N2+NL:N2+2*NL, 11] = (C_ball(Nmax, ell,  0))[:,-1]
            NS = bS.shape[-1]
            N4, N5, N6, N7 = N3 + BC_rows(NS-1, 4)*2
            tau_columns[N4:N4+NS, 3] = (C_shell(NS))[:,-1]
            tau_columns[N4:N4+NS, 4] = (C_shell(NS))[:,-2]
            tau_columns[N5:N5+NS, 5] = (C_shell(NS))[:,-1]
            tau_columns[N5:N5+NS, 6] = (C_shell(NS))[:,-2]
            tau_columns[N6:N6+NS, 7] = (C_shell(NS))[:,-1]
            tau_columns[N6:N6+NS, 8] = (C_shell(NS))[:,-2]
            tau_columns[N4+NS:N4+2*NS, 12] = (C_shell(NS))[:,-1]
            tau_columns[N4+NS:N4+2*NS, 13] = (C_shell(NS))[:,-2]
            tau_columns[N5+NS:N5+2*NS, 14] = (C_shell(NS))[:,-1]
            tau_columns[N5+NS:N5+2*NS, 15] = (C_shell(NS))[:,-2]
            tau_columns[N6+NS:N6+2*NS, 16] = (C_shell(NS))[:,-1]
            tau_columns[N6+NS:N6+2*NS, 17] = (C_shell(NS))[:,-2]
            L[:,-18:] = tau_columns
            L.eliminate_zeros()
        subproblem.L_min = subproblem.left_perm @ L
        if problem.STORE_EXPANDED_MATRICES:
            subproblem.expand_matrices(['M','L'])

## Check condition number and plot matrices
#import matplotlib.pyplot as plt
#plt.figure()
#for subproblem in solver.subproblems:
#    ell = subproblem.group[1]
#    M = subproblem.left_perm.T @ subproblem.M_min
#    L = subproblem.left_perm.T @ subproblem.L_min
#    plt.imshow(np.log10(np.abs(L.A)))
#    plt.colorbar()
#    plt.savefig("matrices/ell_%03i.png" %ell, dpi=300)
#    plt.clf()
#    print(subproblem.group, np.linalg.cond((M + L).A))

# Analysis
t_list = []
E_list = []

weightB_theta = bB.local_colatitude_weights(1)
weightB_r = bB.local_radial_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weightB_r*weightB_theta+0*pB['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correctionB = 4*np.pi/3*0.5**3/vol_test

weightS_theta = bS.local_colatitude_weights(1)
weightS_r = bS.local_radial_weights(1)*r_S**2
reducer = GlobalArrayReducer(d.comm_cart)

vol_test = np.sum(weightS_r*weightS_theta+0*pS['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correctionS = 4*np.pi/3*(1**3 - 0.5**3)/vol_test

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % 10 == 0:
        E0B = np.sum(vol_correctionB*weightB_r*weightB_theta*uB['g'].real**2)
        E0B = 0.5*E0B*(np.pi)/(Lmax+1)/L_dealias
        E0B = reducer.reduce_scalar(E0B, MPI.SUM)
        E0S = np.sum(vol_correctionS*weightS_r*weightS_theta*uS['g'].real**2)
        E0S = 0.5*E0S*(np.pi)/(Lmax+1)/L_dealias
        E0S = reducer.reduce_scalar(E0S, MPI.SUM)
        logger.info("t = %f, E = %e" %(solver.sim_time, E0B + E0S))
        t_list.append(solver.sim_time)
        E_list.append(E0B + E0S)
    solver.step(dt)
end_time = time.time()
logger.info('Run time:', end_time-start_time)

if MPI.COMM_WORLD.rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_hydro_ball_shell.dat',np.array([t_list,E_list]))
