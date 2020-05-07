

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

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
c_S2 = c.S2coordsys
d = distributor.Distributor((c,))
bB = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius/2)
bS = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radii=(radius/2, radius))
bmid = bB.S2_basis(radius=radius/3)
btop = bS.S2_basis(radius=radius)
phi_B, theta_B, r_B = bB.local_grids((1, 1, 1))
phi_S, theta_S, r_S = bS.local_grids((1, 1, 1))

# Fields
uB = field.Field(dist=d, bases=(bB,), dtype=np.complex128, tensorsig=(c,))
pB = field.Field(dist=d, bases=(bB,), dtype=np.complex128)
tB = field.Field(dist=d, bases=(bmid,), dtype=np.complex128, tensorsig=(c,))
uS = field.Field(dist=d, bases=(bS,), dtype=np.complex128, tensorsig=(c,))
pS = field.Field(dist=d, bases=(bS,), dtype=np.complex128)
tS_ang = field.Field(dist=d, bases=(bmid,), dtype=np.complex128, tensorsig=(c_S2,))
tS_rad = field.Field(dist=d, bases=(bmid,), dtype=np.complex128)
tS2 = field.Field(dist=d, bases=(btop,), dtype=np.complex128, tensorsig=(c,))

# Boundary conditions
utop = field.Field(dist=d, bases=(btop,), dtype=np.complex128, tensorsig=(c,))
utop['g'][2] = 0. # u_r = 0
utop['g'][1] = - u0*np.cos(theta_S)*np.cos(phi_S)
utop['g'][0] = u0*np.sin(phi_S)

# Parameters and operators
ezB = field.Field(dist=d, bases=(bB,), dtype=np.complex128, tensorsig=(c,))
ezB['g'][1] = -np.sin(theta_B)
ezB['g'][2] =  np.cos(theta_B)
ezS = field.Field(dist=d, bases=(bS,), dtype=np.complex128, tensorsig=(c,))
ezS['g'][1] = -np.sin(theta_S)
ezS['g'][2] =  np.cos(theta_S)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
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
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
for subproblem in solver.subproblems:
    if subproblem.group[1] != 0:
        ell = subproblem.group[1]
        L = subproblem.L_min
        NB = Nmax - ell//2 + 1
        L[2*NB-1, -9] = 1
        L[3*NB-1, -8] = 1
        L[4*NB-1, -7] = 1
        NS = bS.shape[-1]
        L[4*NB+2*NS-1, -6] = 1
        L[4*NB+2*NS-2, -3] = 1
        L[4*NB+3*NS-1, -5] = 1
        L[4*NB+3*NS-2, -2] = 1
        L[4*NB+4*NS-1, -4] = 1
        L[4*NB+4*NS-2, -1] = 1
        L.eliminate_zeros()
        subproblem.expand_matrices(['M','L'])

# Check condition number and plot matrices
import matplotlib.pyplot as plt
plt.figure()
for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    M = subproblem.M_min
    L = subproblem.L_min
#    plt.imshow(np.log10(np.abs(L.A)))
#    plt.colorbar()
#    plt.savefig("matrices/ell_%03i.png" %ell, dpi=300)
#    plt.clf()
    print(subproblem.group, np.linalg.cond((M + L).A))

# Analysis
t_list = []
E_list = []

weightB_theta = bB.local_colatitude_weights(1)
weightB_r = bB.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weightB_r*weightB_theta+0*pB['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correctionB = 4*np.pi/3*0.5**3/vol_test

weightS_theta = bS.local_colatitude_weights(1)
weightS_r = bS.local_radius_weights(1)
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
print('Run time:', end_time-start_time)
