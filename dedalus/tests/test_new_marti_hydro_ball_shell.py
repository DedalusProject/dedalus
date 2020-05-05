

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
d = distributor.Distributor((c,))
bB = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius/2)
bB2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius/2, k=2)
bS = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radii=(radius/2, radius))
bS2 = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radii=(radius/2, radius), k=2)
bmid = bB.S2_basis(radius=radius/2)
btop = bS.S2_basis(radius=radius)
phi_B, theta_B, r_B = bB.local_grids((1, 1, 1))
phi_S, theta_S, r_S = bS.local_grids((1, 1, 1))

# Fields
uB = field.Field(dist=d, bases=(bB,), dtype=np.complex128, tensorsig=(c,))
pB = field.Field(dist=d, bases=(bB,), dtype=np.complex128)
tB = field.Field(dist=d, bases=(bmid,), dtype=np.complex128, tensorsig=(c,))
uS = field.Field(dist=d, bases=(bS,), dtype=np.complex128, tensorsig=(c,))
pS = field.Field(dist=d, bases=(bS,), dtype=np.complex128)
tS1 = field.Field(dist=d, bases=(bmid,), dtype=np.complex128, tensorsig=(c,))
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

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([pB, uB, pS, uS, tB, tS1, tS2])
# Equations for ell != 0, ball
problem.add_equation(eq_eval("div(uB) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(uB) - nu*lap(uB) + grad(pB) = - dot(uB,grad(uB)) - Om*cross(ezB, uB)"), condition="ntheta != 0")
# Equations for ell != 0, shell
problem.add_equation(eq_eval("div(uS) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(uS) - nu*lap(uS) + grad(pS) = - dot(uS,grad(uS)) - Om*cross(ezS, uS)"), condition="ntheta != 0")
# Boundary conditions for ell != 0
problem.add_equation(eq_eval("uB(r=1/2) - uS(r=1/2) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("radcomp(grad(uB)(r=1/2) - grad(uS)(r=1/2)) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("uS(r=1) = utop"), condition="ntheta != 0")
# Equations for ell == 0
problem.add_equation(eq_eval("pB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("uB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("pS = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("uS = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tB = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("tS1 = 0"), condition="ntheta == 0")
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
        L[4*NB+2*NS-2, -6] = 1
        L[4*NB+2*NS-1, -5] = 1
        L[4*NB+3*NS-2, -4] = 1
        L[4*NB+3*NS-1, -3] = 1
        L[4*NB+4*NS-2, -2] = 1
        L[4*NB+4*NS-1, -1] = 1
        L.eliminate_zeros()
        subproblem.expand_matrices(['M','L'])

# Check condition number and plot matrices
import matplotlib.pyplot as plt
plt.figure()
for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    M = subproblem.M_min
    L = subproblem.L_min
    print(subproblem.group, np.linalg.cond((M + L).A))
    plt.imshow(np.log10(np.abs(L.A)))
    plt.colorbar()
    plt.savefig("matrices/ell_%03i.png" %ell, dpi=300)
    plt.clf()
raise

# Analysis
t_list = []
E_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)
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
print('Run time:', end_time-start_time)
