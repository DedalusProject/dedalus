

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.libraries.dedalus_sphere import jacobi
from scipy import sparse
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
radius = 1
Lmax = 7
L_dealias = 1
Nmax = 7
N_dealias = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,),comm=MPI.COMM_SELF)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
A = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
V = field.Field(dist=d, bases=(b,), dtype=np.complex128)
B = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
tau_A = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)

# initial toroidal magnetic field
B['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
B['g'][0] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))

# Potential BC on A
r_out = 1
ell_func = lambda ell: ell + 1
A_potential_bc = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=1)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=1)/r_out

# Parameters and operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
curl = lambda A: operators.Curl(A)
LiftTau = lambda A: operators.LiftTau(A, b, -1)

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

# BVP for initial A
BVP = problems.LBVP([A, V, tau_A])

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
#BVP.add_equation(eq_eval("curl(A) + grad(V) + LiftTau(tau_A) = B"), condition="ntheta != 0")
BVP.add_equation(eq_eval("curl(A) + grad(V) = B"), condition="ntheta != 0")
BVP.add_equation(eq_eval("div(A) = 0"), condition="ntheta != 0")
BVP.add_equation(eq_eval("A_potential_bc = 0"), condition="ntheta != 0")
BVP.add_equation(eq_eval("A = 0"), condition="ntheta == 0")
BVP.add_equation(eq_eval("V = 0"), condition="ntheta == 0")
BVP.add_equation(eq_eval("tau_A = 0"), condition="ntheta == 0")

solver = solvers.LinearBoundaryValueSolver(BVP)

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (1,       ell+deg+0.5)
    return jacobi.coefficient_connection(N + 1 - ell//2,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.left_perm.T @ subproblem.L_min
    shape = L.shape
    tau_columns = np.zeros((shape[0], 3))
    BCs         = np.zeros((3, shape[1]))
    N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)
    if ell != 0:
        tau_columns[N0:N1,0] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax, ell,  0))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
        L[:,-3:] = tau_columns
    subproblem.L_min = subproblem.left_perm @ L
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['L'])

plot_matrices = False

if plot_matrices:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    I = 2
    J = 1
    for i, sp in enumerate(solver.subproblems[:I]):
        for j, mat in enumerate(['L_min']):
            axes = plt.subplot(I,J,i*J+j+1)
            A = getattr(sp, mat)
            A = sp.left_perm.T @ A
            im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
            axes.set_title('sp %i, %s' %(i, mat))
            axes.set_aspect('equal')
            plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("nmh_matrices.pdf")

solver.solve()

def err(a,b):
    logger.info(np.max(np.abs(a-b))/np.max(np.abs(b)))

B2 = (curl(A) + grad(V)).evaluate()

logger.info('magnetic field error:')
err(B2['g'],B['g'])

c_par = coords.SphericalCoordinates('phi', 'theta', 'r')
d_par = distributor.Distributor((c_par,))
b_par = basis.BallBasis(c_par, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
phi, theta, r = b_par.local_grids((1, 1, 1))

A_par = field.Field(dist=d_par, bases=(b_par,), tensorsig=(c_par,), dtype=np.complex128)
slices = d_par.grid_layout.slices(A_par.domain,(1,1,1))
A_par['g'][:] = A['g'][:,slices[0],slices[1],slices[2]]

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

logger.info('errors in components of A (r, theta, phi):')
err(A_par['g'][2],A_analytic_2)
err(A_par['g'][1],A_analytic_1)
err(A_par['g'][0],A_analytic_0)

