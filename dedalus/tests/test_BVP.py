

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
Lmax = 7
L_dealias = 1
Nmax = 7
N_dealias = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
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

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

# BVP for initial A
BVP = problems.LBVP([A, V, tau_A])

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
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
    return dedalus_sphere.jacobi128.coefficient_connection(N - ell//2,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.L_min
    shape = L.shape
    tau_columns = np.zeros((shape[0], 3))
    BCs         = np.zeros((3, shape[1]))
    N0, N1, N2, N3 = BC_rows(Nmax, ell, 4)
    if ell != 0:
        tau_columns[N0:N1,0] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax, ell,  0))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
        subproblem.L_min[:,-3:] = tau_columns

        BCs[0,  :N0] = b.operator_matrix('r=R', ell, -1)
        BCs[1,N0:N1] = b.operator_matrix('r=R', ell,  0, dk=1) @ b.operator_matrix('D-', ell, +1)
        BCs[2,N1:N2] = b.operator_matrix('r=R', ell, -1, dk=1) @ b.operator_matrix('D-', ell,  0)
        subproblem.L_min[-3:,:] = BCs
    subproblem.L_min.eliminate_zeros()
    print(ell, np.linalg.cond(subproblem.L_min.A))
    subproblem.expand_matrices(['L'])

#import matplotlib.pyplot as plt
#plt.figure()
#I = 2
#J = 1
#for i, sp in enumerate(solver.subproblems[:I]):
#    for j, mat in enumerate(['L_min']):
#        axes = plt.subplot(I,J,i*J+j+1)
#        A = getattr(sp, mat)
#        im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
#        axes.set_title('sp %i, %s' %(i, mat))
#        axes.set_aspect('equal')
#        plt.colorbar(im)
#plt.tight_layout()
#plt.savefig("bvp_matrices.pdf")

solver.solve()

A_analytic_0 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
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
A_analytic_2 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *(np.cos(phi)+np.sin(phi)))

def err(a,b):
    print(np.max(np.abs(a-b))/np.max(np.abs(b)))

print(np.max(np.abs(V['g'])))

B2 = (operators.Curl(A) + operators.Gradient(V, c)).evaluate()

print('magnetic field error:')
err(B2['g'],B['g'])

print('errors:')
err(A['g'][2],A_analytic_0)
err(A['g'][1],A_analytic_1)
err(A['g'][0],A_analytic_2)
