

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
from dedalus.libraries.dedalus_sphere import jacobi
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

Lmax = 31
L_dealias = 1
Nmax = 31
N_dealias = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b    = basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii)
b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
A = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
V = field.Field(dist=d, bases=(b,), dtype=np.complex128)
B = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
tau_A_inner = field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
tau_A_outer = field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)

## initial toroidal magnetic field
#B['g'][0] = 5*np.sin(np.pi*(r-r_inner))*np.sin(2*theta)
#B['g'][1] = 5/8*(9*r -8*r_outer -r_inner**4/r**3)*np.sin(theta)
#B['g'][2] = 5/8*(8*r_outer -6*r -2*r_inner**4/r**3)*np.cos(theta)

B['g'][0] = -3/4 * r * (r**2 - 1) * np.cos(theta) * (3 * r * (2-5*r**2+4*r**4) * np.sin(theta) + 2*(1-3*r**2+3*r**4)*(np.cos(phi) - np.sin(phi)))
B['g'][1] = -3/2 * r * (-1 + 4*r**2 - 6*r**4 + 3*r**6) * (np.cos(phi) + np.sin(phi))

# Parameters and operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
curl = lambda A: operators.Curl(A)

# Potential BC on A
r_out = 1
ell_func = lambda ell: ell + 1
radial = lambda A: operators.RadialComponent(A)
ell_op = lambda A: operators.SphericalEllProduct(A, c, ell_func)

A_bc_outer = radial(grad(A)(r=r_outer)) + ell_op(A)(r=r_outer)/r_outer
A_bc_inner = radial(grad(A)(r=r_inner)) + ell_op(A)(r=r_inner)/r_inner

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

# BVP for initial A
BVP = problems.LBVP([A, V, tau_A_inner, tau_A_outer])

BVP.add_equation(eq_eval("curl(A) + grad(V) = B"), condition="ntheta != 0")
BVP.add_equation(eq_eval("div(A) = 0"), condition="ntheta != 0")
BVP.add_equation(eq_eval("A_bc_inner = 0"), condition="ntheta != 0")
BVP.add_equation(eq_eval("A_bc_outer = 0"), condition="ntheta != 0")
BVP.add_equation(eq_eval("A = 0"), condition="ntheta == 0")
BVP.add_equation(eq_eval("V = 0"), condition="ntheta == 0")
BVP.add_equation(eq_eval("tau_A_inner = 0"), condition="ntheta == 0")
BVP.add_equation(eq_eval("tau_A_outer = 0"), condition="ntheta == 0")

solver = solvers.LinearBoundaryValueSolver(BVP)

# ChebyshevV
alpha_BC = (2-1/2, 2-1/2)

def C(N):
    ab = alpha_BC
    cd = (b.radial_basis.alpha[0]+2,b.radial_basis.alpha[1]+2)
    return jacobi.coefficient_connection(N + 1,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.left_perm.T @ subproblem.L_min
    shape = L.shape
    tau_columns = np.zeros((shape[0], 6))
    BCs         = np.zeros((3, shape[1]))
    N0, N1, N2, N3 = BC_rows(Nmax, 4)
    if ell != 0:
        tau_columns[N0:N1,0] = (C(Nmax))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax))[:,-1]
        tau_columns[N0:N1,3] = (C(Nmax))[:,-2]
        tau_columns[N1:N2,4] = (C(Nmax))[:,-2]
        tau_columns[N2:N3,5] = (C(Nmax))[:,-2]
        L[:,-6:] = tau_columns
    subproblem.L_min = subproblem.left_perm @ L
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['L'])

logger.info("built BVP")
solver.solve()
logger.info("solved BVP")

def err(a,b):
    print(np.max(np.abs(a-b))/np.max(np.abs(b)))

#print('largest entry of V:')
#print(np.max(np.abs(V['g'])))

B2 = (operators.Curl(A) + operators.Gradient(V, c)).evaluate()
divB = (operators.Divergence(B)).evaluate()
divB2 = (operators.Divergence(B2)).evaluate()
print("|divB| = {}".format(np.max(np.abs(divB['g']))))
print("|divB2| = {}".format(np.max(np.abs(divB2['g']))))
print("|B2 - B| = {}".format(np.max(np.abs(B2['g']-B['g']))/np.max(np.abs(B['g']))))
print("|V| = {}".format(np.max(np.abs(V['g']))))

c_par = coords.SphericalCoordinates('phi', 'theta', 'r')
d_par = distributor.Distributor((c_par,))
b_par = basis.SphericalShellBasis(c_par, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii)
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

print('errors in components of A (r, theta, phi):')
err(A_par['g'][2],A_analytic_2)
err(A_par['g'][1],A_analytic_1)
err(A_par['g'][0],A_analytic_0)
