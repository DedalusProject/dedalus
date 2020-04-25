

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation

import logging
logger = logging.getLogger(__name__)


# Parameters
radius = 1
Lmax = 3
Nmax = 9
Om = 20.
u0 = np.sqrt(3/(2*np.pi))
nu = 1e-2

# Integration parameters
dt = 0.02
t_end = 20

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = field.Field(dist=d, bases=(b,), dtype=np.complex128)

# create boundary conditions
u_BC = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
u_BC['g'][2] = 0. # u_r = 0
u_BC['g'][1] = - u0*np.cos(theta)*np.cos(phi)
u_BC['g'][0] = u0*np.sin(phi)

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

class Subproblem_l:

    def __init__(self,ell):
        self.ell = ell

ell = 2
sp = solver.subproblems[ell]
sp_l = Subproblem_l(ell)

# Making old M operator:
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius)
op = operators.convert(u, (bk2,))
M11 = op.subproblem_matrix(sp)
M10 = operators.ZeroMatrix(p, (c,)).subproblem_matrix(sp_l)
M01 = operators.ZeroMatrix(u, ()).subproblem_matrix(sp_l)
M00 = operators.ZeroMatrix(p, ()).subproblem_matrix(sp_l)
from scipy import sparse
M=sparse.bmat([[M00,M01],
               [M10,M11]])
M.tocsr()

M_new = solver.subproblems[2].M_min

print('New M matrix matches with old M matrix:')
print(np.allclose(M.toarray(), M_new.toarray()[:-3,:]))

# Making old L operator:
op = operators.convert(-nu*operators.Laplacian(u, c) + operators.Gradient(p, c), (bk2,))
op_matrices = op.expression_matrices(sp, (p,u))
L11 = op_matrices[u]
L10 = op_matrices[p]
op = operators.Divergence(u)
L01 = op.subproblem_matrix(sp)
L00 = operators.ZeroMatrix(p, ()).subproblem_matrix(sp_l)
op = operators.interpolate(u,r=1)
R = op.subproblem_matrix(sp)
Z = operators.ZeroVector(p, (c,)).subproblem_matrix(sp_l)
L=sparse.bmat([[L00,L01],
               [L10,L11],
               [Z  ,R  ]])

L = L.tocsr()

L_new = solver.subproblems[2].L_min

print('New L matrix matches with old L matrix:')
print( np.allclose(L.toarray(), L_new.toarray() ))

# Plot matrices
#import matplotlib.pyplot as plt
#plt.figure()
#I = 2
#J = 2
#for i, sp in enumerate(solver.subproblems[:I]):
#    for j, mat in enumerate(['M_min', 'L_min']):
#        axes = plt.subplot(I,J,i*J+j+1)
#        A = getattr(sp, mat)
#        im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
#        axes.set_title('sp %i, %s' %(i, mat))
#        axes.set_aspect('equal')
#        plt.colorbar(im)
#plt.tight_layout()
#plt.savefig("nmh_matrices.pdf")
