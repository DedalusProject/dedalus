

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
import dedalus_sphere
import logging
logger = logging.getLogger(__name__)

dtype = np.float64

# Parameters
radius = 1
Mmax = 3
Nmax = 9

# Bases
c = coords.PolarCoordinates('phi', 'r')
d = distributor.Distributor((c,))
b = basis.DiskBasis(c, (Mmax+1, Nmax+1), radius=radius, dtype=dtype)

cb = b.S1_basis(r=radius)

phi, r = b.local_grids((1, 1, 1))

# Fields
f = field.Field(dist=d, bases=(b,), dtype=dtype)
tau = field.Field(dist=d, bases=(cb,), dtype=dtype)

# Parameters and operators
lap = lambda A: operators.Laplacian(A, c)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([f,tau])
problem.add_equation(eq_eval("dt(f) - lap(f) = 0"))
problem.add_equation(eq_eval("f(r=1) = 0"))
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK111)

# Add taus
alpha_BC = 0

def C(N, m, s):
    ab = (alpha_BC, m+s)
    cd = (2,        m+s)
    return dedalus_sphere.jacobi.coefficient_connection(N - m//2 + 1,ab,cd)

for subproblem in solver.subproblems:
    m = subproblem.group[0]
    L = subproblem.L_min
    M = subproblem.M_min
    if dtype == np.complex128:
        raise NotImplementedError()
    elif dtype == np.float64:
        NM = Nmax - m//2 + 1
        tau_columns = np.zeros((L.shape[0], 2))
        tau_columns[:NM,0] = (C(Nmax, m, 0))[:,-1]
        tau_columns[NM:2*NM,1] = (C(Nmax, m, 0))[:,-1]
        L[:,-2:] = tau_columns
    L.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])



# Plot matrices
import matplotlib.pyplot as plt
plt.figure()
I = 2
J = 2
for i, sp in enumerate(solver.subproblems[:I]):
   for j, mat in enumerate(['M_min', 'L_min']):
       axes = plt.subplot(I,J,i*J+j+1)
       A = getattr(sp, mat)
       im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
       axes.set_title('sp %i, %s' %(i, mat))
       axes.set_aspect('equal')
       plt.colorbar(im)
plt.tight_layout()
plt.savefig("nmh_matrices.pdf")
