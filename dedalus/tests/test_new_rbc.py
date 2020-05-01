

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = (4, 4, 1)
Nx, Ny, Nz = 8, 8, 16
Prandtl = 1
Rayleigh = 3000

# Bases
c = coords.CartesianCoordinates('x', 'y', 'z')
d = distributor.Distributor((c,))
xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, Ly))
zb = basis.ChebyshevT(c.coords[2], size=Nz, bounds=(0, Lz))

# Fields
p = field.Field(name='p', dist=d, bases=(xb,yb,zb), dtype=np.complex128)
b = field.Field(name='b', dist=d, bases=(xb,yb,zb), dtype=np.complex128)
u = field.Field(name='u', dist=d, bases=(xb,yb,zb), dtype=np.complex128, tensorsig=(c,))

# Taus
zb2 = basis.ChebyshevV(c.coords[2], size=Nz, bounds=(0, Lz), alpha0=0)
t1 = field.Field(name='t4', dist=d, bases=(xb,yb), dtype=np.complex128)
t2 = field.Field(name='t4', dist=d, bases=(xb,yb), dtype=np.complex128)
t3 = field.Field(name='t4', dist=d, bases=(xb,yb), dtype=np.complex128, tensorsig=(c,))
t4 = field.Field(name='t4', dist=d, bases=(xb,yb), dtype=np.complex128, tensorsig=(c,))
P1 = field.Field(name='P1', dist=d, bases=(zb2,), dtype=np.complex128)
P2 = field.Field(name='P2', dist=d, bases=(zb2,), dtype=np.complex128)
P1['c'][0,0,-1] = 1
P2['c'][0,0,-2] = 1

# Parameters and operators
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)
ez = field.Field(name='ez', dist=d, bases=(xb,yb,zb), dtype=np.complex128, tensorsig=(c,))
ez['g'][2] = 1
ghat = - ez
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, b, u, t1, t2, t3, t4])
problem.add_equation(eq_eval("div(u) = 0"))
problem.add_equation(eq_eval("dt(b) - P*lap(b) + P1*t1 + P2*t2 = - dot(u,grad(b))"))
problem.add_equation(eq_eval("dt(u) - R*lap(u) + grad(p) = - dot(u,grad(u)) - b*ghat"))
problem.add_equation(eq_eval("u(z=0) = 0"))
problem.add_equation(eq_eval("u(z=Lz) = 0"))
problem.add_equation(eq_eval("b(z=0) = Lz"))
problem.add_equation(eq_eval("b(z=Lz) = 0"))
# pressure gauge?
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK111)

# Plot matrices
import matplotlib.pyplot as plt
plt.figure()
I = 2
J = 2
for i, sp in enumerate(solver.subproblems[:I]):
    print(np.linalg.cond((sp.M_min + sp.L_min).A))
    for j, mat in enumerate(['M_min', 'L_min']):
        axes = plt.subplot(I,J,i*J+j+1)
        A = getattr(sp, mat)
        im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
        axes.set_title('sp %i, %s' %(i, mat))
        axes.set_aspect('equal')
        plt.colorbar(im)
plt.tight_layout()
plt.savefig("rbc_matrices.pdf")
