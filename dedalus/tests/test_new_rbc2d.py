

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
import matplotlib.pyplot as plt
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (4, 1)
Nx, Nz = 128, 32
Prandtl = 1
Rayleigh = 1e5
timestep = 0.01
stop_iteration = 100

# Bases
c = coords.CartesianCoordinates('x', 'z')
d = distributor.Distributor((c,))
xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz))
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
p = field.Field(name='p', dist=d, bases=(xb,zb), dtype=np.complex128)
b = field.Field(name='b', dist=d, bases=(xb,zb), dtype=np.complex128)
u = field.Field(name='u', dist=d, bases=(xb,zb), dtype=np.complex128, tensorsig=(c,))

# Taus
zb2 = basis.ChebyshevV(c.coords[1], size=Nz, bounds=(0, Lz), alpha0=0)
t1 = field.Field(name='t1', dist=d, bases=(xb,), dtype=np.complex128)
t2 = field.Field(name='t2', dist=d, bases=(xb,), dtype=np.complex128)
t3 = field.Field(name='t3', dist=d, bases=(xb,), dtype=np.complex128, tensorsig=(c,))
t4 = field.Field(name='t4', dist=d, bases=(xb,), dtype=np.complex128, tensorsig=(c,))
P1 = field.Field(name='P1', dist=d, bases=(zb2,), dtype=np.complex128)
P2 = field.Field(name='P2', dist=d, bases=(zb2,), dtype=np.complex128)
if rank == 0:
    P1['c'][0,-1] = 1
    P2['c'][0,-2] = 1

# Parameters and operators
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)

ez = field.Field(name='ez', dist=d, bases=None, dtype=np.complex128, tensorsig=(c,))
ez['g'][1] = 1
ghat = - ez

B = field.Field(name='B', dist=d, bases=(zb,), dtype=np.complex128)
B['g'] = Lz - z

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, b, u, t1, t2, t3, t4])
problem.add_equation(eq_eval("div(u) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("dt(b) - P*lap(b) + P1*t1 + P2*t2 = - dot(u,grad(b)) - dot(u,grad(B)) + P*lap(B)"))
problem.add_equation(eq_eval("dt(u) - R*lap(u) + grad(p) + P1*t3 + P2*t4 + b*ghat = - dot(u,grad(u)) - B*ghat"), condition="nx != 0")
problem.add_equation(eq_eval("b(z=0) = Lz - B(z=0)"))
problem.add_equation(eq_eval("b(z=Lz) = 0 - B(z=Lz)"))
problem.add_equation(eq_eval("u(z=0) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("u(z=Lz) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("p = 0"), condition="nx == 0")
problem.add_equation(eq_eval("u = 0"), condition="nx == 0")
problem.add_equation(eq_eval("t3 = 0"), condition="nx == 0")
problem.add_equation(eq_eval("t4 = 0"), condition="nx == 0")
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
solver.stop_iteration = stop_iteration

# Add vector taus by hand
for i, subproblem in enumerate(solver.subproblems):
    M = subproblem.M_min
    L = subproblem.L_min
    print(i, subproblem.group, np.linalg.cond((M+L).A))

# Initial conditions
b['g'] = 1e-1 * np.random.randn(*b['g'].shape)

# Main loop
plt.figure()
while solver.ok:
    if solver.iteration % 10 == 0:
        print('Iteration: %i' %solver.iteration)
        plt.imshow(b['g'].real.T, origin='lower')
        plt.savefig('frames/frame_%06i.png' %solver.iteration)
        plt.clf()
    solver.step(timestep)

# Plot matrices
import matplotlib.pyplot as plt
spi = [0, 1]
I = len(spi)
J = 2
plt.figure(figsize=(3*J,3*I))
for i, ind in enumerate(spi):
    sp = solver.subproblems[ind]
    for j, mat in enumerate(['M_min', 'L_min']):
        axes = plt.subplot(I,J,i*J+j+1)
        A = getattr(sp, mat)
        im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
        axes.set_title('sp %i, %s' %(i, mat))
        axes.set_aspect('equal')
        plt.colorbar(im)
plt.tight_layout()
plt.savefig("rbc2d_matrices.pdf")
