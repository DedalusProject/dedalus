import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers
from dedalus.tools import logging
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
import logging
logger = logging.getLogger(__name__)
dtype = np.complex128 #np.float64

Nx = 64
Nz = 32
Lz = 2*np.pi
# Bases
c = coords.CartesianCoordinates('x', 'z')
d = distributor.Distributor((c,))
if dtype == np.complex128:
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, 2*np.pi), dealias=3/2)
elif dtype == np.float64:
    xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, 2*np.pi), dealias=3/2)
zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz),dealias=3/2)
x = xb.local_grid(1)
z = zb.local_grid(1)
zb1 = basis.ChebyshevU(c.coords[1], size=Nz, bounds=(0, Lz), alpha0=0)
t1 = field.Field(name='t1', dist=d, bases=(xb,), dtype=dtype)
t2 = field.Field(name='t2', dist=d, bases=(xb,), dtype=dtype)
P1 = field.Field(name='P1', dist=d, bases=(zb1,), dtype=dtype)
if rank == 0:
    P1['c'][0,-1] = 1
dz = lambda A: operators.Differentiate(A, c.coords[1])
P2 = dz(P1).evaluate()

# Fields
u = field.Field(name='u', dist=d, bases=(xb,zb), dtype=dtype)
F = field.Field(name='F', dist=d, bases=(xb,zb), dtype=dtype)
u_true = 0
for n in np.arange(1,Nz//8):
    u_current_n = 0
    for m in np.arange(Nx//4):
        u_current_m = np.sin(n*z)*np.cos(m*x)
        u_current_n += u_current_m
        F['g'] += -m**2*u_current_m
    u_true += u_current_n
    F['g'] += -n**2*u_current_n
# Problem

lap = lambda A: operators.Laplacian(A, c)
problem = problems.LBVP([u,t1,t2])
problem.add_equation((lap(u) + P1*t1 + P2*t2, F))
problem.add_equation((u(z=0), 0))
problem.add_equation((u(z=Lz), 0))
# Solver
solver = solvers.LinearBoundaryValueSolver(problem)
solver.solve()
# Check solution
print('L1 error on rank {} = {}'.format(rank, np.max(np.abs(u['g']-u_true))))
assert np.allclose(u['g'],u_true)
